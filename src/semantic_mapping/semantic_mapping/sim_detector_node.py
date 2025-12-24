#!/usr/bin/env python3
"""
Simulation Detector - YOLOv8 on CPU/GPU for Gazebo

Usage:
    ros2 run semantic_mapping sim_detector
    
Topics:
    Subscribes: /camera/image_raw, /camera/depth/image_raw
    Publishes: /object_detections, /object_markers
"""

import time
import json
import numpy as np
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

try:
    from message_filters import Subscriber, ApproximateTimeSynchronizer
    HAS_MESSAGE_FILTERS = True
except ImportError:
    HAS_MESSAGE_FILTERS = False

from .detection_interface import (
    BaseDetector, Detection, DetectionResult, BoundingBox, COCO_CLASSES
)


class SimulationDetector(BaseDetector):
    """YOLOv8 detector for simulation"""
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        super().__init__(model_path, confidence_threshold)
        self.device = device
        self.model = None
        self.class_names = COCO_CLASSES
        
    def initialize(self) -> bool:
        try:
            from ultralytics import YOLO
            
            print(f"📦 Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"🖥️  Device: {self.device}")
            
            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            
            self._initialized = True
            print("✅ Detector initialized")
            return True
            
        except ImportError:
            print("❌ ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def detect(self, image: np.ndarray, 
               depth: Optional[np.ndarray] = None,
               camera_intrinsics: Optional[dict] = None) -> DetectionResult:
        if not self._initialized:
            return DetectionResult(
                timestamp=time.time(), frame_id="camera",
                image_width=image.shape[1], image_height=image.shape[0],
                detections=[]
            )
        
        start_time = time.time()
        results = self.model.predict(
            image, device=self.device, 
            conf=self.confidence_threshold, verbose=False
        )
        inference_time = (time.time() - start_time) * 1000
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                bbox = BoundingBox(
                    x_min=int(xyxy[0]), y_min=int(xyxy[1]),
                    x_max=int(xyxy[2]), y_max=int(xyxy[3])
                )
                
                position_3d = None
                if depth is not None and camera_intrinsics is not None:
                    position_3d = self.get_3d_position(bbox, depth, camera_intrinsics)
                
                detection = Detection(
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    confidence=conf,
                    bbox=bbox,
                    position_3d=position_3d
                )
                detections.append(detection)
        
        return DetectionResult(
            timestamp=time.time(), frame_id="camera",
            image_width=image.shape[1], image_height=image.shape[0],
            detections=detections, inference_time_ms=inference_time
        )
    
    def shutdown(self):
        self.model = None
        self._initialized = False


class SimDetectorNode(Node):
    """ROS2 node for simulation detector"""
    
    def __init__(self):
        super().__init__('sim_detector_node')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'auto')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('use_depth', True)
        self.declare_parameter('target_classes', [
            'chair', 'couch', 'potted plant', 'bottle', 'cup', 'person',
            'dining table', 'tv', 'laptop', 'bed', 'toilet'
        ])
        self.declare_parameter('throttle_rate', 5.0)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        conf_thresh = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        self.use_depth = self.get_parameter('use_depth').value
        self.target_classes = list(self.get_parameter('target_classes').value)
        self.throttle_rate = self.get_parameter('throttle_rate').value
        
        # Initialize detector
        self.detector = SimulationDetector(model_path, conf_thresh, device)
        if not self.detector.initialize():
            self.get_logger().error("Failed to initialize detector!")
            return
        
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.last_detection_time = 0.0
        self.min_interval = 1.0 / self.throttle_rate
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        
        # Subscribers
        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )
        
        if self.use_depth and HAS_MESSAGE_FILTERS:
            self.image_sub = Subscriber(self, Image, image_topic, qos_profile=qos)
            self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=qos)
            self.sync = ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub], queue_size=5, slop=0.1
            )
            self.sync.registerCallback(self.synced_callback)
        else:
            self.image_sub = self.create_subscription(
                Image, image_topic, self.image_callback, qos
            )
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/object_markers', 10)
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)
        
        self.get_logger().info(f"🚀 Sim Detector started")
        self.get_logger().info(f"   Model: {model_path}, Device: {device}")
        self.get_logger().info(f"   Rate: {self.throttle_rate} Hz")
    
    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5]
            }
            self.get_logger().info(f"📷 Camera intrinsics received")
    
    def should_process(self) -> bool:
        now = time.time()
        if now - self.last_detection_time >= self.min_interval:
            self.last_detection_time = now
            return True
        return False
    
    def image_callback(self, msg: Image):
        if not self.should_process():
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            result = self.detector.detect(image)
            self.process_result(result, msg.header)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
    
    def synced_callback(self, image_msg: Image, depth_msg: Image):
        if not self.should_process():
            return
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
            
            if depth_msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            elif depth_msg.encoding == "16UC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1").astype(np.float32) / 1000.0
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
            
            result = self.detector.detect(image, depth, self.camera_intrinsics)
            self.process_result(result, image_msg.header)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
    
    def process_result(self, result: DetectionResult, header: Header):
        # Filter by target classes
        if self.target_classes:
            result = result.filter_by_class(self.target_classes)
        
        if len(result.detections) == 0:
            return
        
        self.get_logger().info(
            f"🎯 Detected {len(result.detections)} objects ({result.inference_time_ms:.0f}ms)"
        )
        
        # Publish markers
        self.publish_markers(result, header)
        
        # Publish JSON
        data = {
            'timestamp': result.timestamp,
            'inference_time_ms': result.inference_time_ms,
            'detections': [d.to_dict() for d in result.detections]
        }
        msg = String()
        msg.data = json.dumps(data)
        self.detection_pub.publish(msg)
    
    def publish_markers(self, result: DetectionResult, header: Header):
        marker_array = MarkerArray()
        
        for i, det in enumerate(result.detections):
            if det.position_3d is None:
                continue
            
            marker = Marker()
            marker.header = header
            marker.header.frame_id = "camera_link"
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Camera to ROS coordinate conversion
            marker.pose.position.x = det.position_3d[2]   # Z forward
            marker.pose.position.y = -det.position_3d[0]  # X right -> Y left
            marker.pose.position.z = -det.position_3d[1]  # Y down -> Z up
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            marker.color.a = 0.8
            
            # Color by class
            if det.class_name == 'person':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif det.class_name in ['chair', 'couch', 'bed']:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.5, 1.0
            
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
            
            # Text label
            text = Marker()
            text.header = marker.header
            text.ns = "labels"
            text.id = i + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = marker.pose
            text.pose.position.z += 0.25
            text.text = f"{det.class_name}: {det.confidence:.2f}"
            text.scale.z = 0.12
            text.color.a = 1.0
            text.color.r = text.color.g = text.color.b = 1.0
            text.lifetime.sec = 1
            marker_array.markers.append(text)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SimDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.detector.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
