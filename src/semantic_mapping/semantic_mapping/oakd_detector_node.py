#!/usr/bin/env python3
"""
OAK-D Detector - YOLOv8 on OAK-D VPU for real TurtleBot4

Runs YOLO on Myriad X VPU, offloading from Pi4 CPU.
Performance: 15-25 FPS, ~5-10% Pi4 CPU usage.

Usage:
    ros2 run semantic_mapping oakd_detector
    
Topics:
    Publishes: /object_detections, /object_markers, /oakd/rgb, /oakd/depth
"""

import time
import json
import numpy as np
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from .detection_interface import (
    BaseDetector, Detection, DetectionResult, BoundingBox, COCO_CLASSES
)


class OakDDetector(BaseDetector):
    """YOLOv8 detector on OAK-D VPU"""
    
    def __init__(self, blob_path: str, confidence_threshold: float = 0.5, camera_fps: int = 15):
        super().__init__(blob_path, confidence_threshold)
        self.camera_fps = camera_fps
        self.device = None
        self.pipeline = None
        self.class_names = COCO_CLASSES
        
        self.rgb_queue = None
        self.depth_queue = None
        self.detection_queue = None
        self.rgb_intrinsics = None
        
    def initialize(self) -> bool:
        try:
            import depthai as dai
            
            print(f"📦 Initializing OAK-D: {self.model_path}")
            
            self.pipeline = dai.Pipeline()
            
            # RGB Camera
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setFps(self.camera_fps)
            
            # Stereo Depth
            mono_left = self.pipeline.create(dai.node.MonoCamera)
            mono_right = self.pipeline.create(dai.node.MonoCamera)
            stereo = self.pipeline.create(dai.node.StereoDepth)
            
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(640, 480)
            
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)
            
            # Resize for NN
            manip = self.pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setResize(640, 640)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
            cam_rgb.preview.link(manip.inputImage)
            
            # YOLO Neural Network
            nn = self.pipeline.create(dai.node.YoloDetectionNetwork)
            nn.setBlobPath(self.model_path)
            nn.setConfidenceThreshold(self.confidence_threshold)
            nn.setNumClasses(80)
            nn.setCoordinateSize(4)
            nn.setAnchors([])
            nn.setAnchorMasks({})
            nn.setIouThreshold(0.5)
            nn.setNumInferenceThreads(2)
            nn.input.setBlocking(False)
            manip.out.link(nn.input)
            
            # Outputs
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)
            
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)
            
            xout_nn = self.pipeline.create(dai.node.XLinkOut)
            xout_nn.setStreamName("detections")
            nn.out.link(xout_nn.input)
            
            # Start device
            self.device = dai.Device(self.pipeline)
            
            self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
            self.detection_queue = self.device.getOutputQueue("detections", maxSize=4, blocking=False)
            
            # Get intrinsics
            calib = self.device.readCalibration()
            self.rgb_intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
            
            self._initialized = True
            print(f"✅ OAK-D initialized @ {self.camera_fps} FPS")
            return True
            
        except ImportError:
            print("❌ depthai not installed. Run: pip install depthai")
            return False
        except Exception as e:
            print(f"❌ OAK-D error: {e}")
            return False
    
    def detect(self, image=None, depth=None, camera_intrinsics=None) -> DetectionResult:
        """Get detection from OAK-D (ignores input params, uses internal camera)"""
        if not self._initialized:
            return DetectionResult(
                timestamp=time.time(), frame_id="oakd_rgb",
                image_width=640, image_height=480, detections=[]
            )
        
        start_time = time.time()
        
        in_rgb = self.rgb_queue.tryGet()
        in_depth = self.depth_queue.tryGet()
        in_det = self.detection_queue.tryGet()
        
        if in_rgb is None or in_det is None:
            return DetectionResult(
                timestamp=time.time(), frame_id="oakd_rgb",
                image_width=640, image_height=480, detections=[]
            )
        
        rgb_frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame() if in_depth else None
        
        detections = []
        for det in in_det.detections:
            x_min = int(det.xmin * rgb_frame.shape[1])
            y_min = int(det.ymin * rgb_frame.shape[0])
            x_max = int(det.xmax * rgb_frame.shape[1])
            y_max = int(det.ymax * rgb_frame.shape[0])
            
            bbox = BoundingBox(x_min, y_min, x_max, y_max)
            
            position_3d = None
            if depth_frame is not None and self.rgb_intrinsics is not None:
                intrinsics = {
                    'fx': self.rgb_intrinsics[0][0],
                    'fy': self.rgb_intrinsics[1][1],
                    'cx': self.rgb_intrinsics[0][2],
                    'cy': self.rgb_intrinsics[1][2]
                }
                depth_m = depth_frame.astype(np.float32) / 1000.0
                position_3d = self.get_3d_position(bbox, depth_m, intrinsics)
            
            detection = Detection(
                class_id=det.label,
                class_name=self.class_names[det.label] if det.label < len(self.class_names) else f"class_{det.label}",
                confidence=det.confidence,
                bbox=bbox,
                position_3d=position_3d
            )
            detections.append(detection)
        
        return DetectionResult(
            timestamp=time.time(), frame_id="oakd_rgb",
            image_width=rgb_frame.shape[1], image_height=rgb_frame.shape[0],
            detections=detections, inference_time_ms=(time.time() - start_time) * 1000
        )
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get RGB and depth frames"""
        in_rgb = self.rgb_queue.tryGet()
        in_depth = self.depth_queue.tryGet()
        
        rgb = in_rgb.getCvFrame() if in_rgb else None
        depth = in_depth.getFrame() if in_depth else None
        
        return rgb, depth
    
    def shutdown(self):
        if self.device:
            self.device.close()
        self._initialized = False
        print("🛑 OAK-D shutdown")


class OakDDetectorNode(Node):
    """ROS2 node for OAK-D detector"""
    
    def __init__(self):
        super().__init__('oakd_detector_node')
        
        self.declare_parameter('blob_path', 'yolov8n_6shaves.blob')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_fps', 15)
        self.declare_parameter('target_classes', [
            'chair', 'couch', 'potted plant', 'bottle', 'cup', 'person'
        ])
        self.declare_parameter('publish_images', True)
        
        blob_path = self.get_parameter('blob_path').value
        conf = self.get_parameter('confidence_threshold').value
        fps = self.get_parameter('camera_fps').value
        self.target_classes = list(self.get_parameter('target_classes').value)
        self.publish_images = self.get_parameter('publish_images').value
        
        self.detector = OakDDetector(blob_path, conf, fps)
        if not self.detector.initialize():
            self.get_logger().error("Failed to initialize OAK-D!")
            return
        
        self.bridge = CvBridge()
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/object_markers', 10)
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)
        
        if self.publish_images:
            self.rgb_pub = self.create_publisher(Image, '/oakd/rgb', 10)
            self.depth_pub = self.create_publisher(Image, '/oakd/depth', 10)
        
        # Timer
        self.timer = self.create_timer(1.0 / fps, self.detection_callback)
        
        self.get_logger().info(f"🚀 OAK-D Detector started @ {fps} FPS")
    
    def detection_callback(self):
        result = self.detector.detect()
        
        if self.target_classes:
            result = result.filter_by_class(self.target_classes)
        
        if len(result.detections) == 0:
            return
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "oakd_rgb_camera_optical_frame"
        
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
        
        # Publish images
        if self.publish_images:
            rgb, depth = self.detector.get_frames()
            if rgb is not None:
                self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb, "rgb8"))
            if depth is not None:
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth, "16UC1"))
    
    def publish_markers(self, result: DetectionResult, header: Header):
        marker_array = MarkerArray()
        
        for i, det in enumerate(result.detections):
            if det.position_3d is None:
                continue
            
            marker = Marker()
            marker.header = header
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = det.position_3d[2]
            marker.pose.position.y = -det.position_3d[0]
            marker.pose.position.z = -det.position_3d[1]
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            marker.color.a = 0.8
            
            if det.class_name == 'person':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif det.class_name in ['chair', 'couch']:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.5, 1.0
            
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = OakDDetectorNode()
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
