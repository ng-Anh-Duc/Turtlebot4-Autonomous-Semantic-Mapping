#!/usr/bin/env python3
"""
semantic_mapper.py - Object Detection and Semantic Mapping
Hỗ trợ cả TurtleBot3 (simulation) và TurtleBot4 (real robot)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import json
import os
from datetime import datetime

from semantic_mapping.utils.tf_utils import CoordinateTransformer, calculate_distance


class DetectedObject:
    def __init__(self, class_name, x, y, z, confidence):
        self.class_name = class_name
        self.x, self.y, self.z = x, y, z
        self.confidence = confidence
        self.detection_count = 1
        self.last_seen = datetime.now()
        
    def update(self, x, y, z, confidence):
        alpha = 0.3
        self.x = alpha * x + (1 - alpha) * self.x
        self.y = alpha * y + (1 - alpha) * self.y
        self.z = alpha * z + (1 - alpha) * self.z
        self.confidence = max(self.confidence, confidence)
        self.detection_count += 1
        self.last_seen = datetime.now()
        
    def to_dict(self):
        return {
            'class': self.class_name,
            'x': round(self.x, 3),
            'y': round(self.y, 3),
            'z': round(self.z, 3),
            'confidence': round(self.confidence, 3),
            'detection_count': self.detection_count
        }


class SemanticMapper(Node):
    CLASS_COLORS = {
        'person': (1.0, 1.0, 0.0),
        'chair': (0.5, 0.0, 1.0),
        'backpack': (1.0, 0.5, 0.0),
        'bottle': (0.0, 0.5, 1.0),
        'laptop': (1.0, 0.0, 0.5),
        'cup': (0.0, 1.0, 0.5),
    }
    
    def __init__(self):
        super().__init__('semantic_mapper')
        
        # Declare parameters với default values
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('detection_rate', 0.5)
        self.declare_parameter('min_object_distance', 0.5)
        self.declare_parameter('save_path', '/ros2_ws/maps/semantic_map.json')
        self.declare_parameter('target_classes', ['person', 'chair', 'backpack', 'bottle', 'laptop', 'cup'])
        
        # Topic parameters - có thể config cho TurtleBot3 hoặc TurtleBot4
        self.declare_parameter('rgb_topic', '/oakd/rgb/preview/image_raw')
        self.declare_parameter('depth_topic', '/oakd/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/oakd/rgb/preview/camera_info')
        self.declare_parameter('camera_frame', 'oakd_rgb_camera_optical_frame')
        
        # Load parameters
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.detection_rate = self.get_parameter('detection_rate').value
        self.min_object_distance = self.get_parameter('min_object_distance').value
        self.save_path = self.get_parameter('save_path').value
        self.target_classes = self.get_parameter('target_classes').value
        
        # Topic names
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        
        self.get_logger().info(f'📷 RGB Topic: {rgb_topic}')
        self.get_logger().info(f'📷 Depth Topic: {depth_topic}')
        self.get_logger().info(f'📷 Camera Frame: {self.camera_frame}')
        
        # Load YOLO
        model_path = self.get_parameter('yolo_model').value
        self.get_logger().info(f'📦 Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('✅ YOLO model loaded!')
        
        # Components
        self.bridge = CvBridge()
        self.transformer = CoordinateTransformer(self)
        
        # Data
        self.camera_info = None
        self.detected_objects = []
        self.latest_rgb = None
        self.latest_depth = None
        
        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscribers - dùng topic names từ config
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, sensor_qos
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_map/markers', 10)
        self.detection_image_pub = self.create_publisher(Image, '/semantic_map/detection_image', 10)
        
        # Timers
        self.detection_timer = self.create_timer(self.detection_rate, self.detection_callback)
        self.save_timer = self.create_timer(30.0, self.save_map)
        
        self.get_logger().info('🚀 Semantic Mapper initialized!')
        
    def camera_info_callback(self, msg):
        self.camera_info = msg
        
    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB error: {e}')
            
    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')
    
    def find_existing_object(self, class_name, x, y):
        for obj in self.detected_objects:
            if obj.class_name == class_name:
                if calculate_distance((obj.x, obj.y), (x, y)) < self.min_object_distance:
                    return obj
        return None
    
    def detection_callback(self):
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            return
        
        results = self.model(self.latest_rgb, verbose=False, conf=self.confidence_threshold)
        
        import cv2
        detection_image = self.latest_rgb.copy()
        
        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])]
                if class_name not in self.target_classes:
                    continue
                
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_u, center_v = (x1 + x2) // 2, (y1 + y2) // 2
                
                if not (0 <= center_v < self.latest_depth.shape[0] and 
                        0 <= center_u < self.latest_depth.shape[1]):
                    continue
                
                depth = self.latest_depth[center_v, center_u]
                point_3d = self.transformer.pixel_to_camera_3d(
                    center_u, center_v, depth, self.camera_info
                )
                if point_3d is None:
                    continue
                
                point_map = self.transformer.transform_point(
                    point_3d, self.camera_frame, 'map'
                )
                if point_map is None:
                    continue
                
                existing = self.find_existing_object(class_name, point_map[0], point_map[1])
                if existing:
                    existing.update(point_map[0], point_map[1], point_map[2], confidence)
                else:
                    self.detected_objects.append(
                        DetectedObject(class_name, *point_map, confidence)
                    )
                    self.get_logger().info(
                        f'🎯 NEW: {class_name} at ({point_map[0]:.2f}, {point_map[1]:.2f})'
                    )
                
                cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(detection_image, f'{class_name}: {confidence:.2f}',
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        try:
            self.detection_image_pub.publish(
                self.bridge.cv2_to_imgmsg(detection_image, 'bgr8')
            )
        except:
            pass
        
        self.publish_markers()
    
    def publish_markers(self):
        marker_array = MarkerArray()
        
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)
        
        for i, obj in enumerate(self.detected_objects):
            color = self.CLASS_COLORS.get(obj.class_name, (0.5, 0.5, 0.5))
            
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns, sphere.id = 'objects', i * 2
            sphere.type, sphere.action = Marker.SPHERE, Marker.ADD
            sphere.pose.position.x = obj.x
            sphere.pose.position.y = obj.y
            sphere.pose.position.z = obj.z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
            sphere.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.8)
            marker_array.markers.append(sphere)
            
            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns, text.id = 'labels', i * 2 + 1
            text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.x = obj.x
            text.pose.position.y = obj.y
            text.pose.position.z = obj.z + 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = obj.class_name
            marker_array.markers.append(text)
        
        self.marker_pub.publish(marker_array)
    
    def save_map(self):
        if not self.detected_objects:
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'frame': 'map',
            'objects': [obj.to_dict() for obj in self.detected_objects]
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.get_logger().info(f'💾 Saved {len(self.detected_objects)} objects')


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_map()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()