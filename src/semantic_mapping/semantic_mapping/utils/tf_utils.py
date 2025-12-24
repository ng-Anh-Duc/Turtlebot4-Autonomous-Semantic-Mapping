#!/usr/bin/env python3
"""
tf_utils.py - Coordinate transformation utilities
Chuyển đổi tọa độ giữa các frame: camera -> base_link -> map
"""

import numpy as np
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
import rclpy


class CoordinateTransformer:
    """Transform coordinates between different frames"""
    
    def __init__(self, node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        
    def pixel_to_camera_3d(self, u, v, depth, camera_info):
        """
        Convert pixel coordinates to 3D point in camera frame.
        
        Args:
            u, v: pixel coordinates
            depth: depth value (mm or m)
            camera_info: CameraInfo message
            
        Returns:
            tuple (x, y, z) or None if invalid
        """
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        
        # Convert depth to meters if in mm
        z = depth / 1000.0 if depth > 100 else depth
        
        if z <= 0.1 or z > 10.0:
            return None
            
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return (x, y, z)
    
    def transform_point(self, point_3d, source_frame, target_frame):
        """
        Transform point from source_frame to target_frame.
        
        Args:
            point_3d: tuple (x, y, z)
            source_frame: e.g., 'oakd_rgb_camera_optical_frame'
            target_frame: e.g., 'map'
            
        Returns:
            tuple (x, y, z) or None if failed
        """
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = self.node.get_clock().now().to_msg()
            point_stamped.point.x = float(point_3d[0])
            point_stamped.point.y = float(point_3d[1])
            point_stamped.point.z = float(point_3d[2])
            
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            point_transformed = tf2_geometry_msgs.do_transform_point(
                point_stamped, transform
            )
            
            return (
                point_transformed.point.x,
                point_transformed.point.y,
                point_transformed.point.z
            )
        except Exception as e:
            self.node.get_logger().warn(f'Transform failed: {e}')
            return None


def calculate_distance(point1, point2):
    """Calculate 2D Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)