from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'blob_path',
            default_value='/home/ubuntu/yolov8n.blob',
            description='Path to YOLOv8 blob file'
        ),
        
        DeclareLaunchArgument(
            'camera_fps',
            default_value='15',
            description='OAK-D FPS'
        ),
        
        # OAK-D Detector - TURTLEBOT4
        Node(
            package='semantic_mapping',
            executable='oakd_detector',
            name='oakd_detector_node',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'blob_path': LaunchConfiguration('blob_path'),
                'confidence_threshold': 0.6,
                'camera_fps': LaunchConfiguration('camera_fps'),
                'target_classes': [
                    'person', 'chair', 'couch', 'potted plant',
                    'bottle', 'cup', 'laptop', 'backpack'
                ],
                'publish_images': True,
            }]
        ),
    ])