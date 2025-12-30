from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    pkg_semantic = get_package_share_directory('semantic_mapping')
    
    config_file = os.path.join(pkg_semantic, 'config', 'turtlebot4_params.yaml')
    # detection_config = os.path.join(pkg_semantic, 'config', 'detection_params.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        DeclareLaunchArgument(
            'blob_path',
            default_value='/home/phong/yolov8n.blob',
            description='Path to YOLOv8 blob file for OAK-D'
        ),
        
        DeclareLaunchArgument(
            'camera_fps',
            default_value='15',
            description='OAK-D camera FPS (10-30)'
        ),
        
        # 1. Sim Detector (Quan trọng: Để detect vật thể trong Gazebo)
        Node(
            package='semantic_mapping',
            executable='oakd_detector',
            name='oakd_detector_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
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

        # 2. Semantic Mapper (Vẽ vật thể lên Map)
        Node(
            package='semantic_mapping',
            executable='semantic_mapper',
            name='semantic_mapper',
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim_time': use_sim_time,
                    # TurtleBot4 OAK-D topics (check với: ros2 topic list)
                    'rgb_topic': '/oakd/rgb/preview/image_raw',
                    'depth_topic': '/oakd/rgb/preview/image_raw/compressedDepth',
                    'camera_info_topic': '/oakd/rgb/preview/camera_info',
                    'camera_frame': 'oakd_rgb_camera_optical_frame',
                    'save_path': '/home/phong/semantic_map.json',
                }
            ]
        ),

        # 3. Frontier Explorer (Tự động tìm đường)
        # Delay 10s để đợi map ổn định
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='semantic_mapping',
                    executable='frontier_explorer',
                    name='frontier_explorer',
                    output='screen',
                    parameters=[
                        config_file,
                        {
                            'use_sim_time': use_sim_time,
                            'min_frontier_size': 5,  # Tăng từ 3 -> 5 để filter noise
                            'cluster_distance': 0.6,  # Tăng để merge frontiers gần nhau
                            'exploration_rate': 3.0,  # Giảm từ 10 -> 5 để ổn định hơn
                            'min_goal_distance': 0.8,  # Tăng để tránh goals quá gần
                            'max_goal_distance': 15.0,  # Giảm từ 20 -> 15 cho an toàn
                            'enable_rotation_goals': True,  # Thêm rotation để scan
                        }
                    ]
                )
            ]
        ),
    ])