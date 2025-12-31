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
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        
        # Semantic Mapper - JETSON
        Node(
            package='semantic_mapping',
            executable='semantic_mapper',
            name='semantic_mapper',
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim_time': use_sim_time,
                    # Subscribe topics TurtleBot4
                    'rgb_topic': '/oakd/rgb/preview/image_raw',
                    'depth_topic': '/oakd/rgb/preview/depth',
                    'camera_info_topic': '/oakd/rgb/preview/camera_info',
                    'camera_frame': 'oakd_rgb_camera_optical_frame',
                    'save_path': '/home/phong/semantic_map.json',
                }
            ]
        ),

        # Frontier Explorer - JETSON
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
                            'min_frontier_size': 5,
                            'cluster_distance': 0.6,
                            'exploration_rate': 3.0,
                            'min_goal_distance': 0.8,
                            'max_goal_distance': 15.0,
                            'enable_rotation_goals': True,
                        }
                    ]
                )
            ]
        ),
    ])