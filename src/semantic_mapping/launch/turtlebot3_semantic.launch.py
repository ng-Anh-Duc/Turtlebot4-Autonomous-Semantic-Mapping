"""
Launch semantic mapping với TurtleBot3 Simulation
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    pkg_semantic = get_package_share_directory('semantic_mapping')
    # pkg_share = FindPackageShare('semantic_mapping')
    config_file = os.path.join(pkg_semantic, 'config', 'turtlebot3_params.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        
        # Frontier Explorer (delay 10s để đợi Nav2)
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='semantic_mapping',
                    executable='frontier_explorer',
                    name='frontier_explorer',
                    output='screen',
                    parameters=[config_file, {'use_sim_time': use_sim_time}]
                )
            ]
        ),
        
        # Semantic Mapper (delay 5s)
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='semantic_mapping',
                    executable='semantic_mapper',
                    name='semantic_mapper',
                    output='screen',
                    parameters=[config_file, {'use_sim_time': use_sim_time}]
                )
            ]
        ),

        # Simulation Detector
        Node(
            package='semantic_mapping',
            executable='sim_detector',
            name='sim_detector_node',
            parameters=[
                PathJoinSubstitution([pkg_semantic, 'config', 'detection_params.yaml']),
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),
    ])