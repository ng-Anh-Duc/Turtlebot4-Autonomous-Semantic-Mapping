"""
Launch semantic mapping với TurtleBot4 Real Robot
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    pkg_semantic = get_package_share_directory('semantic_mapping')
    pkg_tb4_nav = get_package_share_directory('turtlebot4_navigation')
    config_file = os.path.join(pkg_semantic, 'config', 'turtlebot4_params.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        
        # TurtleBot4 SLAM
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_tb4_nav, 'launch', 'slam.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),
        
        # TurtleBot4 Nav2
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_tb4_nav, 'launch', 'nav2.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),
        
        # Frontier Explorer (delay 15s để đợi Nav2)
        TimerAction(
            period=15.0,
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
        
        # Semantic Mapper (delay 10s)
        TimerAction(
            period=10.0,
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
        
        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
    ])