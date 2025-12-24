#!/usr/bin/env python3
"""
frontier_explorer.py - Autonomous Exploration using Frontier-Based Algorithm

Robot tự động tìm và di chuyển đến các frontier (ranh giới giữa vùng đã biết và chưa biết)
để khám phá toàn bộ bản đồ.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import numpy as np
from sklearn.cluster import DBSCAN
from enum import Enum
import threading
import tf2_ros


class ExplorationState(Enum):
    IDLE = 0
    EXPLORING = 1
    NAVIGATING = 2
    COMPLETED = 3


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Parameters
        self.declare_parameter('min_frontier_size', 3)
        self.declare_parameter('cluster_distance', 0.5)
        self.declare_parameter('exploration_rate', 10.0)
        self.declare_parameter('min_goal_distance', 0.5)
        self.declare_parameter('max_goal_distance', 20.0)
        
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.cluster_distance = self.get_parameter('cluster_distance').value
        self.exploration_rate = self.get_parameter('exploration_rate').value
        self.min_goal_distance = self.get_parameter('min_goal_distance').value
        self.max_goal_distance = self.get_parameter('max_goal_distance').value
        
        # TF Buffer để lấy robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Map data
        self.map_data = None
        self.map_info = None
        self.map_lock = threading.Lock()
        
        # Navigation
        self.navigator = BasicNavigator()
        self.state = ExplorationState.IDLE
        
        # QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos
        )
        
        # Publishers
        self.frontier_pub = self.create_publisher(MarkerArray, '/frontiers', 10)
        self.goal_pub = self.create_publisher(Marker, '/exploration_goal', 10)
        
        # Wait for Nav2
        self.get_logger().info('⏳ Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('✅ Nav2 is active!')
        
        # Timer
        self.timer = self.create_timer(self.exploration_rate, self.exploration_callback)
        self.get_logger().info('🚀 Frontier Explorer initialized!')
        
    def map_callback(self, msg):
        """Receive map from SLAM"""
        with self.map_lock:
            self.map_data = np.array(msg.data, dtype=np.int8).reshape(
                (msg.info.height, msg.info.width)
            )
            self.map_info = msg.info
            
    def find_frontiers(self):
        """
        Find all frontier cells on the map.
        Frontier = FREE cell (0) with at least one UNKNOWN neighbor (-1)
        """
        with self.map_lock:
            if self.map_data is None:
                self.get_logger().warn('Map data is None!')
                return []
            
            height, width = self.map_data.shape

            # Debug: Count cell types
            free_cells = np.sum(self.map_data == 0)
            occupied_cells = np.sum(self.map_data > 0)
            unknown_cells = np.sum(self.map_data == -1)
            self.get_logger().info(
                f'🗺️ Map: {width}x{height}, Free={free_cells}, Occupied={occupied_cells}, Unknown={unknown_cells}'
            )
            
            if free_cells == 0:
                self.get_logger().warn('No free cells in map yet!')
                return []
        
            frontiers = []
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # if self.map_data[y, x] != 0:  # Only FREE cells
                    #     continue

                    cell_value = self.map_data[y, x]
                
                    # Frontier = FREE cell (giá trị thấp, không phải occupied)
                    # Trong Cartographer: 0 = unknown, thấp = free, cao = occupied
                    # Nhưng trong OccupancyGrid chuẩn: -1 = unknown, 0 = free, 100 = occupied
                    
                    # Thử cả 2 cách:
                    is_free = (cell_value == 0) or (0 < cell_value < 50)
                    
                    if not is_free:
                        continue

                    # Check 8 neighbors
                    # neighbors = self.map_data[y-1:y+2, x-1:x+2].flatten()
                    # if -1 in neighbors:  # Has UNKNOWN neighbor
                    #     world_x = x * self.map_info.resolution + self.map_info.origin.position.x
                    #     world_y = y * self.map_info.resolution + self.map_info.origin.position.y
                    #     frontiers.append((world_x, world_y))

                    has_unknown_neighbor = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if self.map_data[ny, nx] == -1:
                                    has_unknown_neighbor = True
                                    break
                        if has_unknown_neighbor:
                            break
                    
                    if has_unknown_neighbor:
                        world_x = x * self.map_info.resolution + self.map_info.origin.position.x
                        world_y = y * self.map_info.resolution + self.map_info.origin.position.y
                        frontiers.append((world_x, world_y))
            
            return frontiers
    
    def cluster_frontiers(self, frontiers):
        """Cluster nearby frontier cells using DBSCAN"""
        if not frontiers or len(frontiers) < self.min_frontier_size:
            return []
        
        points = np.array(frontiers)
        clustering = DBSCAN(
            eps=self.cluster_distance, 
            min_samples=self.min_frontier_size
        ).fit(points)
        
        clusters = []
        for label in set(clustering.labels_):
            if label == -1:  # Noise
                continue
            cluster_points = points[clustering.labels_ == label]
            centroid = cluster_points.mean(axis=0)
            clusters.append({
                'x': centroid[0], 
                'y': centroid[1], 
                'size': len(cluster_points)
            })
        
        clusters.sort(key=lambda c: c['size'], reverse=True)
        return clusters
    
    def select_best_frontier(self, clusters):
        """Select best frontier based on size and distance"""
        if not clusters:
            return None
        
        # current_pose = self.navigator.get_current_pose()
        # if current_pose is None:
        #     return clusters[0]
        
        # robot_x = current_pose.pose.position.x
        # robot_y = current_pose.pose.position.y

        # Lấy robot pose từ TF thay vì navigator
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f'Could not get robot pose: {e}')
            # Nếu không lấy được pose, chọn cluster lớn nhất
            return clusters[0]
        
        best_score = -float('inf')
        best_frontier = None
        
        for cluster in clusters:
            distance = np.sqrt(
                (cluster['x'] - robot_x)**2 + 
                (cluster['y'] - robot_y)**2
            )
            
            if distance < self.min_goal_distance or distance > self.max_goal_distance:
                continue
            
            # Score = size / distance (prefer large and close frontiers)
            score = cluster['size'] / max(distance, 0.1)
            
            if score > best_score:
                best_score = score
                best_frontier = cluster
        
        return best_frontier
    
    def publish_visualization(self, clusters, goal):
        """Publish visualization markers for RViz"""
        # Publish frontier markers
        marker_array = MarkerArray()
        
        # Clear old markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontiers'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = cluster['x']
            marker.pose.position.y = cluster['y']
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.2 + (cluster['size'] / 50.0)
            marker.scale.z = 0.1
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 0.7
            marker_array.markers.append(marker)
        
        self.frontier_pub.publish(marker_array)
        
        # Publish goal marker
        if goal:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'goal'
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = goal['x']
            marker.pose.position.y = goal['y']
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 0.707
            marker.pose.orientation.z = 0.707
            marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.1, 0.1
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0
            self.goal_pub.publish(marker)
    
    def exploration_callback(self):
        """Main exploration loop - called periodically"""
        # Check if currently navigating
        if self.state == ExplorationState.NAVIGATING:
            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().info('✅ Reached goal!')
                    self.failed_goals = []  # Reset failed goals
                else:
                    self.get_logger().warn(f'⚠️ Navigation result: {result}')
                    # Lưu goal đã fail để tránh chọn lại
                    if hasattr(self, 'current_goal') and self.current_goal:
                        if not hasattr(self, 'failed_goals'):
                            self.failed_goals = []
                        self.failed_goals.append(self.current_goal)
                        self.get_logger().info(f'Added to failed goals list ({len(self.failed_goals)} total)')
                self.state = ExplorationState.IDLE
            return
        
        # if self.state == ExplorationState.COMPLETED:
        #     return
        
        self.state = ExplorationState.EXPLORING
        
        # Find frontiers
        frontiers = self.find_frontiers()
        clusters = self.cluster_frontiers(frontiers)
        
        self.get_logger().info(
            f'📍 Found {len(frontiers)} frontier cells, {len(clusters)} clusters'
        )
        
        if not clusters:
            # self.get_logger().info('🎉 Exploration COMPLETE! No more frontiers.')
            self.get_logger().info('⏳ No frontiers yet, waiting for map data...')
            # self.state = ExplorationState.COMPLETED
            self.state = ExplorationState.IDLE
            return
        
        # Lọc bỏ các goals đã fail
        if hasattr(self, 'failed_goals') and self.failed_goals:
            valid_clusters = []
            for cluster in clusters:
                is_failed = False
                for failed in self.failed_goals:
                    dist = np.sqrt((cluster['x'] - failed['x'])**2 + (cluster['y'] - failed['y'])**2)
                    if dist < 0.5:  # Nếu gần goal đã fail
                        is_failed = True
                        break
                if not is_failed:
                    valid_clusters.append(cluster)
            clusters = valid_clusters
            
            if not clusters:
                self.get_logger().warn('All frontiers have failed before, clearing failed list...')
                self.failed_goals = []
                return

        # Select best frontier
        target = self.select_best_frontier(clusters)
        if target is None:
            self.get_logger().warn('No valid frontier found')
            self.state = ExplorationState.IDLE
            return
        
        self.current_goal = target  # Lưu goal hiện tại
        # Visualize
        self.publish_visualization(clusters, target)
        
        self.get_logger().info(
            f'🚀 Navigating to frontier: ({target["x"]:.2f}, {target["y"]:.2f}) '
            f'[size: {target["size"]}]'
        )
        
        # Create and send goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = target['x']
        goal_pose.pose.position.y = target['y']
        goal_pose.pose.orientation.w = 1.0
        
        self.navigator.goToPose(goal_pose)
        self.state = ExplorationState.NAVIGATING


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()