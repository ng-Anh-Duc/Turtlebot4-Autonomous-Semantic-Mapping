#!/usr/bin/env python3
"""
Frontier Explorer V2.1 - Fixed Cost Map Based Exploration
Fixes:
1. Better obstacle avoidance with clear direction finding
2. Stuck detection and recovery
3. Local waypoint planning (not just gradient)
4. Smoother velocity control
5. Frontier validation (reachability check)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
import numpy as np
import math
from collections import deque
from enum import Enum
import time


class RobotState(Enum):
    EXPLORING = 1
    ROTATING_TO_TARGET = 2
    MOVING_TO_TARGET = 3
    AVOIDING_OBSTACLE = 4
    STUCK_RECOVERY = 5
    COMPLETED = 6


class FrontierExplorerV2_1(Node):
    """Improved Cost Map based frontier exploration with bug fixes."""
    
    # Map cell values
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100
    
    def __init__(self):
        super().__init__('frontier_explorer_v2_1')
        
        # Parameters
        self.declare_parameter('linear_speed', 0.20)
        self.declare_parameter('angular_speed', 0.6)
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('min_frontier_size', 2)
        self.declare_parameter('update_rate', 3.0)
        self.declare_parameter('safety_distance', 0.45)
        self.declare_parameter('obstacle_threshold', 0.4)
        self.declare_parameter('stuck_timeout', 5.0)
        self.declare_parameter('waypoint_distance', 1.0)  # Local waypoint distance
        
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.update_rate = self.get_parameter('update_rate').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.stuck_timeout = self.get_parameter('stuck_timeout').value
        self.waypoint_distance = self.get_parameter('waypoint_distance').value
        
        # State
        self.state = RobotState.EXPLORING
        self.map_data = None
        self.map_info = None
        self.robot_pose = None
        self.scan_data = None
        self.cost_map = None
        self.current_target = None
        self.local_waypoint = None
        
        # Stuck detection
        self.last_position = None
        self.last_position_time = None
        self.stuck_count = 0
        self.recovery_direction = 1  # 1 = left, -1 = right
        self.recovery_start_time = None  # For time-based recovery
        self.obstacle_avoid_start = None  # For obstacle avoidance timeout
        
        # Exploration tracking
        self.explored_targets = set()
        self.failed_targets = set()
        
        # QoS for map
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.exploration_loop)
        
        self.get_logger().info('🚀 Frontier Explorer V2.1 (Improved) started!')
        self.get_logger().info(f'   Linear speed: {self.linear_speed} m/s')
        self.get_logger().info(f'   Angular speed: {self.angular_speed} rad/s')
        self.get_logger().info(f'   Obstacle threshold: {self.obstacle_threshold} m')
    
    def map_callback(self, msg):
        """Store map data."""
        # Use int8 to correctly handle -1 for unknown cells
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        if self.state == RobotState.EXPLORING:
            self.get_logger().info('📍 Map received, starting exploration...')
    
    def scan_callback(self, msg):
        """Store scan data."""
        self.scan_data = msg
    
    def get_robot_pose(self):
        """Get robot pose from TF."""
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None
    
    def world_to_map(self, x, y):
        """Convert world coordinates to map cell."""
        mx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return (mx, my)
    
    def map_to_world(self, mx, my):
        """Convert map cell to world coordinates."""
        x = mx * self.map_info.resolution + self.map_info.origin.position.x
        y = my * self.map_info.resolution + self.map_info.origin.position.y
        return (x, y)
    
    def is_valid_cell(self, x, y):
        """Check if cell is within map bounds."""
        return 0 <= x < self.map_info.width and 0 <= y < self.map_info.height
    
    def is_free(self, x, y):
        """Check if cell is free (low probability of obstacle)."""
        if not self.is_valid_cell(x, y):
            return False
        cell_value = self.map_data[y, x]
        # Cartographer uses probability values:
        # -1 = unknown, 0-50 = likely free, 50-100 = likely occupied
        # Expand threshold to 50 to include frontier-adjacent cells
        return 0 <= cell_value <= 50
    
    def is_unknown(self, x, y):
        """Check if cell is unknown."""
        if not self.is_valid_cell(x, y):
            return False
        cell_value = self.map_data[y, x]
        # Unknown can be -1 (int8) or stored differently
        # Cartographer uses -1 for unknown
        return cell_value == -1 or cell_value == self.UNKNOWN
    
    def is_occupied(self, x, y):
        """Check if cell is occupied."""
        if not self.is_valid_cell(x, y):
            return True
        return self.map_data[y, x] > 50
    
    def has_unknown_neighbor(self, x, y):
        """Check if cell has unknown neighbor (8 directions)."""
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_cell(nx, ny) and self.is_unknown(nx, ny):
                return True
        return False
    
    def wfd_find_frontiers(self, start_cell):
        """
        Wavefront Frontier Detection - find frontiers reachable from robot.
        Only returns frontiers that are connected to robot's position.
        """
        # Debug: count cell types with actual values
        unique_values, counts = np.unique(self.map_data, return_counts=True)
        value_dist = dict(zip(unique_values, counts))
        
        # Log value distribution (only first time or occasionally)
        if not hasattr(self, '_logged_dist') or self._logged_dist < 3:
            self._logged_dist = getattr(self, '_logged_dist', 0) + 1
            self.get_logger().info(f'📊 Map value distribution: {value_dist}')
        
        free_count = np.sum((self.map_data >= 0) & (self.map_data <= 50))
        unknown_count = np.sum(self.map_data == self.UNKNOWN)
        
        # Also check for other potential unknown values
        neg_count = np.sum(self.map_data < 0)
        high_unknown = np.sum(self.map_data == 255)  # In case stored as uint8
        
        if free_count < 10:
            self.get_logger().warn(f'⚠️ Map has very few free cells: {free_count} free, {unknown_count} unknown (neg={neg_count}, 255={high_unknown})')
            return []
        
        # Check if robot position is valid
        robot_cell_value = self.map_data[start_cell[1], start_cell[0]] if self.is_valid_cell(start_cell[0], start_cell[1]) else -999
        
        if not self.is_free(start_cell[0], start_cell[1]):
            self.get_logger().info(f'📍 Robot cell ({start_cell[0]}, {start_cell[1]}) value={robot_cell_value}, finding nearest free cell...')
            # Find nearest free cell with larger search radius
            start_cell = self.find_nearest_free_cell(start_cell)
            if start_cell is None:
                self.get_logger().warn('⚠️ Could not find any free cell near robot!')
                return []
            self.get_logger().info(f'📍 Using free cell: ({start_cell[0]}, {start_cell[1]})')
        
        frontiers = []
        visited = set()
        frontier_visited = set()
        
        # BFS from robot position
        queue = deque([start_cell])
        visited.add(start_cell)
        
        while queue:
            x, y = queue.popleft()
            
            # Check if this is a frontier cell
            if self.has_unknown_neighbor(x, y) and (x, y) not in frontier_visited:
                # Start inner BFS to find connected frontier cells
                frontier_cluster = self.extract_frontier_cluster(x, y, frontier_visited)
                if len(frontier_cluster) >= self.min_frontier_size:
                    frontiers.append(frontier_cluster)
            
            # Expand to free neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and self.is_free(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Debug output
        if not frontiers:
            self.get_logger().warn(f'⚠️ No frontiers found! Visited {len(visited)} cells. Free cells in map: {free_count}')
        
        return frontiers
    
    def find_nearest_free_cell(self, start):
        """Find nearest free cell using BFS with larger search radius."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        # Search up to 1000 cells
        max_iterations = 1000
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.popleft()
            
            # Check if this cell is free (value 0-50)
            if self.is_valid_cell(x, y):
                cell_value = self.map_data[y, x]
                if 0 <= cell_value <= 50:  # Free cell
                    return (x, y)
            
            # Expand search
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if self.is_valid_cell(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        self.get_logger().warn(f'⚠️ find_nearest_free_cell: searched {iterations} cells, no free cell found')
        return None
    
    def extract_frontier_cluster(self, start_x, start_y, visited):
        """Extract connected frontier cells starting from a point."""
        cluster = []
        queue = deque([(start_x, start_y)])
        visited.add((start_x, start_y))
        
        while queue:
            x, y = queue.popleft()
            
            if self.is_free(x, y) and self.has_unknown_neighbor(x, y):
                cluster.append((x, y))
                
                # Add neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited and self.is_valid_cell(nx, ny):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return cluster
    
    def compute_cost_map(self, frontier_clusters, robot_cell):
        """
        Compute cost map using multi-source BFS from frontier cells.
        Only considers frontiers reachable from robot.
        """
        height, width = self.map_data.shape
        cost_map = np.full((height, width), np.inf)
        
        # Initialize queue with all frontier cells
        queue = deque()
        for cluster in frontier_clusters:
            for fx, fy in cluster:
                if self.is_free(fx, fy):
                    cost_map[fy, fx] = 0
                    queue.append((fx, fy, 0))
        
        # BFS propagation
        while queue:
            x, y, cost = queue.popleft()
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                new_cost = cost + 1
                
                if (self.is_valid_cell(nx, ny) and 
                    self.is_free(nx, ny) and 
                    new_cost < cost_map[ny, nx]):
                    cost_map[ny, nx] = new_cost
                    queue.append((nx, ny, new_cost))
        
        return cost_map
    
    def select_best_frontier(self, frontier_clusters, robot_cell):
        """Select best frontier based on size and distance."""
        if not frontier_clusters:
            return None
        
        best_frontier = None
        best_score = -np.inf
        
        robot_world = self.map_to_world(robot_cell[0], robot_cell[1])
        
        for cluster in frontier_clusters:
            # Compute centroid
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            centroid = (int(cx), int(cy))
            
            # Skip if already explored or failed
            centroid_key = (centroid[0] // 5, centroid[1] // 5)  # Grid key
            if centroid_key in self.explored_targets or centroid_key in self.failed_targets:
                continue
            
            # Compute score: balance size vs distance
            centroid_world = self.map_to_world(centroid[0], centroid[1])
            distance = math.sqrt((centroid_world[0] - robot_world[0])**2 + 
                                (centroid_world[1] - robot_world[1])**2)
            
            # Prefer larger frontiers that are closer
            # Score = size / (distance + 1)
            score = len(cluster) / (distance + 1.0)
            
            if score > best_score:
                best_score = score
                best_frontier = {
                    'cells': cluster,
                    'centroid': centroid,
                    'size': len(cluster),
                    'distance': distance
                }
        
        return best_frontier
    
    def find_local_waypoint(self, robot_cell, target_world):
        """
        Find a local waypoint toward the target that is reachable.
        Uses A* like approach for short distance.
        """
        if self.cost_map is None:
            return target_world
        
        robot_world = self.map_to_world(robot_cell[0], robot_cell[1])
        
        # Direction to target
        dx = target_world[0] - robot_world[0]
        dy = target_world[1] - robot_world[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 0.1:
            return target_world
        
        # Normalize and scale to waypoint distance
        waypoint_dist = min(self.waypoint_distance, dist)
        wx = robot_world[0] + (dx / dist) * waypoint_dist
        wy = robot_world[1] + (dy / dist) * waypoint_dist
        
        # Check if direct path is clear
        if self.is_path_clear(robot_world, (wx, wy)):
            return (wx, wy)
        
        # If not clear, find alternative waypoint using cost map gradient
        return self.find_gradient_waypoint(robot_cell)
    
    def is_path_clear(self, start, end, num_checks=10):
        """Check if path between two world points is clear."""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            mx, my = self.world_to_map(x, y)
            if not self.is_valid_cell(mx, my) or self.is_occupied(mx, my):
                return False
        
        return True
    
    def find_gradient_waypoint(self, robot_cell):
        """Find waypoint by following cost map gradient."""
        if self.cost_map is None:
            return None
        
        x, y = robot_cell
        current_cost = self.cost_map[y, x]
        
        if np.isinf(current_cost):
            return None
        
        # Find neighbor with lowest cost that is not occupied
        best_cell = None
        best_cost = current_cost
        
        # Check in larger radius for smoother movement
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if not self.is_valid_cell(nx, ny):
                    continue
                
                if self.is_occupied(nx, ny):
                    continue
                
                cell_cost = self.cost_map[ny, nx]
                
                # Prefer cells that reduce cost AND are free
                if cell_cost < best_cost and self.is_free(nx, ny):
                    # Additional check: path should be clear
                    start_world = self.map_to_world(x, y)
                    end_world = self.map_to_world(nx, ny)
                    if self.is_path_clear(start_world, end_world, 5):
                        best_cost = cell_cost
                        best_cell = (nx, ny)
        
        if best_cell:
            return self.map_to_world(best_cell[0], best_cell[1])
        
        return None
    
    def check_obstacle_ahead(self):
        """Check for obstacles using laser scan with directional info."""
        if self.scan_data is None:
            return False, 0
        
        ranges = np.array(self.scan_data.ranges)
        angle_min = self.scan_data.angle_min
        angle_increment = self.scan_data.angle_increment
        
        # Check front arc (-45 to +45 degrees)
        front_start = int((-math.pi/4 - angle_min) / angle_increment)
        front_end = int((math.pi/4 - angle_min) / angle_increment)
        
        front_start = max(0, front_start)
        front_end = min(len(ranges), front_end)
        
        front_ranges = ranges[front_start:front_end]
        valid_ranges = front_ranges[np.isfinite(front_ranges) & (front_ranges > 0.1)]
        
        if len(valid_ranges) == 0:
            return False, 0
        
        min_range = np.min(valid_ranges)
        
        # Find direction of closest obstacle
        min_idx = np.argmin(valid_ranges)
        obstacle_angle = angle_min + (front_start + min_idx) * angle_increment
        
        return min_range < self.obstacle_threshold, obstacle_angle
    
    def find_clear_direction(self):
        """Find the clearest direction to move, preferring direction toward target."""
        if self.scan_data is None:
            return self.robot_pose[2] if self.robot_pose else 0
        
        ranges = np.array(self.scan_data.ranges)
        angle_min = self.scan_data.angle_min
        angle_increment = self.scan_data.angle_increment
        
        # Calculate target direction if available
        target_angle = None
        if self.current_target and self.robot_pose:
            target_world = self.map_to_world(
                self.current_target['centroid'][0],
                self.current_target['centroid'][1])
            dx = target_world[0] - self.robot_pose[0]
            dy = target_world[1] - self.robot_pose[1]
            target_angle = math.atan2(dy, dx)
        
        # Divide into sectors and find clearest
        num_sectors = 12  # More sectors for finer resolution
        sector_size = len(ranges) // num_sectors
        
        best_sector = 0
        best_score = -float('inf')
        
        for i in range(num_sectors):
            start = i * sector_size
            end = (i + 1) * sector_size
            sector_ranges = ranges[start:end]
            valid = sector_ranges[np.isfinite(sector_ranges) & (sector_ranges > 0.1)]
            
            if len(valid) == 0:
                continue
            
            # Average range in sector (higher is better)
            avg_range = np.mean(valid)
            min_range = np.min(valid)
            
            # Skip sectors with obstacles too close
            if min_range < 0.3:
                continue
            
            # Calculate sector angle
            sector_angle = angle_min + (i + 0.5) * sector_size * angle_increment
            
            # Score = range + bonus for being toward target
            score = avg_range
            
            if target_angle is not None:
                # Add bonus for directions toward target
                angle_to_target = abs(self.normalize_angle(sector_angle - (target_angle - self.robot_pose[2])))
                target_bonus = 1.0 - (angle_to_target / math.pi)  # 0-1 bonus
                score += target_bonus * 2.0  # Weight target direction
            
            if score > best_score:
                best_score = score
                best_sector = i
        
        # Convert sector to world angle
        sector_angle = angle_min + (best_sector + 0.5) * sector_size * angle_increment
        return self.robot_pose[2] + sector_angle if self.robot_pose else sector_angle
    
    def check_stuck(self):
        """Check if robot is stuck (not moving)."""
        if self.robot_pose is None:
            return False
        
        current_pos = (self.robot_pose[0], self.robot_pose[1])
        current_time = time.time()
        
        if self.last_position is None:
            self.last_position = current_pos
            self.last_position_time = current_time
            return False
        
        # Check distance moved
        dist = math.sqrt((current_pos[0] - self.last_position[0])**2 +
                        (current_pos[1] - self.last_position[1])**2)
        
        time_elapsed = current_time - self.last_position_time
        
        if time_elapsed > self.stuck_timeout:
            if dist < 0.1:  # Moved less than 10cm in stuck_timeout seconds
                self.get_logger().warn(f'🚫 Robot appears stuck! Moved only {dist:.2f}m in {time_elapsed:.1f}s')
                self.last_position = current_pos
                self.last_position_time = current_time
                return True
            else:
                self.last_position = current_pos
                self.last_position_time = current_time
        
        return False
    
    def compute_velocity(self, robot_pose, target):
        """Compute velocity command to reach target."""
        dx = target[0] - robot_pose[0]
        dy = target[1] - robot_pose[1]
        
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_angle - robot_pose[2])
        
        twist = Twist()
        
        # If angle is large, rotate first
        if abs(angle_diff) > 0.3:
            twist.angular.z = self.angular_speed * np.sign(angle_diff)
            twist.linear.x = 0.0
        else:
            # Move forward with proportional angular correction
            twist.linear.x = min(self.linear_speed, distance * 0.5)
            twist.angular.z = 1.5 * angle_diff  # Proportional control
        
        return twist
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def exploration_loop(self):
        """Main exploration loop."""
        # Check prerequisites
        if self.map_data is None or self.map_info is None:
            return
        
        self.robot_pose = self.get_robot_pose()
        if self.robot_pose is None:
            return
        
        robot_cell = self.world_to_map(self.robot_pose[0], self.robot_pose[1])
        
        # State machine
        if self.state == RobotState.COMPLETED:
            self.cmd_vel_pub.publish(Twist())
            return
        
        # Warmup: wait for map to have enough data
        free_count = np.sum((self.map_data >= 0) & (self.map_data <= 50))
        if free_count < 50:
            self.get_logger().info(f'⏳ Waiting for map data... ({free_count} free cells)')
            # Rotate slowly to build initial map
            twist = Twist()
            twist.angular.z = 0.3
            self.cmd_vel_pub.publish(twist)
            return
        
        # Check if stuck
        if self.check_stuck():
            self.state = RobotState.STUCK_RECOVERY
            self.stuck_count += 1
            self.recovery_direction *= -1  # Alternate direction
            self.recovery_start_time = None  # Reset recovery timer
            self.obstacle_avoid_start = None  # Reset obstacle timer
            
            if self.stuck_count > 5:
                # Mark current target as failed
                if self.current_target:
                    centroid_key = (self.current_target['centroid'][0] // 5,
                                   self.current_target['centroid'][1] // 5)
                    self.failed_targets.add(centroid_key)
                    self.current_target = None
                self.stuck_count = 0
        
        # Handle stuck recovery
        if self.state == RobotState.STUCK_RECOVERY:
            self.handle_stuck_recovery()
            return
        
        # Find frontiers
        frontier_clusters = self.wfd_find_frontiers(robot_cell)
        
        if not frontier_clusters:
            # Don't immediately declare complete - try rotating to discover more
            if not hasattr(self, 'no_frontier_count'):
                self.no_frontier_count = 0
            
            self.no_frontier_count += 1
            
            if self.no_frontier_count < 20:  # Try for ~4 seconds
                self.get_logger().info(f'🔄 No frontiers found, rotating to discover... ({self.no_frontier_count}/20)')
                twist = Twist()
                twist.angular.z = 0.4
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self.get_logger().info('✅ Exploration complete! No more frontiers.')
                self.state = RobotState.COMPLETED
                self.cmd_vel_pub.publish(Twist())
                return
        else:
            self.no_frontier_count = 0  # Reset counter when frontiers found
        
        total_cells = sum(len(c) for c in frontier_clusters)
        
        # Select best frontier if needed
        if self.current_target is None:
            self.current_target = self.select_best_frontier(frontier_clusters, robot_cell)
            
            if self.current_target is None:
                self.get_logger().warn('⚠️ No valid frontier found, clearing failed list...')
                self.failed_targets.clear()
                return
            
            self.get_logger().info(
                f'🎯 New target: size={self.current_target["size"]}, '
                f'dist={self.current_target["distance"]:.1f}m')
        
        # Compute cost map
        self.cost_map = self.compute_cost_map(frontier_clusters, robot_cell)
        
        # Get target world position
        target_world = self.map_to_world(
            self.current_target['centroid'][0],
            self.current_target['centroid'][1])
        
        # Check if reached target
        dist_to_target = math.sqrt(
            (target_world[0] - self.robot_pose[0])**2 +
            (target_world[1] - self.robot_pose[1])**2)
        
        if dist_to_target < self.goal_tolerance:
            self.get_logger().info(f'✓ Reached frontier!')
            centroid_key = (self.current_target['centroid'][0] // 5,
                           self.current_target['centroid'][1] // 5)
            self.explored_targets.add(centroid_key)
            self.current_target = None
            self.stuck_count = 0
            return
        
        # Find local waypoint
        self.local_waypoint = self.find_local_waypoint(robot_cell, target_world)
        
        if self.local_waypoint is None:
            self.get_logger().warn('⚠️ No valid waypoint, trying recovery...')
            self.recovery_start_time = None  # Ensure fresh recovery
            self.obstacle_avoid_start = None
            self.state = RobotState.STUCK_RECOVERY
            return
        
        # Check for obstacles
        obstacle_ahead, obstacle_angle = self.check_obstacle_ahead()
        
        if obstacle_ahead:
            self.state = RobotState.AVOIDING_OBSTACLE
            self.handle_obstacle_avoidance(obstacle_angle)
        else:
            self.state = RobotState.MOVING_TO_TARGET
            self.obstacle_avoid_start = None  # Reset obstacle timer when path is clear
            # Move toward local waypoint
            twist = self.compute_velocity(self.robot_pose, self.local_waypoint)
            self.cmd_vel_pub.publish(twist)
        
        # Log periodically
        self.get_logger().info(
            f'🔍 Frontiers: {len(frontier_clusters)} ({total_cells} cells) | '
            f'Target dist: {dist_to_target:.1f}m | State: {self.state.name}')
    
    def handle_obstacle_avoidance(self, obstacle_angle):
        """Handle obstacle avoidance - more aggressive movement."""
        twist = Twist()
        
        # Track obstacle avoidance time
        if not hasattr(self, 'obstacle_avoid_start') or self.obstacle_avoid_start is None:
            self.obstacle_avoid_start = self.get_clock().now()
        
        elapsed = (self.get_clock().now() - self.obstacle_avoid_start).nanoseconds / 1e9
        
        # If stuck in obstacle avoidance for too long, trigger recovery
        if elapsed > 15.0:
            self.get_logger().warn('⚠️ Obstacle avoidance timeout, triggering recovery...')
            self.obstacle_avoid_start = None
            self.recovery_start_time = None  # Ensure fresh recovery
            self.state = RobotState.STUCK_RECOVERY
            return
        
        # Find clear direction
        clear_direction = self.find_clear_direction()
        
        # Calculate angle difference to clear direction
        angle_diff = self.normalize_angle(clear_direction - self.robot_pose[2])
        
        if abs(angle_diff) > 0.3:
            # Need to rotate significantly
            twist.angular.z = self.angular_speed * np.sign(angle_diff)
            twist.linear.x = 0.0
        else:
            # Aligned enough, move forward aggressively
            twist.linear.x = self.linear_speed * 0.8  # Move faster
            twist.angular.z = 0.3 * angle_diff  # Small correction
            # Reset obstacle timer since we're making progress
            self.obstacle_avoid_start = None
        
        self.cmd_vel_pub.publish(twist)
    
    def handle_stuck_recovery(self):
        """Recovery behavior when stuck - time-based."""
        twist = Twist()
        
        # Initialize recovery timer if not set
        if not hasattr(self, 'recovery_start_time') or self.recovery_start_time is None:
            self.recovery_start_time = self.get_clock().now()
            self.recovery_phase = 0  # 0=backup, 1=rotate
        
        # Calculate elapsed time
        elapsed = (self.get_clock().now() - self.recovery_start_time).nanoseconds / 1e9
        
        # Recovery phases: 2s backup, 3s rotate, then exit
        if elapsed < 2.0:
            # Phase 1: Back up
            twist.linear.x = -0.15
            twist.angular.z = 0.0
            if self.recovery_phase != 0:
                self.recovery_phase = 0
                self.get_logger().info('🔄 Recovery: backing up (2s)...')
        elif elapsed < 5.0:
            # Phase 2: Rotate to find clear path
            twist.linear.x = 0.0
            twist.angular.z = self.angular_speed * self.recovery_direction
            if self.recovery_phase != 1:
                self.recovery_phase = 1
                direction = "left" if self.recovery_direction > 0 else "right"
                self.get_logger().info(f'🔄 Recovery: rotating {direction} (3s)...')
        else:
            # Recovery complete - exit and try new target
            self.get_logger().info('✅ Recovery complete, finding new target...')
            self.recovery_start_time = None
            self.state = RobotState.EXPLORING
            
            # Mark current target as problematic
            if self.current_target:
                centroid_key = (self.current_target['centroid'][0] // 5,
                               self.current_target['centroid'][1] // 5)
                self.failed_targets.add(centroid_key)
            
            self.current_target = None
            self.stuck_count = 0
            return
        
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorerV2_1()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Stop robot
        try:
            node.cmd_vel_pub.publish(Twist())
        except:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()