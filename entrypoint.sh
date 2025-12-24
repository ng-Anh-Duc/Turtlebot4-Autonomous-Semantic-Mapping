#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

exec "$@"