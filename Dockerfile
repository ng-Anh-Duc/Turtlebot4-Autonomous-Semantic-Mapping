#############################################
# Dockerfile cho TurtleBot4 Semantic Mapping
# Base: ROS2 Humble + Ubuntu 22.04 + VNC
#############################################

FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# ============================================
# Cài đặt tất cả packages
# ============================================
RUN apt-get update && apt-get install -y \
    # Python tools
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    # ROS2 Navigation packages
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-tf2-tools \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf-transformations \
    # TurtleBot4 packages
    ros-humble-turtlebot4-simulator \
    ros-humble-turtlebot4-desktop \
    ros-humble-turtlebot4-navigation \
    ros-humble-turtlebot4-msgs \
    ros-humble-nav2-simple-commander \
    ros-humble-rosbridge-server \
    # Development tools
    git \
    vim \
    wget \
    curl \
    htop \
    # VNC packages
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    # OpenGL software rendering
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgbm1 \
    # X11 for GUI
    x11-apps \
    xterm \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Python packages cho AI/ML
# ============================================
RUN pip3 install --no-cache-dir \
    ultralytics \
    scikit-learn \
    numpy \
    opencv-python-headless \
    transforms3d

# ============================================
# Clone explore_lite
# ============================================
WORKDIR /ros2_ws/src
RUN git clone https://github.com/robo-friends/m-explore-ros2.git

# ============================================
# Build workspace
# ============================================
WORKDIR /ros2_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# ============================================
# VNC Startup Script
# ============================================
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Cleanup old X locks\n\
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true\n\
\n\
# Environment for software rendering\n\
export DISPLAY=:1\n\
export LIBGL_ALWAYS_SOFTWARE=1\n\
export MESA_GL_VERSION_OVERRIDE=3.3\n\
export GALLIUM_DRIVER=llvmpipe\n\
export QT_QUICK_BACKEND=software\n\
\n\
# Start Xvfb (virtual framebuffer)\n\
Xvfb :1 -screen 0 1920x1080x24 &\n\
sleep 2\n\
\n\
# Start window manager\n\
fluxbox &\n\
sleep 1\n\
\n\
# Start VNC server\n\
x11vnc -display :1 -forever -nopw -shared -rfbport 5900 &\n\
\n\
# Start noVNC (web-based VNC client)\n\
websockify --web /usr/share/novnc 6080 localhost:5900 &\n\
\n\
# Source ROS2\n\
source /opt/ros/humble/setup.bash\n\
source /ros2_ws/install/setup.bash 2>/dev/null || true\n\
\n\
echo ""\n\
echo "========================================"\n\
echo "  VNC Ready!"\n\
echo "  Open browser: http://localhost:6080/vnc.html"\n\
echo "========================================"\n\
echo ""\n\
\n\
exec "$@"' > /start-vnc.sh && chmod +x /start-vnc.sh

# ============================================
# Setup bashrc
# ============================================
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> ~/.bashrc && \
    echo "export DISPLAY=:1" >> ~/.bashrc && \
    echo "export LIBGL_ALWAYS_SOFTWARE=1" >> ~/.bashrc && \
    echo "export MESA_GL_VERSION_OVERRIDE=3.3" >> ~/.bashrc && \
    echo "export GALLIUM_DRIVER=llvmpipe" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc

# ============================================
# Create directories
# ============================================
RUN mkdir -p /ros2_ws/maps

WORKDIR /ros2_ws

ENTRYPOINT ["/start-vnc.sh"]
CMD ["bash"]