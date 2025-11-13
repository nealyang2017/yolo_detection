# =========================================================
# ROS Kinetic Dockerfile (for Shared Memory Reader Node)
# =========================================================
FROM ros:kinetic-ros-base

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=kinetic
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Update & basic tools
RUN apt-get update && apt-get install -y \
    python-pip \
    python-dev \
    python-numpy \
    python-opencv \
    net-tools \
    iputils-ping \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Optional: install roscpp, rospy, std_msgs if not included
RUN apt-get update && apt-get install -y \
    ros-kinetic-roscpp \
    ros-kinetic-rospy \
    ros-kinetic-std-msgs \
    ros-kinetic-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# Source ROS setup on startup
RUN echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# Copy your ROS node source (if exists)
# COPY ./src /root/catkin_ws/src

# Build if necessary
# RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash && catkin_make"

# Shared memory directory
VOLUME ["/dev/shm"]

# Default command
CMD ["/bin/bash"]
