FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-opencv git wget curl \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Install YOLOv8 ---
RUN pip install --upgrade pip
RUN pip install ultralytics aiortc opencv-python-headless

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libqt5gui5 \
    libqt5core5a \
    qtbase5-dev

RUN pip install aiohttp aiortc av opencv-python-headless


# --- Optional: small WebRTC server script later ---
WORKDIR /workspace
CMD ["bash"]
