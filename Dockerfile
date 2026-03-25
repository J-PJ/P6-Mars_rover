FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps + RealSense SDK from Intel's apt repo ─────────────────────
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl wget gnupg2 ca-certificates \
    python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    libusb-1.0-0 libusb-1.0-0-dev \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# ── Add Intel RealSense apt repo ───────────────────────────────────────────
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
    https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ────────────────────────────────────────────────────────────
RUN python3 -m pip install --upgrade pip setuptools wheel

# ── Install PyTorch for Jetson aarch64 ────────────────────────────────────
RUN pip3 install --no-cache-dir \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v61 \
    "torch==2.3.0" "torchvision==0.18.0"

# ── Install RealSense Python wrapper ──────────────────────────────────────
RUN pip3 install pyrealsense2

# ── Install ultralytics and deps ──────────────────────────────────────────
RUN pip3 install --no-deps ultralytics==8.3.0

RUN pip3 install \
    "opencv-python>=4.8.0" \
    "PyYAML>=5.3.1" \
    "requests>=2.23.0" \
    "tqdm>=4.64.0" \
    "matplotlib>=3.3.0" \
    "seaborn>=0.11.0" \
    "Pillow>=9.0.0" \
    "numpy>=1.23.5,<2.0.0" \
    psutil \
    py-cpuinfo

WORKDIR /app
COPY . /app

CMD ["python3", "run_detection.py"]
