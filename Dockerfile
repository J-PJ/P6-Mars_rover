# Jetson Nano = L4T r32.7.1 (JetPack 4.6.1/4.6.3). Use matching ML image.
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-opencv ffmpeg libgl1 libglib2.0-0 v4l-utils \
    gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-{base,good,bad} \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
# Pin ultralytics to a conservative version that works well on Nano
RUN python3 -m pip install "ultralytics<=8.1.0"

WORKDIR /app
COPY app.py /app/app.py

# Runtime configuration (override via -e)
ENV MODEL_PATH=/models/best.engine
ENV CAM_INDEX=0
ENV CONF=0.25

CMD ["python3", "app.py"]
