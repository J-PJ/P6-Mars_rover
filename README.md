# P6-Mars_rover
This is a 6 semester projekt about atonemus navigation on mars

How to start yolo

source ~/yolo/bin/activate

yolo detect predict   
model=/home/jens0913/runs/detect/runs/train_probe/y26s_probe8/weights/best.pt   #parth to where you have best.pt
source=0   
show=True





Dockerbild


sudo apt update
sudo apt install -y nvidia-container-runtime
sudo systemctl restart docker




cd ~/yolo_nano_docker
docker build -t yolo-nano:latest .




docker run --rm -it --runtime nvidia \
  -v /home/yolo_nano_docker/models:/models \
  yolo-nano:latest \
  python3 - << 'PY'
from ultralytics import YOLO
m = YOLO("/models/best.pt")
m.export(format="engine", imgsz=640, half=True)  # will write /models/best.engine
print("Export complete -> /models/best.engine")
PY


xhost +local:root




docker run --rm -it --runtime nvidia \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e MODEL_PATH=/models/best.engine \
  -e SOURCE=0 \
  -v /home/yolo_nano_docker/models:/models:ro \
  --device /dev/video0:/dev/video0 \
  yolo-nano:latest


