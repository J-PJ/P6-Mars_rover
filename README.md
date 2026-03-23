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




# USB camera not needed for export; we just want TensorRT build
docker run --rm -it --runtime nvidia \
  -v /home/$USER/models/probe_yolo26:/models \
  yolo-nano:latest \
  python3 - << 'PY'
from ultralytics import YOLO
m = YOLO("/models/best.pt")
m.export(format="engine", imgsz=640, half=True)  # creates /models/best.engine
print("Export done.")
PY




xhost +local:root   # allow X from containers
docker run --rm -it --runtime nvidia \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -e MODEL_PATH=/models/best.engine \
  -e SOURCE=0 \
  -v /home/$USER/models/probe_yolo26:/models:ro \
  --device /dev/video0:/dev/video0 \
  yolo-nano:latest
