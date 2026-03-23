import os, cv2
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/best.engine")
CONF = float(os.environ.get("CONF", "0.25"))
# Use USB by default; for CSI pass a GStreamer string via SOURCE env
SOURCE = os.environ.get("SOURCE", "0")

def open_source(src):
    # If SOURCE is digit-like, open as int; else treat as GStreamer string
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    else:
        return cv2.VideoCapture(src, cv2.CAP_GSTREAMER)

def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Opening source: {SOURCE}")
    cap = open_source(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {SOURCE}")

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] frame read failed")
            break
        results = model.predict(frame, conf=CONF, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("YOLO Nano (TensorRT)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
