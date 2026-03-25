import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

MODEL_PATH  = "models/best.pt"
CONFIDENCE  = 0.5
DISPLAY     = True

def init_realsense():
    pipeline = rs.pipeline()
    config   = rs.config()

    # Enable RGB and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Align depth to color frame so pixels match 1:1
    align = rs.align(rs.stream.color)

    # Get depth scale (converts raw units to metres)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    return pipeline, align, depth_scale

def get_3d_position(depth_frame, x_img, y_img, depth_scale, intrinsics):
    """Return real-world X, Y, Z in metres for a given pixel."""
    depth_value = depth_frame.get_distance(x_img, y_img)  # already in metres
    if depth_value == 0:
        return None, None, None

    # Deproject pixel to 3D point using camera intrinsics
    point = rs.rs2_deproject_pixel_to_point(
        intrinsics, [x_img, y_img], depth_value
    )
    return point[0], point[1], point[2]

def main():
    print("Opening RealSense camera...")
    pipeline, align, depth_scale = init_realsense()
    print("RealSense camera opened.")

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    try:
        # Get camera intrinsics for 3D deprojection
        profile     = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics   = color_stream.as_video_stream_profile().get_intrinsics()

        while True:
            frames        = pipeline.wait_for_frames()
            aligned       = align.process(frames)
            color_frame   = aligned.get_color_frame()
            depth_frame   = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            results = model(frame, conf=CONFIDENCE, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = model.names[cls_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                X, Y, Z = get_3d_position(
                    depth_frame, cx, cy, depth_scale, intrinsics
                )

                if Z is not None:
                    info = f"{label} {conf:.2f} | X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m"
                else:
                    info = f"{label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, info, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if DISPLAY:
                cv2.imshow("YOLO + RealSense D435i", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        pipeline.stop()
        if DISPLAY:
            cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
