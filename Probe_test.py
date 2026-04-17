#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import pyrealsense2 as rs
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

# ---------------- USER SETTINGS ----------------
MODEL_PATH = "models/best.pt"
YOLO_CONF  = 0.25
YOLO_INTERVAL = 1.0   # seconds → 1 FPS
MIN_DEPTH = 0.05      # meters
MAX_DEPTH = 5.0       # meters
# ----------------------------------------------

def get_depth_bottom_strip(depth_frame, x1, y1, x2, y2):
    h = depth_frame.get_height()
    y = min(h - 1, y2 + 10)
    depths = []
    for x in range(x1, x2, 4):
        d = depth_frame.get_distance(x, y)
        if d > 0:
            depths.append(d)
    return float(np.median(depths)) if depths else float("nan")

def draw_detections(frame, boxes, depth_frame, fx, cx0):
    """Draw bounding boxes and info onto frame, return annotated copy."""
    vis = frame.copy()
    if boxes is None:
        return vis

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        depth = get_depth_bottom_strip(depth_frame, x1, y1, x2, y2)
        valid = MIN_DEPTH <= depth <= MAX_DEPTH

        color = (0, 255, 0) if valid else (0, 0, 255)  # green=valid, red=out of range

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        if valid:
            cx = (x1 + x2) / 2.0
            dx = cx - cx0
            angle_deg = math.degrees(math.atan(dx / fx))
            lateral   = depth * math.tan(math.atan(dx / fx))
            label = f"d={depth:.2f}m  lat={lateral:.2f}m  {angle_deg:.1f}deg  [{conf:.2f}]"
        else:
            depth_str = f"{depth:.2f}m" if not math.isnan(depth) else "no depth"
            label = f"OUT OF RANGE ({depth_str})  [{conf:.2f}]"

        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Centre crosshair
        cx_px = (x1 + x2) // 2
        cy_px = (y1 + y2) // 2
        cv2.drawMarker(vis, (cx_px, cy_px), color,
                       cv2.MARKER_CROSS, 12, 2)

    # HUD: timestamp + FPS hint
    cv2.putText(vis, f"YOLO @ 1 FPS  |  press Q to quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    return vis


class ProbeDetector(Node):
    def __init__(self):
        super().__init__("probe_detector")

        self.pub = self.create_publisher(PointStamped, "probe/coordinates", 10)

        self.get_logger().info("Loading YOLO (CPU-only, 1 FPS)...")
        self.model = YOLO(MODEL_PATH)

        # --- RealSense setup ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        intr = (profile.get_stream(rs.stream.color)
                       .as_video_stream_profile()
                       .get_intrinsics())
        self.fx  = intr.fx
        self.cx0 = intr.ppx

        self.get_logger().info(
            f"Camera intrinsics: fx={self.fx:.1f}, cx={self.cx0:.1f}"
        )

        self.last_yolo_time = 0.0
        self.last_vis_frame = None   # cache last annotated frame for display

        self.timer = self.create_timer(0.05, self.loop)   # 20 Hz spin
        self.get_logger().info("Probe detector running ✅")

    def loop(self):
        # ---- Poll camera at full rate so the window stays responsive ----
        frames = self.pipeline.wait_for_frames(timeout_ms=100)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())

        # ---- Run YOLO only at 1 FPS ----
        now = time.time()
        if now - self.last_yolo_time >= YOLO_INTERVAL:
            self.last_yolo_time = now

            results = self.model(
                frame,
                conf=YOLO_CONF,
                device="cpu",
                verbose=False
            )[0]

            boxes = results.boxes

            # Publish detections
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    depth = get_depth_bottom_strip(depth_frame, x1, y1, x2, y2)
                    if not (MIN_DEPTH <= depth <= MAX_DEPTH):
                        continue

                    cx      = (x1 + x2) / 2.0
                    dx      = cx - self.cx0
                    angle_rad = math.atan(dx / self.fx)
                    angle_deg = math.degrees(angle_rad)
                    lateral   = depth * math.tan(angle_rad)

                    msg = PointStamped()
                    msg.header.stamp    = self.get_clock().now().to_msg()
                    msg.header.frame_id = "camera_link"
                    msg.point.x = depth
                    msg.point.y = lateral
                    msg.point.z = angle_deg
                    self.pub.publish(msg)

            # Build and cache annotated frame
            self.last_vis_frame = draw_detections(
                frame, boxes, depth_frame, self.fx, self.cx0
            )

        # ---- Display (uses cached frame between YOLO runs) ----
        display = self.last_vis_frame if self.last_vis_frame is not None else frame
        cv2.imshow("Probe Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Q pressed — shutting down.")
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        self.pipeline.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = ProbeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
