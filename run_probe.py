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
    """
    Robust depth sampling from a strip BELOW the bounding box.
    Very stable for ground probes / objects.
    """
    h = depth_frame.get_height()
    y = min(h - 1, y2 + 10)

    depths = []
    for x in range(x1, x2, 4):
        d = depth_frame.get_distance(x, y)
        if d > 0:
            depths.append(d)

    return float(np.median(depths)) if depths else float("nan")


class ProbeDetector(Node):
    def __init__(self):
        super().__init__("probe_detector")

        # --- ROS 2 publisher ---
        self.pub = self.create_publisher(
            PointStamped,
            "probe/coordinates",
            10
        )

        self.get_logger().info("Loading YOLO (CPU-only, 1 FPS)...")
        self.model = YOLO(MODEL_PATH)

        # --- RealSense setup ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color)

        intr = profile.get_stream(rs.stream.color)\
                      .as_video_stream_profile()\
                      .get_intrinsics()

        self.fx  = intr.fx
        self.cx0 = intr.ppx

        self.get_logger().info(
            f"Camera intrinsics: fx={self.fx:.1f}, cx={self.cx0:.1f}"
        )

        self.last_yolo_time = 0.0

        # Run loop at 20 Hz, YOLO throttled internally to 1 Hz
        self.timer = self.create_timer(0.05, self.loop)
        self.get_logger().info("Probe detector running ✅")

    def loop(self):
        # --- Hard YOLO rate limit (1 FPS) ---
        now = time.time()
        if now - self.last_yolo_time < YOLO_INTERVAL:
            return
        self.last_yolo_time = now

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = self.align.process(frames).get_color_frame()

        if not depth_frame or not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())

        # ✅ Explicit CPU inference
        results = self.model(
            frame,
            conf=YOLO_CONF,
            device="cpu",
            verbose=False
        )[0]

        if results.boxes is None:
            return

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            depth = get_depth_bottom_strip(
                depth_frame, x1, y1, x2, y2
            )

            if not (MIN_DEPTH <= depth <= MAX_DEPTH):
                continue

            # --- Simple, robust geometry ---
            cx = (x1 + x2) / 2.0
            dx = cx - self.cx0

            angle_rad = math.atan(dx / self.fx)
            angle_deg = math.degrees(angle_rad)
            lateral   = depth * math.tan(angle_rad)

            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"

            # Convention:
            # x = forward distance (m)
            # y = lateral offset (m)
            # z = bearing angle (deg)
            msg.point.x = depth
            msg.point.y = lateral
            msg.point.z = angle_deg

            self.pub.publish(msg)


def main():
    rclpy.init()
    node = ProbeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
