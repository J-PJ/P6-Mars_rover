import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import pyzed.sl as sl

MODEL_PATH = "best.pt"
YOLO_CONF = 0.5
YOLO_INTERVAL = 0.1
MAX_DISTANCE = 5.0


def pick_probe_pixel(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Return the mean valid-depth pixel (u, v) near the bottom of the bounding box."""
    h, w = depth_map.shape
    v = min(h - 1, y2 - 10)

    us = [
        u for u in range(x1, x2, 4)
        if np.isfinite(depth_map[v, u]) and depth_map[v, u] > 0
    ]

    if not us:
        return None, None

    return int(np.mean(us)), v


class ProbeDetector(Node):
    def __init__(self):
        super().__init__("probe_detector")

        self.pub = self.create_publisher(String, "probe/data", 10)

        self.model = YOLO(MODEL_PATH)

        # ── ZED setup ──────────────────────────────────────────────────────────
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode        = sl.DEPTH_MODE.NEURAL
        init.coordinate_units  = sl.UNIT.METER
        init.depth_maximum_distance = MAX_DISTANCE

        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED camera")

        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.xyz   = sl.Mat()

        self.last_yolo = 0.0
        self.timer = self.create_timer(0.05, self.loop)

    # ── Main loop ──────────────────────────────────────────────────────────────
    def loop(self):
        now = time.time()
        if now - self.last_yolo < YOLO_INTERVAL:
            return
        self.last_yolo = now

        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        self.zed.retrieve_measure(self.xyz, sl.MEASURE.XYZ)

        frame     = np.ascontiguousarray(self.image.get_data()[:, :, :3])
        depth_map = self.depth.get_data()

        results = self.model(frame, conf=YOLO_CONF, device="cpu", verbose=False)[0]

        if results.boxes is None:
            return

        for box in results.boxes:
            self._process_box(box, depth_map)

    def _process_box(self, box, depth_map: np.ndarray):
        """Validate a single detection box and publish its 3-D coordinates."""
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        u, v = pick_probe_pixel(depth_map, x1, y1, x2, y2)
        if u is None:
            return

        err, point = self.xyz.get_value(u, v)
        if err != sl.ERROR_CODE.SUCCESS:
            return

        X, Y, Z = float(point[0]), float(point[1]), float(point[2])

        if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
            return

        distance = math.sqrt(X * X + Y * Y + Z * Z)
        if not (0 < distance <= MAX_DISTANCE):
            return

        msg = String()
        msg.data = f"x={X:.3f} y={Y:.3f} z={Z:.3f} d={distance:.3f}"
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ProbeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.zed.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
