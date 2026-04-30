# 🚀 P6 — Mars Rover Autonomous Navigation

> **6th Semester Project** | Autonomous probe detection and navigation on Mars using SLAM, YOLO and ROS2

---

## 📖 Overview

This project explores **autonomous navigation** for a Mars rover, using computer vision (YOLO) to detect probes and publishing their coordinates via ROS2 topics. The system is designed to operate without human intervention, simulating real-world Mars surface exploration scenarios.

---

## 🗂️ Project Structure

```
P6-Mars_rover/
├── Probe_run.py       # Main entry point — runs YOLO detection
├── best.pt            # Trained YOLO model weights (must be in same folder)
└── ...                # (Add more files/folders here as the project grows)
```

---

## ⚙️ Requirements

> *(Add your dependencies here as the project grows)*

- Python 3.x
- ROS2 (tested on: *(add distro, e.g. Humble)*)
- YOLOv8 / Ultralytics
- *(Add any other dependencies)*

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/P6-Mars_rover.git
cd P6-Mars_rover
```

### 2. Install dependencies

```bash
# Example — update this as needed
pip install ultralytics
```

### 3. Run YOLO Probe Detection

Make sure `best.pt` is in the **same folder** as `Probe_run.py`, then run:

```bash
python Probe_run.py
```

> This starts the YOLO detection pipeline and publishes probe coordinates to a ROS2 topic.

### 4. View Detected Coordinates

In a **separate terminal**, echo the ROS2 topic:

```bash
ros2 topic echo /probe/data
```

---

## 📡 ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/probe/data` | *(add msg type)* | Coordinates of detected probes |

---

## 🧠 How It Works

1. `Probe_run.py` captures input (camera / simulation feed)
2. YOLO model (`best.pt`) detects probes in each frame
3. Detection coordinates are published to `/probe/data` via ROS2
4. Any ROS2 node can subscribe to this topic for navigation decisions

---

## 🗺️ Roadmap

- [x] YOLO probe detection
- [x] ROS2 coordinate publishing
- [ ] *(Add upcoming features here)*
- [ ] *(Autonomous navigation integration)*
- [ ] *(Simulation environment setup)*

---

## 👥 Contributors

> *(Add team members here)*

- Member 1
- Member 2

---

## 📄 License

> *(Add your license here, e.g. MIT)*

---

## 📝 Notes

> *(Use this section for any extra notes, known issues, or references)*
