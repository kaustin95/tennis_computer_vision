# Tennis Match Vision

A hands-on computer vision project that automatically analyzes tennis match footage to extract match insights

---

## Overview

**Tennis Match Vision** processes raw video of tennis matches, detecting players and the ball, estimating court keypoints, and computing performance metrics:

- **Player Tracking & Speed**  
  – Real-time detection of each player using YOLOv8  
  – Calculates and visualizes player movement speeds over time

- **Ball Detection & Shot Speed**  
  – Fine-tuned YOLOv5 model to localize fast-moving tennis balls  
  – Estimates shot velocity by frame-wise displacement analysis

- **Court Keypoint Extraction**  
  – CNN-based model identifies court lines & corners  
  – Transforms video frames to a normalized bird’s-eye view for accurate measurements

---

## Technology Stack

| Component                       | Framework / Library        |
| ------------------------------- | -------------------------- |
| Object Detection                | [Ultralytics YOLOv8][yolov8] / YOLOv5 |
| Deep Learning & Training        | PyTorch                    |
| Image Processing & Analysis     | OpenCV, NumPy              |
| Data Handling                   | pandas                     |
| Visualization                   | Matplotlib                 |
| Environment                     | Python 3.8                 |

---

## Skills and takeaways

Advanced Object Detection: Custom fine-tuning of YOLO models for high-speed ball tracking.

Deep Learning Workflows: End-to-end training in PyTorch with data augmentation and evaluation.

Geometric Computer Vision: Court line detection and homography transforms for frame normalization.

Performance Optimization: Real-time inference pipeline with OpenCV optimizations.

Data Analysis & Visualization: From raw pixel data to clear, actionable plots and CSV reports.

