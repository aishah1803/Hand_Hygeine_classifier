# Hand_Hygeine_classifier
a group project for year two to create a model that detects how clean the hand is
# YOLOv13 Experiment — Hand Hygiene Detection

## Overview

This branch documents the exploration and experimentation with YOLOv13
as a potential replacement for YOLOv8 in the hand hygiene detection pipeline.
YOLOv13 was developed by iMoonLab and introduces a new architecture called
HyperACE (Hypergraph-based Adaptive Correlation Enhancement) which improves
detection accuracy by capturing complex relationships across the whole image.

---

## Motivation

The original pipeline built by the team used YOLOv8 for the segmentation
gating stage. The goal of this experiment was to explore whether upgrading
to a more recent YOLO version would improve detection performance on the
hand hygiene dataset.

---

## What is YOLOv13

- Released in 2025 by iMoonLab
- Based on a modified Ultralytics codebase
- Introduces HyperACE mechanism for better feature correlation
- Uses A2C2f blocks instead of standard C2f blocks
- Achieves approximately 3% mAP improvement over YOLOv11 on COCO benchmark
- Available at: https://github.com/iMoonLab/yolov13

---

## Experiment Setup

### Environment
- Platform: Google Colab (T4 GPU)
- Python: 3.12
- PyTorch: 2.10.0 with CUDA 12.8
- YOLOv13 installed from: pip install git+https://github.com/iMoonLab/yolov13.git

### Dataset
- 26 images labelled using Label Studio
- Classes: Hand (class 0) and Gel (class 1)
- Bounding box detection format (YOLO txt format)
- Split: 20 train, 3 validation, 3 test

### Training Configuration
- Model: yolov13n.pt (nano version)
- Task: detect
- Epochs: 20
- Image size: 640
- Batch size: 2
- Device: T4 GPU
- Optimiser: AdamW (auto selected)

---

## Training Results

| Epoch | Box Loss | Class Loss | mAP50 |
|-------|----------|------------|-------|
| 1     | 1.560    | 3.496      | 0.216 |
| 5     | 1.589    | 3.484      | 0.448 |
| 10    | 1.402    | 3.025      | 0.394 |
| 15    | 1.731    | 3.030      | 0.459 |
| 20    | 1.613    | 3.026      | 0.462 |

### Final validation results
- Overall mAP50: 0.499
- Hand mAP50: 0.00222
- Gel mAP50: 0.995
- Model size: 5.4 MB

---

## YOLOv13 Architecture

YOLOv13n has 648 layers and 2,460,301 parameters compared to
YOLOv8n's 152 layers and 3,263,811 parameters.

Key new components:
- DSConv — Depthwise Separable Convolution for efficiency
- A2C2f — Attention-based Cross Stage Partial feature fusion
- HyperACE — Hypergraph Adaptive Correlation Enhancement
- FullPAD Tunnel — Feature pyramid path aggregation
- DownsampleConv — Efficient downsampling

---

## Issues Discovered

### Issue 1 — Segmentation not supported
YOLOv13 in its current release only supports object detection
(bounding boxes). It does not provide segmentation masks like
YOLOv8-seg. The original pipeline requires result.masks for
the ROI extraction stage, which YOLOv13 cannot provide.

### Issue 2 — Prediction bug
Running model.predict() on test images produced the following error:

ValueError: too many values to unpack (expected 3)
File: ultralytics/utils/ops.py in process_mask

This is a known bug in the current iMoonLab YOLOv13 implementation
related to mask processing being called even in detection mode.

### Issue 3 — Dependency conflict
YOLOv13 requires sympy 1.13.3 which conflicts with torch 2.10.0
which requires sympy >= 1.13.3. This was resolved by installing
sympy 1.13.3 explicitly.

---

## Comparison with YOLO11

| Feature | YOLOv13n | YOLO11n |
|---|---|---|
| Parameters | 2.46M | 2.58M |
| Layers | 648 | 101 |
| GFLOPs | 6.4 | 6.3 |
| Segmentation | No | Yes |
| Prediction stable | No (bug) | Yes |
| pip install | GitHub only | pip install ultralytics |
| mAP50 (our dataset) | 0.499 | 0.852 |

---

## Conclusion

YOLOv13 shows promising architecture improvements and trained
successfully on our hand hygiene dataset. However it is not
suitable as a drop-in replacement for the segmentation pipeline
at this time due to:

1. No segmentation mask support in current release
2. Unstable prediction API with known bugs
3. Lower mAP50 on our dataset compared to YOLO11 (0.499 vs 0.852)

YOLO11 was chosen as the final model for the pipeline due to:
- Official Ultralytics support and stability
- Segmentation support if needed
- Higher mAP50 on our dataset
- Reliable prediction API

YOLOv13 is recommended for future exploration once the iMoonLab
repository matures and segmentation support is added.

---

## Files in this branch

- README_yolov13.md — this file
- yolov13_experiment.ipynb — full Colab notebook with all cells

---

## How to reproduce

1. Open yolov13_experiment.ipynb in Google Colab
2. Connect to a T4 GPU runtime
3. Run all cells in order
4. Note: prediction cells will produce the ValueError bug described above

---

## References

- YOLOv13 GitHub: https://github.com/iMoonLab/yolov13
- YOLOv13 Paper: Hypergraph-based Adaptive Correlation Enhancement
- Ultralytics YOLO11: https://docs.ultralytics.com
- Label Studio: https://labelstud.io

---

## Author

Aishah — University of Bradford, 2026
Discipline Specific Assignment
