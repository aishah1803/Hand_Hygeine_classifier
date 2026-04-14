# Hand Hygiene Analyser

A computer vision pipeline that analyses hand hygiene by detecting 
gel coverage and calculating a cleanliness percentage from a hand image.

Built from scratch using YOLO11, OpenCV, MediaPipe, and PyTorch.

---

## What it does

- Detects hands and gel using a custom trained YOLO11 model
- Calculates gel coverage percentage from bounding box overlap
- Analyses colour and brightness of hand and gel regions
- Analyses surface texture and smoothness
- Detects hand landmarks and finger positions using MediaPipe
- Analyses pore visibility as a gel coverage indicator
- Combines all features into one final cleanliness score
- Evaluates using 4 ML classifiers with full metrics
- Includes a graphical user interface

---

## Project Structure
Hand_hygeine_project/
├── main.py # Run analysis on a single image
├── batch_run.py # Run analysis on all test images
├── metrics.py # ML evaluation and charts
├── gui.py # Graphical user interface
├── detector.py # YOLO11 hand and gel detection
├── analyser.py # Gel coverage calculation
├── colour_analyser.py # Colour and brightness analysis
├── texture_analyser.py # Texture and smoothness analysis
├── landmark_analyser.py # MediaPipe hand landmarks
├── pore_analyser.py # Pore visibility analysis
├── scorer.py # Weighted feature scoring
├── models/ # Trained model weights
├── test_images/ # Test images
└── output/ # Results and charts

---

## Requirements

- Python 3.11
- See requirements.txt for all packages

---

## Installation

### Step 1 - Clone or download the project

### Step 2 - Create a virtual environment
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### Step 3 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 - Download MediaPipe hand model
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'models/hand_landmarker.task'); print('Downloaded OK')"
```

---

## How to run

### Option 1 - Graphical Interface (easiest)
```bash
python gui.py
```
Opens a window where you can browse for an image and click Analyse.

### Option 2 - Single image analysis
```bash
python main.py
```
Edit the IMAGE_PATH in main.py to point to your image.

### Option 3 - Batch analysis on all test images
```bash
python batch_run.py
```
Runs the pipeline on every image in the test_images folder.

### Option 4 - ML metrics and evaluation
```bash
python metrics.py
```
Runs 4 classifiers and saves charts to output/metrics/

---

## How it works

### Stage 1 - Detection
YOLO11 detects bounding boxes around hands and gel regions.

### Stage 2 - Coverage Analysis
Calculates what percentage of the hand box is covered by gel boxes.
Multiple gel detections add bonus points to the coverage score.

### Stage 3 - Colour Analysis
OpenCV analyses the HSV colour space of hand and gel regions.
Brightness difference between hand and gel indicates gel visibility.

### Stage 4 - Texture Analysis
Laplacian filter detects surface roughness.
Standard deviation measures consistency of gel spread.

### Stage 5 - Landmark Analysis
MediaPipe detects 21 hand keypoints.
Checks which fingers are extended and how open the hand is.

### Stage 6 - Pore Analysis
CLAHE enhancement and blob detection finds visible pores.
More pores visible means less gel coverage.
Fewer pores means gel is filling and covering them.

### Stage 7 - Scoring
All 5 features combined with weighted scoring:
- Gel coverage:   20%
- Colour diff:    25%
- Consistency:    20%
- Confidence:     15%
- Pore score:     20%

### Output labels
- 70% and above  = Clean
- 40% to 69%     = Partially Clean
- Below 40%      = Unclean

---

## ML Evaluation

metrics.py evaluates the pipeline using:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)

Metrics calculated:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

Uses Leave One Out cross validation due to small dataset size.

---

## Dataset

- 26 images labelled in Label Studio with bounding boxes
- Classes: Hand (class 1) and Gel (class 0)
- Split: 20 train, 3 validation, 3 test for YOLO training
- 6 unlabelled images used for pipeline testing

---

## Model

- Architecture: YOLO11n (nano)
- Task: Object detection
- Classes: 2 (Hand, Gel)
- Training: 50 epochs, batch size 4, image size 640
- Results: mAP50 = 0.852 overall
  - Hand: mAP50 = 0.708
  - Gel:  mAP50 = 0.995

---

## Notes on YOLOv13

YOLOv13 was explored as an alternative backbone.
Training completed successfully but prediction has a known bug
in the current iMoonLab implementation.
YOLOv13 also does not support segmentation masks in its current
release, making it unsuitable as a direct replacement for 
segmentation-based pipelines.
YOLO11 was chosen as the final model due to stability and 
official segmentation support.

---

## Author

Built as part of a discipline-specific assignment.
University of Bradford, 2026. 