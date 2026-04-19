# Hand Hygiene Classifier — YOLOv26 Segmentation (Gel Remaining)

This project uses **YOLOv26 segmentation** to detect **hand** and **gel** using polygon labels, then estimates **how much gel is left** on the hand.

We compute:

**gel% = (gel mask area ÷ hand mask area) × 100**

Classes used:
- `0 = gel`
- `1 = hand`

---

## What We Built

- A YOLOv26 **segmentation** model trained on polygon annotations (Label Studio export).
- A pipeline that:
  1) trains YOLOv26-seg
  2) evaluates on a labeled validation set
  3) predicts masks on images
  4) calculates gel% per image
  5) saves results to CSV
  6) creates simple plots for reporting

---

## Why Segmentation (not Classification)

Classification would give **one label per image** (e.g., “low gel”), but we need a **percentage**.  
Segmentation gives pixel masks for gel and hand, so gel% can be calculated from mask area.

---

## Dataset & Labeling

### Labeling Tool
- **Label Studio**
- Annotation type: **polygons** (segmentation)

### Export Format
- YOLO segmentation labels (`.txt` files)
- Each line contains: `class_id x1 y1 x2 y2 ...` (normalized polygon points)

### YOLO Folder Structure

This is the structure YOLO expects:

dataset_split/
images/
train/
val/
labels/
train/
val/
data.yaml

**Important rule:** label files must match image filenames (same stem).


### `data.yaml` Example
```yaml
path: /content/drive/MyDrive/hand_hygeine_project/dataset_split
train: images/train
val: images/val
names:
  0: gel
  1: hand

## Outputs you should expect

Saved (usually to Google Drive):
- `best_split_fast.pt` (trained model weights)
- CSV files containing gel% per image:
  - gel% values
  - optional `gel_level` category (none/low/medium/high)
  - optional `status` flag (`ok` / `no_hand_detected`)
- Prediction overlay images (masks drawn on the input)

---

## Key challenges we faced

- Colab GPU limits (sometimes forced CPU training, which is slow)
- Colab resets deleting `/content/runs/...` (fixed by saving weights to Drive)
- Validation labels missing at first (fixed by resplitting labeled pairs into `dataset_split/`)
- Duplicate hand detections (fixed by using the **largest hand mask** when calculating gel%)
- Small dataset size (gel detection is harder; more labeled images improves results)

---

## Limitations

- Small dataset → results can be unstable and miss gel in some images.
- Best improvement is adding more labeled data and training longer (ideally with GPU).

---

## Next steps

- Label more images (recommended)
- Create a proper test set (train/val/test all labeled)
- Improve model accuracy and reduce duplicate hand detections

---



