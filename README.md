# Hand Hygiene Pipeline

This package implements the requested sequential pipeline:

1. Data preparation with duplicate removal, mislabel filtering, Albumentations augmentation, and skin-tone-aware split balancing.
2. YOLOv8 segmentation gating with confidence thresholding and ROI extraction.
3. MediaPipe Hand Landmarker on the YOLO ROI, with a full-image fallback when ROI landmarks fail.
4. OpenCV anatomical region splitting into palm, thumb, index, middle, ring, and pinky.
5. UV gel coverage estimation in each anatomical region using HSV fluorescence thresholds plus an adaptive brightness fallback.
6. Weighted aggregation with a palm-heavy score and a minimum-region fail rule.
7. Threshold calibration across lighting, camera, and skin-tone metadata.

Every stage records latency, and failures are logged separately per stage under `logs/failures/`.
For documentation, each `predict` run now also writes a flat CSV row per frame under `logs/predictions.csv`.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Project-local setup on this machine:

```bash
cd /Users/ericsvzn/Documents/Hand_Classification_Python/hand_hygiene_pipeline
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Expected data inputs

Stage 1 accepts either:

- A folder dataset: `raw_dir/<label>/*.jpg`
- A manifest CSV with at least `image_path,label`

Optional manifest columns:

- `mask_path`
- `split`
- `skin_tone`
- `camera_id`
- `lighting_id`
- `is_mislabeled`

For real deployment, the segmentation model should be a YOLOv8 segmentation checkpoint fine-tuned on a hand class. The default `yolov8n-seg.pt` is only a starting backbone and will not reliably segment hands without fine-tuning.

For Stage 3, place a MediaPipe `hand_landmarker.task` file at `models/hand_landmarker.task` or update `default_config.json`.

## Commands

Prepare data:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json prepare-data \
  --raw-dir /path/to/raw_dataset \
  --output-dir /path/to/prepared_dataset \
  --manifest /path/to/manifest.csv
```

Prepare data directly from a Label Studio export:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json prepare-data \
  --label-studio-json /path/to/project-export.json \
  --image-root /path/to/exported-or-local-images \
  --output-dir /path/to/prepared_dataset
```

Train the hand segmenter:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json train-segmentation \
  --dataset-yaml /path/to/hand_segmentation_dataset.yaml \
  --run-dir runs/hand_yolov8n_seg
```

Run inference:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json predict \
  --input /path/to/images \
  --summary-out logs/latency_summary.json
```

Calibrate thresholds:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json calibrate \
  --predictions-jsonl logs/predictions.jsonl \
  --metadata-csv /path/to/heldout_metadata.csv \
  --output calibration.json
```

Tune the UV fluorescence ranges for your lamp/camera setup:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json calibrate-gel \
  --image /path/to/sample_uv_hand.jpg
```

Derive image-level `clean` / `unclean` labels from a YOLO export where `Gel` means contamination and `Hand` means hand-only:

```bash
python3 -m hand_hygiene_pipeline --config default_config.json derive-image-labels \
  --yolo-export /path/to/label_studio_yolo_export \
  --split-manifest data/strict_segmentation_yolo/strict_split_manifest.csv \
  --output-dir data/derived_image_labels
```

## Best-results workflow

1. Fine-tune or provide a hand-specific YOLO segmentation checkpoint. The stock `yolov8n-seg.pt` backbone is not a reliable hand detector on its own.
2. Use the tuned defaults in `default_config.json` or `small_dataset_config.json` and update `segmentation.weights_path` to your hand model.
3. Run `calibrate-gel` on several representative UV images from your real setup and copy the final HSV ranges into `gel_detection.hsv_ranges`.
4. Run `predict` on a held-out validation folder with unique filenames and a fresh `log_dir`.
5. Run `analyse` before `calibrate` so you can inspect failure rates as well as accuracy.
6. Run `calibrate` on the held-out set, then copy the recommended thresholds into `aggregation.min_region_score` and `aggregation.pass_threshold`.
7. Reserve a final untouched test split for the last evaluation only.

## Label Studio hand-off

Use the Label Studio export as soon as you have stable image-level labels.

- For dataset creation and split balancing: feed the export directly to `prepare-data --label-studio-json ... --image-root ...`.
- For threshold calibration and final evaluation: export a held-out CSV matching `data/heldout_metadata_template.csv`, where `frame_id` matches the image filename stem exactly.
- If your export is YOLO segmentation with `Gel` and `Hand`, you can convert it into image-level `unclean` / `clean` labels with `derive-image-labels`. The rule used is: any `Gel` polygon => `unclean`; otherwise `Hand` only => `clean`.

## Outputs

- `logs/predictions.jsonl`: per-frame decision, per-region scores, confidences, and landmark mode
- `logs/predictions.csv`: documentation-friendly per-frame table with source image path, decision, landmark mode, region scores, thresholds, failures, and latencies
- `logs/latency.jsonl`: stage-by-stage latency records
- `logs/failures/<stage>.jsonl`: separate failure logs for decode, segmentation, landmarks, and region splitting
- `prepared_dataset_report.json`: data cleaning and skin-tone coverage summary
- `calibration.json`: recommended thresholds and subgroup validation metrics
