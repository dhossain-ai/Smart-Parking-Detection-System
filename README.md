# Smart Parking Detection System

Smart Parking Detection System is a fully offline computer vision project for parking-slot occupancy classification using the PKLot dataset.

The goal is to classify each annotated parking slot as:

- Vacant
- Occupied

This project uses known parking-slot coordinates from the local PKLot annotations or saved local calibration files. The current local dataset is a Roboflow COCO export detected by `_annotations.coco.json` files. It is not a general car detector.

## Offline Constraint

All processing must run locally. The project must not use external internet inference APIs or hosted computer vision services such as Google Vision, Cloud OCR, AWS Rekognition, Azure Computer Vision, Roboflow hosted APIs, OpenAI Vision APIs, or similar services.

## Required Methods

The final system must implement and compare two local approaches:

- Classical computer vision / machine learning: LBP, HSV histogram, optional HOG, with SVM or Random Forest.
- Neural network: a local CNN trained on parking-slot crops.

## Expected Metrics

Professor target metrics:

| Method | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Classical | > 88% | > 85% | > 85% | > 0.85 |
| CNN | > 98% | > 97% | > 97% | > 0.97 |

Additional evaluation should include inference speed, weather-wise robustness, confusion matrices, and false occupancy rate.

Current local results from saved Phase 4, Phase 5, and Phase 6 outputs:

| Method | Accuracy | Precision | Recall | F1-score | Requirement status |
|---|---:|---:|---:|---:|---|
| Classical LBP + HSV + HOG + LinearSVC | 0.94025 | 0.93871 | 0.94200 | 0.94035 | Met |
| Tuned CNN V2 | 0.97300 | 0.95372 | 0.99425 | 0.97356 | Partially met |

The classical method meets the assignment minimum requirements. The current tuned CNN exceeds recall and F1-score requirements but is still below the professor's strict accuracy and precision targets, so it must not be described as fully meeting the CNN requirement yet.

## Repository Structure

```text
data/
  raw/          # Local PKLot dataset files, ignored by Git
  processed/    # Generated crops and processed data, ignored by Git
  samples/      # Small safe samples for documentation or demos
  splits/       # Train/validation/test split files

src/
  data/         # Dataset parsing and crop preparation
  classical/    # Classical feature extraction and ML models
  neural/       # Local CNN model and training code
  evaluation/   # Metrics and model comparison
  visualization/# Overlays, plots, and visual outputs
  utils/        # Shared utilities and configuration

app/            # Local Streamlit app
models/         # Local trained model artifacts, ignored by Git
results/        # Metrics and generated visual outputs
notebooks/      # Exploratory notebooks
slides/         # Presentation materials
docs/           # Project planning and AI context documents
```

## Current Status

Phase 7 image detection demo is implemented. The repository can point to a local Roboflow COCO PKLot dataset, parse parking-slot annotations, normalize occupancy labels, validate train/valid/test splits, create local training manifests, train/evaluate the classical model, train/tune the local CNN, produce comparison tables/reports from saved metrics, and create annotated image-demo outputs from known parking-slot boxes.

Classical model training and evaluation are implemented for Phase 4. Custom CNN training and Phase 5B tuning are implemented, with metrics accepted only from local runs. Phase 6 comparison is completed using the best available real CNN result. Phase 7 image demo is completed for both tuned CNN and classical model selection. Video and UI implementation have not started.

Weather labels are unavailable in the current Roboflow COCO export, so weather robustness is prepared as a workflow/template rather than reported with fake sunny/rainy/cloudy accuracy values. Further CNN training and tuning can be done later.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.utils.check_setup
```

## Phase 1 Dataset Setup

Download the PKLot Dataset manually from Kaggle and unzip it into:

```text
data/raw/PKLot/
```

Alternatively, set a custom dataset path:

```bash
export PKLOT_DATA_DIR=/absolute/path/to/PKLot
```

The full dataset is ignored by Git and should not be committed.

Check the local setup and dataset location with:

```bash
python3 -m src.utils.check_setup
python3 -m src.data.check_pklot_dataset
```

For the current Roboflow COCO version, the checker detects:

```text
data/raw/PKLot/train/_annotations.coco.json
data/raw/PKLot/valid/_annotations.coco.json
data/raw/PKLot/test/_annotations.coco.json
```

If PKLot has not been downloaded or unzipped yet, the dataset checker will fail with setup instructions. That is expected before the local dataset is available.

## Phase 2 COCO Metadata Workflow

Generate project-ready parking-slot metadata:

```bash
python -m src.data.prepare_pklot_coco
```

This writes:

```text
data/processed/metadata/slot_annotations.csv
data/processed/metadata/split_summary.csv
data/processed/metadata/category_summary.csv
```

The metadata keeps image paths, COCO annotation IDs, normalized labels, and clipped bounding boxes. Category names such as `space-empty`, `empty`, `vacant`, `space-occupied`, and `occupied` are normalized to `vacant` or `occupied`.

Export only a small crop sample set for visual verification:

```bash
python -m src.data.export_crop_samples --samples-per-class 20
python -m src.data.make_crop_contact_sheet
```

Full crop export is available but intentionally optional:

```bash
python -m src.data.export_crop_samples --export-all
```

Later training should prefer metadata-driven on-the-fly cropping from source images. That avoids committing or storing hundreds of thousands of generated crop files.

## Phase 3 Dataset Analysis Workflow

Analyze the generated slot metadata:

```bash
python -m src.data.analyze_pklot_metadata
```

Validate the existing Roboflow `train`, `valid`, and `test` splits:

```bash
python -m src.data.validate_splits --check-images --max-image-checks 200
```

Prepare weather labels honestly:

```bash
python -m src.data.prepare_weather_labels
```

The Roboflow COCO export may not preserve original PKLot weather folders. Weather is inferred only from reliable path metadata or a manually supplied weather CSV. If reliable weather labels are unavailable, the project records `unknown`; no sunny, rainy, cloudy, or overcast labels should be invented.

Create local manifests for later training:

```bash
python -m src.data.create_training_manifests --samples-per-class 2000
```

Full manifests are written to `data/splits/train_slots.csv`, `data/splits/valid_slots.csv`, and `data/splits/test_slots.csv`. Small balanced manifests are also created for quick experiments. These generated CSVs are ignored by Git, while training should crop slots on the fly from the original images and known bounding boxes.

## Phase 4 Classical Baseline Workflow

The classical baseline crops slots on the fly from the original images using manifest bounding boxes. It does not require exporting all crop images.

Smoke test handcrafted feature extraction:

```bash
python -m src.classical.smoke_test_features
```

Train and evaluate the default classical model:

```bash
python -m src.classical.train_classical
```

Default features are LBP texture histograms, HSV color histograms, and HOG descriptors from 64x64 slot crops. The default classifier is a Linear SVM with feature scaling.

Generated local outputs:

```text
models/classical/classical_lbp_hsv_hog_svm.joblib
results/metrics/classical/
results/figures/classical_confusion_matrix.png
```

Model files and generated metrics are ignored by Git. Reported accuracy, precision, recall, and F1-score must come from local script output; do not fake or hand-edit metrics.

## Phase 5 CNN Baseline Workflow

The neural baseline is a custom local CNN trained from scratch. It does not download pretrained weights and does not use online inference services.

Smoke test the dataset and model path:

```bash
python -m src.neural.smoke_test_cnn
```

Train and evaluate the default CNN:

```bash
python -m src.neural.train_cnn
```

The CNN reads the same manifest CSVs as the classical baseline, crops parking slots on the fly from original images, resizes crops to 64x64, applies local training augmentations, and predicts:

```text
0 = vacant
1 = occupied
```

Generated local outputs:

```text
models/cnn/best_cnn_model.pth
results/metrics/cnn/
results/figures/cnn_confusion_matrix.png
results/figures/cnn_training_curves.png
```

CNN checkpoints and generated metrics are ignored by Git. Neural metrics should only be reported from real local training/evaluation runs.

## Phase 5B CNN Tuning Workflow

The tuned CNN workflow adds a `v2` architecture option and a threshold sweep to reduce false occupancy predictions while keeping occupied-slot recall high.

Run the V2 smoke test:

```bash
python -m src.neural.smoke_test_cnn
```

Run a tuned CPU-safe experiment:

```bash
python -m src.neural.train_cnn --model-version v2 --epochs 8 --batch-size 64 --samples-per-class 2000 --patience 3 --weight-decay 0.0001 --output-model models/cnn/best_cnn_model_v2.pth --output-dir results/metrics/cnn_tuned
```

Tuned outputs are saved separately:

```text
models/cnn/best_cnn_model_v2.pth
results/metrics/cnn_tuned/
results/figures/cnn_tuned_confusion_matrix.png
results/figures/cnn_tuned_training_curves.png
results/figures/cnn_tuned_threshold_sweep.png
```

The threshold sweep is selected from validation metrics and saved locally. Do not claim the neural requirement is met unless the generated tuned metrics prove it.

## Phase 6 Evaluation Comparison Workflow

Generate comparison tables, markdown reports, and slide/app figures from saved metrics:

```bash
python -m src.evaluation.compare_models
```

Default inputs:

```text
results/metrics/classical/classical_metrics.json
results/metrics/cnn_tuned/cnn_metrics.json
```

If tuned CNN metrics are unavailable, the comparison script falls back to:

```text
results/metrics/cnn/cnn_metrics.json
```

Generated comparison outputs:

```text
results/metrics/comparison/
results/figures/model_metrics_comparison.png
results/figures/requirement_checklist.png
results/figures/false_occupancy_comparison.png
```

The Phase 6 report states that the classical model meets requirements and the current tuned CNN partially meets requirements. It also records that weather-wise robustness cannot be honestly reported until manual weather labels are added.

## Phase 7 Image Demo Workflow

Create an annotated image demo with the tuned CNN:

```bash
python -m src.visualization.demo_image
python -m src.visualization.demo_image --model-type cnn
```

Create an annotated image demo with the classical model:

```bash
python -m src.visualization.demo_image --model-type classical
```

The image demo selects a test image with many parking-slot annotations unless `--image-path` is supplied. It uses known COCO/test-manifest bounding boxes, crops each slot, predicts vacant/occupied locally, and draws green vacant overlays and red occupied overlays on the original image. This is parking-slot image/frame detection, not general car detection.

Default outputs:

```text
results/images/demo_image_cnn_output.jpg
results/images/demo_image_cnn_side_by_side.jpg
results/images/demo_image_classical_output.jpg
results/images/demo_image_classical_side_by_side.jpg
results/metrics/demo_image/
```

Generated demo images, prediction CSVs, and summary JSON files are local ignored outputs.
