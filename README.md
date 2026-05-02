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

No metrics are reported yet because models have not been trained or evaluated.

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

Phase 3 dataset analysis is implemented. The repository can point to a local Roboflow COCO PKLot dataset, parse parking-slot annotations, normalize occupancy labels, validate train/valid/test splits, and create local training manifests for later offline training.

Classical model training and evaluation are implemented for Phase 4. Custom CNN training is implemented for Phase 5. UI implementation and final model comparison have not started.

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
