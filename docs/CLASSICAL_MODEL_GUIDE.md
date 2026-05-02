# Classical Model Guide

Phase 4 trains a fully offline classical baseline for parking-slot occupancy classification.

The pipeline uses known COCO bounding boxes from generated manifests. It crops each slot from the original image on the fly, resizes the crop to 64x64, extracts handcrafted features, and trains a local scikit-learn classifier.

## Features

Default feature set:

- LBP texture histogram
- HSV color histogram
- HOG descriptor

Default classifier:

- StandardScaler + LinearSVC

Occupied is the positive class. Vacant is the negative class.

## Smoke Test

Run a quick feature extraction check:

```bash
python -m src.classical.smoke_test_features
```

The smoke test loads a few records from `data/splits/train_slots_balanced_small.csv`, extracts feature vectors, and checks for invalid labels or NaN/infinite values.

## Train Baseline

Run the default baseline:

```bash
python -m src.classical.train_classical
```

Default manifests:

```text
data/splits/train_slots_balanced_small.csv
data/splits/valid_slots_balanced_small.csv
data/splits/test_slots_balanced_small.csv
```

Outputs:

```text
models/classical/classical_lbp_hsv_hog_svm.joblib
results/metrics/classical/classical_metrics.csv
results/metrics/classical/classical_metrics.json
results/metrics/classical/classification_report.txt
results/metrics/classical/confusion_matrix.csv
results/figures/classical_confusion_matrix.png
```

For a smaller run:

```bash
python -m src.classical.train_classical --max-train-per-class 1000 --max-valid-per-class 500 --max-test-per-class 500
```

Model files and generated metrics are local outputs and are ignored by Git.
