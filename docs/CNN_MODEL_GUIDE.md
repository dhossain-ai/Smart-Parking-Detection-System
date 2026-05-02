# CNN Model Guide

Phase 5 trains a fully offline custom CNN for parking-slot occupancy classification.

The model uses generated manifest CSVs and known COCO bounding boxes. Slot crops are loaded from the original images on the fly, resized to 64x64, normalized, and passed to a compact local CNN. No pretrained weights, online inference APIs, or internet downloads are used.

## Architecture

Default model:

- Conv2d + BatchNorm + ReLU + MaxPool
- Conv2d + BatchNorm + ReLU + MaxPool
- Conv2d + BatchNorm + ReLU + MaxPool
- Conv2d + BatchNorm + ReLU + AdaptiveAvgPool
- Dropout + Linear classifier

The classifier outputs two logits:

```text
0 = vacant
1 = occupied
```

Occupied is the positive class.

## Data

Default manifests:

```text
data/splits/train_slots_balanced_small.csv
data/splits/valid_slots_balanced_small.csv
data/splits/test_slots_balanced_small.csv
```

Training augmentations are local OpenCV/numpy transforms:

- random brightness
- random contrast
- small rotation
- small horizontal and vertical shift
- horizontal flip

## Smoke Test

Run:

```bash
python -m src.neural.smoke_test_cnn
```

The smoke test loads a small balanced manifest sample, creates a dataloader, runs one forward pass, and checks tensor/logit shapes plus NaN/inf values.

## Train Baseline

Run:

```bash
python -m src.neural.train_cnn
```

Default outputs:

```text
models/cnn/best_cnn_model.pth
results/metrics/cnn/cnn_metrics.csv
results/metrics/cnn/cnn_metrics.json
results/metrics/cnn/classification_report.txt
results/metrics/cnn/confusion_matrix.csv
results/figures/cnn_confusion_matrix.png
results/figures/cnn_training_curves.png
```

For a smaller CPU run:

```bash
python -m src.neural.train_cnn --epochs 3 --max-train-per-class 1000 --max-valid-per-class 500 --max-test-per-class 500
```

Model checkpoints, generated metrics, and generated plots are local outputs and are ignored by Git. Reported accuracy, precision, recall, and F1-score must come from local script output.
