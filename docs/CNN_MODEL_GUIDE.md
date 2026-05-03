# CNN Model Guide

Phase 5 trains a fully offline custom CNN for parking-slot occupancy classification. Phase 5B adds tuning tools for improving precision while keeping occupied-slot recall high.

The model uses generated manifest CSVs and known COCO bounding boxes. Slot crops are loaded from the original images on the fly, resized to 64x64, normalized, and passed to a compact local CNN. No pretrained weights, online inference APIs, or internet downloads are used.

## Architecture

Baseline model:

- Conv2d + BatchNorm + ReLU + MaxPool
- Conv2d + BatchNorm + ReLU + MaxPool
- Conv2d + BatchNorm + ReLU + MaxPool
- Conv2d + BatchNorm + ReLU + AdaptiveAvgPool
- Dropout + Linear classifier

Tuned model option:

- `v2` with slightly wider convolution channels
- BatchNorm after convolution layers
- AdaptiveAvgPool before the classifier
- configurable dropout
- same 64x64 input and 2-logit output

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
- low-probability mild Gaussian blur

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

## Tune CNN

Run a tuned V2 experiment:

```bash
python -m src.neural.train_cnn --model-version v2 --epochs 15 --batch-size 64 --samples-per-class 8000 --patience 5 --weight-decay 0.0001 --output-model models/cnn/best_cnn_model_v2.pth --output-dir results/metrics/cnn_tuned
```

On CPU, smaller tuned runs are acceptable:

```bash
python -m src.neural.train_cnn --model-version v2 --epochs 8 --batch-size 64 --samples-per-class 2000 --patience 3 --weight-decay 0.0001 --output-model models/cnn/best_cnn_model_v2.pth --output-dir results/metrics/cnn_tuned
```

The tuning workflow saves a validation/test threshold sweep:

```text
results/metrics/cnn_tuned/threshold_sweep.csv
results/figures/cnn_tuned_threshold_sweep.png
```

Threshold selection is based on validation results. It first looks for thresholds that meet all neural targets, then falls back to the best F1 threshold with recall at least 0.97. Tuned results should be used only if the saved metrics genuinely improve the baseline.

## Phase 6 Comparison Status

Phase 6 compares the saved classical metrics against the best available saved CNN metrics:

```bash
python -m src.evaluation.compare_models
```

The current tuned CNN V2 result is strong but only partially meets the professor's CNN target. It exceeds recall and F1-score requirements, but accuracy and precision are still below the strict CNN thresholds:

```text
accuracy: 0.97300, target > 0.98
precision: 0.95372, target > 0.97
recall: 0.99425, target > 0.97
F1-score: 0.97356, target > 0.97
```

Do not claim the CNN fully meets requirements until a real local run satisfies all four target metrics. Further CNN training can continue later with larger samples, additional threshold tuning, or architecture/augmentation refinement.

Weather robustness is not reported with numeric CNN accuracy because the current Roboflow COCO export does not preserve reliable weather labels. The project keeps `data/splits/weather_labels_template.csv` for optional manual labeling before any weather-wise results are claimed.
