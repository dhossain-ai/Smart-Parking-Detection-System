# CNN Model Guide

Phase 5 trains a fully offline custom CNN for parking-slot occupancy classification. Phase 5B adds tuning tools for improving precision while keeping occupied-slot recall high. Phase B also adds MobileNetV3-Small transfer learning as a professor-aligned modern neural method.

The models use generated manifest CSVs and known COCO bounding boxes. Slot crops are loaded from the original images on the fly, resized, normalized, and passed to a local neural network. The custom CNN uses 64x64 crops. MobileNetV3-Small uses 224x224 crops. No online inference APIs are used.

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

Transfer-learning option:

- `mobilenet_v3_small` uses `torchvision.models.mobilenet_v3_small`
- 224x224 parking-slot crops
- optional ImageNet pretrained weights only when `--pretrained` is explicitly passed
- optional frozen backbone with `--freeze-backbone`
- classifier head replaced with a 2-logit parking occupancy head

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

MobileNetV3 shape smoke test:

```bash
python -m src.neural.smoke_test_cnn --model-version mobilenet_v3_small --image-size 224
```

Pretrained MobileNetV3 smoke test, when torchvision weights are already cached or internet is available:

```bash
python -m src.neural.smoke_test_cnn --model-version mobilenet_v3_small --image-size 224 --pretrained
```

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

## MobileNetV3 Transfer Learning

MobileNetV3-Small is available as a transfer-learning neural method aligned with the professor's suggestion of MobileNetV3 or AlexNet. It keeps the same parking-slot classification target:

```text
0 = vacant
1 = occupied
```

Use 224x224 crops for MobileNetV3. When `--pretrained` is passed, the training dataset uses ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training may download torchvision pretrained weights only when `--pretrained` is explicitly passed. The final saved checkpoint contains local weights and metadata, so later evaluation or app/demo loading can run offline from the saved `.pth` file.

Local transfer-learning command:

```bash
python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained --image-size 224 --epochs 5 --batch-size 64 --samples-per-class 2000 --patience 3 --output-model models/cnn/best_mobilenetv3_transfer.pth --output-dir results/metrics/mobilenetv3_transfer
```

Colab full training command:

```bash
python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained --image-size 224 --epochs 20 --batch-size 64 --samples-per-class 12000 --patience 6 --weight-decay 0.0001 --output-model models/cnn/best_mobilenetv3_transfer.pth --output-dir results/metrics/mobilenetv3_transfer
```

Optional frozen-backbone smoke training:

```bash
python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained --freeze-backbone --image-size 224 --epochs 2 --batch-size 16 --samples-per-class 100 --patience 1 --output-model models/cnn/best_mobilenetv3_transfer_smoke.pth --output-dir results/metrics/mobilenetv3_transfer_smoke
```

MobileNet checkpoints include:

```text
model_version
image_size
normalize_mode
threshold / decision_threshold
class_names
pretrained
freeze_backbone
```

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
