# Image Demo Guide

Phase 7 creates a local image detection demo for parking-slot occupancy classification.

Run the default tuned CNN demo:

```bash
python -m src.visualization.demo_image
```

Run the classical model demo:

```bash
python -m src.visualization.demo_image --model-type classical
```

Run the CNN demo explicitly:

```bash
python -m src.visualization.demo_image --model-type cnn
```

The demo uses known parking-slot bounding boxes from the COCO-derived test manifest. It classifies each annotated parking slot crop, draws predictions on the original image, and saves side-by-side visual output.

Color meaning:

- Green = vacant
- Red = occupied

This is image/frame detection using known parking-slot coordinates. It is not a general car detector and it does not search the whole image for vehicles.

No internet API is used. The script loads local model files only:

```text
models/cnn/best_cnn_model_v2.pth
models/classical/classical_lbp_hsv_hog_svm.joblib
```

Default outputs are saved under:

```text
results/images/
results/metrics/demo_image/
```

CNN outputs:

```text
results/images/demo_image_cnn_output.jpg
results/images/demo_image_cnn_side_by_side.jpg
results/metrics/demo_image/demo_image_cnn_predictions.csv
results/metrics/demo_image/demo_image_cnn_summary.json
```

Classical outputs:

```text
results/images/demo_image_classical_output.jpg
results/images/demo_image_classical_side_by_side.jpg
results/metrics/demo_image/demo_image_classical_predictions.csv
results/metrics/demo_image/demo_image_classical_summary.json
```

The script also writes canonical latest-run files:

```text
results/metrics/demo_image/demo_image_predictions.csv
results/metrics/demo_image/demo_image_summary.json
```

Use a specific annotated image:

```bash
python -m src.visualization.demo_image --image-path data/raw/PKLot/test/example.jpg
```

The image must have matching slot annotations in the selected manifest.

Useful options:

```bash
python -m src.visualization.demo_image --model-type cnn --max-labels 80
python -m src.visualization.demo_image --model-type cnn --threshold 0.48
python -m src.visualization.demo_image --model-type classical --manifest data/splits/test_slots.csv
```

Generated demo images and prediction outputs are local artifacts and should not be committed.
