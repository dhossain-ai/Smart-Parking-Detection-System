# Smart Parking Detection System

Smart Parking Detection System is a fully offline computer vision project for parking-slot occupancy classification using the Roboflow COCO export of the PKLot dataset.

The system classifies each known parking slot as:

- `vacant`
- `occupied`

It uses known parking-slot coordinates from local COCO annotations or saved calibration files. It is not a general car detector. For a new real camera image or video, parking slots must be calibrated once or supplied as known bounding boxes.

## Features

- Fully local/offline workflow with no external vision APIs.
- COCO annotation parsing for PKLot parking-slot boxes.
- Classical ML method using LBP, HSV histogram, HOG, and Linear SVM.
- Custom local CNN V2 trained on 64x64 parking-slot crops.
- MobileNetV3-Small transfer learning support with 224x224 parking-slot crops.
- Accuracy, precision, recall, F1-score, speed, and false occupancy reporting.
- Image demo with green vacant overlays and red occupied overlays.
- Annotated frame-sequence video demo from PKLot test images.
- Local Streamlit dashboard for demos, metrics, model comparison, and limitations.
- Honest weather robustness limitation when reliable weather labels are unavailable.

## Offline / No API Rule

All processing runs locally. The project does not use Google Vision, Cloud OCR, AWS Rekognition, Azure Computer Vision, Roboflow hosted APIs, OpenAI Vision APIs, or any other online inference service.

## Dataset Setup

Download the PKLot dataset manually and place the Roboflow COCO export at:

```text
data/raw/PKLot/
```

Expected COCO files:

```text
data/raw/PKLot/train/_annotations.coco.json
data/raw/PKLot/valid/_annotations.coco.json
data/raw/PKLot/test/_annotations.coco.json
```

You may also set a custom dataset path:

```bash
export PKLOT_DATA_DIR=/absolute/path/to/PKLot
```

The dataset is ignored by Git and should not be committed.

Check setup:

```bash
python -m src.utils.check_setup
python -m src.data.check_pklot_dataset
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.utils.check_setup
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Preprocessing Commands

Prepare COCO slot metadata:

```bash
python -m src.data.prepare_pklot_coco
```

Analyze dataset metadata:

```bash
python -m src.data.analyze_pklot_metadata
```

Validate splits:

```bash
python -m src.data.validate_splits --check-images --max-image-checks 200
```

Create training manifests:

```bash
python -m src.data.create_training_manifests --samples-per-class 2000
```

## Classical Training

Smoke test handcrafted feature extraction:

```bash
python -m src.classical.smoke_test_features
```

Train and evaluate the classical model:

```bash
python -m src.classical.train_classical
```

Expected local model artifact:

```text
models/classical/classical_lbp_hsv_hog_svm.joblib
```

## CNN Training

Smoke test the CNN pipeline:

```bash
python -m src.neural.smoke_test_cnn
```

Train the tuned V2 CNN:

```bash
python -m src.neural.train_cnn --model-version v2 --epochs 8 --batch-size 64 --samples-per-class 2000 --patience 3 --weight-decay 0.0001 --output-model models/cnn/best_cnn_model_v2.pth --output-dir results/metrics/cnn_tuned
```

Expected local model artifact:

```text
models/cnn/best_cnn_model_v2.pth
```

### MobileNetV3 Transfer Learning

MobileNetV3-Small is available as a professor-aligned modern neural method. It uses 224x224 parking-slot crops and replaces the torchvision classifier head with the same two output logits:

```text
0 = vacant
1 = occupied
```

Pretrained ImageNet weights are used only during training when `--pretrained` is explicitly passed. The saved checkpoint contains local weights and metadata, so later evaluation or app/demo integration can run offline from the saved `.pth` file.

Smoke test MobileNetV3 without downloading weights:

```bash
python -m src.neural.smoke_test_cnn --model-version mobilenet_v3_small --image-size 224
```

Local transfer-learning run:

```bash
python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained --image-size 224 --epochs 5 --batch-size 64 --samples-per-class 2000 --patience 3 --output-model models/cnn/best_mobilenetv3_transfer.pth --output-dir results/metrics/mobilenetv3_transfer
```

Colab full training command:

```bash
python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained --image-size 224 --epochs 20 --batch-size 64 --samples-per-class 12000 --patience 6 --weight-decay 0.0001 --output-model models/cnn/best_mobilenetv3_transfer.pth --output-dir results/metrics/mobilenetv3_transfer
```

Model files are generated locally and ignored by Git. See `models/MODEL_ARTIFACTS.md`.

## Evaluation

Generate comparison tables, reports, and slide-ready summaries:

```bash
python -m src.evaluation.compare_models
```

Final validation check:

```bash
python -m src.utils.final_project_check
```

## Image Demo

Run the tuned CNN image demo:

```bash
python -m src.visualization.demo_image --model-type cnn
```

Run the classical image demo:

```bash
python -m src.visualization.demo_image --model-type classical
```

Default CNN output:

```text
results/images/demo_image_cnn_side_by_side.jpg
```

Green means vacant. Red means occupied. The demo uses known COCO parking-slot boxes from the test manifest.

## Video Demo

Run the tuned CNN frame-sequence video demo:

```bash
python -m src.visualization.demo_video --model-type cnn --num-frames 30 --fps 5
```

For a shorter smoke test:

```bash
python -m src.visualization.demo_video --model-type cnn --num-frames 5 --fps 5
```

Default CNN output:

```text
results/videos/demo_video_cnn_side_by_side.mp4
```

The current COCO export contains annotated image frames, not continuous source videos, so the video demo is generated honestly from annotated PKLot frame sequences.

## Streamlit App

Run the local dashboard:

```bash
streamlit run app/streamlit_app.py
```

The app includes:

- Dashboard
- Image Detection
- Video Detection
- Compare Models
- Metrics
- Weather Robustness
- Explainability
- Calibration
- Settings/About

The app runs locally/offline and does not train models on startup.

## Results

| Method | Accuracy | Precision | Recall | F1-score | Requirement status |
|---|---:|---:|---:|---:|---|
| Classical LBP + HSV + HOG + LinearSVC | 0.94025 | 0.93871 | 0.94200 | 0.94035 | Met |
| Tuned CNN V2 | 0.97300 | 0.95372 | 0.99425 | 0.97356 | Partially met |

The classical method meets the assignment's classical minimum requirements.

The tuned CNN exceeds recall and F1-score targets but does not yet meet the strict modern accuracy and precision targets. Do not describe the CNN as fully meeting the modern/CNN requirement unless future real local training improves those metrics.

## Known Limitations

- The system uses known parking-slot boxes; arbitrary new camera views require one-time slot calibration.
- Weather labels are unavailable in the current Roboflow COCO export, so weather-wise accuracy is not reported.
- CNN V2 is strong but only partially meets the strict modern target.
- Dataset files, model checkpoints, generated images, generated videos, and large generated results are ignored by Git.
- Existing demo outputs may need to be regenerated after cloning because generated artifacts are local.

## Final Project Structure

```text
app/
  streamlit_app.py
data/
  raw/                # local dataset, ignored by Git
  processed/          # generated metadata/crops, ignored by Git
  splits/             # generated manifests, ignored by Git
docs/
  FINAL_REPORT.md
  DEMO_SCRIPT.md
  FINAL_SUBMISSION_CHECKLIST.md
  STREAMLIT_APP_GUIDE.md
models/
  MODEL_ARTIFACTS.md
  classical/          # local .joblib model artifacts, ignored by Git
  cnn/                # local .pth model artifacts, ignored by Git
results/
  metrics/            # saved metrics and generated demo summaries
  images/             # generated demo images, ignored by Git
  videos/             # generated demo videos, ignored by Git
slides/
  presentation_outline.md
src/
  classical/          # handcrafted features and classical training
  data/               # COCO parsing, manifests, dataset checks
  evaluation/         # comparison reports
  neural/             # CNN model/training/evaluation
  utils/              # setup and final validation checks
  visualization/      # image/video overlay demos
```

## Final Documentation

- Final report: `docs/FINAL_REPORT.md`
- Demo script: `docs/DEMO_SCRIPT.md`
- Slide outline: `slides/presentation_outline.md`
- Submission checklist: `docs/FINAL_SUBMISSION_CHECKLIST.md`
- Model artifact note: `models/MODEL_ARTIFACTS.md`
