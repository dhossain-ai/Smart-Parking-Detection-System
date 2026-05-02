# Smart Parking Detection System

Smart Parking Detection System is a fully offline computer vision project for parking-slot occupancy classification using the PKLot dataset.

The goal is to classify each annotated parking slot as:

- Vacant
- Occupied

This project uses known parking-slot coordinates from PKLot XML annotations or saved local calibration files. It is not a general car detector.

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

Phase 0 setup is complete or in progress. The repository scaffold, documentation, dependency list, basic path configuration, and setup validation script are the only intended deliverables at this phase.

Dataset processing, model training, UI implementation, and evaluation have not started.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.utils.check_setup
```
