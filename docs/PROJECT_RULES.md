# Project Rules

## Workflow

- Work only on the current approved phase.
- Use `python3` commands in this environment.
- Do not claim dataset statistics, metrics, or model performance unless they were produced by local scripts.

## Offline Implementation

- The system must run fully offline.
- Do not use Google Vision, Cloud OCR, AWS Rekognition, Azure Vision, Roboflow hosted APIs, OpenAI Vision APIs, or any other online inference service.
- Do not download the dataset automatically in project code.
- Do not add Kaggle credentials or tokens.

## Project Scope

- This project is parking-slot occupancy classification using PKLot annotations.
- It is not a general YOLO-style vehicle detection project.
- Use current COCO annotations, optional XML annotations, or saved local calibration coordinates for parking-slot regions.
- Keep both required methods: classical features with SVM or Random Forest, and a local CNN.
- Occupied is the positive class; vacant is the negative class.

## Repository Hygiene

- Do not commit the full dataset, trained models, or large generated results.
- Keep raw data, processed data, local model artifacts, and generated media under ignored paths.
- Do not fake metrics or create placeholder results.
