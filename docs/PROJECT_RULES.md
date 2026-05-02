# Project Rules

## Phase Discipline

- Work only on the current approved phase.
- Do not start dataset processing, model training, evaluation, or UI implementation during Phase 0.
- Do not claim results or model performance before evaluation has been run.

## Offline Constraint

- The Smart Parking Detection System must run fully offline.
- Do not use external inference APIs or hosted computer vision services.
- Do not add dependencies on Google Vision, Cloud OCR, AWS Rekognition, Azure Computer Vision, Roboflow hosted APIs, OpenAI Vision APIs, or similar online services.

## Project Scope

- This project is parking-slot occupancy classification using PKLot annotations.
- It is not a general YOLO-style vehicle detection project.
- Required approaches are:
  - Classical features such as LBP, HSV histograms, optional HOG, with SVM or Random Forest.
  - A local CNN trained and evaluated locally.

## Repository Hygiene

- Do not commit the full dataset or large generated artifacts.
- Keep raw and processed datasets under ignored data directories.
- Keep trained models and generated result media out of Git unless explicitly approved.
