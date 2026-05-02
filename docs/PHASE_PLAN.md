# Phase Plan

## Phase 0: Repo Setup and AI Docs

Goal:
Create the basic repository structure and documentation.

Deliverables:
- README.md
- requirements.txt
- .gitignore
- docs/AI_CONTEXT.md
- docs/PROJECT_RULES.md
- docs/PHASE_PLAN.md

Status:
Not started

---

## Phase 1: Dataset Setup

Goal:
Prepare local PKLot dataset access.

Deliverables:
- data/raw/ structure
- dataset path configuration
- DATASET_GUIDE.md

Status:
Not started

---

## Phase 2: XML Parsing and Slot Cropping

Goal:
Read PKLot XML files, crop parking slots, and save labeled crops.

Deliverables:
- src/data/parse_pklot_xml.py
- src/data/crop_slots.py
- data/processed/crops/
- crop metadata CSV

Status:
Not started

---

## Phase 3: Dataset Splitting and EDA

Goal:
Create train/validation/test splits and check class/weather balance.

Deliverables:
- train.csv
- val.csv
- test.csv
- weather split CSV
- class distribution plots

Status:
Not started

---

## Phase 4: Classical Model

Goal:
Train classical ML model.

Deliverables:
- LBP/HSV/HOG feature extraction
- SVM or Random Forest model
- classical model file
- classical metrics CSV

Status:
Not started

---

## Phase 5: Neural Network Model

Goal:
Train CNN model.

Deliverables:
- PyTorch dataset class
- CNN model
- training script
- saved CNN weights
- CNN metrics CSV

Status:
Not started

---

## Phase 6: Evaluation

Goal:
Compare classical and CNN models.

Deliverables:
- accuracy, precision, recall, F1
- confusion matrices
- weather robustness table
- speed comparison
- false occupancy rate

Status:
Not started

---

## Phase 7: Image Detection Demo

Goal:
Create visual image prediction output.

Deliverables:
- demo_image.py
- output image with green/red overlays
- occupancy summary

Status:
Not started

---

## Phase 8: Video Detection Demo

Goal:
Process a short video frame by frame.

Deliverables:
- demo_video.py
- annotated output video
- occupancy trend data

Status:
Not started

---

## Phase 9: Streamlit App

Goal:
Build modern local dashboard app.

Deliverables:
- app/streamlit_app.py
- image detection tab
- video detection tab
- model comparison tab
- metrics tab
- weather robustness tab

Status:
Not started

---

## Phase 10: Final Packaging

Goal:
Prepare final submission.

Deliverables:
- final README
- slides
- demo screenshots
- demo video
- instructions for running locally

Status:
Not started