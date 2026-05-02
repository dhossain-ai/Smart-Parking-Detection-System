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

## Phase 2: COCO Annotation Parsing and Slot Preparation

Goal:
Read Roboflow COCO annotations, prepare parking-slot metadata, and export a small crop sample set for visual verification.

Deliverables:
- src/data/coco_utils.py
- src/data/prepare_pklot_coco.py
- src/data/export_crop_samples.py
- src/data/make_crop_contact_sheet.py
- data/processed/metadata/slot_annotations.csv
- data/processed/metadata/split_summary.csv
- data/processed/metadata/category_summary.csv
- data/processed/crop_samples/
- results/figures/crop_samples_contact_sheet.jpg

Status:
In progress

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
