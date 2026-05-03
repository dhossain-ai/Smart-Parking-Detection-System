# Smart Parking Detection System: 10-Slide Outline

## Slide 1: Title

Bullet points:

- Smart Parking Detection System
- Offline parking-slot occupancy classification
- PKLot Roboflow COCO export
- Classical model and local CNN comparison

Suggested visual:

- Side-by-side original and annotated parking-lot demo image

Speaker notes:

- Introduce the goal: classify each known parking slot as vacant or occupied.
- State that this is a local/offline system, not a hosted API demo.

## Slide 2: Problem Statement

Bullet points:

- Drivers need to know which spaces are available.
- The task is slot-level occupancy classification.
- Output is a visual green/red parking map.
- Green = vacant, red = occupied.

Suggested visual:

- Annotated frame with green and red parking-slot overlays

Speaker notes:

- Explain that false occupancy is especially important because it can hide an actually free slot.

## Slide 3: Dataset and Offline Constraint

Bullet points:

- Dataset: PKLot via Roboflow COCO export
- Local `train`, `valid`, and `test` folders
- `_annotations.coco.json` files provide slot boxes and labels
- No external vision APIs or online inference services

Suggested visual:

- Simple dataset flow diagram: COCO annotations to slot crops to model prediction

Speaker notes:

- Emphasize that all training, evaluation, and demos run locally.

## Slide 4: Preprocessing and COCO Slot Extraction

Bullet points:

- Parse COCO annotation files
- Normalize labels to `vacant` and `occupied`
- Clip and validate bounding boxes
- Crop each parking slot on the fly
- Resize crops to 64x64

Suggested visual:

- Parking image with several bounding boxes and example slot crops

Speaker notes:

- Clarify that the system uses known slot coordinates and does not scan the whole image as a general detector.

## Slide 5: Classical Method

Bullet points:

- 64x64 parking-slot crops
- LBP texture features
- HSV color histogram
- HOG descriptor
- Linear SVM classifier

Suggested visual:

- Classical pipeline: crop to features to LinearSVC to label

Speaker notes:

- Explain why handcrafted features are a good baseline for a controlled slot-classification task.

## Slide 6: CNN Method

Bullet points:

- Custom local CNN trained from scratch
- 64x64 RGB crop input
- Convolution, batch normalization, pooling, dropout
- Two outputs: vacant and occupied
- Tuned V2 uses threshold `0.48`

Suggested visual:

- CNN architecture block diagram or training curve screenshot

Speaker notes:

- State that the CNN improves recall and F1 but still falls short on the strict accuracy and precision requirements.

## Slide 7: Evaluation Metrics and Requirements

Bullet points:

- Occupied is the positive class
- Metrics: accuracy, precision, recall, F1-score
- Classical target: accuracy > 88%, precision/recall > 85%, F1 > 0.85
- CNN target: accuracy > 98%, precision/recall > 97%, F1 > 0.97
- False occupancy rate is tracked

Suggested visual:

- Requirement checklist table

Speaker notes:

- Explain false occupancy as vacant slots predicted occupied.

## Slide 8: Results Comparison

Bullet points:

- Classical: accuracy 0.94025, precision 0.93871, recall 0.94200, F1 0.94035
- Tuned CNN: accuracy 0.97300, precision 0.95372, recall 0.99425, F1 0.97356
- Classical requirement met
- CNN requirement partially met
- Weather-wise results not claimed without labels

Suggested visual:

- Bar chart comparing accuracy, precision, recall, and F1-score

Speaker notes:

- Be explicit that the CNN does not fully meet strict modern targets yet.

## Slide 9: Visual Demo: Image, Video, and App

Bullet points:

- Image demo draws slot predictions on a PKLot test image
- Video demo is generated from annotated PKLot frame sequences
- Streamlit app displays demos, metrics, comparison, and calibration notes
- All demos run locally/offline

Suggested visual:

- Three-panel screenshot: image demo, video demo frame, Streamlit dashboard

Speaker notes:

- Say that real camera deployment needs one-time slot calibration.

## Slide 10: Conclusion and Future Work

Bullet points:

- Complete offline workflow implemented
- Classical method meets requirements
- Tuned CNN has strong recall and F1 but needs more tuning
- Current COCO export lacks reliable weather labels
- Future work: calibration UI, weather labels, CNN tuning, explainability

Suggested visual:

- Final architecture overview or app screenshot

Speaker notes:

- Close with the honest conclusion: the system is complete and demonstrable locally, while the CNN target remains an improvement area.
