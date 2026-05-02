# AI Context: Smart Parking Detection System

## Project Goal

Build a fully offline Smart Parking Detection System using the PKLot dataset.

The system classifies parking spaces as:

- Vacant
- Occupied

The project must implement and compare:

1. Classical computer vision / machine learning method
2. Neural network method

The project should also include a visually impressive local Streamlit app that supports image and video detection.

## Important Constraint

External internet API services are prohibited.

Do not use:

- Google Vision API
- Cloud OCR
- AWS Rekognition
- Azure Computer Vision
- Roboflow hosted API
- OpenAI Vision API
- Any online inference API

All processing must run locally using Python libraries.

Allowed libraries:

- OpenCV
- NumPy
- Pandas
- scikit-image
- scikit-learn
- PyTorch
- Matplotlib
- Plotly
- Streamlit

## Dataset

Dataset: PKLot Dataset from Kaggle.

The dataset contains parking lot surveillance images and XML annotations.

The XML annotation files provide parking slot coordinates and occupancy labels.

The system should use known parking slot coordinates from annotations or saved calibration JSON files.

This is parking-slot classification, not general car object detection.

## Required Methods

### Classical Method

Use image preprocessing and handcrafted features.

Recommended:

- Resize crops to 64x64
- Normalize images
- Extract LBP features
- Extract HSV color histogram
- Optional HOG features
- Train SVM or Random Forest

### Neural Network Method

Use local neural network training.

Recommended:

- 64x64 RGB parking slot crops
- Custom CNN
- Data augmentation
- Cross entropy loss
- Adam optimizer
- Save trained model locally

## Required Metrics

Professor requirements:

| Method | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Classical | > 88% | > 85% | > 85% | > 0.85 |
| Modern/CNN | > 98% | > 97% | > 97% | > 0.97 |

Also evaluate:

- Speed
- Weather robustness
- Sunny performance
- Rainy performance
- Cloudy performance
- False occupancy rate

## Visual Demo Requirements

The app/demo should show:

- Original image/video feed
- Processed image/video feed
- Green overlays for vacant slots
- Red overlays for occupied slots
- Confidence scores
- Total slot count
- Occupied count
- Vacant count
- Occupancy rate
- Inference time
- Classical vs CNN comparison
- Weather robustness results
- Confusion matrix
- Optional Grad-CAM / activation visualization

## Final Output

The final repo should include:

- Source code
- Trained models
- Evaluation results
- Demo images
- Demo video output
- Streamlit local app
- README
- Slides/report