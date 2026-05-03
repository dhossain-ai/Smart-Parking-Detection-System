# Smart Parking Detection System: Final Technical Report

## Problem Statement

This project builds a fully offline Smart Parking Detection System for parking-slot occupancy classification. Given an image frame and known parking-slot bounding boxes, the system classifies each slot as `vacant` or `occupied`, then visualizes the result with green overlays for vacant spaces and red overlays for occupied spaces.

The project is parking-slot classification, not general vehicle detection. For arbitrary new camera views, the parking slots must be calibrated once or supplied as known bounding boxes before the detector can be used.

## Dataset Description

The system uses the Roboflow COCO export of the PKLot dataset. The local dataset is organized into `train`, `valid`, and `test` folders, each with a `_annotations.coco.json` file. COCO annotations provide image paths, parking-slot bounding boxes, and occupancy labels.

Occupied is treated as the positive class. Vacant is treated as the negative class.

## Offline Constraint

All processing runs locally. The project does not use external internet APIs, hosted inference services, cloud vision tools, Roboflow hosted inference, OpenAI Vision APIs, or other online prediction systems. Training, evaluation, image demo generation, video demo generation, and the Streamlit app all use local files and local Python libraries.

## Preprocessing Pipeline

The preprocessing pipeline reads COCO annotations, normalizes category labels into `vacant` and `occupied`, clips parking-slot bounding boxes to image bounds, and creates manifest CSV files for train, validation, and test splits.

During training and inference, slot crops are loaded from the original images using the known bounding boxes. Crops are resized to 64x64 pixels. This avoids storing a large full crop export and keeps the workflow reproducible from local annotations.

## Classical Method

The classical model uses handcrafted visual features extracted from each 64x64 parking-slot crop:

- Local Binary Pattern texture histogram
- HSV color histogram
- HOG descriptor

These features are scaled and classified with a Linear SVM. The saved local model path is:

```text
models/classical/classical_lbp_hsv_hog_svm.joblib
```

## Neural Network Method

The neural method is a custom local CNN trained from scratch on parking-slot crops. The tuned V2 model uses stacked convolution, batch normalization, ReLU activation, pooling, adaptive average pooling, dropout, and a two-class linear classifier.

The tuned CNN predicts:

```text
0 = vacant
1 = occupied
```

The saved local model path is:

```text
models/cnn/best_cnn_model_v2.pth
```

The selected occupied-probability threshold from local validation is `0.48`.

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- False occupancy rate
- Inference speed
- Requirement pass/fail status

False occupancy means an actual vacant slot is predicted as occupied. This is important because a driver may skip a genuinely available parking space.

## Results

| Method | Accuracy | Precision | Recall | F1-score | Requirement status |
|---|---:|---:|---:|---:|---|
| Classical LBP + HSV + HOG + LinearSVC | 0.94025 | 0.93871 | 0.94200 | 0.94035 | Meets classical requirement |
| Tuned CNN V2 | 0.97300 | 0.95372 | 0.99425 | 0.97356 | Partially meets strict CNN requirement |

The classical method exceeds the assignment's classical targets for accuracy, precision, recall, and F1-score.

The tuned CNN exceeds the strict recall and F1-score targets, but does not yet meet the strict accuracy and precision targets. It should therefore not be described as fully meeting the modern/CNN requirement unless a future real local training run improves those metrics.

## False Occupancy Discussion

The saved comparison report records the following false occupancy rates:

| Method | False occupancy rate |
|---|---:|
| Classical | 0.06150 |
| Tuned CNN V2 | 0.04825 |

The CNN has the lower false occupancy rate in the saved comparison, which is useful for a parking application because fewer free slots are incorrectly marked as occupied. However, its precision is still below the strict modern target, so further tuning is still needed before claiming full CNN requirement satisfaction.

## Image Demo

The image demo uses a PKLot test image and known COCO parking-slot boxes. It crops each annotated slot, runs local model prediction, and draws the result on the original frame:

- Green = vacant
- Red = occupied

Default CNN output files are generated under:

```text
results/images/demo_image_cnn_output.jpg
results/images/demo_image_cnn_side_by_side.jpg
```

## Video Demo

The current Roboflow COCO export contains annotated image frames, not continuous raw video files. The video demo therefore creates an honest annotated frame-sequence video from multiple PKLot test images.

For each frame, the system uses known COCO parking-slot boxes, predicts occupancy locally, draws green/red overlays, and writes annotated MP4 outputs:

```text
results/videos/demo_video_cnn_output.mp4
results/videos/demo_video_cnn_side_by_side.mp4
```

For a real camera feed, parking slots must be calibrated once for that fixed view.

## Streamlit App

The Streamlit app provides a local dashboard for:

- Dashboard summary
- Image detection demo
- Video detection demo
- Model comparison
- Metrics
- Weather robustness notes
- Explainability placeholder
- Calibration notes
- Settings/about information

Run locally with:

```bash
streamlit run app/streamlit_app.py
```

The app does not call external APIs and does not train models on startup.

## Weather Robustness Limitation

The original PKLot dataset includes different weather conditions, but the current Roboflow COCO export does not preserve reliable weather labels in the available metadata. Weather-wise accuracy is therefore not reported with invented sunny, rainy, or cloudy metrics.

The project includes a weather-label template workflow for future manual labeling. Weather robustness can be evaluated honestly after reliable weather labels are added.

## Conclusion

The classical method meets the assignment’s classical minimum requirements.
The tuned CNN exceeds recall and F1 targets but does not yet meet the strict accuracy and precision targets.
The system still demonstrates a complete local image/video smart parking workflow and can be improved with further tuning.

## Future Improvements

- Continue CNN tuning with larger training samples and additional threshold analysis.
- Add reliable weather labels and compute real weather-wise performance.
- Add camera calibration support for arbitrary real parking-lot videos.
- Add optional Grad-CAM or activation visualization for CNN explainability.
- Optimize inference for faster CPU-only demonstration.
