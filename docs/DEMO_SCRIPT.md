# Smart Parking Detection System Demo Script

## 1. Introduce the Project

Say:

> This is a Smart Parking Detection System for parking-slot occupancy classification. It predicts whether each known parking space is vacant or occupied.

Say:

> This system is fully local and uses no external vision APIs.

Mention that the project uses the Roboflow COCO export of PKLot and known parking-slot boxes from the annotations.

## 2. Verify Dataset Setup

Run:

```bash
python -m src.data.check_pklot_dataset
```

Say:

> The dataset checker verifies that the local PKLot COCO export is available with train, validation, and test annotations.

## 3. Explain the Models

Say:

> The project implements two required methods: a classical model using LBP, HSV, HOG, and a Linear SVM; and a local CNN trained on 64x64 parking-slot crops.

Say:

> The classical method meets its required metrics. The tuned CNN has strong recall and F1-score, but it does not yet fully meet the strict CNN accuracy and precision targets.

## 4. Run Image Demo

Run:

```bash
python -m src.visualization.demo_image --model-type cnn
```

Say:

> The system uses known COCO parking-slot boxes.

Say:

> Green means vacant, red means occupied.

Say:

> The image demo crops each annotated parking slot, runs the local CNN, and draws the prediction back onto the original image.

Show:

```text
results/images/demo_image_cnn_side_by_side.jpg
```

## 5. Run Video Demo

Run:

```bash
python -m src.visualization.demo_video --model-type cnn --num-frames 30 --fps 5
```

Say:

> The video demo is generated from annotated PKLot frame sequences.

Say:

> This is not a cloud service or online detector. It runs local model inference on local image frames.

Show:

```text
results/videos/demo_video_cnn_side_by_side.mp4
```

## 6. Launch Streamlit App

Run:

```bash
streamlit run app/streamlit_app.py
```

Say:

> The Streamlit app is a local dashboard for reviewing the image demo, video demo, model comparison, metrics, weather limitation, and calibration notes.

Say:

> For a real camera, parking slots must be calibrated once.

Open the app and show:

- Dashboard
- Image Detection
- Video Detection
- Compare Models
- Metrics
- Weather Robustness
- Calibration

## 7. Results to Present

| Method | Accuracy | Precision | Recall | F1-score | Status |
|---|---:|---:|---:|---:|---|
| Classical | 0.94025 | 0.93871 | 0.94200 | 0.94035 | Meets classical requirement |
| Tuned CNN V2 | 0.97300 | 0.95372 | 0.99425 | 0.97356 | Partially meets strict CNN requirement |

Say:

> The classical method satisfies the assignment's classical threshold. The CNN is useful and visually strong, but I am reporting it honestly as partially meeting the strict modern target because accuracy and precision are still below the requirement.

## 8. Limitations

Say:

> Weather labels are not reliable in the current COCO export, so I do not report fake sunny, rainy, or cloudy accuracy.

Say:

> Arbitrary new camera images or videos require slot calibration or known bounding boxes.

## 9. Closing Statement

Say:

> The final project demonstrates the complete offline workflow: dataset parsing, preprocessing, classical model, CNN model, evaluation, image demo, video demo, and local Streamlit app.
