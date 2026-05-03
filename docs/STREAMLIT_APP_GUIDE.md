# Streamlit App Guide

Phase 9 adds a local Streamlit dashboard for the Smart Parking Detection System.

Run the app:

```bash
streamlit run app/streamlit_app.py
```

The app is fully local/offline. It does not call external internet APIs, hosted model endpoints, or cloud vision services.

Dashboard pages:

- Dashboard
- Image Detection
- Video Detection
- Compare Models
- Metrics
- Weather Robustness
- Explainability
- Calibration
- Settings/About

The Image Detection page displays existing local image-demo outputs and can regenerate them with local commands:

```bash
python -m src.visualization.demo_image --model-type cnn
python -m src.visualization.demo_image --model-type classical
```

The Video Detection page displays existing annotated PKLot frame-sequence video outputs and can regenerate them with local commands:

```bash
python -m src.visualization.demo_video --model-type cnn --num-frames 30 --fps 5
python -m src.visualization.demo_video --model-type classical --num-frames 15 --fps 5
```

The app reads saved comparison metrics, requirement checklists, final evaluation reports, demo outputs, and weather robustness notes from local `results/` files.

Important limitations:

- The CNN currently partially meets the strict modern model target; the app reports that honestly.
- Weather-wise accuracy is not reported because the current Roboflow COCO export does not preserve reliable weather labels.
- Arbitrary real camera videos require slot calibration for the camera view before occupancy detection can be applied.

Generated images, videos, prediction CSVs, and local Streamlit secrets should not be committed.
