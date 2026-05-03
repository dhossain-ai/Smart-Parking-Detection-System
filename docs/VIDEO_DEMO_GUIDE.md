# Video Demo Guide

Phase 8 creates an annotated frame-sequence video demo for parking-slot occupancy classification.

The current PKLot data is a Roboflow COCO export containing annotated image frames, not real continuous video files. The default demo honestly creates a video-style result from multiple annotated PKLot test images:

```text
test image 1 -> annotated frame 1
test image 2 -> annotated frame 2
test image 3 -> annotated frame 3
```

Run the default tuned CNN video demo:

```bash
python -m src.visualization.demo_video
```

Run the CNN demo with explicit frame count:

```bash
python -m src.visualization.demo_video --model-type cnn --num-frames 60
```

Run the classical model demo:

```bash
python -m src.visualization.demo_video --model-type classical --num-frames 30
```

Color meaning:

- Green = vacant
- Red = occupied

The demo uses known parking-slot bounding boxes from the COCO-derived test manifest. It crops each annotated parking slot, predicts vacant/occupied locally, draws overlays on every frame, and writes an annotated MP4 plus a side-by-side original/processed MP4.

Default outputs:

```text
results/videos/demo_video_cnn_output.mp4
results/videos/demo_video_cnn_side_by_side.mp4
results/metrics/demo_video/demo_video_occupancy_trend.csv
results/metrics/demo_video/demo_video_predictions.csv
results/metrics/demo_video/demo_video_summary.json
```

The script also writes model-specific copies:

```text
results/metrics/demo_video/demo_video_cnn_summary.json
results/metrics/demo_video/demo_video_classical_summary.json
```

The occupancy trend CSV stores one row per frame:

```text
frame_index,image_path,total_slots,occupied_count,vacant_count,occupancy_rate,model_type,threshold
```

No internet API is used. The script loads local model files only.

For arbitrary real video, parking slot coordinates must be calibrated or provided for the camera view. This phase uses annotated PKLot image frames to create a video-style demonstration, so it should be described as an annotated frame-sequence video demo.

Generated MP4 files, prediction CSVs, and summary JSON files are local artifacts and should not be committed.
