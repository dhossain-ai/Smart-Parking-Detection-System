# Final Submission Checklist

## Required by Professor

- [x] Classical computer vision / machine learning method
- [x] Neural network method
- [x] Accuracy, precision, recall, and F1-score reported
- [x] Image detection demo
- [x] Video detection demo
- [x] Local Streamlit app
- [x] Offline/no external API rule followed
- [x] False occupancy discussion included
- [ ] Weather-wise accuracy reported only when reliable weather labels are available

## Implemented Status

| Item | Status |
|---|---|
| Classical method implemented | Complete |
| Neural method implemented | Complete |
| Classical requirements | Met |
| CNN strict requirements | Partially met |
| Image demo implemented | Complete |
| Video demo implemented | Complete |
| Streamlit app implemented | Complete |
| Offline/no external API rule | Satisfied |
| Dataset committed | No |
| Models committed | No, unless manually supplied separately under submission rules |
| Weather robustness | Limitation documented honestly |

## Files/Scripts to Show

Dataset check:

```bash
python -m src.data.check_pklot_dataset
```

Classical model:

```bash
python -m src.classical.train_classical
```

CNN model:

```bash
python -m src.neural.train_cnn --model-version v2 --epochs 8 --batch-size 64 --samples-per-class 2000 --patience 3 --weight-decay 0.0001 --output-model models/cnn/best_cnn_model_v2.pth --output-dir results/metrics/cnn_tuned
```

Evaluation:

```bash
python -m src.evaluation.compare_models
```

Image demo:

```bash
python -m src.visualization.demo_image --model-type cnn
```

Video demo:

```bash
python -m src.visualization.demo_video --model-type cnn --num-frames 30 --fps 5
```

Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Final validation:

```bash
python -m src.utils.final_project_check
```

## Known Limitations

- The tuned CNN exceeds recall and F1-score targets but does not yet meet the strict CNN accuracy and precision targets.
- Weather labels are unavailable in the current COCO export, so sunny/rainy/cloudy accuracy is not reported.
- The system uses known COCO parking-slot boxes for PKLot demos.
- Arbitrary real camera images/videos require slot calibration or known bounding boxes.
- Model checkpoints and generated media are local artifacts and are ignored by Git.

## Before Demo Checklist

- [ ] Confirm local dataset exists under `data/raw/PKLot/` or `PKLOT_DATA_DIR`.
- [ ] Confirm `models/classical/classical_lbp_hsv_hog_svm.joblib` exists.
- [ ] Confirm `models/cnn/best_cnn_model_v2.pth` exists.
- [ ] Run final validation script.
- [ ] Generate image demo if missing.
- [ ] Generate video demo if missing.
- [ ] Launch Streamlit locally.
- [ ] Prepare to explain that green means vacant and red means occupied.
- [ ] Prepare to explain that the system uses known parking-slot boxes.
- [ ] Prepare to explain that a real camera needs one-time parking-slot calibration.
