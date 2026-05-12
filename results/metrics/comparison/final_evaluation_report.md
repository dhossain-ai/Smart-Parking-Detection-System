# Final Evaluation Report

## Project Summary

This project evaluates a fully offline Smart Parking Detection System for parking-slot occupancy classification. Occupied is the positive class and vacant is the negative class.

## Dataset Summary

The current local data source is the Roboflow COCO export of PKLot with train, valid, and test folders and `_annotations.coco.json` files. The evaluation uses saved local model metrics from the test split.

## Classical Method Summary

The classical model uses models/classical/classical_lbp_hsv_hog_svm.joblib with LBP, HSV histogram, HOG features, feature scaling, and LinearSVC. Test accuracy is 0.94025, precision is 0.93871, recall is 0.94200, and F1-score is 0.94035.

The classical method meets the assignment minimum requirements.

## CNN Method Summary

The best available CNN result for this comparison is /content/drive/MyDrive/smart_parking_models/best_mobilenetv3_transfer_final.pth. The selected occupied-probability threshold is 0.14. Test accuracy is 0.98013, precision is 0.96922, recall is 0.99175, and F1-score is 0.98035.

The MobileNetV3 model meets the accuracy, recall, and F1-score targets. Precision is slightly below the strict 97% target, so the CNN requirement is reported honestly as partially met.

The neural model is used for the visual demo because it is the strongest saved CNN result and has the lowest false occupancy rate in the current experiments.

## Requirement Checklist

| method | metric | required_value | actual_value | passed |
| --- | --- | --- | --- | --- |
| Classical | accuracy | > 0.88 | 0.94025 | Yes |
| Classical | precision | > 0.85 | 0.93871 | Yes |
| Classical | recall | > 0.85 | 0.94200 | Yes |
| Classical | f1_score | > 0.85 | 0.94035 | Yes |
| CNN | accuracy | > 0.98 | 0.98013 | Yes |
| CNN | precision | > 0.97 | 0.96922 | No |
| CNN | recall | > 0.97 | 0.99175 | Yes |
| CNN | f1_score | > 0.97 | 0.98035 | Yes |

Classical requirement: met.
CNN requirement: partially met with MobileNetV3-Small; precision is the only strict target still slightly below requirement.

## False Occupancy Discussion

| method | false_occupancy_rate | note |
| --- | --- | --- |
| Classical | 0.06150 | Actual vacant slots predicted as occupied; lower is better. |
| CNN | 0.03150 | Actual vacant slots predicted as occupied; lower is better. |

False occupancy means an actual vacant slot is predicted as occupied. This can cause a driver to skip a free space, so lower is better.

## Speed Discussion

| method | records | inference_seconds | inference_time_per_slot_sec | slots_per_second | feature_extraction_time_per_slot_sec | total_estimated_time_per_slot_sec | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classical | 4000 | 0.23778960 | 0.00005945 | 16821.59 | 0.01915145 | 0.01921090 | Classical classifier timing excludes feature extraction unless total is used. |
| CNN | 24000 | 114.08308682 | 0.00475346 | 210.37 | NA | NA | CNN timing includes model inference on the recorded device. |

Classical speed should be read carefully because classifier inference and handcrafted feature extraction are recorded separately. CNN speed reflects the saved model inference timing and depends on the recorded device.

## Threshold Summary

No threshold in the sweep satisfied all CNN requirements.

## Weather Robustness Limitation

Original PKLot includes sunny, rainy, and cloudy conditions, but this Roboflow COCO export does not provide reliable weather labels in the current file paths or metadata. The project provides `data/splits/weather_labels_template.csv` for optional manual labeling. Weather-wise accuracy should not be claimed until those labels are added.

## Honest Conclusion

The classical method meets the assignment minimum requirements. MobileNetV3-Small is the final neural model because it improves accuracy, F1-score, and false occupancy rate over the earlier custom CNN. It meets accuracy, recall, and F1-score targets, while precision remains slightly below the strict 97% target.
