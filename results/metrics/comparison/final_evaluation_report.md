# Final Evaluation Report

## Project Summary

This project evaluates a fully offline Smart Parking Detection System for parking-slot occupancy classification. Occupied is the positive class and vacant is the negative class.

## Dataset Summary

The current local data source is the Roboflow COCO export of PKLot with train, valid, and test folders and `_annotations.coco.json` files. The evaluation uses saved local model metrics from the test split.

## Classical Method Summary

The classical model uses models/classical/classical_lbp_hsv_hog_svm.joblib with LBP, HSV histogram, HOG features, feature scaling, and LinearSVC. Test accuracy is 0.94025, precision is 0.93871, recall is 0.94200, and F1-score is 0.94035.

The classical method meets the assignment minimum requirements.

## CNN Method Summary

The best available CNN result for this comparison is models/cnn/best_cnn_model_v2.pth. The selected occupied-probability threshold is 0.48. Test accuracy is 0.97300, precision is 0.95372, recall is 0.99425, and F1-score is 0.97356.

The current tuned CNN exceeds recall and F1 requirements but does not yet meet the strict accuracy and precision targets.

The CNN is still useful for the visual demo and can be improved later with larger training samples, threshold tuning, or architecture/augmentation refinement.

## Requirement Checklist

| method | metric | required_value | actual_value | passed |
| --- | --- | --- | --- | --- |
| Classical | accuracy | > 0.88 | 0.94025 | Yes |
| Classical | precision | > 0.85 | 0.93871 | Yes |
| Classical | recall | > 0.85 | 0.94200 | Yes |
| Classical | f1_score | > 0.85 | 0.94035 | Yes |
| CNN | accuracy | > 0.98 | 0.97300 | No |
| CNN | precision | > 0.97 | 0.95372 | No |
| CNN | recall | > 0.97 | 0.99425 | Yes |
| CNN | f1_score | > 0.97 | 0.97356 | Yes |

Classical requirement: met.
CNN requirement: partially met with current tuned model; further training/tuning planned.

## False Occupancy Discussion

| method | false_occupancy_rate | note |
| --- | --- | --- |
| Classical | 0.06150 | Actual vacant slots predicted as occupied; lower is better. |
| CNN | 0.04825 | Actual vacant slots predicted as occupied; lower is better. |

False occupancy means an actual vacant slot is predicted as occupied. This can cause a driver to skip a free space, so lower is better.

## Speed Discussion

| method | records | inference_seconds | inference_time_per_slot_sec | slots_per_second | feature_extraction_time_per_slot_sec | total_estimated_time_per_slot_sec | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classical | 4000 | 0.23778960 | 0.00005945 | 16821.59 | 0.01915145 | 0.01921090 | Classical classifier timing excludes feature extraction unless total is used. |
| CNN | 8000 | 47.77885590 | 0.00597236 | 167.44 | NA | NA | CNN timing includes model inference on the recorded device. |

Classical speed should be read carefully because classifier inference and handcrafted feature extraction are recorded separately. CNN speed reflects the saved model inference timing and depends on the recorded device.

## Threshold Summary

No threshold in the sweep satisfied all CNN requirements.

## Weather Robustness Limitation

Original PKLot includes sunny, rainy, and cloudy conditions, but this Roboflow COCO export does not provide reliable weather labels in the current file paths or metadata. The project provides `data/splits/weather_labels_template.csv` for optional manual labeling. Weather-wise accuracy should not be claimed until those labels are added.

## Honest Conclusion

The classical method meets the assignment minimum requirements. The current tuned CNN result is strong, especially for occupied-slot recall, but it is still below the professor's accuracy and precision targets. Further CNN training and tuning can continue later without changing the reported Phase 6 metrics.
