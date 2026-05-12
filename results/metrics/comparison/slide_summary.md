# Slide Summary

- Classical result: accuracy 0.94025, precision 0.93871, recall 0.94200, F1-score 0.94035; requirement met.
- MobileNetV3 result: accuracy 0.98013, precision 0.96922, recall 0.99175, F1-score 0.98035; precision is slightly below the strict target.
- Requirement status: Classical met; CNN partially met with MobileNetV3-Small.
- Visual demo plan: use MobileNetV3-Small for overlays and present the precision limitation honestly.
- Weather limitation: current Roboflow COCO export has no reliable sunny/rainy/cloudy labels, so weather-wise accuracy is not reported.
- Next improvement: continue MobileNetV3 threshold and training tuning to push precision above 0.97.
