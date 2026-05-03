# Slide Summary

- Classical result: accuracy 0.94025, precision 0.93871, recall 0.94200, F1-score 0.94035; requirement met.
- CNN result: accuracy 0.97300, precision 0.95372, recall 0.99425, F1-score 0.97356; strong but still below accuracy/precision targets.
- Requirement status: Classical met; CNN partially met with current tuned model.
- Visual demo plan: use the best available real CNN result for overlays and comparison, while presenting CNN status honestly.
- Weather limitation: current Roboflow COCO export has no reliable sunny/rainy/cloudy labels, so weather-wise accuracy is not reported.
- Next improvement: continue CNN training/tuning with larger samples, threshold analysis, architecture refinement, and augmentation updates.
