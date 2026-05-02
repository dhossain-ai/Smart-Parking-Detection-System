# Metrics Requirements

## Positive Class

Occupied is the positive class.

Vacant is the negative class.

## Required Metrics

### Classical Method

Minimum requirements:

- Accuracy > 88%
- Precision > 85%
- Recall > 85%
- F1-score > 0.85

### Neural Network Method

Minimum requirements:

- Accuracy > 98%
- Precision > 97%
- Recall > 97%
- F1-score > 0.97

## Extra Metrics

Also compute:

- Confusion matrix
- Inference speed per image
- Inference speed per frame
- Weather-wise accuracy
- False occupancy rate

## False Occupancy Rate

False occupancy means:

Actual vacant slot predicted as occupied.

This is important because a driver may skip a free space if the system incorrectly reports it as occupied.

Formula:

False Occupancy Rate = False Occupied Predictions / Total Actual Vacant Slots