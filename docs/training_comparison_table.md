| Training    | Classes | Samples | Best Epoch | mAP@50   | mAP@50-95 | Precision | Recall  | Training Time |
|------------|---------|---------|------------|----------|-----------|-----------|---------|--------------|
| Training 1 |   3     |   100   |    78      | 98.61%   | 89.35%    | 96.17%    | 93.88%  | 34.2 min     |
| Training 3 |   4     |   200   |    68      | 98.06%   | 90.69%    | 92.69%    | 97.42%  | 123.2 min    |
| Training 4 |   6     |   200   |    90      | 98.90%   | 95.75%    | 96.22%    | 96.63%  | 79.5 min     |

**Table: Comparison of YOLOv8 training runs on synthetic GUI dataset.**
- *Best Epoch*: Epoch with highest mAP@50 during training.
- *mAP@50*: Mean Average Precision at IoU 0.5 (higher = better detection accuracy).
- *mAP@50-95*: Stricter mean AP (IoU 0.5 to 0.95).
- *Precision/Recall*: Global values for all classes.
- *Training Time*: Total time for 100 epochs.
