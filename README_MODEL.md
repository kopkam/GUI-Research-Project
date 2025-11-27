# GUI Widget Detection with YOLOv8

This project trains a YOLOv8 object detection model to identify GUI widgets (Buttons, Labels, Entry fields) in screenshots.

## Dataset

- **Total Images**: 100 annotated screenshots
- **Train Set**: 80 images
- **Validation Set**: 20 images
- **Classes**: 
  - Button (class 0)
  - Label (class 1)
  - Entry (class 2)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if using the virtual environment:
```bash
.venv/bin/pip install -r requirements.txt
```

### 2. Prepare Dataset

Convert annotations to YOLO format:
```bash
python prepare_yolo_dataset.py
```

This creates the `yolo_dataset` directory with the following structure:
```
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

## Training

### Start Training

```bash
python train_model.py
```

**Training Configuration:**
- Model: YOLOv8 Nano (yolov8n)
- Epochs: 100
- Batch Size: 16
- Image Size: 640x640
- Early Stopping: Patience of 20 epochs

The training will:
- Use transfer learning from pretrained YOLO weights
- Apply data augmentation (rotation, scaling, HSV adjustments)
- Save checkpoints and training plots
- Validate on the validation set during training

**Output:**
- Best model: `gui_widget_detection/yolov8_training/weights/best.pt`
- Last model: `gui_widget_detection/yolov8_training/weights/last.pt`
- Training plots and metrics: `gui_widget_detection/yolov8_training/`

### Monitor Training

Training metrics will be displayed in the console. Look for:
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Precision**: Percentage of correct detections
- **Recall**: Percentage of widgets found
- **Loss values**: Should decrease over time

## Testing

### Evaluate on Validation Set

```bash
python test_model.py
```

This will:
- Run validation on the test set
- Print mAP50, mAP50-95, Precision, and Recall
- Test on 10 sample images
- Save annotated results to `test_results/`

