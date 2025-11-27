# GUI Widget Detection with YOLOv8 🎯

An AI-powered system for detecting and classifying GUI widgets (Buttons, Labels, Entry fields) in screenshots using YOLOv8 object detection.

## 🎉 Status: Model Trained & Ready!

✓ Model trained with **96.31% mAP@50** (clean data, no text leakage)  
✓ Detects Buttons, Labels, and Entry fields  
✓ Fast inference (~23ms per image on CPU)  
✓ High precision (96.17%) - few false positives  
✓ 100 annotated training examples  

## 📋 Project Overview

This project consists of two main components:

1. **Synthetic GUI Generator** - Creates randomized GUI layouts with annotations
2. **YOLOv8 Object Detector** - Trains and detects GUI widgets in screenshots

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Try the Model

```bash
# Test on specific image
python test_model.py screenshots/screenshot_0.png

# Create visualizations
python visualize_results.py
```

### 3. Generate More Training Data (Optional)

```bash
# Open and run the notebook to generate more GUIs
jupyter notebook randomized_gui_loop.ipynb
```

### 4. Retrain (Optional)

```bash
# Prepare dataset
python prepare_yolo_dataset.py

# Train model
python train_model.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| mAP@50 | 96.31% |
| mAP@50-95 | 88.28% |
| Precision | 96.17% |
| Recall | 90.68% |

See [TRAINING_RESULTS.md](TRAINING_RESULTS.md) for detailed results.

## 📁 Project Structure

```
GUI-Research-Project/
├── 📊 Data
│   ├── annotations/              # JSON annotations for training data
│   ├── screenshots/              # GUI screenshots
│   └── yolo_dataset/            # YOLO-formatted dataset
│
├── 🤖 Models
│   └── gui_widget_detection/    # Trained model weights & results
│
├── 🛠️ Scripts
│   ├── prepare_yolo_dataset.py  # Convert annotations to YOLO format
│   ├── train_model.py           # Train YOLOv8 model
│   ├── test_model.py            # Evaluate model performance
│   ├── visualize_results.py     # Create visualizations
│   └── demo.py                  # Quick demo script
│
├── 🎨 GUI Generation
│   ├── randomized_gui.py        # GUI generation script
│   ├── randomized_gui_loop.ipynb # Batch GUI generation
│   └── randomized_gui.ipynb     # Interactive GUI generation
│
├── 📝 Configuration
│   ├── dataset.yaml             # YOLO dataset config
│   └── requirements.txt         # Python dependencies
│
└── 📖 Documentation
    ├── README.md                # This file
    ├── README_MODEL.md          # Detailed model documentation
    └── TRAINING_RESULTS.md      # Training results & performance
```

## 🎯 Detected Widget Types

- **Button** 🔴 - Interactive buttons (97.3% precision, 96.7% recall)
- **Label** 🟢 - Text labels (93.5% precision, 75.4% recall)
- **Entry** 🔵 - Input fields (97.7% precision, 100% recall)

## 📈 Training Details

- **Model**: YOLOv8 Nano (3M parameters)
- **Training Time**: ~34 minutes (100 epochs)
- **Dataset**: 80 train / 20 validation images
- **Augmentation**: HSV, rotation, scaling, flipping, mosaic
- **Hardware**: Trained on CPU (Apple M4)

## 🔧 Advanced Options

### Retrain with Different Settings

Edit `train_model.py`:
```python
MODEL_SIZE = "yolov8s"  # Use larger model (nano/small/medium)
EPOCHS = 200            # Train longer
BATCH_SIZE = 32         # Increase batch size
```

### Generate More Training Data

Edit `randomized_gui_loop.ipynb`:
```python
for iter in tqdm(range(500)):  # Generate 500 examples
    # ... existing code ...
```


## 📚 Documentation

- [README_MODEL.md](README_MODEL.md) - Comprehensive model documentation
- [TRAINING_RESULTS.md](TRAINING_RESULTS.md) - Training results and analysis
- [Ultralytics Docs](https://docs.ultralytics.com/) - YOLOv8 documentation

## 🎨 Visualization

The project includes visualization tools to see model predictions:

```bash
# Create summary of predictions on multiple images
python visualize_results.py

# Visualize specific image
python visualize_results.py screenshots/screenshot_0.png
```

Results are color-coded:
- 🔴 Red boxes = Buttons
- 🟢 Green boxes = Labels
- 🔵 Blue boxes = Entry fields