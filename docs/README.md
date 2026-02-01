# GUI Widget Detection with YOLOv8 🎯

An AI-powered system for detecting and classifying GUI widgets (Buttons, Labels, Entry fields) in screenshots using YOLOv8 object detection.

## 🎉 Status: Model Trained & Ready!

✓ Model trained with **97.84% mAP@50** (200 samples, 4 classes including Tables)  
✓ Detects Buttons, Labels, Entry fields, and Tables  
✓ Fast inference (~60ms per image on CPU)  
✓ High precision (95.27%) - few false positives  
✓ 200 annotated training examples  

## 📋 Project Overview

This project consists of three main components:

1. **Synthetic GUI Generator** - Creates randomized GUI layouts with annotations
2. **YOLOv8 Object Detector** - Trains and detects GUI widgets in screenshots
3. **GUI Recreation System** - Recreates real-world GUIs from screenshots using AI (NEW! 🎨)

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

### 2b. NEW: GUI Recreation from Real Screenshots 🎨

```bash
# Install OCR dependency
pip install easyocr

# Interactive demo - detect widgets, extract text, recreate GUI
python demo_gui_recreation.py

# Batch process all real screenshots
python batch_recreate_guis.py

# Generate executable code (Tkinter/PyQt/HTML)
python generate_gui_code.py real_padded_screenshots/calculator.png
```

**What it does:**
1. Detects widgets using trained YOLO model
2. Extracts text from each widget using OCR
3. Analyzes layout structure (rows, columns)
4. Recreates GUI in Tkinter, HTML, or as executable code

See [README_GUI_RECREATION.md](README_GUI_RECREATION.md) for details.

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
| mAP@50 | 97.84% |
| mAP@50-95 | 92.02% |
| Precision | 95.27% |
| Recall | 96.84% |

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
├── 🔄 GUI Recreation (NEW!)
│   ├── recreate_gui_from_screenshot.py  # Main recreation pipeline
│   ├── demo_gui_recreation.py           # Interactive demo
│   ├── batch_recreate_guis.py           # Process all screenshots
│   ├── generate_gui_code.py             # Generate Tkinter/PyQt/HTML code
│   ├── compare_original_recreated.py    # Visual comparison tool
│   └── gui_recreations/                 # Output directory
│
├── 📝 Configuration
│   ├── dataset.yaml             # YOLO dataset config
│   └── requirements.txt         # Python dependencies
│
└── 📖 Documentation
    ├── README.md                # This file
    ├── README_MODEL.md          # Detailed model documentation
    ├── README_GUI_RECREATION.md # GUI recreation system (NEW!)
    └── TRAINING_RESULTS.md      # Training results & performance
```

## 🎯 Detected Widget Types

- **Button** 🔴 - Interactive buttons (97.3% precision, 96.7% recall)
- **Label** 🟢 - Text labels (93.5% precision, 75.4% recall)
- **Entry** 🔵 - Input fields (97.7% precision, 100% recall)
- **Table** 🟡 - Data tables (NEW in expanded dataset)

## 📈 Training Details

- **Model**: YOLOv8 Nano (3M parameters)
- **Training Time**: ~60 minutes (100 epochs)
- **Dataset**: 160 train / 40 validation images (200 total)
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