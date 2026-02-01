# GUI Widget Detection with YOLOv8 🎯

An AI-powered system for detecting and classifying GUI widgets (Buttons, Labels, Entry fields) in screenshots using YOLOv8 object detection.

## 🎉 Status: Model Trained & Ready!

✓ Model trained with **97.84% mAP@50** (200 samples, 4 classes including Tables)  
✓ Detects Buttons, Labels, Entry fields, and Tables  
✓ Fast inference (~60ms per image on CPU)  
✓ High precision (95.27%) - few false positives  
✓ 600 annotated training examples  
✓ **NEW!** GUI Recreation System - recreate GUIs from real screenshots using AI + OCR

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
python scripts/inference/test_model.py data/screenshots/screenshot_0.png

# Create visualizations
python scripts/inference/visualize_results.py
```

### 2b. NEW: GUI Recreation from Real Screenshots 🎨

```bash
# Install OCR dependency
pip install easyocr

# Interactive demo - detect widgets, extract text, recreate GUI
python scripts/recreation/demo_gui_recreation.py

# Batch process all real screenshots
python scripts/recreation/batch_recreate_guis.py

# Generate executable code (Tkinter/PyQt/HTML)
python scripts/recreation/generate_gui_code.py data/real_screenshots/calculator.png
```

**What it does:**
1. Detects widgets using trained YOLO model
2. Extracts text from each widget using OCR
3. Analyzes layout structure (rows, columns)
4. Recreates GUI in Tkinter, HTML, or as executable code

See [docs/README_GUI_RECREATION.md](docs/README_GUI_RECREATION.md) for details.

### 3. Generate More Training Data (Optional)

```bash
# Open and run the notebook to generate more GUIs
jupyter notebook notebooks/randomized_gui_loop.ipynb
```

### 4. Retrain (Optional)

```bash
# Prepare dataset
python scripts/training/prepare_yolo_dataset.py

# Train model
python scripts/training/train_model.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| mAP@50 | 97.84% |
| mAP@50-95 | 92.02% |
| Precision | 95.27% |
| Recall | 96.84% |

See [docs/TRAINING_RESULTS.md](docs/TRAINING_RESULTS.md) for detailed results.

## 📁 Project Structure

```
GUI-Research-Project/
├── 📊 data/                     # Training and test data
│   ├── annotations/             # JSON annotations (600 samples)
│   ├── screenshots/             # Synthetic GUI screenshots
│   ├── real_screenshots/        # Real-world screenshots
│   └── improved_data/           # Enhanced training data
│
├── 🤖 models/                   # Trained models
│   ├── gui_widget_detection/    # YOLOv8 training results
│   └── yolov8n.pt              # Base YOLO model
│
├── 📦 datasets/                 # Prepared datasets
│   └── yolo_dataset/           # YOLO-formatted dataset
│
├── 🛠️ scripts/                  # All scripts organized by function
│   ├── training/               # Training & dataset preparation
│   │   ├── train_model.py
│   │   └── prepare_yolo_dataset.py
│   │
│   ├── inference/              # Testing & visualization
│   │   ├── test_model.py
│   │   ├── test_real_world.py
│   │   └── visualize_results.py
│   │
│   ├── recreation/             # GUI Recreation (NEW!)
│   │   ├── recreate_gui_from_screenshot.py
│   │   ├── demo_gui_recreation.py
│   │   ├── batch_recreate_guis.py
│   │   ├── generate_gui_code.py
│   │   └── compare_original_recreated.py
│   │
│   ├── analysis/               # Analysis & metrics
│   │   ├── compare_trainings.py
│   │   ├── get_per_class_metrics.py
│   │   └── print_final_stats.py
│   │
│   └── utils/                  # Utilities
│       └── randomized_gui.py
│
├── 📈 results/                  # All outputs
│   ├── gui_recreations/        # GUI recreation outputs
│   ├── test_outputs/           # Test results
│   ├── real_world/             # Real-world test results
│   ├── presentation/           # Presentation materials
│   └── runs/                   # Training runs
│
├── 📓 notebooks/                # Jupyter notebooks
│   ├── randomized_gui.ipynb
│   ├── randomized_gui_loop.ipynb
│   ├── gui_recreator.ipynb
│   └── sandbox.ipynb
│
├── 📖 docs/                     # Documentation
│   ├── README.md               # This file
│   ├── README_MODEL.md         # Model documentation
│   ├── README_GUI_RECREATION.md # Recreation system docs
│   ├── RECREATION_QUICKSTART.md # Quick reference
│   ├── TRAINING_RESULTS.md     # Training results
│   ├── FULL_PIPELINE.md        # Complete pipeline
│   └── presentation/           # Presentation slides
│
├── 📝 Configuration
│   ├── dataset.yaml            # YOLO dataset config
│   ├── requirements.txt        # Python dependencies
│   └── .gitignore
│
└── 🎯 Quick Access
    └── ideas.txt               # Project ideas & notes
```

## 🎯 Detected Widget Types

- **Button** 🔴 - Interactive buttons (97.3% precision, 96.7% recall)
- **Label** 🟢 - Text labels (93.5% precision, 75.4% recall)
- **Entry** 🔵 - Input fields (97.7% precision, 100% recall)
- **Table** 🟡 - Data tables (NEW in expanded dataset)

## 📈 Training Details

- **Model**: YOLOv8 Nano (3M parameters)
- **Training Time**: ~60 minutes (100 epochs)
- **Dataset**: 160 train / 40 validation images (200 total base, 600 available)
- **Augmentation**: HSV, rotation, scaling, flipping, mosaic
- **Hardware**: Trained on CPU (Apple M4)

## 🔧 Advanced Options

### Retrain with Different Settings

Edit `scripts/training/train_model.py`:
```python
MODEL_SIZE = "yolov8s"  # Use larger model (nano/small/medium)
EPOCHS = 200            # Train longer
BATCH_SIZE = 32         # Increase batch size
```

### Generate More Training Data

Edit `notebooks/randomized_gui_loop.ipynb`:
```python
for iter in tqdm(range(500)):  # Generate 500 examples
    # ... existing code ...
```

## 📚 Documentation

- [docs/README_MODEL.md](docs/README_MODEL.md) - Comprehensive model documentation
- [docs/README_GUI_RECREATION.md](docs/README_GUI_RECREATION.md) - GUI recreation system
- [docs/RECREATION_QUICKSTART.md](docs/RECREATION_QUICKSTART.md) - Quick reference guide
- [docs/TRAINING_RESULTS.md](docs/TRAINING_RESULTS.md) - Training results and analysis
- [Ultralytics Docs](https://docs.ultralytics.com/) - YOLOv8 documentation

## 🎨 Visualization

The project includes visualization tools to see model predictions:

```bash
# Create summary of predictions on multiple images
python scripts/inference/visualize_results.py

# Visualize specific image
python scripts/inference/visualize_results.py data/screenshots/screenshot_0.png
```

Results are color-coded:
- 🔴 Red boxes = Buttons
- 🟢 Green boxes = Labels
- 🔵 Blue boxes = Entry fields

## 🔮 GUI Recreation Features

- **Object Detection** - YOLO detects all widgets
- **Text Extraction** - OCR extracts text (96% success rate)
- **Layout Analysis** - Automatic row/column detection
- **Code Generation** - Outputs working Tkinter/PyQt/HTML code
- **Visual Comparison** - Side-by-side original vs recreated
- **Batch Processing** - Process multiple screenshots at once

## 🚀 Future Improvements

- [ ] Deep learning for better OCR
- [ ] Hierarchical widget relationships (parent-child)
- [ ] Style detection (colors, fonts, themes)
- [ ] Automatic React/Flutter code generation
- [ ] Real-time GUI recreation from video
- [ ] Training on more GUI frameworks (Qt, GTK, Web)

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Text extraction
- Tkinter - GUI generation and recreation
