# 🏗️ Model Architecture - GUI Widget Detection System

## Research Project Presentation
**Date:** January 2026  
**Model:** YOLOv8 Nano for GUI Widget Detection

---

## 🎯 1. Overview

**Objective:** Automated detection and classification of GUI widgets in screenshots using deep learning.

**Model Type:** YOLOv8 Nano (You Only Look Once - Version 8, Nano variant)

**Task:** Multi-class Object Detection

---

## 📊 2. Model Specifications

| Specification | Value |
|--------------|-------|
| **Architecture** | YOLOv8 Nano |
| **Total Parameters** | 3,011,628 (~3M) |
| **Model Layers** | 129 layers |
| **GFLOPs** | 8.2 |
| **Input Size** | 640×640 pixels (RGB) |
| **Output Classes** | 4 (Button, Label, Entry, Table) |
| **Model Size** | ~6 MB |
| **Inference Speed** | ~60ms per image (CPU - Apple M4) |

---

## 🧠 3. Architecture Components

### 3.1 Backbone (Feature Extraction)
```
Input Image (640×640×3)
    ↓
Conv + Batch Norm + SiLU Activation
    ↓
C2f Modules (CSP Bottleneck with 2 Convolutions)
    ↓
Progressive Downsampling (640 → 320 → 160 → 80 → 40)
    ↓
Feature Maps at Multiple Scales
```

**Key Features:**
- **C2f Blocks:** Cross-Stage Partial (CSP) bottleneck with 2 convolutions
- **Efficient design:** Reduced parameters while maintaining accuracy
- **Multi-scale features:** Captures both small and large widgets

### 3.2 Neck (Feature Fusion)
```
Feature Pyramid Network (FPN)
    ↓
Path Aggregation Network (PAN)
    ↓
Multi-scale Feature Fusion
```

**Purpose:**
- Combines features from different scales
- Enhances detection of widgets of various sizes
- Improves localization accuracy

### 3.3 Head (Detection)
```
Classification Branch → Widget Class Probabilities (4 classes)
    ↓
Regression Branch → Bounding Box Coordinates (x, y, w, h)
    ↓
Objectness Score → Confidence of Detection
```

**Output Format:**
- Class predictions: [Button, Label, Entry, Table]
- Bounding boxes: [x_center, y_center, width, height] (normalized 0-1)
- Confidence scores: 0.0 - 1.0

---

## 📈 4. Training Configuration

### 4.1 Dataset
- **Total Images:** 200 annotated GUI screenshots
- **Training Set:** 160 images (80%)
- **Validation Set:** 40 images (20%)
- **Data Augmentation:** 
  - HSV color jittering
  - Random rotation (±10°)
  - Random scaling (0.5-1.5x)
  - Horizontal flipping
  - Mosaic augmentation

### 4.2 Training Parameters
```yaml
Model: yolov8n.pt (pretrained on COCO)
Epochs: 100
Batch Size: 16
Image Size: 640×640
Optimizer: AdamW
Learning Rate: 0.01 (with cosine decay)
Early Stopping: Patience 20 epochs
Hardware: CPU (Apple M4)
Training Time: 123.2 minutes (~2 hours)
```

### 4.3 Loss Function
Multi-task loss combining:
- **Classification Loss:** Binary Cross-Entropy
- **Bounding Box Loss:** Distribution Focal Loss (DFL)
- **Objectness Loss:** Binary Cross-Entropy

---

## 🎯 5. Model Performance

### 5.1 Overall Metrics (Validation Set)
| Metric | Score |
|--------|-------|
| **mAP@50** | 97.84% |
| **mAP@50-95** | 92.02% |
| **Precision** | 95.27% |
| **Recall** | 96.84% |
| **Inference Speed** | ~60ms/image (CPU) |

### 5.2 Per-Class Performance
| Widget Type | Precision | Recall | mAP@50 | mAP@50-95 |
|-------------|-----------|--------|--------|-----------|
| **Button** | 98.23% | 98.97% | 99.12% | 94.31% |
| **Label** | 89.08% | 88.37% | 93.58% | 77.47% |
| **Entry** | 96.78% | 100.00% | 99.15% | 96.81% |
| **Table** | 96.98% | 100.00% | 99.50% | 99.50% |

### 5.3 Comparison with Previous Version
| Metric | Old (100 samples) | New (200 samples) | Improvement |
|--------|-------------------|-------------------|-------------|
| mAP@50 | 97.30% | 97.84% | +0.54% |
| Precision | 92.21% | 95.27% | +2.86% |
| Recall | 95.64% | 96.84% | +0.74% |
| Classes | 3 | 4 | +Table class |

---

## 🔄 6. System Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: GUI Screenshot                     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING                                   │
│  • Resize to 640×640                                        │
│  • Normalize RGB values (0-255 → 0-1)                       │
│  • Convert to tensor                                         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              YOLOV8 NANO MODEL                              │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   BACKBONE   │ → │     NECK     │ → │     HEAD     │ │
│  │  (C2f + Conv)│    │  (FPN + PAN) │    │ (Classify +  │ │
│  │              │    │              │    │   Regress)   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              POSTPROCESSING                                  │
│  • Non-Maximum Suppression (NMS)                            │
│  • Confidence threshold filtering (>0.25)                   │
│  • Coordinate denormalization                               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: Detected Widgets                        │
│  • Class: [Button, Label, Entry, Table]                     │
│  • Bounding Box: [x, y, width, height]                      │
│  • Confidence Score: 0.0 - 1.0                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 💾 7. Model Files Structure

```
gui_widget_detection/
└── yolov8_training3/
    ├── weights/
    │   ├── best.pt          # Best model checkpoint (5.9MB)
    │   └── last.pt          # Last epoch checkpoint
    ├── results.csv          # Training metrics per epoch
    ├── results.png          # Training curves visualization
    ├── confusion_matrix.png # Confusion matrix
    └── args.yaml           # Training configuration
```

---

## 🔬 8. Technical Highlights

### 8.1 Why YOLOv8 Nano?
✅ **Fast:** Real-time inference (~60ms on CPU)  
✅ **Lightweight:** Only 3M parameters, 6MB model size  
✅ **Accurate:** 97.84% mAP@50 on validation set  
✅ **Versatile:** Handles multiple widget types and sizes  
✅ **Pretrained:** Transfer learning from COCO dataset  

### 8.2 Novel Contributions
1. **Synthetic GUI Dataset:** Generated 200 diverse GUI layouts with automatic annotations
2. **Table Detection:** Extended to detect complex UI elements (tables)
3. **High Accuracy:** Achieved near-perfect detection on Entry and Table widgets (100% recall)

### 8.3 Key Innovations in YOLOv8
- **Anchor-free design:** Eliminates manual anchor box tuning
- **Improved backbone:** C2f modules for better feature extraction
- **Task-aligned assigner:** Better positive/negative sample assignment
- **Distribution Focal Loss:** More accurate bounding box regression

---

## 📉 9. Training Progress

### Loss Curves
- **Box Loss:** Decreased from 1.5 to 0.40 (training) and 0.40 (validation)
- **Classification Loss:** Decreased from 1.2 to 0.46 (training) and 0.41 (validation)
- **Convergence:** Stable after ~50 epochs, continued training to 100 for best results

### Learning Rate Schedule
- Initial: 0.01
- Final: 0.000025
- Schedule: Cosine annealing with warmup

---

## 🚀 10. Applications & Use Cases

1. **Automated UI Testing:** Identify and interact with GUI elements
2. **Accessibility Audits:** Analyze UI component distribution
3. **UI/UX Analysis:** Extract layout information from screenshots
4. **Documentation Generation:** Automatic UI element cataloging
5. **Reverse Engineering:** Reconstruct UI structures from images

---

## 📚 11. References & Technologies

**Framework:**
- Ultralytics YOLOv8 (2023)
- PyTorch 2.9.1

**Dataset Tools:**
- Tkinter (GUI generation)
- JSON (annotation format)
- Python 3.13

**Evaluation:**
- COCO metrics (mAP@50, mAP@50-95)
- Precision, Recall, F1-score

---

## 🎓 12. Conclusion

The YOLOv8 Nano model successfully achieves **high-accuracy GUI widget detection** with:
- ✅ 97.84% mAP@50
- ✅ Fast inference (60ms/image)
- ✅ Lightweight architecture (3M parameters)
- ✅ 4 widget classes with excellent per-class performance

The model demonstrates the effectiveness of **transfer learning** and **synthetic data generation** for specialized object detection tasks in the GUI domain.

---

**Project Repository:** [GitHub - GUI-Research-Project](https://github.com/kopkam/GUI-Research-Project)

**Model Weights:** `gui_widget_detection/yolov8_training3/weights/best.pt`
