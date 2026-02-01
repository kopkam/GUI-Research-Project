# 🔄 Full Pipeline - GUI Widget Detection System

## Kompletny proces od początku do końca

---

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FAZA 1: GENEROWANIE DANYCH                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
         ┌──────────────────────────────────────────────┐
         │  randomized_gui_loop.ipynb                   │
         │  • Tkinter GUI Generator                     │
         │  • Losowe układy widgetów                    │
         │  • Różne rozmiary, style, kolory             │
         └──────────────────┬───────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  Wygenerowane dane:     │
              │  • screenshots/*.png    │
              │  • annotations/*.json   │
              └─────────────┬───────────┘
                            │
                            │ 200 par (obraz + anotacja)
                            ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                   FAZA 2: PRZYGOTOWANIE DATASETU                         │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────┐
         │  prepare_yolo_dataset.py                     │
         │  • Konwersja JSON → YOLO format              │
         │  • Train/Val split (80/20)                   │
         │  • Normalizacja bounding boxes               │
         └──────────────────┬───────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  yolo_dataset/          │
              │  ├── train/             │
              │  │   ├── images/ (160)  │
              │  │   └── labels/ (160)  │
              │  └── val/               │
              │      ├── images/ (40)   │
              │      └── labels/ (40)   │
              └─────────────┬───────────┘
                            ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                        FAZA 3: TRENING MODELU                            │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────┐
         │  train_model.py                              │
         │  • YOLOv8 Nano pretrained (COCO)             │
         │  • 100 epochs                                │
         │  • Data augmentation                         │
         │  • Adam optimizer + cosine LR                │
         └──────────────────┬───────────────────────────┘
                            ↓
                   ┌────────────────┐
                   │  Training Loop │
                   └────────┬───────┘
                            ↓
         ┌──────────────────────────────────────────────┐
         │  Każda epoka:                                │
         │  1. Forward pass (batch 16)                  │
         │  2. Loss calculation                         │
         │     • Box loss (DFL)                         │
         │     • Class loss (BCE)                       │
         │     • Objectness loss (BCE)                  │
         │  3. Backward pass (gradient descent)         │
         │  4. Validation co 1 epokę                    │
         │  5. Save best model (mAP@50)                 │
         └──────────────────┬───────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  Wytrenowany model:     │
              │  • best.pt (5.9 MB)     │
              │  • results.csv          │
              │  • confusion_matrix.png │
              └─────────────┬───────────┘
                            ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                        FAZA 4: EWALUACJA                                 │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────┐
         │  test_model.py                               │
         │  • Walidacja na 40 obrazach                  │
         │  • Metryki COCO (mAP@50, mAP@50-95)          │
         │  • Per-class performance                     │
         └──────────────────┬───────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  Wyniki:                │
              │  • mAP@50: 97.84%       │
              │  • Precision: 95.27%    │
              │  • Recall: 96.84%       │
              └─────────────┬───────────┘
                            ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                      FAZA 5: INFERENCE / UŻYCIE                          │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────┐
         │  Nowy screenshot GUI                         │
         └──────────────────┬───────────────────────────┘
                            ↓
         ┌──────────────────────────────────────────────┐
         │  Model Inference                             │
         │  1. Preprocessing (resize 640x640)           │
         │  2. Forward pass (~60ms)                     │
         │  3. NMS (Non-Maximum Suppression)            │
         │  4. Threshold filtering (conf > 0.25)        │
         └──────────────────┬───────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  Wykryte widgety:       │
              │  • Buttons + bbox       │
              │  • Labels + bbox        │
              │  • Entries + bbox       │
              │  • Tables + bbox        │
              └─────────────────────────┘
```

---

## 🔍 Szczegółowy opis każdej fazy

### FAZA 1: Generowanie syntetycznych danych 🎨

**Plik:** `randomized_gui_loop.ipynb`

**Proces:**
1. **Inicjalizacja:** Ustawienie parametrów GUI (rozmiar okna, liczba widgetów)
2. **Losowanie:** 
   - Liczba każdego typu widgetu (1-20)
   - Pozycje (grid layout)
   - Rozmiary (random z zakresu)
   - Kolory tła i tekstu
   - Treść tekstowa (random strings)
3. **Renderowanie:** Tkinter tworzy GUI i robi screenshot
4. **Anotacja:** Automatyczne zapisanie współrzędnych każdego widgetu
5. **Zapis:**
   - `screenshots/screenshot_X.png`
   - `annotations/annotation_X.json`

**Format JSON (annotation_X.json):**
```json
{
  "image": "screenshots/screenshot_X.png",
  "resolution": {"width": 1024, "height": 768},
  "widgets": [
    {
      "id": "button_0",
      "type": "Button",
      "text": "Click Me",
      "bbox": {
        "x_min": 100,
        "y_min": 50,
        "x_max": 200,
        "y_max": 100
      }
    },
    {
      "id": "label_0",
      "type": "Label",
      "text": "Hello World",
      "bbox": {...}
    },
    ...
  ]
}
```

**Wynik:** 200 par (obrazów + anotacji)

---

### FAZA 2: Konwersja do formatu YOLO 📝

**Plik:** `prepare_yolo_dataset.py`

**Proces:**
1. **Wczytanie anotacji:** Odczyt wszystkich 200 plików JSON
2. **Podział train/val:** sklearn.train_test_split (80/20, seed=42)
3. **Konwersja bbox:** 
   ```python
   # Z absolutnych współrzędnych (x_min, y_min, x_max, y_max)
   # Do formatu YOLO (x_center, y_center, width, height) znormalizowane [0,1]
   
   x_center = (x_min + x_max) / (2 * img_width)
   y_center = (y_min + y_max) / (2 * img_height)
   width = (x_max - x_min) / img_width
   height = (y_max - y_min) / img_height
   ```
4. **Mapowanie klas:**
   - Button → 0
   - Label → 1
   - Entry → 2
   - Table → 3
5. **Zapis plików:**
   - Kopiowanie obrazów do `yolo_dataset/train|val/images/`
   - Generowanie labelek `.txt` w `yolo_dataset/train|val/labels/`

**Format YOLO (.txt):**
```
0 0.5 0.3 0.15 0.08    # Button at center (0.5, 0.3), size 15%x8%
1 0.2 0.1 0.1 0.05     # Label at (0.2, 0.1)
2 0.7 0.4 0.2 0.06     # Entry at (0.7, 0.4)
```

**Wynik:** 
- 160 obrazów treningowych + labelki
- 40 obrazów walidacyjnych + labelki
- `dataset.yaml` z konfiguracją

---

### FAZA 3: Trening modelu 🧠

**Plik:** `train_model.py`

**Konfiguracja:**
```python
model = YOLO('yolov8n.pt')  # Pretrained on COCO
model.train(
    data='dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    patience=20,  # Early stopping
    project='gui_widget_detection',
    name='yolov8_training3'
)
```

**Proces treningu (każda epoka):**

1. **Data Loading (batch=16):**
   - Wczytanie 16 obrazów
   - Resize do 640×640
   - Augmentacja:
     - HSV color jittering
     - Random rotation (±10°)
     - Random flip horizontal
     - Random scaling (0.5-1.5x)
     - Mosaic (4 obrazy w jeden)

2. **Forward Pass:**
   ```
   Input (640×640×3)
      ↓
   Backbone (C2f blocks)
      ↓
   Neck (FPN + PAN)
      ↓
   Head (Classify + Regress)
      ↓
   Predictions:
   • Class logits [batch, anchors, 4 classes]
   • Box coords [batch, anchors, 4 (x,y,w,h)]
   • Objectness [batch, anchors, 1]
   ```

3. **Loss Calculation:**
   ```python
   total_loss = box_loss + cls_loss + obj_loss
   
   # Box loss (Distribution Focal Loss)
   box_loss = DFL(pred_boxes, true_boxes)
   
   # Classification loss (Binary Cross Entropy)
   cls_loss = BCE(pred_classes, true_classes)
   
   # Objectness loss
   obj_loss = BCE(pred_obj, true_obj)
   ```

4. **Backward Pass:**
   - Gradient calculation
   - Adam optimizer update
   - Learning rate: 0.01 → 0.000025 (cosine decay)

5. **Validation (co epokę):**
   - Inference na 40 obrazach val
   - Calculation mAP@50, mAP@50-95
   - Zapis best.pt jeśli mAP@50 się poprawił

**Monitoring:**
- Losses: box, cls, dfl (powinny maleć)
- Metrics: precision, recall, mAP (powinny rosnąć)
- Learning rate (maleje według cosine schedule)

**Wynik po 100 epokach:**
- `best.pt` - najlepszy model (mAP@50 = 97.87% w epoce 68)
- `last.pt` - model z ostatniej epoki
- `results.csv` - metryki z każdej epoki
- Wykresy: losses, precision, recall, confusion matrix

---

### FAZA 4: Ewaluacja i testowanie 📊

**Plik:** `test_model.py`

**Proces:**
1. **Load model:** `YOLO('gui_widget_detection/yolov8_training3/weights/best.pt')`
2. **Validation run:**
   ```python
   metrics = model.val(
       data='dataset.yaml',
       split='val',
       batch=16,
       imgsz=640
   )
   ```
3. **Metryki COCO:**
   - **mAP@50:** Average Precision przy IoU threshold = 0.5
   - **mAP@50-95:** AP uśrednione dla IoU 0.5-0.95 (co 0.05)
   - **Precision:** TP / (TP + FP)
   - **Recall:** TP / (TP + FN)

4. **Per-class analysis:**
   - Osobne metryki dla każdej klasy
   - Confusion matrix
   - Analiza błędów

**Wyniki finalne:**
```
Overall:
  mAP@50:     97.84%
  mAP@50-95:  92.02%
  Precision:  95.27%
  Recall:     96.84%

Per-class:
  Button:  99.12% mAP@50, 100% recall
  Label:   93.58% mAP@50, 88.37% recall
  Entry:   99.15% mAP@50, 100% recall
  Table:   99.50% mAP@50, 100% recall
```

---

### FAZA 5: Inference na nowych obrazach 🚀

**Użycie modelu:**

```python
from ultralytics import YOLO

# 1. Load model
model = YOLO('gui_widget_detection/yolov8_training3/weights/best.pt')

# 2. Predict
results = model.predict(
    source='new_screenshot.png',
    conf=0.25,  # Confidence threshold
    iou=0.45    # NMS IoU threshold
)

# 3. Extract predictions
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        
        # Class and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Class name
        class_name = ['Button', 'Label', 'Entry', 'Table'][class_id]
        
        print(f'{class_name}: {confidence:.2f} at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]')
```

**Proces inference:**
1. **Preprocessing:**
   - Resize obrazu do 640×640 (letterbox padding)
   - Normalizacja RGB (0-255 → 0-1)
   - Convert to tensor

2. **Model forward pass:**
   - Backbone extracts features
   - Neck fuses multi-scale features
   - Head predicts boxes + classes
   - Time: ~60ms on CPU

3. **Postprocessing:**
   - **NMS (Non-Maximum Suppression):**
     - Usuwa duplikujące się detekcje
     - Zachowuje box z najwyższym confidence
   - **Confidence filtering:**
     - Odrzuca detekcje < 0.25 confidence
   - **Denormalizacja:**
     - Konwersja bbox z [0,1] → pixels

4. **Output:**
   - Lista wykrytych widgetów
   - Każdy z: class, confidence, bbox

---

## 📈 Timeline całego projektu

```
┌─────────────────────────────────────────────────────────────┐
│  Dzień 1-2: Generowanie danych                              │
│  • Rozwój randomized_gui.py                                 │
│  • Generacja 200 GUI + anotacji                             │
│  • Czas: ~2h                                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Dzień 3: Przygotowanie datasetu                            │
│  • Implementacja prepare_yolo_dataset.py                    │
│  • Konwersja + split                                        │
│  • Czas: ~1h                                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Dzień 4-5: Trening                                         │
│  • Setup train_model.py                                     │
│  • Training (100 epochs = 123 min)                          │
│  • Analiza wyników                                          │
│  • Czas: ~3h                                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Dzień 6: Ewaluacja i dokumentacja                          │
│  • test_model.py - metryki                                  │
│  • Porównanie z poprzednimi wersjami                        │
│  • Dokumentacja (README, MODEL_ARCHITECTURE)                │
│  • Czas: ~2h                                                │
└─────────────────────────────────────────────────────────────┘

Total project time: ~8-10 godzin (bez czasu treningu)
```

---

## 🔄 Iteracyjne ulepszenia

### Wersja 1 (100 próbek, 3 klasy)
- Dataset: 100 GUI screenshots
- Klasy: Button, Label, Entry
- Wynik: 97.30% mAP@50
- Problem: Słaby recall dla Labels (49.92%)

### Wersja 2 (próba poprawy)
- Modyfikacje w augmentacji
- Wynik: 96.00% mAP@50
- Wnioski: Gorsze wyniki, wycofano

### Wersja 3 (200 próbek, 4 klasy) ✅ CURRENT
- Dataset: 200 GUI screenshots (2x więcej)
- Klasy: Button, Label, Entry, **Table** (nowa)
- Wynik: **97.84% mAP@50**
- Ulepszenia:
  - Label recall: 49.92% → **88.37%** (+38.45%)
  - Button precision: 90.31% → **98.23%** (+7.92%)
  - Dodano Table z 99.50% mAP

---

## 💡 Kluczowe decyzje projektowe

1. **Syntetyczne dane zamiast ręcznych:**
   - ✅ Szybka generacja
   - ✅ Automatyczne anotacje (brak błędów)
   - ✅ Pełna kontrola nad diversity
   - ⚠️ Może nie generalizować na real-world UI

2. **YOLOv8 Nano zamiast większych wariantów:**
   - ✅ Szybki inference (60ms vs 200ms+)
   - ✅ Mały rozmiar (6MB vs 50MB+)
   - ✅ Wystarczająca accuracy (97.84%)
   - ⚠️ Mniejsza capacity dla złożonych przypadków

3. **Transfer learning z COCO:**
   - ✅ Szybsza konwergencja
   - ✅ Lepsze feature extraction
   - ✅ Oszczędność czasu treningu
   - ✅ Wyższe wyniki niż training from scratch

4. **80/20 split zamiast 70/15/15:**
   - ✅ Więcej danych treningowych
   - ✅ Wystarczająco dużo validation samples
   - ⚠️ Brak oddzielnego test setu

---

## 🎯 Wyniki vs Założenia projektu

| Cel | Założenie | Osiągnięty wynik | Status |
|-----|-----------|------------------|--------|
| Accuracy | >90% mAP@50 | 97.84% | ✅ PRZEKROCZONY |
| Speed | <100ms | ~60ms | ✅ OSIĄGNIĘTY |
| Classes | 3-5 | 4 | ✅ OSIĄGNIĘTY |
| Dataset | >100 samples | 200 | ✅ OSIĄGNIĘTY |
| Model size | <10MB | 5.9MB | ✅ OSIĄGNIĘTY |

---

## 📚 Narzędzia i technologie użyte w pipeline

### Data Generation
- **Tkinter:** GUI rendering
- **PIL/Pillow:** Screenshot capture
- **Python:** Scripting and automation

### Data Preprocessing
- **scikit-learn:** Train/val split
- **NumPy:** Numerical operations
- **JSON:** Annotation format

### Training
- **PyTorch 2.9.1:** Deep learning framework
- **Ultralytics YOLOv8:** Object detection model
- **AdamW optimizer:** Training optimization
- **Cosine LR scheduler:** Learning rate decay

### Evaluation
- **COCO metrics:** mAP calculation
- **Matplotlib:** Plotting results
- **Pandas:** Data analysis

### Infrastructure
- **Apple M4 CPU:** Training hardware
- **Python 3.13:** Programming language
- **Git:** Version control
- **GitHub:** Code repository

---

## 🚀 Dalsze kierunki rozwoju

1. **Więcej klas widgetów:**
   - Checkboxes, Radio buttons
   - Sliders, Progress bars
   - Dropdowns, ComboBoxes
   - Menus, Toolbars

2. **Real-world testing:**
   - Test na rzeczywistych aplikacjach
   - Różne UI frameworks (Qt, GTK, WPF)
   - Różne platformy (Windows, macOS, Linux)

3. **Optymalizacje:**
   - Quantization dla szybszego inference
   - Export do ONNX/TensorRT
   - Mobile deployment (iOS/Android)

4. **Rozszerzenie funkcjonalności:**
   - Hierarchia widgetów (parent-child)
   - Wykrywanie stanu (enabled/disabled)
   - OCR integracja dla text extraction

---

**Dokumentacja projektu:**
- [README.md](README.md) - Ogólny opis
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) - Architektura modelu
- [TRAINING_RESULTS.md](TRAINING_RESULTS.md) - Wyniki treningu
- [FULL_PIPELINE.md](FULL_PIPELINE.md) - Ten dokument

**Repository:** https://github.com/kopkam/GUI-Research-Project
