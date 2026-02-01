# GUI Recreation from Real Screenshots

Automatic recreation of GUI interfaces from real screenshots using:
1. **Object Detection (YOLO)** - wykrywa widgety i pozycje
2. **OCR (EasyOCR)** - wyciąga tekst z widgetów  
3. **Layout Analysis** - analizuje strukturę i hierarchię
4. **Recreation** - odtwarza w Tkinter lub HTML

## 🚀 Quick Start

### 1. Instalacja zależności
```bash
pip install easyocr
```

### 2. Single screenshot recreation
```bash
python recreate_gui_from_screenshot.py
```

Domyślnie procesuje `real_padded_screenshots/calculator.png` i tworzy:
- `gui_recreations/calculator_analysis.json` - szczegółowa analiza
- `gui_recreations/calculator_recreation.html` - HTML recreation
- Tkinter window - interaktywna recreacja

### 3. Batch processing (wszystkie screenshots)
```bash
python batch_recreate_guis.py
```

Procesuje wszystkie obrazy w `real_padded_screenshots/` i tworzy:
- JSON analysis dla każdego
- HTML recreation dla każdego
- `gui_recreations/index.html` - overview page
- `gui_recreations/batch_summary.json` - statystyki

## 📋 Pipeline Overview

```
Real Screenshot
     ↓
┌────────────────────┐
│ 1. Object Detection│  → YOLO wykrywa widgety
│    (YOLO)          │    (Button, Label, Entry, etc.)
└────────────────────┘
     ↓
┌────────────────────┐
│ 2. Text Extraction │  → OCR wyciąga tekst
│    (EasyOCR)       │    z każdego widgeta
└────────────────────┘
     ↓
┌────────────────────┐
│ 3. Layout Analysis │  → Analiza pozycji,
│                    │    wykrywanie wierszy
└────────────────────┘
     ↓
┌────────────────────┐
│ 4. GUI Recreation  │  → Odtworzenie w
│                    │    Tkinter lub HTML
└────────────────────┘
```

## 📊 Output Files

### JSON Analysis
Zawiera szczegółowe info o każdym widgecie:
```json
{
  "source_image": "calculator.png",
  "image_size": {"width": 800, "height": 600},
  "widgets": [
    {
      "widget_type": "Button",
      "bbox": [100, 200, 150, 230],
      "confidence": 0.95,
      "text": "Calculate",
      "center_x": 125,
      "center_y": 215,
      "width": 50,
      "height": 30
    }
  ]
}
```

### HTML Recreation
Interaktywna strona HTML z odtworzonym GUI:
- Zachowuje pozycje widgetów
- Wyświetla wyciągnięty tekst
- Różne style dla różnych typów widgetów

## 🎯 Use Cases

1. **Reverse Engineering GUI** - odtworzenie struktury GUI z screenshota
2. **Automated Testing** - mapowanie elementów do testów
3. **UI Documentation** - automatyczna dokumentacja interfejsów
4. **Accessibility** - wyciąganie struktury dla screen readers
5. **UI Clone Detection** - porównywanie podobnych interfejsów

## 🔧 Customization

### Zmiana confidence threshold
```python
recreator = GUIRecreator()
recreator.detect_widgets("image.png", confidence=0.3)  # niższy threshold
```

### Zmiana języków OCR
```python
# W GUIRecreator.__init__:
self.ocr_reader = easyocr.Reader(['en', 'pl', 'de'], gpu=True)
```

### Zmiana scale factor dla Tkinter
```python
recreator.recreate_gui_tkinter(scale_factor=0.5)  # 50% rozmiaru
```

## 📈 Advanced Features

### Custom widget mapping
Możesz dodać własne typy widgetów w `recreate_gui_from_screenshot.py`:

```python
def _create_custom_widget(self, canvas, bbox, text):
    x1, y1, x2, y2 = bbox
    # Custom drawing logic
    canvas.create_rectangle(x1, y1, x2, y2, fill='#custom')
```

### Export do innych formatów
Możesz dodać generatory dla:
- PyQt/PySide kod
- React/Vue components  
- Flutter widgets
- Android XML layouts

## 🐛 Troubleshooting

### OCR nie działa
```bash
# EasyOCR wymaga dodatkowych dependencies
pip install torch torchvision
```

### Małe widgety nie są wykrywane
```python
# Zmniejsz confidence threshold
recreator.detect_widgets(image_path, confidence=0.15)
```

### Tekst nie jest wyciągany poprawnie
```python
# Zwiększ padding przy OCR (linia 145 w recreate_gui_from_screenshot.py)
padding = 10  # było 5
```

## 📸 Example Results

```
Input: real_padded_screenshots/calculator.png
Output:
  ✓ Detected 15 widgets
    • 10x Button
    • 3x Label  
    • 2x Entry
  ✓ Extracted text from 13/15 widgets
  ✓ Detected 4 rows
  ✓ Generated HTML recreation
```

## 🔮 Future Improvements

- [ ] Deep learning dla lepszego OCR (Tesseract alternative)
- [ ] Hierarchia parent-child między widgetami
- [ ] Style detection (colors, fonts)
- [ ] Automatic code generation (Tkinter/PyQt/HTML)
- [ ] Interactive editing w recreated GUI
- [ ] Comparison między original a recreated

## 📚 Related Files

- `test_real_world.py` - testowanie modelu na real screenshots
- `randomized_gui.py` - generowanie synthetic training data
- `train_model.py` - trenowanie YOLO modelu
