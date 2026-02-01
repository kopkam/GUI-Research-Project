# GUI Recreation - Quick Reference

## 🚀 Quickstart Commands

```bash
# Install dependencies
pip install easyocr

# Interactive demo
python demo_gui_recreation.py

# Process single screenshot
python recreate_gui_from_screenshot.py

# Process all screenshots
python batch_recreate_guis.py

# Generate code
python generate_gui_code.py real_padded_screenshots/calculator.png

# Create comparisons
python compare_original_recreated.py
```

## 📊 What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `demo_gui_recreation.py` | Interactive step-by-step demo | Tkinter window + HTML |
| `recreate_gui_from_screenshot.py` | Main recreation pipeline | JSON + HTML + Tkinter |
| `batch_recreate_guis.py` | Process all screenshots at once | Multiple JSON/HTML + summary |
| `generate_gui_code.py` | Generate executable code | .py (Tkinter/PyQt) + .html |
| `compare_original_recreated.py` | Visual comparison | PNG comparison images + report |

## 🎯 Common Use Cases

### 1. Quick Preview of a Screenshot
```bash
python demo_gui_recreation.py
# Select screenshot from menu
# See step-by-step analysis
```

### 2. Batch Process All Real Screenshots
```bash
python batch_recreate_guis.py
# Processes all files in real_padded_screenshots/
# Creates gui_recreations/ directory with all outputs
# Open gui_recreations/index.html to browse
```

### 3. Generate Code for Specific GUI
```bash
python generate_gui_code.py real_padded_screenshots/calculator.png
# Creates 3 files in gui_recreations/code/:
#   - calculator_tkinter.py (run with: python calculator_tkinter.py)
#   - calculator_pyqt.py (run with: python calculator_pyqt.py)
#   - calculator_interactive.html (open in browser)
```

### 4. Compare Original vs Recreated
```bash
python compare_original_recreated.py
# Creates side-by-side comparisons
# Generates report with metrics
# Open gui_recreations/comparisons/comparison_report.html
```

## 🔧 Customization

### Change Detection Sensitivity
Edit confidence threshold in script:
```python
recreator = GUIRecreator()
recreator.detect_widgets(image_path, confidence=0.15)  # Lower = more detections
```

### Change OCR Languages
In `recreate_gui_from_screenshot.py`, line ~50:
```python
self.ocr_reader = easyocr.Reader(['en', 'pl', 'de'], gpu=False)
```

### Change Scale Factor for Tkinter
```python
recreator.recreate_gui_tkinter(scale_factor=0.5)  # 50% size
```

## 📁 Output Files Explained

### JSON Analysis (`*_analysis.json`)
```json
{
  "source_image": "calculator.png",
  "widgets": [
    {
      "widget_type": "Button",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "text": "Calculate",
      "center_x": 125,
      "center_y": 215
    }
  ]
}
```

### HTML Recreation (`*_recreation.html`)
- Interactive HTML page
- Click to open in browser
- Widgets positioned exactly as detected

### Generated Code (`*_tkinter.py`, `*_pyqt.py`)
- Executable Python code
- Run to see recreated GUI
- Modify as needed for your project

### Comparison Images (`*_comparison.png`)
- 3-panel view: Original | Detected | Recreated
- Color-coded bounding boxes
- Visual quality assessment

## 💡 Tips & Tricks

1. **Better OCR results**: Use higher resolution screenshots
2. **More detections**: Lower confidence threshold to 0.15-0.20
3. **Cleaner output**: Remove low-confidence detections in post-processing
4. **Custom widgets**: Edit widget creation functions in `recreate_gui_from_screenshot.py`
5. **Faster processing**: Set `gpu=True` in EasyOCR if you have CUDA

## 🐛 Troubleshooting

**OCR not working?**
```bash
pip install torch torchvision
pip install easyocr
```

**No widgets detected?**
- Lower confidence threshold
- Check if model exists at `gui_widget_detection/yolov8_training4/weights/best.pt`
- Try on synthetic screenshots first

**Tkinter window too small/large?**
```python
recreator.recreate_gui_tkinter(scale_factor=0.8)  # Adjust as needed
```

**Text extraction poor?**
- Use higher quality screenshots
- Increase OCR padding (line 145 in recreate_gui_from_screenshot.py)
- Try different OCR languages

## 📈 Performance

Typical processing time per screenshot:
- Object Detection: ~0.1s
- OCR (all widgets): ~2-5s
- HTML generation: <0.1s
- Tkinter recreation: instant

On 26 real screenshots:
- Total time: ~3-5 minutes
- Average: 7-12s per screenshot

## 🎨 Example Workflow

```bash
# 1. Quick preview
python demo_gui_recreation.py
# Select "calculator.png" → inspect results

# 2. Generate all formats
python generate_gui_code.py real_padded_screenshots/calculator.png
# Creates Tkinter, PyQt, and HTML versions

# 3. Run generated code
python gui_recreations/code/calculator_tkinter.py
# See recreated GUI in action!

# 4. Create comparison
python compare_original_recreated.py
# Open gui_recreations/comparisons/comparison_report.html
```

## 📚 See Also

- [README_GUI_RECREATION.md](README_GUI_RECREATION.md) - Full documentation
- [README.md](README.md) - Main project README
- [README_MODEL.md](README_MODEL.md) - Model details
