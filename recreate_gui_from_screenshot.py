"""
GUI Recreation from Real Screenshots
=====================================
Pipeline:
1. Object Detection - wykrywa widgety i ich pozycje (YOLO)
2. OCR - wyciąga tekst z każdego widgeta (EasyOCR)
3. Layout Analysis - analizuje hierarchię i grupowanie
4. Recreation - odtwarza GUI w Tkinter
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import json
import easyocr
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import tkinter as tk
from PIL import Image, ImageTk

# Configuration
MODEL_PATH = "gui_widget_detection/yolov8_training4/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25


@dataclass
class Widget:
    """Reprezentuje wykryty widget z GUI"""
    widget_type: str  # Button, Label, Entry, etc.
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    text: str = ""  # Extracted text from OCR
    center_x: float = 0
    center_y: float = 0
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        """Calculate derived properties"""
        x1, y1, x2, y2 = self.bbox
        self.width = x2 - x1
        self.height = y2 - y1
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2


class GUIRecreator:
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize GUI recreator with YOLO model and OCR"""
        print("🔧 Initializing GUI Recreator...")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        print(f"  ✓ YOLO model loaded: {model_path}")
        
        # Initialize OCR (supports multiple languages)
        print("  ⏳ Loading EasyOCR (this may take a moment)...")
        self.ocr_reader = easyocr.Reader(['en', 'pl'], gpu=False)
        print("  ✓ EasyOCR loaded")
        
        self.widgets: List[Widget] = []
        self.image = None
        self.image_path = None
    
    def detect_widgets(self, image_path: str, confidence: float = CONFIDENCE_THRESHOLD) -> List[Widget]:
        """
        Step 1: Detect widgets using YOLO model
        """
        print(f"\n🔍 Step 1: Detecting widgets in {Path(image_path).name}")
        
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        # Run YOLO detection
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            verbose=False
        )
        
        widgets = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                conf = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                widget = Widget(
                    widget_type=class_name,
                    bbox=bbox,
                    confidence=conf
                )
                widgets.append(widget)
        
        self.widgets = widgets
        print(f"  ✓ Detected {len(widgets)} widgets")
        
        # Group by type
        widget_counts = {}
        for w in widgets:
            widget_counts[w.widget_type] = widget_counts.get(w.widget_type, 0) + 1
        
        for wtype, count in sorted(widget_counts.items()):
            print(f"    • {count}x {wtype}")
        
        return widgets
    
    def extract_text(self) -> List[Widget]:
        """
        Step 2: Extract text from each widget using OCR
        """
        print(f"\n📝 Step 2: Extracting text from {len(self.widgets)} widgets")
        
        for i, widget in enumerate(self.widgets):
            x1, y1, x2, y2 = widget.bbox
            
            # Add padding for better OCR
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(self.image.shape[1], x2 + padding)
            y2 = min(self.image.shape[0], y2 + padding)
            
            # Crop widget region
            widget_crop = self.image[y1:y2, x1:x2]
            
            if widget_crop.size == 0:
                continue
            
            # Run OCR on widget
            try:
                ocr_results = self.ocr_reader.readtext(widget_crop, detail=0)
                text = ' '.join(ocr_results).strip()
                widget.text = text
                
                if text:
                    print(f"  [{i+1}/{len(self.widgets)}] {widget.widget_type}: '{text}'")
            except Exception as e:
                print(f"  ⚠️  OCR failed for widget {i+1}: {e}")
        
        # Count widgets with text
        widgets_with_text = sum(1 for w in self.widgets if w.text)
        print(f"  ✓ Extracted text from {widgets_with_text}/{len(self.widgets)} widgets")
        
        return self.widgets
    
    def analyze_layout(self) -> Dict:
        """
        Step 3: Analyze layout structure
        - Group widgets by proximity
        - Detect rows/columns
        - Build hierarchy
        """
        print(f"\n📐 Step 3: Analyzing layout structure")
        
        if not self.widgets:
            return {}
        
        # Sort by vertical position (top to bottom)
        sorted_widgets = sorted(self.widgets, key=lambda w: w.center_y)
        
        # Detect rows (widgets with similar Y coordinates)
        rows = []
        current_row = [sorted_widgets[0]]
        row_threshold = 30  # pixels
        
        for widget in sorted_widgets[1:]:
            if abs(widget.center_y - current_row[0].center_y) < row_threshold:
                current_row.append(widget)
            else:
                rows.append(sorted(current_row, key=lambda w: w.center_x))
                current_row = [widget]
        
        if current_row:
            rows.append(sorted(current_row, key=lambda w: w.center_x))
        
        print(f"  ✓ Detected {len(rows)} rows")
        for i, row in enumerate(rows):
            widget_types = [w.widget_type for w in row]
            print(f"    Row {i+1}: {len(row)} widgets - {widget_types}")
        
        layout = {
            'total_widgets': len(self.widgets),
            'num_rows': len(rows),
            'rows': rows,
            'image_size': {
                'width': self.image.shape[1],
                'height': self.image.shape[0]
            }
        }
        
        return layout
    
    def save_analysis(self, output_path: str):
        """Save detected widgets and text to JSON"""
        data = {
            'source_image': str(self.image_path),
            'image_size': {
                'width': self.image.shape[1],
                'height': self.image.shape[0]
            },
            'widgets': [asdict(w) for w in self.widgets]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Analysis saved to: {output_path}")
    
    def recreate_gui_tkinter(self, scale_factor: float = 1.0):
        """
        Step 4: Recreate GUI in Tkinter based on detected widgets
        """
        print(f"\n🎨 Step 4: Recreating GUI in Tkinter")
        
        if not self.widgets:
            print("  ⚠️  No widgets to recreate")
            return
        
        # Create Tkinter window
        root = tk.Tk()
        root.title(f"Recreated: {Path(self.image_path).name}")
        
        # Get original dimensions
        orig_width = self.image.shape[1]
        orig_height = self.image.shape[0]
        
        # Scale dimensions
        canvas_width = int(orig_width * scale_factor)
        canvas_height = int(orig_height * scale_factor)
        
        # Create canvas
        canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
        canvas.pack()
        
        # Draw each widget
        widget_map = {
            'Button': self._create_button,
            'Label': self._create_label,
            'Entry': self._create_entry,
            'Table': self._create_table,
            'Plot': self._create_plot,
            'Video': self._create_video
        }
        
        created_count = 0
        for widget in self.widgets:
            x1, y1, x2, y2 = widget.bbox
            
            # Scale coordinates
            x1 = int(x1 * scale_factor)
            y1 = int(y1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y2 = int(y2 * scale_factor)
            
            # Create widget based on type
            creator_func = widget_map.get(widget.widget_type)
            if creator_func:
                creator_func(canvas, (x1, y1, x2, y2), widget.text)
                created_count += 1
        
        print(f"  ✓ Created {created_count} widgets in Tkinter")
        print(f"  📏 Scale factor: {scale_factor}")
        
        root.mainloop()
    
    # Widget creation helpers
    def _create_button(self, canvas, bbox, text):
        x1, y1, x2, y2 = bbox
        # Draw button rectangle
        canvas.create_rectangle(x1, y1, x2, y2, fill='#e0e0e0', outline='#888', width=2)
        # Add text
        if text:
            canvas.create_text((x1+x2)/2, (y1+y2)/2, text=text, font=('Arial', 10, 'bold'))
        else:
            canvas.create_text((x1+x2)/2, (y1+y2)/2, text='Button', font=('Arial', 10))
    
    def _create_label(self, canvas, bbox, text):
        x1, y1, x2, y2 = bbox
        # Draw label background
        canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='#ccc', width=1)
        # Add text
        if text:
            canvas.create_text(x1+5, (y1+y2)/2, text=text, anchor='w', font=('Arial', 9))
        else:
            canvas.create_text(x1+5, (y1+y2)/2, text='Label', anchor='w', font=('Arial', 9))
    
    def _create_entry(self, canvas, bbox, text):
        x1, y1, x2, y2 = bbox
        # Draw entry field
        canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='#666', width=2)
        # Add text if any
        if text:
            canvas.create_text(x1+5, (y1+y2)/2, text=text, anchor='w', font=('Arial', 9))
    
    def _create_table(self, canvas, bbox, text):
        x1, y1, x2, y2 = bbox
        # Draw table
        canvas.create_rectangle(x1, y1, x2, y2, fill='#f9f9f9', outline='#333', width=2)
        canvas.create_text((x1+x2)/2, (y1+y2)/2, text='[TABLE]', font=('Arial', 10, 'italic'))
    
    def _create_plot(self, canvas, bbox, text):
        x1, y1, x2, y2 = bbox
        # Draw plot area
        canvas.create_rectangle(x1, y1, x2, y2, fill='#f0f0ff', outline='#333', width=2)
        canvas.create_text((x1+x2)/2, (y1+y2)/2, text='[PLOT]', font=('Arial', 10, 'italic'))
    
    def _create_video(self, canvas, bbox, text):
        x1, y1, x2, y2 = bbox
        # Draw video area
        canvas.create_rectangle(x1, y1, x2, y2, fill='#000', outline='#333', width=2)
        canvas.create_text((x1+x2)/2, (y1+y2)/2, text='[VIDEO]', fill='white', font=('Arial', 10, 'italic'))
    
    def generate_html_recreation(self, output_path: str):
        """
        Alternative: Generate HTML/CSS recreation
        """
        print(f"\n🌐 Generating HTML recreation")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recreated GUI - {Path(self.image_path).name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .gui-container {{
            position: relative;
            width: {self.image.shape[1]}px;
            height: {self.image.shape[0]}px;
            border: 2px solid #ccc;
            background: white;
        }}
        .widget {{
            position: absolute;
            border: 1px solid #888;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .button {{
            background: #e0e0e0;
            border: 2px solid #888;
            font-weight: bold;
            cursor: pointer;
        }}
        .label {{
            background: white;
            border: 1px solid #ccc;
            text-align: left;
            padding-left: 5px;
        }}
        .entry {{
            background: white;
            border: 2px solid #666;
            text-align: left;
            padding-left: 5px;
        }}
        .table {{
            background: #f9f9f9;
            border: 2px solid #333;
            font-style: italic;
        }}
        .plot {{
            background: #f0f0ff;
            border: 2px solid #333;
            font-style: italic;
        }}
        .video {{
            background: #000;
            border: 2px solid #333;
            color: white;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <h1>Recreated GUI: {Path(self.image_path).name}</h1>
    <p>Detected {len(self.widgets)} widgets</p>
    <div class="gui-container">
"""
        
        # Add each widget
        for i, widget in enumerate(self.widgets):
            x1, y1, x2, y2 = widget.bbox
            width = x2 - x1
            height = y2 - y1
            
            widget_class = widget.widget_type.lower()
            text = widget.text if widget.text else widget.widget_type.upper()
            
            html += f"""        <div class="widget {widget_class}" 
             style="left: {x1}px; top: {y1}px; width: {width}px; height: {height}px;">
            {text}
        </div>
"""
        
        html += """    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  ✓ HTML saved to: {output_path}")


def main():
    """
    Full pipeline demo
    """
    print("=" * 70)
    print("GUI Recreation from Real Screenshots")
    print("=" * 70)
    
    # Example usage
    screenshot_path = "real_padded_screenshots/calculator.png"
    output_dir = Path("gui_recreations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize recreator
    recreator = GUIRecreator()
    
    # Run full pipeline
    recreator.detect_widgets(screenshot_path)
    recreator.extract_text()
    layout = recreator.analyze_layout()
    
    # Save analysis
    json_output = output_dir / f"{Path(screenshot_path).stem}_analysis.json"
    recreator.save_analysis(str(json_output))
    
    # Generate HTML recreation
    html_output = output_dir / f"{Path(screenshot_path).stem}_recreation.html"
    recreator.generate_html_recreation(str(html_output))
    
    # Show Tkinter recreation
    print("\n" + "=" * 70)
    print("Opening Tkinter recreation...")
    print("(Close the window to continue)")
    print("=" * 70)
    recreator.recreate_gui_tkinter(scale_factor=1.0)


if __name__ == "__main__":
    main()
