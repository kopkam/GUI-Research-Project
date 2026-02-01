"""
Advanced GUI Recreation - Code Generation
==========================================
Generuje kod Tkinter/PyQt/HTML który odtwarza wykryte GUI
"""

from recreate_gui_from_screenshot import GUIRecreator, Widget
from pathlib import Path
from typing import List


class CodeGenerator:
    """Generate executable code for recreating detected GUIs"""
    
    def __init__(self, widgets: List[Widget], image_size: tuple):
        self.widgets = widgets
        self.width, self.height = image_size
    
    def generate_tkinter_code(self) -> str:
        """
        Generate standalone Tkinter Python code
        """
        code = f'''"""
Auto-generated Tkinter GUI
Generated from screenshot analysis
Total widgets: {len(self.widgets)}
"""

import tkinter as tk
from tkinter import ttk


def create_gui():
    """Create the recreated GUI"""
    root = tk.Tk()
    root.title("Recreated GUI")
    root.geometry("{self.width}x{self.height}")
    root.configure(bg='white')
    
    # Create widgets
'''
        
        # Generate widget creation code
        for i, widget in enumerate(self.widgets):
            x1, y1, x2, y2 = widget.bbox
            width = x2 - x1
            height = y2 - y1
            
            widget_var = f"widget_{i}"
            text = widget.text if widget.text else widget.widget_type
            
            if widget.widget_type == 'Button':
                code += f'''    
    # Button {i}
    {widget_var} = tk.Button(root, text="{text}", 
                             bg='#e0e0e0', fg='black',
                             relief='raised', borderwidth=2)
    {widget_var}.place(x={x1}, y={y1}, width={width}, height={height})
'''
            
            elif widget.widget_type == 'Label':
                code += f'''    
    # Label {i}
    {widget_var} = tk.Label(root, text="{text}",
                           bg='white', fg='black',
                           anchor='w', relief='flat')
    {widget_var}.place(x={x1}, y={y1}, width={width}, height={height})
'''
            
            elif widget.widget_type == 'Entry':
                code += f'''    
    # Entry {i}
    {widget_var} = tk.Entry(root, bg='white', fg='black',
                           relief='sunken', borderwidth=2)
    {widget_var}.place(x={x1}, y={y1}, width={width}, height={height})
    {widget_var}.insert(0, "{text}")
'''
            
            elif widget.widget_type == 'Table':
                code += f'''    
    # Table {i} (using Text widget as placeholder)
    {widget_var} = tk.Text(root, bg='#f9f9f9', fg='black',
                          relief='solid', borderwidth=2)
    {widget_var}.place(x={x1}, y={y1}, width={width}, height={height})
    {widget_var}.insert('1.0', '[TABLE DATA]')
    {widget_var}.config(state='disabled')
'''
            
            elif widget.widget_type == 'Plot':
                code += f'''    
    # Plot {i} (Canvas placeholder)
    {widget_var} = tk.Canvas(root, bg='#f0f0ff',
                            relief='solid', borderwidth=2)
    {widget_var}.place(x={x1}, y={y1}, width={width}, height={height})
    {widget_var}.create_text({width//2}, {height//2}, 
                            text="[PLOT]", font=('Arial', 12, 'italic'))
'''
            
            elif widget.widget_type == 'Video':
                code += f'''    
    # Video {i} (Canvas placeholder)
    {widget_var} = tk.Canvas(root, bg='black',
                            relief='solid', borderwidth=2)
    {widget_var}.place(x={x1}, y={y1}, width={width}, height={height})
    {widget_var}.create_text({width//2}, {height//2},
                            text="[VIDEO]", fill='white',
                            font=('Arial', 12, 'italic'))
'''
        
        code += '''    
    root.mainloop()


if __name__ == "__main__":
    create_gui()
'''
        
        return code
    
    def generate_pyqt_code(self) -> str:
        """
        Generate standalone PyQt5 code
        """
        code = f'''"""
Auto-generated PyQt5 GUI
Generated from screenshot analysis
Total widgets: {len(self.widgets)}
"""

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QPushButton, QLabel, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt
import sys


class RecreatedGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Recreated GUI - PyQt5")
        self.setGeometry(100, 100, {self.width}, {self.height})
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: white;")
        
        # Create widgets
'''
        
        # Generate widget creation code
        for i, widget in enumerate(self.widgets):
            x1, y1, x2, y2 = widget.bbox
            width = x2 - x1
            height = y2 - y1
            
            widget_var = f"widget_{i}"
            text = widget.text if widget.text else widget.widget_type
            
            if widget.widget_type == 'Button':
                code += f'''        
        # Button {i}
        {widget_var} = QPushButton("{text}", central)
        {widget_var}.setGeometry({x1}, {y1}, {width}, {height})
        {widget_var}.setStyleSheet("background-color: #e0e0e0;")
'''
            
            elif widget.widget_type == 'Label':
                code += f'''        
        # Label {i}
        {widget_var} = QLabel("{text}", central)
        {widget_var}.setGeometry({x1}, {y1}, {width}, {height})
        {widget_var}.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
'''
            
            elif widget.widget_type == 'Entry':
                code += f'''        
        # Entry {i}
        {widget_var} = QLineEdit(central)
        {widget_var}.setGeometry({x1}, {y1}, {width}, {height})
        {widget_var}.setText("{text}")
'''
            
            elif widget.widget_type == 'Table':
                code += f'''        
        # Table {i}
        {widget_var} = QTextEdit(central)
        {widget_var}.setGeometry({x1}, {y1}, {width}, {height})
        {widget_var}.setPlainText("[TABLE DATA]")
        {widget_var}.setReadOnly(True)
        {widget_var}.setStyleSheet("background-color: #f9f9f9;")
'''
        
        code += '''

def main():
    app = QApplication(sys.argv)
    gui = RecreatedGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
'''
        
        return code
    
    def generate_html_code(self) -> str:
        """
        Generate standalone HTML/CSS/JS
        """
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recreated GUI</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .container {{
            position: relative;
            width: {self.width}px;
            height: {self.height}px;
            background: white;
            margin: 0 auto;
            border: 2px solid #ccc;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .widget {{
            position: absolute;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            cursor: default;
        }}
        
        .button {{
            background: #e0e0e0;
            border: 2px solid #888;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .button:hover {{
            background: #d0d0d0;
        }}
        
        .label {{
            background: white;
            border: 1px solid #ccc;
            text-align: left;
            padding-left: 8px;
        }}
        
        .entry {{
            background: white;
            border: 2px solid #666;
            border-radius: 3px;
            padding: 0 8px;
            text-align: left;
        }}
        
        .table {{
            background: #f9f9f9;
            border: 2px solid #333;
            font-style: italic;
            color: #666;
        }}
        
        .plot {{
            background: linear-gradient(135deg, #f0f0ff 0%, #e0e0ff 100%);
            border: 2px solid #333;
            font-style: italic;
            color: #666;
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
    <div class="container" id="gui-container">
'''
        
        # Generate widgets
        for i, widget in enumerate(self.widgets):
            x1, y1, x2, y2 = widget.bbox
            width = x2 - x1
            height = y2 - y1
            
            widget_class = widget.widget_type.lower()
            text = widget.text if widget.text else widget.widget_type.upper()
            
            html += f'''        <div class="widget {widget_class}" 
             id="widget-{i}"
             style="left: {x1}px; top: {y1}px; width: {width}px; height: {height}px;"
             data-type="{widget.widget_type}"
             data-confidence="{widget.confidence:.2f}">
            {text}
        </div>
'''
        
        html += '''    </div>
    
    <script>
        // Add interactivity
        document.querySelectorAll('.button').forEach(btn => {
            btn.addEventListener('click', function() {
                console.log('Button clicked:', this.textContent);
                alert('Button clicked: ' + this.textContent);
            });
        });
        
        // Show widget info on hover
        document.querySelectorAll('.widget').forEach(widget => {
            widget.addEventListener('mouseenter', function() {
                const type = this.dataset.type;
                const conf = this.dataset.confidence;
                this.title = `Type: ${type} | Confidence: ${conf}`;
            });
        });
    </script>
</body>
</html>
'''
        
        return html


def generate_all_code_versions(screenshot_path: str, output_dir: str = "gui_recreations/code"):
    """
    Generate all code versions (Tkinter, PyQt, HTML) for a screenshot
    """
    print("=" * 70)
    print("Code Generation from Screenshot")
    print("=" * 70)
    
    # Run detection and analysis
    recreator = GUIRecreator()
    recreator.detect_widgets(screenshot_path)
    recreator.extract_text()
    
    if not recreator.widgets:
        print("❌ No widgets detected")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(screenshot_path).stem
    
    # Initialize code generator
    image_size = (recreator.image.shape[1], recreator.image.shape[0])
    generator = CodeGenerator(recreator.widgets, image_size)
    
    # Generate Tkinter code
    print("\n🐍 Generating Tkinter code...")
    tkinter_code = generator.generate_tkinter_code()
    tkinter_file = output_path / f"{base_name}_tkinter.py"
    with open(tkinter_file, 'w', encoding='utf-8') as f:
        f.write(tkinter_code)
    print(f"  ✓ Saved: {tkinter_file}")
    
    # Generate PyQt code
    print("\n🎨 Generating PyQt5 code...")
    pyqt_code = generator.generate_pyqt_code()
    pyqt_file = output_path / f"{base_name}_pyqt.py"
    with open(pyqt_file, 'w', encoding='utf-8') as f:
        f.write(pyqt_code)
    print(f"  ✓ Saved: {pyqt_file}")
    
    # Generate HTML code
    print("\n🌐 Generating HTML code...")
    html_code = generator.generate_html_code()
    html_file = output_path / f"{base_name}_interactive.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_code)
    print(f"  ✓ Saved: {html_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Code Generation Complete!")
    print("=" * 70)
    print(f"\n📊 Generated code for {len(recreator.widgets)} widgets")
    print(f"\n📁 Output files:")
    print(f"   • Tkinter: {tkinter_file}")
    print(f"   • PyQt5:   {pyqt_file}")
    print(f"   • HTML:    {html_file}")
    
    print(f"\n💡 To run:")
    print(f"   python {tkinter_file}")
    print(f"   python {pyqt_file}  # requires: pip install PyQt5")
    print(f"   open {html_file}    # in browser")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        screenshot = sys.argv[1]
    else:
        screenshot = "real_padded_screenshots/calculator.png"
    
    generate_all_code_versions(screenshot)
