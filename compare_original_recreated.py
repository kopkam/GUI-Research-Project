"""
Visual Comparison - Original vs Recreated GUI
==============================================
Side-by-side comparison z metrykami similarity
"""

from recreate_gui_from_screenshot import GUIRecreator
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json


def create_comparison_image(original_path: str, output_path: str):
    """
    Create side-by-side comparison: Original | Detected | Recreated
    """
    print(f"\n🖼️  Creating visual comparison for: {Path(original_path).name}")
    
    # Run detection
    recreator = GUIRecreator()
    recreator.detect_widgets(original_path)
    recreator.extract_text()
    
    # Load original image
    original = cv2.imread(original_path)
    height, width = original.shape[:2]
    
    # Create detected version (with bounding boxes)
    detected = original.copy()
    for widget in recreator.widgets:
        x1, y1, x2, y2 = widget.bbox
        
        # Color by type
        colors = {
            'Button': (0, 255, 0),     # Green
            'Label': (255, 165, 0),    # Orange
            'Entry': (0, 165, 255),    # Blue
            'Table': (255, 0, 255),    # Magenta
            'Plot': (255, 255, 0),     # Cyan
            'Video': (128, 0, 128)     # Purple
        }
        color = colors.get(widget.widget_type, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(detected, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{widget.widget_type} {widget.confidence:.2f}"
        cv2.putText(detected, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create recreated version (simplified rendering)
    recreated = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    for widget in recreator.widgets:
        x1, y1, x2, y2 = widget.bbox
        
        # Draw widget based on type
        if widget.widget_type == 'Button':
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (224, 224, 224), -1)
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (136, 136, 136), 2)
        elif widget.widget_type == 'Label':
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (204, 204, 204), 1)
        elif widget.widget_type == 'Entry':
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (102, 102, 102), 2)
        elif widget.widget_type == 'Table':
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (249, 249, 249), -1)
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (51, 51, 51), 2)
        elif widget.widget_type == 'Plot':
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (255, 240, 240), -1)
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (51, 51, 51), 2)
        elif widget.widget_type == 'Video':
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.rectangle(recreated, (x1, y1), (x2, y2), (51, 51, 51), 2)
        
        # Add text if available
        if widget.text:
            text_size = cv2.getTextSize(widget.text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x1 + 5
            text_y = y1 + (y2-y1)//2 + text_size[1]//2
            cv2.putText(recreated, widget.text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Create combined image (3 columns)
    padding = 20
    title_height = 50
    
    combined_width = width * 3 + padding * 4
    combined_height = height + title_height + padding * 2
    
    combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 240
    
    # Add titles
    titles = ['Original', 'Detected Widgets', 'Recreated']
    for i, title in enumerate(titles):
        x = padding + i * (width + padding) + width // 2
        cv2.putText(combined, title, (x - 60, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Place images
    y_start = title_height + padding
    combined[y_start:y_start+height, padding:padding+width] = original
    combined[y_start:y_start+height, padding*2+width:padding*2+width*2] = detected
    combined[y_start:y_start+height, padding*3+width*2:padding*3+width*3] = recreated
    
    # Save
    cv2.imwrite(output_path, combined)
    print(f"  ✓ Saved comparison: {output_path}")
    
    return recreator.widgets


def calculate_metrics(widgets, image_size):
    """
    Calculate metrics about the recreation
    """
    metrics = {
        'total_widgets': len(widgets),
        'widgets_with_text': sum(1 for w in widgets if w.text),
        'text_coverage': sum(1 for w in widgets if w.text) / len(widgets) if widgets else 0,
        'average_confidence': sum(w.confidence for w in widgets) / len(widgets) if widgets else 0,
        'widget_types': {},
        'average_widget_size': 0,
        'coverage_area': 0
    }
    
    # Widget type distribution
    for w in widgets:
        metrics['widget_types'][w.widget_type] = metrics['widget_types'].get(w.widget_type, 0) + 1
    
    # Size and coverage
    total_area = 0
    for w in widgets:
        total_area += w.width * w.height
    
    image_area = image_size[0] * image_size[1]
    metrics['coverage_area'] = total_area / image_area if image_area > 0 else 0
    metrics['average_widget_size'] = total_area / len(widgets) if widgets else 0
    
    return metrics


def generate_comparison_report(screenshot_dir: str = "real_padded_screenshots",
                               output_dir: str = "gui_recreations/comparisons"):
    """
    Generate comparison report for all screenshots
    """
    print("=" * 70)
    print("Visual Comparison Report Generator")
    print("=" * 70)
    
    screenshots = sorted(Path(screenshot_dir).glob("*.png"))
    
    if not screenshots:
        print(f"❌ No screenshots found in {screenshot_dir}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Found {len(screenshots)} screenshots")
    print("─" * 70)
    
    all_metrics = []
    
    for i, screenshot in enumerate(screenshots, 1):
        print(f"\n[{i}/{len(screenshots)}] Processing: {screenshot.name}")
        
        # Create comparison image
        comparison_output = output_path / f"{screenshot.stem}_comparison.png"
        widgets = create_comparison_image(str(screenshot), str(comparison_output))
        
        # Calculate metrics
        image = cv2.imread(str(screenshot))
        image_size = (image.shape[1], image.shape[0])
        metrics = calculate_metrics(widgets, image_size)
        metrics['screenshot'] = screenshot.name
        
        all_metrics.append(metrics)
        
        # Print summary
        print(f"  • Widgets: {metrics['total_widgets']}")
        print(f"  • With text: {metrics['widgets_with_text']} ({metrics['text_coverage']*100:.1f}%)")
        print(f"  • Avg confidence: {metrics['average_confidence']:.2f}")
        print(f"  • Coverage: {metrics['coverage_area']*100:.1f}%")
    
    # Save metrics
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Generate HTML report
    generate_html_report(all_metrics, output_path)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    
    avg_widgets = sum(m['total_widgets'] for m in all_metrics) / len(all_metrics)
    avg_text_coverage = sum(m['text_coverage'] for m in all_metrics) / len(all_metrics)
    avg_confidence = sum(m['average_confidence'] for m in all_metrics) / len(all_metrics)
    
    print(f"\n✅ Processed: {len(screenshots)} screenshots")
    print(f"📈 Average widgets per screenshot: {avg_widgets:.1f}")
    print(f"📝 Average text extraction rate: {avg_text_coverage*100:.1f}%")
    print(f"🎯 Average detection confidence: {avg_confidence:.2f}")
    
    print(f"\n📁 Outputs saved to: {output_path}/")
    print(f"   • Comparison images: *_comparison.png")
    print(f"   • Metrics: metrics.json")
    print(f"   • Report: comparison_report.html")


def generate_html_report(metrics_list, output_dir):
    """
    Generate interactive HTML report
    """
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GUI Recreation Comparison Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .card-content {
            padding: 15px;
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: bold;
            color: #555;
        }
        .metric-value {
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #3498db;
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
    </style>
</head>
<body>
    <h1>🎨 GUI Recreation Comparison Report</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
"""
    
    # Calculate summary stats
    total_screenshots = len(metrics_list)
    total_widgets = sum(m['total_widgets'] for m in metrics_list)
    avg_text_coverage = sum(m['text_coverage'] for m in metrics_list) / len(metrics_list) * 100
    avg_confidence = sum(m['average_confidence'] for m in metrics_list) / len(metrics_list)
    
    html += f"""
        <div class="metric">
            <span class="metric-label">Total Screenshots:</span>
            <span class="metric-value">{total_screenshots}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Widgets Detected:</span>
            <span class="metric-value">{total_widgets}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Text Extraction Rate:</span>
            <span class="metric-value">{avg_text_coverage:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Detection Confidence:</span>
            <span class="metric-value">{avg_confidence:.2f}</span>
        </div>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Screenshot</th>
            <th>Widgets</th>
            <th>With Text</th>
            <th>Text Coverage</th>
            <th>Avg Confidence</th>
            <th>Coverage Area</th>
        </tr>
"""
    
    for m in metrics_list:
        html += f"""        <tr>
            <td>{m['screenshot']}</td>
            <td>{m['total_widgets']}</td>
            <td>{m['widgets_with_text']}</td>
            <td>{m['text_coverage']*100:.1f}%</td>
            <td>{m['average_confidence']:.2f}</td>
            <td>{m['coverage_area']*100:.1f}%</td>
        </tr>
"""
    
    html += """    </table>
    
    <h2>Visual Comparisons</h2>
    <div class="grid">
"""
    
    for m in metrics_list:
        screenshot_name = m['screenshot']
        base_name = Path(screenshot_name).stem
        comparison_img = f"{base_name}_comparison.png"
        
        html += f"""        <div class="card">
            <img src="{comparison_img}" alt="{screenshot_name}">
            <div class="card-content">
                <h3>{screenshot_name}</h3>
                <div class="metric">
                    <span class="metric-label">Widgets:</span>
                    <span class="metric-value">{m['total_widgets']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Text Extraction:</span>
                    <span class="metric-value">{m['text_coverage']*100:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence:</span>
                    <span class="metric-value">{m['average_confidence']:.2f}</span>
                </div>
            </div>
        </div>
"""
    
    html += """    </div>
</body>
</html>
"""
    
    report_file = output_dir / "comparison_report.html"
    with open(report_file, 'w') as f:
        f.write(html)
    
    print(f"  ✓ HTML report generated: {report_file}")


if __name__ == "__main__":
    generate_comparison_report()
