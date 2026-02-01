"""
Batch GUI Recreation - Process all real screenshots
"""

from recreate_gui_from_screenshot import GUIRecreator
from pathlib import Path
import json

# Configuration
REAL_SCREENSHOTS_DIR = "real_padded_screenshots"
OUTPUT_DIR = "gui_recreations"
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def batch_process_screenshots():
    """
    Process all screenshots in the directory
    """
    print("=" * 70)
    print("Batch GUI Recreation")
    print("=" * 70)
    
    # Get all screenshots
    screenshots_dir = Path(REAL_SCREENSHOTS_DIR)
    screenshots = []
    
    for ext in IMAGE_EXTENSIONS:
        screenshots.extend(screenshots_dir.glob(f'*{ext}'))
    
    screenshots = sorted(screenshots)
    
    if not screenshots:
        print(f"❌ No screenshots found in {REAL_SCREENSHOTS_DIR}")
        return
    
    print(f"\n📂 Found {len(screenshots)} screenshots to process")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize recreator once (reuse OCR and model)
    recreator = GUIRecreator()
    
    # Process each screenshot
    results = []
    
    for i, screenshot in enumerate(screenshots, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{i}/{len(screenshots)}]: {screenshot.name}")
        print('='*70)
        
        try:
            # Run pipeline
            widgets = recreator.detect_widgets(str(screenshot))
            recreator.extract_text()
            layout = recreator.analyze_layout()
            
            # Save outputs
            base_name = screenshot.stem
            
            # 1. JSON analysis
            json_output = output_dir / f"{base_name}_analysis.json"
            recreator.save_analysis(str(json_output))
            
            # 2. HTML recreation
            html_output = output_dir / f"{base_name}_recreation.html"
            recreator.generate_html_recreation(str(html_output))
            
            # Store results
            results.append({
                'screenshot': screenshot.name,
                'num_widgets': len(widgets),
                'widgets_with_text': sum(1 for w in widgets if w.text),
                'num_rows': layout.get('num_rows', 0),
                'json_output': str(json_output),
                'html_output': str(html_output)
            })
            
            print(f"✅ Completed: {screenshot.name}")
            
        except Exception as e:
            print(f"❌ Error processing {screenshot.name}: {e}")
            results.append({
                'screenshot': screenshot.name,
                'error': str(e)
            })
    
    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if 'error' not in r)
    print(f"\n✅ Successfully processed: {successful}/{len(screenshots)}")
    
    total_widgets = sum(r.get('num_widgets', 0) for r in results)
    total_widgets_with_text = sum(r.get('widgets_with_text', 0) for r in results)
    
    print(f"🎯 Total widgets detected: {total_widgets}")
    print(f"📝 Widgets with extracted text: {total_widgets_with_text}")
    
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
    print(f"📄 Summary saved to: {summary_file}")
    
    # Show individual results
    print(f"\n📋 Individual Results:")
    print("-" * 70)
    for r in results:
        if 'error' not in r:
            print(f"  ✓ {r['screenshot']:30s} → {r['num_widgets']:3d} widgets, {r['widgets_with_text']:3d} with text")
        else:
            print(f"  ✗ {r['screenshot']:30s} → ERROR")


def generate_overview_html():
    """
    Generate an overview HTML page showing all recreations
    """
    output_dir = Path(OUTPUT_DIR)
    html_files = sorted(output_dir.glob('*_recreation.html'))
    
    if not html_files:
        print("No HTML files found to create overview")
        return
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GUI Recreations Overview</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .card a {
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .card a:hover {
            background: #2980b9;
        }
    </style>
</head>
<body>
    <h1>🎨 GUI Recreations Overview</h1>
    <p>Total recreations: """ + str(len(html_files)) + """</p>
    <div class="grid">
"""
    
    for html_file in html_files:
        name = html_file.stem.replace('_recreation', '')
        json_file = html_file.parent / f"{name}_analysis.json"
        
        # Read widget count from JSON
        widget_count = "?"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                widget_count = len(data.get('widgets', []))
        
        html += f"""        <div class="card">
            <h3>{name}</h3>
            <p>Widgets: {widget_count}</p>
            <a href="{html_file.name}" target="_blank">View Recreation</a>
            <a href="{json_file.name}" target="_blank">View JSON</a>
        </div>
"""
    
    html += """    </div>
</body>
</html>
"""
    
    overview_file = output_dir / "index.html"
    with open(overview_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n🌐 Overview page created: {overview_file}")
    print(f"   Open in browser to see all recreations")


if __name__ == "__main__":
    batch_process_screenshots()
    generate_overview_html()
    
    print("\n" + "="*70)
    print("✅ Batch processing complete!")
    print("="*70)
