"""
Quick Demo - GUI Recreation
============================
Pokazuje możliwości recreacji GUI step by step
"""

from recreate_gui_from_screenshot import GUIRecreator
from pathlib import Path
import sys


def demo_single_screenshot(screenshot_path: str):
    """
    Demonstracja na pojedynczym screenshocie
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "GUI RECREATION DEMO" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    
    if not Path(screenshot_path).exists():
        print(f"\n❌ Screenshot not found: {screenshot_path}")
        print("\nAvailable screenshots in real_padded_screenshots/:")
        for img in sorted(Path("real_padded_screenshots").glob("*.png"))[:5]:
            print(f"  • {img.name}")
        return
    
    print(f"\n📸 Input: {screenshot_path}")
    print("─" * 70)
    
    # Create output directory
    output_dir = Path("gui_recreations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize recreator
    print("\n⏳ Initializing AI models...")
    recreator = GUIRecreator()
    
    # Step 1: Object Detection
    print("\n" + "─" * 70)
    input("Press ENTER to run Step 1: Object Detection...")
    widgets = recreator.detect_widgets(screenshot_path)
    
    if not widgets:
        print("❌ No widgets detected. Try lowering confidence threshold.")
        return
    
    # Step 2: Text Extraction
    print("\n" + "─" * 70)
    input("Press ENTER to run Step 2: Text Extraction (OCR)...")
    recreator.extract_text()
    
    # Step 3: Layout Analysis
    print("\n" + "─" * 70)
    input("Press ENTER to run Step 3: Layout Analysis...")
    layout = recreator.analyze_layout()
    
    # Step 4: Save outputs
    print("\n" + "─" * 70)
    print("💾 Saving outputs...")
    
    base_name = Path(screenshot_path).stem
    
    # JSON
    json_output = output_dir / f"{base_name}_analysis.json"
    recreator.save_analysis(str(json_output))
    
    # HTML
    html_output = output_dir / f"{base_name}_recreation.html"
    recreator.generate_html_recreation(str(html_output))
    
    # Summary
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 26 + "RESULTS" + " " * 35 + "║")
    print("╚" + "═" * 68 + "╝")
    
    widgets_with_text = sum(1 for w in widgets if w.text)
    
    print(f"\n✅ Detection:")
    print(f"   • Total widgets: {len(widgets)}")
    print(f"   • Widgets with text: {widgets_with_text} ({widgets_with_text/len(widgets)*100:.0f}%)")
    
    print(f"\n✅ Layout:")
    print(f"   • Rows detected: {layout['num_rows']}")
    print(f"   • Image size: {layout['image_size']['width']}x{layout['image_size']['height']}")
    
    print(f"\n✅ Outputs:")
    print(f"   • JSON: {json_output}")
    print(f"   • HTML: {html_output}")
    
    # Widget breakdown
    print(f"\n📊 Widget Breakdown:")
    widget_counts = {}
    for w in widgets:
        widget_counts[w.widget_type] = widget_counts.get(w.widget_type, 0) + 1
    
    for wtype, count in sorted(widget_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 2)
        print(f"   {wtype:8s}: {bar} {count}")
    
    # Examples of extracted text
    print(f"\n💬 Sample Extracted Text:")
    text_samples = [w for w in widgets if w.text][:5]
    if text_samples:
        for w in text_samples:
            print(f"   • [{w.widget_type}] \"{w.text}\"")
    else:
        print("   (No text extracted)")
    
    # Step 5: Tkinter Recreation
    print("\n" + "─" * 70)
    choice = input("\n🎨 Open Tkinter recreation? (y/n): ").lower()
    
    if choice == 'y':
        print("\n📱 Opening Tkinter window...")
        print("   (Close window to continue)")
        recreator.recreate_gui_tkinter(scale_factor=1.0)
    
    print("\n" + "═" * 70)
    print("✅ Demo complete!")
    print(f"\n💡 Next steps:")
    print(f"   1. Open {html_output} in browser")
    print(f"   2. Check {json_output} for full analysis")
    print(f"   3. Run batch_recreate_guis.py for all screenshots")
    print("═" * 70)


def interactive_menu():
    """
    Interactive menu dla wyboru screenshota
    """
    screenshots_dir = Path("real_padded_screenshots")
    screenshots = sorted(screenshots_dir.glob("*.png"))
    
    if not screenshots:
        print("❌ No screenshots found in real_padded_screenshots/")
        return
    
    print("\n📂 Available Screenshots:")
    print("─" * 70)
    
    for i, screenshot in enumerate(screenshots, 1):
        print(f"  {i:2d}. {screenshot.name}")
    
    print(f"  {len(screenshots)+1:2d}. Exit")
    
    print("─" * 70)
    
    while True:
        try:
            choice = input(f"\nSelect screenshot (1-{len(screenshots)+1}): ").strip()
            
            if not choice:
                continue
            
            choice_num = int(choice)
            
            if choice_num == len(screenshots) + 1:
                print("👋 Goodbye!")
                sys.exit(0)
            
            if 1 <= choice_num <= len(screenshots):
                selected = screenshots[choice_num - 1]
                demo_single_screenshot(str(selected))
                break
            else:
                print(f"❌ Please enter number between 1 and {len(screenshots)+1}")
        
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Screenshot provided as argument
        demo_single_screenshot(sys.argv[1])
    else:
        # Interactive menu
        interactive_menu()
