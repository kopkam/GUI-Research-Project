"""
Generate presentation-ready visualizations of model predictions.
Creates annotated screenshots showing detected widgets.
"""

from ultralytics import YOLO
from pathlib import Path
import random

# Configuration
MODEL_PATH = "gui_widget_detection/yolov8_training4/weights/best.pt"
SCREENSHOTS_DIR = "screenshots"
OUTPUT_DIR = "presentation_results"
NUM_EXAMPLES = 12  # Number of examples for presentation
CONFIDENCE_THRESHOLD = 0.25

def generate_presentation_examples():
    """
    Generate high-quality prediction visualizations for presentation.
    """
    print("=" * 70)
    print("Generating Presentation Examples - GUI Widget Detection")
    print("=" * 70)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first!")
        return
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Get all screenshots
    all_screenshots = sorted(list(Path(SCREENSHOTS_DIR).glob("*.png")))
    print(f"📂 Found {len(all_screenshots)} screenshots")
    
    # Randomly select diverse examples
    random.seed(42)
    if len(all_screenshots) > NUM_EXAMPLES:
        selected = random.sample(all_screenshots, NUM_EXAMPLES)
    else:
        selected = all_screenshots[:NUM_EXAMPLES]
    
    print(f"🎯 Selected {len(selected)} examples for presentation")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"\n🔍 Running inference...")
    print("-" * 70)
    
    # Process each image
    for i, img_path in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {img_path.name}")
        
        # Run prediction with visualization
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            save=True,
            project=OUTPUT_DIR,
            name="examples",
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=3  # Thicker lines for presentation
        )
        
        # Print what was detected
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                print("  ⚠️  No widgets detected")
                continue
            
            print(f"  ✓ Detected {len(boxes)} widgets:")
            
            # Group by class
            detections = {}
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name not in detections:
                    detections[class_name] = []
                detections[class_name].append(confidence)
            
            # Print summary
            for widget_type, confidences in sorted(detections.items()):
                avg_conf = sum(confidences) / len(confidences)
                print(f"    • {len(confidences)}x {widget_type} (avg conf: {avg_conf:.2f})")
    
    print("\n" + "=" * 70)
    print("✅ Presentation Examples Generated Successfully!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/examples/")
    print(f"📊 Total images: {len(selected)}")
    print("\n💡 Use these images in your presentation to show:")
    print("   • Model accuracy on diverse GUI layouts")
    print("   • Bounding box precision")
    print("   • Confidence scores for different widget types")
    print("   • Performance on new widget classes (Plot, Video)")
    print("\n" + "=" * 70)


def generate_comparison_grid():
    """
    Generate a grid comparing input vs output (optional enhancement).
    """
    print("\n📸 Tip: For even better presentation visuals, consider creating:")
    print("   • Side-by-side comparisons (original | detected)")
    print("   • Confusion matrix overlay")
    print("   • Per-class detection examples")


if __name__ == "__main__":
    generate_presentation_examples()
    generate_comparison_grid()
