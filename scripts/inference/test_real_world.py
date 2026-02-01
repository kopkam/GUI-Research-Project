"""
Test trained YOLOv8 model on real-world GUI screenshots.
Evaluates model generalization beyond synthetic training data.
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# Configuration
MODEL_PATH = "gui_widget_detection/yolov8_training4/weights/best.pt"
REAL_IMAGES_DIR = "real_padded_screenshots"
OUTPUT_DIR = "real_world_results"
CONFIDENCE_THRESHOLD = 0.25

def test_on_real_world():
    """
    Test model on real-world GUI screenshots to evaluate generalization.
    """
    print("=" * 70)
    print("Testing Model on Real-World GUI Screenshots")
    print("=" * 70)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first!")
        return
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Get all real screenshots
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    real_images = []
    for ext in image_extensions:
        real_images.extend(Path(REAL_IMAGES_DIR).glob(ext))
    
    real_images = sorted(real_images)
    
    if not real_images:
        print(f"❌ No images found in {REAL_IMAGES_DIR}")
        return
    
    print(f"📂 Found {len(real_images)} real-world screenshots")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"\n🔍 Running inference with confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("-" * 70)
    
    # Statistics
    total_detections = 0
    detection_stats = {
        'Button': 0,
        'Label': 0,
        'Entry': 0,
        'Table': 0,
        'Plot': 0,
        'Video': 0
    }
    
    # Process each image
    for i, img_path in enumerate(real_images, 1):
        print(f"\n[{i}/{len(real_images)}] {img_path.name}")
        
        # Run prediction
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            save=True,
            project=OUTPUT_DIR,
            name="predictions",
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        # Analyze results
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                print("  ⚠️  No widgets detected")
                continue
            
            total_detections += len(boxes)
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
                
                # Update stats
                detection_stats[class_name] += 1
            
            # Print summary for this image
            for widget_type, confidences in sorted(detections.items()):
                avg_conf = sum(confidences) / len(confidences)
                print(f"    • {len(confidences)}x {widget_type} (avg conf: {avg_conf:.2f})")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("📊 REAL-WORLD TESTING RESULTS")
    print("=" * 70)
    print(f"\n📂 Images tested: {len(real_images)}")
    print(f"🎯 Total widgets detected: {total_detections}")
    print(f"📈 Average detections per image: {total_detections / len(real_images):.1f}")
    
    print(f"\n🔢 Per-class detections:")
    for widget_type, count in sorted(detection_stats.items(), key=lambda x: -x[1]):
        if count > 0:
            percentage = (count / total_detections) * 100
            print(f"   • {widget_type:8s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/predictions/")
    print("\n💡 Analysis:")
    print("   • Check if the model generalizes well to real GUIs")
    print("   • Compare detections with your expectations")
    print("   • Note any systematic errors or missed widgets")
    print("   • Consider if synthetic data covers real-world patterns")
    
    print("\n" + "=" * 70)
    
    return detection_stats


def generate_comparison_report():
    """
    Generate a side-by-side comparison showing model performance.
    """
    print("\n📋 Generating comparison report...")
    print("\nReal-world testing helps evaluate:")
    print("  1. Generalization beyond synthetic training data")
    print("  2. Robustness to different GUI frameworks (Qt, GTK, Web, etc.)")
    print("  3. Handling of varied styles, fonts, and layouts")
    print("  4. Performance on complex real-world interfaces")
    print("\n⚠️  Note: Lower performance on real-world data is expected")
    print("   since the model was trained exclusively on Tkinter GUIs.")


if __name__ == "__main__":
    stats = test_on_real_world()
    generate_comparison_report()
    
    print("\n✅ Testing complete! Review the annotated images in:")
    print(f"   {OUTPUT_DIR}/predictions/")
