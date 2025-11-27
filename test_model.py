"""
Test the trained YOLOv8 model on GUI screenshots.
Performs inference and visualizes detected widgets.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import os

# Configuration  
MODEL_PATH = "gui_widget_detection/yolov8_training2/weights/best.pt"  # Path to trained model (LATEST - no text prefixes)
TEST_IMAGES_DIR = "screenshots"  # Directory with test images
OUTPUT_DIR = "test_results"  # Where to save results
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections


def test_model(image_path=None, save_results=True, show=False):
    """
    Test the trained model on images.
    
    Args:
        image_path: Path to a single image (if None, tests on all images in TEST_IMAGES_DIR)
        save_results: Whether to save annotated images
        show: Whether to display results
    """
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using 'python train_model.py'")
        return
    
    # Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Create output directory
    if save_results:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Determine which images to test
    if image_path:
        test_images = [image_path]
    else:
        # Test on all images in the test directory
        test_images = sorted(Path(TEST_IMAGES_DIR).glob("*.png"))[:10]  # Test on first 10
        print(f"Testing on {len(test_images)} images from {TEST_IMAGES_DIR}")
    
    # Run inference
    for img_path in test_images:
        print(f"\nProcessing: {img_path}")
        
        # Perform prediction
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            save=save_results,
            project=OUTPUT_DIR,
            name="predictions",
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        # Print detection results
        for result in results:
            boxes = result.boxes
            print(f"  Detected {len(boxes)} widgets:")
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                print(f"    - {class_name}: {confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)
    if save_results:
        print(f"Results saved to: {OUTPUT_DIR}/predictions/")
    
    return results


def test_single_image(image_path):
    """
    Test model on a single image and return detailed results.
    """
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return None
    
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        save=True,
        project=OUTPUT_DIR,
        name="single_test",
        exist_ok=True
    )
    
    return results


def evaluate_on_validation():
    """
    Evaluate model performance on the validation set.
    """
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    print("Running validation on test set...")
    model = YOLO(MODEL_PATH)
    
    # Validate the model
    metrics = model.val(
        data="dataset.yaml",
        split="val",
        batch=16,
        imgsz=640,
        plots=True,
        save_json=True,
        project=OUTPUT_DIR,
        name="validation"
    )
    
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on specific image
        img_path = sys.argv[1]
        print(f"Testing on single image: {img_path}")
        test_single_image(img_path)
    else:
        # Run full evaluation
        print("Running evaluation on validation set...")
        evaluate_on_validation()
        
        print("\n" + "=" * 60)
        print("Testing on sample images...")
        print("=" * 60)
        test_model()
