"""
Visualize model predictions with bounding boxes and labels.
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

MODEL_PATH = "gui_widget_detection/yolov8_training/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.3


def visualize_predictions(image_path, save_path=None):
    """
    Visualize model predictions on an image using matplotlib.
    """
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Read image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Show original image
    ax1.imshow(img_rgb)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Show image with predictions
    ax2.imshow(img_rgb)
    ax2.set_title("Model Predictions", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Define colors for each class
    colors = {
        0: (255, 0, 0),     # Button - Red
        1: (0, 255, 0),     # Label - Green
        2: (0, 0, 255)      # Entry - Blue
    }
    
    # Draw predictions
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Get color for this class
            color = colors.get(class_id, (255, 255, 255))
            color_normalized = tuple(c/255.0 for c in color)
            
            # Draw bounding box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False,
                edgecolor=color_normalized,
                linewidth=2
            )
            ax2.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            ax2.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color_normalized, alpha=0.7)
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_summary():
    """
    Create a summary visualization showing multiple test images.
    """
    model = YOLO(MODEL_PATH)
    test_images = sorted(Path("screenshots").glob("screenshot_[0-9].png"))[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {
        0: (1.0, 0, 0),     # Button - Red
        1: (0, 1.0, 0),     # Label - Green
        2: (0, 0, 1.0)      # Entry - Blue
    }
    
    for idx, img_path in enumerate(test_images):
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model.predict(source=img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Show image
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{img_path.name}", fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        # Draw predictions
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                color = colors.get(class_id, (1, 1, 1))
                
                # Draw bounding box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2
                )
                axes[idx].add_patch(rect)
                
                # Add label
                label = f"{class_name[:3]}: {confidence:.2f}"
                axes[idx].text(
                    x1, y1 - 3,
                    label,
                    color='white',
                    fontsize=8,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
                )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Button'),
        Patch(facecolor='green', edgecolor='green', label='Label'),
        Patch(facecolor='blue', edgecolor='blue', label='Entry')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = "results_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results summary saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Visualize specific image
        img_path = sys.argv[1]
        output_path = "visualization.png"
        print(f"Creating visualization for: {img_path}")
        visualize_predictions(img_path, save_path=output_path)
    else:
        # Create summary of multiple images
        print("Creating results summary...")
        create_results_summary()
