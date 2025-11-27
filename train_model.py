"""
Train YOLOv8 model on GUI widget detection dataset.
"""

from ultralytics import YOLO
import torch
from pathlib import Path

# Configuration
DATA_CONFIG = "dataset.yaml"
MODEL_SIZE = "yolov8n"  # Options: yolov8n (nano), yolov8s (small), yolov8m (medium)
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Output directory
PROJECT_NAME = "gui_widget_detection"
RUN_NAME = "yolov8_training"


def train_model():
    """
    Train YOLOv8 model on the GUI widget dataset.
    """
    print("=" * 60)
    print("GUI Widget Detection - YOLOv8 Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Dataset Config: {DATA_CONFIG}")
    print()
    
    # Load a pretrained YOLO model (recommended for transfer learning)
    model = YOLO(f"{MODEL_SIZE}.pt")
    
    # Train the model
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        patience=20,  # Early stopping patience
        save=True,
        plots=True,  # Save training plots
        val=True,  # Validate during training
        verbose=True,
        # Augmentation settings
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,    # Image HSV-Saturation augmentation
        hsv_v=0.4,    # Image HSV-Value augmentation
        degrees=10.0,  # Image rotation
        translate=0.1,  # Image translation
        scale=0.5,     # Image scale
        flipud=0.0,    # Image flip up-down
        fliplr=0.5,    # Image flip left-right
        mosaic=1.0,    # Mosaic augmentation
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"\nBest model saved to: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"Last model saved to: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
    print(f"Training results: {PROJECT_NAME}/{RUN_NAME}/")
    print("\nNext step: Run 'python test_model.py' to test the model!")
    
    return results


if __name__ == "__main__":
    # Check if dataset is prepared
    if not Path(DATA_CONFIG).exists():
        print("Error: dataset.yaml not found!")
        print("Please run 'python prepare_yolo_dataset.py' first.")
        exit(1)
    
    if not Path("yolo_dataset").exists():
        print("Error: yolo_dataset directory not found!")
        print("Please run 'python prepare_yolo_dataset.py' first.")
        exit(1)
    
    train_model()
