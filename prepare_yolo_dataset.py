"""
Convert GUI annotations from JSON format to YOLO format for training.
Creates train/val split and organizes data for YOLOv8.
"""

import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Set random seed for reproducibility
random.seed(42)

# Configuration
ANNOTATIONS_DIR = "annotations"
IMAGES_DIR = "screenshots"
OUTPUT_DIR = "yolo_dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# Widget types to class indices
CLASS_MAPPING = {
    "Button": 0,
    "Label": 1,
    "Entry": 2
}

CLASS_NAMES = ["Button", "Label", "Entry"]


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) to YOLO format:
    (x_center, y_center, width, height) - all normalized to [0, 1]
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to [0, 1]
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def prepare_yolo_dataset():
    """
    Main function to convert dataset to YOLO format.
    """
    # Create output directories
    output_path = Path(OUTPUT_DIR)
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Get all annotation files
    annotation_files = sorted([f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')])
    print(f"Found {len(annotation_files)} annotation files")
    
    # Split into train and validation
    train_files, val_files = train_test_split(
        annotation_files, 
        train_size=TRAIN_SPLIT, 
        random_state=42
    )
    
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    
    # Process each split
    for split_name, files in [("train", train_files), ("val", val_files)]:
        print(f"\nProcessing {split_name} split...")
        
        for ann_file in files:
            # Load annotation
            with open(os.path.join(ANNOTATIONS_DIR, ann_file), 'r') as f:
                data = json.load(f)
            
            # Get image info
            img_path = data["image"]
            img_width = data["resolution"]["width"]
            img_height = data["resolution"]["height"]
            
            # Copy image to output directory
            src_img = os.path.join(img_path)
            base_name = Path(img_path).stem  # e.g., "screenshot_0"
            dst_img = output_path / split_name / "images" / f"{base_name}.png"
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Image not found: {src_img}")
                continue
            
            # Create YOLO label file
            label_file = output_path / split_name / "labels" / f"{base_name}.txt"
            
            with open(label_file, 'w') as f:
                for widget in data["widgets"]:
                    widget_type = widget["type"]
                    
                    if widget_type not in CLASS_MAPPING:
                        print(f"Warning: Unknown widget type: {widget_type}")
                        continue
                    
                    class_id = CLASS_MAPPING[widget_type]
                    
                    # Convert bbox to YOLO format
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        widget["bbox"], img_width, img_height
                    )
                    
                    # Write to label file: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"\n✓ Dataset prepared successfully in '{OUTPUT_DIR}' directory")
    print(f"✓ Classes: {CLASS_NAMES}")
    return output_path


if __name__ == "__main__":
    prepare_yolo_dataset()
    print("\nNext step: Run 'python train_model.py' to start training!")
