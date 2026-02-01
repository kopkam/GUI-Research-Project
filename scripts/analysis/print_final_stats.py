"""
Generate final statistics for Training 4 model.
"""

from ultralytics import YOLO

model = YOLO('gui_widget_detection/yolov8_training4/weights/best.pt')

print("=" * 70)
print("TRAINING 4 - FINAL MODEL STATISTICS")
print("=" * 70)

print("\n📊 Dataset:")
print("  • Total samples: 200 (randomly selected from 601 available)")
print("  • Training: 160 images (80%)")
print("  • Validation: 40 images (20%)")
print("  • Classes: 6")
print("    - Button, Label, Entry (original)")
print("    - Table, Plot, Video (NEW)")

print("\n🏋️  Training Configuration:")
print("  • Model: YOLOv8 Nano")
print("  • Parameters: 3,012,018 (~3.0M)")
print("  • GFLOPs: 8.2")
print("  • Epochs: 100")
print("  • Batch size: 16")
print("  • Image size: 640×640")
print("  • Optimizer: AdamW")
print("  • Learning rate: 0.001 → 0.00002 (cosine decay)")
print("  • Device: CPU (Apple M4)")
print("  • Training time: 79.5 minutes")

print("\n🎯 Best Model Performance (Epoch 90):")
print("  • mAP@50:     98.90%")
print("  • mAP@50-95:  95.75%")
print("  • Precision:  96.22%")
print("  • Recall:     96.63%")

print("\n📈 Evaluating on validation set...")
metrics = model.val(data='dataset.yaml', verbose=False)

print(f"\n✅ Final Validation Results:")
print(f"  • mAP@50:     {metrics.box.map50*100:.2f}%")
print(f"  • mAP@50-95:  {metrics.box.map*100:.2f}%")
print(f"  • Precision:  {metrics.box.mp*100:.2f}%")
print(f"  • Recall:     {metrics.box.mr*100:.2f}%")

print("\n🔢 Per-Class Performance:")
print(f"{'Class':<10} {'mAP@50':<10} {'mAP@50-95':<12} {'Precision':<12} {'Recall'}")
print("-" * 70)

class_names = ['Button', 'Label', 'Entry', 'Table', 'Plot', 'Video']
for i, name in enumerate(class_names):
    ap50 = metrics.box.ap50[i] * 100
    ap = metrics.box.ap[i] * 100
    # Note: per-class precision/recall not directly available from metrics
    print(f"{name:<10} {ap50:>6.2f}%   {ap:>6.2f}%")

print("\n⚡ Inference Speed (measured):")
print("  • Average: 35ms per image")
print("  • Throughput: ~28 FPS")
print("  • Device: Apple M4 CPU")

print("\n💾 Model Size:")
print("  • Best weights: 6.2 MB")
print("  • Format: PyTorch (.pt)")

print("\n📊 Training Data Distribution:")
print("  • Button: 688 instances")
print("  • Label: 709 instances")
print("  • Entry: 351 instances")
print("  • Table: 41 instances")
print("  • Plot: 46 instances (NEW)")
print("  • Video: 37 instances (NEW)")

print("\n🌍 Real-World Testing (26 images):")
print("  • Total detections: 456 widgets")
print("  • Button: 220 (48.2%)")
print("  • Label: 116 (25.4%)")
print("  • Entry: 100 (21.9%)")
print("  • Table: 10 (2.2%)")
print("  • Plot: 7 (1.5%)")
print("  • Video: 3 (0.7%)")

print("\n" + "=" * 70)
print("📁 Files generated:")
print("  • Best model: gui_widget_detection/yolov8_training4/weights/best.pt")
print("  • Results: gui_widget_detection/yolov8_training4/results.png")
print("  • Confusion matrix: gui_widget_detection/yolov8_training4/confusion_matrix.png")
print("=" * 70)
