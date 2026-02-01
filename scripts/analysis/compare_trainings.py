"""
Compare metrics across 3 training iterations with different number of classes.
"""

from ultralytics import YOLO
import os

print("=" * 80)
print("TRAINING COMPARISON - Evolution Across 3 Iterations")
print("=" * 80)
print()

# Define the 3 trainings to compare
trainings = [
    {
        'name': 'Training 1',
        'path': 'gui_widget_detection/yolov8_training/weights/best.pt',
        'classes': 3,
        'class_names': ['Button', 'Label', 'Entry'],
        'dataset_size': 100,
        'description': 'Baseline (3 classes)'
    },
    {
        'name': 'Training 3',
        'path': 'gui_widget_detection/yolov8_training3/weights/best.pt',
        'classes': 4,
        'class_names': ['Button', 'Label', 'Entry', 'Table'],
        'dataset_size': 200,
        'description': 'Added Table class'
    },
    {
        'name': 'Training 4',
        'path': 'gui_widget_detection/yolov8_training4/weights/best.pt',
        'classes': 6,
        'class_names': ['Button', 'Label', 'Entry', 'Table', 'Plot', 'Video'],
        'dataset_size': 200,
        'description': 'Added Plot & Video'
    }
]

# Collect results
results = []

for training in trainings:
    if not os.path.exists(training['path']):
        print(f"⚠️  {training['name']}: Model not found at {training['path']}")
        continue
    
    print(f"📊 Evaluating {training['name']} ({training['description']})...")
    model = YOLO(training['path'])
    
    # Note: We need the correct dataset.yaml for each training
    # For simplicity, we'll use the current one and note limitations
    
    # Get model info
    try:
        # Run validation - this might fail for old models with different class counts
        metrics = model.val(data='dataset.yaml', verbose=False, plots=False)
        
        results.append({
            'name': training['name'],
            'classes': training['classes'],
            'class_names': training['class_names'],
            'dataset_size': training['dataset_size'],
            'description': training['description'],
            'mAP50': metrics.box.map50,
            'mAP50_95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'valid': True
        })
    except Exception as e:
        print(f"   ⚠️  Validation failed: {str(e)[:50]}...")
        # Try to get basic info from model
        results.append({
            'name': training['name'],
            'classes': training['classes'],
            'class_names': training['class_names'],
            'dataset_size': training['dataset_size'],
            'description': training['description'],
            'mAP50': None,
            'mAP50_95': None,
            'precision': None,
            'recall': None,
            'valid': False
        })

print()
print("=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
print()

# Header
print(f"{'Training':<15} {'Classes':<10} {'Dataset':<10} {'mAP@50':<12} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 80)

# Results
for r in results:
    if r['valid']:
        print(f"{r['name']:<15} {r['classes']:<10} {r['dataset_size']:<10} "
              f"{r['mAP50']*100:>9.2f}%  {r['mAP50_95']*100:>9.2f}%  "
              f"{r['precision']*100:>9.2f}%  {r['recall']*100:>9.2f}%")
    else:
        print(f"{r['name']:<15} {r['classes']:<10} {r['dataset_size']:<10} "
              f"{'N/A':>11}  {'N/A':>11}  {'N/A':>11}  {'N/A':>11}")

print("-" * 80)

print()
print("📝 Detailed Breakdown:")
print()

for r in results:
    print(f"🔹 {r['name']} - {r['description']}")
    print(f"   Classes: {r['classes']} ({', '.join(r['class_names'])})")
    print(f"   Dataset size: {r['dataset_size']} samples")
    if r['valid']:
        print(f"   Performance: mAP@50={r['mAP50']*100:.2f}%, Precision={r['precision']*100:.2f}%, Recall={r['recall']*100:.2f}%")
    else:
        print(f"   Performance: Unable to validate with current dataset")
    print()

print()
print("⚠️  Note: Old models (Training 1 & 3) may show N/A if they can't be validated")
print("   with the current 6-class dataset.yaml. For accurate comparison, we'd need")
print("   to use the original dataset configuration for each training.")
print()
print("=" * 80)
