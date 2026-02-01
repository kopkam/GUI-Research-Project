"""
Get detailed per-class metrics for Training 4 model.
"""

from ultralytics import YOLO

model = YOLO('gui_widget_detection/yolov8_training4/weights/best.pt')

print('Running validation to extract per-class metrics...\n')
m = model.val(data='dataset.yaml', verbose=False, plots=False)

print('=' * 80)
print('PER-CLASS PERFORMANCE METRICS - TRAINING 4')
print('=' * 80)
print()

# Header
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
print('-' * 80)

# Class names
names = ['Button', 'Label', 'Entry', 'Table', 'Plot', 'Video']

# Note: YOLOv8 provides overall precision/recall, not per-class
# We can only accurately show per-class AP values
# For precision/recall, we'd need to parse the confusion matrix

# Show per-class AP values with overall P/R as reference
for i, name in enumerate(names):
    ap50 = m.box.ap50[i]
    ap = m.box.ap[i]
    
    # Overall metrics (same for all classes as YOLO doesn't provide per-class P/R directly)
    precision = m.box.mp
    recall = m.box.mr
    
    print(f"{name:<12} {precision:>10.2%}   {recall:>10.2%}   {ap50:>10.2%}   {ap:>10.2%}")

print('-' * 80)
print(f"{'AVERAGE':<12} {m.box.mp:>10.2%}   {m.box.mr:>10.2%}   {m.box.map50:>10.2%}   {m.box.map:>10.2%}")
print('=' * 80)

print()
print('📝 Note: Precision and Recall shown are overall metrics.')
print('   YOLOv8 calculates these across all classes combined.')
print('   mAP@50 and mAP@50-95 are accurately computed per-class.')
print()

# Alternative: show just mAP per class clearly
print()
print('=' * 80)
print('SIMPLIFIED PER-CLASS TABLE (mAP only)')
print('=' * 80)
print()
print(f"{'Class':<12} {'mAP@50':<15} {'mAP@50-95':<15} {'Status'}")
print('-' * 80)

for i, name in enumerate(names):
    ap50 = m.box.ap50[i]
    ap = m.box.ap[i]
    
    # Status indicator
    if ap50 >= 0.99:
        status = '🔥 Perfect'
    elif ap50 >= 0.95:
        status = '✅ Excellent'
    elif ap50 >= 0.90:
        status = '✓ Good'
    else:
        status = '⚠️ Fair'
    
    print(f"{name:<12} {ap50:>10.2%}      {ap:>10.2%}      {status}")

print('-' * 80)
print(f"{'AVERAGE':<12} {m.box.map50:>10.2%}      {m.box.map:>10.2%}      Overall")
print('=' * 80)

print()
print('✅ Overall Metrics:')
print(f'   Precision: {m.box.mp:.2%}')
print(f'   Recall:    {m.box.mr:.2%}')
print(f'   mAP@50:    {m.box.map50:.2%}')
print(f'   mAP@50-95: {m.box.map:.2%}')
