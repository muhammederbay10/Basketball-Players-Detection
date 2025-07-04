from ultralytics import YOLO
from callbacks import CustomLRScheduler

# Create callback instance
lr_scheduler = CustomLRScheduler(
    monitor='metrics/mAP50(B)',  
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# The Model
model = YOLO("yolov8s.pt")

# Register Callbacks
model.add_callback("on_train_start", lr_scheduler.on_train_start)
model.add_callback("on_fit_epoch_end", lr_scheduler.on_fit_epoch_end)

# Train
model.train(
    data="data/data.yaml",
    epochs=100,
    patience=15,          # Early stopping
    batch=16,
    imgsz=640,
    lr0=0.01,             # Initial learning rate
    weight_decay=0.0005,  # L2 regularization
    hsv_h=0.015,          # Hue augmentation
    hsv_s=0.7,            # Saturation augmentation
    hsv_v=0.4,            # Value augmentation
    degrees=10.0,         # Rotation augmentation
    translate=0.1,        # Translation augmentation
    scale=0.5,            # Scale augmentation
    fliplr=0.5,           # Horizontal flip
    mosaic=1.0,           # Mosaic augmentation
    copy_paste=0.5,       # Copy-paste augmentation
    close_mosaic=10,      # Disable mosaic after 10 epochs
    cos_lr=True,          # Cosine learning rate schedule
    erasing=0.4,          # Random erasing
    dropout=0.2           # Dropout (for larger models)
)