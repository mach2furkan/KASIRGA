#!/usr/bin/env python3
"""Train YOLO11s on FLIR ADAS v2 RGB dataset."""

from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = YOLO('yolo11s.pt')

results = model.train(
    data='/home/bisavunma/yolo_project/flir_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    project='/home/bisavunma/yolo_project/runs/flir',
    name='yolo11s_flir',
    exist_ok=False,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    patience=20,
    save_period=10,
    plots=True,
    verbose=True,
)

print("\n=== Training Complete ===")
print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

# Validate
val_results = model.val(data='/home/bisavunma/yolo_project/flir_yolo/data.yaml', device=0)
print(f"\nValidation mAP50: {val_results.box.map50:.4f}")
print(f"Validation mAP50-95: {val_results.box.map:.4f}")
