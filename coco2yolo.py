#!/usr/bin/env python3
"""Convert FLIR ADAS v2 COCO annotations to YOLO format."""

import json, os, random, shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

COCO_JSON = '/home/bisavunma/yolo_project/flir_dataset/annotations/images_rgb_train_coco.json'
IMAGES_DIR = Path('/home/bisavunma/yolo_project/flir_dataset/images/images_rgb_train')
OUT_DIR = Path('/home/bisavunma/yolo_project/flir_yolo')

# Key ADAS categories to keep (COCO category_id -> YOLO class_id)
KEEP_CATS = {
    1: 0,   # person
    2: 1,   # bike
    3: 2,   # car
    4: 3,   # motor
    6: 4,   # bus
    8: 5,   # truck
}
CLASS_NAMES = {0: 'person', 1: 'bike', 2: 'car', 3: 'motor', 4: 'bus', 5: 'truck'}

TRAIN_RATIO = 0.85
SEED = 42

def main():
    random.seed(SEED)

    print("Loading COCO JSON...")
    with open(COCO_JSON) as f:
        coco = json.load(f)

    # Index annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] in KEEP_CATS:
            ann_by_img[ann['image_id']].append(ann)

    # Index images by filename
    img_by_id = {img['id']: img for img in coco['images']}

    # Find images we actually have on disk
    available_files = {p.name for p in IMAGES_DIR.glob('*.jpg')}
    print(f"Images on disk: {len(available_files)}")

    # Filter to only images we have + have at least 1 annotation
    usable = []
    for img_id, anns in ann_by_img.items():
        img_info = img_by_id.get(img_id)
        if img_info is None:
            continue
        fname = Path(img_info['file_name']).name
        if fname in available_files:
            usable.append((img_info, anns))

    print(f"Usable images (have file + annotations): {len(usable)}")

    # Shuffle and split
    random.shuffle(usable)
    split_idx = int(len(usable) * TRAIN_RATIO)
    train_set = usable[:split_idx]
    val_set = usable[split_idx:]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Create output dirs
    for split in ['train', 'val']:
        (OUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def convert_bbox(bbox, img_w, img_h):
        """COCO [x,y,w,h] -> YOLO [cx,cy,w,h] normalized."""
        x, y, w, h = bbox
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        # Clamp
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))
        return cx, cy, nw, nh

    def process_split(dataset, split_name):
        print(f"\nProcessing {split_name}...")
        skipped = 0
        for img_info, anns in tqdm(dataset, desc=split_name):
            fname = Path(img_info['file_name']).name
            src = IMAGES_DIR / fname
            dst_img = OUT_DIR / 'images' / split_name / fname
            dst_lbl = OUT_DIR / 'labels' / split_name / (Path(fname).stem + '.txt')

            # Copy image
            shutil.copy2(src, dst_img)

            # Write YOLO label
            img_w, img_h = img_info['width'], img_info['height']
            lines = []
            for ann in anns:
                yolo_cls = KEEP_CATS[ann['category_id']]
                bbox = ann['bbox']
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                cx, cy, nw, nh = convert_bbox(bbox, img_w, img_h)
                if nw < 0.001 or nh < 0.001:
                    skipped += 1
                    continue
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            dst_lbl.write_text('\n'.join(lines))

        print(f"  Skipped {skipped} tiny bboxes")

    process_split(train_set, 'train')
    process_split(val_set, 'val')

    # Write data.yaml
    yaml_content = f"""path: {OUT_DIR}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for i in range(len(CLASS_NAMES)):
        yaml_content += f"  {i}: {CLASS_NAMES[i]}\n"

    yaml_path = OUT_DIR / 'data.yaml'
    yaml_path.write_text(yaml_content)
    print(f"\nWrote {yaml_path}")

    # Stats
    train_imgs = len(list((OUT_DIR / 'images' / 'train').glob('*.jpg')))
    val_imgs = len(list((OUT_DIR / 'images' / 'val').glob('*.jpg')))
    print(f"\nFinal dataset:")
    print(f"  Train: {train_imgs} images")
    print(f"  Val:   {val_imgs} images")
    print(f"  Classes: {CLASS_NAMES}")

    # Annotation stats per class
    cls_count = defaultdict(int)
    for img_info, anns in usable:
        for ann in anns:
            cls_count[KEEP_CATS[ann['category_id']]] += 1
    print("\nAnnotation counts per class:")
    for cls_id, name in CLASS_NAMES.items():
        print(f"  {name}: {cls_count[cls_id]}")


if __name__ == '__main__':
    main()
