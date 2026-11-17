"""
Build a unified, authentic, real-world YOLOv8 dataset for border security.
Sources:
  1. Roboflow Universe Real Weapon Detection (Guns, Knives, Persons)
  2. Real COCO Surveillance/Street Images (Vehicles, Persons, Knives)
  3. Real Military Vehicles (Tanks, APCs, Armored Vehicles from DVIDS)

Class Mapping:
  0: person
  1: weapon
  2: vehicle
  3: military_vehicle
"""
import os
import shutil
import random
import json
import glob
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = DATA_DIR / "dataset"
RESULTS_DIR = DATA_DIR / "results"
SAMPLES_DIR = DATA_DIR / "dataset_samples"

CLASS_NAMES = ["person", "weapon", "vehicle", "military_vehicle"]
CLASS_COLORS = {
    0: (50, 205, 50),     # Lime green: person
    1: (0, 0, 255),       # Red: weapon
    2: (255, 165, 0),     # Orange: vehicle
    3: (255, 0, 128)      # Magenta: military_vehicle
}

def clean_and_prepare_dirs():
    for split in ["train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def process_roboflow_parquet(max_samples=350):
    parquet_path = RAW_DIR / "weapon_val.parquet"
    if not parquet_path.exists():
        print(f"[!] Parquet file {parquet_path} not found.")
        return []
    
    print(f"[*] Reading Roboflow Parquet: {parquet_path}...")
    table = pq.read_table(str(parquet_path))
    num_rows = table.num_rows
    indices = list(range(num_rows))
    random.shuffle(indices)
    selected_indices = indices[:max_samples]
    
    items = []
    # Class mapping in parquet: 0: GUN -> 1, 1: KNIFE -> 1, 2: PERSON -> 0
    cat_map = {0: 1, 1: 1, 2: 0}
    
    for idx in selected_indices:
        img_id = table["image_id"][idx].as_py()
        width = table["width"][idx].as_py()
        height = table["height"][idx].as_py()
        img_bytes = table["image"][idx].as_py()["bytes"]
        objects = table["objects"][idx].as_py()
        
        bboxes = objects["bbox"]  # list of [xmin, ymin, w, h]
        categories = objects["category"]
        
        yolo_labels = []
        for bbox, cat in zip(bboxes, categories):
            if cat not in cat_map:
                continue
            target_cls = cat_map[cat]
            xmin, ymin, bw, bh = bbox
            if bw <= 1 or bh <= 1:
                continue
            # Convert to YOLO (xc, yc, w, h) normalized
            xc = (xmin + bw / 2.0) / width
            yc = (ymin + bh / 2.0) / height
            nw = bw / width
            nh = bh / height
            
            # Clip
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))
            yolo_labels.append(f"{target_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        
        if yolo_labels:
            items.append({
                "id": f"rf_weapon_{img_id:05d}",
                "image_bytes": img_bytes,
                "labels": yolo_labels,
                "classes": list(set(int(l.split()[0]) for l in yolo_labels))
            })
            
    print(f"[+] Processed {len(items)} real weapon/person images from Roboflow Universe.")
    return items

def process_coco128():
    coco_img_dir = RAW_DIR / "coco128" / "coco128" / "images" / "train2017"
    coco_lbl_dir = RAW_DIR / "coco128" / "coco128" / "labels" / "train2017"
    if not coco_img_dir.exists():
        print(f"[!] COCO128 not found at {coco_img_dir}")
        return []
    
    print("[*] Processing COCO128 real images for vehicles, persons, weapons...")
    # COCO classes:
    # 0 -> 0 (person)
    # 2 (car), 3 (motorcycle), 5 (bus), 7 (truck) -> 2 (vehicle)
    # 43 (knife) -> 1 (weapon)
    coco_map = {0: 0, 2: 2, 3: 2, 5: 2, 7: 2, 43: 1}
    
    items = []
    for lbl_file in coco_lbl_dir.glob("*.txt"):
        stem = lbl_file.stem
        img_file = coco_img_dir / f"{stem}.jpg"
        if not img_file.exists():
            continue
            
        yolo_labels = []
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                c = int(parts[0])
                if c in coco_map:
                    target_cls = coco_map[c]
                    yolo_labels.append(f"{target_cls} {' '.join(parts[1:5])}")
                    
        if yolo_labels:
            items.append({
                "id": f"coco_{stem}",
                "image_path": str(img_file),
                "labels": yolo_labels,
                "classes": list(set(int(l.split()[0]) for l in yolo_labels))
            })
            
    print(f"[+] Processed {len(items)} real images from COCO128.")
    return items

def process_military():
    mil_img_dir = RAW_DIR / "military" / "images"
    mil_lbl_dir = RAW_DIR / "military" / "labels"
    if not mil_img_dir.exists() or not mil_lbl_dir.exists():
        print("[!] Military dirs not found.")
        return []
        
    print("[*] Processing DVIDS military real images for tanks/APCs...")
    # Military classes:
    # 0 (soldier) -> 0 (person)
    # 1 (tank), 2 (apc), 10, 11 -> 3 (military_vehicle)
    mil_map = {0: 0, 1: 3, 2: 3, 10: 3, 11: 3}
    
    items = []
    for img_file in mil_img_dir.glob("*.jpg"):
        stem = img_file.stem
        lbl_file = mil_lbl_dir / f"{stem}.txt"
        if not lbl_file.exists():
            continue
            
        yolo_labels = []
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                c = int(parts[0])
                if c in mil_map:
                    target_cls = mil_map[c]
                    yolo_labels.append(f"{target_cls} {' '.join(parts[1:5])}")
                    
        if yolo_labels:
            items.append({
                "id": f"mil_{stem}",
                "image_path": str(img_file),
                "labels": yolo_labels,
                "classes": list(set(int(l.split()[0]) for l in yolo_labels))
            })
            
    print(f"[+] Processed {len(items)} real military vehicle images.")
    return items

def main():
    clean_and_prepare_dirs()
    
    rf_items = process_roboflow_parquet(max_samples=350)
    coco_items = process_coco128()
    mil_items = process_military()
    
    all_items = rf_items + coco_items + mil_items
    print(f"\n[+] Total real images gathered: {len(all_items)}")
    
    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    splits = {
        "train": all_items[:n_train],
        "val": all_items[n_train:n_train + n_val],
        "test": all_items[n_train + n_val:]
    }
    
    # Write dataset files
    stats = {
        "splits": {},
        "class_counts": {name: {"train": 0, "val": 0, "test": 0, "total": 0} for name in CLASS_NAMES},
        "total_images": n,
        "total_boxes": 0
    }
    
    print("\n[*] Writing dataset to data/dataset/ ...")
    for split_name, items in splits.items():
        img_out_dir = DATASET_DIR / "images" / split_name
        lbl_out_dir = DATASET_DIR / "labels" / split_name
        
        # Clear out existing files in this split
        for f in img_out_dir.glob("*.*"):
            f.unlink()
        for f in lbl_out_dir.glob("*.txt"):
            f.unlink()
            
        split_boxes = 0
        for item in items:
            item_id = item["id"]
            img_dest = img_out_dir / f"{item_id}.jpg"
            lbl_dest = lbl_out_dir / f"{item_id}.txt"
            
            # Save image
            if "image_bytes" in item:
                with open(img_dest, "wb") as f:
                    f.write(item["image_bytes"])
            elif "image_path" in item:
                shutil.copy2(item["image_path"], img_dest)
                
            # Save label
            with open(lbl_dest, "w") as f:
                f.write("\n".join(item["labels"]) + "\n")
                
            # Stats
            for line in item["labels"]:
                cls_idx = int(line.split()[0])
                cls_name = CLASS_NAMES[cls_idx]
                stats["class_counts"][cls_name][split_name] += 1
                stats["class_counts"][cls_name]["total"] += 1
                split_boxes += 1
                stats["total_boxes"] += 1
                
        stats["splits"][split_name] = {
            "images": len(items),
            "boxes": split_boxes
        }
        print(f"  {split_name.upper()}: {len(items)} images, {split_boxes} bounding boxes")
        
    # Write data.yaml
    data_yaml_path = DATASET_DIR / "data.yaml"
    data_yaml_content = f"""path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: weapon
  2: vehicle
  3: military_vehicle
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
    print(f"[+] Wrote dataset configuration: {data_yaml_path}")
    
    # Save dataset_stats.json
    stats_path = RESULTS_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[+] Saved dataset stats: {stats_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print(f"{'Class':<20}{'Train':<10}{'Val':<10}{'Test':<10}{'Total':<10}")
    print("="*60)
    for cname in CLASS_NAMES:
        c_stats = stats["class_counts"][cname]
        print(f"{cname:<20}{c_stats['train']:<10}{c_stats['val']:<10}{c_stats['test']:<10}{c_stats['total']:<10}")
    print("="*60)
    print(f"{'TOTAL IMAGES':<20}{stats['splits']['train']['images']:<10}{stats['splits']['val']['images']:<10}{stats['splits']['test']['images']:<10}{stats['total_images']:<10}")
    print(f"{'TOTAL BOXES':<20}{stats['splits']['train']['boxes']:<10}{stats['splits']['val']['boxes']:<10}{stats['splits']['test']['boxes']:<10}{stats['total_boxes']:<10}")
    print("="*60 + "\n")
    
    # Plot class distribution chart
    plot_class_distribution(stats)
    
    # Generate 24 ground truth samples
    generate_gt_samples()

def plot_class_distribution(stats):
    fig, ax = plt.subplots(figsize=(10, 6))
    import numpy as np
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    train_vals = [stats["class_counts"][c]["train"] for c in CLASS_NAMES]
    val_vals = [stats["class_counts"][c]["val"] for c in CLASS_NAMES]
    test_vals = [stats["class_counts"][c]["test"] for c in CLASS_NAMES]
    
    rects1 = ax.bar(x - width, train_vals, width, label='Train', color='#FF9933')
    rects2 = ax.bar(x, val_vals, width, label='Validation', color='#FFFFFF', edgecolor='#333333')
    rects3 = ax.bar(x + width, test_vals, width, label='Test', color='#138808')
    
    ax.set_facecolor('#1a1a24')
    fig.patch.set_facecolor('#111118')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444455')
    ax.set_ylabel('Instance Count', color='white', fontsize=12)
    ax.set_title('Real Dataset Class Distribution (Train / Val / Test)', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color='white', fontsize=11)
    ax.legend(facecolor='#222233', edgecolor='#444455', labelcolor='white')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#888888')
    
    chart_path = RESULTS_DIR / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved distribution chart to {chart_path}")

def generate_gt_samples(count=24):
    for f in SAMPLES_DIR.glob("*.*"):
        f.unlink()
        
    all_imgs = list((DATASET_DIR / "images" / "train").glob("*.jpg")) + list((DATASET_DIR / "images" / "val").glob("*.jpg"))
    random.shuffle(all_imgs)
    selected = all_imgs[:count]
    
    saved_count = 0
    for img_path in selected:
        lbl_path = DATASET_DIR / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_idx = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)
                
                color = CLASS_COLORS.get(cls_idx, (0, 255, 0))
                cname = CLASS_NAMES[cls_idx]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_txt = f"{cname}"
                cv2.putText(img, label_txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        out_path = SAMPLES_DIR / f"gt_sample_{saved_count+1:02d}_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        saved_count += 1
        
    print(f"[+] Saved {saved_count} ground truth sample images to {SAMPLES_DIR}")

if __name__ == "__main__":
    main()

