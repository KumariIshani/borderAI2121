"""Dataset Preparation, Validation, and Visualization for Sentinel Border Watch.

Converts, remaps, and validates bounding box annotations into YOLOv8 format:
Classes:
0: person
1: weapon
2: vehicle
3: military_vehicle

Produces:
- data/dataset/data.yaml
- data/results/dataset_stats.json
- data/results/class_distribution.png
- 24 color-coded sample images in data/dataset_samples/
"""

from __future__ import annotations

import os
import glob
import json
import logging
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("sentinel.prepare_dataset")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CLASS_NAMES = ["person", "weapon", "vehicle", "military_vehicle"]
CLASS_COLORS = {
    0: (255, 180, 50),   # Sky Blue / Cyan (Person)
    1: (30, 30, 240),    # Red (Weapon)
    2: (50, 220, 50),    # Green (Vehicle)
    3: (0, 140, 255),    # Orange (Military Vehicle)
}


def create_data_yaml(dataset_dir: str = os.path.join("data", "dataset")) -> str:
    """Generates the data.yaml file required by Ultralytics YOLOv8."""
    abs_dir = os.path.abspath(dataset_dir).replace("\\", "/")
    yaml_content = f"""# Sentinel Border Watch - YOLOv8 Dataset Configuration
path: {abs_dir}
train: images/train
val: images/val
test: images/test

# Classes
nc: 4
names:
  0: person
  1: weapon
  2: vehicle
  3: military_vehicle
"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info(f"Generated data.yaml at {yaml_path}")
    return yaml_path


def compute_dataset_stats(dataset_dir: str = os.path.join("data", "dataset")) -> Dict[str, Any]:
    """Scans dataset/labels/{train,val,test} to compute class and box statistics."""
    splits = ["train", "val", "test"]
    stats: Dict[str, Any] = {
        "splits": {},
        "total_images": 0,
        "total_boxes": 0,
        "class_totals": {name: 0 for name in CLASS_NAMES},
    }

    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        images = glob.glob(os.path.join(img_dir, "*.*"))
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [f for f in images if os.path.splitext(f)[1].lower() in valid_exts]

        class_counts = {name: 0 for name in CLASS_NAMES}
        box_count = 0
        corrupt_or_empty = 0

        for img_path in images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
                corrupt_or_empty += 1
                continue

            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        if 0 <= cid < len(CLASS_NAMES):
                            cname = CLASS_NAMES[cid]
                            class_counts[cname] += 1
                            stats["class_totals"][cname] += 1
                            box_count += 1
                    except ValueError:
                        pass

        split_info = {
            "image_count": len(images),
            "box_count": box_count,
            "corrupt_or_empty_labels": corrupt_or_empty,
            "class_distribution": class_counts,
        }
        stats["splits"][split] = split_info
        stats["total_images"] += len(images)
        stats["total_boxes"] += box_count

    stats_out = os.path.join("data", "results", "dataset_stats.json")
    os.makedirs(os.path.dirname(stats_out), exist_ok=True)
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset stats saved to {stats_out}")
    print("\n" + "=" * 60)
    print("SENTINEL BORDER WATCH — DATASET STATISTICS")
    print("=" * 60)
    for s, data in stats["splits"].items():
        print(f"[{s.upper():<5}] Images: {data['image_count']:<5} | Total Boxes: {data['box_count']:<6} | Classes: {data['class_distribution']}")
    print("-" * 60)
    print(f"TOTAL: {stats['total_images']} Images, {stats['total_boxes']} Annotations")
    print(f"Class Totals: {stats['class_totals']}")
    print("=" * 60 + "\n")

    return stats


def plot_class_distribution(stats: Dict[str, Any], output_path: str = os.path.join("data", "results", "class_distribution.png")) -> None:
    """Renders a high-contrast publication-grade bar chart of class counts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    splits = ["train", "val", "test"]
    class_names = CLASS_NAMES

    x = np.arange(len(class_names))
    width = 0.25

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Tricolor / tactical palette
    colors = {"train": "#FF9933", "val": "#00E5FF", "test": "#138808"}

    for i, s in enumerate(splits):
        counts = [stats["splits"].get(s, {}).get("class_distribution", {}).get(c, 0) for c in class_names]
        ax.bar(x + (i - 1) * width, counts, width, label=s.capitalize(), color=colors[s], alpha=0.9, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Bounding Box Instances", fontsize=11, fontweight="bold", color="white")
    ax.set_title("Sentinel Border Watch — Dataset Class Distribution Across Splits", fontsize=12, fontweight="bold", pad=12, color="#FF9933")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper().replace("_", " ") for c in class_names], fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="#1e222d", edgecolor="#444")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Class distribution chart saved to {output_path}")


def render_ground_truth_samples(
    dataset_dir: str = os.path.join("data", "dataset"),
    output_dir: str = os.path.join("data", "dataset_samples"),
    max_samples: int = 24,
) -> List[str]:
    """Renders ground-truth annotated bounding boxes for visual verification."""
    os.makedirs(output_dir, exist_ok=True)

    # Pick samples from train split
    img_dir = os.path.join(dataset_dir, "images", "train")
    lbl_dir = os.path.join(dataset_dir, "labels", "train")

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]

    saved_paths = []
    sample_idx = 0

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h

                x1 = int(max(0, cx - bw / 2.0))
                y1 = int(max(0, cy - bh / 2.0))
                x2 = int(min(w - 1, cx + bw / 2.0))
                y2 = int(min(h - 1, cy + bh / 2.0))

                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
                color = CLASS_COLORS.get(cid, (255, 255, 255))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                lbl_text = f"GT: {cname.upper()}"
                cv2.putText(img, lbl_text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        out_name = f"sample_gt_{sample_idx + 1:02d}_{base_name[:12]}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img)
        saved_paths.append(out_path)
        sample_idx += 1

        if sample_idx >= max_samples:
            break

    logger.info(f"Rendered {len(saved_paths)} ground-truth sample images to {output_dir}")
    return saved_paths


if __name__ == "__main__":
    create_data_yaml()
    stats = compute_dataset_stats()
    plot_class_distribution(stats)
    render_ground_truth_samples()
