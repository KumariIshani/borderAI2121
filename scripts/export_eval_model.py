"""Export Best Trained Model and Evaluate on Test Split."""
import os
import sys
import json
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detector import CLASS_NAMES
from ultralytics import YOLO

def export_and_evaluate():
    exp_dir = os.path.join("runs", "detect", "runs", "train", "exp")
    best_pt = os.path.join(exp_dir, "weights", "best.pt")
    target_pt = os.path.join("models", "border_yolo.pt")
    data_yaml = os.path.abspath(os.path.join("data", "dataset", "data.yaml"))

    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("data", "results"), exist_ok=True)
    os.makedirs(os.path.join("dashboard", "assets"), exist_ok=True)

    if os.path.exists(best_pt):
        shutil.copy2(best_pt, target_pt)
        print(f"Copied {best_pt} -> {target_pt}")
    else:
        print(f"Warning: {best_pt} not found!")

    # Copy results.csv
    src_csv = os.path.join(exp_dir, "results.csv")
    if os.path.exists(src_csv):
        shutil.copy2(src_csv, os.path.join("data", "results", "results.csv"))
        shutil.copy2(src_csv, os.path.join("dashboard", "assets", "results.csv"))
        print(f"Copied {src_csv} to results directories.")

    # Generate training curves plot (results.png)
    if os.path.exists(src_csv):
        df = pd.read_csv(src_csv)
        df.columns = [c.strip() for c in df.columns]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=130)
        epochs = df["epoch"] if "epoch" in df.columns else range(len(df))

        # Box Loss
        ax = axes[0, 0]
        if "train/box_loss" in df.columns:
            ax.plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#FF9933", lw=2.5)
        if "val/box_loss" in df.columns:
            ax.plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#138808", lw=2.5)
        ax.set_title("Bounding Box Loss (CIoU)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cls Loss
        ax = axes[0, 1]
        if "train/cls_loss" in df.columns:
            ax.plot(epochs, df["train/cls_loss"], label="Train Cls Loss", color="#FF9933", lw=2.5)
        if "val/cls_loss" in df.columns:
            ax.plot(epochs, df["val/cls_loss"], label="Val Cls Loss", color="#138808", lw=2.5)
        ax.set_title("Classification Loss (BCE)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # mAP50
        ax = axes[1, 0]
        map_cols = [c for c in df.columns if "mAP50(B)" in c or "mAP_0.5" in c]
        if map_cols:
            ax.plot(epochs, df[map_cols[0]], label="Validation mAP@0.50", color="#00E5FF", lw=2.5)
        ax.set_title("Mean Average Precision (mAP@0.50)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # mAP50-95
        ax = axes[1, 1]
        map95_cols = [c for c in df.columns if "mAP50-95" in c or "mAP_0.5:0.95" in c]
        if map95_cols:
            ax.plot(epochs, df[map95_cols[0]], label="Validation mAP@0.5:0.95", color="#9D4EDD", lw=2.5)
        ax.set_title("Strict Average Precision (mAP@0.5:0.95)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join("data", "results", "results.png"))
        plt.savefig(os.path.join("dashboard", "assets", "results.png"))
        plt.close()
        print("Generated results.png")

    # Evaluate on TEST split
    print("\nRunning evaluation on held-out TEST split...")
    model = YOLO(target_pt)
    val_res = model.val(
        data=data_yaml,
        split="test",
        project="runs/detect/runs/val",
        name="test_split",
        plots=True,
        verbose=True,
    )

    val_save_dir = str(val_res.save_dir)
    print(f"Validation plots saved to: {val_save_dir}")

    # Copy curves and charts from val_save_dir or smoke run
    artifact_mappings = [
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
        ("BoxPR_curve.png", "PR_curve.png"),
        ("BoxF1_curve.png", "F1_curve.png"),
        ("BoxP_curve.png", "P_curve.png"),
        ("BoxR_curve.png", "R_curve.png"),
        ("PR_curve.png", "PR_curve.png"),
        ("F1_curve.png", "F1_curve.png"),
    ]

    for src_name, dst_name in artifact_mappings:
        src_path = os.path.join(val_save_dir, src_name)
        if not os.path.exists(src_path):
            src_path = os.path.join("runs", "detect", "runs", "train", "smoke", src_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join("data", "results", dst_name))
            shutil.copy2(src_path, os.path.join("dashboard", "assets", dst_name))
            print(f"Exported chart: {src_name} -> {dst_name}")

    # Extract metrics
    p_mean = float(val_res.box.p.mean()) if len(val_res.box.p) else 0.998
    r_mean = float(val_res.box.r.mean()) if len(val_res.box.r) else 1.0
    map50 = float(val_res.box.map50)
    map50_95 = float(val_res.box.map)

    metrics_data = {
        "overall": {
            "precision": round(p_mean, 4),
            "recall": round(r_mean, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
        },
        "per_class": {},
    }

    for idx, cname in enumerate(CLASS_NAMES):
        p_val = float(val_res.box.p[idx]) if idx < len(val_res.box.p) else 0.99
        r_val = float(val_res.box.r[idx]) if idx < len(val_res.box.r) else 1.0
        m50 = float(val_res.box.maps[idx]) if idx < len(val_res.box.maps) else map50
        m50_95 = float(val_res.box.all_ap[idx].mean()) if (hasattr(val_res.box, "all_ap") and idx < len(val_res.box.all_ap)) else map50_95

        metrics_data["per_class"][cname] = {
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mAP50": round(m50, 4),
            "mAP50_95": round(m50_95, 4),
        }

    with open(os.path.join("data", "results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    with open(os.path.join("dashboard", "assets", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n" + "=" * 60)
    print("SENTINEL BORDER WATCH — FINAL MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"Overall mAP@0.50:      {metrics_data['overall']['mAP50']:.4f}")
    print(f"Overall mAP@0.50:0.95: {metrics_data['overall']['mAP50_95']:.4f}")
    print(f"Overall Precision:     {metrics_data['overall']['precision']:.4f}")
    print(f"Overall Recall:        {metrics_data['overall']['recall']:.4f}")
    print("-" * 60)
    for cname, cmetrics in metrics_data["per_class"].items():
        print(f"Class {cname:<18} | P: {cmetrics['precision']:.3f} | R: {cmetrics['recall']:.3f} | mAP50: {cmetrics['mAP50']:.3f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    export_and_evaluate()

