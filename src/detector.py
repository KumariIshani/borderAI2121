"""YOLOv8 Object Detector Wrapper for Sentinel Border Watch.

Detects:
0: person
1: weapon
2: vehicle
3: military_vehicle
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("sentinel.detector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CLASS_NAMES: Dict[int, str] = {
    0: "person",
    1: "weapon",
    2: "vehicle",
    3: "military_vehicle",
}

DEFAULT_MODEL_PATH = os.path.join("models", "border_yolo.pt")
DEFAULT_FIREARM_MODEL_PATH = os.path.join("models", "firearm_yolo.pt")


@dataclass
class Detection:
    """Represents a single bounding box detection in an image/frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    center: Tuple[float, float]             # (cx, cy)
    bottom_center: Tuple[float, float]      # (cx, y2)
    normalized_area: float                  # (w * h) / (img_w * img_h)
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "center": [round(float(v), 2) for v in self.center],
            "bottom_center": [round(float(v), 2) for v in self.bottom_center],
            "normalized_area": round(float(self.normalized_area), 6),
            "track_id": self.track_id,
        }


def compute_box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class BorderDetector:
    """Wrapper around Ultralytics YOLOv8 for border security surveillance.
    
    Combines foundation YOLOv8 precision for reliable person and vehicle detection with
    dedicated weights for firearms/weapons and armored military vehicles.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        firearm_model_path: str = DEFAULT_FIREARM_MODEL_PATH,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.firearm_model_path = firearm_model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

        from ultralytics import YOLO
        import torch

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Detector initialized on device: {self.device}")

        # 1. Foundation model for robust person and vehicle detection
        base_weights = "yolov8s.pt" if os.path.exists("yolov8s.pt") else "yolov8n.pt"
        logger.info(f"Loading foundation YOLO model from: {base_weights}")
        self.foundation_model = YOLO(base_weights)

        # 2. Dedicated firearm/weapon detection model (trained on firearms, handguns, rifles)
        if os.path.exists(self.firearm_model_path):
            logger.info(f"Loading dedicated firearm YOLO model from: {self.firearm_model_path}")
            self.firearm_model = YOLO(self.firearm_model_path)
        else:
            self.firearm_model = None

        # 3. Domain border weights for military vehicles & additional border threats
        if os.path.exists(self.model_path) and os.path.abspath(self.model_path) not in [
            os.path.abspath(base_weights),
            os.path.abspath(self.firearm_model_path) if self.firearm_model_path else "",
        ]:
            logger.info(f"Loading domain-specific YOLO model from: {self.model_path}")
            self.custom_model = YOLO(self.model_path)
        else:
            self.custom_model = None

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Runs multi-class border threat detection on a BGR image or video frame."""
        h, w = image.shape[:2]
        img_area = float(max(1, w * h))

        raw_persons: List[Tuple[int, str, float, Tuple[float, float, float, float]]] = []
        raw_weapons: List[Tuple[int, str, float, Tuple[float, float, float, float]]] = []
        raw_vehicles: List[Tuple[int, str, float, Tuple[float, float, float, float]]] = []

        # 1. Predict with foundation detector
        r_f = self.foundation_model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )[0]

        if r_f.boxes is not None and len(r_f.boxes) > 0:
            for b in r_f.boxes:
                c = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                if c == 0:  # person
                    raw_persons.append((0, "person", conf, (x1, y1, x2, y2)))
                elif c in [2, 3, 5, 7]:  # car, motorcycle, bus, truck -> vehicle
                    raw_vehicles.append((2, "vehicle", conf, (x1, y1, x2, y2)))
                elif c == 43:  # knife -> weapon
                    raw_weapons.append((1, "weapon", conf, (x1, y1, x2, y2)))

        # 2. Predict with dedicated firearm detector (Guns, Rifles, Firearms)
        if self.firearm_model is not None:
            r_w = self.firearm_model.predict(
                source=image,
                conf=max(0.20, self.conf_thresh),
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )[0]
            if r_w.boxes is not None and len(r_w.boxes) > 0:
                for b in r_w.boxes:
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    raw_weapons.append((1, "weapon", conf, (x1, y1, x2, y2)))

        # 3. Predict with domain custom detector (military armored units / additional border threats)
        if self.custom_model is not None:
            r_c = self.custom_model.predict(
                source=image,
                conf=max(0.35, self.conf_thresh),
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )[0]
            if r_c.boxes is not None and len(r_c.boxes) > 0:
                for b in r_c.boxes:
                    c = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    if c == 1 and conf >= 0.35:  # weapon
                        raw_weapons.append((1, "weapon", conf, (x1, y1, x2, y2)))
                    elif c == 3 and conf >= 0.45:  # military_vehicle (tank, APC)
                        raw_vehicles.append((3, "military_vehicle", conf, (x1, y1, x2, y2)))
                    elif c == 0 and conf >= 0.70:
                        raw_persons.append((0, "person", conf, (x1, y1, x2, y2)))

        # 4. Class-aware Non-Maximum Suppression (NMS)
        def nms_filter(items: List[Tuple[int, str, float, Tuple[float, float, float, float]]], iou_th: float = 0.45):
            items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
            kept = []
            for item in items_sorted:
                overlap = False
                for k in kept:
                    if compute_box_iou(item[3], k[3]) > iou_th:
                        overlap = True
                        break
                if not overlap:
                    kept.append(item)
            return kept

        filtered_persons = nms_filter(raw_persons, 0.45)
        filtered_weapons = nms_filter(raw_weapons, 0.45)

        # Vehicle NMS with military_vehicle priority
        raw_vehicles.sort(key=lambda x: x[2], reverse=True)
        filtered_vehicles = []
        for v_item in raw_vehicles:
            v_cid, v_cname, v_conf, v_bbox = v_item
            match_idx = -1
            for idx, ex in enumerate(filtered_vehicles):
                if compute_box_iou(v_bbox, ex[3]) > 0.45:
                    match_idx = idx
                    break
            if match_idx >= 0:
                ex_cid, ex_cname, ex_conf, ex_bbox = filtered_vehicles[match_idx]
                if v_cid == 3 and ex_cid == 2:
                    filtered_vehicles[match_idx] = (3, "military_vehicle", max(v_conf, ex_conf), ex_bbox)
            else:
                filtered_vehicles.append(v_item)

        all_candidates = filtered_persons + filtered_weapons + filtered_vehicles

        # 5. Construct Detection dataclasses
        detections: List[Detection] = []
        for cls_id, cls_name, conf, (x1, y1, x2, y2) in all_candidates:
            # Clip coordinates
            x1 = max(0.0, min(float(w - 1), x1))
            x2 = max(0.0, min(float(w - 1), x2))
            y1 = max(0.0, min(float(h - 1), y1))
            y2 = max(0.0, min(float(h - 1), y2))

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            bottom_center = (cx, y2)
            norm_area = (bw * bh) / img_area

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    bottom_center=bottom_center,
                    normalized_area=norm_area,
                )
            )

        return detections
