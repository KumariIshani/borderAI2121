"""Armed Person Spatial Linking Logic for Sentinel Border Watch.

Associates detected weapons with detected persons based on geometric containment
and bounding-box overlap heuristics.

Methodology & Honesty Note:
---------------------------
True armed-person verification in high-risk border environments ideally utilizes
keypoint pose estimation (e.g. wrist/hand alignment with weapon triggers) or 3D
spatial reasoning. In this YOLOv8 pipeline, an association heuristic is employed:
1. For each detected weapon, candidate persons are evaluated based on whether
   the weapon's center point lies within the person's bounding box (with an optional
   expansion margin to account for outstretched arms / holsters).
2. If multiple candidates exist, the person whose center is closest to the weapon
   center (or shares maximum intersection area) is linked.
3. Once linked, the person detection is flagged as "ARMED PERSON" (status: ARMED).
4. Any weapon not matching a person within the proximity threshold is classified
   as an "UNATTACHED WEAPON" (e.g. discarded firearm, cache, or ground hazard).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from .detector import Detection


@dataclass
class ArmedPerson:
    """Represents a person detection identified as carrying one or more weapons."""
    person_detection: Detection
    linked_weapons: List[Detection] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.person_detection.bbox

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return self.person_detection.bottom_center

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person": self.person_detection.to_dict(),
            "weapons": [w.to_dict() for w in self.linked_weapons],
            "confidence": round(self.confidence, 4),
            "weapon_count": len(self.linked_weapons),
        }


@dataclass
class ArmedPersonResult:
    """Aggregated output of the spatial linking process."""
    armed_persons: List[ArmedPerson]
    unarmed_persons: List[Detection]
    unattached_weapons: List[Detection]
    other_detections: List[Detection]

    @property
    def total_armed_count(self) -> int:
        return len(self.armed_persons)

    @property
    def total_unattached_weapon_count(self) -> int:
        return len(self.unattached_weapons)


def _point_in_box(
    point: Tuple[float, float],
    box: Tuple[float, float, float, float],
    margin_ratio: float = 0.15,
) -> bool:
    """Checks if a point (cx, cy) is within a box, expanded by a margin ratio."""
    px, py = point
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    mx = w * margin_ratio
    my = h * margin_ratio

    return (x1 - mx) <= px <= (x2 + mx) and (y1 - my) <= py <= (y2 + my)


def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter_area / float(areaA + areaB - inter_area)


def _box_intersection_ratio(box_small: Tuple[float, float, float, float], box_large: Tuple[float, float, float, float]) -> float:
    """Computes the fraction of box_small that is covered by box_large."""
    xA = max(box_small[0], box_large[0])
    yA = max(box_small[1], box_large[1])
    xB = min(box_small[2], box_large[2])
    yB = min(box_small[3], box_large[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    small_area = max(1.0, (box_small[2] - box_small[0]) * (box_small[3] - box_small[1]))

    return inter_area / float(small_area)


def link_weapons_to_persons(
    detections: List[Detection],
    containment_margin: float = 0.20,
    min_intersection_ratio: float = 0.25,
) -> ArmedPersonResult:
    """Associates weapon detections with person detections.

    Args:
        detections: All detected objects in the frame.
        containment_margin: Relative bounding-box expansion ratio for outstretched arms/hands.
        min_intersection_ratio: Minimum overlap of weapon area inside person box to link.

    Returns:
        ArmedPersonResult segregating armed persons, unarmed persons,
        unattached weapons, and vehicles.
    """
    persons: List[Detection] = []
    weapons: List[Detection] = []
    others: List[Detection] = []

    for d in detections:
        if d.class_id == 0 or d.class_name.lower() == "person":
            persons.append(d)
        elif d.class_id == 1 or d.class_name.lower() == "weapon":
            weapons.append(d)
        else:
            others.append(d)

    # Track matches: person_idx -> list of linked weapon detections
    person_weapons: Dict[int, List[Detection]] = {i: [] for i in range(len(persons))}
    unattached_weapons: List[Detection] = []

    for weapon in weapons:
        best_person_idx: Optional[int] = None
        min_dist = float("inf")

        for p_idx, person in enumerate(persons):
            wcx, wcy = weapon.center
            center_inside = _point_in_box((wcx, wcy), person.bbox, margin_ratio=containment_margin)
            inter_ratio = _box_intersection_ratio(weapon.bbox, person.bbox)

            if center_inside or inter_ratio >= min_intersection_ratio:
                # Compute distance between centers to break ties
                pcx, pcy = person.center
                dist = math.hypot(wcx - pcx, wcy - pcy)
                if dist < min_dist:
                    min_dist = dist
                    best_person_idx = p_idx

        if best_person_idx is not None:
            person_weapons[best_person_idx].append(weapon)
        else:
            unattached_weapons.append(weapon)

    armed_persons: List[ArmedPerson] = []
    unarmed_persons: List[Detection] = []

    for p_idx, person in enumerate(persons):
        linked = person_weapons[p_idx]
        if linked:
            # Combined confidence based on person conf and max weapon conf
            max_w_conf = max(w.confidence for w in linked)
            joint_conf = 0.5 * (person.confidence + max_w_conf)
            armed_persons.append(
                ArmedPerson(
                    person_detection=person,
                    linked_weapons=linked,
                    confidence=joint_conf,
                )
            )
        else:
            unarmed_persons.append(person)

    return ArmedPersonResult(
        armed_persons=armed_persons,
        unarmed_persons=unarmed_persons,
        unattached_weapons=unattached_weapons,
        other_detections=others,
    )
