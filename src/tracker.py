"""Real-time Multi-Object IoU Tracker for Sentinel Border Surveillance.

Assigns persistent Track IDs to objects across video frames, counts unique entities,
prevents multi-frame double-counting (e.g. reporting 200 persons for 3 individuals),
and tracks armed status across the lifespan of each tracked person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from .detector import Detection, compute_box_iou


@dataclass
class Track:
    track_id: int
    class_name: str
    class_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    max_confidence: float
    is_armed: bool = False
    hits: int = 1
    age: int = 0
    time_since_update: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.max_confidence, 3),
            "bbox": [round(v, 1) for v in self.bbox],
            "is_armed": self.is_armed,
            "total_frames_seen": self.hits,
        }


class VideoEntityTracker:
    """Multi-object tracker with persistence, IoU matching, and armed-threat tracking."""

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 25, min_hits_to_confirm: int = 2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits_to_confirm = min_hits_to_confirm
        self.next_id = 1
        self.active_tracks: List[Track] = []
        self.all_confirmed_tracks: Dict[int, Track] = {}
        self.peak_concurrent: int = 0

    def update(
        self,
        unarmed_persons: List[Detection],
        armed_persons: List[Any],  # ArmedPerson objects
        other_detections: List[Detection],
        unattached_weapons: List[Detection],
    ) -> List[Tuple[int, str, Tuple[float, float, float, float], float, bool]]:
        """Updates tracks with current frame detections.

        Returns list of active renderable tuples: (track_id, label, bbox, conf, is_armed).
        """
        # Formulate candidates for this frame
        frame_candidates = []

        for p in unarmed_persons:
            frame_candidates.append({
                "class_name": "person",
                "class_id": 0,
                "bbox": p.bbox,
                "conf": p.confidence,
                "is_armed": False,
            })

        for ap in armed_persons:
            frame_candidates.append({
                "class_name": "person",
                "class_id": 0,
                "bbox": ap.bbox,
                "conf": ap.person_detection.confidence,
                "is_armed": True,
            })

        for d in other_detections:
            is_mil = (d.class_id == 3 or "military" in d.class_name.lower())
            c_name = "military_vehicle" if is_mil else "vehicle"
            c_id = 3 if is_mil else 2
            frame_candidates.append({
                "class_name": c_name,
                "class_id": c_id,
                "bbox": d.bbox,
                "conf": d.confidence,
                "is_armed": False,
            })

        for w_det in unattached_weapons:
            frame_candidates.append({
                "class_name": "weapon",
                "class_id": 1,
                "bbox": w_det.bbox,
                "conf": w_det.confidence,
                "is_armed": False,
            })

        self.peak_concurrent = max(self.peak_concurrent, len(frame_candidates))

        # Match candidates to active tracks of same class category
        matched_track_ids = set()
        active_render_boxes = []

        for cand in frame_candidates:
            best_track = None
            best_iou = self.iou_thresh

            for t in self.active_tracks:
                if t.track_id in matched_track_ids:
                    continue
                # Same general category
                cat_match = (
                    cand["class_name"] == t.class_name
                    or ("vehicle" in cand["class_name"] and "vehicle" in t.class_name)
                )
                if not cat_match:
                    continue

                iou = compute_box_iou(cand["bbox"], t.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = t

            if best_track is not None:
                # Update existing track
                best_track.bbox = cand["bbox"]
                best_track.confidence = cand["conf"]
                best_track.max_confidence = max(best_track.max_confidence, cand["conf"])
                if cand["is_armed"]:
                    best_track.is_armed = True
                if cand["class_name"] == "military_vehicle":
                    best_track.class_name = "military_vehicle"
                    best_track.class_id = 3
                best_track.hits += 1
                best_track.time_since_update = 0
                matched_track_ids.add(best_track.track_id)

                if best_track.hits >= self.min_hits_to_confirm:
                    self.all_confirmed_tracks[best_track.track_id] = best_track

                active_render_boxes.append((
                    best_track.track_id,
                    best_track.class_name,
                    best_track.bbox,
                    best_track.confidence,
                    best_track.is_armed,
                ))
            else:
                # Initialize new track
                new_t = Track(
                    track_id=self.next_id,
                    class_name=cand["class_name"],
                    class_id=cand["class_id"],
                    bbox=cand["bbox"],
                    confidence=cand["conf"],
                    max_confidence=cand["conf"],
                    is_armed=cand["is_armed"],
                    hits=1,
                    age=0,
                    time_since_update=0,
                )
                self.next_id += 1
                self.active_tracks.append(new_t)
                matched_track_ids.add(new_t.track_id)

                # If first frame or strong detection, confirm immediately
                if new_t.confidence >= 0.40:
                    self.all_confirmed_tracks[new_t.track_id] = new_t

                active_render_boxes.append((
                    new_t.track_id,
                    new_t.class_name,
                    new_t.bbox,
                    new_t.confidence,
                    new_t.is_armed,
                ))

        # Age unmatched tracks
        surviving_tracks = []
        for t in self.active_tracks:
            if t.track_id not in matched_track_ids:
                t.time_since_update += 1
            t.age += 1
            if t.time_since_update <= self.max_age:
                surviving_tracks.append(t)
        self.active_tracks = surviving_tracks

        return active_render_boxes

    def get_summary(self) -> Dict[str, Any]:
        """Calculates final deduplicated entity counts across the entire video."""
        confirmed = list(self.all_confirmed_tracks.values())

        # If very few confirmed (e.g. short 1-second video), fall back to any track with hits >= 1
        if len(confirmed) == 0:
            confirmed = self.active_tracks

        persons = [t for t in confirmed if t.class_id == 0]
        armed_persons = [t for t in persons if t.is_armed]
        weapons = [t for t in confirmed if t.class_id == 1 or "weapon" in t.class_name]
        vehicles = [t for t in confirmed if t.class_id == 2]
        military = [t for t in confirmed if t.class_id == 3]

        # Total weapons includes weapons carried by armed persons + unattached weapons
        total_unique_weapons = len(weapons) + len(armed_persons)

        return {
            "persons": len(persons),
            "armed_persons": len(armed_persons),
            "weapons": total_unique_weapons,
            "vehicles": len(vehicles),
            "military_vehicles": len(military),
            "total_entities": len(persons) + total_unique_weapons + len(vehicles) + len(military),
            "peak_concurrent": self.peak_concurrent,
            "tracks": [t.to_dict() for t in confirmed],
        }

