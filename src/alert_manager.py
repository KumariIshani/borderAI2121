"""Prioritized Alert Management and Cooldown System for Sentinel Border Watch.

Priority Matrix:
- CRITICAL: Armed person in RESTRICTED or HIGH zone.
- HIGH:     Armed person in other zones (MEDIUM/Observation) OR Military vehicle in RESTRICTED zone.
- MEDIUM:   Unarmed person or civilian vehicle in RESTRICTED zone, or Military vehicle in Buffer/Observation zone.
- LOW:      Routine detections (civilian vehicle / unarmed person in normal buffer/observation zones).

Includes deduplication cooldown to prevent flooding logs during continuous video streaming.
"""

from __future__ import annotations

import os
import json
import csv
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

from .detector import Detection
from .armed_person_logic import ArmedPerson, ArmedPersonResult
from .zone_analyzer import ZoneLevel


class AlertPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Alert:
    """Represents a generated tactical alert event."""
    alert_id: str
    timestamp: str
    epoch_time: float
    priority: AlertPriority
    zone: str
    zone_level: ZoneLevel
    target_type: str
    description: str
    confidence: float
    bbox: List[float]
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["priority"] = self.priority.value
        d["zone_level"] = self.zone_level.value
        return d


class AlertManager:
    """Manages evaluation, priority routing, cooldown filtering, and logging of alerts."""

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        log_csv_path: str = os.path.join("data", "results", "alerts.csv"),
        log_json_path: str = os.path.join("data", "results", "alerts.json"),
    ):
        self.cooldown_seconds = cooldown_seconds
        self.log_csv_path = log_csv_path
        self.log_json_path = log_json_path
        self.alerts: List[Alert] = []
        self._last_alert_time: Dict[str, float] = {}

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
        self._init_files()

    def _init_files(self) -> None:
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "alert_id", "timestamp", "priority", "zone", "zone_level",
                    "target_type", "confidence", "description", "bbox"
                ])
        if not os.path.exists(self.log_json_path):
            with open(self.log_json_path, mode="w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def evaluate_threat(
        self,
        armed_result: ArmedPersonResult,
        zone_evaluator_fn,  # Callable[[bottom_center], Tuple[ZoneLevel, str]]
        frame_time: Optional[float] = None,
    ) -> List[Alert]:
        """Evaluates detected threats for a frame and yields new non-cooled-down alerts."""
        now = frame_time or time.time()
        time_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        generated: List[Alert] = []

        # 1. Evaluate Armed Persons
        for ap in armed_result.armed_persons:
            zone_lvl, zone_name = zone_evaluator_fn(ap.bottom_center)
            if zone_lvl in (ZoneLevel.RESTRICTED, ZoneLevel.HIGH):
                prio = AlertPriority.CRITICAL
                desc = f"RED ALERT: Armed individual breached {zone_name}! {len(ap.linked_weapons)} weapon(s) detected."
            else:
                prio = AlertPriority.HIGH
                desc = f"HIGH ALERT: Armed individual identified in {zone_name}."

            alert = self._create_alert(
                prio=prio,
                zone_lvl=zone_lvl,
                zone_name=zone_name,
                target_type="ARMED PERSON",
                desc=desc,
                conf=ap.confidence,
                bbox=list(ap.bbox),
                now=now,
                time_str=time_str,
            )
            if alert:
                generated.append(alert)

        # 2. Evaluate Unattached Weapons
        for uw in armed_result.unattached_weapons:
            zone_lvl, zone_name = zone_evaluator_fn(uw.bottom_center)
            if zone_lvl == ZoneLevel.RESTRICTED:
                prio = AlertPriority.CRITICAL
                desc = f"RED ALERT: Unattended firearm/weapon discovered in {zone_name}!"
            elif zone_lvl == ZoneLevel.HIGH:
                prio = AlertPriority.HIGH
                desc = f"HIGH ALERT: Unattended weapon identified in {zone_name}."
            else:
                prio = AlertPriority.MEDIUM
                desc = f"CAUTION: Unattended weapon in {zone_name}."

            alert = self._create_alert(
                prio=prio,
                zone_lvl=zone_lvl,
                zone_name=zone_name,
                target_type="UNATTACHED WEAPON",
                desc=desc,
                conf=uw.confidence,
                bbox=list(uw.bbox),
                now=now,
                time_str=time_str,
            )
            if alert:
                generated.append(alert)

        # 3. Evaluate Unarmed Persons
        for up in armed_result.unarmed_persons:
            zone_lvl, zone_name = zone_evaluator_fn(up.bottom_center)
            if zone_lvl == ZoneLevel.RESTRICTED:
                prio = AlertPriority.MEDIUM
                desc = f"WARNING: Unauthorized pedestrian detected in {zone_name}."
            elif zone_lvl == ZoneLevel.HIGH:
                prio = AlertPriority.LOW
                desc = f"MONITORING: Pedestrian traversing {zone_name}."
            else:
                prio = AlertPriority.LOW
                desc = f"INFO: Routine pedestrian observed in {zone_name}."

            alert = self._create_alert(
                prio=prio,
                zone_lvl=zone_lvl,
                zone_name=zone_name,
                target_type="PERSON",
                desc=desc,
                conf=up.confidence,
                bbox=list(up.bbox),
                now=now,
                time_str=time_str,
            )
            if alert:
                generated.append(alert)

        # 4. Evaluate Other Detections (Vehicles / Military Vehicles)
        for obj in armed_result.other_detections:
            zone_lvl, zone_name = zone_evaluator_fn(obj.bottom_center)
            is_military = (obj.class_id == 3 or "military" in obj.class_name.lower())

            if is_military:
                if zone_lvl == ZoneLevel.RESTRICTED:
                    prio = AlertPriority.HIGH
                    desc = f"HOSTILE THREAT: Armored/Military vehicle entered {zone_name}!"
                else:
                    prio = AlertPriority.MEDIUM
                    desc = f"ATTENTION: Armored/Military vehicle tracking in {zone_name}."
                target = "MILITARY VEHICLE"
            else:
                if zone_lvl == ZoneLevel.RESTRICTED:
                    prio = AlertPriority.MEDIUM
                    desc = f"SECURITY BREACH: Unauthorized vehicle detected in {zone_name}."
                else:
                    prio = AlertPriority.LOW
                    desc = f"ROUTINE: Vehicle transit logged in {zone_name}."
                target = "VEHICLE"

            alert = self._create_alert(
                prio=prio,
                zone_lvl=zone_lvl,
                zone_name=zone_name,
                target_type=target,
                desc=desc,
                conf=obj.confidence,
                bbox=list(obj.bbox),
                now=now,
                time_str=time_str,
            )
            if alert:
                generated.append(alert)

        if generated:
            self.persist_alerts(generated)

        return generated

    def _create_alert(
        self,
        prio: AlertPriority,
        zone_lvl: ZoneLevel,
        zone_name: str,
        target_type: str,
        desc: str,
        conf: float,
        bbox: List[float],
        now: float,
        time_str: str,
    ) -> Optional[Alert]:
        """Cooldown check: skips duplicate alerts of same type and zone within cooldown."""
        # Quantize bbox center to ~80px grid to identify distinct physical targets
        cx = int((bbox[0] + bbox[2]) / 2.0 / 80.0)
        cy = int((bbox[1] + bbox[3]) / 2.0 / 80.0)
        cooldown_key = f"{prio.value}_{target_type}_{zone_lvl.value}_{cx}_{cy}"

        last_time = self._last_alert_time.get(cooldown_key, 0.0)
        if (now - last_time) < self.cooldown_seconds:
            return None

        self._last_alert_time[cooldown_key] = now
        alert_id = f"ALT-{int(now * 1000) % 10000000:07d}"

        alert = Alert(
            alert_id=alert_id,
            timestamp=time_str,
            epoch_time=now,
            priority=prio,
            zone=zone_name,
            zone_level=zone_lvl,
            target_type=target_type,
            description=desc,
            confidence=round(conf, 4),
            bbox=[round(b, 2) for b in bbox],
        )
        self.alerts.append(alert)
        return alert

    def persist_alerts(self, new_alerts: List[Alert]) -> None:
        """Appends new alerts to CSV and JSON."""
        # Append to CSV
        with open(self.log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for a in new_alerts:
                writer.writerow([
                    a.alert_id, a.timestamp, a.priority.value, a.zone,
                    a.zone_level.value, a.target_type, a.confidence,
                    a.description, json.dumps(a.bbox)
                ])

        # Update JSON
        try:
            existing = []
            if os.path.exists(self.log_json_path):
                with open(self.log_json_path, mode="r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                    except Exception:
                        existing = []
            existing.extend([a.to_dict() for a in new_alerts])
            with open(self.log_json_path, mode="w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass
