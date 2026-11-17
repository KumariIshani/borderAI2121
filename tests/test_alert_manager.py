"""Unit tests for Alert Priority and Cooldown Logic."""

import os
import pytest
from src.detector import Detection
from src.armed_person_logic import link_weapons_to_persons, ArmedPersonResult
from src.zone_analyzer import ZoneLevel
from src.alert_manager import AlertManager, AlertPriority


def make_det(cls_id: int, cls_name: str, bbox: tuple, conf: float = 0.90) -> Detection:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return Detection(
        class_id=cls_id,
        class_name=cls_name,
        confidence=conf,
        bbox=bbox,
        center=(cx, cy),
        bottom_center=(cx, y2),
        normalized_area=((x2 - x1) * (y2 - y1)) / 1000000.0,
    )


def test_alert_priorities(tmp_path):
    csv_path = str(tmp_path / "alerts.csv")
    json_path = str(tmp_path / "alerts.json")
    mgr = AlertManager(cooldown_seconds=1.0, log_csv_path=csv_path, log_json_path=json_path)

    # 1. Armed Person in Restricted Zone -> CRITICAL
    p = make_det(0, "person", (100, 700, 200, 950))
    w = make_det(1, "weapon", (130, 800, 160, 850))
    armed_res = link_weapons_to_persons([p, w])

    alerts = mgr.evaluate_threat(
        armed_result=armed_res,
        zone_evaluator_fn=lambda pt: (ZoneLevel.RESTRICTED, "Border Fence Zone"),
        frame_time=100.0,
    )
    assert len(alerts) == 1
    assert alerts[0].priority == AlertPriority.CRITICAL
    assert alerts[0].target_type == "ARMED PERSON"

    # 2. Armed Person in Observation Zone -> HIGH
    mgr_fresh = AlertManager(cooldown_seconds=0.0, log_csv_path=csv_path, log_json_path=json_path)
    alerts_obs = mgr_fresh.evaluate_threat(
        armed_result=armed_res,
        zone_evaluator_fn=lambda pt: (ZoneLevel.MEDIUM, "Observation Zone"),
        frame_time=100.0,
    )
    assert len(alerts_obs) == 1
    assert alerts_obs[0].priority == AlertPriority.HIGH

    # 3. Military vehicle in Restricted Zone -> HIGH
    tank = make_det(3, "military_vehicle", (300, 700, 500, 900))
    tank_res = link_weapons_to_persons([tank])
    alerts_tank = mgr_fresh.evaluate_threat(
        armed_result=tank_res,
        zone_evaluator_fn=lambda pt: (ZoneLevel.RESTRICTED, "Border Fence Zone"),
        frame_time=100.0,
    )
    assert len(alerts_tank) == 1
    assert alerts_tank[0].priority == AlertPriority.HIGH

    # 4. Civilian vehicle in Restricted Zone -> MEDIUM
    car = make_det(2, "vehicle", (300, 700, 500, 900))
    car_res = link_weapons_to_persons([car])
    alerts_car = mgr_fresh.evaluate_threat(
        armed_result=car_res,
        zone_evaluator_fn=lambda pt: (ZoneLevel.RESTRICTED, "Border Fence Zone"),
        frame_time=100.0,
    )
    assert len(alerts_car) == 1
    assert alerts_car[0].priority == AlertPriority.MEDIUM


def test_cooldown_deduplication(tmp_path):
    csv_path = str(tmp_path / "alerts.csv")
    json_path = str(tmp_path / "alerts.json")
    mgr = AlertManager(cooldown_seconds=5.0, log_csv_path=csv_path, log_json_path=json_path)

    p = make_det(0, "person", (100, 700, 200, 950))
    w = make_det(1, "weapon", (130, 800, 160, 850))
    res = link_weapons_to_persons([p, w])

    # First call at t=10.0 -> alert produced
    a1 = mgr.evaluate_threat(res, lambda pt: (ZoneLevel.RESTRICTED, "Fence"), frame_time=10.0)
    assert len(a1) == 1

    # Second call at t=12.0 (within 5s cooldown) -> suppressed
    a2 = mgr.evaluate_threat(res, lambda pt: (ZoneLevel.RESTRICTED, "Fence"), frame_time=12.0)
    assert len(a2) == 0

    # Third call at t=16.0 (past 5s cooldown) -> alert produced
    a3 = mgr.evaluate_threat(res, lambda pt: (ZoneLevel.RESTRICTED, "Fence"), frame_time=16.0)
    assert len(a3) == 1
