"""Unit tests for Armed Person Spatial Linking Logic."""

import pytest
from src.detector import Detection
from src.armed_person_logic import link_weapons_to_persons, ArmedPersonResult


def make_detection(cls_id: int, cls_name: str, bbox: tuple, conf: float = 0.85) -> Detection:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bottom_center = (cx, y2)
    norm_area = ((x2 - x1) * (y2 - y1)) / (1920.0 * 1080.0)
    return Detection(
        class_id=cls_id,
        class_name=cls_name,
        confidence=conf,
        bbox=bbox,
        center=(cx, cy),
        bottom_center=bottom_center,
        normalized_area=norm_area,
    )


def test_armed_person_direct_containment():
    # Person at (100, 100, 200, 400)
    person = make_detection(0, "person", (100.0, 100.0, 200.0, 400.0))
    # Weapon held at waist (140, 220, 180, 270) -> inside person box
    weapon = make_detection(1, "weapon", (140.0, 220.0, 180.0, 270.0))

    result = link_weapons_to_persons([person, weapon])

    assert len(result.armed_persons) == 1
    assert len(result.unarmed_persons) == 0
    assert len(result.unattached_weapons) == 0
    assert result.armed_persons[0].person_detection == person
    assert result.armed_persons[0].linked_weapons == [weapon]


def test_unattached_weapon_isolated():
    # Person on left side of image
    person = make_detection(0, "person", (50.0, 100.0, 150.0, 350.0))
    # Weapon dropped on right side of image far away
    weapon = make_detection(1, "weapon", (800.0, 500.0, 850.0, 550.0))

    result = link_weapons_to_persons([person, weapon])

    assert len(result.armed_persons) == 0
    assert len(result.unarmed_persons) == 1
    assert len(result.unattached_weapons) == 1
    assert result.unattached_weapons[0] == weapon


def test_multiple_persons_closest_assignment():
    # Two persons standing nearby
    p1 = make_detection(0, "person", (100.0, 100.0, 200.0, 400.0))
    p2 = make_detection(0, "person", (300.0, 100.0, 400.0, 400.0))
    # Weapon inside p2's box
    weapon = make_detection(1, "weapon", (330.0, 220.0, 370.0, 280.0))

    result = link_weapons_to_persons([p1, p2, weapon])

    assert len(result.armed_persons) == 1
    assert len(result.unarmed_persons) == 1
    assert result.armed_persons[0].person_detection == p2
    assert result.unarmed_persons[0] == p1
