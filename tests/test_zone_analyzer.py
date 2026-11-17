"""Unit tests for ZoneAnalyzer in Sentinel Border Watch."""

import pytest
from src.zone_analyzer import ZoneAnalyzer, ZoneConfig, ZoneLevel


def test_zone_classification_defaults():
    config = ZoneConfig(
        fence_min_y=0.65,
        fence_max_y=1.00,
        buffer_min_y=0.35,
        buffer_max_y=0.65,
        obs_min_y=0.00,
        obs_max_y=0.35,
    )
    analyzer = ZoneAnalyzer(config=config)
    frame_h = 1000

    # Observation zone (y < 350)
    lvl, name = analyzer.get_zone((500, 200), frame_height=frame_h)
    assert lvl == ZoneLevel.MEDIUM
    assert "Observation" in name

    # Buffer zone (350 <= y < 650)
    lvl, name = analyzer.get_zone((500, 500), frame_height=frame_h)
    assert lvl == ZoneLevel.HIGH
    assert "Buffer" in name

    # Border fence / restricted zone (650 <= y <= 1000)
    lvl, name = analyzer.get_zone((500, 800), frame_height=frame_h)
    assert lvl == ZoneLevel.RESTRICTED
    assert "Fence" in name


def test_zone_boundary_clamping():
    analyzer = ZoneAnalyzer()
    # Out of bounds coordinates should clamp safely
    lvl, _ = analyzer.get_zone((500, -50), frame_height=720)
    assert lvl == ZoneLevel.MEDIUM

    lvl, _ = analyzer.get_zone((500, 9999), frame_height=720)
    assert lvl == ZoneLevel.RESTRICTED
