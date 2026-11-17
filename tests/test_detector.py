"""Unit tests for BorderDetector ensemble."""

import os
import numpy as np
import pytest
from src.detector import BorderDetector, Detection


def test_detector_initialization():
    detector = BorderDetector()
    assert detector is not None
    assert detector.foundation_model is not None


def test_detector_inference_on_synthetic_frame():
    detector = BorderDetector()
    # Create empty black image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)
    # On a solid black frame, detections should be empty or valid Detection instances
    for d in detections:
        assert isinstance(d, Detection)
        assert d.class_id in [0, 1, 2, 3]

