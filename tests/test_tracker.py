import pytest
from src.tracker import VideoEntityTracker, Track, calculate_match_score
from src.detector import Detection
from src.armed_person_logic import ArmedPerson


def test_match_score_overlap():
    b1 = (100.0, 100.0, 200.0, 200.0)
    b2 = (110.0, 105.0, 210.0, 205.0)
    score = calculate_match_score(b1, b2)
    assert score > 0.5


def test_match_score_center_fallback():
    # Box moved but within 1.5 body dimensions
    b1 = (100.0, 100.0, 200.0, 200.0)
    b2 = (170.0, 100.0, 270.0, 200.0)
    score = calculate_match_score(b1, b2)
    assert score > 0.05


def test_tracker_deduplication_and_counts():
    tracker = VideoEntityTracker(iou_thresh=0.25, max_age=10, min_hits_to_confirm=2)

    # Frame 1: 2 persons
    p1_f1 = Detection(0, "person", 0.9, (100.0, 100.0, 200.0, 200.0), (150.0, 150.0), (150.0, 200.0), 0.05)
    p2_f1 = Detection(0, "person", 0.85, (300.0, 100.0, 400.0, 200.0), (350.0, 150.0), (350.0, 200.0), 0.05)
    tracker.update([p1_f1, p2_f1], [], [], [])

    # Frame 2: same 2 persons slightly shifted
    p1_f2 = Detection(0, "person", 0.92, (105.0, 102.0, 205.0, 202.0), (155.0, 152.0), (155.0, 202.0), 0.05)
    p2_f2 = Detection(0, "person", 0.88, (302.0, 101.0, 402.0, 201.0), (352.0, 151.0), (352.0, 201.0), 0.05)
    boxes = tracker.update([p1_f2, p2_f2], [], [], [])

    assert len(boxes) == 2
    # Verify track IDs remained #1 and #2
    tids = {b[0] for b in boxes}
    assert tids == {1, 2}

    summary = tracker.get_summary()
    # Confirmed unique persons must be exactly 2, not 4
    assert summary["persons"] == 2
    assert summary["peak_concurrent"] == 2

