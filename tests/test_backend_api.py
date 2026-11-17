import pytest
import backend.main as bm

def test_api_health():
    data = bm.get_health()
    assert data["status"] == "online"
    assert "classes" in data
    assert "person" in data["classes"]

def test_api_cameras():
    data = bm.get_cameras()
    assert len(data) >= 4
    cam_ids = [c["id"] for c in data]
    assert "cam01" in cam_ids
    assert "cam02" in cam_ids

def test_api_stats():
    data = bm.get_stats()
    assert "kpis" in data
    assert data["kpis"]["total_detections"] > 0
    assert "trend_data" in data
    assert "class_distribution" in data

def test_api_detections_cam01():
    data = bm.get_detections("cam01")
    assert isinstance(data, list)
    assert len(data) > 0
    assert "boxes" in data[0]

def test_api_alerts():
    data = bm.get_alerts()
    assert isinstance(data, list)

def test_api_dataset_stats():
    data = bm.get_dataset_stats()
    assert "splits" in data or "class_counts" in data

def test_api_model_metrics():
    data = bm.get_model_metrics()
    assert data["architecture"] == "YOLOv8s"
    assert "per_class" in data

def test_api_model_assets():
    data = bm.get_model_assets()
    assert "weights_url" in data
    assert "results_png" in data
