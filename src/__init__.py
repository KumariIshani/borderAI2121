"""Sentinel Border Watch - Core Surveillance Package."""

from .detector import BorderDetector, Detection
from .armed_person_logic import link_weapons_to_persons, ArmedPersonResult
from .zone_analyzer import ZoneAnalyzer, ZoneLevel
from .alert_manager import AlertManager, AlertPriority, Alert
from .pipeline import SurveillancePipeline

__all__ = [
    "BorderDetector",
    "Detection",
    "link_weapons_to_persons",
    "ArmedPersonResult",
    "ZoneAnalyzer",
    "ZoneLevel",
    "AlertManager",
    "AlertPriority",
    "Alert",
    "SurveillancePipeline",
]
