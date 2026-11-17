"""Zone-Based Security Level Analyzer for Sentinel Border Watch.

Partitions surveillance frames into three configurable security sectors:
1. Border Fence Zone (RESTRICTED) - Immediate zero-line / perimeter fence.
2. Buffer Zone (HIGH)             - Tactical sterile approach corridor.
3. Observation Zone (MEDIUM)      - Outer surveillance / patrol sector.

Zone classification is determined by the ground contact point (bottom-center)
of detected entities.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class ZoneLevel(str, Enum):
    RESTRICTED = "RESTRICTED"   # Red alert / Fence Zone
    HIGH = "HIGH"               # Amber alert / Buffer Corridor
    MEDIUM = "MEDIUM"           # Yellow alert / Observation Track
    SAFE = "SAFE"               # Green / Verified Friendly Base


@dataclass
class ZoneConfig:
    """Configurable boundaries for horizontal surveillance bands (normalized 0.0 - 1.0).

    Default camera perspective assumes:
    - 0.00 to 0.35: Observation Zone (distance / outer perimeter)
    - 0.35 to 0.65: Buffer Zone (tactical approach corridor)
    - 0.65 to 1.00: Border Fence Zone (immediate perimeter / zero line)
    """
    fence_min_y: float = 0.65
    fence_max_y: float = 1.00

    buffer_min_y: float = 0.35
    buffer_max_y: float = 0.65

    obs_min_y: float = 0.00
    obs_max_y: float = 0.35

    fence_name: str = "Fence Zone"
    buffer_name: str = "Buffer Zone"
    obs_name: str = "Observation Zone"


class ZoneAnalyzer:
    """Analyzes spatial locations of detections and assigns security threat zones."""

    ZONE_COLORS: Dict[ZoneLevel, Tuple[int, int, int]] = {
        ZoneLevel.RESTRICTED: (30, 30, 220),   # BGR Red
        ZoneLevel.HIGH: (0, 140, 255),         # BGR Orange
        ZoneLevel.MEDIUM: (0, 215, 255),       # BGR Amber/Yellow
        ZoneLevel.SAFE: (50, 205, 50),         # BGR Lime Green
    }

    def __init__(self, config: Optional[ZoneConfig] = None):
        self.config = config or ZoneConfig()

    def get_zone(self, bottom_center: Tuple[float, float], frame_height: int) -> Tuple[ZoneLevel, str]:
        """Determines the security zone given a bottom-center pixel coordinate."""
        _, cy = bottom_center
        norm_y = float(cy) / float(max(1, frame_height))
        norm_y = max(0.0, min(1.0, norm_y))

        if self.config.fence_min_y <= norm_y <= self.config.fence_max_y:
            return ZoneLevel.RESTRICTED, self.config.fence_name
        elif self.config.buffer_min_y <= norm_y < self.config.fence_min_y:
            return ZoneLevel.HIGH, self.config.buffer_name
        else:
            return ZoneLevel.MEDIUM, self.config.obs_name

    def draw_zone_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.18,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Renders translucent colored zone bands and boundary markers onto the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Boundaries in pixels
        y_obs_end = int(self.config.obs_max_y * h)
        y_buf_end = int(self.config.buffer_max_y * h)

        # 1. Observation Zone
        color_obs = self.ZONE_COLORS[ZoneLevel.MEDIUM]
        cv2.rectangle(overlay, (0, 0), (w, y_obs_end), color_obs, -1)

        # 2. Buffer Zone
        color_buf = self.ZONE_COLORS[ZoneLevel.HIGH]
        cv2.rectangle(overlay, (0, y_obs_end), (w, y_buf_end), color_buf, -1)

        # 3. Fence Zone (Restricted)
        color_fence = self.ZONE_COLORS[ZoneLevel.RESTRICTED]
        cv2.rectangle(overlay, (0, y_buf_end), (w, h), color_fence, -1)

        # Blend overlay
        blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

        # Draw demarcation lines
        cv2.line(blended, (0, y_obs_end), (w, y_obs_end), (0, 215, 255), 2, cv2.LINE_AA)
        cv2.line(blended, (0, y_buf_end), (w, y_buf_end), (30, 30, 240), 2, cv2.LINE_AA)

        if show_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Observation label
            cv2.putText(blended, f"[ZONE-III] {self.config.obs_name.upper()} (MEDIUM RISK)",
                        (20, min(30, max(20, y_obs_end - 10))), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            # Buffer label
            cv2.putText(blended, f"[ZONE-II] {self.config.buffer_name.upper()} (HIGH RISK)",
                        (20, y_obs_end + 30), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            # Fence label
            cv2.putText(blended, f"[ZONE-I] {self.config.fence_name.upper()} (RESTRICTED / LETHAL)",
                        (20, y_buf_end + 30), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return blended
