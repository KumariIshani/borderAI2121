"""End-to-End Surveillance Video Inference Pipeline for Sentinel Border Watch.

Performs:
1. Frame extraction from video or camera.
2. YOLOv8 detection.
3. Armed person proximity linking.
4. Zone analysis (Restricted, High, Medium).
5. Threat evaluation & alert management.
6. Visual annotation (Translucent zones, bounding boxes, labels, tactical HUD).
7. Web-compatible H.264 MP4 encoding and JSON pipeline summary output.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

try:
    from .detector import BorderDetector, Detection
    from .armed_person_logic import link_weapons_to_persons, ArmedPersonResult
    from .zone_analyzer import ZoneAnalyzer, ZoneLevel
    from .alert_manager import AlertManager, AlertPriority, Alert
except (ImportError, ValueError):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from src.detector import BorderDetector, Detection
    from src.armed_person_logic import link_weapons_to_persons, ArmedPersonResult
    from src.zone_analyzer import ZoneAnalyzer, ZoneLevel
    from src.alert_manager import AlertManager, AlertPriority, Alert

logger = logging.getLogger("sentinel.pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Visualization Color Scheme (BGR)
COLOR_ARMED_PERSON = (30, 30, 235)       # High-viz Red
COLOR_UNATTACHED_WEAPON = (210, 0, 255)  # Magenta
COLOR_PERSON = (255, 180, 50)            # Sky Blue / Aqua
COLOR_VEHICLE = (60, 220, 60)            # Bright Green
COLOR_MILITARY = (0, 140, 255)           # High-Threat Orange


class SurveillancePipeline:
    """Integrated pipeline processing live video feeds or recorded files."""

    def __init__(
        self,
        model_path: str = os.path.join("models", "border_yolo.pt"),
        conf_thresh: float = 0.25,
        zone_analyzer: Optional[ZoneAnalyzer] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        self.detector = BorderDetector(model_path=model_path, conf_thresh=conf_thresh)
        self.zone_analyzer = zone_analyzer or ZoneAnalyzer()
        self.alert_manager = alert_manager or AlertManager()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, ArmedPersonResult, List[Alert], Dict[str, Any]]:
        """Processes a single video frame and produces the tactical visualization."""
        h, w = frame.shape[:2]
        current_time = frame_time or time.time()

        # 1. Detection
        detections = self.detector.detect(frame)

        # 2. Armed Person Linking
        armed_result = link_weapons_to_persons(detections)

        # 3. Zone Evaluator helper
        def evaluate_zone(bottom_center: Tuple[float, float]) -> Tuple[ZoneLevel, str]:
            return self.zone_analyzer.get_zone(bottom_center, frame_height=h)

        # 4. Alert Evaluation
        new_alerts = self.alert_manager.evaluate_threat(
            armed_result=armed_result,
            zone_evaluator_fn=evaluate_zone,
            frame_time=current_time,
        )

        # 5. Render Visualization
        annotated = self._render_frame(
            frame=frame,
            armed_result=armed_result,
            zone_evaluator_fn=evaluate_zone,
            active_alerts=new_alerts,
        )

        stats = {
            "total_detections": len(detections),
            "armed_persons": len(armed_result.armed_persons),
            "unarmed_persons": len(armed_result.unarmed_persons),
            "unattached_weapons": len(armed_result.unattached_weapons),
            "vehicles": sum(1 for d in armed_result.other_detections if d.class_id == 2),
            "military_vehicles": sum(1 for d in armed_result.other_detections if d.class_id == 3),
        }

        return annotated, armed_result, new_alerts, stats

    def _render_frame(
        self,
        frame: np.ndarray,
        armed_result: ArmedPersonResult,
        zone_evaluator_fn,
        active_alerts: List[Alert],
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Clean video canvas (no zone 1 2 3 overlays)
        canvas = frame.copy()

        # Draw Unarmed Persons
        for p in armed_result.unarmed_persons:
            self._draw_box(canvas, p.bbox, f"PERSON {p.confidence:.2f}", COLOR_PERSON)

        # Draw Armed Persons
        for ap in armed_result.armed_persons:
            label = f"! ARMED PERSON ({len(ap.linked_weapons)} WEAPON)"
            self._draw_box(canvas, ap.bbox, label, COLOR_ARMED_PERSON, thickness=3, fill_header=True)
            # Draw linked weapon boxes inside / near
            for w_det in ap.linked_weapons:
                self._draw_box(canvas, w_det.bbox, f"WEAPON {w_det.confidence:.2f}", COLOR_UNATTACHED_WEAPON, thickness=1)
                # Draw connecting line between person center and weapon center
                p_pt = (int(ap.person_detection.center[0]), int(ap.person_detection.center[1]))
                w_pt = (int(w_det.center[0]), int(w_det.center[1]))
                cv2.line(canvas, p_pt, w_pt, COLOR_ARMED_PERSON, 1, cv2.LINE_AA)

        # Draw Unattached Weapons
        for uw in armed_result.unattached_weapons:
            self._draw_box(canvas, uw.bbox, f"WEAPON {uw.confidence:.2f}", COLOR_UNATTACHED_WEAPON, thickness=2)

        # Draw Vehicles & Military Vehicles
        for d in armed_result.other_detections:
            if d.class_id == 3 or "military" in d.class_name.lower():
                self._draw_box(canvas, d.bbox, "MILITARY VEHICLE", COLOR_MILITARY, thickness=2)
            else:
                self._draw_box(canvas, d.bbox, "VEHICLE", COLOR_VEHICLE, thickness=2)

        # Render Top HUD Banner
        self._render_hud(canvas, armed_result, active_alerts)

        return canvas

    def _draw_box(
        self,
        img: np.ndarray,
        bbox: Tuple[float, float, float, float],
        label: str,
        color: Tuple[int, int, int],
        thickness: int = 3,
        fill_header: bool = True,
    ) -> None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return

        # 1. Main box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 2. Corner tactical brackets (5px)
        c_len = min(20, max(6, int((x2 - x1) * 0.2)), max(6, int((y2 - y1) * 0.2)))
        cv2.line(img, (x1, y1), (x1 + c_len, y1), color, 5)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), color, 5)
        cv2.line(img, (x2, y1), (x2 - c_len, y1), color, 5)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), color, 5)
        cv2.line(img, (x1, y2), (x1 + c_len, y2), color, 5)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), color, 5)
        cv2.line(img, (x2, y2), (x2 - c_len, y2), color, 5)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), color, 5)

        # 3. Solid Header Badge
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        pad = 6
        badge_h = th + pad * 2
        badge_w = tw + pad * 2
        lbl_y1 = max(0, y1 - badge_h)
        lbl_y2 = y1 if lbl_y1 > 0 else (y1 + badge_h)
        lbl_x2 = min(w, x1 + badge_w)

        cv2.rectangle(img, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        brightness = color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299
        text_color = (0, 0, 0) if brightness > 130 else (255, 255, 255)
        text_y = lbl_y2 - pad - baseline // 2
        cv2.putText(img, label, (x1 + pad, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def _render_hud(
        self,
        img: np.ndarray,
        armed_result: ArmedPersonResult,
        active_alerts: List[Alert],
    ) -> None:
        h, w = img.shape[:2]
        hud_h = 42

        # Semi-transparent top HUD bar
        hud_bar = img[0:hud_h, 0:w].copy()
        cv2.rectangle(hud_bar, (0, 0), (w, hud_h), (15, 18, 24), -1)
        cv2.addWeighted(hud_bar, 0.85, img[0:hud_h, 0:w], 0.15, 0, img[0:hud_h, 0:w])

        # Neutral orange-cyan tactical accent line at top
        cv2.line(img, (0, 0), (w // 2, 0), (20, 115, 249), 3)         # Orange accent
        cv2.line(img, (w // 2, 0), (w, 0), (238, 211, 34), 3)         # Cyan accent

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "SENTINEL // AI BORDER SURVEILLANCE"
        cv2.putText(img, title, (14, 25), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Counts
        armed_cnt = len(armed_result.armed_persons)
        person_cnt = len(armed_result.unarmed_persons)
        veh_cnt = sum(1 for d in armed_result.other_detections if d.class_id == 2)
        mil_cnt = sum(1 for d in armed_result.other_detections if d.class_id == 3)

        status_text = f"ARMED: {armed_cnt} | PERSON: {person_cnt} | VEH: {veh_cnt} | MIL: {mil_cnt}"
        (tw, _), _ = cv2.getTextSize(status_text, font, 0.48, 1)
        cv2.putText(img, status_text, (w - tw - 16, 25), font, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        # Bottom Critical Alert Banner if armed persons detected
        if armed_cnt > 0:
            alert_bar_h = 32
            alert_y = h - alert_bar_h
            bar = img[alert_y:h, 0:w].copy()
            cv2.rectangle(bar, (0, 0), (w, alert_bar_h), (20, 20, 180), -1)
            cv2.addWeighted(bar, 0.85, img[alert_y:h, 0:w], 0.15, 0, img[alert_y:h, 0:w])
            warn_msg = f"[CRITICAL THREAT] {armed_cnt} ARMED HOSTILE(S) ACTIVE IN SURVEILLANCE SECTOR"
            (mtw, _), _ = cv2.getTextSize(warn_msg, font, 0.55, 2)
            cv2.putText(img, warn_msg, (max(10, (w - mtw) // 2), h - 10), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def run_video(
        self,
        video_source: Any,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = False,
    ) -> Dict[str, Any]:
        """Runs the surveillance pipeline over a video file or stream."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {video_source} ({width}x{height} @ {fps:.1f} FPS, {total_in_video} frames)")

        writer = None
        temp_out = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            temp_out = output_path.replace(".mp4", "_temp.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

        frame_idx = 0
        t0 = time.time()
        class_counts = {"person": 0, "weapon": 0, "vehicle": 0, "military_vehicle": 0, "armed_person": 0}
        alerts_generated = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, armed_res, alerts, stats = self.process_frame(frame, frame_time=t0 + (frame_idx / fps))
            alerts_generated += len(alerts)

            class_counts["person"] += stats["unarmed_persons"]
            class_counts["armed_person"] += stats["armed_persons"]
            class_counts["vehicle"] += stats["vehicles"]
            class_counts["military_vehicle"] += stats["military_vehicles"]
            class_counts["weapon"] += stats["unattached_weapons"]

            if writer:
                writer.write(annotated)

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

            if display:
                cv2.imshow("Sentinel Border Watch", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()

        elapsed = max(0.001, time.time() - t0)
        proc_fps = frame_idx / elapsed

        # Re-encode with H.264 if possible
        if output_path and temp_out and os.path.exists(temp_out):
            reencode_to_h264(temp_out, output_path)

        summary = {
            "source": str(video_source),
            "output_video": output_path,
            "frames_processed": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(proc_fps, 2),
            "total_alerts": alerts_generated,
            "accumulated_counts": class_counts,
        }

        # Save summary
        summary_path = os.path.join("data", "results", "pipeline_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write summary: {e}")

        logger.info(f"Pipeline finished: {frame_idx} frames processed at {proc_fps:.1f} FPS.")
        return summary


def reencode_to_h264(input_path: str, output_path: str) -> None:
    """Re-encodes an mp4v video into browser-friendly H.264 format using imageio_ffmpeg or system ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23", output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_path):
            try:
                os.remove(input_path)
            except Exception:
                pass
            return
    except Exception as e:
        logger.warning(f"imageio_ffmpeg re-encoding failed: {e}")

    # Fallback: rename temp to output
    if os.path.exists(input_path) and input_path != output_path:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(input_path, output_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Sentinel Border Watch Pipeline CLI")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index for live feed")
    parser.add_argument("--model", type=str, default=os.path.join("models", "border_yolo.pt"), help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--output", type=str, default=os.path.join("data", "detections", "output.mp4"), help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    source = args.camera if args.camera is not None else args.video
    if source is None:
        # Default sample video check
        sample = os.path.join("data", "sample_videos", "border_patrol.mp4")
        if os.path.exists(sample):
            source = sample
        else:
            print("No video source provided and default sample video not found. Exiting.")
            sys.exit(1)

    out_path = args.output if args.save_video else None
    pipeline = SurveillancePipeline(model_path=args.model, conf_thresh=args.conf)
    summary = pipeline.run_video(video_source=source, output_path=out_path, max_frames=args.max_frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
