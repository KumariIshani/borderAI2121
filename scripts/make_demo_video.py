"""Generates the Master Demo Surveillance Video for Sentinel Border Watch.

Produces:
- data/detections/demo_annotated.mp4 (30-60s H.264 video with tactical HUD, zones, and alerts)
- data/results/demo_frames/frame_01.jpg .. frame_06.jpg (6 keyframe screenshots)
"""

from __future__ import annotations

import os
import sys
import glob
import logging
import cv2
import numpy as np

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import SurveillancePipeline, reencode_to_h264
from src.detector import BorderDetector, Detection
from src.armed_person_logic import link_weapons_to_persons
from src.zone_analyzer import ZoneAnalyzer, ZoneLevel
from src.alert_manager import AlertManager, AlertPriority

logger = logging.getLogger("sentinel.make_demo_video")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def generate_master_demo_video(
    output_path: str = os.path.join("data", "detections", "demo_annotated.mp4"),
    frames_dir: str = os.path.join("data", "results", "demo_frames"),
    model_path: str = os.path.join("models", "border_yolo.pt"),
    max_duration_sec: int = 36,
) -> str:
    """Executes the pipeline over sample videos and test scenarios to generate the final demo video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Ensure sample videos are present
    from scripts.download_videos import obtain_all_sample_videos
    sample_videos = obtain_all_sample_videos()

    pipeline = SurveillancePipeline(model_path=model_path, conf_thresh=0.25)
    temp_output = output_path.replace(".mp4", "_raw.mp4")

    # We will stitch / process 3 dynamic segments:
    # 1. border_patrol.mp4 (Perimeter watch)
    # 2. checkpoint_approach.mp4 (Checkpost approach)
    # 3. perimeter_intrusion.mp4 (High-threat armored / armed intrusion)
    fps = 25.0
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    total_frames_target = int(max_duration_sec * fps)
    frames_per_video = total_frames_target // len(sample_videos)

    all_annotated_frames = []
    global_frame_idx = 0

    for vid_path in sample_videos:
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        read_count = 0

        while read_count < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            annotated, armed_res, alerts, stats = pipeline.process_frame(
                frame,
                frame_time=global_frame_idx / fps,
            )

            writer.write(annotated)
            all_annotated_frames.append(annotated)
            read_count += 1
            global_frame_idx += 1

        cap.release()

    writer.release()
    logger.info(f"Raw demo video rendered ({len(all_annotated_frames)} frames)")

    # Re-encode to H.264
    reencode_to_h264(temp_output, output_path)

    # Extract 6 screenshots evenly spaced
    total_f = len(all_annotated_frames)
    if total_f >= 6:
        indices = np.linspace(total_f * 0.1, total_f * 0.9, 6, dtype=int)
        for i, idx in enumerate(indices):
            shot_path = os.path.join(frames_dir, f"frame_{i + 1:02d}.jpg")
            cv2.imwrite(shot_path, all_annotated_frames[idx])
            logger.info(f"Saved demo keyframe: {shot_path}")

    # Verify video
    test_cap = cv2.VideoCapture(output_path)
    if test_cap.isOpened() and os.path.getsize(output_path) > 1000:
        logger.info(f"SUCCESS: Demo video verified at {output_path} ({os.path.getsize(output_path)} bytes)")
    else:
        logger.error(f"FAILURE: Demo video verification failed for {output_path}")

    return output_path


if __name__ == "__main__":
    generate_master_demo_video()
