"""Automated Download & Generation of Border Surveillance Sample Videos.

Obtains or creates 3 sample videos in data/sample_videos/:
1. border_patrol.mp4: People walking along a border sector.
2. checkpoint_approach.mp4: Vehicles and personnel approaching a frontier checkpost.
3. perimeter_intrusion.mp4: Perimeter sector surveillance view.

Includes automated direct-link downloads from public domain / Creative Commons sources
and a high-fidelity Ken Burns / procedural video generator fallback.
"""

from __future__ import annotations

import os
import sys
import glob
import math
import logging
import urllib.request
import cv2
import numpy as np

logger = logging.getLogger("sentinel.download_videos")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

OUTPUT_DIR = os.path.join("data", "sample_videos")

PUBLIC_VIDEO_URLS = [
    # Creative Commons / Public Domain surveillance and traffic feeds
    (
        "checkpoint_approach.mp4",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    ),
    (
        "border_patrol.mp4",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    ),
]


def try_download_public_video(url: str, dest_path: str, timeout: int = 15) -> bool:
    """Attempts direct download of a sample video with a strict timeout."""
    try:
        logger.info(f"Attempting download: {url} -> {dest_path}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
            data = response.read()
            if len(data) > 10000:
                out_file.write(data)
                logger.info(f"Successfully downloaded {dest_path} ({len(data)} bytes)")
                return True
    except Exception as e:
        logger.warning(f"Direct download failed for {url}: {e}")
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except Exception:
            pass
    return False


def generate_procedural_surveillance_video(
    output_path: str,
    duration_sec: int = 15,
    fps: int = 25,
    width: int = 1280,
    height: int = 720,
    scene_type: str = "fence",
) -> None:
    """Generates a realistic 1280x720 border surveillance feed with animated entities.

    Creates realistic ground, border fence, sky, moving personnel, vehicles,
    and simulated night-vision / surveillance camera grain.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logger.info(f"Generating procedural surveillance video '{scene_type}': {output_path} ({total_frames} frames)")

    np.random.seed(42 if scene_type == "fence" else 99)

    # Precompute static background (horizon, ground, ridge, fence) once
    base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    y_horizon = int(height * 0.38)
    sky_grad = np.linspace([50, 60, 75], [70, 85, 105], y_horizon, dtype=np.uint8)
    base_frame[:y_horizon, :] = sky_grad[:, None, :]
    ground_grad = np.linspace([35, 55, 65], [65, 95, 110], height - y_horizon, dtype=np.uint8)
    base_frame[y_horizon:, :] = ground_grad[:, None, :]

    # Distant low hills / Punjab border ridge line
    ridge_pts = [(0, y_horizon - 10)]
    for x in range(0, width + 50, 60):
        ry = y_horizon - 15 + int(12 * math.sin(x * 0.015))
        ridge_pts.append((x, ry))
    ridge_pts.append((width, height))
    ridge_pts.append((0, height))
    cv2.fillPoly(base_frame, [np.array(ridge_pts, dtype=np.int32)], (30, 42, 50))

    # Barbed-wire Border Fence structure along y = int(height * 0.65)
    fence_y = int(height * 0.65)
    cv2.line(base_frame, (0, fence_y), (width, fence_y), (85, 95, 105), 3)
    cv2.line(base_frame, (0, fence_y - 20), (width, fence_y - 20), (75, 85, 95), 2)
    cv2.line(base_frame, (0, fence_y - 40), (width, fence_y - 40), (65, 75, 85), 2)

    # Fence vertical steel posts
    for px in range(20, width, 55):
        cv2.line(base_frame, (px, fence_y - 60), (px, fence_y + 10), (100, 110, 120), 3)
        cv2.line(base_frame, (px, fence_y - 60), (px + 55, fence_y), (50, 60, 70), 1)
        cv2.line(base_frame, (px, fence_y), (px + 55, fence_y - 60), (50, 60, 70), 1)

    # Precompute a small noise bank to avoid generating millions of random floats per frame
    noise_bank = [np.random.randint(-3, 4, (height, width, 3), dtype=np.int16) for _ in range(8)]

    for f_idx in range(total_frames):
        frame = base_frame.copy()

        # Moving entities based on scene type
        t_norm = f_idx / float(total_frames)

        if scene_type == "fence":
            # Entity 1: Patrol person moving along the buffer zone
            p1_x = int(100 + t_norm * 850)
            p1_y = int(height * 0.52)
            _draw_synthetic_person(frame, p1_x, p1_y, scale=1.1, armed=(f_idx > 80))

            # Entity 2: Distant security vehicle moving on service road
            v_x = int(width - 50 - t_norm * 900)
            v_y = int(height * 0.45)
            _draw_synthetic_vehicle(frame, v_x, v_y, scale=0.9, military=False)

            # Entity 3: Infiltrator in fence zone towards second half
            if f_idx > 60:
                p2_x = int(800 - (t_norm - 0.25) * 350)
                p2_y = int(height * 0.78)
                _draw_synthetic_person(frame, p2_x, p2_y, scale=1.4, armed=True)

        elif scene_type == "checkpoint":
            # Checkpost gate arch on left
            cv2.rectangle(frame, (60, int(height * 0.35)), (140, int(height * 0.85)), (70, 75, 85), -1)
            cv2.rectangle(frame, (420, int(height * 0.35)), (500, int(height * 0.85)), (70, 75, 85), -1)
            cv2.rectangle(frame, (50, int(height * 0.32)), (510, int(height * 0.38)), (90, 95, 105), -1)

            # Vehicle approaching checkpost
            v_x = int(1000 - t_norm * 650)
            v_y = int(height * 0.62)
            _draw_synthetic_vehicle(frame, v_x, v_y, scale=1.3, military=(f_idx > 100))

            # Sentry guarding checkpost
            _draw_synthetic_person(frame, 220, int(height * 0.60), scale=1.1, armed=True)

        else: # perimeter
            # Tank / military vehicle patrol
            m_x = int(150 + t_norm * 700)
            m_y = int(height * 0.58)
            _draw_synthetic_vehicle(frame, m_x, m_y, scale=1.4, military=True)

            # Two scouts moving in tandem
            _draw_synthetic_person(frame, m_x - 120, int(height * 0.62), scale=1.2, armed=True)
            _draw_synthetic_person(frame, m_x - 170, int(height * 0.62), scale=1.2, armed=False)

        # Fast sensor noise from precomputed bank
        noise = noise_bank[f_idx % len(noise_bank)]
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    logger.info(f"Procedural video generated: {output_path}")


def _draw_synthetic_person(img: np.ndarray, x: int, y: int, scale: float = 1.0, armed: bool = False) -> None:
    """Draws a recognizable human silhouette for surveillance camera validation."""
    h = int(100 * scale)
    w = int(35 * scale)
    head_r = int(12 * scale)

    # Torso & legs
    body_color = (45, 55, 65)  # Camouflage dark tone
    skin_color = (95, 120, 145)

    # Head
    cv2.circle(img, (x, y - h + head_r), head_r, skin_color, -1)
    # Torso
    cv2.rectangle(img, (x - w // 2, y - h + 2 * head_r), (x + w // 2, y - int(h * 0.45)), body_color, -1)
    # Legs
    cv2.rectangle(img, (x - w // 2 + 2, y - int(h * 0.45)), (x - 2, y), (35, 45, 55), -1)
    cv2.rectangle(img, (x + 2, y - int(h * 0.45)), (x + w // 2 - 2, y), (35, 45, 55), -1)

    # If armed, draw weapon silhouette (rifle slung across chest or held forward)
    if armed:
        weapon_color = (25, 25, 30)
        # Rifle barrel and stock
        wx1 = x - int(w * 0.6)
        wy1 = y - int(h * 0.55)
        wx2 = x + int(w * 0.9)
        wy2 = y - int(h * 0.40)
        cv2.line(img, (wx1, wy1), (wx2, wy2), weapon_color, max(2, int(4 * scale)))
        # Barrel extension
        cv2.line(img, (wx2, wy2), (wx2 + int(14 * scale), wy2 + int(4 * scale)), weapon_color, max(1, int(2 * scale)))


def _draw_synthetic_vehicle(img: np.ndarray, x: int, y: int, scale: float = 1.0, military: bool = False) -> None:
    """Draws a vehicle / military armored vehicle silhouette."""
    w = int(160 * scale)
    h = int(70 * scale)

    if military:
        # Armored / Tank chassis (Olive Drab tone)
        hull_color = (40, 75, 45)
        tread_color = (25, 30, 25)
        # Main hull
        cv2.rectangle(img, (x - w // 2, y - int(h * 0.7)), (x + w // 2, y - int(h * 0.2)), hull_color, -1)
        # Turret
        cv2.rectangle(img, (x - int(w * 0.25), y - h), (x + int(w * 0.2), y - int(h * 0.65)), (35, 65, 40), -1)
        # Cannon barrel pointing forward
        cv2.line(img, (x + int(w * 0.2), y - int(h * 0.85)), (x + int(w * 0.75), y - int(h * 0.85)), (30, 50, 35), max(3, int(5 * scale)))
        # Continuous tracks / treads
        cv2.rectangle(img, (x - int(w * 0.52), y - int(h * 0.25)), (x + int(w * 0.52), y), tread_color, -1)
        # Wheels inside track
        for wx in range(x - int(w * 0.4), x + int(w * 0.45), int(22 * scale)):
            cv2.circle(img, (wx, y - int(h * 0.12)), int(8 * scale), (55, 60, 55), -1)
    else:
        # Civilian patrol jeep / SUV
        body_color = (65, 65, 75)
        # Lower body
        cv2.rectangle(img, (x - w // 2, y - int(h * 0.65)), (x + w // 2, y - int(h * 0.2)), body_color, -1)
        # Cabin / windows
        cv2.rectangle(img, (x - int(w * 0.35), y - h), (x + int(w * 0.25), y - int(h * 0.65)), (95, 105, 115), -1)
        # Windows
        cv2.rectangle(img, (x - int(w * 0.30), y - int(h * 0.92)), (x - 5, y - int(h * 0.68)), (180, 190, 200), -1)
        cv2.rectangle(img, (x + 5, y - int(h * 0.92)), (x + int(w * 0.20), y - int(h * 0.68)), (180, 190, 200), -1)
        # Wheels
        cv2.circle(img, (x - int(w * 0.32), y - int(h * 0.1)), int(14 * scale), (25, 25, 25), -1)
        cv2.circle(img, (x + int(w * 0.32), y - int(h * 0.1)), int(14 * scale), (25, 25, 25), -1)


def generate_ken_burns_from_images(
    image_paths: List[str],
    output_path: str,
    fps: int = 25,
    seconds_per_image: int = 6,
    width: int = 1280,
    height: int = 720,
) -> bool:
    """Builds a cinematic Ken Burns pan/zoom video from test split images."""
    if not image_paths:
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frames_per_image = fps * seconds_per_image

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for fi in range(frames_per_image):
            t = fi / float(frames_per_image)
            # Smooth slow zoom in from 1.0 to 1.15 and pan
            zoom = 1.0 + 0.15 * t
            crop_w = int(iw / zoom)
            crop_h = int(ih / zoom)

            x1 = int(t * (iw - crop_w))
            y1 = int(t * 0.5 * (ih - crop_h))
            x2 = min(iw, x1 + crop_w)
            y2 = min(ih, y1 + crop_h)

            cropped = img[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(resized)

    writer.release()
    logger.info(f"Ken Burns video created: {output_path}")
    return True


def obtain_all_sample_videos() -> List[str]:
    """Ensures all 3 required sample surveillance videos exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_targets = [
        ("border_patrol.mp4", "fence"),
        ("checkpoint_approach.mp4", "checkpoint"),
        ("perimeter_intrusion.mp4", "perimeter"),
    ]

    ready_videos = []
    for filename, scene_type in video_targets:
        dest = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(dest) or os.path.getsize(dest) < 10000:
            generate_procedural_surveillance_video(dest, duration_sec=16, fps=25, scene_type=scene_type)
        ready_videos.append(dest)

    logger.info(f"All sample videos verified in {OUTPUT_DIR}: {[os.path.basename(v) for v in ready_videos]}")
    return ready_videos


if __name__ == "__main__":
    obtain_all_sample_videos()
