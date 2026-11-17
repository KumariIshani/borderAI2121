"""
Fast, high-fidelity CCTV surveillance video generator for 4 border sectors:
  CAM-001: Northern Sector (Jammu & Kashmir) - Night Vision Mountain Border Fence
  CAM-002: Western Sector (Rajasthan Border) - Thermal/Monochrome Desert Barbed Wire
  CAM-003: Eastern Sector (Northeast Border) - River & Forest Patrol Road
  CAM-004: Southern Sector (Gujarat Border) - Rann of Kutch Salt Flats & Perimeter Fence
"""
import os
import cv2
import numpy as np

OUTPUT_DIR = "data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMS_DIR = "dashboard/assets/cams"

SECTORS = [
    {
        "id": "CAM-001",
        "name": "cam_001_jk.mp4",
        "base_img": "cam_001_jk.jpg",
        "fps": 25,
        "duration_sec": 8,
    },
    {
        "id": "CAM-002",
        "name": "cam_002_rajasthan.mp4",
        "base_img": "cam_002_rajasthan.jpg",
        "fps": 25,
        "duration_sec": 8,
    },
    {
        "id": "CAM-003",
        "name": "cam_003_northeast.mp4",
        "base_img": "cam_003_northeast.jpg",
        "fps": 25,
        "duration_sec": 8,
    },
    {
        "id": "CAM-004",
        "name": "cam_004_gujarat.mp4",
        "base_img": "cam_004_gujarat.jpg",
        "fps": 25,
        "duration_sec": 8,
    }
]

def generate_video(sector):
    img_path = os.path.join(CAMS_DIR, sector["base_img"])
    if not os.path.exists(img_path):
        print(f"[!] Base image {img_path} not found.")
        return
    
    base = cv2.imread(img_path)
    if base is None:
        return
    
    W, H = 1280, 720
    base = cv2.resize(base, (W, H))
    
    # Precompute scanlines mask
    scanlines = np.ones((H, 1, 1), dtype=np.float32)
    scanlines[0::2, :, :] = 0.94
    scanlines[1::2, :, :] = 1.00
    base_scan = (base.astype(np.float32) * scanlines).clip(0, 255).astype(np.uint8)
    
    out_path = os.path.join(OUTPUT_DIR, sector["name"])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    total_frames = int(sector["fps"] * sector["duration_sec"])
    writer = cv2.VideoWriter(out_path, fourcc, sector["fps"], (W, H))
    
    print(f"[*] Generating {sector['name']} ({total_frames} frames)...")
    for fi in range(total_frames):
        t = fi / float(total_frames)
        # Smooth camera pan / tilt
        dx = int(np.sin(t * 2 * np.pi) * 6)
        dy = int(np.cos(t * 2 * np.pi) * 3)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(base_scan, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
        
    writer.release()
    print(f"[+] Rendered {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    for s in SECTORS:
        generate_video(s)
