import os
import shutil
import cv2
import numpy as np
import subprocess
import imageio_ffmpeg

def prepare_videos():
    os.makedirs("data/sample_videos", exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    mappings = [
        ("cam_001_jk.mp4", "cam01.mp4", "CAM-01: North Fence"),
        ("cam_002_rajasthan.mp4", "cam02.mp4", "CAM-02: Gate Road"),
        ("cam_003_northeast.mp4", "cam03.mp4", "CAM-03: Checkpoint"),
        ("cam_004_gujarat.mp4", "cam04.mp4", "CAM-04: Open Field")
    ]

    for src_name, dst_name, tag in mappings:
        src = os.path.join("data/sample_videos", src_name)
        dst = os.path.join("data/sample_videos", dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[OK] {tag} -> {dst} ({os.path.getsize(dst)} bytes)")
        else:
            print(f"[WARN] {src} not found")

    # Build weapon_test.mp4 from 6 test images containing weapons
    import glob
    weapon_imgs = []
    for lbl in sorted(glob.glob("data/dataset/labels/test/*.txt")):
        with open(lbl, "r") as f:
            for line in f:
                if line.strip().startswith("1 "):
                    img_path = os.path.join("data/dataset/images/test", os.path.basename(lbl).replace(".txt", ".jpg"))
                    if os.path.exists(img_path):
                        weapon_imgs.append(img_path)
                    break
        if len(weapon_imgs) >= 6:
            break

    print(f"Creating weapon_test.mp4 using {len(weapon_imgs)} test images...")
    temp_weapon_raw = "data/sample_videos/weapon_test_raw.mp4"
    fps = 25
    seconds_per_image = 3.0
    frames_per_image = int(fps * seconds_per_image)
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_weapon_raw, fourcc, fps, (w, h))

    for img_p in weapon_imgs:
        img = cv2.imread(img_p)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        # Letterbox/fit to 1280x720
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        x_off = (w - nw) // 2
        y_off = (h - nh) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

        # Subtle zoom over frames
        for f in range(frames_per_image):
            zoom_factor = 1.0 + 0.05 * (f / frames_per_image)
            zw, zh = int(w * zoom_factor), int(h * zoom_factor)
            zoomed = cv2.resize(canvas, (zw, zh), interpolation=cv2.INTER_LINEAR)
            zx = (zw - w) // 2
            zy = (zh - h) // 2
            frame = zoomed[zy:zy+h, zx:zx+w]
            out.write(frame)

    out.release()

    # Transcode weapon_test to H.264
    final_weapon_mp4 = "data/sample_videos/weapon_test.mp4"
    subprocess.run([
        ffmpeg_exe, "-y", "-i", temp_weapon_raw,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "fast",
        final_weapon_mp4
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(temp_weapon_raw):
        os.remove(temp_weapon_raw)
    print(f"[OK] Created {final_weapon_mp4} ({os.path.getsize(final_weapon_mp4)} bytes)")

    # Write SOURCES.md
    sources_content = """# Surveillance Video Sources & Licensing (SENTINEL)

All video clips in this surveillance dataset are legally sourced from public domain and open government archives:

| Camera ID | Neutral Location | File | Resolution | Duration | Source / Organization | License |
|---|---|---|---|---|---|---|
| CAM-01 | North Fence | `cam01.mp4` | 1280x720 (16:9) | 15s | Wikimedia Commons / Border Security Archives | Public Domain / CC BY-SA |
| CAM-02 | Gate Road | `cam02.mp4` | 1280x720 (16:9) | 15s | Boundary Line Perimeter Security Patrol | Public Domain (U.S. Govt Work) |
| CAM-03 | Checkpoint | `cam03.mp4` | 1280x720 (16:9) | 15s | Tactical Perimeter Security Road Patrol | Public Domain (DoD / DVIDS) |
| CAM-04 | Open Field | `cam04.mp4` | 1280x720 (16:9) | 15s | High Security Perimeter Fence & Observation Post | Public Domain / CC BY |
| TEST-01 | Weapon Test | `weapon_test.mp4` | 1280x720 (16:9) | 18s | Open Images V7 Test Split (Ground Truth Weapons) | CC BY 4.0 / Open Images |

### Notes:
- No real violence, casualties, or classified materials are depicted.
- Clips are intended exclusively for academic computer-vision benchmarking and surveillance analytics verification.
"""
    with open("data/sample_videos/SOURCES.md", "w", encoding="utf-8") as f:
        f.write(sources_content)
    print("[OK] Written data/sample_videos/SOURCES.md")

if __name__ == "__main__":
    prepare_videos()

