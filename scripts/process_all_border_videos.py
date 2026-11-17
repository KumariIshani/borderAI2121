import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
import imageio_ffmpeg
from src.pipeline import SurveillancePipeline

def process_all():
    os.makedirs("data/detections", exist_ok=True)
    pipeline = SurveillancePipeline(
        model_path="models/border_yolo.pt",
        conf_thresh=0.25
    )

    cams = [
        ("data/sample_videos/cam_001_jk.mp4", "data/detections/annotated_cam_001_jk.mp4", 150),
        ("data/sample_videos/cam_002_rajasthan.mp4", "data/detections/annotated_cam_002_rajasthan.mp4", 150),
        ("data/sample_videos/cam_003_northeast.mp4", "data/detections/annotated_cam_003_northeast.mp4", 150),
        ("data/sample_videos/cam_004_gujarat.mp4", "data/detections/annotated_cam_004_gujarat.mp4", 150),
    ]

    for in_vid, out_vid, max_f in cams:
        if os.path.exists(in_vid):
            print(f"\n==================================================")
            print(f"Processing {in_vid} -> {out_vid} (max {max_f} frames)")
            print(f"==================================================")
            summary = pipeline.run_video(
                video_source=in_vid,
                output_path=out_vid,
                max_frames=max_f,
                display=False
            )
            print(f"Done: {summary['frames_processed']} frames, {summary['total_alerts']} alerts.")
        else:
            print(f"Warning: {in_vid} not found!")

    # Master demo_annotated.mp4
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    master_target = "data/detections/demo_annotated.mp4"
    if os.path.exists("data/detections/annotated_cam_001_jk.mp4"):
        print(f"\n[*] Creating master demo video {master_target}...")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", "data/detections/annotated_cam_001_jk.mp4",
            "-c:v", "copy",
            master_target
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[OK] Master demo video ready ({os.path.getsize(master_target)} bytes)")

if __name__ == "__main__":
    process_all()
