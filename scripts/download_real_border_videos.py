import os
import sys
import subprocess
import urllib.request
import imageio_ffmpeg

VIDEOS = [
    {
        "name": "cam_001_jk.mp4",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Indian_Border_Police_visit_to_Jerusalem_Envelope_area%2C_November_2018.webm",
        "desc": "Indian Border Police Patrol & Inspection",
        "start": "00:00:05",
        "duration": "15"
    },
    {
        "name": "cam_002_rajasthan.mp4",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2b/Administrative_Boundary_Line_Patrol_Aerial_Insertion_%28956589%29.webm",
        "desc": "Boundary Line Perimeter Security Patrol",
        "start": "00:00:02",
        "duration": "15"
    },
    {
        "name": "cam_003_northeast.mp4",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/16/215th_Military_Police_Detachment_conducts_joint_patrol_with_German_Military_Police_%28987175%29.webm",
        "desc": "Joint Tactical Perimeter Patrol",
        "start": "00:00:10",
        "duration": "15"
    },
    {
        "name": "cam_004_gujarat.mp4",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/19/Lithuania_Boosts_NATO%E2%80%99s_Baltics_Border_Defenses_With_Eye_on_Russia.webmhd.webm",
        "desc": "High Security Perimeter Fence & Guard Post",
        "start": "00:00:04",
        "duration": "15"
    }
]

def download_and_convert():
    os.makedirs("data/raw/videos", exist_ok=True)
    os.makedirs("data/sample_videos", exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    for item in VIDEOS:
        out_target = os.path.join("data/sample_videos", item["name"])
        raw_file = os.path.join("data/raw/videos", os.path.basename(item["url"].split("?")[0]))
        
        print(f"\n[+] Downloading {item['desc']} -> {raw_file}")
        if not os.path.exists(raw_file):
            req = urllib.request.Request(item["url"], headers={"User-Agent": "BorderAISurveillance/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(raw_file, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"    Downloaded {os.path.getsize(raw_file)} bytes")
        else:
            print(f"    Already exists ({os.path.getsize(raw_file)} bytes)")

        # Convert / extract clean 15-second H.264 mp4 clip
        print(f"[*] Transcoding to H.264: {out_target}")
        cmd = [
            ffmpeg_exe, "-y",
            "-ss", item["start"],
            "-i", raw_file,
            "-t", item["duration"],
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-an",
            out_target
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            print(f"[OK] Generated {out_target} ({os.path.getsize(out_target)} bytes)")
        else:
            print(f"[ERROR] Failed transcoding: {res.stderr.decode(errors='ignore')[:300]}")

if __name__ == "__main__":
    download_and_convert()

