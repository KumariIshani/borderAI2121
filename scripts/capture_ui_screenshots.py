import os
import time
import subprocess

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_DIR = os.path.abspath(r"data\results\ui_screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

PAGES = [
    ("page_01_dashboard.png", "http://localhost:5173/"),
    ("page_02_live_monitor.png", "http://localhost:5173/monitor"),
    ("page_03_upload_analyze.png", "http://localhost:5173/analyze"),
    ("page_04_alerts.png", "http://localhost:5173/alerts"),
    ("page_05_dataset.png", "http://localhost:5173/dataset"),
    ("page_06_model_performance.png", "http://localhost:5173/model"),
    ("page_07_about.png", "http://localhost:5173/about"),
]

def capture_all():
    print("Capturing all 7 page screenshots with headless browser...")
    for filename, url in PAGES:
        out_path = os.path.join(OUT_DIR, filename)
        cmd = [
            EDGE_EXE,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=1600,1000",
            f"--screenshot={out_path}",
            url
        ]
        print(f"[*] Capturing {url} -> {filename}...")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(out_path):
            print(f"    [OK] Captured ({os.path.getsize(out_path)} bytes)")
        else:
            print(f"    [WARN] Failed to capture {filename}")

if __name__ == "__main__":
    capture_all()

