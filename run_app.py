"""
SENTINEL — Unified Application Launcher
Starts both the FastAPI neural backend (Port 8000) and the React Vite frontend (Port 5173),
verifies service health, and opens the application in your default web browser.
"""

from __future__ import annotations

import os
import sys
import time
import signal
import webbrowser
import subprocess
import urllib.request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

PYTHON_EXE = sys.executable

# Add portable node to PATH if present
NODE_DIR = os.path.join(ROOT_DIR, "node")
if os.path.exists(NODE_DIR):
    os.environ["PATH"] = f"{NODE_DIR};{os.environ.get('PATH', '')}"

def is_service_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SentinelHealth/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 304)
    except Exception:
        return False

def main():
    print("=" * 65, flush=True)
    print("   SENTINEL — AI Border Surveillance System (Unified Launcher)", flush=True)
    print("=" * 65, flush=True)

    # 1. Start FastAPI Backend (Port 8000)
    print("\n[1/4] Starting FastAPI Neural Backend on port 8000...", flush=True)
    backend_log = open(os.path.join(LOGS_DIR, "backend_app.log"), "w", encoding="utf-8")
    backend_cmd = [
        PYTHON_EXE, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=ROOT_DIR,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 2. Start Vite Frontend (Port 5173)
    print("[2/4] Starting React Vite Frontend on port 5173...", flush=True)
    frontend_log = open(os.path.join(LOGS_DIR, "frontend_app.log"), "w", encoding="utf-8")
    npm_path = os.path.join(NODE_DIR, "npm.cmd")
    npm_cmd = npm_path if os.path.exists(npm_path) else ("npm.cmd" if sys.platform == "win32" else "npm")
    frontend_cmd = [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=os.path.join(ROOT_DIR, "frontend"),
        stdout=frontend_log,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 3. Wait for readiness
    print("[3/4] Waiting for services to initialize...", flush=True)
    backend_ready = False
    frontend_ready = False

    for attempt in range(30):
        time.sleep(1.0)
        if not backend_ready and is_service_ready("http://127.0.0.1:8000/api/health"):
            backend_ready = True
            print("  [OK] Backend API is live at http://127.0.0.1:8000/api/health", flush=True)

        if not frontend_ready and is_service_ready("http://127.0.0.1:5173"):
            frontend_ready = True
            print("  [OK] Frontend UI is live at http://localhost:5173", flush=True)

        if backend_ready and frontend_ready:
            break

    if not (backend_ready and frontend_ready):
        print("  [NOTE] Services initializing in background. Connecting...", flush=True)

    # 4. Open in default browser
    app_url = "http://localhost:5173"
    print(f"\n[4/4] Opening application in browser: {app_url}", flush=True)
    try:
        webbrowser.open(app_url)
    except Exception as e:
        print(f"  Note: Open {app_url} in your browser manually ({e})", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("   SENTINEL is running! Press Ctrl+C in this terminal to stop.", flush=True)
    print("   Backend API:   http://127.0.0.1:8000 (Swagger docs at /docs)", flush=True)
    print("   Web Dashboard: http://localhost:5173", flush=True)
    print("=" * 65 + "\n", flush=True)

    def handle_exit(sig=None, frame=None):
        print("\nStopping services...", flush=True)
        try:
            backend_proc.terminate()
            frontend_proc.terminate()
            backend_log.close()
            frontend_log.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_exit)

    # Keep alive
    try:
        while True:
            time.sleep(1.0)
            if backend_proc.poll() is not None:
                print("\n[ERROR] Backend process terminated unexpectedly.", flush=True)
                break
            if frontend_proc.poll() is not None:
                print("\n[ERROR] Frontend process terminated unexpectedly.", flush=True)
                break
    except KeyboardInterrupt:
        handle_exit()

if __name__ == "__main__":
    main()
