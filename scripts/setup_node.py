import os
import sys
import urllib.request
import zipfile

NODE_URL = "https://nodejs.org/dist/v20.18.0/node-v20.18.0-win-x64.zip"
TARGET_DIR = os.path.abspath("node")
ZIP_PATH = os.path.abspath("node.zip")

def install_node():
    if os.path.exists(os.path.join(TARGET_DIR, "node.exe")):
        print(f"Node already installed in {TARGET_DIR}")
        return

    print(f"Downloading portable Node.js LTS from {NODE_URL}...")
    req = urllib.request.Request(NODE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({(downloaded/total)*100:.1f}%)")

    print("Extracting Node.js zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for member in z.infolist():
            # Strip the root directory 'node-v20.18.0-win-x64/'
            parts = member.filename.split("/", 1)
            if len(parts) > 1 and parts[1]:
                target_path = os.path.join(TARGET_DIR, parts[1])
                if member.is_dir():
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with z.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[OK] Node.js extracted to {TARGET_DIR}")
    print(f"Node binary: {os.path.join(TARGET_DIR, 'node.exe')}")

if __name__ == "__main__":
    install_node()

