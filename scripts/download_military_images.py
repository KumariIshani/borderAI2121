"""
Download real military vehicle images (Tanks, APCs) from Hugging Face DVIDS dataset.
Resizes large images to max dimension 1024 to save disk and memory while preserving detail.
"""
import os
import json
import urllib.request
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor

IMG_DIR = "data/raw/military/images"
os.makedirs(IMG_DIR, exist_ok=True)

with open("data/raw/military/tank_files.json") as f:
    tank_files = json.load(f)

# Take up to 35 high-yield tank/APC images
selected = tank_files[:35]

BASE_URL = "https://huggingface.co/datasets/llama-farm/military-labeled-yolo/resolve/main/train/images/"

def download_and_optimize(fname):
    dest = os.path.join(IMG_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return fname, True, "cached"
    
    url = BASE_URL + fname
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        
        # Optimize image with PIL
        img = Image.open(io.BytesIO(data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        max_dim = 1024
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        img.save(dest, "JPEG", quality=85)
        return fname, True, f"saved ({os.path.getsize(dest) // 1024} KB)"
    except Exception as e:
        return fname, False, str(e)

print(f"Starting download of {len(selected)} military images...")
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(download_and_optimize, selected))

successes = [r for r in results if r[1]]
print(f"Downloaded {len(successes)} / {len(selected)} military images.")
for r in results:
    if not r[1]:
        print(f"  Failed: {r[0]} - {r[2]}")

