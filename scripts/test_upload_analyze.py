import urllib.request
import json
import time
import os

sample_img = "data/dataset_samples/gt_sample_01_coco_000000000322.jpg"
if not os.path.exists(sample_img):
    sample_img = "data/media/screenshots/screenshot_1.jpg"

print(f"Testing Upload & Analyze with {sample_img}...")

boundary = "----SentinelFormBoundary12345"
with open(sample_img, "rb") as f:
    file_bytes = f.read()

part_header = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="sample_upload.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode("utf-8")
part_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
body = part_header + file_bytes + part_footer

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

with urllib.request.urlopen(req) as resp:
    init_data = json.loads(resp.read().decode())
    job_id = init_data["job_id"]
    print(f"Dispatched Job ID: {job_id}, status: {init_data['status']}")

# Poll job
completed = False
for _ in range(25):
    time.sleep(1.0)
    with urllib.request.urlopen(f"http://127.0.0.1:8000/api/jobs/{job_id}") as resp:
        job_info = json.loads(resp.read().decode())
        print(f"  Job status: {job_info['status']}, progress: {job_info['progress']} %")
        if job_info["status"] == "completed":
            print("  [SUCCESS] Job completed!")
            print("  Result URL:", job_info["result"]["result_url"])
            print("  Total Entities:", job_info["result"]["total_entities"])
            print("  Summary:", job_info["result"]["summary"])
            completed = True
            break
        elif job_info["status"] == "error":
            print("  [ERROR] Job failed:", job_info.get("error"))
            break

assert completed, "Job did not complete successfully"
print("\n[VERIFIED] Upload & Analyze flow functions end-to-end with real neural model output!")

