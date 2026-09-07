"""
SmartVisionAI - Pure stdlib FastAPI HTTP Endpoint Tests
Uses uvicorn in a daemon thread + urllib.request (zero extra dependencies).
"""
import sys
import time
import threading
import json
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(r"D:\SmartVisionAI_New")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend.app import app

PORT = 8009
BASE_URL = f"http://127.0.0.1:{PORT}"

def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

def http_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")

def http_post_multipart(path, filepath):
    url = f"{BASE_URL}{path}"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{Path(filepath).name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")

def main():
    print("=" * 65)
    print("STARTING TEST SERVER ON PORT 8009...")
    print("=" * 65)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    print("Waiting for server to be ready...")
    ready = False
    for i in range(45):
        try:
            status, _ = http_get("/health")
            if status == 200:
                print(f"Server is ready after {i+1}s!")
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)
    
    assert ready, "Server failed to start within 45 seconds"

    # 1. GET /
    status, data = http_get("/")
    print(f"[TEST 1] GET / -> {status}")
    assert status == 200 and data.get("success") is True, f"Failed: {data}"
    print(f"         Root response: project={data.get('project')}")

    # 2. GET /health
    status, data = http_get("/health")
    print(f"[TEST 2] GET /health -> {status}")
    assert status == 200 and data.get("status") == "healthy", f"Failed: {data}"

    # 3. GET /system/status
    status, data = http_get("/system/status")
    print(f"[TEST 3] GET /system/status -> {status}")
    assert status == 200 and data.get("status") == "ready", f"Failed: {data}"

    # 4. GET /detect/status
    status, data = http_get("/detect/status")
    print(f"[TEST 4] GET /detect/status -> {status}")
    assert status == 200 and data.get("success") is True, f"Failed: {data}"
    for f in ["active_backend", "requested_backend", "yolo_backend", "depth_backend", "models_loaded", "warmup_status"]:
        assert f in data, f"Missing field: {f}"
    print(f"         active_backend   : {data['active_backend']}")
    print(f"         requested_backend: {data['requested_backend']}")
    print(f"         yolo_backend     : {data['yolo_backend']}")
    print(f"         depth_backend    : {data['depth_backend']}")
    print(f"         models_loaded    : {data['models_loaded']}")
    print(f"         warmup_status    : {data['warmup_status']}")

    # 5. POST /detect
    test_img = PROJECT_ROOT / "backend" / "test.jpg"
    print(f"[TEST 5] POST /detect (uploading test.jpg)...")
    status, data = http_post_multipart("/detect", str(test_img))
    print(f"         Status Code: {status}")
    assert status == 200 and data.get("success") is True, f"Failed: {data}"
    print(f"         Objects count : {len(data.get('objects', []))}")
    print(f"         Action        : {data.get('navigation', {}).get('action')}")
    print(f"         Safety Level  : {data.get('safety_level')}")
    print(f"         Breakdown (ms): {data.get('processing_breakdown_ms')}")

    # 6. GET/POST /detect-video (MUST 404)
    print(f"[TEST 6] Verifying /detect-video returns 404...")
    status_get, _ = http_get("/detect-video")
    print(f"         GET  /detect-video -> {status_get}")
    assert status_get == 404
    status_post, _ = http_post_multipart("/detect-video", str(test_img))
    print(f"         POST /detect-video -> {status_post}")
    assert status_post == 404

    print("=" * 65)
    print("ALL API ENDPOINT VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
