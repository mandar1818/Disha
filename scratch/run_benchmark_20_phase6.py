import time
import urllib.request
import json
import os
import sys
import numpy as np
import cv2
import psutil

API_URL = "http://127.0.0.1:8000/detect"

def load_or_create_frame() -> bytes:
    # Use real image if available
    img_path = r'D:\SmartVisionAI_New\backend\test.jpg'
    if os.path.exists(img_path):
        return open(img_path, 'rb').read()

    # Synthetic fallback
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    cv2.rectangle(img, (240, 150), (400, 420), (30, 30, 160), -1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def send_multipart_image(url: str, img_bytes: bytes) -> tuple[int, dict, float]:
    boundary = "----WebKitFormBoundaryPhase6Benchmark"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = (time.perf_counter() - t0) * 1000.0
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data, elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return 500, {"error": str(e)}, elapsed

def run_benchmark():
    print("=" * 65)
    print("SMARTVISIONAI PHASE 6 — 20 CONSECUTIVE REQUESTS BENCHMARK")
    print("=" * 65)

    img_bytes = load_or_create_frame()

    # Warm-up request
    print("Performing 1 warm-up request...")
    status, _, w_lat = send_multipart_image(API_URL, img_bytes)
    print(f"Warm-up complete: status={status}, latency={w_lat:.1f}ms\n")

    # Initial memory measurement
    uvicorn_procs = [p for p in psutil.process_iter(['name', 'cmdline']) if 'uvicorn' in (p.info['name'] or '') or any('uvicorn' in c for c in (p.info['cmdline'] or []))]
    initial_server_mem = sum(p.memory_info().rss for p in uvicorn_procs) / (1024 * 1024) if uvicorn_procs else 0.0

    latencies = []
    statuses = []
    schema_valid_count = 0

    for i in range(1, 21):
        code, resp, lat = send_multipart_image(API_URL, img_bytes)
        statuses.append(code)
        latencies.append(lat)

        mo = resp.get("multiple_objects")
        primary = resp.get("primary_obstacle")
        is_valid = False

        if code == 200 and resp.get("success") is True and isinstance(mo, dict):
            # Check canonical MultipleObjects fields
            required_keys = [
                "object_count", "left_obstacles", "center_obstacles", "right_obstacles",
                "human_detected", "vehicle_detected", "highest_risk_object_id", "closest_object_id"
            ]
            if all(k in mo for k in required_keys):
                is_valid = True

        if is_valid:
            schema_valid_count += 1

        p_class = primary.get("class_name", "none") if primary else "none"
        obj_cnt = mo.get("object_count", 0) if mo else 0
        c_id = mo.get("closest_object_id") if mo else None
        hr_id = mo.get("highest_risk_object_id") if mo else None
        print(f"Request {i:2d}/20 | HTTP {code} | {lat:6.1f}ms | Objects: {obj_cnt} | Closest: {c_id} | HighestRisk: {hr_id} | Primary: {p_class}")

    # Final memory measurement
    final_server_mem = sum(p.memory_info().rss for p in uvicorn_procs) / (1024 * 1024) if uvicorn_procs else 0.0
    mem_delta = final_server_mem - initial_server_mem

    print("\n" + "=" * 65)
    print("PHASE 6 BENCHMARK SUMMARY RESULTS")
    print("=" * 65)
    print(f"Total Requests Sent   : {len(latencies)}")
    print(f"Successful (HTTP 200) : {statuses.count(200)} / {len(statuses)}")
    print(f"Failures              : {len(statuses) - statuses.count(200)}")
    print(f"Valid Schema Rate     : {schema_valid_count} / {len(latencies)} ({schema_valid_count/len(latencies)*100:.1f}%)")
    print(f"Min Latency           : {min(latencies):.1f} ms")
    print(f"Max Latency           : {max(latencies):.1f} ms")
    print(f"Mean Latency          : {np.mean(latencies):.1f} ms")
    print(f"Median Latency        : {np.median(latencies):.1f} ms")
    print(f"Server Initial RSS    : {initial_server_mem:.2f} MB")
    print(f"Server Final RSS      : {final_server_mem:.2f} MB")
    print(f"Server Memory Delta   : {mem_delta:+.2f} MB")
    print("=" * 65)

    assert statuses.count(200) == 20, "All 20 requests must return HTTP 200"
    assert schema_valid_count == 20, "All 20 responses must satisfy canonical multiple_objects contract"
    assert mem_delta < 50.0, "No memory leak detected"
    print("ALL 20 PHASE 6 BENCHMARK CRITERIA MET SUCCESSFULLY!")

if __name__ == "__main__":
    run_benchmark()
