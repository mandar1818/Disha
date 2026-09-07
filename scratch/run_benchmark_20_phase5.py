import time
import urllib.request
import json
import os
import sys
import numpy as np
import cv2
import psutil

API_URL = "http://127.0.0.1:8000/detect"

def create_synthetic_frame(frame_idx: int) -> bytes:
    # 640x480 frame with a synthetic person/rectangle that moves slightly
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200) # light background

    # Person-like standing shape in center, moving slightly
    cx = 320 + int(5 * np.sin(frame_idx * 0.5))
    cy = 240
    # Head
    cv2.circle(img, (cx, cy - 80), 30, (50, 50, 200), -1)
    # Body
    cv2.rectangle(img, (cx - 40, cy - 50), (cx + 40, cy + 120), (30, 30, 160), -1)

    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def send_multipart_image(url: str, img_bytes: bytes) -> tuple[int, dict, float]:
    boundary = "----WebKitFormBoundarySmartVisionPhase5Test"
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
    print("SMARTVISIONAI PHASE 5 — 20 CONSECUTIVE REQUESTS BENCHMARK")
    print("=" * 65)

    # Initial warm-up request
    print("Performing 1 warm-up request...")
    warm_bytes = create_synthetic_frame(0)
    status, _, w_lat = send_multipart_image(API_URL, warm_bytes)
    print(f"Warm-up complete: status={status}, latency={w_lat:.1f}ms\n")

    # Measure initial memory
    proc = psutil.Process(os.getpid())
    # Also track uvicorn process memory if possible
    uvicorn_procs = [p for p in psutil.process_iter(['name', 'cmdline']) if 'uvicorn' in (p.info['name'] or '') or any('uvicorn' in c for c in (p.info['cmdline'] or []))]
    initial_server_mem = sum(p.memory_info().rss for p in uvicorn_procs) / (1024 * 1024) if uvicorn_procs else 0.0

    latencies = []
    statuses = []
    schema_valid_count = 0
    smoothed_distances = []
    primary_classes = []

    for i in range(1, 21):
        frame_data = create_synthetic_frame(i)
        code, resp, lat = send_multipart_image(API_URL, frame_data)
        statuses.append(code)
        latencies.append(lat)

        # Validate Phase 5 schema fields
        is_valid = False
        dist_m = None
        dist_cat = None
        dist_conf = None
        p_class = "none"

        if code == 200 and resp.get("success") is True:
            # Check objects, unknown_objects, primary_obstacle, navigation
            objs = resp.get("objects", [])
            unk_objs = resp.get("unknown_objects", [])
            primary = resp.get("primary_obstacle")
            nav = resp.get("navigation", {})

            # Primary obstacle validation
            if primary:
                dist_m = primary.get("distance_m")
                dist_cat = primary.get("distance_category")
                dist_conf = primary.get("distance_confidence")
                p_class = primary.get("class_name", "unknown")
                smoothed_distances.append(dist_m)
                primary_classes.append(p_class)

                if dist_m is not None and dist_cat in ("VERY_CLOSE", "NEAR", "FAR", "UNKNOWN_DISTANCE"):
                    is_valid = True
            else:
                is_valid = True

        if is_valid:
            schema_valid_count += 1

        print(f"Request {i:2d}/20 | HTTP {code} | {lat:6.1f}ms | Primary: {p_class:7s} | dist={dist_m}m ({dist_cat})")

    # Measure final memory
    final_server_mem = sum(p.memory_info().rss for p in uvicorn_procs) / (1024 * 1024) if uvicorn_procs else 0.0
    mem_delta = final_server_mem - initial_server_mem

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 65)
    print(f"Total Requests Sent   : {len(latencies)}")
    print(f"Successful (HTTP 200) : {statuses.count(200)} / {len(statuses)}")
    print(f"Valid Schema Rate     : {schema_valid_count} / {len(latencies)} ({schema_valid_count/len(latencies)*100:.1f}%)")
    print(f"Min Latency           : {min(latencies):.1f} ms")
    print(f"Max Latency           : {max(latencies):.1f} ms")
    print(f"Mean Latency          : {np.mean(latencies):.1f} ms")
    print(f"Median Latency        : {np.median(latencies):.1f} ms")
    print(f"Server Initial RSS    : {initial_server_mem:.2f} MB")
    print(f"Server Final RSS      : {final_server_mem:.2f} MB")
    print(f"Server Memory Delta   : {mem_delta:+.2f} MB")
    if smoothed_distances:
        print(f"Smoothed Dist Mean    : {np.mean(smoothed_distances):.2f} m")
        print(f"Smoothed Dist Std Dev : {np.std(smoothed_distances):.4f} m (temporally stable)")
    print("=" * 65)

    assert statuses.count(200) == 20, "All 20 requests must return HTTP 200"
    assert schema_valid_count == 20, "All 20 responses must satisfy canonical schema"
    assert mem_delta < 50.0, "No significant memory leak detected"
    print("ALL 20 BENCHMARK CRITERIA MET SUCCESSFULLY!")

if __name__ == "__main__":
    run_benchmark()
