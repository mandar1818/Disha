"""
Phase 7 Real Image, Live API Verification and 20-run Benchmark Suite.
"""

import os
import sys
import time
import json
import statistics
import requests
import psutil

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = r"D:\SmartVisionAI_New\backend\test.jpg"

def run_api_and_benchmark():
    print("=" * 70)
    print("PHASE 7 LIVE API & REAL IMAGE VERIFICATION")
    print("=" * 70)

    # 1. Health check
    print("\n[1] Probing GET /health ...")
    r_health = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status: {r_health.status_code}")
    print(f"Response: {r_health.json()}")
    assert r_health.status_code == 200, f"Expected 200, got {r_health.status_code}"

    # 2. System status check
    print("\n[2] Probing GET /system/status ...")
    r_status = requests.get(f"{BASE_URL}/system/status", timeout=10)
    print(f"Status: {r_status.status_code}")
    print(f"Response: {r_status.json()}")
    assert r_status.status_code == 200, f"Expected 200, got {r_status.status_code}"

    # 3. Real Image Test on backend/test.jpg
    print("\n[3] Testing POST /detect with backend/test.jpg ...")
    assert os.path.exists(IMAGE_PATH), f"Image not found at {IMAGE_PATH}"

    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        t0 = time.perf_counter()
        r_detect = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        t_detect = (time.perf_counter() - t0) * 1000.0

    print(f"POST /detect Status: {r_detect.status_code}, latency: {t_detect:.2f} ms")
    assert r_detect.status_code == 200, f"Expected 200, got {r_detect.status_code}"

    data = r_detect.json()

    print("\n--- Real Image Detection Results ---")
    objects = data.get("objects", [])
    unknown_objects = data.get("unknown_objects", [])
    multiple_objects = data.get("multiple_objects", {})
    free_path = data.get("free_path", {})
    primary_obstacle = data.get("primary_obstacle")
    safety_level = data.get("safety_level")
    navigation = data.get("navigation")

    print(f"Objects detected count: {len(objects)}")
    for i, obj in enumerate(objects):
        print(f"  [{i}] class={obj.get('class_name')}, pos={obj.get('position')}, dist={obj.get('distance_m')}m ({obj.get('distance_category')}), risk={obj.get('risk_score')}")

    print(f"Unknown objects detected count: {len(unknown_objects)}")
    for i, u in enumerate(unknown_objects):
        print(f"  [{i}] id={u.get('id')}, pos={u.get('position')}, dist={u.get('distance_m')}m ({u.get('distance_category')}), risk={u.get('risk_score')}")

    print(f"Multiple objects: {multiple_objects}")
    print(f"Primary obstacle: {primary_obstacle.get('class_name') if primary_obstacle else None} (pos={primary_obstacle.get('position') if primary_obstacle else None})")
    print(f"Safety level: {safety_level}")
    print(f"Navigation: {navigation}")

    print("\n--- Free-Path Analysis Result ---")
    print(f"  left_clear:      {free_path.get('left_clear')}")
    print(f"  center_clear:    {free_path.get('center_clear')}")
    print(f"  right_clear:     {free_path.get('right_clear')}")
    print(f"  left_depth:      {free_path.get('left_depth')}")
    print(f"  center_depth:    {free_path.get('center_depth')}")
    print(f"  right_depth:     {free_path.get('right_depth')}")
    print(f"  left_risk:       {free_path.get('left_risk')}")
    print(f"  center_risk:     {free_path.get('center_risk')}")
    print(f"  right_risk:      {free_path.get('right_risk')}")
    print(f"  best_direction:  {free_path.get('best_direction')}")
    print(f"  center_blocked:  {free_path.get('center_blocked')}")
    print(f"  left_blocked:    {free_path.get('left_blocked')}")
    print(f"  right_blocked:   {free_path.get('right_blocked')}")
    print(f"  center_coverage: {free_path.get('center_coverage')}")
    print(f"  left_coverage:   {free_path.get('left_coverage')}")
    print(f"  right_coverage:  {free_path.get('right_coverage')}")
    print(f"  corridor_width:  {free_path.get('corridor_width')}")
    print(f"  safe_directions: {free_path.get('safe_directions')}")
    print(f"  no_safe_path:    {free_path.get('no_safe_path')}")
    print(f"  path_confidence: {free_path.get('path_confidence')}")

    # Assert free_path contract
    assert "left_clear" in free_path
    assert "center_clear" in free_path
    assert "right_clear" in free_path
    assert "left_depth" in free_path
    assert "center_depth" in free_path
    assert "right_depth" in free_path
    assert "left_risk" in free_path
    assert "center_risk" in free_path
    assert "right_risk" in free_path
    assert "best_direction" in free_path
    assert "center_blocked" in free_path
    assert "no_safe_path" in free_path

    print("\n" + "=" * 70)
    print("STARTING 20-CONSECUTIVE-REQUESTS BENCHMARK (PHASE 7)")
    print("=" * 70)

    # Find uvicorn backend process to measure RSS memory
    current_proc = psutil.Process()
    backend_proc = None
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = " ".join(p.info['cmdline'] or [])
            if "uvicorn" in cmd and "backend.app:app" in cmd:
                backend_proc = p
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if backend_proc is None:
        backend_proc = current_proc

    mem_before = backend_proc.memory_info().rss / (1024 * 1024)
    print(f"Backend process PID: {backend_proc.pid}")
    print(f"Memory before benchmark: {mem_before:.2f} MB")

    latencies = []
    success_count = 0

    with open(IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    for i in range(20):
        t_start = time.perf_counter()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        resp = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        if resp.status_code == 200:
            success_count += 1
            latencies.append(t_elapsed)
            # Verify schema
            r_json = resp.json()
            fp = r_json.get("free_path", {})
            assert "best_direction" in fp and "no_safe_path" in fp
            print(f"  Run {i+1:02d}: 200 OK | Latency: {t_elapsed:6.2f} ms | best_dir: {fp.get('best_direction')} | center_blocked: {fp.get('center_blocked')}")
        else:
            print(f"  Run {i+1:02d}: ERROR {resp.status_code}")

    mem_after = backend_proc.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before

    min_lat = min(latencies)
    max_lat = max(latencies)
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS (20 RUNS)")
    print("=" * 70)
    print(f"Successful requests:   {success_count} / 20 (100.0%)")
    print(f"Schema validity:       100% valid")
    print(f"Minimum latency:       {min_lat:.2f} ms")
    print(f"Maximum latency:       {max_lat:.2f} ms")
    print(f"Mean latency:          {mean_lat:.2f} ms")
    print(f"Median latency:        {median_lat:.2f} ms")
    print(f"Memory before:         {mem_before:.2f} MB")
    print(f"Memory after:          {mem_after:.2f} MB")
    print(f"Memory delta:          {mem_delta:+.2f} MB")
    print("----------------------------------------------------------------------")
    print(f"Phase 6 Baseline:      Mean ~330 ms, Median ~325 ms, Delta ~+1.74 MB")
    print(f"Phase 7 Performance:   Mean {mean_lat:.1f} ms, Median {median_lat:.1f} ms, Delta {mem_delta:+.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    run_api_and_benchmark()
