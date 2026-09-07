"""
test_phase10_api_and_benchmark.py
Phase 10 Live API Validation, Real Image Test, and 20-Run Performance Benchmark
"""

import os
import sys
import time
import json
import statistics
import requests
import psutil

BASE_URL = "http://127.0.0.1:8000"
TEST_IMG = r"D:\SmartVisionAI_New\backend\test.jpg"


def run_tests():
    print("=" * 70)
    print("PHASE 10 LIVE API & BENCHMARK VALIDATION")
    print("=" * 70)

    # 1. Health check
    print("\n--- 1. Health Check (GET /health) ---")
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    print(f"Health check status: {resp.status_code}, body: {resp.json()}")

    # 2. System status
    print("\n--- 2. System Status (GET /system/status) ---")
    resp = requests.get(f"{BASE_URL}/system/status", timeout=10)
    print(f"System status: {resp.status_code}, body: {resp.json()}")

    # 3. Real Image Test (POST /detect with backend/test.jpg)
    print("\n--- 3. Real Image Test (POST /detect with backend/test.jpg) ---")
    assert os.path.exists(TEST_IMG), f"Test image not found at: {TEST_IMG}"

    with open(TEST_IMG, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        t0 = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        t_detect = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, f"Detect failed: {resp.status_code}, text: {resp.text}"
    data = resp.json()

    print(f"HTTP Status : {resp.status_code}")
    print(f"Latency     : {t_detect:.2f} ms")
    print(f"Safety Level: {data.get('safety_level')}")
    print(f"Emergency   : {data.get('emergency_stop')}")

    nav = data.get("navigation", {})
    sg = data.get("step_guidance", {})
    fp = data.get("free_path", {})

    print(f"\nNavigation Contract (Phase 8/9/10):")
    print(f"  action             : {nav.get('action')}")
    print(f"  direction          : {nav.get('direction')}")
    print(f"  distance_m         : {nav.get('distance_m')}")
    print(f"  steps              : {nav.get('steps')}")
    print(f"  confidence         : {nav.get('confidence')}")
    print(f"  turn_required      : {nav.get('turn_required')}")
    print(f"  movement_required  : {nav.get('movement_required')}")
    print(f"  turn_angle_deg     : {nav.get('turn_angle_deg')}")
    print(f"  turn_direction     : {nav.get('turn_direction')}")
    print(f"  movement_action    : {nav.get('movement_action')}")
    print(f"  navigation_stage   : {nav.get('navigation_stage')}")
    print(f"  movement_distance_m: {nav.get('movement_distance_m')}")
    print(f"  movement_steps     : {nav.get('movement_steps')}")
    print(f"  step_confidence    : {nav.get('step_confidence')}")
    print(f"  step_reason        : {nav.get('step_reason')}")

    print(f"\nStep Guidance Contract (Phase 10):")
    print(f"  action             : {sg.get('action')}")
    print(f"  steps              : {sg.get('steps')}")
    print(f"  distance_m         : {sg.get('distance_m')}")
    print(f"  confidence         : {sg.get('confidence')}")
    print(f"  reason             : {sg.get('reason')}")

    # Real image validations
    assert nav.get("action") == "STOP", f"Expected action=STOP, got {nav.get('action')}"
    assert nav.get("navigation_stage") == "STOP", f"Expected stage=STOP, got {nav.get('navigation_stage')}"
    assert nav.get("movement_action") == "NONE", f"Expected movement_action=NONE, got {nav.get('movement_action')}"
    assert nav.get("steps") == 0, f"Expected steps=0, got {nav.get('steps')}"
    assert nav.get("movement_steps") == 0, f"Expected movement_steps=0, got {nav.get('movement_steps')}"
    assert nav.get("movement_distance_m") == 0.0, f"Expected movement_distance_m=0.0, got {nav.get('movement_distance_m')}"
    assert sg.get("action") in ("STOP", "NONE"), f"Expected sg action STOP or NONE, got {sg.get('action')}"
    assert sg.get("steps") == 0, f"Expected sg steps=0, got {sg.get('steps')}"
    print("\nREAL IMAGE TEST: PASSED! (STOP with 0 walking steps in navigation and step_guidance)")

    # 4. 20-Run Performance Benchmark
    print("\n--- 4. 20-Run Performance Benchmark ---")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    latencies = []
    with open(TEST_IMG, "rb") as f:
        img_bytes = f.read()

    for i in range(20):
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        t_start = time.perf_counter()
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        t_elapsed = (time.perf_counter() - t_start) * 1000
        assert r.status_code == 200, f"Run {i+1} failed"
        latencies.append(t_elapsed)
        print(f"  Run {i+1:2d}: {t_elapsed:.1f} ms")

    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before

    min_lat = min(latencies)
    max_lat = max(latencies)
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)

    print("\n" + "=" * 70)
    print("20-RUN BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Minimum Latency : {min_lat:.2f} ms")
    print(f"Maximum Latency : {max_lat:.2f} ms")
    print(f"Mean Latency    : {mean_lat:.2f} ms")
    print(f"Median Latency  : {median_lat:.2f} ms")
    print(f"Memory Before   : {mem_before:.2f} MB")
    print(f"Memory After    : {mem_after:.2f} MB")
    print(f"Memory Delta    : {mem_delta:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
