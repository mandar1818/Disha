import os
import sys
import time
import requests
import json
import statistics

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = r"D:\SmartVisionAI_New\backend\test.jpg"

def main():
    print("=" * 70)
    print("PHASE 8 LIVE API VALIDATION & 20-RUN BENCHMARK")
    print("=" * 70)

    # 1. Health check
    print("\n--- Testing GET /health ---")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status: {r.status_code}, Response: {r.json()}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    # 2. System status
    print("\n--- Testing GET /system/status ---")
    r = requests.get(f"{BASE_URL}/system/status", timeout=10)
    print(f"Status: {r.status_code}")
    assert r.status_code == 200

    # 3. Real image POST /detect
    print("\n--- Testing POST /detect with backend/test.jpg ---")
    with open(IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
    print(f"Status: {r.status_code}")
    assert r.status_code == 200, f"Detect failed: {r.text}"
    data = r.json()

    # Verify canonical contract
    assert "navigation" in data, "Missing 'navigation' field"
    nav = data["navigation"]
    print("\nLive /detect Navigation Result:")
    print(json.dumps(nav, indent=2))

    assert "action" in nav, "Missing 'action' in navigation"
    assert "direction" in nav, "Missing 'direction' in navigation"
    assert "distance_m" in nav, "Missing 'distance_m' in navigation"
    assert "steps" in nav, "Missing 'steps' in navigation"
    assert "confidence" in nav, "Missing 'confidence' in navigation"
    assert "turn_required" in nav, "Missing 'turn_required' in navigation"
    assert "movement_required" in nav, "Missing 'movement_required' in navigation"
    assert "reason" in nav, "Missing 'reason' in navigation"

    assert nav["action"] in ("GO_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "GO_BACK")
    assert nav["direction"] in ("LEFT", "CENTER", "RIGHT", "BACK", "NONE")
    assert isinstance(nav["turn_required"], bool)
    assert isinstance(nav["movement_required"], bool)
    assert isinstance(nav["reason"], str) and len(nav["reason"]) > 0

    # Check free_path and real image condition
    fp = data.get("free_path", {})
    print(f"\nFree path: no_safe_path={fp.get('no_safe_path')}, left_clear={fp.get('left_clear')}, center_clear={fp.get('center_clear')}, right_clear={fp.get('right_clear')}")
    print(f"Navigation decision: action={nav['action']}, direction={nav['direction']}, reason='{nav['reason']}'")

    if fp.get("no_safe_path"):
        assert nav["action"] == "STOP", f"Expected STOP when no_safe_path is True, got {nav['action']}"
        assert nav["turn_required"] is False, "turn_required should be False on STOP"
        assert nav["movement_required"] is False, "movement_required should be False on STOP"
        print("PASS: Real image blocked corridors correctly trigger STOP!")

    # 4. 20-run Benchmark
    print("\n--- Running 20-iteration benchmark on POST /detect ---")
    latencies = []
    for i in range(20):
        t0 = time.perf_counter()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        assert r.status_code == 200
        latencies.append(dt)
        print(f"Run {i+1:2d}: {dt:.2f} ms")

    min_l = min(latencies)
    max_l = max(latencies)
    mean_l = statistics.mean(latencies)
    median_l = statistics.median(latencies)

    print("\nBenchmark Results (20 runs):")
    print(f"  Min latency:    {min_l:.2f} ms")
    print(f"  Max latency:    {max_l:.2f} ms")
    print(f"  Mean latency:   {mean_l:.2f} ms")
    print(f"  Median latency: {median_l:.2f} ms")

    print("\n========================================================")
    print("PHASE 8 LIVE API & BENCHMARK VALIDATION COMPLETE - PASS")
    print("========================================================")

if __name__ == "__main__":
    main()
