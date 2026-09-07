import os
import sys
import time
import requests
import json
import statistics
import psutil

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = r"D:\SmartVisionAI_New\backend\test.jpg"

def get_process_memory():
    current_process = psutil.Process(os.getpid())
    return current_process.memory_info().rss / (1024 * 1024)

def main():
    print("=" * 70)
    print("PHASE 9 LIVE API VALIDATION, REAL IMAGE, & 20-RUN BENCHMARK")
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

    # Verify canonical contract including Phase 9 extensions
    assert "navigation" in data, "Missing 'navigation' field"
    nav = data["navigation"]
    print("\nLive /detect Navigation Result:")
    print(json.dumps(nav, indent=2))

    required_keys = [
        "action", "direction", "distance_m", "steps", "confidence",
        "turn_required", "movement_required", "reason",
        "turn_angle_deg", "turn_direction", "movement_action", "navigation_stage"
    ]
    for key in required_keys:
        assert key in nav, f"Missing required key '{key}' in navigation"

    assert nav["action"] in ("GO_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "GO_BACK")
    assert nav["direction"] in ("LEFT", "CENTER", "RIGHT", "BACK", "NONE")
    assert nav["turn_direction"] in ("LEFT", "RIGHT", "NONE")
    assert nav["movement_action"] in ("WALK_FORWARD", "WALK_LEFT", "WALK_RIGHT", "WALK_BACK", "NONE")
    assert nav["navigation_stage"] in ("TURN", "MOVE", "STOP", "IDLE")
    assert isinstance(nav["turn_required"], bool)
    assert isinstance(nav["movement_required"], bool)
    assert isinstance(nav["turn_angle_deg"], (int, float))
    assert nav["turn_angle_deg"] >= 0.0

    # Check real image condition
    fp = data.get("free_path", {})
    print(f"\nFree path: no_safe_path={fp.get('no_safe_path')}, left_clear={fp.get('left_clear')}, center_clear={fp.get('center_clear')}, right_clear={fp.get('right_clear')}")
    print(f"Navigation decision: action={nav['action']}, stage={nav['navigation_stage']}, angle={nav['turn_angle_deg']}, turn_dir={nav['turn_direction']}, mov_act={nav['movement_action']}")

    if fp.get("no_safe_path"):
        assert nav["action"] == "STOP", f"Expected STOP when no_safe_path is True, got {nav['action']}"
        assert nav["navigation_stage"] == "STOP", f"Expected navigation_stage=STOP, got {nav['navigation_stage']}"
        assert nav["turn_angle_deg"] == 0.0, f"Expected turn_angle_deg=0.0, got {nav['turn_angle_deg']}"
        assert nav["turn_direction"] == "NONE", f"Expected turn_direction=NONE, got {nav['turn_direction']}"
        assert nav["movement_action"] == "NONE", f"Expected movement_action=NONE, got {nav['movement_action']}"
        assert nav["turn_required"] is False
        assert nav["movement_required"] is False
        print("PASS: Real image blocked corridors correctly trigger stage=STOP, angle=0.0, mov_act=NONE!")

    # 4. 20-run Benchmark
    print("\n--- Running 20-iteration benchmark on POST /detect ---")
    mem_before = get_process_memory()
    latencies = []
    for i in range(20):
        t0 = time.perf_counter()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        assert r.status_code == 200
        latencies.append(dt)
        print(f"Run {i+1:2d}: {dt:.2f} ms")

    mem_after = get_process_memory()
    min_l = min(latencies)
    max_l = max(latencies)
    mean_l = statistics.mean(latencies)
    median_l = statistics.median(latencies)

    print("\nBenchmark Results (20 runs):")
    print(f"  Min latency:    {min_l:.2f} ms")
    print(f"  Max latency:    {max_l:.2f} ms")
    print(f"  Mean latency:   {mean_l:.2f} ms")
    print(f"  Median latency: {median_l:.2f} ms")
    print(f"  Memory Before:  {mem_before:.2f} MB")
    print(f"  Memory After:   {mem_after:.2f} MB")
    print(f"  Memory Delta:   {mem_after - mem_before:.2f} MB")

    print("\n========================================================")
    print("PHASE 9 LIVE API & BENCHMARK VALIDATION COMPLETE - PASS")
    print("========================================================")

if __name__ == "__main__":
    main()
