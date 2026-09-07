"""
test_phase11_live_benchmark.py
Direct pipeline integration test and 20-run latency benchmark for Phase 11.
"""

import sys
import os
import time
import json
import statistics
import cv2

sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.app import health, system_status
from backend.routes.detection import get_processor

def run_live_benchmark():
    print("=" * 65)
    print("PHASE 11: LIVE PIPELINE & BENCHMARK VALIDATION")
    print("=" * 65)

    # 1. Health check
    print("\n--- 1. Testing GET /health ---")
    h_res = health()
    assert h_res["success"] is True and h_res["status"] == "healthy"
    print(f"PASS: /health returned {h_res}")

    # 2. System status
    print("\n--- 2. Testing GET /system/status ---")
    s_res = system_status()
    assert s_res["success"] is True
    print(f"PASS: /system/status returned pipeline ready")

    # 3. Live processor frame execution with backend/test.jpg
    print("\n--- 3. Testing live FrameProcessor with backend/test.jpg ---")
    img_path = r"D:\SmartVisionAI_New\backend\test.jpg"
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to load {img_path}"

    processor = get_processor()
    data = processor.process_frame(img)

    # Validate Phase 11 fields in live response
    print("\n--- 4. Validating Phase 11 Response Contract ---")
    assert "predictive_threat" in data, "Missing predictive_threat in response"
    pt = data["predictive_threat"]
    assert isinstance(pt, dict), "predictive_threat must be a dict"
    print(f"  predictive_threat: position={pt.get('position')}, motion_state={pt.get('motion_state')}, threat_level={pt.get('threat_level')}")

    required_pt_keys = [
        "obstacle_id", "class_name", "position", "distance_m", "motion_state",
        "approach_rate_mps", "time_to_collision_s", "approach_score", "threat_level",
        "is_approaching", "warning_issued"
    ]
    for k in required_pt_keys:
        assert k in pt, f"Missing '{k}' in predictive_threat: {pt}"

    assert "approaching" in data, "Missing approaching backward-compat field"
    assert "approach_rate" in data, "Missing approach_rate backward-compat field"
    assert "step_guidance" in data, "Missing step_guidance"
    assert "navigation" in data, "Missing navigation"

    print("  Detected objects count:", len(data.get("objects", [])))
    for idx, obj in enumerate(data.get("objects", [])):
        print(f"    Object {idx+1}: {obj.get('class_name')} @ {obj.get('distance_m')}m | motion={obj.get('motion_state')} | rate={obj.get('approach_rate_mps')}m/s | is_app={obj.get('is_approaching')}")
        assert "motion_state" in obj, f"Object {idx+1} missing motion_state"
        assert "approach_rate_mps" in obj, f"Object {idx+1} missing approach_rate_mps"
        assert "time_to_collision_s" in obj, f"Object {idx+1} missing time_to_collision_s"
        assert "approach_score" in obj, f"Object {idx+1} missing approach_score"
        assert "is_approaching" in obj, f"Object {idx+1} missing is_approaching"

    print("PASS: Live FrameProcessor response strictly conforms to Phase 11 contract")

    # 5. 20-run Latency & Stability Benchmark
    print("\n--- 5. Running 20-run Consecutive Live Benchmark ---")
    latencies = []
    for run in range(20):
        t0 = time.perf_counter()
        resp = processor.process_frame(img)
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies.append(lat)
        assert "predictive_threat" in resp

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    stdev_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    print(f"Benchmark Results (20 consecutive frames on CPU):")
    print(f"  Min latency:    {min_lat:.1f} ms")
    print(f"  Median latency: {median_lat:.1f} ms")
    print(f"  Mean latency:   {mean_lat:.1f} ms")
    print(f"  Max latency:    {max_lat:.1f} ms")
    print(f"  Std Dev:        {stdev_lat:.1f} ms")

    print("\n=================================================================")
    print("PHASE 11 LIVE BENCHMARK COMPLETED SUCCESSFULLY (20/20 RUNS OK)")
    print("=================================================================")

if __name__ == "__main__":
    run_live_benchmark()
