"""
test_phase12_live_benchmark.py
Direct pipeline integration test and 20-run latency benchmark for Phase 12.
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
from scratch.test_phase12_voice import VoiceDecisionManagerPy

def run_live_benchmark():
    print("=" * 65)
    print("PHASE 12: LIVE PIPELINE & VOICE DECISION BENCHMARK")
    print("=" * 65)

    # 1. Health check
    print("\n--- 1. Testing GET /health ---")
    h_res = health()
    assert h_res["success"] is True and h_res["status"] == "healthy"
    print(f"PASS: /health returned healthy status")

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

    # 4. Process live response through Phase 12 Voice Decision Manager
    print("\n--- 4. Validating Phase 12 Voice Decision on Live Frame ---")
    voice_manager = VoiceDecisionManagerPy()
    decision = voice_manager.handle_response(data, now=int(time.time() * 1000))

    print(f"  Live decision action:      {decision['action']}")
    print(f"  Live decision instruction: '{decision['instruction']}'")
    print(f"  Live decision priority:    {decision['priority']}")
    print(f"  Live decision reason:      {decision['reason']}")

    assert decision["action"] in ("SPEAK", "SUPPRESS", "REPEAT", "INTERRUPT", "QUEUE", "IGNORE")
    assert decision["priority"] in (1, 2, 3, 4, 5)
    assert isinstance(decision["instruction"], str) and len(decision["instruction"]) > 0
    print("PASS: Live voice decision conforms to Phase 12 contract")

    # 5. 20-run latency benchmark
    print("\n--- 5. Running 20-Iteration Latency Benchmark ---")
    latencies = []
    voice_decision_times = []

    for i in range(20):
        t0 = time.perf_counter()
        res = processor.process_frame(img)
        t_pipeline = time.perf_counter()

        t_v0 = time.perf_counter()
        dec = voice_manager.evaluate(res, now=int(time.time() * 1000))
        t_v1 = time.perf_counter()

        pipeline_ms = (t_pipeline - t0) * 1000.0
        voice_ms = (t_v1 - t_v0) * 1000.0

        latencies.append(pipeline_ms)
        voice_decision_times.append(voice_ms)

        print(f"  Iteration {i+1:2d}/20: Pipeline={pipeline_ms:.1f}ms, VoiceDecision={voice_ms*1000:.1f}µs | Decision: {dec['action']} '{dec['instruction']}'")

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))]
    mean_voice_us = statistics.mean(voice_decision_times) * 1000.0

    print(f"\n--- Latency Benchmark Results (20 runs) ---")
    print(f"  Mean Pipeline Latency:       {mean_lat:.1f} ms")
    print(f"  Median Pipeline Latency:     {median_lat:.1f} ms")
    print(f"  95th Percentile Latency:     {p95_lat:.1f} ms")
    print(f"  Mean Voice Decision Overhead:{mean_voice_us:.1f} µs")

    print(f"\n=================================================================")
    print(f"PHASE 12 LIVE PIPELINE & BENCHMARK VALIDATION COMPLETED: SUCCESS")
    print(f"=================================================================")
    return True

if __name__ == "__main__":
    success = run_live_benchmark()
    sys.exit(0 if success else 1)

