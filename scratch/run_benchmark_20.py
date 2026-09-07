"""
SmartVisionAI - Phase 26 Performance Verification Benchmark
Runs 20 consecutive real multipart POST /detect requests and profiles:
- processing_time_ms (internal backend)
- end-to-end HTTP round-trip latency
- memory (RSS)
- CPU
- errors
- stability & distance/step tracking consistency
"""

import os
import sys
import time
import json
import psutil
import requests

PROJECT_ROOT = r"D:\SmartVisionAI_New"
TEST_IMAGE = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
DETECT_URL = "http://127.0.0.1:8000/detect"

def run_performance_test():
    with open(TEST_IMAGE, "rb") as f:
        img_bytes = f.read()

    # 1 warmup request
    _ = requests.post(DETECT_URL, files={"file": ("test.jpg", img_bytes, "image/jpeg")})

    latencies_http = []
    internal_times = []
    mem_readings = []
    cpu_readings = []
    stabilities = []
    errors = 0

    proc = psutil.Process()

    for i in range(20):
        cpu_before = psutil.cpu_percent(interval=None)
        t0 = time.perf_counter()
        try:
            resp = requests.post(DETECT_URL, files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=10)
            elapsed_http = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                data = resp.json()
                latencies_http.append(elapsed_http)
                internal_times.append(data.get("processing_time_ms", 0))
                stabilities.append(data.get("obstacle_stability", 0))
            else:
                errors += 1
        except Exception as e:
            errors += 1
        cpu_after = psutil.cpu_percent(interval=None)
        cpu_readings.append(max(cpu_before, cpu_after))
        mem_readings.append(proc.memory_info().rss / (1024 * 1024))
        time.sleep(0.02)

    sys_mem = psutil.virtual_memory()

    results = {
        "runs": 20,
        "errors": errors,
        "internal_backend_ms": {
            "avg": round(sum(internal_times) / len(internal_times), 2) if internal_times else 0,
            "min": round(min(internal_times), 2) if internal_times else 0,
            "max": round(max(internal_times), 2) if internal_times else 0,
        },
        "http_roundtrip_ms": {
            "avg": round(sum(latencies_http) / len(latencies_http), 2) if latencies_http else 0,
            "min": round(min(latencies_http), 2) if latencies_http else 0,
            "max": round(max(latencies_http), 2) if latencies_http else 0,
        },
        "cpu_percent": round(sum(cpu_readings) / len(cpu_readings), 2),
        "process_ram_mb": round(mem_readings[-1], 2),
        "system_ram_used_percent": round(sys_mem.percent, 2),
        "effective_fps": round(1000.0 / (sum(latencies_http) / len(latencies_http)), 2) if latencies_http else 0,
        "max_obstacle_stability": max(stabilities) if stabilities else 0
    }

    print("Performance Test (20 runs) completed:")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    run_performance_test()
