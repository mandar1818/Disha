import sys
sys.path.insert(0, r"D:\SmartVisionAI_New")
import time
import requests
import json
import psutil
import os

def run_20_frame_benchmark():
    base_url = "http://127.0.0.1:8000"
    test_image_path = "backend/test.jpg"

    print("=" * 60)
    print("STARTING 20-RUN CONSECUTIVE API TEST (PHASE 4)")
    print("=" * 60)

    # Check health first
    r_health = requests.get(f"{base_url}/health", timeout=10)
    assert r_health.status_code == 200, "Backend not reachable"
    print("Backend confirmed healthy.")

    process = psutil.Process(os.getpid())
    # Try finding the uvicorn process for memory monitoring
    server_procs = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) if 'uvicorn' in str(p.info.get('cmdline'))]
    server_proc = server_procs[0] if server_procs else process
    mem_start_mb = server_proc.memory_info().rss / (1024 * 1024)
    print(f"Initial server memory (RSS): {mem_start_mb:.2f} MB (PID {server_proc.pid})")

    latencies = []
    failures = 0
    known_counts = []
    unknown_counts = []
    schema_failures = 0
    id_failures = 0

    required_keys = [
        "objects", "unknown_objects", "scene", "free_path", "multiple_objects",
        "navigation", "step_guidance", "primary_obstacle", "safety_level",
        "emergency_stop", "voice_instruction", "approaching", "approach_rate",
        "obstacle_stability", "emergency_alert", "processing_time_ms", "api"
    ]

    for i in range(1, 21):
        t0 = time.perf_counter()
        try:
            with open(test_image_path, "rb") as f:
                files = {"file": ("test.jpg", f, "image/jpeg")}
                res = requests.post(f"{base_url}/detect", files=files, timeout=30)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if res.status_code != 200:
                print(f"Run {i}: FAILED with status {res.status_code}")
                failures += 1
                continue

            data = res.json()
            latencies.append(elapsed_ms)

            # Schema check
            missing = [k for k in required_keys if k not in data]
            if missing:
                print(f"Run {i}: Missing keys: {missing}")
                schema_failures += 1

            k_objs = data.get("objects", [])
            u_objs = data.get("unknown_objects", [])
            known_counts.append(len(k_objs))
            unknown_counts.append(len(u_objs))

            # ID validity check
            for u in u_objs:
                if u.get("id", 0) < 1001:
                    print(f"Run {i}: Invalid unknown ID: {u.get('id')}")
                    id_failures += 1

            backend_proc_ms = data.get("processing_time_ms", 0.0)
            print(f"Run {i:02d}: {elapsed_ms:.1f} ms (backend: {backend_proc_ms:.1f} ms) | Known: {len(k_objs)} | Unknown: {len(u_objs)}")

        except Exception as exc:
            print(f"Run {i}: Exception {exc}")
            failures += 1

    mem_end_mb = server_proc.memory_info().rss / (1024 * 1024)
    mem_diff_mb = mem_end_mb - mem_start_mb

    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    min_ms = min(latencies) if latencies else 0.0
    max_ms = max(latencies) if latencies else 0.0
    avg_k = sum(known_counts) / len(known_counts) if known_counts else 0.0
    avg_u = sum(unknown_counts) / len(unknown_counts) if unknown_counts else 0.0

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (20 CONSECUTIVE RUNS)")
    print("=" * 60)
    print(f"Total Runs:                 20")
    print(f"Successful Runs:            {len(latencies)}")
    print(f"Failures:                   {failures}")
    print(f"Schema Validations Passed:  {20 - schema_failures} / 20")
    print(f"Unknown ID Checks Passed:   {20 - id_failures} / 20")
    print(f"Average Round-Trip Time:    {avg_ms:.2f} ms")
    print(f"Minimum Round-Trip Time:    {min_ms:.2f} ms")
    print(f"Maximum Round-Trip Time:    {max_ms:.2f} ms")
    print(f"Avg Known Objects / Frame:  {avg_k:.1f}")
    print(f"Avg Unknown Objects / Frame:{avg_u:.1f}")
    print(f"Server Memory Start:        {mem_start_mb:.2f} MB")
    print(f"Server Memory End:          {mem_end_mb:.2f} MB")
    print(f"Server Memory Delta:        {mem_diff_mb:+.2f} MB")
    print("=" * 60)

    # Assertions
    assert failures == 0, f"Benchmark encountered {failures} failures"
    assert schema_failures == 0, f"Schema validation failed on {schema_failures} runs"
    assert id_failures == 0, f"Unknown ID check failed on {id_failures} runs"
    assert abs(mem_diff_mb) < 50.0, f"Abnormal memory growth: {mem_diff_mb} MB"
    print(">>> 20-RUN CONSECUTIVE BENCHMARK COMPLETED AND VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    run_20_frame_benchmark()
