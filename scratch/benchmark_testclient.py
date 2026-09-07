import sys
import os
import time
import statistics
from fastapi.testclient import TestClient
from backend.app import app

def benchmark():
    client = TestClient(app)
    test_img = os.path.join(r"D:\SmartVisionAI_New", "backend", "test.jpg")
    with open(test_img, "rb") as f:
        img_bytes = f.read()

    print("[Benchmark] Executing Request 1 (Cold Start)...")
    t0 = time.perf_counter()
    r1 = client.post("/detect", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    t1 = time.perf_counter()
    data1 = r1.json()
    print(f"Request 1: HTTP Total={(t1-t0)*1000:.2f}ms | JSON processing_time_ms={data1.get('processing_time_ms')}")

    print("\n[Benchmark] Executing 20 Consecutive Requests...")
    internal_ms = []
    http_ms = []
    for i in range(20):
        t0 = time.perf_counter()
        r = client.post("/detect", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        t1 = time.perf_counter()
        data = r.json()
        internal_ms.append(data.get("processing_time_ms", 0))
        http_ms.append((t1 - t0) * 1000.0)
        print(f"  Run {i+1:2d}/20: HTTP={http_ms[-1]:.1f}ms | processing_time_ms={internal_ms[-1]:.1f}ms")

    print("\n========================================================")
    print("TESTCLIENT 20-RUN BENCHMARK SUMMARY")
    print("========================================================")
    print(f"Internal processing_time_ms:")
    print(f"  Mean:   {statistics.mean(internal_ms):.2f} ms")
    print(f"  Median: {statistics.median(internal_ms):.2f} ms")
    print(f"  Min:    {min(internal_ms):.2f} ms")
    print(f"  Max:    {max(internal_ms):.2f} ms")
    print(f"  P95:    {sorted(internal_ms)[int(0.95*len(internal_ms))]:.2f} ms")
    print(f"HTTP Round-Trip:")
    print(f"  Mean:   {statistics.mean(http_ms):.2f} ms")
    print(f"  Median: {statistics.median(http_ms):.2f} ms")
    print(f"  Min:    {min(http_ms):.2f} ms")
    print(f"  Max:    {max(http_ms):.2f} ms")
    print(f"  P95:    {sorted(http_ms)[int(0.95*len(http_ms))]:.2f} ms")
    print("========================================================")

if __name__ == "__main__":
    benchmark()

