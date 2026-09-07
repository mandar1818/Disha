import sys
import os
import io
import time
import asyncio
import statistics
from fastapi import UploadFile
from starlette.datastructures import Headers

PROJECT_ROOT = r"D:\SmartVisionAI_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.routes.detection import detect

async def benchmark_endpoint():
    test_img = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
    with open(test_img, "rb") as f:
        img_bytes = f.read()

    def make_upload_file():
        file_obj = io.BytesIO(img_bytes)
        return UploadFile(
            file=file_obj,
            size=len(img_bytes),
            filename="test.jpg",
            headers=Headers({"content-type": "image/jpeg"}),
        )

    print("[Benchmark] Executing Request 1 (Cold Start)...")
    t0 = time.perf_counter()
    r1 = await detect(file=make_upload_file())
    t1 = time.perf_counter()
    print(f"Request 1: Total Route Time={(t1-t0)*1000:.2f}ms | JSON processing_time_ms={r1.get('processing_time_ms')}")

    print("\n[Benchmark] Executing 20 Consecutive Requests...")
    internal_ms = []
    total_ms = []
    for i in range(20):
        t0 = time.perf_counter()
        r = await detect(file=make_upload_file())
        t1 = time.perf_counter()
        internal_ms.append(r.get("processing_time_ms", 0))
        total_ms.append((t1 - t0) * 1000.0)
        print(f"  Run {i+1:2d}/20: RouteTotal={total_ms[-1]:.1f}ms | processing_time_ms={internal_ms[-1]:.1f}ms")

    print("\n========================================================")
    print("ENDPOINT 20-RUN BENCHMARK SUMMARY")
    print("========================================================")
    print(f"Internal processing_time_ms (FrameProcessor):")
    print(f"  Mean:   {statistics.mean(internal_ms):.2f} ms")
    print(f"  Median: {statistics.median(internal_ms):.2f} ms")
    print(f"  Min:    {min(internal_ms):.2f} ms")
    print(f"  Max:    {max(internal_ms):.2f} ms")
    print(f"  P95:    {sorted(internal_ms)[int(0.95*len(internal_ms))]:.2f} ms")
    print(f"Total Route Time (including decode + thread dispatch):")
    print(f"  Mean:   {statistics.mean(total_ms):.2f} ms")
    print(f"  Median: {statistics.median(total_ms):.2f} ms")
    print(f"  Min:    {min(total_ms):.2f} ms")
    print(f"  Max:    {max(total_ms):.2f} ms")
    print(f"  P95:    {sorted(total_ms)[int(0.95*len(total_ms))]:.2f} ms")
    print("========================================================")

if __name__ == "__main__":
    asyncio.run(benchmark_endpoint())

