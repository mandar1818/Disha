import os
import sys
import time
import numpy as np
import cv2
import torch

PROJECT_ROOT = r"D:\SmartVisionAI_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.depth.midas import MiDaSDepthEstimator

def compare_interpolation():
    img_path = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to load {img_path}"

    midas = MiDaSDepthEstimator(model_name="MiDaS_small")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = midas.transform(rgb).to(midas.device)

    # Warmup
    with torch.no_grad():
        _ = midas.model(input_batch)

    # 1. Benchmark Bicubic with no_grad
    bicubic_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            raw = midas.model(input_batch)
            pred = torch.nn.functional.interpolate(
                raw.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_bicubic = pred.cpu().numpy()
            depth_bicubic = np.nan_to_num(depth_bicubic, nan=0.0, posinf=1.0, neginf=0.0)
            mn, mx = float(depth_bicubic.min()), float(depth_bicubic.max())
            if mx - mn > 1e-8:
                depth_bicubic = (depth_bicubic - mn) / (mx - mn)
            else:
                depth_bicubic = np.zeros_like(depth_bicubic, dtype=np.float32)
        bicubic_times.append((time.perf_counter() - t0) * 1000)

    # 2. Benchmark Bilinear with inference_mode
    bilinear_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw = midas.model(input_batch)
            pred = torch.nn.functional.interpolate(
                raw.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            depth_bilinear = pred.cpu().numpy()
            depth_bilinear = np.nan_to_num(depth_bilinear, nan=0.0, posinf=1.0, neginf=0.0)
            mn, mx = float(depth_bilinear.min()), float(depth_bilinear.max())
            if mx - mn > 1e-8:
                depth_bilinear = (depth_bilinear - mn) / (mx - mn)
            else:
                depth_bilinear = np.zeros_like(depth_bilinear, dtype=np.float32)
        bilinear_times.append((time.perf_counter() - t0) * 1000)

    # Statistical comparison
    abs_diff = np.abs(depth_bicubic - depth_bilinear)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    print("========================================================")
    print("MIDAS INTERPOLATION BENCHMARK (20 runs)")
    print("========================================================")
    print(f"Bicubic (no_grad) Mean:        {np.mean(bicubic_times):.2f} ms")
    print(f"Bilinear (inference_mode) Mean:{np.mean(bilinear_times):.2f} ms")
    print(f"Latency Improvement:           {np.mean(bicubic_times) - np.mean(bilinear_times):.2f} ms")
    print(f"--------------------------------------------------------")
    print(f"Depth Statistics (Bicubic):")
    print(f"  Min: {depth_bicubic.min():.4f}, Max: {depth_bicubic.max():.4f}, Mean: {depth_bicubic.mean():.4f}, Std: {depth_bicubic.std():.4f}")
    print(f"Depth Statistics (Bilinear):")
    print(f"  Min: {depth_bilinear.min():.4f}, Max: {depth_bilinear.max():.4f}, Mean: {depth_bilinear.mean():.4f}, Std: {depth_bilinear.std():.4f}")
    print(f"--------------------------------------------------------")
    print(f"Difference Metrics:")
    print(f"  Max Absolute Difference:     {max_abs_diff:.6f}")
    print(f"  Mean Absolute Difference:    {mean_abs_diff:.6f}")
    print(f"  Relative Deviation:          {(mean_abs_diff / depth_bicubic.mean()) * 100:.3f}%")
    print("========================================================")

if __name__ == "__main__":
    compare_interpolation()

