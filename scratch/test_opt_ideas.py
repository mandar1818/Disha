import os
import sys
import time
import numpy as np
import cv2
import torch

PROJECT_ROOT = r"D:\SmartVisionAI_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.processors.frame_processor import FrameProcessor
from backend.routes.detection import decode_image

def test_optimizations():
    test_img_path = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    img = decode_image(img_bytes)
    processor = FrameProcessor()

    # Baseline 10 runs
    base_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = processor.process_frame(img)
        base_times.append((time.perf_counter() - t0) * 1000)

    print(f"Current Unoptimized Mean: {np.mean(base_times):.2f} ms | Median: {np.median(base_times):.2f} ms")

    # Test Idea 1: MiDaS with torch.inference_mode() and bilinear vs bicubic interpolation
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = processor.midas.transform(rgb).to(processor.midas.device)
    
    # Measure bicubic with no_grad
    t_bicubic = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = processor.midas.model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            _ = pred.cpu().numpy()
        t_bicubic.append((time.perf_counter() - t0) * 1000)

    # Measure bilinear with inference_mode
    t_bilinear = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = processor.midas.model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            _ = pred.cpu().numpy()
        t_bilinear.append((time.perf_counter() - t0) * 1000)

    print(f"MiDaS Bicubic (no_grad):        {np.mean(t_bicubic):.2f} ms")
    print(f"MiDaS Bilinear (inference_mode):{np.mean(t_bilinear):.2f} ms (Speedup: {np.mean(t_bicubic)-np.mean(t_bilinear):.2f} ms)")

    # Compare depth values difference between bicubic and bilinear
    with torch.no_grad():
        d_bicubic = torch.nn.functional.interpolate(pred.unsqueeze(0).unsqueeze(0) if pred.ndim==2 else pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    with torch.inference_mode():
        d_bilinear = torch.nn.functional.interpolate(pred.unsqueeze(0).unsqueeze(0) if pred.ndim==2 else pred.unsqueeze(1), size=rgb.shape[:2], mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    max_diff = np.max(np.abs(d_bicubic - d_bilinear))
    print(f"Max absolute difference in depth map: {max_diff:.6f}")

    # Test Idea 2: Remove redundant np.nanmin/max in _sample_bbox_depth
    # Test Idea 3: Downsampled morphological operations for unknown obstacle detection
    # Let's test morphology on 1408x768 vs downsampled (704x384)
    h, w = img.shape[:2]
    clean_depth = np.nan_to_num(d_bilinear, nan=1.0, posinf=1.0, neginf=0.0)
    depth_u8 = (np.clip(clean_depth, 0.0, 1.0) * 255).astype(np.uint8)
    fg_mask = (depth_u8 < int(0.55 * 255)).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    t_morph_full = []
    for _ in range(10):
        t0 = time.perf_counter()
        m1 = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernel_close)
        cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_morph_full.append((time.perf_counter() - t0) * 1000)

    # Downsampled
    scale = 0.5
    fg_small = cv2.resize(fg_mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    k_open_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    t_morph_small = []
    for _ in range(10):
        t0 = time.perf_counter()
        fg_s = cv2.resize(fg_mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        m1 = cv2.morphologyEx(fg_s, cv2.MORPH_OPEN, k_open_s)
        m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k_close_s)
        cnts_s, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_morph_small.append((time.perf_counter() - t0) * 1000)

    print(f"Morphology Full (1408x768):      {np.mean(t_morph_full):.2f} ms")
    print(f"Morphology Downsampled (704x384):{np.mean(t_morph_small):.2f} ms (Speedup: {np.mean(t_morph_full)-np.mean(t_morph_small):.2f} ms)")

if __name__ == "__main__":
    test_optimizations()

