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

def profile_breakdown():
    test_img_path = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    print("[Profiler] Initializing FrameProcessor...")
    processor = FrameProcessor()

    # Warmup
    img_warmup = decode_image(img_bytes)
    _ = processor.process_frame(img_warmup)

    print("\n[Profiler] Running 5 timed iterations for precise stage profiling...")
    
    decode_times = []
    yolo_prep_times = []
    yolo_infer_times = []
    yolo_parse_times = []
    midas_prep_times = []
    midas_infer_times = []
    midas_post_times = []
    unknown_times = []
    decision_times = []
    total_times = []

    for i in range(5):
        # 1. Decode
        t0 = time.perf_counter()
        img = decode_image(img_bytes)
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000)

        # 2. YOLO
        t_y_prep_0 = time.perf_counter()
        prep_img = processor.yolo._prepare_image(img)
        t_y_prep_1 = time.perf_counter()
        yolo_prep_times.append((t_y_prep_1 - t_y_prep_0) * 1000)

        t_y_inf_0 = time.perf_counter()
        y_results = processor.yolo.model.predict(
            source=prep_img,
            conf=processor.yolo.confidence,
            device=processor.yolo.device,
            verbose=False,
        )
        t_y_inf_1 = time.perf_counter()
        yolo_infer_times.append((t_y_inf_1 - t_y_inf_0) * 1000)

        t_y_parse_0 = time.perf_counter()
        objects = processor.yolo._parse_results(y_results)
        t_y_parse_1 = time.perf_counter()
        yolo_parse_times.append((t_y_parse_1 - t_y_parse_0) * 1000)

        # 3. MiDaS
        t_m_prep_0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = processor.midas.transform(rgb).to(processor.midas.device)
        t_m_prep_1 = time.perf_counter()
        midas_prep_times.append((t_m_prep_1 - t_m_prep_0) * 1000)

        t_m_inf_0 = time.perf_counter()
        with torch.no_grad():
            prediction = processor.midas.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        t_m_inf_1 = time.perf_counter()
        midas_infer_times.append((t_m_inf_1 - t_m_inf_0) * 1000)

        t_m_post_0 = time.perf_counter()
        depth = prediction.cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
        minimum = float(depth.min())
        maximum = float(depth.max())
        if maximum - minimum > 1e-8:
            depth = (depth - minimum) / (maximum - minimum)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)
        depth_map = depth.astype(np.float32)
        scene_depth = processor._depth_map_to_scene(depth_map)
        fused_objects = processor.fuse_objects_and_depth(img, objects, scene_depth, depth_map=depth_map)
        t_m_post_1 = time.perf_counter()
        midas_post_times.append((t_m_post_1 - t_m_post_0) * 1000)

        # 4. Unknown
        t_u_0 = time.perf_counter()
        unknown_candidates = processor.detect_unknown_candidates(
            frame=img,
            depth_map=depth_map,
            known_objects=fused_objects,
            frame_width=img.shape[1],
            frame_height=img.shape[0],
        )
        t_u_1 = time.perf_counter()
        unknown_times.append((t_u_1 - t_u_0) * 1000)

        # 5. Decision Engine
        t_d_0 = time.perf_counter()
        decision = processor.decision_engine.analyze(
            objects=fused_objects,
            scene_depth=scene_depth,
            frame_width=img.shape[1],
            frame_height=img.shape[0],
            unknown_objects=unknown_candidates,
        )
        t_d_1 = time.perf_counter()
        decision_times.append((t_d_1 - t_d_0) * 1000)

        # Full frame run
        t_tot_0 = time.perf_counter()
        full_res = processor.process_frame(img)
        t_tot_1 = time.perf_counter()
        total_times.append((t_tot_1 - t_tot_0) * 1000)

    print("\n========================================================")
    print("STAGE BREAKDOWN (Average of 5 runs)")
    print("========================================================")
    print(f"1. Image Decode (cv2.imdecode):          {np.mean(decode_times):7.2f} ms")
    print(f"2. YOLO Image Prepare:                   {np.mean(yolo_prep_times):7.2f} ms")
    print(f"3. YOLO Inference (CPU):                 {np.mean(yolo_infer_times):7.2f} ms")
    print(f"4. YOLO Result Parsing:                  {np.mean(yolo_parse_times):7.2f} ms")
    print(f"5. MiDaS Preprocessing (transform):      {np.mean(midas_prep_times):7.2f} ms")
    print(f"6. MiDaS Inference & Bicubic Interp:     {np.mean(midas_infer_times):7.2f} ms")
    print(f"7. MiDaS Postprocessing & Fusion:        {np.mean(midas_post_times):7.2f} ms")
    print(f"8. Unknown Obstacle Detection (OpenCV):  {np.mean(unknown_times):7.2f} ms")
    print(f"9. Decision Engine:                      {np.mean(decision_times):7.2f} ms")
    print(f"--------------------------------------------------------")
    subtotal = (np.mean(decode_times) + np.mean(yolo_prep_times) + np.mean(yolo_infer_times) +
                np.mean(yolo_parse_times) + np.mean(midas_prep_times) + np.mean(midas_infer_times) +
                np.mean(midas_post_times) + np.mean(unknown_times) + np.mean(decision_times))
    print(f"Subtotal of Stages:                      {subtotal:7.2f} ms")
    print(f"Full process_frame() measured:           {np.mean(total_times):7.2f} ms")
    print("========================================================\n")

if __name__ == "__main__":
    profile_breakdown()

