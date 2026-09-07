"""
======================================================================
SmartVisionAI - Edge-AI Integration & Benchmark Test Suite
======================================================================
Validates the complete Edge-AI integration:
  1. TFLite YOLOv8n detector contract & accuracy
  2. ONNX MiDaS depth estimator contract & statistics
  3. FrameProcessor TFLite+ONNX end-to-end pipeline
  4. Safety invariants (STOP -> 0 steps, Emergency priority)
  5. Automatic fail-safe fallback to PyTorch on model failure
  6. Direct comparative benchmark (TFLite+ONNX vs PyTorch baseline)
     - 20 warm sequential requests each
     - YOLO latency (mean, median, P95)
     - MiDaS latency (mean, median, P95)
     - Total FrameProcessor latency (mean, median, P95)
     - First cold-start request latency
======================================================================
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import cv2
from typing import Any, Dict, List

# Ensure project root is in path
PROJECT_ROOT = r"D:\SmartVisionAI_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import (
    INFERENCE_BACKEND,
    YOLO_TFLITE_PATH,
    YOLO_TFLITE_MODEL,
    MIDAS_ONNX_PATH,
    MIDAS_ONNX_MODEL,
    YOLO_PYTORCH_PATH,
    YOLO_PYTORCH_MODEL,
    MIDAS_PYTORCH_MODEL,
)
from backend.yolo.detector import YOLODetector
from backend.depth.midas import MiDaSDepthEstimator
from backend.processors.frame_processor import FrameProcessor


class EdgeAIIntegrationTestSuite:
    def __init__(self):
        self.passed_count = 0
        self.total_count = 0
        self.test_img_path = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
        self.img = cv2.imread(self.test_img_path)
        assert self.img is not None, f"Failed to load test image: {self.test_img_path}"
        self.benchmark_data: Dict[str, Any] = {}

    def log_result(self, name: str, passed: bool, details: str = ""):
        self.total_count += 1
        if passed:
            self.passed_count += 1
            print(f"[PASS] {name}")
            if details:
                print(f"       {details}")
        else:
            print(f"[FAIL] {name}")
            if details:
                print(f"       ERROR: {details}")

    # ----------------------------------------------------------------
    # 1. TFLite YOLO Detector Tests
    # ----------------------------------------------------------------
    def test_tflite_yolo(self):
        print("\n--- 1. Testing TFLite YOLOv8n Detector ---")
        try:
            detector = YOLODetector(model_path=YOLO_TFLITE_PATH, backend="tflite")
            st = detector.status()
            self.log_result(
                "TFLite detector initialization & status",
                st.get("loaded") is True and st.get("backend") == "tflite",
                f"Status: {st}"
            )

            # Warmup test
            warm_ok = detector.warmup(640, 640)
            self.log_result("TFLite detector warmup", warm_ok is True)

            # Detection on test.jpg
            detections = detector.detect(self.img)
            self.log_result(
                "TFLite detection execution",
                isinstance(detections, list) and len(detections) > 0,
                f"Found {len(detections)} detections"
            )

            # Check schema contract
            valid_schema = True
            classes_found = set()
            for d in detections:
                if not all(k in d for k in ("class_name", "class_id", "confidence", "bbox")):
                    valid_schema = False
                bbox = d.get("bbox", {})
                if not all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                    valid_schema = False
                classes_found.add(d["class_name"])

            self.log_result("TFLite detection schema contract", valid_schema)
            self.log_result(
                "TFLite detection accuracy (detects person, chair, cup, plant)",
                {"person", "chair", "cup", "potted plant"}.issubset(classes_found),
                f"Detected classes: {classes_found}"
            )
            del detector
            import gc
            gc.collect()
        except Exception as exc:
            self.log_result("TFLite YOLO detector suite", False, str(exc))

    # ----------------------------------------------------------------
    # 2. ONNX MiDaS Depth Estimator Tests
    # ----------------------------------------------------------------
    def test_onnx_midas(self):
        print("\n--- 2. Testing ONNX MiDaS Depth Estimator ---")
        try:
            midas = MiDaSDepthEstimator(onnx_model_path=MIDAS_ONNX_PATH, backend="onnx")
            st = midas.status()
            self.log_result(
                "ONNX MiDaS initialization & status",
                st.get("loaded") is True and st.get("backend") == "onnx",
                f"Status: {st}"
            )

            # Warmup
            warm_ok = midas.warmup(640, 480)
            self.log_result("ONNX MiDaS warmup", warm_ok is True)

            # Estimate depth map
            depth_map = midas.estimate_depth_map(self.img)
            h, w = self.img.shape[:2]
            shape_ok = depth_map.shape == (h, w)
            range_ok = 0.0 <= depth_map.min() and depth_map.max() <= 1.0
            self.log_result(
                "ONNX depth map shape and [0.0, 1.0] contract",
                shape_ok and range_ok,
                f"Shape: {depth_map.shape}, Range: [{depth_map.min():.3f}, {depth_map.max():.3f}]"
            )

            # Scene analysis
            scene = midas.analyze_scene(depth_map)
            scene_keys = {"average_depth", "center_depth", "left_depth", "center_region_depth", "right_depth"}
            scene_ok = scene_keys.issubset(scene.keys())
            self.log_result(
                "ONNX scene depth spatial regions",
                scene_ok,
                f"Scene: {scene}"
            )
            del midas
            import gc
            gc.collect()
        except Exception as exc:
            self.log_result("ONNX MiDaS depth estimator suite", False, str(exc))

    # ----------------------------------------------------------------
    # 3. FrameProcessor TFLite+ONNX End-to-End Pipeline
    # ----------------------------------------------------------------
    def test_frame_processor_tflite_onnx(self):
        print("\n--- 3. Testing FrameProcessor with TFLite+ONNX ---")
        try:
            processor = FrameProcessor(inference_backend="tflite_onnx")
            st = processor.status()
            self.log_result(
                "FrameProcessor TFLite+ONNX backend active",
                st.get("active_backend") == "tflite_onnx" and st.get("loaded") is True,
                f"Status: {st}"
            )

            # Process frame
            result = processor.process_frame(self.img)
            req_keys = {
                "success", "objects", "unknown_objects", "scene", "free_path",
                "multiple_objects", "safety_level", "emergency_stop", "navigation",
                "step_guidance", "predictive_threat", "voice_instruction",
                "processing_time_ms", "processing_breakdown_ms", "backend"
            }
            keys_ok = req_keys.issubset(result.keys())
            self.log_result("FrameProcessor result schema completeness", keys_ok)

            # Check action and safety invariants
            nav_action = result.get("navigation", {}).get("action")
            steps = result.get("step_guidance", {}).get("steps", -1)
            action_ok = nav_action in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
            step_invariant = (nav_action != "STOP") or (steps == 0)

            self.log_result(
                "Navigation action and Step Invariant (STOP -> 0 steps)",
                action_ok and step_invariant,
                f"Action: {nav_action}, Steps: {steps}, Safety: {result.get('safety_level')}"
            )

            # Check breakdown telemetry
            breakdown = result.get("processing_breakdown_ms", {})
            b_keys = {"yolo", "midas", "fusion", "unknown", "decision", "total"}
            b_ok = b_keys.issubset(breakdown.keys())
            self.log_result(
                "Processing breakdown telemetry",
                b_ok,
                f"Breakdown: {breakdown}"
            )
            del processor
            import gc
            gc.collect()
        except Exception as exc:
            self.log_result("FrameProcessor TFLite+ONNX suite", False, str(exc))

    # ----------------------------------------------------------------
    # 4. Fail-Safe Automatic Fallback to PyTorch
    # ----------------------------------------------------------------
    def test_failsafe_fallback(self):
        print("\n--- 4. Testing Fail-Safe Automatic PyTorch Fallback ---")
        try:
            # Deliberately request tflite_onnx with a corrupt/nonexistent model path
            fallback_proc = FrameProcessor(
                yolo_model="nonexistent_invalid_path.tflite",
                inference_backend="tflite_onnx"
            )
            st = fallback_proc.status()
            fallback_succeeded = (
                st.get("active_backend") == "pytorch" and
                st.get("fallback_active") is True and
                st.get("loaded") is True and
                getattr(fallback_proc.yolo, "model_path", "") == YOLO_PYTORCH_PATH
            )
            self.log_result(
                "Automatic fallback to PyTorch on invalid TFLite path",
                fallback_succeeded,
                f"Active backend: {st.get('active_backend')}, Fallback active: {st.get('fallback_active')}, YOLO: {getattr(fallback_proc.yolo, 'model_path', '')}"
            )

            # Verify that frame processing still succeeds seamlessly in fallback mode
            res = fallback_proc.process_frame(self.img)
            self.log_result(
                "Fallback pipeline frame processing success",
                res.get("success") is True and len(res.get("objects", [])) > 0,
                f"Fallback objects: {len(res.get('objects', []))}, Action: {res.get('navigation', {}).get('action')}"
            )
            del fallback_proc, res
            import gc
            gc.collect()
        except Exception as exc:
            self.log_result("Fail-safe fallback suite", False, str(exc))

    # ----------------------------------------------------------------
    # 5. Comparative Latency Benchmark (20 warm sequential requests)
    # ----------------------------------------------------------------
    def run_comparative_benchmark(self):
        print("\n--- 5. Running 20-Frame Comparative Latency Benchmark ---")
        N_RUNS = 20

        # Benchmark A: TFLite + ONNX
        print("Benchmarking TFLite + ONNX...")
        t_cold_start = time.perf_counter()
        proc_tf = FrameProcessor(inference_backend="tflite_onnx")
        assert proc_tf.active_backend == "tflite_onnx", f"Expected tflite_onnx, got {proc_tf.active_backend}"
        res_tf_cold = proc_tf.process_frame(self.img)
        cold_tf_ms = (time.perf_counter() - t_cold_start) * 1000.0

        # Warmup
        for _ in range(3):
            proc_tf.process_frame(self.img)

        tf_yolo_times = []
        tf_midas_times = []
        tf_proc_times = []
        tf_total_times = []

        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            res = proc_tf.process_frame(self.img)
            t_tot = (time.perf_counter() - t0) * 1000.0
            bd = res.get("processing_breakdown_ms", {})
            tf_yolo_times.append(bd.get("yolo", 0.0))
            tf_midas_times.append(bd.get("midas", 0.0))
            tf_proc_times.append(res.get("processing_time_ms", t_tot))
            tf_total_times.append(t_tot)

        del proc_tf, res, res_tf_cold
        import gc
        gc.collect()

        # Benchmark B: PyTorch
        print("Benchmarking PyTorch Baseline...")
        t_cold_start = time.perf_counter()
        proc_pt = FrameProcessor(inference_backend="pytorch")
        assert proc_pt.active_backend == "pytorch", f"Expected pytorch, got {proc_pt.active_backend}"
        res_pt_cold = proc_pt.process_frame(self.img)
        cold_pt_ms = (time.perf_counter() - t_cold_start) * 1000.0

        # Warmup
        for _ in range(3):
            proc_pt.process_frame(self.img)

        pt_yolo_times = []
        pt_midas_times = []
        pt_proc_times = []
        pt_total_times = []

        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            res = proc_pt.process_frame(self.img)
            t_tot = (time.perf_counter() - t0) * 1000.0
            bd = res.get("processing_breakdown_ms", {})
            pt_yolo_times.append(bd.get("yolo", 0.0))
            pt_midas_times.append(bd.get("midas", 0.0))
            pt_proc_times.append(res.get("processing_time_ms", t_tot))
            pt_total_times.append(t_tot)

        del proc_pt, res, res_pt_cold
        gc.collect()

        def stats(arr):
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        self.benchmark_data = {
            "runs": N_RUNS,
            "tflite_onnx": {
                "cold_ms": cold_tf_ms,
                "yolo": stats(tf_yolo_times),
                "midas": stats(tf_midas_times),
                "processor": stats(tf_proc_times),
                "total": stats(tf_total_times),
            },
            "pytorch": {
                "cold_ms": cold_pt_ms,
                "yolo": stats(pt_yolo_times),
                "midas": stats(pt_midas_times),
                "processor": stats(pt_proc_times),
                "total": stats(pt_total_times),
            },
        }

        speedup_total = (
            (self.benchmark_data["pytorch"]["total"]["mean"] - self.benchmark_data["tflite_onnx"]["total"]["mean"])
            / self.benchmark_data["pytorch"]["total"]["mean"]
        ) * 100.0

        print("\n==========================================================================================")
        print("BENCHMARK COMPARISON RESULTS (20 Warm Frames, backend/test.jpg)")
        print("==========================================================================================")
        print(f"{'Backend':<22} | {'Success':<7} | {'Failed':<6} | {'Mean':<10} | {'Median':<10} | {'Min':<10} | {'Max':<10} | {'P95':<10}")
        print("-" * 90)
        print(f"{'TFLite+ONNX':<22} | {N_RUNS:<7} | {0:<6} | {self.benchmark_data['tflite_onnx']['total']['mean']:>7.2f} ms | {self.benchmark_data['tflite_onnx']['total']['median']:>7.2f} ms | {self.benchmark_data['tflite_onnx']['total']['min']:>7.2f} ms | {self.benchmark_data['tflite_onnx']['total']['max']:>7.2f} ms | {self.benchmark_data['tflite_onnx']['total']['p95']:>7.2f} ms")
        print(f"{'PyTorch Baseline':<22} | {N_RUNS:<7} | {0:<6} | {self.benchmark_data['pytorch']['total']['mean']:>7.2f} ms | {self.benchmark_data['pytorch']['total']['median']:>7.2f} ms | {self.benchmark_data['pytorch']['total']['min']:>7.2f} ms | {self.benchmark_data['pytorch']['total']['max']:>7.2f} ms | {self.benchmark_data['pytorch']['total']['p95']:>7.2f} ms")
        print("=" * 90)
        print("\nCOMPONENT BREAKDOWN (Mean Latency):")
        print(f"{'Component':<25} | {'PyTorch Baseline':<18} | {'TFLite+ONNX Edge-AI':<18} | {'Delta'}")
        print("-" * 75)
        print(f"{'First Cold Request':<25} | {cold_pt_ms:>15.2f} ms | {cold_tf_ms:>15.2f} ms | {((cold_pt_ms - cold_tf_ms)/cold_pt_ms)*100:+.1f}%")
        print(f"{'YOLO mean':<25} | {self.benchmark_data['pytorch']['yolo']['mean']:>15.2f} ms | {self.benchmark_data['tflite_onnx']['yolo']['mean']:>15.2f} ms | {((self.benchmark_data['pytorch']['yolo']['mean'] - self.benchmark_data['tflite_onnx']['yolo']['mean'])/self.benchmark_data['pytorch']['yolo']['mean'])*100:+.1f}%")
        print(f"{'MiDaS mean':<25} | {self.benchmark_data['pytorch']['midas']['mean']:>15.2f} ms | {self.benchmark_data['tflite_onnx']['midas']['mean']:>15.2f} ms | {((self.benchmark_data['pytorch']['midas']['mean'] - self.benchmark_data['tflite_onnx']['midas']['mean'])/self.benchmark_data['pytorch']['midas']['mean'])*100:+.1f}%")
        print(f"{'FrameProcessor mean':<25} | {self.benchmark_data['pytorch']['processor']['mean']:>15.2f} ms | {self.benchmark_data['tflite_onnx']['processor']['mean']:>15.2f} ms | {((self.benchmark_data['pytorch']['processor']['mean'] - self.benchmark_data['tflite_onnx']['processor']['mean'])/self.benchmark_data['pytorch']['processor']['mean'])*100:+.1f}%")
        print(f"{'Total /detect mean':<25} | {self.benchmark_data['pytorch']['total']['mean']:>15.2f} ms | {self.benchmark_data['tflite_onnx']['total']['mean']:>15.2f} ms | {speedup_total:+.1f}%")
        print("=" * 75)

        self.log_result(
            "Benchmark completed and TFLite+ONNX measured faster than PyTorch",
            speedup_total > 0,
            f"Overall latency improved by {speedup_total:.1f}% ({self.benchmark_data['pytorch']['total']['mean']:.1f}ms -> {self.benchmark_data['tflite_onnx']['total']['mean']:.1f}ms)"
        )


def main():
    print("=" * 65)
    print("SmartVisionAI Edge-AI Integration & Benchmark Suite")
    print("=" * 65)
    suite = EdgeAIIntegrationTestSuite()
    suite.test_tflite_yolo()
    suite.test_onnx_midas()
    suite.test_frame_processor_tflite_onnx()
    suite.run_comparative_benchmark()
    suite.test_failsafe_fallback()

    print("\n" + "=" * 65)
    print(f"TEST SUMMARY: {suite.passed_count}/{suite.total_count} PASSED ({(suite.passed_count/max(1, suite.total_count))*100:.1f}%)")
    print("=" * 65)
    if suite.passed_count == suite.total_count:
        print("ALL EDGE-AI INTEGRATION TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME EDGE-AI INTEGRATION TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
