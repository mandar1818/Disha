"""
SmartVisionAI - Phase 14 Comprehensive Real-Time Optimization Test Suite
========================================================================
Validates all 25 test categories required by Phase 14:
  01. Timing instrumentation (monotonic, processing_breakdown_ms)
  02. YOLO model reuse across frames
  03. MiDaS model reuse across frames
  04. No duplicate model initialization (singleton safety)
  05. Image decoding robustness (valid JPEG, invalid bytes, empty)
  06. Detection contract (classes, boxes, confidences, coordinates)
  07. Distance contract (monocular estimation, categories, confidence, fallbacks)
  08. Unknown obstacle contract (candidate segmentation, ROI extraction)
  09. Multiple-object contract (spatial distribution, primary obstacle)
  10. Free-path contract (corridor evaluation, corridor blocking)
  11. Navigation contract (STOP, direction, action correctness)
  12. Step invariant (STOP implies 0 steps strictly)
  13. Predictive threat contract (motion tracking, approach rate, threat levels)
  14. Safety contract (safety levels, emergency stop)
  15. Performance benchmark (20 sequential requests, cold vs warm, mean, median, min, max, P95)
  16. Phase 4 regression (Unknown obstacle detection)
  17. Phase 5 regression (Distance estimation & calibration)
  18. Phase 6 regression (Multiple-object spatial reasoning)
  19. Phase 7 regression (Free path corridor analysis)
  20. Phase 8 regression (Navigation decision engine)
  21. Phase 9 regression (Turn and movement separation)
  22. Phase 10 regression (Walking steps and movement distance)
  23. Phase 11 regression (Motion tracking and predictive threat)
  24. Phase 12 regression (VoiceDecisionManager hierarchy and rules)
  25. Phase 13 regression (Mobile resilience contracts and watchdog)
"""

from __future__ import annotations

import asyncio
import io
import math
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = r"D:\SmartVisionAI_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import UploadFile

from backend.app import app, health, root, startup_event, system_status
from backend.decision.engine import DecisionEngine
from backend.processors.frame_processor import FrameProcessor
from backend.routes.detection import (
    decode_image,
    detect,
    detection_status,
    get_processor,
    warmup_processor,
)


class Phase14TestSuite:
    def __init__(self):
        self.passed_count = 0
        self.total_count = 25
        self.results: Dict[str, str] = {}
        self.test_img_path = os.path.join(PROJECT_ROOT, "backend", "test.jpg")
        with open(self.test_img_path, "rb") as f:
            self.test_img_bytes = f.read()
        self.test_img_cv = cv2.imread(self.test_img_path)

    def log_test(self, number: int, name: str, status: bool, details: str = ""):
        tag = f"Test {number:02d}: {name}"
        if status:
            self.passed_count += 1
            self.results[tag] = "PASS"
            print(f"[{tag}] ... PASS")
            if details:
                print(f"    {details}")
        else:
            self.results[tag] = "FAIL"
            print(f"[{tag}] ... FAIL")
            if details:
                print(f"    ERROR: {details}")

    # -------------------------------------------------------------
    # Test 01: Timing instrumentation
    # -------------------------------------------------------------
    def test_01_timing_instrumentation(self, proc_res: Dict[str, Any]):
        try:
            assert "processing_time_ms" in proc_res, "Missing processing_time_ms in result"
            t = proc_res["processing_time_ms"]
            assert isinstance(t, (int, float)) and t > 0, f"Invalid processing_time_ms: {t}"

            # Check breakdown telemetry
            breakdown = proc_res.get("processing_breakdown_ms")
            assert breakdown is not None, "Missing processing_breakdown_ms telemetry"
            required_stages = ["yolo", "midas", "fusion", "unknown", "decision", "total"]
            for s in required_stages:
                assert s in breakdown, f"Missing stage '{s}' in breakdown"
                assert breakdown[s] >= 0, f"Negative time in stage '{s}'"

            self.log_test(
                1,
                "Timing Instrumentation",
                True,
                f"Total: {t:.2f} ms | Breakdown: YOLO={breakdown['yolo']:.1f}ms, MiDaS={breakdown['midas']:.1f}ms, Fusion={breakdown['fusion']:.1f}ms, Unknown={breakdown['unknown']:.1f}ms, Decision={breakdown['decision']:.1f}ms",
            )
        except Exception as exc:
            self.log_test(1, "Timing Instrumentation", False, str(exc))

    # -------------------------------------------------------------
    # Test 02: YOLO model reuse
    # -------------------------------------------------------------
    def test_02_yolo_model_reuse(self, processor: FrameProcessor):
        try:
            yolo_detector = processor.yolo
            yolo_model = yolo_detector.model
            model_id_1 = id(yolo_model)

            # Process 2 more frames
            _ = processor.process_frame(self.test_img_cv)
            _ = processor.process_frame(self.test_img_cv)

            model_id_2 = id(processor.yolo.model)
            assert model_id_1 == model_id_2, "YOLO model instance was re-created instead of reused!"
            self.log_test(2, "YOLO Model Reuse", True, f"Model instance id({model_id_1}) preserved across frames")
        except Exception as exc:
            self.log_test(2, "YOLO Model Reuse", False, str(exc))

    # -------------------------------------------------------------
    # Test 03: MiDaS model reuse
    # -------------------------------------------------------------
    def test_03_midas_model_reuse(self, processor: FrameProcessor):
        try:
            midas_estimator = processor.midas
            midas_model = midas_estimator.model
            model_id_1 = id(midas_model)

            # Process another frame
            _ = processor.process_frame(self.test_img_cv)

            model_id_2 = id(processor.midas.model)
            assert model_id_1 == model_id_2, "MiDaS model instance was re-created instead of reused!"
            self.log_test(3, "MiDaS Model Reuse", True, f"Depth model id({model_id_1}) preserved across frames")
        except Exception as exc:
            self.log_test(3, "MiDaS Model Reuse", False, str(exc))

    # -------------------------------------------------------------
    # Test 04: No duplicate model initialization (Singleton)
    # -------------------------------------------------------------
    def test_04_no_duplicate_model_initialization(self):
        try:
            p1 = get_processor()
            p2 = get_processor()
            assert p1 is p2, "get_processor() returned different instances!"

            st = p1.status()
            assert st["loaded"] is True, "Processor reports loaded=False"
            assert st["yolo"]["loaded"] is True
            assert st["midas"]["loaded"] is True
            assert st["decision_engine"]["loaded"] is True
            self.log_test(4, "No Duplicate Model Initialization", True, "get_processor() maintains strict singleton")
        except Exception as exc:
            self.log_test(4, "No Duplicate Model Initialization", False, str(exc))

    # -------------------------------------------------------------
    # Test 05: Image decoding robustness
    # -------------------------------------------------------------
    def test_05_image_decoding(self):
        try:
            # Valid image
            img = decode_image(self.test_img_bytes)
            assert isinstance(img, np.ndarray), "Decoded image is not np.ndarray"
            assert img.ndim == 3 and img.shape[2] == 3, f"Unexpected shape: {img.shape}"
            assert img.dtype == np.uint8

            # Empty bytes
            empty_failed = False
            try:
                decode_image(b"")
            except ValueError:
                empty_failed = True
            assert empty_failed, "Empty bytes did not raise ValueError"

            # Corrupt bytes
            corrupt_failed = False
            try:
                decode_image(b"NotAValidJpegOrPngImageDataGarbage12345")
            except ValueError:
                corrupt_failed = True
            assert corrupt_failed, "Corrupt bytes did not raise ValueError"

            self.log_test(5, "Image Decoding Robustness", True, f"Decoded test.jpg to {img.shape}; invalid inputs rejected safely")
        except Exception as exc:
            self.log_test(5, "Image Decoding Robustness", False, str(exc))

    # -------------------------------------------------------------
    # Test 06: Detection contract
    # -------------------------------------------------------------
    def test_06_detection_contract(self, res: Dict[str, Any]):
        try:
            assert res.get("success") is True, "Result success is not True"
            objects = res.get("objects", [])
            assert len(objects) >= 1, f"Expected detections, got {len(objects)}"

            for obj in objects:
                assert "class_name" in obj, "Missing class_name"
                assert "confidence" in obj and obj["confidence"] >= 0.5, f"Low conf: {obj.get('confidence')}"
                assert "bbox" in obj, "Missing bbox"
                bbox = obj["bbox"]
                assert bbox["x2"] > bbox["x1"] and bbox["y2"] > bbox["y1"]
                assert "position" in obj and obj["position"] in ("LEFT", "CENTER", "RIGHT")
                assert "distance_m" in obj and obj["distance_m"] > 0
                assert "risk_score" in obj

            detected_names = [o["class_name"] for o in objects]
            self.log_test(6, "Detection Contract", True, f"Detected {len(objects)} objects: {detected_names}")
        except Exception as exc:
            self.log_test(6, "Detection Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 07: Distance contract
    # -------------------------------------------------------------
    def test_07_distance_contract(self):
        try:
            # Check canonical estimation & fallbacks
            d_near = DecisionEngine.estimate_distance_and_steps(0.35, [100, 100, 300, 400], "chair", 0.8)
            assert d_near["distance_category"] in ("VERY_CLOSE", "NEAR", "FAR")
            assert d_near["distance_m"] > 0
            assert d_near["distance_confidence"] in ("HIGH", "MEDIUM", "LOW")

            # Fallback on NaN / Inf
            d_nan = DecisionEngine.estimate_distance_and_steps(float("nan"), [100, 100, 300, 400], "person", 0.8)
            assert d_nan["distance_category"] == "UNKNOWN_DISTANCE"
            assert d_nan["distance_confidence"] == "LOW"

            d_inf = DecisionEngine.estimate_distance_and_steps(float("inf"), [100, 100, 300, 400], "person", 0.8)
            assert d_inf["distance_category"] == "UNKNOWN_DISTANCE"

            self.log_test(7, "Distance Contract", True, "Canonical distance estimation and non-finite fallbacks verified")
        except Exception as exc:
            self.log_test(7, "Distance Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 08: Unknown obstacle contract
    # -------------------------------------------------------------
    def test_08_unknown_obstacle_contract(self, res: Dict[str, Any]):
        try:
            assert "unknown_objects" in res, "Missing unknown_objects in result"
            unknowns = res["unknown_objects"]
            # Check unknown object structure if any candidate was segmented
            for u in unknowns:
                assert "bbox" in u
                assert "distance_m" in u and u["distance_m"] > 0
                assert "position" in u and u["position"] in ("LEFT", "CENTER", "RIGHT")
                assert "risk_score" in u
            self.log_test(8, "Unknown Obstacle Contract", True, f"unknown_objects list verified ({len(unknowns)} candidates)")
        except Exception as exc:
            self.log_test(8, "Unknown Obstacle Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 09: Multiple-object contract
    # -------------------------------------------------------------
    def test_09_multiple_object_contract(self, res: Dict[str, Any]):
        try:
            assert "multiple_objects" in res, "Missing multiple_objects in result"
            mo = res["multiple_objects"]
            assert "object_count" in mo and mo["object_count"] >= 1
            assert "left_obstacles" in mo
            assert "center_obstacles" in mo
            assert "right_obstacles" in mo
            assert "highest_risk_object_id" in mo
            assert "primary_obstacle" in res
            self.log_test(9, "Multiple-Object Contract", True, f"Total count={mo['object_count']} (L={mo['left_obstacles']}, C={mo['center_obstacles']}, R={mo['right_obstacles']})")
        except Exception as exc:
            self.log_test(9, "Multiple-Object Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 10: Free-path contract
    # -------------------------------------------------------------
    def test_10_free_path_contract(self, res: Dict[str, Any]):
        try:
            assert "free_path" in res, "Missing free_path in result"
            fp = res["free_path"]
            assert "left_clear" in fp
            assert "center_clear" in fp
            assert "right_clear" in fp
            assert "no_safe_path" in fp
            assert "best_direction" in fp
            self.log_test(10, "Free-Path Contract", True, f"Corridors: L={fp['left_clear']}, C={fp['center_clear']}, R={fp['right_clear']}, no_safe_path={fp['no_safe_path']}")
        except Exception as exc:
            self.log_test(10, "Free-Path Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 11: Navigation contract
    # -------------------------------------------------------------
    def test_11_navigation_contract(self, res: Dict[str, Any]):
        try:
            assert "navigation" in res, "Missing navigation in result"
            nav = res["navigation"]
            assert "action" in nav
            assert "direction" in nav
            assert "confidence" in nav and (nav["confidence"] in ("HIGH", "MEDIUM", "LOW") or 0.0 <= float(nav["confidence"]) <= 1.0)

            # On test.jpg, all corridors are blocked -> STOP
            assert nav["action"] == "STOP", f"Expected STOP on test.jpg, got {nav['action']}"
            assert nav["direction"] == "NONE", f"Expected NONE direction on STOP, got {nav['direction']}"
            self.log_test(11, "Navigation Contract", True, f"Action: {nav['action']}, Direction: {nav['direction']}, Confidence: {nav['confidence']}")
        except Exception as exc:
            self.log_test(11, "Navigation Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 12: Step invariant
    # -------------------------------------------------------------
    def test_12_step_invariant(self, res: Dict[str, Any]):
        try:
            assert "step_guidance" in res, "Missing step_guidance in result"
            sg = res["step_guidance"]
            nav = res["navigation"]
            if nav["action"] == "STOP":
                assert sg["steps"] == 0, f"VIOLATION: STOP action had non-zero steps: {sg['steps']}"
                assert sg["action"] in ("STOP", "NONE"), f"VIOLATION: STOP action had step action: {sg['action']}"
            self.log_test(12, "Step Invariant", True, f"Invariant verified: Action={nav['action']} strictly mandates steps={sg['steps']}")
        except Exception as exc:
            self.log_test(12, "Step Invariant", False, str(exc))

    # -------------------------------------------------------------
    # Test 13: Predictive threat contract
    # -------------------------------------------------------------
    def test_13_predictive_threat_contract(self):
        try:
            engine = DecisionEngine()
            # Frame 1: Initial observation
            obs_f1 = [{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.4, "distance_m": 4.0, "risk_score": 0.3}]
            r1 = engine.analyze(objects=obs_f1)
            o1 = r1["objects"][0]
            assert o1["motion_state"] == "UNKNOWN_MOTION", f"Expected UNKNOWN_MOTION on frame 1, got {o1['motion_state']}"
            assert o1["approach_rate_mps"] == 0.0

            # Frame 2: Approaching rapidly
            time.sleep(0.05)
            obs_f2 = [{"id": 1, "class_name": "person", "position": "CENTER", "depth": 0.7, "distance_m": 1.5, "risk_score": 0.7}]
            r2 = engine.analyze(objects=obs_f2)
            o2 = r2["objects"][0]
            assert o2["is_approaching"] is True, "Target failed to be marked approaching"
            assert o2["approach_rate_mps"] > 0, "Approach rate should be positive"
            assert "predictive_threat" in r2

            self.log_test(13, "Predictive Threat Contract", True, f"Frame 1: UNKNOWN_MOTION -> Frame 2: {o2['motion_state']} ({o2['approach_rate_mps']:.2f} m/s)")
        except Exception as exc:
            self.log_test(13, "Predictive Threat Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 14: Safety contract
    # -------------------------------------------------------------
    def test_14_safety_contract(self, res: Dict[str, Any]):
        try:
            assert "safety_level" in res, "Missing safety_level"
            assert res["safety_level"] in ("SAFE", "AWARE", "CAUTION", "DANGER", "EMERGENCY")
            assert "emergency_stop" in res and isinstance(res["emergency_stop"], bool)
            assert "voice_instruction" in res and len(res["voice_instruction"]) > 0
            self.log_test(14, "Safety Contract", True, f"Safety Level: {res['safety_level']}, Emergency Stop: {res['emergency_stop']}, Instruction: '{res['voice_instruction']}'")
        except Exception as exc:
            self.log_test(14, "Safety Contract", False, str(exc))

    # -------------------------------------------------------------
    # Test 15: Performance benchmark (20 sequential requests)
    # -------------------------------------------------------------
    async def test_15_performance_benchmark(self):
        try:
            print("\n    Running 20 sequential POST /detect requests on backend/test.jpg...")
            times_ms: List[float] = []
            wall_times_ms: List[float] = []

            for i in range(1, 21):
                upload = UploadFile(filename="test.jpg", file=io.BytesIO(self.test_img_bytes))
                t0 = time.perf_counter()
                resp = await detect(upload)
                t_wall = (time.perf_counter() - t0) * 1000
                p_time = resp["processing_time_ms"]
                times_ms.append(p_time)
                wall_times_ms.append(t_wall)
                print(f"      Request {i:02d}: processing = {p_time:.2f} ms | wall = {t_wall:.2f} ms")

            mean_val = float(np.mean(times_ms))
            median_val = float(np.median(times_ms))
            min_val = float(np.min(times_ms))
            max_val = float(np.max(times_ms))
            p95_val = float(np.percentile(times_ms, 95))

            print(f"\n    --- Benchmark Statistics (N=20) ---")
            print(f"    Mean   : {mean_val:.2f} ms")
            print(f"    Median : {median_val:.2f} ms")
            print(f"    Min    : {min_val:.2f} ms")
            print(f"    Max    : {max_val:.2f} ms")
            print(f"    P95    : {p95_val:.2f} ms")

            # Validate against acceptance criteria
            assert mean_val < 600.0, f"Benchmark mean {mean_val:.2f} ms exceeds acceptable limit (600 ms)"

            tier = "EXCELLENT (<400 ms)" if mean_val < 400.0 else ("GOOD (400-500 ms)" if mean_val < 500.0 else "ACCEPTABLE (500-600 ms)")
            self.log_test(
                15,
                "Performance Benchmark (20 Runs)",
                True,
                f"Tier: {tier} | Mean: {mean_val:.2f} ms | Median: {median_val:.2f} ms | P95: {p95_val:.2f} ms",
            )
            # Store stats on self for completion report
            self.benchmark_stats = {
                "mean": mean_val,
                "median": median_val,
                "min": min_val,
                "max": max_val,
                "p95": p95_val,
                "runs": times_ms,
                "wall_runs": wall_times_ms,
                "tier": tier,
            }
        except Exception as exc:
            self.log_test(15, "Performance Benchmark (20 Runs)", False, str(exc))

    # -------------------------------------------------------------
    # Test 16: Phase 4 regression (Unknown obstacles)
    # -------------------------------------------------------------
    def test_16_phase4_regression(self):
        try:
            from scratch.test_phase4_verification import test_unknown_primary_selection
            test_unknown_primary_selection()
            self.log_test(16, "Phase 4 Regression (Unknown Obstacles)", True, "Phase 4 unknown primary selection verified")
        except Exception as exc:
            self.log_test(16, "Phase 4 Regression (Unknown Obstacles)", False, str(exc))

    # -------------------------------------------------------------
    # Test 17: Phase 5 regression (Distance estimation)
    # -------------------------------------------------------------
    def test_17_phase5_regression(self):
        try:
            from scratch.test_phase5_verification import run_phase5_tests
            run_phase5_tests()
            self.log_test(17, "Phase 5 Regression (Distance Estimation)", True, "Phase 5 9/9 verification scenarios passed")
        except Exception as exc:
            self.log_test(17, "Phase 5 Regression (Distance Estimation)", False, str(exc))

    # -------------------------------------------------------------
    # Test 18: Phase 6 regression (Multiple-object reasoning)
    # -------------------------------------------------------------
    def test_18_phase6_regression(self):
        try:
            from scratch.test_phase6_verification import run_phase6_tests
            run_phase6_tests()
            self.log_test(18, "Phase 6 Regression (Multiple Objects)", True, "Phase 6 multi-object verification passed")
        except Exception as exc:
            self.log_test(18, "Phase 6 Regression (Multiple Objects)", False, str(exc))

    # -------------------------------------------------------------
    # Test 19: Phase 7 regression (Free path corridor analysis)
    # -------------------------------------------------------------
    def test_19_phase7_regression(self):
        try:
            from scratch.test_phase7_verification import run_phase7_verification
            run_phase7_verification()
            self.log_test(19, "Phase 7 Regression (Free Path)", True, "Phase 7 16-scenario corridor verification passed")
        except Exception as exc:
            self.log_test(19, "Phase 7 Regression (Free Path)", False, str(exc))

    # -------------------------------------------------------------
    # Test 20: Phase 8 regression (Navigation decision engine)
    # -------------------------------------------------------------
    def test_20_phase8_regression(self):
        try:
            from scratch.test_phase8_verification import run_phase8_verification
            run_phase8_verification()
            self.log_test(20, "Phase 8 Regression (Navigation Engine)", True, "Phase 8 23-scenario navigation suite passed")
        except Exception as exc:
            self.log_test(20, "Phase 8 Regression (Navigation Engine)", False, str(exc))

    # -------------------------------------------------------------
    # Test 21: Phase 9 regression (Turn vs movement separation)
    # -------------------------------------------------------------
    def test_21_phase9_regression(self):
        try:
            from scratch.test_phase9_verification import run_phase9_verification
            run_phase9_verification()
            self.log_test(21, "Phase 9 Regression (Turn vs Movement)", True, "Phase 9 30-scenario turn suite passed")
        except Exception as exc:
            self.log_test(21, "Phase 9 Regression (Turn vs Movement)", False, str(exc))

    # -------------------------------------------------------------
    # Test 22: Phase 10 regression (Walking steps guidance)
    # -------------------------------------------------------------
    def test_22_phase10_regression(self):
        try:
            from scratch.test_phase10_verification import run_all_tests as run_phase10_tests
            run_phase10_tests()
            self.log_test(22, "Phase 10 Regression (Walking Steps)", True, "Phase 10 35-scenario walking step suite passed")
        except Exception as exc:
            self.log_test(22, "Phase 10 Regression (Walking Steps)", False, str(exc))

    # -------------------------------------------------------------
    # Test 23: Phase 11 regression (Motion & predictive threat)
    # -------------------------------------------------------------
    def test_23_phase11_regression(self):
        try:
            from scratch.test_phase11_verification import run_phase11_tests
            run_phase11_tests()
            self.log_test(23, "Phase 11 Regression (Predictive Safety)", True, "Phase 11 37-scenario predictive safety suite passed")
        except Exception as exc:
            self.log_test(23, "Phase 11 Regression (Predictive Safety)", False, str(exc))

    # -------------------------------------------------------------
    # Test 24: Phase 12 regression (VoiceDecisionManager)
    # -------------------------------------------------------------
    def test_24_phase12_regression(self):
        try:
            from scratch.test_phase12_voice import run_tests as run_phase12_tests
            ok = run_phase12_tests()
            assert ok is True, "Phase 12 voice test suite failed"
            self.log_test(24, "Phase 12 Regression (Voice Guidance)", True, "Phase 12 voice guidance suite passed")
        except Exception as exc:
            self.log_test(24, "Phase 12 Regression (Voice Guidance)", False, str(exc))

    # -------------------------------------------------------------
    # Test 25: Phase 13 regression (Mobile resilience & watchdog)
    # -------------------------------------------------------------
    def test_25_phase13_regression(self):
        try:
            from scratch.test_phase13_resilience import run_phase13_suite
            ok = run_phase13_suite(include_historical=False)
            assert ok is True, "Phase 13 resilience test suite failed"
            self.log_test(25, "Phase 13 Regression (Resilience & Watchdog)", True, "Phase 13 resilience scenarios passed")
        except Exception as exc:
            self.log_test(25, "Phase 13 Regression (Resilience & Watchdog)", False, str(exc))


async def main():
    print("=" * 70)
    print("SMARTVISIONAI - PHASE 14 COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 70)

    # 0. Startup and prewarm
    print("Pre-warming backend...")
    t0 = time.perf_counter()
    await startup_event()
    print(f"Pre-warm complete in {(time.perf_counter() - t0)*1000:.2f} ms")

    suite = Phase14TestSuite()
    processor = get_processor()

    # Initial frame processing
    proc_res = processor.process_frame(suite.test_img_cv)

    # Run tests 01 - 14
    suite.test_01_timing_instrumentation(proc_res)
    suite.test_02_yolo_model_reuse(processor)
    suite.test_03_midas_model_reuse(processor)
    suite.test_04_no_duplicate_model_initialization()
    suite.test_05_image_decoding()
    suite.test_06_detection_contract(proc_res)
    suite.test_07_distance_contract()
    suite.test_08_unknown_obstacle_contract(proc_res)
    suite.test_09_multiple_object_contract(proc_res)
    suite.test_10_free_path_contract(proc_res)
    suite.test_11_navigation_contract(proc_res)
    suite.test_12_step_invariant(proc_res)
    suite.test_13_predictive_threat_contract()
    suite.test_14_safety_contract(proc_res)

    # Run test 15 (Benchmark)
    await suite.test_15_performance_benchmark()

    # Run tests 16 - 25 (Historical regressions)
    suite.test_16_phase4_regression()
    suite.test_17_phase5_regression()
    suite.test_18_phase6_regression()
    suite.test_19_phase7_regression()
    suite.test_20_phase8_regression()
    suite.test_21_phase9_regression()
    suite.test_22_phase10_regression()
    suite.test_23_phase11_regression()
    suite.test_24_phase12_regression()
    # Free memory before running subprocesses in Test 25
    import backend.routes.detection as det_mod
    det_mod._processor = None
    del processor, proc_res
    import gc
    gc.collect()
    suite.test_25_phase13_regression()

    print("\n" + "=" * 70)
    print("PHASE 14 VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total Tests  : {suite.total_count}")
    print(f"Passed Tests : {suite.passed_count}")
    print(f"Failed Tests : {suite.total_count - suite.passed_count}")
    print("=" * 70)

    if suite.passed_count == suite.total_count:
        print("ALL 25 PHASE 14 VERIFICATIONS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME TESTS FAILED! Check output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
