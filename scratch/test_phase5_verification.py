import sys
import os
import math
import numpy as np

# Add project root to sys.path
sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.decision.engine import DecisionEngine

def run_phase5_tests():
    print("=" * 60)
    print("STARTING PHASE 5 VERIFICATION SUITE")
    print("=" * 60)
    passed = 0
    total = 0

    # -------------------------------------------------------------
    # Test 1: Canonical Distance Categorization & Fallback
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 1: Canonical Distance Categorization & Fallback ---")
    d1 = DecisionEngine.estimate_distance_and_steps(depth=0.1, bbox=[100, 100, 300, 400], class_name="person", confidence=0.8)
    d2 = DecisionEngine.estimate_distance_and_steps(depth=0.4, bbox=[100, 100, 200, 250], class_name="chair", confidence=0.7)
    d3 = DecisionEngine.estimate_distance_and_steps(depth=0.9, bbox=[50, 50, 80, 80], class_name="car", confidence=0.6)
    d_nan = DecisionEngine.estimate_distance_and_steps(depth=float('nan'), bbox=[100, 100, 200, 200], class_name="person", confidence=0.8)
    d_inf = DecisionEngine.estimate_distance_and_steps(depth=float('inf'), bbox=[100, 100, 200, 200], class_name="person", confidence=0.8)
    d_neg = DecisionEngine.estimate_distance_and_steps(depth=-1.0, bbox=[100, 100, 200, 200], class_name="person", confidence=0.8)

    print(f"Close dist: {d1['estimated_distance_m']}m -> {d1['distance_category']}")
    print(f"Near dist: {d2['estimated_distance_m']}m -> {d2['distance_category']}")
    print(f"Far dist: {d3['estimated_distance_m']}m -> {d3['distance_category']}")
    print(f"NaN dist: {d_nan['estimated_distance_m']}m -> {d_nan['distance_category']}, conf={d_nan['distance_confidence']}")
    print(f"Inf dist: {d_inf['estimated_distance_m']}m -> {d_inf['distance_category']}, conf={d_inf['distance_confidence']}")
    print(f"Neg dist: {d_neg['estimated_distance_m']}m -> {d_neg['distance_category']}, conf={d_neg['distance_confidence']}")

    assert d1["distance_category"] == "VERY_CLOSE", f"Expected VERY_CLOSE, got {d1['distance_category']}"
    assert d2["distance_category"] == "NEAR", f"Expected NEAR, got {d2['distance_category']}"
    assert d3["distance_category"] == "FAR", f"Expected FAR, got {d3['distance_category']}"
    assert d_nan["distance_category"] == "UNKNOWN_DISTANCE" and d_nan["distance_confidence"] == "LOW"
    assert d_inf["distance_category"] == "UNKNOWN_DISTANCE" and d_inf["distance_confidence"] == "LOW"
    assert d_neg["distance_category"] == "UNKNOWN_DISTANCE" and d_neg["distance_confidence"] == "LOW"
    print("PASS: Distance categorization and non-finite fallbacks verified.")
    passed += 1

    # -------------------------------------------------------------
    # Test 2: Distance Confidence Calibration
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 2: Distance Confidence Calibration ---")
    c_high = DecisionEngine.estimate_distance_and_steps(depth=0.5, bbox=[100, 50, 400, 400], class_name="person", confidence=0.85, frame_width=640, frame_height=480)
    c_med = DecisionEngine.estimate_distance_and_steps(depth=0.5, bbox=[100, 50, 150, 100], class_name="person", confidence=0.55, frame_width=640, frame_height=480)
    c_low = DecisionEngine.estimate_distance_and_steps(depth=0.5, bbox=[100, 50, 150, 100], class_name="person", confidence=0.35, frame_width=640, frame_height=480)

    print(f"Confidence HIGH case: {c_high['distance_confidence']}")
    print(f"Confidence MEDIUM case: {c_med['distance_confidence']}")
    print(f"Confidence LOW case: {c_low['distance_confidence']}")
    assert c_high["distance_confidence"] == "HIGH"
    assert c_med["distance_confidence"] == "MEDIUM"
    assert c_low["distance_confidence"] == "LOW"
    print("PASS: Distance confidence levels verified.")
    passed += 1

    # -------------------------------------------------------------
    # Test 3: Stationary Object 10-Frame Smoothing & Stability
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 3: Stationary Object 10-Frame Smoothing ---")
    engine = DecisionEngine()
    raw_depths = [0.35, 0.48, 0.32, 0.46, 0.36, 0.44, 0.33, 0.47, 0.35, 0.46]
    smoothed_results = []
    raw_results = []

    for f_idx, d in enumerate(raw_depths):
        objs = [{
            "class_name": "chair",
            "confidence": 0.80,
            "bbox": {"x1": 220, "y1": 150, "x2": 420, "y2": 380},
            "depth": d
        }]
        res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
        p = res["primary_obstacle"]
        raw_dist = DecisionEngine.estimate_distance_and_steps(depth=d, bbox=[220, 150, 420, 380], class_name="chair", confidence=0.8)["estimated_distance_m"]
        raw_results.append(raw_dist)
        smoothed_results.append(p["distance_m"])
        print(f" Frame {f_idx+1}: raw={raw_dist:.2f}m -> smoothed={p['distance_m']:.2f}m, stability={p['obstacle_stability']}, approaching={res['approaching']}")

    raw_var = np.var(raw_results)
    sm_var = np.var(smoothed_results[3:]) # after warm-up
    print(f"Raw variance: {raw_var:.4f}, Smoothed variance (steady): {sm_var:.4f}")
    assert sm_var < raw_var * 0.40, f"Expected smoothing to reduce variance by >60%, raw={raw_var}, sm={sm_var}"
    assert res["approaching"] is False, "Stationary oscillating object should NOT be marked as approaching"
    assert res["obstacle_stability"] >= 5, f"Expected high stability counter, got {res['obstacle_stability']}"
    print("PASS: Stationary object smoothing significantly dampens variance and maintains stability.")
    passed += 1

    # -------------------------------------------------------------
    # Test 4: Approaching Object Detection
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 4: Approaching Object Detection ---")
    engine = DecisionEngine()
    # Monotonic descent in depth / distance
    approach_depths = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.12]
    approaching_detected = False
    for f_idx, d in enumerate(approach_depths):
        objs = [{
            "class_name": "person",
            "confidence": 0.85,
            "bbox": {"x1": 200, "y1": 100, "x2": 440, "y2": 400},
            "depth": d
        }]
        res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
        p = res["primary_obstacle"]
        print(f" Frame {f_idx+1} (depth={d:.2f}): smoothed={p['distance_m']:.2f}m, approaching={res['approaching']}, rate={res['approach_rate']}m/s")
        if res["approaching"] is True and res["approach_rate"] > 0:
            approaching_detected = True

    assert approaching_detected, "Approaching trend must be detected for consistently closing obstacle"
    print("PASS: Approaching obstacle correctly flagged with positive approach_rate.")
    passed += 1

    # -------------------------------------------------------------
    # Test 5: Noisy Non-Approaching Oscillations
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 5: Noisy Non-Approaching Oscillations ---")
    engine = DecisionEngine()
    noisy_depths = [0.40, 0.45, 0.39, 0.44, 0.40, 0.45, 0.39, 0.44]
    never_approaching = True
    for f_idx, d in enumerate(noisy_depths):
        objs = [{
            "class_name": "person",
            "confidence": 0.85,
            "bbox": {"x1": 200, "y1": 100, "x2": 440, "y2": 400},
            "depth": d
        }]
        res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
        if res["approaching"] is True:
            never_approaching = False
        print(f" Frame {f_idx+1}: smoothed={res['primary_obstacle']['distance_m']:.2f}m, approaching={res['approaching']}")

    assert never_approaching, "Random sensor noise should NOT trigger approaching alert"
    print("PASS: Noisy fluctuations correctly rejected from approaching trigger.")
    passed += 1

    # -------------------------------------------------------------
    # Test 6: Safety-Critical Bypass (Sudden Close Obstacle)
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 6: Safety-Critical Bypass ---")
    engine = DecisionEngine()
    # Frame 1-3: far object at ~3.5m
    for _ in range(3):
        res = engine.analyze(objects=[{
            "class_name": "person",
            "confidence": 0.85,
            "bbox": {"x1": 280, "y1": 150, "x2": 360, "y2": 300},
            "depth": 0.65
        }], frame_width=640, frame_height=480)
    print(f" Far steady state distance: {res['primary_obstacle']['distance_m']:.2f}m")

    # Frame 4: Sudden jump to very close 0.5m (raw depth 0.05)
    res_jump = engine.analyze(objects=[{
        "class_name": "person",
        "confidence": 0.90,
        "bbox": {"x1": 150, "y1": 50, "x2": 490, "y2": 450},
        "depth": 0.05
    }], frame_width=640, frame_height=480)
    jump_p = res_jump["primary_obstacle"]
    print(f" Jump frame: smoothed={jump_p['distance_m']:.2f}m, category={jump_p['distance_category']}, emergency={res_jump['emergency_stop']}")

    assert jump_p["distance_m"] <= 1.2, f"Safety bypass failed: expected <= 1.2m immediately, got {jump_p['distance_m']}"
    assert jump_p["distance_category"] == "VERY_CLOSE"
    assert res_jump["emergency_stop"] is True or res_jump["safety_level"] in ("EMERGENCY", "DANGER")
    print("PASS: Safety-critical bypass immediately snaps to close distance with zero lag.")
    passed += 1

    # -------------------------------------------------------------
    # Test 7: Missing Detection (Track Retention across frame drop)
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 7: Missing Detection Track Retention ---")
    engine = DecisionEngine()
    # Frame 1: Detected
    res1 = engine.analyze(objects=[{
        "class_name": "chair",
        "confidence": 0.80,
        "bbox": {"x1": 200, "y1": 200, "x2": 350, "y2": 400},
        "depth": 0.40
    }], frame_width=640, frame_height=480)
    dist1 = res1["primary_obstacle"]["distance_m"]
    stab1 = res1["primary_obstacle"]["obstacle_stability"]

    # Frame 2: Missed detection (empty list)
    res2 = engine.analyze(objects=[], frame_width=640, frame_height=480)
    assert res2["primary_obstacle"] is None

    # Frame 3: Reappears at similar position
    res3 = engine.analyze(objects=[{
        "class_name": "chair",
        "confidence": 0.82,
        "bbox": {"x1": 205, "y1": 195, "x2": 355, "y2": 395},
        "depth": 0.41
    }], frame_width=640, frame_height=480)
    dist3 = res3["primary_obstacle"]["distance_m"]
    stab3 = res3["primary_obstacle"]["obstacle_stability"]
    print(f" Frame 1 dist={dist1:.2f}m, stab={stab1}")
    print(f" Frame 2 missed")
    print(f" Frame 3 dist={dist3:.2f}m, stab={stab3}")

    assert stab3 >= 2, f"Expected track to be retained across 1 missed frame (stab >= 2), got {stab3}"
    print("PASS: Track preserved across intermittent frame drop without crashing or reset.")
    passed += 1

    # -------------------------------------------------------------
    # Test 8: Multi-Object Distance Independence
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 8: Multi-Object Distance Independence ---")
    engine = DecisionEngine()
    for _ in range(3):
        objs = [
            {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 50, "y1": 50, "x2": 200, "y2": 400}, "depth": 0.55},
            {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 350, "y1": 200, "x2": 550, "y2": 450}, "depth": 0.20}
        ]
        res = engine.analyze(objects=objs, frame_width=640, frame_height=480)

    out_objs = {o["class_name"]: o for o in res["objects"]}
    print(f" Person smoothed dist: {out_objs['person']['distance_m']:.2f}m, category={out_objs['person']['distance_category']}")
    print(f" Chair smoothed dist: {out_objs['chair']['distance_m']:.2f}m, category={out_objs['chair']['distance_category']}")

    assert out_objs["person"]["distance_m"] > out_objs["chair"]["distance_m"] + 0.5, "Person and Chair distances must not mix"
    assert len(engine.object_tracks) == 2, f"Expected exactly 2 independent tracks, found {len(engine.object_tracks)}"
    print("PASS: Multi-object distances tracked independently without cross-pollination.")
    passed += 1

    # -------------------------------------------------------------
    # Test 9: Phase 4 Regression (Unknown Obstacle Smoothing & Reasoning)
    # -------------------------------------------------------------
    total += 1
    print("\n--- Test 9: Phase 4 Unknown Obstacle Regression ---")
    engine = DecisionEngine()
    unk_cand = [{
        "id": 1001,
        "bbox": {"x1": 250, "y1": 250, "x2": 400, "y2": 450},
        "cand_depth": 0.22,
        "confidence": 0.72
    }]
    for _ in range(3):
        res = engine.analyze(objects=[], unknown_objects=unk_cand, frame_width=640, frame_height=480)

    assert len(res["unknown_objects"]) == 1, "Unknown obstacle must be retained in canonical unknown_objects"
    u = res["unknown_objects"][0]
    print(f" Unknown object: id={u['id']}, distance_m={u['distance_m']}, category={u['distance_category']}, conf={u['distance_confidence']}")
    assert u["distance_category"] in ("VERY_CLOSE", "NEAR")
    assert u["distance_m"] > 0
    assert res["primary_obstacle"]["class_name"] == "unknown"
    print("PASS: Unknown obstacle successfully participates in Phase 5 tracking and smoothing.")
    passed += 1

    print("\n" + "=" * 60)
    print(f"ALL {passed}/{total} PHASE 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_phase5_tests()
