import sys
import os
import math

sys.path.insert(0, r"D:\SmartVisionAI_New")
from backend.decision.engine import DecisionEngine

def run_phase6_tests():
    print("=" * 65)
    print("STARTING PHASE 6 MULTIPLE-OBJECT REASONING VERIFICATION SUITE")
    print("=" * 65)
    passed = 0
    total = 0

    # -------------------------------------------------------------
    # TEST A: One known object
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST A: One Known Object ---")
    engine = DecisionEngine()
    objs = [{
        "class_name": "person",
        "confidence": 0.88,
        "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 420},
        "depth": 0.35
    }]
    res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, center={mo['center_obstacles']}, human={mo['human_detected']}, vehicle={mo['vehicle_detected']}, highest_risk_id={mo['highest_risk_object_id']}, closest_id={mo['closest_object_id']}")
    assert mo["object_count"] == 1
    assert mo["center_obstacles"] == 1
    assert mo["left_obstacles"] == 0
    assert mo["right_obstacles"] == 0
    assert mo["human_detected"] is True
    assert mo["vehicle_detected"] is False
    assert mo["highest_risk_object_id"] == 1
    assert mo["closest_object_id"] == 1
    print("PASS: Test A verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST B: Two known objects
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST B: Two Known Objects ---")
    engine = DecisionEngine()
    objs = [
        {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 420}, "depth": 0.35},
        {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 50, "y1": 250, "x2": 180, "y2": 450}, "depth": 0.45}
    ]
    res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, left={mo['left_obstacles']}, center={mo['center_obstacles']}, right={mo['right_obstacles']}, human={mo['human_detected']}")
    assert mo["object_count"] == 2
    assert mo["left_obstacles"] == 1
    assert mo["center_obstacles"] == 1
    assert mo["right_obstacles"] == 0
    assert mo["human_detected"] is True
    assert mo["vehicle_detected"] is False
    print("PASS: Test B verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST C: Three known objects
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST C: Three Known Objects (including vehicle) ---")
    engine = DecisionEngine()
    objs = [
        {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 420}, "depth": 0.40},
        {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 50, "y1": 250, "x2": 180, "y2": 450}, "depth": 0.50},
        {"class_name": "car", "confidence": 0.90, "bbox": {"x1": 450, "y1": 150, "x2": 620, "y2": 380}, "depth": 0.30}
    ]
    res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, human={mo['human_detected']}, vehicle={mo['vehicle_detected']}")
    assert mo["object_count"] == 3
    assert mo["left_obstacles"] == 1
    assert mo["center_obstacles"] == 1
    assert mo["right_obstacles"] == 1
    assert mo["human_detected"] is True
    assert mo["vehicle_detected"] is True
    print("PASS: Test C verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST D: Known + unknown
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST D: Known + Unknown Mixed ---")
    engine = DecisionEngine()
    objs = [
        {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 50, "y1": 250, "x2": 180, "y2": 450}, "depth": 0.50}
    ]
    unk = [
        {"id": 1001, "bbox": {"x1": 250, "y1": 200, "x2": 390, "y2": 420}, "cand_depth": 0.25, "confidence": 0.75}
    ]
    res = engine.analyze(objects=objs, unknown_objects=unk, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, left={mo['left_obstacles']}, center={mo['center_obstacles']}, closest={mo['closest_object_id']}, highest_risk={mo['highest_risk_object_id']}")
    assert mo["object_count"] == 2
    assert mo["left_obstacles"] == 1
    assert mo["center_obstacles"] == 1
    assert mo["closest_object_id"] == 1001 # unknown is closer
    assert res["primary_obstacle"]["class_name"] == "unknown"
    print("PASS: Test D verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST E: Multiple unknown objects
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST E: Multiple Unknown Objects ---")
    engine = DecisionEngine()
    unk = [
        {"id": 1001, "bbox": {"x1": 50, "y1": 200, "x2": 180, "y2": 400}, "cand_depth": 0.40, "confidence": 0.70},
        {"id": 1002, "bbox": {"x1": 250, "y1": 200, "x2": 390, "y2": 400}, "cand_depth": 0.30, "confidence": 0.75},
        {"id": 1003, "bbox": {"x1": 460, "y1": 200, "x2": 590, "y2": 400}, "cand_depth": 0.50, "confidence": 0.65}
    ]
    res = engine.analyze(objects=[], unknown_objects=unk, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, left={mo['left_obstacles']}, center={mo['center_obstacles']}, right={mo['right_obstacles']}, IDs={[u['id'] for u in res['unknown_objects']]}")
    assert mo["object_count"] == 3
    assert mo["left_obstacles"] == 1
    assert mo["center_obstacles"] == 1
    assert mo["right_obstacles"] == 1
    assert mo["human_detected"] is False
    assert mo["vehicle_detected"] is False
    assert mo["closest_object_id"] == 1002 # 1002 depth 0.30 is closest
    print("PASS: Test E verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST F: LEFT + CENTER + RIGHT
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST F: Side-by-Side (LEFT + CENTER + RIGHT) ---")
    engine = DecisionEngine()
    objs = [
        {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 50, "y1": 200, "x2": 180, "y2": 400}, "depth": 0.30},
        {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 250, "y1": 150, "x2": 390, "y2": 450}, "depth": 0.40}
    ]
    unk = [
        {"id": 1001, "bbox": {"x1": 460, "y1": 200, "x2": 590, "y2": 400}, "cand_depth": 0.60, "confidence": 0.70}
    ]
    res = engine.analyze(objects=objs, unknown_objects=unk, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, left={mo['left_obstacles']}, center={mo['center_obstacles']}, right={mo['right_obstacles']}")
    assert mo["object_count"] == 3
    assert mo["left_obstacles"] == 1
    assert mo["center_obstacles"] == 1
    assert mo["right_obstacles"] == 1
    print("PASS: Test F verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST G: Multiple CENTER obstacles
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST G: Multiple Center Obstacles & Collective Blocking ---")
    engine = DecisionEngine()
    objs = [
        {"class_name": "person", "confidence": 0.65, "bbox": {"x1": 230, "y1": 100, "x2": 320, "y2": 400}, "depth": 0.45},
        {"class_name": "chair", "confidence": 0.65, "bbox": {"x1": 320, "y1": 200, "x2": 410, "y2": 420}, "depth": 0.45}
    ]
    res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: center_obstacles={mo['center_obstacles']}, safety_level={res['safety_level']}, is_center_blocked={res['free_path'].get('is_center_blocked')}")
    assert mo["center_obstacles"] == 2
    assert res["safety_level"] != "SAFE", "Multiple center obstacles must not produce SAFE level"
    assert res["free_path"].get("is_center_blocked") is True, "Collective center blockage must be flagged"
    print("PASS: Test G verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST H: Closest != Highest Risk
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST H: Closest != Highest Risk ---")
    engine = DecisionEngine()
    # Chair is very close but low priority; Person is farther but high priority/risk
    objs = [
        {"class_name": "chair", "confidence": 0.70, "bbox": {"x1": 50, "y1": 250, "x2": 180, "y2": 450}, "depth": 0.15}, # dist ~1.3m, risk ~0.55
        {"class_name": "person", "confidence": 0.90, "bbox": {"x1": 250, "y1": 100, "x2": 400, "y2": 420}, "depth": 0.35} # dist ~2.4m, risk ~0.75
    ]
    res = engine.analyze(objects=objs, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    c_id = mo["closest_object_id"]
    hr_id = mo["highest_risk_object_id"]
    print(f"Result: closest_id={c_id}, highest_risk_id={hr_id}")
    assert c_id != hr_id, f"Closest and highest-risk must be independent objects! got {c_id} and {hr_id}"
    assert c_id == 2 or c_id == 1
    print("PASS: Test H verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST I: Two independently tracked objects
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST I: Two Independently Tracked Objects ---")
    engine = DecisionEngine()
    for frame in range(1, 4):
        objs = [
            {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 50, "y1": 100, "x2": 180, "y2": 400}, "depth": 0.40},
            {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 450, "y1": 200, "x2": 600, "y2": 450}, "depth": 0.20}
        ]
        res = engine.analyze(objects=objs, frame_width=640, frame_height=480)

    assert len(engine.object_tracks) == 2, f"Expected 2 tracks, found {len(engine.object_tracks)}"
    trks = list(engine.object_tracks.values())
    print(f"Track 1: {trks[0]['class_name']} dist={trks[0]['smoothed_distance_m']}m, stab={trks[0]['stability']}")
    print(f"Track 2: {trks[1]['class_name']} dist={trks[1]['smoothed_distance_m']}m, stab={trks[1]['stability']}")
    assert trks[0]["class_name"] != trks[1]["class_name"]
    assert trks[0]["stability"] == 3 and trks[1]["stability"] == 3
    print("PASS: Test I verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST J: One approaching + one stationary
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST J: One Approaching + One Stationary ---")
    engine = DecisionEngine()
    # Object A (person) approaches (depth 0.7 -> 0.5 -> 0.3 -> 0.15)
    # Object B (chair) stays stationary (depth 0.4 -> 0.4 -> 0.4 -> 0.4)
    approach_depths = [0.70, 0.50, 0.30, 0.15]
    for d in approach_depths:
        objs = [
            {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 230, "y1": 100, "x2": 410, "y2": 420}, "depth": d},
            {"class_name": "chair", "confidence": 0.80, "bbox": {"x1": 50, "y1": 200, "x2": 180, "y2": 400}, "depth": 0.40}
        ]
        res = engine.analyze(objects=objs, frame_width=640, frame_height=480)

    p_obj = [o for o in res["objects"] if o["class_name"] == "person"][0]
    c_obj = [o for o in res["objects"] if o["class_name"] == "chair"][0]
    print(f"Person: approaching={p_obj['approaching']}, rate={p_obj['approach_rate']}m/s")
    print(f"Chair:  approaching={c_obj['approaching']}, rate={c_obj['approach_rate']}m/s")
    assert p_obj["approaching"] is True, "Approaching person must have approaching=True"
    assert c_obj["approaching"] is False, "Stationary chair must have approaching=False"
    print("PASS: Test J verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST K: Duplicate / Overlap Objects
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST K: Duplicate / Overlapping Objects Suppression ---")
    engine = DecisionEngine()
    objs = [
        {"class_name": "person", "confidence": 0.85, "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 420}, "depth": 0.35},
        {"class_name": "person", "confidence": 0.80, "bbox": {"x1": 255, "y1": 105, "x2": 395, "y2": 425}, "depth": 0.36} # high IoU duplicate
    ]
    unk = [
        {"id": 1001, "bbox": {"x1": 250, "y1": 100, "x2": 390, "y2": 420}, "cand_depth": 0.35, "confidence": 0.70} # overlapping with person
    ]
    res = engine.analyze(objects=objs, unknown_objects=unk, frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: object_count={mo['object_count']}, objects={len(res['objects'])}, unknowns={len(res['unknown_objects'])}")
    assert mo["object_count"] == 1, f"Expected duplicates to be suppressed to 1, got {mo['object_count']}"
    assert len(res["unknown_objects"]) == 0, "Unknown overlapping with known must be suppressed"
    print("PASS: Test K verified.")
    passed += 1

    # -------------------------------------------------------------
    # TEST L: No Obstacles
    # -------------------------------------------------------------
    total += 1
    print("\n--- TEST L: No Obstacles ---")
    engine = DecisionEngine()
    res = engine.analyze(objects=[], unknown_objects=[], frame_width=640, frame_height=480)
    mo = res["multiple_objects"]
    print(f"Result: count={mo['object_count']}, center={mo['center_obstacles']}, highest_risk={mo['highest_risk_object_id']}, closest={mo['closest_object_id']}")
    assert mo["object_count"] == 0
    assert mo["left_obstacles"] == 0
    assert mo["center_obstacles"] == 0
    assert mo["right_obstacles"] == 0
    assert mo["human_detected"] is False
    assert mo["vehicle_detected"] is False
    assert mo["highest_risk_object_id"] is None
    assert mo["closest_object_id"] is None
    assert res["primary_obstacle"] is None
    print("PASS: Test L verified.")
    passed += 1

    print("\n" + "=" * 65)
    print(f"ALL {passed}/{total} PHASE 6 VERIFICATION SCENARIOS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    run_phase6_tests()
