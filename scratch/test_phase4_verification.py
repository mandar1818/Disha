import sys
sys.path.insert(0, r"D:\SmartVisionAI_New")
import cv2
import numpy as np
from backend.processors.frame_processor import FrameProcessor
from backend.decision.engine import DecisionEngine

def test_real_image():
    print("=" * 60)
    print("TEST 1: Real test image (backend/test.jpg)")
    print("=" * 60)
    frame = cv2.imread("backend/test.jpg")
    processor = FrameProcessor()
    result = processor.process_frame(frame)

    print("Success:", result.get("success"))
    print("Frame:", result.get("frame"))
    
    known_objects = result.get("objects", [])
    print(f"\nKnown Objects ({len(known_objects)}):")
    for obj in known_objects:
        print(f"  ID={obj.get('id')}: {obj.get('class_name')} (conf={obj.get('confidence')}, pos={obj.get('position')}, dist={obj.get('distance_m')}m, risk={obj.get('risk_score')})")

    unknown_objects = result.get("unknown_objects", [])
    print(f"\nUnknown Objects ({len(unknown_objects)}):")
    for u in unknown_objects:
        print(f"  ID={u.get('id')}: pos={u.get('position')}, dist={u.get('distance_m')}m ({u.get('distance_category')}, {u.get('distance_confidence')}), risk={u.get('risk_score')}, rel={u.get('navigation_relevance')}, conf={u.get('confidence')}")

    print("\nPrimary Obstacle:", result.get("primary_obstacle", {}).get("class_name") if result.get("primary_obstacle") else None)
    print("Safety Level:", result.get("safety_level"))
    print("Emergency Stop:", result.get("emergency_stop"))
    print("Navigation:", result.get("navigation"))
    print("Voice Instruction:", result.get("voice_instruction"))
    print("Multi-Object:", result.get("multiple_objects"))

    # Assertions
    assert result.get("success") is True, "Processing failed"
    assert len(known_objects) > 0, "No known objects detected"
    
    # Check known classes are not "unknown"
    for obj in known_objects:
        assert obj.get("class_name") != "unknown", f"Known object was misclassified as unknown: {obj}"
        assert obj.get("id") < 1000, f"Known object ID collided with unknown ID range: {obj.get('id')}"

    # Check unknown objects format
    for u in unknown_objects:
        assert u.get("id") >= 1001, f"Unknown object ID must be >= 1001, got {u.get('id')}"
        assert u.get("position") in ("LEFT", "CENTER", "RIGHT"), f"Invalid position: {u.get('position')}"
        assert u.get("distance_category") in ("VERY_CLOSE", "NEAR", "FAR", "UNKNOWN_DISTANCE"), f"Invalid distance category: {u.get('distance_category')}"
        assert u.get("distance_confidence") in ("HIGH", "MEDIUM", "LOW"), f"Invalid distance confidence: {u.get('distance_confidence')}"
        assert u.get("navigation_relevance") in ("HIGH", "MEDIUM", "LOW", "NONE"), f"Invalid navigation relevance: {u.get('navigation_relevance')}"
        assert 0.0 <= u.get("confidence") <= 1.0, f"Invalid confidence: {u.get('confidence')}"
        assert "bbox" in u and all(k in u["bbox"] for k in ("x1", "y1", "x2", "y2")), f"Invalid bbox: {u.get('bbox')}"

    # Check suppression: no unknown object should significantly overlap a known object
    for u in unknown_objects:
        ub = u["bbox"]
        for k in known_objects:
            kb = k["bbox"]
            ix1 = max(ub["x1"], kb["x1"])
            iy1 = max(ub["y1"], kb["y1"])
            ix2 = min(ub["x2"], kb["x2"])
            iy2 = min(ub["y2"], kb["y2"])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter > 0:
                ua = (ub["x2"] - ub["x1"]) * (ub["y2"] - ub["y1"])
                ka = (kb["x2"] - kb["x1"]) * (kb["y2"] - kb["y1"])
                iou = inter / (ua + ka - inter)
                assert iou < 0.30, f"Suppression failed: unknown object {u['id']} overlaps {k['class_name']} with IoU {iou:.2f}"
    print("\n>>> TEST 1 PASSED: Known objects intact, unknown objects valid & non-overlapping!")

def test_unknown_primary_selection():
    print("\n" + "=" * 60)
    print("TEST 2: Unknown obstacle becomes primary obstacle when risk is highest")
    print("=" * 60)
    engine = DecisionEngine()
    # Scenario: Low-risk known object on side (cup/bottle, risk 0.20)
    # High-risk unknown obstacle right in the CENTER (risk 0.85, depth 0.15)
    known_objects = [
        {
            "id": 1,
            "class_name": "cup",
            "confidence": 0.80,
            "bbox": {"x1": 50, "y1": 300, "x2": 100, "y2": 400},
            "position": "LEFT",
            "distance_m": 3.0,
            "distance_category": "NEAR",
            "distance_confidence": "HIGH",
            "priority": 0.15,
            "risk_score": 0.22,
            "navigation_relevance": "LOW",
        }
    ]

    unknown_candidates = [
        {
            "bbox": {"x1": 200, "y1": 150, "x2": 440, "y2": 460},
            "cand_depth": 0.12,
            "confidence": 0.82,
        }
    ]

    scene_depth = {"average_depth": 0.4, "center_depth": 0.15, "left_depth": 0.5, "right_depth": 0.5}

    result = engine.analyze(
        objects=known_objects,
        scene_depth=scene_depth,
        frame_width=640,
        frame_height=480,
        unknown_objects=unknown_candidates,
    )

    print("Objects:", len(result["objects"]))
    print("Unknown Objects:", len(result["unknown_objects"]))
    primary = result.get("primary_obstacle")
    print("Primary Obstacle:", primary)
    print("Voice Instruction:", result.get("voice_instruction"))
    print("Safety Level:", result.get("safety_level"))
    print("Navigation:", result.get("navigation"))

    assert primary is not None, "Primary obstacle should exist"
    assert primary.get("class_name") == "unknown", f"Expected primary obstacle to be unknown, got {primary.get('class_name')}"
    assert primary.get("id") == 1001, f"Expected ID 1001, got {primary.get('id')}"
    assert primary.get("position") == "CENTER", f"Expected CENTER, got {primary.get('position')}"
    assert result.get("emergency_stop") is True, "Emergency stop should trigger for very close center obstacle"
    assert result.get("voice_instruction") == "Stop immediately.", f"Expected Stop immediately., got: {result.get('voice_instruction')}"
    print(">>> Emergency unknown obstacle test passed!")

    # Scenario 2B: Caution unknown obstacle in CENTER (not emergency)
    unknown_caution = [
        {
            "bbox": {"x1": 200, "y1": 150, "x2": 440, "y2": 460},
            "cand_depth": 0.30,
            "confidence": 0.75,
        }
    ]
    scene_caution = {"average_depth": 0.5, "center_depth": 0.35, "left_depth": 0.5, "right_depth": 0.5}
    res_caution = engine.analyze(
        objects=known_objects,
        scene_depth=scene_caution,
        frame_width=640,
        frame_height=480,
        unknown_objects=unknown_caution,
    )
    print("Caution Voice Instruction:", res_caution.get("voice_instruction"))
    assert "unknown obstacle" in res_caution.get("voice_instruction", "").lower(), f"Expected voice instruction to mention unknown obstacle, got: {res_caution.get('voice_instruction')}"
    print(">>> Caution unknown obstacle voice test passed!")

    # Scenario 2C: Unknown obstacle on LEFT
    unknown_left = [
        {
            "bbox": {"x1": 20, "y1": 150, "x2": 180, "y2": 460},
            "cand_depth": 0.30,
            "confidence": 0.75,
        }
    ]
    res_left = engine.analyze(
        objects=[],
        scene_depth=scene_caution,
        frame_width=640,
        frame_height=480,
        unknown_objects=unknown_left,
    )
    print("Left Voice Instruction:", res_left.get("voice_instruction"))
    assert "unknown obstacle on your left" in res_left.get("voice_instruction", "").lower(), f"Expected voice instruction to say unknown obstacle on your left, got: {res_left.get('voice_instruction')}"
    print(">>> Side unknown obstacle voice test passed!")

if __name__ == "__main__":
    test_real_image()
    test_unknown_primary_selection()
    print("\nALL IN-PROCESS TESTS PASSED SUCCESSFULLY!")
