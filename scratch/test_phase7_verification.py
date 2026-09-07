"""
======================================================================
SmartVisionAI - Phase 7 Dedicated Verification Suite
======================================================================
Tests 1-16:
1. Empty scene
2. One far small center object
3. One near center chair
4. One near center person
5. Large unknown center obstacle
6. LEFT obstacle only
7. RIGHT obstacle only
8. LEFT + CENTER obstacles
9. CENTER + RIGHT obstacles
10. LEFT + RIGHT obstacles
11. All three paths blocked (no_safe_path = True)
12. Known + unknown obstacles both participating
13. Multiple center obstacles
14. Small far obstacle + large near obstacle (near dominates)
15. Top-of-frame / background object (corridor not blocked)
16. Symmetry test (LEFT mirrored to RIGHT)
"""

import sys
import os
sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.decision.engine import DecisionEngine

def run_phase7_verification():
    print("=" * 70)
    print("STARTING PHASE 7 FREE-PATH & SAFE WALKING CORRIDOR VERIFICATION")
    print("=" * 70)

    engine = DecisionEngine()
    scene_clear = {"left_depth": 0.85, "center_region_depth": 0.85, "right_depth": 0.85}

    # ----------------------------------------------------------------
    # TEST 1: Empty Scene
    # ----------------------------------------------------------------
    print("\n--- TEST 1: Empty Scene ---")
    res1 = engine.analyze_free_path(scene_depth=scene_clear, objects=[], unknown_objects=[])
    print(f"Result: left_clear={res1['left_clear']}, center_clear={res1['center_clear']}, right_clear={res1['right_clear']}, best={res1['best_direction']}")
    assert res1["left_clear"] is True, "Test 1 failed: left should be clear"
    assert res1["center_clear"] is True, "Test 1 failed: center should be clear"
    assert res1["right_clear"] is True, "Test 1 failed: right should be clear"
    assert res1["best_direction"] == "CENTER", f"Test 1 failed: best should be CENTER, got {res1['best_direction']}"
    assert res1["no_safe_path"] is False, "Test 1 failed: no_safe_path should be False"
    print("TEST 1 Result: PASS (Empty scene -> all paths clear, best=CENTER)")

    # ----------------------------------------------------------------
    # TEST 2: One Far Small Center Object
    # ----------------------------------------------------------------
    print("\n--- TEST 2: Far Small Center Object ---")
    res2 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[{
            "id": 1,
            "class_name": "cup",
            "bbox": {"x1": 300, "y1": 200, "x2": 340, "y2": 260},
            "distance_m": 4.8,
            "distance_category": "FAR",
            "distance_confidence": "HIGH",
            "risk_score": 0.15,
            "confidence": 0.85
        }]
    )
    print(f"Result: center_clear={res2['center_clear']}, center_blocked={res2['center_blocked']}, center_risk={res2['center_risk']}")
    assert res2["center_clear"] is True, "Test 2 failed: small far center object should leave center clear"
    assert res2["center_blocked"] is False, "Test 2 failed: center_blocked should be False"
    print("TEST 2 Result: PASS (Small far center object -> center remains clear)")

    # ----------------------------------------------------------------
    # TEST 3: One Near Center Chair
    # ----------------------------------------------------------------
    print("\n--- TEST 3: Near Center Chair ---")
    res3 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[{
            "id": 1,
            "class_name": "chair",
            "bbox": {"x1": 240, "y1": 200, "x2": 400, "y2": 450},
            "distance_m": 1.7,
            "distance_category": "NEAR",
            "distance_confidence": "HIGH",
            "risk_score": 0.65,
            "confidence": 0.90
        }]
    )
    print(f"Result: center_clear={res3['center_clear']}, center_blocked={res3['center_blocked']}, center_cov={res3['center_coverage']}")
    assert res3["center_clear"] is False, "Test 3 failed: center should be blocked"
    assert res3["center_blocked"] is True, "Test 3 failed: center_blocked should be True"
    print("TEST 3 Result: PASS (Near center chair -> center blocked)")

    # ----------------------------------------------------------------
    # TEST 4: One Near Center Person
    # ----------------------------------------------------------------
    print("\n--- TEST 4: Near Center Person ---")
    res4 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[{
            "id": 1,
            "class_name": "person",
            "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460},
            "distance_m": 1.5,
            "distance_category": "NEAR",
            "distance_confidence": "HIGH",
            "risk_score": 0.70,
            "confidence": 0.92
        }]
    )
    print(f"Result: center_clear={res4['center_clear']}, center_blocked={res4['center_blocked']}")
    assert res4["center_clear"] is False, "Test 4 failed: center should be blocked by person"
    assert res4["center_blocked"] is True, "Test 4 failed: center_blocked should be True"
    print("TEST 4 Result: PASS (Near center person -> center blocked)")

    # ----------------------------------------------------------------
    # TEST 5: Large Unknown Center Obstacle
    # ----------------------------------------------------------------
    print("\n--- TEST 5: Large Unknown Center Obstacle ---")
    res5 = engine.analyze_free_path(
        scene_depth=scene_clear,
        unknown_objects=[{
            "id": 1001,
            "bbox": {"x1": 230, "y1": 180, "x2": 410, "y2": 440},
            "distance_m": 1.6,
            "distance_category": "NEAR",
            "distance_confidence": "HIGH",
            "risk_score": 0.65,
            "confidence": 0.85
        }]
    )
    print(f"Result: center_clear={res5['center_clear']}, center_blocked={res5['center_blocked']}")
    assert res5["center_clear"] is False, "Test 5 failed: unknown center obstacle should block center"
    assert res5["center_blocked"] is True, "Test 5 failed: center_blocked should be True"
    print("TEST 5 Result: PASS (Large unknown center obstacle -> center blocked)")

    # ----------------------------------------------------------------
    # TEST 6: LEFT Obstacle Only
    # ----------------------------------------------------------------
    print("\n--- TEST 6: LEFT Obstacle Only ---")
    res6 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[{
            "id": 1,
            "class_name": "chair",
            "bbox": {"x1": 30, "y1": 200, "x2": 190, "y2": 440},
            "distance_m": 1.5,
            "distance_category": "NEAR",
            "distance_confidence": "HIGH",
            "risk_score": 0.65,
            "confidence": 0.90
        }]
    )
    print(f"Result: left_clear={res6['left_clear']}, center_clear={res6['center_clear']}, right_clear={res6['right_clear']}, best={res6['best_direction']}")
    assert res6["left_clear"] is False and res6["left_blocked"] is True, "Test 6 failed: left should be blocked"
    assert res6["center_clear"] is True and res6["center_blocked"] is False, "Test 6 failed: center should be clear"
    assert res6["right_clear"] is True and res6["right_blocked"] is False, "Test 6 failed: right should be clear"
    assert res6["best_direction"] == "CENTER", f"Test 6 failed: best should be CENTER, got {res6['best_direction']}"
    print("TEST 6 Result: PASS (LEFT obstacle only -> left blocked, center & right clear, best=CENTER)")

    # ----------------------------------------------------------------
    # TEST 7: RIGHT Obstacle Only
    # ----------------------------------------------------------------
    print("\n--- TEST 7: RIGHT Obstacle Only ---")
    res7 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[{
            "id": 1,
            "class_name": "chair",
            "bbox": {"x1": 450, "y1": 200, "x2": 610, "y2": 440},
            "distance_m": 1.5,
            "distance_category": "NEAR",
            "distance_confidence": "HIGH",
            "risk_score": 0.65,
            "confidence": 0.90
        }]
    )
    print(f"Result: left_clear={res7['left_clear']}, center_clear={res7['center_clear']}, right_clear={res7['right_clear']}, best={res7['best_direction']}")
    assert res7["right_clear"] is False and res7["right_blocked"] is True, "Test 7 failed: right should be blocked"
    assert res7["center_clear"] is True and res7["center_blocked"] is False, "Test 7 failed: center should be clear"
    assert res7["left_clear"] is True and res7["left_blocked"] is False, "Test 7 failed: left should be clear"
    assert res7["best_direction"] == "CENTER", f"Test 7 failed: best should be CENTER, got {res7['best_direction']}"
    print("TEST 7 Result: PASS (RIGHT obstacle only -> right blocked, center & left clear, best=CENTER)")

    # ----------------------------------------------------------------
    # TEST 8: LEFT + CENTER Obstacles
    # ----------------------------------------------------------------
    print("\n--- TEST 8: LEFT + CENTER Obstacles ---")
    res8 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 190, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90},
            {"id": 2, "class_name": "person", "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70, "confidence": 0.92}
        ]
    )
    print(f"Result: left_clear={res8['left_clear']}, center_clear={res8['center_clear']}, right_clear={res8['right_clear']}, best={res8['best_direction']}")
    assert res8["left_clear"] is False, "Test 8 failed: left should be blocked"
    assert res8["center_clear"] is False, "Test 8 failed: center should be blocked"
    assert res8["right_clear"] is True, "Test 8 failed: right should be clear"
    assert res8["best_direction"] == "RIGHT", f"Test 8 failed: best should be RIGHT, got {res8['best_direction']}"
    print("TEST 8 Result: PASS (LEFT + CENTER blocked -> right clear, best=RIGHT)")

    # ----------------------------------------------------------------
    # TEST 9: CENTER + RIGHT Obstacles
    # ----------------------------------------------------------------
    print("\n--- TEST 9: CENTER + RIGHT Obstacles ---")
    res9 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "person", "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70, "confidence": 0.92},
            {"id": 2, "class_name": "chair", "bbox": {"x1": 450, "y1": 200, "x2": 610, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90}
        ]
    )
    print(f"Result: left_clear={res9['left_clear']}, center_clear={res9['center_clear']}, right_clear={res9['right_clear']}, best={res9['best_direction']}")
    assert res9["center_clear"] is False, "Test 9 failed: center should be blocked"
    assert res9["right_clear"] is False, "Test 9 failed: right should be blocked"
    assert res9["left_clear"] is True, "Test 9 failed: left should be clear"
    assert res9["best_direction"] == "LEFT", f"Test 9 failed: best should be LEFT, got {res9['best_direction']}"
    print("TEST 9 Result: PASS (CENTER + RIGHT blocked -> left clear, best=LEFT)")

    # ----------------------------------------------------------------
    # TEST 10: LEFT + RIGHT Obstacles
    # ----------------------------------------------------------------
    print("\n--- TEST 10: LEFT + RIGHT Obstacles ---")
    res10 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 190, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90},
            {"id": 2, "class_name": "chair", "bbox": {"x1": 450, "y1": 200, "x2": 610, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90}
        ]
    )
    print(f"Result: left_clear={res10['left_clear']}, center_clear={res10['center_clear']}, right_clear={res10['right_clear']}, best={res10['best_direction']}")
    assert res10["left_clear"] is False, "Test 10 failed: left should be blocked"
    assert res10["right_clear"] is False, "Test 10 failed: right should be blocked"
    assert res10["center_clear"] is True, "Test 10 failed: center should be clear"
    assert res10["best_direction"] == "CENTER", f"Test 10 failed: best should be CENTER, got {res10['best_direction']}"
    print("TEST 10 Result: PASS (LEFT + RIGHT blocked -> center clear, best=CENTER)")

    # ----------------------------------------------------------------
    # TEST 11: All Three Paths Blocked
    # ----------------------------------------------------------------
    print("\n--- TEST 11: All Three Paths Blocked ---")
    res11 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 190, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90},
            {"id": 2, "class_name": "person", "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70, "confidence": 0.92},
            {"id": 3, "class_name": "chair", "bbox": {"x1": 450, "y1": 200, "x2": 610, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90}
        ]
    )
    print(f"Result: left_clear={res11['left_clear']}, center_clear={res11['center_clear']}, right_clear={res11['right_clear']}, no_safe_path={res11['no_safe_path']}, best={res11['best_direction']}")
    assert res11["left_clear"] is False, "Test 11 failed: left should be blocked"
    assert res11["center_clear"] is False, "Test 11 failed: center should be blocked"
    assert res11["right_clear"] is False, "Test 11 failed: right should be blocked"
    assert res11["no_safe_path"] is True, "Test 11 failed: no_safe_path should be True"
    assert len(res11["safe_directions"]) == 0, "Test 11 failed: safe_directions should be empty"
    print("TEST 11 Result: PASS (All three paths blocked -> no_safe_path=True, safe_directions=[])")

    # ----------------------------------------------------------------
    # TEST 12: Known + Unknown Obstacles
    # ----------------------------------------------------------------
    print("\n--- TEST 12: Known + Unknown Obstacles ---")
    res12 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 190, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.90}
        ],
        unknown_objects=[
            {"id": 1001, "bbox": {"x1": 230, "y1": 180, "x2": 410, "y2": 440}, "distance_m": 1.6, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.85}
        ]
    )
    print(f"Result: left_clear={res12['left_clear']}, center_clear={res12['center_clear']}, right_clear={res12['right_clear']}")
    assert res12["left_clear"] is False, "Test 12 failed: left should be blocked by known chair"
    assert res12["center_clear"] is False, "Test 12 failed: center should be blocked by unknown obstacle"
    assert res12["right_clear"] is True, "Test 12 failed: right should be clear"
    print("TEST 12 Result: PASS (Known + unknown obstacles both participate in path analysis)")

    # ----------------------------------------------------------------
    # TEST 13: Multiple Center Obstacles
    # ----------------------------------------------------------------
    print("\n--- TEST 13: Multiple Center Obstacles ---")
    res13 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "chair", "bbox": {"x1": 230, "y1": 200, "x2": 320, "y2": 440}, "distance_m": 2.0, "distance_category": "NEAR", "risk_score": 0.55, "confidence": 0.85},
            {"id": 2, "class_name": "suitcase", "bbox": {"x1": 320, "y1": 220, "x2": 410, "y2": 440}, "distance_m": 2.2, "distance_category": "NEAR", "risk_score": 0.50, "confidence": 0.80}
        ]
    )
    print(f"Result: center_clear={res13['center_clear']}, center_blocked={res13['center_blocked']}, center_coverage={res13['center_coverage']}")
    assert res13["center_clear"] is False, "Test 13 failed: multiple center obstacles should block center"
    assert res13["center_blocked"] is True, "Test 13 failed: center_blocked should be True"
    print("TEST 13 Result: PASS (Multiple center obstacles -> center blocked)")

    # ----------------------------------------------------------------
    # TEST 14: Small Far + Large Near Obstacle
    # ----------------------------------------------------------------
    print("\n--- TEST 14: Small Far + Large Near Obstacle ---")
    res14 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[
            {"id": 1, "class_name": "cup", "bbox": {"x1": 300, "y1": 200, "x2": 330, "y2": 240}, "distance_m": 5.0, "distance_category": "FAR", "risk_score": 0.15, "confidence": 0.80},
            {"id": 2, "class_name": "chair", "bbox": {"x1": 240, "y1": 200, "x2": 400, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70, "confidence": 0.90}
        ]
    )
    print(f"Result: center_clear={res14['center_clear']}, center_risk={res14['center_risk']}")
    assert res14["center_clear"] is False, "Test 14 failed: near obstacle should dominate and block center"
    assert res14["center_blocked"] is True, "Test 14 failed: center_blocked should be True"
    print("TEST 14 Result: PASS (Large near obstacle dominates blocking)")

    # ----------------------------------------------------------------
    # TEST 15: Top-of-Frame Background Object
    # ----------------------------------------------------------------
    print("\n--- TEST 15: Top-of-Frame Background Object ---")
    res15 = engine.analyze_free_path(
        scene_depth=scene_clear,
        objects=[{
            "id": 1,
            "class_name": "traffic light",
            "bbox": {"x1": 250, "y1": 20, "x2": 390, "y2": 120},  # y2 = 120 / 480 = 0.25 < 0.35
            "distance_m": 1.2,
            "distance_category": "VERY_CLOSE",
            "risk_score": 0.75,
            "confidence": 0.88
        }]
    )
    print(f"Result: center_clear={res15['center_clear']}, center_blocked={res15['center_blocked']}, center_cov={res15['center_coverage']}")
    assert res15["center_clear"] is True, "Test 15 failed: top-of-frame object should not block walking corridor"
    assert res15["center_blocked"] is False, "Test 15 failed: center_blocked should be False"
    print("TEST 15 Result: PASS (Top-of-frame object does not incorrectly block walking corridor)")

    # ----------------------------------------------------------------
    # TEST 16: Symmetry Test
    # ----------------------------------------------------------------
    print("\n--- TEST 16: Symmetry Test ---")
    res16_left = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{"id": 1, "class_name": "chair", "bbox": {"x1": 50, "y1": 200, "x2": 200, "y2": 440}, "distance_m": 1.6, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.9}]
    )
    # Mirrored coordinates: 640 - 200 = 440, 640 - 50 = 590
    res16_right = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{"id": 1, "class_name": "chair", "bbox": {"x1": 440, "y1": 200, "x2": 590, "y2": 440}, "distance_m": 1.6, "distance_category": "NEAR", "risk_score": 0.65, "confidence": 0.9}]
    )
    print(f"Left obstacle: left_risk={res16_left['left_risk']}, right_risk={res16_left['right_risk']}, left_clear={res16_left['left_clear']}")
    print(f"Right obstacle: left_risk={res16_right['left_risk']}, right_risk={res16_right['right_risk']}, right_clear={res16_right['right_clear']}")
    assert res16_left["left_clear"] == res16_right["right_clear"] == False, "Test 16 failed: clearance asymmetry"
    assert res16_left["right_clear"] == res16_right["left_clear"] == True, "Test 16 failed: clearance asymmetry"
    assert res16_left["left_risk"] == res16_right["right_risk"], f"Test 16 failed: risk asymmetry {res16_left['left_risk']} != {res16_right['right_risk']}"
    assert res16_left["right_risk"] == res16_right["left_risk"], f"Test 16 failed: risk asymmetry {res16_left['right_risk']} != {res16_right['left_risk']}"
    print("TEST 16 Result: PASS (Symmetry test verified)")

    print("\n" + "=" * 70)
    print("ALL 16 PHASE 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase7_verification()
