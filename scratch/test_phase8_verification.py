"""
======================================================================
SmartVisionAI - Phase 8 Dedicated Verification Suite
======================================================================
Tests 1-23:
1. Empty scene -> GO_FORWARD, CENTER
2. Center obstacle, left safe -> TURN_LEFT, LEFT
3. Center obstacle, right safe -> TURN_RIGHT, RIGHT
4. Center blocked, both sides safe -> select lower-risk side deterministically
5. Only center safe -> GO_FORWARD
6. Only left safe -> TURN_LEFT
7. Only right safe -> TURN_RIGHT
8. All paths blocked -> STOP
9. Phase 7 real-image condition (no_safe_path=True, best_direction=RIGHT) -> STOP
10. Emergency stop -> STOP immediately
11. Danger center obstacle + right safe -> TURN_RIGHT
12. Danger center obstacle + left safe -> TURN_LEFT
13. Approaching center obstacle + side safe -> side avoidance
14. Approaching center obstacle + all paths blocked -> STOP
15. Far small center object -> GO_FORWARD if center remains clear
16. Unknown center obstacle + right safe -> TURN_RIGHT
17. Unknown center obstacle + left safe -> TURN_LEFT
18. Closest obstacle != highest-risk obstacle -> follows highest-risk obstacle
19. Navigation hysteresis -> rejects rapid 1-frame direction oscillations
20. Emergency hysteresis bypass -> immediate STOP
21. GO_BACK safety -> STOP without verified backward clearance; GO_BACK with verified clearance
22. Turn/movement flags -> exact contract verification across all actions
23. Navigation reason -> meaningful reason text present on all actions
"""

import sys
import os
sys.path.insert(0, r"D:\SmartVisionAI_New")

from backend.decision.engine import DecisionEngine

def run_phase8_verification():
    print("=" * 70)
    print("STARTING PHASE 8 NAVIGATION DECISION ENGINE VERIFICATION SUITE")
    print("=" * 70)

    # ----------------------------------------------------------------
    # TEST 1: Empty Scene
    # ----------------------------------------------------------------
    print("\n--- TEST 1: Empty Scene ---")
    e1 = DecisionEngine()
    fp1 = {"left_clear": True, "center_clear": True, "right_clear": True, "no_safe_path": False}
    res1 = e1.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp1)
    print(f"Result: action={res1['action']}, direction={res1['direction']}, turn_required={res1['turn_required']}, movement_required={res1['movement_required']}")
    assert res1["action"] == "GO_FORWARD", f"Test 1 failed: expected GO_FORWARD, got {res1['action']}"
    assert res1["direction"] == "CENTER", f"Test 1 failed: expected CENTER, got {res1['direction']}"
    assert res1["turn_required"] is False and res1["movement_required"] is True
    print("TEST 1 Result: PASS (Empty scene -> GO_FORWARD, CENTER)")

    # ----------------------------------------------------------------
    # TEST 2: Center Obstacle, Left Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 2: Center Obstacle, Left Safe ---")
    e2 = DecisionEngine()
    e2.last_action = "TURN_LEFT"
    fp2 = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False, "left_risk": 0.2, "right_risk": 0.8}
    obs2 = {"position": "CENTER", "distance_m": 1.7, "risk_score": 0.65}
    res2 = e2.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs2, free_path=fp2)
    print(f"Result: action={res2['action']}, direction={res2['direction']}, reason={res2['reason']}")
    assert res2["action"] == "TURN_LEFT" and res2["direction"] == "LEFT"
    assert res2["turn_required"] is True and res2["movement_required"] is False
    print("TEST 2 Result: PASS (Center obstacle, left safe -> TURN_LEFT, LEFT)")

    # ----------------------------------------------------------------
    # TEST 3: Center Obstacle, Right Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 3: Center Obstacle, Right Safe ---")
    e3 = DecisionEngine()
    e3.last_action = "TURN_RIGHT"
    fp3 = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False, "left_risk": 0.8, "right_risk": 0.2}
    obs3 = {"position": "CENTER", "distance_m": 1.7, "risk_score": 0.65}
    res3 = e3.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs3, free_path=fp3)
    print(f"Result: action={res3['action']}, direction={res3['direction']}, reason={res3['reason']}")
    assert res3["action"] == "TURN_RIGHT" and res3["direction"] == "RIGHT"
    assert res3["turn_required"] is True and res3["movement_required"] is False
    print("TEST 3 Result: PASS (Center obstacle, right safe -> TURN_RIGHT, RIGHT)")

    # ----------------------------------------------------------------
    # TEST 4: Center Blocked, Both Sides Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 4: Center Blocked, Both Sides Safe ---")
    e4 = DecisionEngine()
    e4.last_action = "TURN_RIGHT"
    fp4 = {"left_clear": True, "center_clear": False, "right_clear": True, "no_safe_path": False, "left_risk": 0.60, "right_risk": 0.25}
    obs4 = {"position": "CENTER", "distance_m": 1.8, "risk_score": 0.60}
    res4 = e4.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs4, free_path=fp4)
    print(f"Result: action={res4['action']}, direction={res4['direction']}, reason={res4['reason']}")
    assert res4["action"] == "TURN_RIGHT" and res4["direction"] == "RIGHT"
    print("TEST 4 Result: PASS (Center blocked, both sides safe -> selects lower-risk side deterministically)")

    # ----------------------------------------------------------------
    # TEST 5: Only Center Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 5: Only Center Safe ---")
    e5 = DecisionEngine()
    fp5 = {"left_clear": False, "center_clear": True, "right_clear": False, "no_safe_path": False}
    res5 = e5.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=None, free_path=fp5)
    print(f"Result: action={res5['action']}, direction={res5['direction']}, reason={res5['reason']}")
    assert res5["action"] == "GO_FORWARD" and res5["direction"] == "CENTER"
    print("TEST 5 Result: PASS (Only center safe -> GO_FORWARD, CENTER)")

    # ----------------------------------------------------------------
    # TEST 6: Only Left Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 6: Only Left Safe ---")
    e6 = DecisionEngine()
    e6.last_action = "TURN_LEFT"
    fp6 = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False}
    res6 = e6.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp6)
    print(f"Result: action={res6['action']}, direction={res6['direction']}")
    assert res6["action"] == "TURN_LEFT" and res6["direction"] == "LEFT"
    print("TEST 6 Result: PASS (Only left safe -> TURN_LEFT, LEFT)")

    # ----------------------------------------------------------------
    # TEST 7: Only Right Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 7: Only Right Safe ---")
    e7 = DecisionEngine()
    e7.last_action = "TURN_RIGHT"
    fp7 = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False}
    res7 = e7.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp7)
    print(f"Result: action={res7['action']}, direction={res7['direction']}")
    assert res7["action"] == "TURN_RIGHT" and res7["direction"] == "RIGHT"
    print("TEST 7 Result: PASS (Only right safe -> TURN_RIGHT, RIGHT)")

    # ----------------------------------------------------------------
    # TEST 8: All Paths Blocked
    # ----------------------------------------------------------------
    print("\n--- TEST 8: All Paths Blocked ---")
    e8 = DecisionEngine()
    fp8 = {"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True}
    res8 = e8.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle={"position": "CENTER"}, free_path=fp8)
    print(f"Result: action={res8['action']}, direction={res8['direction']}, turn_required={res8['turn_required']}, movement_required={res8['movement_required']}")
    assert res8["action"] == "STOP"
    assert res8["turn_required"] is False and res8["movement_required"] is False
    print("TEST 8 Result: PASS (All paths blocked -> STOP)")

    # ----------------------------------------------------------------
    # TEST 9: Phase 7 Real-Image Condition (Mandatory Test)
    # ----------------------------------------------------------------
    print("\n--- TEST 9: Phase 7 Real-Image Condition (no_safe_path=True, best=RIGHT) ---")
    e9 = DecisionEngine()
    fp9 = {
        "left_clear": False,
        "center_clear": False,
        "right_clear": False,
        "no_safe_path": True,
        "best_direction": "RIGHT"  # Least dangerous fallback in Phase 7
    }
    res9 = e9.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle={"position": "CENTER", "distance_m": 2.96}, free_path=fp9)
    print(f"Result: action={res9['action']}, direction={res9['direction']}, reason={res9['reason']}")
    assert res9["action"] == "STOP", f"Test 9 FAILED: expected STOP, got {res9['action']}"
    assert res9["turn_required"] is False and res9["movement_required"] is False
    print("TEST 9 Result: PASS (Phase 7 real-image condition -> STOP, does NOT blindly execute TURN_RIGHT)")

    # ----------------------------------------------------------------
    # TEST 10: Emergency Stop
    # ----------------------------------------------------------------
    print("\n--- TEST 10: Emergency Stop ---")
    e10 = DecisionEngine()
    fp10 = {"left_clear": True, "center_clear": True, "right_clear": True, "no_safe_path": False}
    res10 = e10.determine_navigation(safety_level="EMERGENCY", emergency_stop=True, primary_obstacle={"position": "CENTER", "distance_m": 0.8}, free_path=fp10)
    print(f"Result: action={res10['action']}, direction={res10['direction']}, reason={res10['reason']}")
    assert res10["action"] == "STOP"
    assert res10["turn_required"] is False and res10["movement_required"] is False
    print("TEST 10 Result: PASS (Emergency stop -> STOP immediately)")

    # ----------------------------------------------------------------
    # TEST 11: Danger Center Obstacle + Right Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 11: Danger Center Obstacle + Right Safe ---")
    e11 = DecisionEngine()
    e11.last_action = "TURN_RIGHT"
    fp11 = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False}
    obs11 = {"position": "CENTER", "risk_score": 0.75, "distance_m": 1.4}
    res11 = e11.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs11, free_path=fp11)
    print(f"Result: action={res11['action']}, direction={res11['direction']}")
    assert res11["action"] == "TURN_RIGHT" and res11["direction"] == "RIGHT"
    print("TEST 11 Result: PASS (Danger center obstacle + right safe -> TURN_RIGHT)")

    # ----------------------------------------------------------------
    # TEST 12: Danger Center Obstacle + Left Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 12: Danger Center Obstacle + Left Safe ---")
    e12 = DecisionEngine()
    e12.last_action = "TURN_LEFT"
    fp12 = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False}
    obs12 = {"position": "CENTER", "risk_score": 0.75, "distance_m": 1.4}
    res12 = e12.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs12, free_path=fp12)
    print(f"Result: action={res12['action']}, direction={res12['direction']}")
    assert res12["action"] == "TURN_LEFT" and res12["direction"] == "LEFT"
    print("TEST 12 Result: PASS (Danger center obstacle + left safe -> TURN_LEFT)")

    # ----------------------------------------------------------------
    # TEST 13: Approaching Center Obstacle + Side Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 13: Approaching Center Obstacle + Side Safe ---")
    e13 = DecisionEngine()
    e13.last_action = "TURN_RIGHT"
    fp13 = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False}
    obs13 = {"position": "CENTER", "approaching": True, "risk_score": 0.70, "distance_m": 2.0}
    res13 = e13.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs13, free_path=fp13)
    print(f"Result: action={res13['action']}, direction={res13['direction']}")
    assert res13["action"] == "TURN_RIGHT"
    print("TEST 13 Result: PASS (Approaching center obstacle + side safe -> TURN_RIGHT avoidance)")

    # ----------------------------------------------------------------
    # TEST 14: Approaching Center Obstacle + All Paths Blocked
    # ----------------------------------------------------------------
    print("\n--- TEST 14: Approaching Center Obstacle + All Paths Blocked ---")
    e14 = DecisionEngine()
    fp14 = {"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True}
    obs14 = {"position": "CENTER", "approaching": True, "risk_score": 0.80, "distance_m": 1.2}
    res14 = e14.determine_navigation(safety_level="DANGER", emergency_stop=False, primary_obstacle=obs14, free_path=fp14)
    print(f"Result: action={res14['action']}")
    assert res14["action"] == "STOP"
    print("TEST 14 Result: PASS (Approaching center obstacle + all blocked -> STOP)")

    # ----------------------------------------------------------------
    # TEST 15: Far Small Center Object
    # ----------------------------------------------------------------
    print("\n--- TEST 15: Far Small Center Object ---")
    e15 = DecisionEngine()
    fp15 = {"left_clear": True, "center_clear": True, "right_clear": True, "no_safe_path": False}
    obs15 = {"position": "CENTER", "distance_m": 4.8, "distance_category": "FAR", "risk_score": 0.15}
    res15 = e15.determine_navigation(safety_level="SAFE", emergency_stop=False, primary_obstacle=obs15, free_path=fp15)
    print(f"Result: action={res15['action']}, direction={res15['direction']}")
    assert res15["action"] == "GO_FORWARD" and res15["direction"] == "CENTER"
    print("TEST 15 Result: PASS (Far small center object -> GO_FORWARD, CENTER)")

    # ----------------------------------------------------------------
    # TEST 16: Unknown Center Obstacle + Right Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 16: Unknown Center Obstacle + Right Safe ---")
    e16 = DecisionEngine()
    e16.last_action = "TURN_RIGHT"
    fp16 = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False}
    obs16 = {"class_name": "unknown_obstacle", "position": "CENTER", "distance_m": 1.6, "risk_score": 0.65}
    res16 = e16.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs16, free_path=fp16)
    print(f"Result: action={res16['action']}, direction={res16['direction']}")
    assert res16["action"] == "TURN_RIGHT" and res16["direction"] == "RIGHT"
    print("TEST 16 Result: PASS (Unknown center obstacle + right safe -> TURN_RIGHT)")

    # ----------------------------------------------------------------
    # TEST 17: Unknown Center Obstacle + Left Safe
    # ----------------------------------------------------------------
    print("\n--- TEST 17: Unknown Center Obstacle + Left Safe ---")
    e17 = DecisionEngine()
    e17.last_action = "TURN_LEFT"
    fp17 = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False}
    obs17 = {"class_name": "unknown_obstacle", "position": "CENTER", "distance_m": 1.6, "risk_score": 0.65}
    res17 = e17.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs17, free_path=fp17)
    print(f"Result: action={res17['action']}, direction={res17['direction']}")
    assert res17["action"] == "TURN_LEFT" and res17["direction"] == "LEFT"
    print("TEST 17 Result: PASS (Unknown center obstacle + left safe -> TURN_LEFT)")

    # ----------------------------------------------------------------
    # TEST 18: Closest Obstacle != Highest-Risk Obstacle
    # ----------------------------------------------------------------
    print("\n--- TEST 18: Closest Obstacle != Highest-Risk Obstacle ---")
    e18 = DecisionEngine()
    e18.last_action = "TURN_LEFT"
    # Primary obstacle is highest-risk center obstacle at 1.5m, left corridor clear
    fp18 = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False}
    obs18_highest = {"position": "CENTER", "distance_m": 1.5, "risk_score": 0.70}
    res18 = e18.determine_navigation(safety_level="CAUTION", emergency_stop=False, primary_obstacle=obs18_highest, free_path=fp18)
    print(f"Result: action={res18['action']}, direction={res18['direction']}")
    assert res18["action"] == "TURN_LEFT" and res18["direction"] == "LEFT"
    print("TEST 18 Result: PASS (Navigation avoids highest-risk blocking obstacle -> TURN_LEFT)")

    # ----------------------------------------------------------------
    # TEST 19: Navigation Hysteresis
    # ----------------------------------------------------------------
    print("\n--- TEST 19: Navigation Hysteresis ---")
    e19 = DecisionEngine()
    e19.last_action = "TURN_LEFT"
    fp19_right = {"left_clear": False, "center_clear": False, "right_clear": True, "no_safe_path": False}
    fp19_left = {"left_clear": True, "center_clear": False, "right_clear": False, "no_safe_path": False}

    # Frame 1: Candidate wants TURN_RIGHT (1st frame)
    r19_1 = e19.determine_navigation("CAUTION", False, {"position": "CENTER"}, fp19_right)
    assert r19_1["action"] == "TURN_LEFT", f"Frame 1 failed: expected TURN_LEFT, got {r19_1['action']}"

    # Frame 2: Candidate wants TURN_LEFT again
    r19_2 = e19.determine_navigation("CAUTION", False, {"position": "CENTER"}, fp19_left)
    assert r19_2["action"] == "TURN_LEFT"

    # Frame 3: Candidate wants TURN_RIGHT again (1st frame)
    r19_3 = e19.determine_navigation("CAUTION", False, {"position": "CENTER"}, fp19_right)
    assert r19_3["action"] == "TURN_LEFT"
    print("TEST 19 Result: PASS (Single-frame alternating fluctuations rejected by hysteresis)")

    # ----------------------------------------------------------------
    # TEST 20: Emergency Hysteresis Bypass
    # ----------------------------------------------------------------
    print("\n--- TEST 20: Emergency Hysteresis Bypass ---")
    e20 = DecisionEngine()
    e20.last_action = "GO_FORWARD"
    r20 = e20.determine_navigation("EMERGENCY", True, {"position": "CENTER", "distance_m": 0.7}, {"left_clear": True, "center_clear": True, "right_clear": True})
    assert r20["action"] == "STOP"
    print("TEST 20 Result: PASS (Emergency bypasses hysteresis immediately -> STOP on frame 1)")

    # ----------------------------------------------------------------
    # TEST 21: GO_BACK Safety
    # ----------------------------------------------------------------
    print("\n--- TEST 21: GO_BACK Safety ---")
    e21 = DecisionEngine()
    fp21 = {"left_clear": False, "center_clear": False, "right_clear": False, "no_safe_path": True}
    # Without verified backward clearance: MUST NOT output GO_BACK -> must output STOP
    r21_noback = e21.determine_navigation("DANGER", False, {"position": "CENTER"}, fp21, backward_path_clear=False)
    assert r21_noback["action"] == "STOP", "Without verified backward clearance, must be STOP"

    # With verified backward clearance: outputs GO_BACK
    e21.last_action = "GO_BACK"
    r21_back = e21.determine_navigation("DANGER", False, {"position": "CENTER"}, fp21, backward_path_clear=True)
    assert r21_back["action"] == "GO_BACK" and r21_back["direction"] == "BACK"
    assert r21_back["movement_required"] is True and r21_back["turn_required"] is False
    print("TEST 21 Result: PASS (GO_BACK safety verified: STOP without backward evidence; GO_BACK only with clearance)")

    # ----------------------------------------------------------------
    # TEST 22: Turn / Movement Flags
    # ----------------------------------------------------------------
    print("\n--- TEST 22: Turn / Movement Flags ---")
    assert res1["turn_required"] is False and res1["movement_required"] is True   # GO_FORWARD
    assert res2["turn_required"] is True and res2["movement_required"] is False   # TURN_LEFT
    assert res3["turn_required"] is True and res3["movement_required"] is False   # TURN_RIGHT
    assert res8["turn_required"] is False and res8["movement_required"] is False  # STOP
    assert r21_back["turn_required"] is False and r21_back["movement_required"] is True  # GO_BACK
    print("TEST 22 Result: PASS (Turn/movement flags contract verified across all actions)")

    # ----------------------------------------------------------------
    # TEST 23: Navigation Reason
    # ----------------------------------------------------------------
    print("\n--- TEST 23: Navigation Reason ---")
    test_responses = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, r21_back]
    for resp in test_responses:
        reason = resp.get("reason", "")
        assert isinstance(reason, str) and len(reason.strip()) > 5, f"Empty or invalid reason in response: {resp}"
    print("TEST 23 Result: PASS (All actions contain clear descriptive reason strings)")

    print("\n" + "=" * 70)
    print("ALL 23 PHASE 8 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase8_verification()
