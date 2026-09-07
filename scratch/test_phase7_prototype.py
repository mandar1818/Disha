"""
Prototype test script for Phase 7 Free-Path Analysis & Safe Walking Corridor Detection.
Tests all 16 required test cases.
"""

import sys
from typing import Dict, Any, List, Optional, Tuple

class PrototypeDecisionEngine:
    CLOSE_DEPTH = 0.30

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            r = float(val)
            if r != r:
                return default
            return r
        except (ValueError, TypeError):
            return default

    def normalize_depth(self, d: Any) -> float:
        return max(0.0, min(1.0, self._safe_float(d, 1.0)))

    def _compute_interval_coverage(self, intervals: List[Tuple[float, float]], span_start: float, span_end: float) -> float:
        if not intervals:
            return 0.0
        clamped = []
        for s, e in intervals:
            cs = max(span_start, s)
            ce = min(span_end, e)
            if ce > cs:
                clamped.append((cs, ce))
        if not clamped:
            return 0.0
        clamped.sort(key=lambda x: x[0])
        merged = [clamped[0]]
        for cur in clamped[1:]:
            prev_s, prev_e = merged[-1]
            if cur[0] <= prev_e:
                merged[-1] = (prev_s, max(prev_e, cur[1]))
            else:
                merged.append(cur)
        total_len = sum(e - s for s, e in merged)
        return min(1.0, total_len / (span_end - span_start))

    def analyze_free_path(
        self,
        scene_depth: Optional[Dict[str, Any]] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        unknown_objects: Optional[List[Dict[str, Any]]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> Dict[str, Any]:
        scene = scene_depth if isinstance(scene_depth, dict) else {}

        left_depth = self.normalize_depth(scene.get("left_depth", scene.get("left", 1.0)))
        center_depth = self.normalize_depth(scene.get("center_region_depth", scene.get("center_depth", 1.0)))
        right_depth = self.normalize_depth(scene.get("right_depth", scene.get("right", 1.0)))

        scene_left_risk = max(0.0, 1.0 - left_depth)
        scene_center_risk = max(0.0, 1.0 - center_depth)
        scene_right_risk = max(0.0, 1.0 - right_depth)

        width = float(frame_width) if frame_width and frame_width > 0 else 640.0
        height = float(frame_height) if frame_height and frame_height > 0 else 480.0

        all_obstacles: List[Dict[str, Any]] = []
        if isinstance(objects, list):
            all_obstacles.extend(objects)
        if isinstance(unknown_objects, list):
            all_obstacles.extend(unknown_objects)

        center_intervals = []
        left_intervals = []
        right_intervals = []

        center_contribs = []
        left_contribs = []
        right_contribs = []

        center_blocking_flags = []
        left_blocking_flags = []
        right_blocking_flags = []

        max_center_dist_weight = 0.0
        max_left_dist_weight = 0.0
        max_right_dist_weight = 0.0

        has_low_conf = False
        has_med_conf = False

        for obs in all_obstacles:
            bbox = obs.get("bbox") if isinstance(obs.get("bbox"), dict) else {}
            x1 = max(0.0, min(1.0, self._safe_float(bbox.get("x1", 0)) / width))
            y1 = max(0.0, min(1.0, self._safe_float(bbox.get("y1", 0)) / height))
            x2 = max(0.0, min(1.0, self._safe_float(bbox.get("x2", 0)) / width))
            y2 = max(0.0, min(1.0, self._safe_float(bbox.get("y2", 0)) / height))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            obj_width = x2 - x1

            # Vertical walking relevance:
            # Bottom edge y2 < 0.35 is in top background/sky/ceiling -> not a walking hazard
            if y2 < 0.35:
                vertical_factor = 0.0
            else:
                vertical_factor = min(1.0, max(0.0, (y2 - 0.30) / 0.30))

            # Horizontal overlaps
            cov_c = max(0.0, min(x2, 0.65) - max(x1, 0.35)) / 0.30
            cov_l = max(0.0, min(x2, 0.35) - max(x1, 0.0)) / 0.35
            cov_r = max(0.0, min(x2, 1.0) - max(x1, 0.65)) / 0.35

            dist_m = self._safe_float(obs.get("distance_m", obs.get("estimated_distance_m", 2.0)))
            dist_cat = str(obs.get("distance_category", "")).upper()
            if not dist_cat or dist_cat == "NONE":
                if dist_m < 1.4:
                    dist_cat = "VERY_CLOSE"
                elif dist_m < 3.5:
                    dist_cat = "NEAR"
                else:
                    dist_cat = "FAR"

            risk = self._safe_float(obs.get("risk_score", 0.0))

            if dist_cat == "VERY_CLOSE":
                dist_weight = 1.0
            elif dist_cat == "NEAR":
                dist_weight = 0.85
            elif dist_cat == "FAR":
                dist_weight = 0.50 if obj_width > 0.60 else 0.25
            else:
                dist_weight = 0.60

            # Confidence assessment
            dist_conf = str(obs.get("distance_confidence", "HIGH")).upper()
            det_conf = self._safe_float(obs.get("confidence", 0.8))
            is_unknown = bool(obs.get("class_name") in ("unknown_obstacle", "obstacle", None, ""))
            if dist_conf == "LOW" or det_conf < 0.40:
                has_low_conf = True
            elif dist_conf == "MEDIUM" or det_conf < 0.65 or is_unknown:
                has_med_conf = True

            if vertical_factor > 0.0:
                # Center
                if cov_c > 0.02:
                    center_intervals.append((x1, x2))
                    c_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_c * 1.5)
                    center_contribs.append(c_contrib)
                    max_center_dist_weight = max(max_center_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_c >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_c >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_c >= 0.25) or \
                           (cov_c >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            center_blocking_flags.append(True)

                # Left
                if cov_l > 0.02:
                    left_intervals.append((x1, x2))
                    l_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_l * 1.5)
                    left_contribs.append(l_contrib)
                    max_left_dist_weight = max(max_left_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_l >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_l >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_l >= 0.25) or \
                           (cov_l >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            left_blocking_flags.append(True)

                # Right
                if cov_r > 0.02:
                    right_intervals.append((x1, x2))
                    r_contrib = risk * dist_weight * vertical_factor * min(1.0, cov_r * 1.5)
                    right_contribs.append(r_contrib)
                    max_right_dist_weight = max(max_right_dist_weight, dist_weight)
                    if vertical_factor >= 0.35:
                        if (dist_cat == "VERY_CLOSE" and cov_r >= 0.15 and risk >= 0.40) or \
                           (dist_cat == "NEAR" and cov_r >= 0.20 and risk >= 0.45) or \
                           (risk >= 0.65 and cov_r >= 0.25) or \
                           (cov_r >= 0.50 and dist_cat in ("VERY_CLOSE", "NEAR")):
                            right_blocking_flags.append(True)

        # Region coverage
        center_coverage = self._compute_interval_coverage(center_intervals, 0.35, 0.65)
        left_coverage = self._compute_interval_coverage(left_intervals, 0.0, 0.35)
        right_coverage = self._compute_interval_coverage(right_intervals, 0.65, 1.0)

        # Region risks
        max_c_risk = max(center_contribs) if center_contribs else 0.0
        max_l_risk = max(left_contribs) if left_contribs else 0.0
        max_r_risk = max(right_contribs) if right_contribs else 0.0

        center_risk = min(1.0, max(scene_center_risk, max_c_risk, center_coverage * max_center_dist_weight))
        left_risk = min(1.0, max(scene_left_risk, max_l_risk, left_coverage * max_left_dist_weight))
        right_risk = min(1.0, max(scene_right_risk, max_r_risk, right_coverage * max_right_dist_weight))

        # Clearance decisions
        center_blocked = bool(
            center_depth <= self.CLOSE_DEPTH
            or any(center_blocking_flags)
            or center_risk >= 0.50
            or (center_coverage >= 0.45 and max_center_dist_weight >= 0.70)
        )
        center_clear = bool(not center_blocked)

        left_blocked = bool(
            left_depth <= self.CLOSE_DEPTH
            or any(left_blocking_flags)
            or left_risk >= 0.50
            or (left_coverage >= 0.45 and max_left_dist_weight >= 0.70)
        )
        left_clear = bool(not left_blocked)

        right_blocked = bool(
            right_depth <= self.CLOSE_DEPTH
            or any(right_blocking_flags)
            or right_risk >= 0.50
            or (right_coverage >= 0.45 and max_right_dist_weight >= 0.70)
        )
        right_clear = bool(not right_blocked)

        safe_directions = []
        if left_clear:
            safe_directions.append("LEFT")
        if center_clear:
            safe_directions.append("CENTER")
        if right_clear:
            safe_directions.append("RIGHT")

        no_safe_path = bool(len(safe_directions) == 0)

        # Best direction selection
        if center_clear and (center_risk <= min(left_risk, right_risk) + 0.15):
            best_direction = "CENTER"
        elif safe_directions:
            # Choose from safe directions with lowest risk
            best_direction = min(safe_directions, key=lambda d: {"LEFT": left_risk, "CENTER": center_risk, "RIGHT": right_risk}[d])
        else:
            # All blocked: least dangerous direction
            risk_map = {"LEFT": left_risk, "CENTER": center_risk, "RIGHT": right_risk}
            best_direction = min(risk_map, key=risk_map.get)

        # Available corridor width in best direction
        if best_direction == "CENTER":
            corridor_width = round(0.30 * max(0.0, 1.0 - center_coverage), 4)
        elif best_direction == "LEFT":
            corridor_width = round(0.35 * max(0.0, 1.0 - left_coverage), 4)
        else:
            corridor_width = round(0.35 * max(0.0, 1.0 - right_coverage), 4)

        if has_low_conf:
            path_confidence = "LOW"
        elif has_med_conf:
            path_confidence = "MEDIUM"
        else:
            path_confidence = "HIGH"

        return {
            "left_clear": left_clear,
            "center_clear": center_clear,
            "right_clear": right_clear,
            "left_depth": round(left_depth, 4),
            "center_depth": round(center_depth, 4),
            "right_depth": round(right_depth, 4),
            "left_risk": round(left_risk, 4),
            "center_risk": round(center_risk, 4),
            "right_risk": round(right_risk, 4),
            "best_direction": best_direction,

            # Phase 7 canonical / optional corridor fields
            "center_blocked": center_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "center_coverage": round(center_coverage, 4),
            "left_coverage": round(left_coverage, 4),
            "right_coverage": round(right_coverage, 4),
            "corridor_width": corridor_width,
            "safe_directions": safe_directions,
            "no_safe_path": no_safe_path,
            "path_confidence": path_confidence,

            # Backward compatibility
            "left": round(left_depth, 4),
            "center": round(center_depth, 4),
            "right": round(right_depth, 4),
            "is_center_blocked": center_blocked,
        }

def run_tests():
    engine = PrototypeDecisionEngine()
    print("=" * 60)
    print("RUNNING PROTOTYPE PHASE 7 TESTS 1-16")
    print("=" * 60)

    # TEST 1: Empty scene
    res1 = engine.analyze_free_path(scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8})
    assert res1["left_clear"] and res1["center_clear"] and res1["right_clear"], f"Test 1 failed: {res1}"
    assert res1["best_direction"] == "CENTER"
    print("Test 1 PASS: Empty scene -> all clear, best=CENTER")

    # TEST 2: One far small center object
    res2 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{
            "class_name": "cup",
            "bbox": {"x1": 300, "y1": 200, "x2": 340, "y2": 260},
            "distance_m": 4.5,
            "distance_category": "FAR",
            "risk_score": 0.15
        }]
    )
    assert res2["center_clear"] is True, f"Test 2 failed: {res2}"
    print("Test 2 PASS: Far small center object -> center remains clear")

    # TEST 3: One near center chair
    res3 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{
            "class_name": "chair",
            "bbox": {"x1": 240, "y1": 200, "x2": 400, "y2": 440},
            "distance_m": 1.8,
            "distance_category": "NEAR",
            "risk_score": 0.65
        }]
    )
    assert res3["center_clear"] is False and res3["center_blocked"] is True, f"Test 3 failed: {res3}"
    print("Test 3 PASS: Near center chair -> center blocked")

    # TEST 4: One near center person
    res4 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{
            "class_name": "person",
            "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460},
            "distance_m": 1.5,
            "distance_category": "NEAR",
            "risk_score": 0.70
        }]
    )
    assert res4["center_clear"] is False and res4["center_blocked"] is True, f"Test 4 failed: {res4}"
    print("Test 4 PASS: Near center person -> center blocked")

    # TEST 5: Large unknown center obstacle
    res5 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        unknown_objects=[{
            "id": 1001,
            "bbox": {"x1": 230, "y1": 200, "x2": 410, "y2": 440},
            "distance_m": 1.7,
            "distance_category": "NEAR",
            "risk_score": 0.65
        }]
    )
    assert res5["center_clear"] is False and res5["center_blocked"] is True, f"Test 5 failed: {res5}"
    print("Test 5 PASS: Large unknown center obstacle -> center blocked")

    # TEST 6: LEFT obstacle only
    res6 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{
            "class_name": "chair",
            "bbox": {"x1": 30, "y1": 200, "x2": 180, "y2": 440},
            "distance_m": 1.5,
            "distance_category": "NEAR",
            "risk_score": 0.65
        }]
    )
    assert res6["left_clear"] is False and res6["center_clear"] is True and res6["right_clear"] is True, f"Test 6 failed: {res6}"
    print("Test 6 PASS: LEFT obstacle only -> left blocked, center & right clear")

    # TEST 7: RIGHT obstacle only
    res7 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{
            "class_name": "chair",
            "bbox": {"x1": 460, "y1": 200, "x2": 610, "y2": 440},
            "distance_m": 1.5,
            "distance_category": "NEAR",
            "risk_score": 0.65
        }]
    )
    assert res7["right_clear"] is False and res7["center_clear"] is True and res7["left_clear"] is True, f"Test 7 failed: {res7}"
    print("Test 7 PASS: RIGHT obstacle only -> right blocked, center & left clear")

    # TEST 8: LEFT + CENTER obstacles
    res8 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 180, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65},
            {"class_name": "person", "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70}
        ]
    )
    assert res8["left_clear"] is False and res8["center_clear"] is False and res8["right_clear"] is True, f"Test 8 failed: {res8}"
    assert res8["best_direction"] == "RIGHT", f"Test 8 best_direction failed: {res8}"
    print("Test 8 PASS: LEFT + CENTER obstacles -> right clear, best=RIGHT")

    # TEST 9: CENTER + RIGHT obstacles
    res9 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "person", "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70},
            {"class_name": "chair", "bbox": {"x1": 460, "y1": 200, "x2": 610, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65}
        ]
    )
    assert res9["center_clear"] is False and res9["right_clear"] is False and res9["left_clear"] is True, f"Test 9 failed: {res9}"
    assert res9["best_direction"] == "LEFT", f"Test 9 best_direction failed: {res9}"
    print("Test 9 PASS: CENTER + RIGHT obstacles -> left clear, best=LEFT")

    # TEST 10: LEFT + RIGHT obstacles
    res10 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 180, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65},
            {"class_name": "chair", "bbox": {"x1": 460, "y1": 200, "x2": 610, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65}
        ]
    )
    assert res10["left_clear"] is False and res10["right_clear"] is False and res10["center_clear"] is True, f"Test 10 failed: {res10}"
    assert res10["best_direction"] == "CENTER", f"Test 10 best_direction failed: {res10}"
    print("Test 10 PASS: LEFT + RIGHT obstacles -> center clear, best=CENTER")

    # TEST 11: All three paths blocked
    res11 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 180, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65},
            {"class_name": "person", "bbox": {"x1": 260, "y1": 100, "x2": 380, "y2": 460}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70},
            {"class_name": "chair", "bbox": {"x1": 460, "y1": 200, "x2": 610, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65}
        ]
    )
    assert res11["left_clear"] is False and res11["center_clear"] is False and res11["right_clear"] is False, f"Test 11 failed: {res11}"
    assert res11["no_safe_path"] is True, f"Test 11 no_safe_path failed: {res11}"
    assert len(res11["safe_directions"]) == 0
    print("Test 11 PASS: All three paths blocked -> no safe path")

    # TEST 12: Known + unknown obstacles
    res12 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "chair", "bbox": {"x1": 30, "y1": 200, "x2": 180, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.65}
        ],
        unknown_objects=[
            {"id": 1001, "bbox": {"x1": 230, "y1": 200, "x2": 410, "y2": 440}, "distance_m": 1.7, "distance_category": "NEAR", "risk_score": 0.65}
        ]
    )
    assert res12["left_clear"] is False and res12["center_clear"] is False and res12["right_clear"] is True, f"Test 12 failed: {res12}"
    print("Test 12 PASS: Known + unknown obstacles both participate in path analysis")

    # TEST 13: Multiple center obstacles
    res13 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "chair", "bbox": {"x1": 230, "y1": 200, "x2": 320, "y2": 440}, "distance_m": 2.0, "distance_category": "NEAR", "risk_score": 0.55},
            {"class_name": "box", "bbox": {"x1": 320, "y1": 220, "x2": 410, "y2": 440}, "distance_m": 2.2, "distance_category": "NEAR", "risk_score": 0.50}
        ]
    )
    assert res13["center_clear"] is False and res13["center_blocked"] is True, f"Test 13 failed: {res13}"
    print("Test 13 PASS: Multiple center obstacles -> center blocked")

    # TEST 14: Small far obstacle + large near obstacle
    res14 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[
            {"class_name": "cup", "bbox": {"x1": 300, "y1": 200, "x2": 330, "y2": 240}, "distance_m": 5.0, "distance_category": "FAR", "risk_score": 0.15},
            {"class_name": "chair", "bbox": {"x1": 240, "y1": 200, "x2": 400, "y2": 440}, "distance_m": 1.5, "distance_category": "NEAR", "risk_score": 0.70}
        ]
    )
    assert res14["center_clear"] is False and res14["center_blocked"] is True, f"Test 14 failed: {res14}"
    print("Test 14 PASS: Small far + large near obstacle -> near dominates blocking")

    # TEST 15: Top-of-frame/background object
    res15 = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{
            "class_name": "light",
            "bbox": {"x1": 250, "y1": 20, "x2": 390, "y2": 120},  # y2 = 120/480 = 0.25 < 0.35
            "distance_m": 1.2,
            "distance_category": "VERY_CLOSE",
            "risk_score": 0.75
        }]
    )
    assert res15["center_clear"] is True and res15["center_blocked"] is False, f"Test 15 failed: {res15}"
    print("Test 15 PASS: Top-of-frame object -> does not incorrectly block walking corridor")

    # TEST 16: Symmetry test
    res16_left = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{"class_name": "chair", "bbox": {"x1": 50, "y1": 200, "x2": 200, "y2": 440}, "distance_m": 1.6, "distance_category": "NEAR", "risk_score": 0.65}]
    )
    # Mirrored to right: 640 - 200 = 440, 640 - 50 = 590
    res16_right = engine.analyze_free_path(
        scene_depth={"left_depth": 0.8, "center_region_depth": 0.8, "right_depth": 0.8},
        objects=[{"class_name": "chair", "bbox": {"x1": 440, "y1": 200, "x2": 590, "y2": 440}, "distance_m": 1.6, "distance_category": "NEAR", "risk_score": 0.65}]
    )
    assert res16_left["left_clear"] == res16_right["right_clear"] == False
    assert res16_left["right_clear"] == res16_right["left_clear"] == True
    assert res16_left["left_risk"] == res16_right["right_risk"]
    assert res16_left["right_risk"] == res16_right["left_risk"]
    print(f"Test 16 PASS: Symmetry verified: Left risk {res16_left['left_risk']} == Right mirrored risk {res16_right['right_risk']}")

    print("=" * 60)
    print("ALL 16 PROTOTYPE UNIT TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
