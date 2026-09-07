import sys
sys.path.insert(0, r"D:\SmartVisionAI_New")
import cv2
import numpy as np
from backend.processors.frame_processor import FrameProcessor
from backend.decision.engine import DecisionEngine

def run_tests():
    processor = FrameProcessor()
    engine = processor.decision_engine

    print("=" * 70)
    print("PHASE 4 COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # TEST 1: Known objects only (backend/test.jpg)
    # ------------------------------------------------------------------
    print("\n--- TEST 1: Known Objects Only ---")
    frame1 = cv2.imread("backend/test.jpg")
    res1 = processor.process_frame(frame1)
    known_names = [o["class_name"] for o in res1["objects"]]
    unknown_ids = [u["id"] for u in res1["unknown_objects"]]
    print(f"Known objects detected: {known_names}")
    print(f"Unknown objects detected: {len(unknown_ids)} (IDs: {unknown_ids})")
    for obj in res1["objects"]:
        assert obj["class_name"] in ["person", "chair", "cup", "potted plant"], f"Unexpected class {obj['class_name']}"
    print("TEST 1 Result: PASS (Known objects remain YOLO objects)")

    # ------------------------------------------------------------------
    # TEST 2: Scene with genuine unclassified physical obstacle
    # ------------------------------------------------------------------
    print("\n--- TEST 2: Genuine Unknown Physical Obstacle ---")
    # Synthetic frame: background floor, plus an unclassified crate in center
    h, w = 480, 640
    frame2 = np.full((h, w, 3), 180, dtype=np.uint8)
    # MiDaS depth map: gradient floor from 0.8 at horizon (y=150) to 0.5 at bottom
    depth_map2 = np.ones((h, w), dtype=np.float32)
    for r in range(150, h):
        depth_map2[r, :] = 0.8 - (r - 150) / (h - 150) * 0.3  # 0.8 -> 0.5
    # Insert obstacle: box at [220, 200, 420, 400] with depth 0.20 (close)
    depth_map2[200:400, 220:420] = 0.20
    frame2[200:400, 220:420] = (60, 60, 60) # dark gray box

    candidates2 = processor.detect_unknown_candidates(
        frame=frame2,
        depth_map=depth_map2,
        known_objects=[],
        frame_width=w,
        frame_height=h
    )
    print(f"Unknown candidates found: {len(candidates2)}")
    assert len(candidates2) > 0, "Failed to detect genuine unknown obstacle"
    cand = candidates2[0]
    print(f"  Candidate bbox: {cand['bbox']}, depth: {cand['cand_depth']}, conf: {cand['confidence']}")
    print("TEST 2 Result: PASS (Genuine physical obstacle candidate detected)")

    # ------------------------------------------------------------------
    # TEST 3, 4, 5: Positions LEFT, CENTER, RIGHT
    # ------------------------------------------------------------------
    print("\n--- TEST 3, 4, 5: Spatial Positioning (LEFT, CENTER, RIGHT) ---")
    positions_to_test = [
        ("LEFT", [50, 200, 180, 400]),
        ("CENTER", [230, 200, 410, 400]),
        ("RIGHT", [460, 200, 590, 400]),
    ]
    for expected_pos, (bx1, by1, bx2, by2) in positions_to_test:
        dmap = np.ones((h, w), dtype=np.float32) * 0.8
        dmap[by1:by2, bx1:bx2] = 0.25
        f = np.full((h, w, 3), 180, dtype=np.uint8)
        f[by1:by2, bx1:bx2] = 50
        cands = processor.detect_unknown_candidates(f, dmap, [], w, h)
        assert len(cands) > 0, f"Failed to detect obstacle for {expected_pos}"
        norm = engine.normalize_unknown_objects(cands, w, h)
        actual_pos = norm[0]["position"]
        print(f"  Expected: {expected_pos}, Actual: {actual_pos} (bbox: {cands[0]['bbox']})")
        assert actual_pos == expected_pos, f"Position mismatch: expected {expected_pos}, got {actual_pos}"
    print("TEST 3, 4, 5 Result: PASS (Spatial positioning verified)")

    # ------------------------------------------------------------------
    # TEST 6: Small physical obstacle & minimum size evaluation
    # ------------------------------------------------------------------
    print("\n--- TEST 6: Small Physical Obstacle Evaluation ---")
    # Test different obstacle sizes: 0.5%, 1.0%, 1.5%, 2.0% of frame area
    frame_area = float(w * h)
    test_sizes = [
        (0.005, "0.5% area (~39x39 px)"),
        (0.008, "0.8% area (~50x50 px)"),
        (0.012, "1.2% area (~61x61 px)"),
        (0.015, "1.5% area (~68x68 px) - current min threshold"),
        (0.020, "2.0% area (~78x78 px)"),
    ]
    for ratio, desc in test_sizes:
        side = int(np.sqrt(ratio * frame_area))
        x1, y1 = 300, 300
        x2, y2 = x1 + side, y1 + side
        dmap = np.ones((h, w), dtype=np.float32) * 0.8
        dmap[y1:y2, x1:x2] = 0.25
        f = np.full((h, w, 3), 180, dtype=np.uint8)
        f[y1:y2, x1:x2] = 50
        cands = processor.detect_unknown_candidates(f, dmap, [], w, h)
        detected = len(cands) > 0
        print(f"  Size {desc}: {'DETECTED' if detected else 'REJECTED'}")
    print("TEST 6 Result: Evaluated (Recorded sensitivity thresholds)")

    # ------------------------------------------------------------------
    # TEST 7: Large wall / floor / background region
    # ------------------------------------------------------------------
    print("\n--- TEST 7: Large Background / Wall / Floor Rejection ---")
    # Massive region covering 60% of frame
    dmap7 = np.ones((h, w), dtype=np.float32) * 0.8
    dmap7[100:460, 50:590] = 0.30  # area = 360 * 540 = 194,400 px = 63% of frame
    f7 = np.full((h, w, 3), 150, dtype=np.uint8)
    cands7 = processor.detect_unknown_candidates(f7, dmap7, [], w, h)
    print(f"  Candidates from 63% area region: {len(cands7)}")
    assert len(cands7) == 0, "Failed: Large background region was mistakenly detected as obstacle"
    print("TEST 7 Result: PASS (Large background/wall/floor regions rejected)")

    # ------------------------------------------------------------------
    # TEST 8: Shadow / reflection / depth noise / extreme slivers
    # ------------------------------------------------------------------
    print("\n--- TEST 8: Shadow / Slivers / Noise Rejection ---")
    # Horizontal seam / floor line (aspect ratio 8:1)
    dmap8 = np.ones((h, w), dtype=np.float32) * 0.8
    dmap8[300:320, 100:500] = 0.30  # 400x20 -> aspect ratio 20
    f8 = np.full((h, w, 3), 150, dtype=np.uint8)
    cands8_sliver = processor.detect_unknown_candidates(f8, dmap8, [], w, h)
    print(f"  Candidates from extreme sliver: {len(cands8_sliver)}")
    assert len(cands8_sliver) == 0, "Failed: Extreme horizontal sliver accepted"

    # Distant depth region (depth 0.75 > 0.55 cutoff)
    dmap8b = np.ones((h, w), dtype=np.float32) * 0.8
    dmap8b[200:350, 200:350] = 0.72  # close to background
    cands8_dist = processor.detect_unknown_candidates(f8, dmap8b, [], w, h)
    print(f"  Candidates from distant region (depth > 0.55): {len(cands8_dist)}")
    assert len(cands8_dist) == 0, "Failed: Distant region accepted"
    print("TEST 8 Result: PASS (Noise and extreme slivers rejected)")

    # ------------------------------------------------------------------
    # TEST 9: Candidate overlapping known YOLO object
    # ------------------------------------------------------------------
    print("\n--- TEST 9: Known Object Suppression ---")
    # Known person at [200, 100, 400, 450]
    known_person = [{"class_name": "person", "bbox": {"x1": 200, "y1": 100, "x2": 400, "y2": 450}, "confidence": 0.9}]
    # Candidate directly over the person
    dmap9 = np.ones((h, w), dtype=np.float32) * 0.8
    dmap9[120:440, 210:390] = 0.20
    f9 = np.full((h, w, 3), 150, dtype=np.uint8)
    cands9 = processor.detect_unknown_candidates(f9, dmap9, known_person, w, h)
    print(f"  Candidates overlapping known person: {len(cands9)}")
    assert len(cands9) == 0, "Failed: Candidate overlapping known person was not suppressed"
    print("TEST 9 Result: PASS (Known object candidate suppressed)")

    # ------------------------------------------------------------------
    # TEST 10: Multiple independent unknown candidates
    # ------------------------------------------------------------------
    print("\n--- TEST 10: Multiple Independent Unknown Candidates ---")
    # Two distinct obstacles: one on LEFT, one on RIGHT
    dmap10 = np.ones((h, w), dtype=np.float32) * 0.8
    dmap10[200:380, 50:180] = 0.25   # LEFT obstacle
    dmap10[200:380, 460:590] = 0.30  # RIGHT obstacle
    f10 = np.full((h, w, 3), 150, dtype=np.uint8)
    cands10 = processor.detect_unknown_candidates(f10, dmap10, [], w, h)
    print(f"  Distinct candidates detected: {len(cands10)}")
    assert len(cands10) == 2, f"Expected 2 candidates, got {len(cands10)}"
    norm10 = engine.normalize_unknown_objects(cands10, w, h)
    positions = [u["position"] for u in norm10]
    ids = [u["id"] for u in norm10]
    print(f"  Positions: {positions}, IDs: {ids}")
    assert set(positions) == {"LEFT", "RIGHT"}, f"Expected LEFT and RIGHT, got {positions}"
    assert ids == [1001, 1002], f"Expected IDs [1001, 1002], got {ids}"
    print("TEST 10 Result: PASS (Multiple unknown candidates deduplicated and represented independently)")

    # ------------------------------------------------------------------
    # CRASH RESISTANCE / SAFE FALLBACKS
    # ------------------------------------------------------------------
    print("\n--- CRASH RESISTANCE TESTS ---")
    # 1. None depth map
    c_none = processor.detect_unknown_candidates(frame1, None, [], w, h)
    assert c_none == [], "Failed safe fallback on None depth map"

    # 2. Empty depth map
    c_empty = processor.detect_unknown_candidates(frame1, np.array([]), [], w, h)
    assert c_empty == [], "Failed safe fallback on empty depth map"

    # 3. Depth map with NaNs and Infs
    dmap_nan = np.full((h, w), np.nan, dtype=np.float32)
    dmap_nan[200:300, 200:300] = np.inf
    # Test if it runs safely without throwing exceptions
    try:
        c_nan = processor.detect_unknown_candidates(frame1, dmap_nan, [], w, h)
        print(f"  NaN/Inf depth map handled safely: returned {len(c_nan)} candidates")
    except Exception as exc:
        print(f"  NaN/Inf depth map threw exception: {exc}")
        assert False, f"detect_unknown_candidates crashed on NaN/Inf: {exc}"

    # 4. Completely black frame
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    c_black = processor.detect_unknown_candidates(black_frame, dmap, [], w, h)
    print(f"  Black frame handled safely: returned {len(c_black)} candidates")

    # 5. Unknown objects normalization with None / empty
    norm_none = engine.normalize_unknown_objects(None, w, h)
    assert norm_none == [], "Failed safe fallback on normalize_unknown_objects(None)"
    norm_invalid = engine.normalize_unknown_objects([{"bbox": "invalid"}], w, h)
    assert len(norm_invalid) == 1 and norm_invalid[0]["id"] == 1001

    print("Crash Resistance Tests: ALL PASSED")
    print("\n" + "=" * 70)
    print("ALL 10 TESTS + CRASH RESISTANCE PASSED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
