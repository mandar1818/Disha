import sys
sys.path.insert(0, r"D:\SmartVisionAI_New")
import cv2
import numpy as np
from backend.yolo.detector import YOLODetector
from backend.depth.midas import MiDaSDepthEstimator
from backend.decision.engine import DecisionEngine

def detect_unknown_candidates(frame, depth_map, known_objects, frame_width, frame_height):
    h, w = depth_map.shape[:2]
    frame_area = float(w * h)

    # 1. Depth threshold: relative foreground objects (< 0.55)
    depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255).astype(np.uint8)
    fg_mask = (depth_u8 < int(0.55 * 255)).astype(np.uint8) * 255

    # 2. Morphological cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    # 3. Contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    engine = DecisionEngine()
    candidates = []

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)
        cnt_area = float(cv2.contourArea(cnt))

        # Size filters
        if box_area < 0.015 * frame_area or box_area > 0.45 * frame_area:
            continue
        if cnt_area < 0.010 * frame_area:
            continue

        # Aspect ratio
        ar = bw / float(bh)
        if ar > 4.5 or ar < 0.15:
            continue

        # Vertical location: must extend into bottom 70% of frame (not purely ceiling/sky)
        if (by + bh) / float(h) < 0.30:
            continue

        # Sample depth
        roi = depth_map[by : by + bh, bx : bx + bw]
        if roi.size == 0:
            continue
        cand_depth = float(np.median(roi))
        if cand_depth > 0.55:
            continue

        cand_box = {
            "x1": float(bx),
            "y1": float(by),
            "x2": float(bx + bw),
            "y2": float(by + bh),
        }

        # 4. Known YOLO box suppression
        suppressed = False
        for k_obj in known_objects:
            k_box = k_obj.get("bbox", {})
            kx1 = float(k_box.get("x1", 0))
            ky1 = float(k_box.get("y1", 0))
            kx2 = float(k_box.get("x2", 0))
            ky2 = float(k_box.get("y2", 0))

            ix1 = max(cand_box["x1"], kx1)
            iy1 = max(cand_box["y1"], ky1)
            ix2 = min(cand_box["x2"], kx2)
            iy2 = min(cand_box["y2"], ky2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter_area = iw * ih

            if inter_area <= 0:
                continue

            k_area = max(1.0, (kx2 - kx1) * (ky2 - ky1))
            union_area = box_area + k_area - inter_area
            iou = inter_area / union_area
            cand_overlap = inter_area / box_area
            k_overlap = inter_area / k_area

            if iou >= 0.25 or cand_overlap >= 0.35 or k_overlap >= 0.50:
                suppressed = True
                print(f"Candidate {cand_box} suppressed by {k_obj.get('class_name')} (iou={iou:.2f}, overlap={cand_overlap:.2f})")
                break

        if suppressed:
            continue

        # Confidence based on solidity and depth contrast
        solidity = cnt_area / max(1.0, box_area)
        depth_contrast = max(0.0, 0.60 - cand_depth) / 0.60
        conf = round(float(np.clip(0.40 + 0.30 * solidity + 0.30 * depth_contrast, 0.40, 0.85)), 4)

        position = engine.get_position(cand_box, frame_width)

        dist_info = engine.estimate_distance_and_steps(
            depth=cand_depth,
            bbox=cand_box,
            class_name="unknown",
            confidence=conf,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        dist_m = float(dist_info["estimated_distance_m"])
        if dist_m < 1.4:
            dist_cat = "VERY_CLOSE"
        elif dist_m <= 3.5:
            dist_cat = "NEAR"
        else:
            dist_cat = "FAR"

        risk = engine.calculate_risk(
            confidence=conf,
            depth=cand_depth,
            priority=0.60,
            position=position,
            bbox=cand_box,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if risk >= 0.70 or (position == "CENTER" and dist_m < 2.5):
            nav_relevance = "HIGH"
        elif risk >= 0.40 or dist_m <= 3.5:
            nav_relevance = "MEDIUM"
        elif risk >= 0.15:
            nav_relevance = "LOW"
        else:
            nav_relevance = "NONE"

        candidates.append({
            "bbox": cand_box,
            "cand_depth": cand_depth,
            "confidence": conf,
            "position": position,
            "distance_m": dist_m,
            "distance_category": dist_cat,
            "distance_confidence": dist_info["distance_confidence"],
            "risk_score": risk,
            "navigation_relevance": nav_relevance,
        })

    # Deduplicate candidates (NMS)
    kept = []
    candidates.sort(key=lambda c: c["risk_score"], reverse=True)
    for c in candidates:
        dup = False
        for k in kept:
            # check iou
            b1, b2 = c["bbox"], k["bbox"]
            ix1 = max(b1["x1"], b2["x1"])
            iy1 = max(b1["y1"], b2["y1"])
            ix2 = min(b1["x2"], b2["x2"])
            iy2 = min(b1["y2"], b2["y2"])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            ia = iw * ih
            a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
            a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
            if ia / (a1 + a2 - ia) >= 0.35:
                dup = True
                break
        if not dup:
            kept.append(c)

    # Assign IDs
    unknown_objects = []
    for idx, c in enumerate(kept[:3]):
        unknown_objects.append({
            "id": 1001 + idx,
            "position": c["position"],
            "bbox": c["bbox"],
            "distance_m": c["distance_m"],
            "distance_category": c["distance_category"],
            "distance_confidence": c["distance_confidence"],
            "risk_score": c["risk_score"],
            "navigation_relevance": c["navigation_relevance"],
            "confidence": c["confidence"],
            # internal / backward compat
            "class_name": "unknown",
            "depth": round(c["cand_depth"], 4),
        })

    return unknown_objects

def main():
    image_path = "backend/test.jpg"
    frame = cv2.imread(image_path)
    h, w = frame.shape[:2]

    detector = YOLODetector()
    known = detector.detect(frame)

    estimator = MiDaSDepthEstimator()
    depth_map = estimator.estimate_depth_map(frame)

    unknowns = detect_unknown_candidates(frame, depth_map, known, w, h)
    print(f"\nResulting Unknown Objects ({len(unknowns)}):")
    for u in unknowns:
        print(" ", u)

if __name__ == "__main__":
    main()
