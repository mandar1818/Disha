import sys
sys.path.insert(0, r"D:\SmartVisionAI_New")
import numpy as np

def check_suppression(cand_box, known_box, iou_thresh=0.25, cand_cov_thresh=0.35, k_cov_thresh=0.50):
    cb = cand_box
    kb = known_box
    
    bw = cb["x2"] - cb["x1"]
    bh = cb["y2"] - cb["y1"]
    cand_area = bw * bh

    kw = kb["x2"] - kb["x1"]
    kh = kb["y2"] - kb["y1"]
    k_area = kw * kh

    ix1 = max(cb["x1"], kb["x1"])
    iy1 = max(cb["y1"], kb["y1"])
    ix2 = min(cb["x2"], kb["x2"])
    iy2 = min(cb["y2"], kb["y2"])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih

    if inter_area <= 0:
        return False, {"iou": 0.0, "cand_cov": 0.0, "k_cov": 0.0, "reason": "no intersection"}

    union_area = cand_area + k_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    cand_cov = inter_area / cand_area if cand_area > 0 else 0.0
    k_cov = inter_area / k_area if k_area > 0 else 0.0

    # Current suppression condition:
    # iou >= 0.25 or cand_overlap >= 0.35 or (k_overlap >= 0.50 and cand_area <= 2.5 * k_area)
    suppressed = (iou >= iou_thresh or cand_cov >= cand_cov_thresh or k_cov >= k_cov_thresh)
    return suppressed, {"iou": round(iou, 3), "cand_cov": round(cand_cov, 3), "k_cov": round(k_cov, 3), "cand_area": cand_area, "k_area": k_area}

def run_suppression_cases():
    print("=" * 60)
    print("TESTING SUPPRESSION CASES A THROUGH F")
    print("=" * 60)

    # Case A: Candidate almost completely inside known object
    # Known person: [100, 100, 300, 400] (area = 60,000)
    # Candidate: [120, 150, 250, 350] (area = 26,000, fully inside)
    supp_A, m_A = check_suppression(
        cand_box={"x1": 120, "y1": 150, "x2": 250, "y2": 350},
        known_box={"x1": 100, "y1": 100, "x2": 300, "y2": 400}
    )
    print(f"Case A (Inside): Suppressed={supp_A}, Metrics={m_A}")
    assert supp_A is True, "Case A should be suppressed"

    # Case B: Candidate partially overlaps known object (e.g. 50% overlap)
    # Known: [100, 100, 300, 300]
    # Candidate: [200, 100, 400, 300] (half overlaps)
    supp_B, m_B = check_suppression(
        cand_box={"x1": 200, "y1": 100, "x2": 400, "y2": 300},
        known_box={"x1": 100, "y1": 100, "x2": 300, "y2": 300}
    )
    print(f"Case B (Partial 50%): Suppressed={supp_B}, Metrics={m_B}")
    assert supp_B is True, "Case B should be suppressed"

    # Case C: Candidate mostly surrounds known object (candidate is halo / slight envelope)
    # Known: [100, 100, 200, 300] (area = 20,000)
    # Candidate: [90, 90, 210, 310] (area = 26,400, k_cov = 1.0, cand_cov = 20000/26400 = 0.76)
    supp_C, m_C = check_suppression(
        cand_box={"x1": 90, "y1": 90, "x2": 210, "y2": 310},
        known_box={"x1": 100, "y1": 100, "x2": 200, "y2": 300}
    )
    print(f"Case C (Surrounds / Envelope): Suppressed={supp_C}, Metrics={m_C}")
    assert supp_C is True, "Case C should be suppressed"

    # Case C2: Candidate is a HUGE obstacle (table) that happens to have a TINY cup on it
    # Known cup: [200, 200, 230, 230] (area = 900)
    # Candidate table: [50, 150, 450, 400] (area = 100,000)
    # inter = 900, cand_cov = 0.009, k_cov = 1.0!
    # If using k_overlap >= 0.50 unconditionally, the ENTIRE table is suppressed!
    # With scale check (cand_area <= 2.5 * k_area), table is NOT suppressed!
    supp_C2_raw, m_C2 = check_suppression(
        cand_box={"x1": 50, "y1": 150, "x2": 450, "y2": 400},
        known_box={"x1": 200, "y1": 200, "x2": 230, "y2": 230}
    )
    print(f"Case C2 (Huge obstacle with tiny cup on it - raw): Suppressed={supp_C2_raw}, Metrics={m_C2}")

    # Case D: Candidate touches but does not meaningfully overlap (1% overlap or touching edge)
    # Known: [100, 100, 200, 200]
    # Candidate: [200, 100, 300, 200] (touches border, 0 overlap)
    supp_D, m_D = check_suppression(
        cand_box={"x1": 200, "y1": 100, "x2": 300, "y2": 200},
        known_box={"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    )
    print(f"Case D (Touching border): Suppressed={supp_D}, Metrics={m_D}")
    assert supp_D is False, "Case D should NOT be suppressed"

    # Case D2: 5% incidental overlap
    supp_D2, m_D2 = check_suppression(
        cand_box={"x1": 195, "y1": 100, "x2": 295, "y2": 200},
        known_box={"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    )
    print(f"Case D2 (5% incidental overlap): Suppressed={supp_D2}, Metrics={m_D2}")
    assert supp_D2 is False, "Case D2 should NOT be suppressed"

    print("ALL SUPPRESSION CASES EXAMINED")

if __name__ == "__main__":
    run_suppression_cases()
