import time

import cv2

from src.detector import YOLOv8TFLiteDetector
from src.depth_estimator import DepthEstimator


def main():

    print("=" * 80)
    print("YOLO + MiDaS MODEL TIMING TEST")
    print("=" * 80)

    image_path = "tests/test.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print()
        print("ERROR: Could not load test image.")
        print(f"Expected: {image_path}")
        return

    height, width = frame.shape[:2]

    print()
    print(f"Image size: {width} x {height}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------

    print()
    print("Loading YOLO...")

    detector = YOLOv8TFLiteDetector(
        model_path="models/yolov8n.tflite",
        confidence_threshold=0.50,
        iou_threshold=0.45
    )

    print()
    print("Loading MiDaS...")

    depth_estimator = DepthEstimator(
        model_path="models/midas_small.onnx"
    )

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("WARM-UP")
    print("=" * 80)

    print()
    print("Running YOLO warm-up...")

    detector.detect(frame)

    print("Running MiDaS warm-up...")

    depth_estimator.estimate(frame)

    print("Warm-up complete.")

    # --------------------------------------------------
    # Test settings
    # --------------------------------------------------

    number_of_runs = 10

    yolo_times = []
    midas_times = []

    # --------------------------------------------------
    # YOLO timing
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("YOLO TIMING")
    print("=" * 80)

    for i in range(number_of_runs):

        start = time.perf_counter()

        detections = detector.detect(frame)

        end = time.perf_counter()

        elapsed = end - start

        yolo_times.append(elapsed)

        print(
            f"Run {i + 1:02d} | "
            f"{elapsed * 1000:.2f} ms | "
            f"Detections: {len(detections)}"
        )

    # --------------------------------------------------
    # MiDaS timing
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("MiDaS TIMING")
    print("=" * 80)

    for i in range(number_of_runs):

        start = time.perf_counter()

        depth_map = depth_estimator.estimate(frame)

        end = time.perf_counter()

        elapsed = end - start

        midas_times.append(elapsed)

        print(
            f"Run {i + 1:02d} | "
            f"{elapsed * 1000:.2f} ms | "
            f"Depth shape: {depth_map.shape}"
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    avg_yolo = sum(yolo_times) / len(yolo_times)
    avg_midas = sum(midas_times) / len(midas_times)

    min_yolo = min(yolo_times)
    max_yolo = max(yolo_times)

    min_midas = min(midas_times)
    max_midas = max(midas_times)

    # --------------------------------------------------
    # Estimated performance
    # --------------------------------------------------

    # If both models run on every frame:
    full_pipeline_time = (
        avg_yolo + avg_midas
    )

    full_pipeline_fps = (
        1.0 / full_pipeline_time
        if full_pipeline_time > 0
        else 0.0
    )

    # With MiDaS running once every 3 frames:
    #
    # Average depth cost per frame =
    # MiDaS time / 3
    #
    optimized_model_time = (
        avg_yolo +
        (avg_midas / 3.0)
    )

    optimized_fps = (
        1.0 / optimized_model_time
        if optimized_model_time > 0
        else 0.0
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("MODEL TIMING RESULTS")
    print("=" * 80)

    print()

    print(
        f"YOLO average time:       "
        f"{avg_yolo * 1000:.2f} ms"
    )

    print(
        f"YOLO minimum time:       "
        f"{min_yolo * 1000:.2f} ms"
    )

    print(
        f"YOLO maximum time:       "
        f"{max_yolo * 1000:.2f} ms"
    )

    print()

    print(
        f"MiDaS average time:      "
        f"{avg_midas * 1000:.2f} ms"
    )

    print(
        f"MiDaS minimum time:      "
        f"{min_midas * 1000:.2f} ms"
    )

    print(
        f"MiDaS maximum time:      "
        f"{max_midas * 1000:.2f} ms"
    )

    print()

    print(
        f"Estimated FPS if both run "
        f"every frame:            "
        f"{full_pipeline_fps:.2f}"
    )

    print(
        f"Estimated FPS with MiDaS "
        f"every 3 frames:         "
        f"{optimized_fps:.2f}"
    )

    print()

    print("=" * 80)
    print("MODEL TIMING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()