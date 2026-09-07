import time

import cv2
import numpy as np

from src.detector import YOLOv8TFLiteDetector


def main():

    print("=" * 80)
    print("YOLO DETECTOR COMPONENT TIMING TEST")
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
    # Load detector
    # --------------------------------------------------

    detector = YOLOv8TFLiteDetector(
        model_path="models/yolov8n.tflite",
        confidence_threshold=0.50,
        iou_threshold=0.45
    )

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("WARM-UP")
    print("=" * 80)

    detector.detect(frame)

    print("Warm-up complete.")

    # --------------------------------------------------
    # Number of runs
    # --------------------------------------------------

    runs = 10

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    # --------------------------------------------------
    # Profiling
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("PROFILING")
    print("=" * 80)

    for i in range(runs):

        # ==============================================
        # PREPROCESSING
        # ==============================================

        start_total = time.perf_counter()

        start_preprocess = time.perf_counter()

        input_tensor, scale, pad_x, pad_y = (
            detector.preprocess(frame)
        )

        end_preprocess = time.perf_counter()

        preprocess_time = (
            end_preprocess - start_preprocess
        )

        # ==============================================
        # TFLITE INFERENCE
        # ==============================================

        start_inference = time.perf_counter()

        detector.interpreter.set_tensor(
            detector.input_details[0]["index"],
            input_tensor
        )

        detector.interpreter.invoke()

        output = detector.interpreter.get_tensor(
            detector.output_details[0]["index"]
        )

        end_inference = time.perf_counter()

        inference_time = (
            end_inference - start_inference
        )

        # ==============================================
        # POSTPROCESSING
        # ==============================================

        start_postprocess = time.perf_counter()

        predictions = np.squeeze(
            output,
            axis=0
        )

        predictions = predictions.T

        detections = []

        for prediction in predictions:

            class_scores = prediction[4:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

            if confidence < detector.confidence_threshold:
                continue

            cx, cy, w, h = prediction[:4]

            # YOLO output is normalized
            cx *= 640
            cy *= 640
            w *= 640
            h *= 640

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Undo letterbox
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "box": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ]
                }
            )

        # Run detector's NMS implementation
        final_detections = detector.nms(
            detections
        )

        end_postprocess = time.perf_counter()

        postprocess_time = (
            end_postprocess - start_postprocess
        )

        # ==============================================
        # TOTAL
        # ==============================================

        end_total = time.perf_counter()

        total_time = (
            end_total - start_total
        )

        # Store
        preprocess_times.append(
            preprocess_time
        )

        inference_times.append(
            inference_time
        )

        postprocess_times.append(
            postprocess_time
        )

        total_times.append(
            total_time
        )

        print(
            f"Run {i + 1:02d} | "
            f"Pre: {preprocess_time * 1000:.2f} ms | "
            f"Inference: {inference_time * 1000:.2f} ms | "
            f"Post: {postprocess_time * 1000:.2f} ms | "
            f"Total: {total_time * 1000:.2f} ms | "
            f"Detections: {len(final_detections)}"
        )

    # --------------------------------------------------
    # Averages
    # --------------------------------------------------

    avg_preprocess = (
        sum(preprocess_times)
        / len(preprocess_times)
    )

    avg_inference = (
        sum(inference_times)
        / len(inference_times)
    )

    avg_postprocess = (
        sum(postprocess_times)
        / len(postprocess_times)
    )

    avg_total = (
        sum(total_times)
        / len(total_times)
    )

    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    fps = (
        1.0 / avg_total
        if avg_total > 0
        else 0.0
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("YOLO COMPONENT TIMING RESULTS")
    print("=" * 80)

    print()

    print(
        f"Average preprocessing:   "
        f"{avg_preprocess * 1000:.2f} ms"
    )

    print(
        f"Average TFLite inference:"
        f" {avg_inference * 1000:.2f} ms"
    )

    print(
        f"Average postprocessing:   "
        f"{avg_postprocess * 1000:.2f} ms"
    )

    print(
        f"Average total:            "
        f"{avg_total * 1000:.2f} ms"
    )

    print(
        f"Estimated detector FPS:   "
        f"{fps:.2f}"
    )

    print()

    # --------------------------------------------------
    # Percentage breakdown
    # --------------------------------------------------

    if avg_total > 0:

        preprocess_percent = (
            avg_preprocess
            / avg_total
            * 100
        )

        inference_percent = (
            avg_inference
            / avg_total
            * 100
        )

        postprocess_percent = (
            avg_postprocess
            / avg_total
            * 100
        )

        print("=" * 80)
        print("TIME BREAKDOWN")
        print("=" * 80)

        print()

        print(
            f"Preprocessing:   "
            f"{preprocess_percent:.1f}%"
        )

        print(
            f"Inference:      "
            f"{inference_percent:.1f}%"
        )

        print(
            f"Postprocessing:  "
            f"{postprocess_percent:.1f}%"
        )

    print()
    print("=" * 80)
    print("YOLO COMPONENT TIMING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()