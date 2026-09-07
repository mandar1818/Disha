import time

import cv2

from src.edge_ai_pipeline import EdgeAIPipeline


def main():

    print("=" * 80)
    print("EDGE AI PERFORMANCE TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Test image
    # --------------------------------------------------

    image_path = "tests/test.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print()
        print("ERROR: Could not load test image.")
        print("Expected image:")
        print(image_path)
        return

    height, width = frame.shape[:2]

    print()
    print(f"Image size: {width} x {height}")

    # --------------------------------------------------
    # Create pipeline
    # --------------------------------------------------

    pipeline = EdgeAIPipeline()

    # --------------------------------------------------
    # Performance settings
    # --------------------------------------------------

    number_of_frames = 15

    print()
    print(f"Testing {number_of_frames} frames...")
    print(
        f"MiDaS update interval: "
        f"{pipeline.depth_update_interval} frames"
    )

    print()

    # --------------------------------------------------
    # Timing variables
    # --------------------------------------------------

    total_times = []

    # We cannot directly measure internal MiDaS time
    # from the current pipeline, so this test counts
    # how many times the depth map changes/recomputes
    # through the pipeline state.
    #
    # The expected MiDaS execution count is calculated
    # from the configured interval.
    # --------------------------------------------------

    expected_depth_runs = 0

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print("Running warm-up frame...")

    pipeline.process_frame(frame)

    print("Warm-up complete.")

    print()

    # Reset before actual measurement
    pipeline.reset()

    # --------------------------------------------------
    # Performance test
    # --------------------------------------------------

    for i in range(number_of_frames):

        frame_number = i + 1

        start_time = time.perf_counter()

        result = pipeline.process_frame(frame)

        end_time = time.perf_counter()

        elapsed = end_time - start_time

        total_times.append(elapsed)

        # MiDaS runs on:
        # Frame 1, 4, 7, 10, 13...
        if (
            frame_number - 1
        ) % pipeline.depth_update_interval == 0:

            expected_depth_runs += 1

        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        print(
            f"Frame {frame_number:02d} | "
            f"Time: {elapsed * 1000:.2f} ms | "
            f"FPS: {fps:.2f}"
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    total_time = sum(total_times)

    average_time = (
        total_time / len(total_times)
        if total_times
        else 0.0
    )

    min_time = (
        min(total_times)
        if total_times
        else 0.0
    )

    max_time = (
        max(total_times)
        if total_times
        else 0.0
    )

    average_fps = (
        1.0 / average_time
        if average_time > 0
        else 0.0
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)

    print()
    print(
        f"Frames processed:        "
        f"{number_of_frames}"
    )

    print(
        f"Average frame time:      "
        f"{average_time * 1000:.2f} ms"
    )

    print(
        f"Minimum frame time:      "
        f"{min_time * 1000:.2f} ms"
    )

    print(
        f"Maximum frame time:      "
        f"{max_time * 1000:.2f} ms"
    )

    print(
        f"Average FPS:             "
        f"{average_fps:.2f}"
    )

    print(
        f"MiDaS update interval:   "
        f"{pipeline.depth_update_interval} frames"
    )

    print(
        f"Expected MiDaS runs:     "
        f"{expected_depth_runs}"
    )

    print()

    # --------------------------------------------------
    # Detection result from final frame
    # --------------------------------------------------

    print("=" * 80)
    print("FINAL FRAME SUMMARY")
    print("=" * 80)

    print()

    print(
        "YOLO detections:",
        len(result["detections"])
    )

    print(
        "Confirmed objects:",
        len(result["confirmed"])
    )

    print(
        "Processed objects:",
        len(result["objects"])
    )

    print(
        "Path blocked:",
        result["path"]["path_blocked"]
    )

    print(
        "Beep required:",
        result["safety_decision"]["beep_required"]
    )

    print()

    print("=" * 80)
    print("PERFORMANCE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()