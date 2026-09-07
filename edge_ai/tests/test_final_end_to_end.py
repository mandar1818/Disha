import cv2
import time

from src.edge_ai import EdgeAI


def print_result(label, result):

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    print()

    print("Objects:")

    if not result["objects"]:

        print("  None")

    else:

        for obj in result["objects"]:

            print(
                f"  {obj['name']:10} "
                f"confidence={obj['confidence']:.3f} "
                f"priority={obj['priority']:7} "
                f"proximity={obj['proximity']:10} "
                f"direction={obj['direction']:6} "
                f"in_path={obj['in_path']}"
            )

    print()

    print("Path blocked:", result["path_blocked"])
    print("Multiple objects:", result["multiple_objects"])
    print("Nearby objects:", result["nearby_objects"])
    print("Risk level:", result["risk_level"])
    print("Situation:", result["situation"])
    print("Voice message:", result["voice_message"])
    print("Beep:", result["beep"])


def main():

    print("=" * 80)
    print("FINAL EDGE AI END-TO-END TEST")
    print("=" * 80)

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------

    image_path = "tests/test.jpg"

    frame = cv2.imread(image_path)

    if frame is None:

        print()
        print("ERROR: Could not load test image.")
        print("Expected:", image_path)

        return

    height, width = frame.shape[:2]

    print()
    print(f"Image size: {width} x {height}")

    # --------------------------------------------------
    # CREATE EDGE AI
    # --------------------------------------------------

    edge_ai = EdgeAI()

    print()
    print("Edge AI initialized.")

    # ==================================================
    # TEST 1
    # OBJECT CONFIRMATION
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 1: THREE-FRAME OBJECT CONFIRMATION")
    print("=" * 80)

    results = []

    for i in range(1, 4):

        result = edge_ai.process(frame)

        results.append(result)

        print()
        print(f"Frame {i}")

        print(
            "Confirmed objects:",
            len(result["objects"])
        )

        print(
            "Voice message:",
            result["voice_message"]
        )

        print(
            "Beep:",
            result["beep"]
        )

    # --------------------------------------------------
    # CHECK CONFIRMATION
    # --------------------------------------------------

    if len(results[0]["objects"]) == 0:
        print()
        print("PASS: Frame 1 has no confirmed objects.")

    else:
        print()
        print(
            "WARNING: Frame 1 already has confirmed objects."
        )

    if len(results[1]["objects"]) == 0:
        print("PASS: Frame 2 has no confirmed objects.")

    else:
        print(
            "WARNING: Frame 2 already has confirmed objects."
        )

    if len(results[2]["objects"]) > 0:
        print(
            "PASS: Objects confirmed on Frame 3."
        )

    else:
        print(
            "FAIL: Objects were not confirmed on Frame 3."
        )

    # ==================================================
    # TEST 2
    # VOICE ALERT
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 2: VOICE ALERT")
    print("=" * 80)

    frame_3 = results[2]

    if frame_3["voice_message"] is not None:

        print(
            "PASS: Voice alert generated:",
            frame_3["voice_message"]
        )

    else:

        print(
            "FAIL: No voice alert generated."
        )

    # ==================================================
    # TEST 3
    # BEEP
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 3: PROXIMITY BEEP")
    print("=" * 80)

    if frame_3["beep"]:

        print(
            "PASS: Beep required while path is blocked."
        )

    else:

        print(
            "FAIL: Beep was not required."
        )

    # ==================================================
    # TEST 4
    # ALERT COOLDOWN
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 4: ALERT COOLDOWN")
    print("=" * 80)

    result_4 = edge_ai.process(frame)

    print(
        "Voice message on next frame:",
        result_4["voice_message"]
    )

    if result_4["voice_message"] is None:

        print(
            "PASS: Repeated voice alert suppressed."
        )

    else:

        print(
            "FAIL: Voice alert repeated too soon."
        )

    # ==================================================
    # TEST 5
    # CONTINUOUS BEEP
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 5: CONTINUOUS BEEP")
    print("=" * 80)

    result_5 = edge_ai.process(frame)

    if result_5["beep"]:

        print(
            "PASS: Beep remains active while "
            "path is blocked."
        )

    else:

        print(
            "FAIL: Beep stopped unexpectedly."
        )

    # ==================================================
    # TEST 6
    # RESET
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 6: RESET")
    print("=" * 80)

    edge_ai.reset()

    print(
        "PASS: Edge AI reset completed."
    )

    # ==================================================
    # TEST 7
    # NEW SESSION
    # ==================================================

    print()
    print("=" * 80)
    print("TEST 7: NEW SESSION AFTER RESET")
    print("=" * 80)

    new_results = []

    for i in range(1, 4):

        result = edge_ai.process(frame)

        new_results.append(result)

        print()
        print(f"New session frame {i}")

        print(
            "Objects:",
            len(result["objects"])
        )

        print(
            "Voice:",
            result["voice_message"]
        )

        print(
            "Beep:",
            result["beep"]
        )

    if len(new_results[2]["objects"]) > 0:

        print()
        print(
            "PASS: New detection session works after reset."
        )

    else:

        print()
        print(
            "FAIL: Detection did not recover after reset."
        )

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print()
    print("=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)

    print()
    print("Edge AI public interface:")
    print("    EdgeAI()")
    print("    edge_ai.process(frame)")
    print("    edge_ai.reset()")

    print()
    print("END-TO-END TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()