import cv2

from src.edge_ai import EdgeAI


def main():

    print("=" * 80)
    print("EDGE AI PUBLIC INTERFACE TEST")
    print("=" * 80)

    # --------------------------------------------------
    # LOAD TEST IMAGE
    # --------------------------------------------------

    image_path = "tests/test.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print("ERROR: Could not load test image.")
        print(f"Expected: {image_path}")
        return

    height, width = frame.shape[:2]

    print()
    print(f"Image size: {width} x {height}")

    # --------------------------------------------------
    # CREATE EDGE AI
    # --------------------------------------------------

    edge_ai = EdgeAI()

    print()
    print("Edge AI initialized successfully.")

    # --------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------

    for frame_number in range(1, 6):

        print()
        print("-" * 80)
        print(f"FRAME {frame_number}")
        print("-" * 80)

        result = edge_ai.process(frame)

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
        print(
            "Multiple objects:",
            result["multiple_objects"]
        )
        print(
            "Nearby objects:",
            result["nearby_objects"]
        )
        print(
            "Risk level:",
            result["risk_level"]
        )
        print(
            "Situation:",
            result["situation"]
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
    # RESET
    # --------------------------------------------------

    print()
    print("-" * 80)
    print("TESTING RESET")
    print("-" * 80)

    edge_ai.reset()

    print("Edge AI reset successfully.")

    print()
    print("=" * 80)
    print("EDGE AI PUBLIC INTERFACE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()