import cv2

from src.edge_ai_pipeline import EdgeAIPipeline


IMAGE_PATH = "tests/test.jpg"


print("=" * 80)
print("COMPLETE EDGE AI PIPELINE TEST")
print("=" * 80)


# ------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width = frame.shape[:2]

print(f"Image size: {width} x {height}")


# ------------------------------------------------------
# CREATE PIPELINE
# ------------------------------------------------------

pipeline = EdgeAIPipeline()


# ------------------------------------------------------
# PROCESS MULTIPLE FRAMES
# ------------------------------------------------------

for frame_number in range(1, 6):

    print("\n" + "-" * 80)
    print(f"FRAME {frame_number}")
    print("-" * 80)

    result = pipeline.process_frame(frame)

    # --------------------------------------------------
    # YOLO DETECTIONS
    # --------------------------------------------------

    print("\nYOLO detections:")

    for detection in result["detections"]:

        print(
            f"  {detection['class_name']:<15}"
            f" confidence={detection['confidence']:.3f}"
        )

    # --------------------------------------------------
    # CONFIRMED OBJECTS
    # --------------------------------------------------

    print("\nConfirmed objects:")

    if not result["confirmed"]:

        print("  None")

    else:

        for obj in result["confirmed"]:

            print(
                f"  {obj['class_name']:<15}"
                f" confidence={obj['confidence']:.3f}"
                f" priority={obj['priority']:<7}"
                f" frames={obj['confirmation_frames']}"
            )

    # --------------------------------------------------
    # DEPTH + PROXIMITY
    # --------------------------------------------------

    print("\nProximity analysis:")

    if not result["objects"]:

        print("  None")

    else:

        for obj in result["objects"]:

            print(
                f"  {obj['class_name']:<15}"
                f" area={obj['area_ratio']:.3f}"
                f" depth={obj.get('relative_depth', 0.0):.3f}"
                f" depth_state={obj.get('depth_proximity', 'N/A'):<11}"
                f" proximity={obj['proximity']:<11}"
                f" in_path={obj['in_path']}"
            )

    # --------------------------------------------------
    # PATH ANALYSIS
    # --------------------------------------------------

    path = result["path"]

    print("\nPath analysis:")

    print(
        f"  Path blocked:       {path.get('path_blocked')}"
    )

    print(
        f"  Nearby objects:     {path.get('nearby_objects')}"
    )

    print(
        f"  Multiple objects:   {path.get('multiple_objects')}"
    )

    print(
        f"  Beep required:      {path.get('beep_required')}"
    )

    # --------------------------------------------------
    # SELECTED ALERT
    # --------------------------------------------------

    selected = result["selected_alert"]

    print("\nSelected alert:")

    if selected is None:

        print("  None")

    else:

        print(
            f"  {selected['class_name']}"
            f" | priority={selected['priority']}"
            f" | proximity={selected['proximity']}"
        )

    # --------------------------------------------------
    # VOICE ALERT
    # --------------------------------------------------

    print("\nVoice alerts:")

    if not result["alerts"]:

        print("  None")

    else:

        for alert in result["alerts"]:

            print(
                f"  {alert}"
            )

    # --------------------------------------------------
    # SAFETY DECISION
    # --------------------------------------------------

    safety = result["safety_decision"]

    print("\nSafety decision:")

    print(
        f"  Situation:          {safety['situation']}"
    )

    print(
        f"  Risk level:         {safety['risk_level']}"
    )

    print(
        f"  Path blocked:       {safety['path_blocked']}"
    )

    print(
        f"  Multiple objects:   {safety['multiple_objects']}"
    )

    print(
        f"  Nearby objects:     {safety['nearby_objects']}"
    )

    print(
        f"  Voice required:     {safety['voice_required']}"
    )

    print(
        f"  Beep required:      {safety['beep_required']}"
    )

    # --------------------------------------------------
    # BEEP
    # --------------------------------------------------

    print("\nBeep action:")

    print(
        f"  {result['beep_action']}"
    )


print("\n" + "=" * 80)
print("COMPLETE EDGE AI PIPELINE TEST COMPLETE")
print("=" * 80)