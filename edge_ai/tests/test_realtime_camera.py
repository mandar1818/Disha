import cv2

from src.edge_ai_pipeline import EdgeAIPipeline


print("=" * 70)
print("REAL-TIME EDGE AI CAMERA TEST")
print("=" * 70)

pipeline = EdgeAIPipeline()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print()
print("Camera started.")
print("Press Q to quit.")
print()

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    result = pipeline.process_frame(frame)

    # --------------------------------------------------
    # Display detections
    # --------------------------------------------------

    for detection in result["detections"]:

        x1, y1, x2, y2 = map(
            int,
            detection["box"]
        )

        label = detection["class_name"]
        confidence = detection["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------
    # Display path status
    # --------------------------------------------------

    path = result["path"]

    if path["path_blocked"]:
        path_text = "PATH BLOCKED"
    else:
        path_text = "PATH CLEAR"

    cv2.putText(
        frame,
        path_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255) if path["path_blocked"] else (0, 255, 0),
        2
    )

    # --------------------------------------------------
    # Display selected voice alert
    # --------------------------------------------------

    selected = result["selected_alert"]

    if selected:

        alert_text = (
            f"{selected['class_name']} "
            f"{selected['proximity']}"
        )

        cv2.putText(
            frame,
            alert_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # --------------------------------------------------
    # Display beep state
    # --------------------------------------------------

    beep_action = result["beep_action"]

    cv2.putText(
        frame,
        f"BEEP: {beep_action}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------
    # Show frame
    # --------------------------------------------------

    cv2.imshow(
        "Edge AI - Real Time",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

pipeline.reset()

print()
print("Camera test stopped.")  