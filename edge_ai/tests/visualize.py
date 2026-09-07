import cv2
import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.detector import YOLOv8TFLiteDetector


# Load image
image_path = "tests/test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image:", image_path)
    sys.exit(1)


# Create detector
detector = YOLOv8TFLiteDetector(
    confidence_threshold=0.50,
    iou_threshold=0.45
)


# Run detection
detections = detector.detect(image)


# Draw detections
for detection in detections:

    class_name = detection["class_name"]
    confidence = detection["confidence"]

    x1, y1, x2, y2 = detection["box"]

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Draw box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label
    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# Save result
output_path = "tests/result.jpg"

cv2.imwrite(
    output_path,
    image
)

print()
print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("Saved:", output_path)
print("Detections:", len(detections))

for detection in detections:
    print(
        f"{detection['class_name']:20s} "
        f"{detection['confidence']:.3f} "
        f"{detection['box']}"
    )

print("=" * 60)