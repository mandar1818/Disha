import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/yolov8n.tflite"
IMAGE_PATH = "tests/test.jpg"


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


print("=" * 70)
print("YOLOv8 TFLITE RAW OUTPUT INSPECTION")
print("=" * 70)


# --------------------------------------------------
# Load model
# --------------------------------------------------

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\nINPUT")
print(input_details[0])


print("\nOUTPUT")
print(output_details[0])


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE_PATH}"
    )


# --------------------------------------------------
# Preprocess
# --------------------------------------------------

image_resized = cv2.resize(
    image,
    (640, 640)
)

image_resized = cv2.cvtColor(
    image_resized,
    cv2.COLOR_BGR2RGB
)

image_resized = (
    image_resized.astype(np.float32) / 255.0
)

image_resized = np.transpose(
    image_resized,
    (2, 0, 1)
)

input_tensor = np.expand_dims(
    image_resized,
    axis=0
)


# --------------------------------------------------
# Inference
# --------------------------------------------------

interpreter.set_tensor(
    input_details[0]["index"],
    input_tensor
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)


print("\nRAW OUTPUT SHAPE:")
print(output.shape)


# --------------------------------------------------
# Convert [1,84,8400] → [8400,84]
# --------------------------------------------------

predictions = np.squeeze(
    output,
    axis=0
).T


# --------------------------------------------------
# Find high-confidence predictions
# --------------------------------------------------

results = []

for i, prediction in enumerate(predictions):

    box = prediction[:4]

    class_scores = prediction[4:]

    class_id = int(
        np.argmax(class_scores)
    )

    confidence = float(
        class_scores[class_id]
    )

    if confidence >= 0.50:

        results.append({
            "index": i,
            "box": box,
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id],
            "confidence": confidence
        })


results.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


# --------------------------------------------------
# Print high-confidence raw predictions
# --------------------------------------------------

print("\nHIGH-CONFIDENCE RAW PREDICTIONS")
print("=" * 70)


for result in results[:30]:

    print(
        f"\nIndex       : {result['index']}"
    )

    print(
        f"Class       : {result['class_name']}"
    )

    print(
        f"Confidence  : {result['confidence']:.4f}"
    )

    print(
        f"RAW BOX     : {result['box']}"
    )


# --------------------------------------------------
# Statistics
# --------------------------------------------------

if results:

    boxes = np.array([
        r["box"]
        for r in results
    ])

    print("\n")
    print("=" * 70)
    print("HIGH-CONFIDENCE BOX STATISTICS")
    print("=" * 70)

    print(
        "Minimum:",
        boxes.min(axis=0)
    )

    print(
        "Maximum:",
        boxes.max(axis=0)
    )

else:

    print(
        "\nNo predictions above confidence 0.50."
    )


print("\n")
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)