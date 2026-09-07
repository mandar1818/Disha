import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/yolov8n.tflite"
IMAGE_PATH = "tests/test.jpg"

CONFIDENCE_THRESHOLD = 0.50

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


print("=" * 60)
print("YOLOv8n TFLite MODEL TEST")
print("=" * 60)

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print("Shape:", input_details[0]["shape"])
print("Type:", input_details[0]["dtype"])

print("\nOutput:")
print("Shape:", output_details[0]["shape"])
print("Type:", output_details[0]["dtype"])

# Read image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not find image: {IMAGE_PATH}"
    )

original_h, original_w = image.shape[:2]

print("\nImage:")
print("Width:", original_w)
print("Height:", original_h)

# Resize
image_resized = cv2.resize(image, (640, 640))

# BGR -> RGB
image_resized = cv2.cvtColor(
    image_resized,
    cv2.COLOR_BGR2RGB
)

# Normalize
image_resized = image_resized.astype(np.float32) / 255.0

# HWC -> CHW
image_resized = np.transpose(
    image_resized,
    (2, 0, 1)
)

# Add batch
input_tensor = np.expand_dims(
    image_resized,
    axis=0
)

# Run inference
interpreter.set_tensor(
    input_details[0]["index"],
    input_tensor
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nRaw output shape:", output.shape)

# [1, 84, 8400] -> [8400, 84]
predictions = np.squeeze(output, axis=0).T

detections = []

for prediction in predictions:

    class_scores = prediction[4:]

    class_id = int(np.argmax(class_scores))

    confidence = float(class_scores[class_id])

    if confidence >= CONFIDENCE_THRESHOLD:

        detections.append({
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id],
            "confidence": confidence
        })

# Sort highest confidence first
detections.sort(
    key=lambda x: x["confidence"],
    reverse=True
)

print("\nDETECTIONS WITH CONFIDENCE >= 0.50")
print("-" * 60)

if len(detections) == 0:
    print("No objects detected.")
else:
    for detection in detections[:20]:
        print(
            f"{detection['class_name']:20s}"
            f" confidence = "
            f"{detection['confidence']:.3f}"
        )

print("-" * 60)
print("Total candidates:", len(detections))

print("\nMODEL TEST COMPLETED")