import cv2
import numpy as np
import tensorflow as tf


class YOLOv8TFLiteDetector:

    def __init__(
        self,
        model_path="models/yolov8n.tflite",
        labels_path="config/coco_classes.txt",
        confidence_threshold=0.50,
        iou_threshold=0.45
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Load COCO class labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.class_names = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]

        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        print("Model loaded successfully")
        print("Input shape:", self.input_shape)
        print("Output shape:", self.output_details[0]["shape"])
        print("Number of classes:", len(self.class_names))

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

    def preprocess(self, image):

        original_h, original_w = image.shape[:2]

        # Scale image while preserving aspect ratio
        scale = min(
            self.input_width / original_w,
            self.input_height / original_h
        )

        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding
        pad_w = self.input_width - new_w
        pad_h = self.input_height - new_h

        left = pad_w // 2
        right = pad_w - left

        top = pad_h // 2
        bottom = pad_h - top

        # Letterbox
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # BGR -> RGB
        padded = cv2.cvtColor(
            padded,
            cv2.COLOR_BGR2RGB
        )

        # Normalize
        padded = padded.astype(np.float32) / 255.0

        # HWC -> CHW
        padded = np.transpose(
            padded,
            (2, 0, 1)
        )

        # Add batch dimension
        input_tensor = np.expand_dims(
            padded,
            axis=0
        )

        return input_tensor, scale, left, top

    # ---------------------------------------------------------
    # IOU
    # ---------------------------------------------------------

    def calculate_iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_w = max(
            0.0,
            x2 - x1
        )

        intersection_h = max(
            0.0,
            y2 - y1
        )

        intersection = (
            intersection_w *
            intersection_h
        )

        area1 = (
            max(0.0, box1[2] - box1[0]) *
            max(0.0, box1[3] - box1[1])
        )

        area2 = (
            max(0.0, box2[2] - box2[0]) *
            max(0.0, box2[3] - box2[1])
        )

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    # ---------------------------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # ---------------------------------------------------------

    def nms(self, detections):

        if not detections:
            return []

        final_detections = []

        class_ids = set(
            d["class_id"]
            for d in detections
        )

        # NMS separately for each class
        for class_id in class_ids:

            class_detections = [
                d
                for d in detections
                if d["class_id"] == class_id
            ]

            # Highest confidence first
            class_detections.sort(
                key=lambda x: x["confidence"],
                reverse=True
            )

            while class_detections:

                best = class_detections.pop(0)

                final_detections.append(best)

                remaining = []

                for detection in class_detections:

                    iou = self.calculate_iou(
                        best["box"],
                        detection["box"]
                    )

                    # Keep boxes that don't overlap too much
                    if iou < self.iou_threshold:
                        remaining.append(detection)

                class_detections = remaining

        return final_detections

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    def detect(self, image):

        original_h, original_w = image.shape[:2]

        # Preprocess
        (
            input_tensor,
            scale,
            pad_x,
            pad_y
        ) = self.preprocess(image)

        # Run model
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_tensor
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Output:
        # [1, 84, 8400]
        #
        # Convert to:
        # [8400, 84]

        predictions = np.squeeze(
            output,
            axis=0
        )

        predictions = predictions.T

        detections = []

        for prediction in predictions:

            # -------------------------------------------------
            # YOLOv8 output
            #
            # 0-3  = cx, cy, width, height
            # 4-83 = 80 COCO class scores
            # -------------------------------------------------

            cx, cy, width, height = prediction[:4]

            class_scores = prediction[4:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

            # Confidence threshold
            if confidence < self.confidence_threshold:
                continue

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Raw box values are normalized 0-1.
            #
            # Convert to 640x640 coordinates first.
            # -------------------------------------------------

            cx *= self.input_width
            cy *= self.input_height
            width *= self.input_width
            height *= self.input_height

            # -------------------------------------------------
            # xywh -> xyxy
            # -------------------------------------------------

            x1 = cx - width / 2
            y1 = cy - height / 2

            x2 = cx + width / 2
            y2 = cy + height / 2

            # -------------------------------------------------
            # Remove letterbox padding
            # -------------------------------------------------

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale

            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # -------------------------------------------------
            # Clip to original image
            # -------------------------------------------------

            x1 = max(
                0.0,
                min(float(original_w), x1)
            )

            y1 = max(
                0.0,
                min(float(original_h), y1)
            )

            x2 = max(
                0.0,
                min(float(original_w), x2)
            )

            y2 = max(
                0.0,
                min(float(original_h), y2)
            )

            # Ignore invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": confidence,
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            })

        # Apply NMS
        detections = self.nms(
            detections
        )

        # Sort by confidence
        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return detections