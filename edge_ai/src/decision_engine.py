from collections import defaultdict


class DecisionEngine:

    def __init__(
        self,
        confirmation_frames=3
    ):
        # Object must be detected in this many
        # consecutive frames before being confirmed.
        self.confirmation_frames = confirmation_frames

        # Tracks how many consecutive frames
        # each object class has appeared.
        self.frame_counts = defaultdict(int)

        # Store previously confirmed objects.
        self.confirmed_objects = set()

        # --------------------------------------------------
        # PRIORITY MAP
        # --------------------------------------------------

        self.priority_map = {

            # 🔴 DANGER
            "car": "danger",
            "motorcycle": "danger",
            "bus": "danger",
            "truck": "danger",
            "train": "danger",
            "airplane": "danger",
            "boat": "danger",
            "fire hydrant": "danger",
            "stop sign": "danger",
            "knife": "danger",
            "scissors": "danger",

            # 🟠 WARNING
            "person": "warning",
            "bicycle": "warning",
            "traffic light": "warning",
            "parking meter": "warning",
            "bench": "warning",
            "backpack": "warning",
            "umbrella": "warning",
            "handbag": "warning",
            "suitcase": "warning",
            "skis": "warning",
            "snowboard": "warning",
            "skateboard": "warning",
            "surfboard": "warning",
            "sports ball": "warning",
            "baseball bat": "warning",
            "baseball glove": "warning",
            "chair": "warning",
            "couch": "warning",
            "bed": "warning",
            "dining table": "warning",
            "toilet": "warning",
            "potted plant": "warning",
            "microwave": "warning",
            "oven": "warning",
            "toaster": "warning",
            "sink": "warning",
            "refrigerator": "warning",

            # 🟢 NORMAL
            "bird": "normal",
            "cat": "normal",
            "dog": "normal",
            "horse": "normal",
            "sheep": "normal",
            "cow": "normal",
            "elephant": "normal",
            "bear": "normal",
            "zebra": "normal",
            "giraffe": "normal",
            "frisbee": "normal",
            "kite": "normal",
            "tennis racket": "normal",
            "bottle": "normal",
            "wine glass": "normal",
            "cup": "normal",
            "fork": "normal",
            "spoon": "normal",
            "bowl": "normal",
            "banana": "normal",
            "apple": "normal",
            "sandwich": "normal",
            "orange": "normal",
            "broccoli": "normal",
            "carrot": "normal",
            "hot dog": "normal",
            "pizza": "normal",
            "donut": "normal",
            "cake": "normal",
            "tv": "normal",
            "laptop": "normal",
            "mouse": "normal",
            "remote": "normal",
            "keyboard": "normal",
            "cell phone": "normal",
            "book": "normal",
            "clock": "normal",
            "vase": "normal",
            "teddy bear": "normal",
            "hair drier": "normal",
            "toothbrush": "normal",
            "tie": "normal",
        }

    # ------------------------------------------------------
    # GET PRIORITY
    # ------------------------------------------------------

    def get_priority(self, class_name):

        return self.priority_map.get(
            class_name,
            "normal"
        )

    # ------------------------------------------------------
    # PROCESS ONE FRAME
    # ------------------------------------------------------

    def process_frame(self, detections):

        # Classes detected in current frame
        current_classes = set(
            detection["class_name"]
            for detection in detections
        )

        # --------------------------------------------------
        # Update consecutive frame counters
        # --------------------------------------------------

        for class_name in list(self.frame_counts.keys()):

            if class_name not in current_classes:

                # Object disappeared.
                # Reset its counter.
                self.frame_counts[class_name] = 0

        for class_name in current_classes:

            self.frame_counts[class_name] += 1

        # --------------------------------------------------
        # Find confirmed objects
        # --------------------------------------------------

        confirmed = []

        for detection in detections:

            class_name = detection["class_name"]

            count = self.frame_counts[class_name]

            if count >= self.confirmation_frames:

                priority = self.get_priority(
                    class_name
                )

                confirmed.append({
                    "class_id": detection["class_id"],
                    "class_name": class_name,
                    "confidence": detection["confidence"],
                    "box": detection["box"],
                    "priority": priority,
                    "confirmation_frames": count
                })

        # --------------------------------------------------
        # Sort by priority
        #
        # danger > warning > normal
        # --------------------------------------------------

        priority_order = {
            "danger": 0,
            "warning": 1,
            "normal": 2
        }

        confirmed.sort(
            key=lambda x: (
                priority_order[x["priority"]],
                -x["confidence"]
            )
        )

        return confirmed

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------

    def reset(self):

        self.frame_counts.clear()
        self.confirmed_objects.clear()