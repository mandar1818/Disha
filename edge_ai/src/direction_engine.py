class DirectionEngine:
    """
    Determines whether a detected object is on the
    left, center, or right side of the camera frame.
    """

    def __init__(
        self,
        left_threshold=0.33,
        right_threshold=0.67
    ):
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold

    def get_direction(
        self,
        box,
        image_width
    ):
        """
        Determine object direction from its bounding box.

        Returns:
            "left"
            "center"
            "right"
        """

        x1, _, x2, _ = box

        object_center_x = (x1 + x2) / 2

        normalized_x = object_center_x / image_width

        if normalized_x < self.left_threshold:
            return "left"

        elif normalized_x > self.right_threshold:
            return "right"

        else:
            return "center"

    def process(
        self,
        detections,
        image_width
    ):
        """
        Add direction information to every detection.
        """

        results = []

        for detection in detections:

            result = detection.copy()

            result["direction"] = self.get_direction(
                detection["box"],
                image_width
            )

            results.append(result)

        return results