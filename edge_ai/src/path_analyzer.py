class PathAnalyzer:

    def __init__(
        self,
        minimum_proximity="near",
        multi_object_threshold=2
    ):
        self.minimum_proximity = minimum_proximity
        self.multi_object_threshold = multi_object_threshold

    # ------------------------------------------------------
    # PROXIMITY SEVERITY
    # ------------------------------------------------------

    def proximity_level(self, proximity):

        levels = {
            "far": 0,
            "near": 1,
            "very_close": 2
        }

        return levels.get(
            proximity,
            0
        )

    # ------------------------------------------------------
    # CHECK WHETHER OBJECT IS A NEARBY OBSTACLE
    # ------------------------------------------------------

    def is_nearby_obstacle(self, detection):

        proximity = detection["proximity"]
        in_path = detection["in_path"]

        return (
            in_path
            and
            self.proximity_level(proximity) >=
            self.proximity_level(self.minimum_proximity)
        )

    # ------------------------------------------------------
    # ANALYZE PATH
    # ------------------------------------------------------

    def analyze(self, detections):

        nearby_objects = []

        for detection in detections:

            if self.is_nearby_obstacle(
                detection
            ):
                nearby_objects.append(
                    detection
                )

        # --------------------------------------------------
        # COUNT NEARBY OBJECTS
        # --------------------------------------------------

        nearby_count = len(
            nearby_objects
        )

        # --------------------------------------------------
        # DETERMINE WHETHER PATH IS BLOCKED
        # --------------------------------------------------

        path_blocked = (
            nearby_count > 0
        )

        # --------------------------------------------------
        # MULTIPLE OBJECTS
        # --------------------------------------------------

        multiple_objects = (
            nearby_count >=
            self.multi_object_threshold
        )

        # --------------------------------------------------
        # BEEP DECISION
        #
        # Beep when:
        #
        # 1. At least one nearby obstacle is in path
        # OR
        # 2. Multiple nearby obstacles are present
        # --------------------------------------------------

        beep_required = (
            path_blocked
            or
            multiple_objects
        )

        # --------------------------------------------------
        # HIGHEST PRIORITY OBJECT
        # --------------------------------------------------

        priority_order = {
            "danger": 0,
            "warning": 1,
            "normal": 2
        }

        highest_priority = None

        if nearby_objects:

            sorted_objects = sorted(
                nearby_objects,
                key=lambda x: (
                    priority_order.get(
                        x["priority"],
                        2
                    ),
                    -x["confidence"]
                )
            )

            highest_priority = sorted_objects[0]

        return {
            "path_blocked": path_blocked,
            "nearby_count": nearby_count,
            "multiple_objects": multiple_objects,
            "beep_required": beep_required,
            "nearby_objects": nearby_objects,
            "highest_priority": highest_priority
        }