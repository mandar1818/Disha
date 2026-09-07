class ProximityEngine:

    def __init__(
        self,
        near_area=0.12,
        very_close_area=0.30,
        center_threshold=0.25,
        near_depth=0.65,
        very_close_depth=0.82
    ):
        """
        Proximity estimation using:
        - Bounding-box area
        - MiDaS relative depth
        - Object position in walking path

        MiDaS provides relative depth, NOT meters.
        """

        self.near_area = near_area
        self.very_close_area = very_close_area

        self.near_depth = near_depth
        self.very_close_depth = very_close_depth

        self.center_threshold = center_threshold

    def calculate_box_area(
        self,
        box,
        image_width,
        image_height
    ):
        x1, y1, x2, y2 = box

        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)

        box_area = box_width * box_height
        image_area = image_width * image_height

        if image_area <= 0:
            return 0.0

        return box_area / image_area

    def is_in_path(
        self,
        box,
        image_width
    ):
        x1, _, x2, _ = box

        object_center_x = (x1 + x2) / 2
        image_center_x = image_width / 2

        normalized_offset = (
            abs(object_center_x - image_center_x)
            / image_width
        )

        return normalized_offset <= self.center_threshold

    def get_depth_state(self, relative_depth):
        if relative_depth >= self.very_close_depth:
            return "very_close"

        if relative_depth >= self.near_depth:
            return "near"

        return "far"

    def fuse_proximity(
        self,
        area_ratio,
        depth_state,
        in_path
    ):
        """
        Combine bbox size and MiDaS depth.

        Important:
        A tiny object should NOT become a dangerous
        obstacle only because MiDaS reports high depth.

        Depth can strengthen an area-based estimate,
        but cannot independently make a tiny object
        very_close.
        """

        # Object outside walking path
        if not in_path:
            return "far"

        # Large bbox = strong proximity signal
        if area_ratio >= self.very_close_area:
            return "very_close"

        if area_ratio >= self.near_area:

            if depth_state == "very_close":
                return "very_close"

            return "near"

        # Small bbox
        if area_ratio < self.near_area:

            # Depth alone is not enough to call it
            # very_close.
            if depth_state == "very_close":
                return "near"

            # Moderate depth + small object
            # remains far.
            return "far"

        return "far"

    def estimate(
        self,
        detection,
        image_width,
        image_height,
        depth_map=None
    ):
        box = detection["box"]

        area_ratio = self.calculate_box_area(
            box,
            image_width,
            image_height
        )

        in_path = self.is_in_path(
            box,
            image_width
        )

        relative_depth = None
        depth_state = None

        if depth_map is not None:

            from .depth_estimator import DepthEstimator

            depth_estimator = DepthEstimator.__new__(
                DepthEstimator
            )

            relative_depth = (
                depth_estimator.estimate_bbox_depth(
                    depth_map,
                    box,
                    image_width,
                    image_height
                )
            )

            depth_state = self.get_depth_state(
                relative_depth
            )

        # Bbox-only fallback
        if depth_state is None:

            if area_ratio >= self.very_close_area:
                proximity = "very_close"

            elif area_ratio >= self.near_area:
                proximity = "near"

            else:
                proximity = "far"

        else:

            proximity = self.fuse_proximity(
                area_ratio,
                depth_state,
                in_path
            )

        result = detection.copy()

        result["proximity"] = proximity
        result["area_ratio"] = area_ratio
        result["in_path"] = in_path

        if relative_depth is not None:
            result["relative_depth"] = relative_depth
            result["depth_proximity"] = depth_state

        return result

    def process(
        self,
        detections,
        image_width,
        image_height,
        depth_map=None
    ):
        results = []

        for detection in detections:

            result = self.estimate(
                detection,
                image_width,
                image_height,
                depth_map
            )

            results.append(result)

        return results