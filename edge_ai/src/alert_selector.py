class AlertSelector:
    """
    Selects the most important object for voice announcement.

    Priority:
        danger > warning > normal

    Within the same priority:
        very_close > near > far

    Then:
        higher confidence wins
    """

    PRIORITY_RANK = {
        "danger": 3,
        "warning": 2,
        "normal": 1
    }

    PROXIMITY_RANK = {
        "very_close": 3,
        "near": 2,
        "far": 1
    }

    def select(self, objects):
        """
        Select one object for voice announcement.

        Args:
            objects: list of confirmed/proximity-enriched detections

        Returns:
            Selected object or None
        """

        if not objects:
            return None

        selected = max(
            objects,
            key=lambda obj: (
                self.PRIORITY_RANK.get(obj.get("priority", "normal"), 1),
                self.PROXIMITY_RANK.get(obj.get("proximity", "far"), 1),
                obj.get("confidence", 0.0)
            )
        )

        return selected