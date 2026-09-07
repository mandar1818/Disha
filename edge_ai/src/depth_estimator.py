import cv2
import numpy as np
import onnxruntime as ort


class DepthEstimator:
    """
    Lightweight monocular depth estimation using
    MiDaS v2.1 Small ONNX.

    Output is relative depth, not meters.
    """

    def __init__(
        self,
        model_path="models/midas_small.onnx"
    ):

        self.model_path = model_path

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_width = 256
        self.input_height = 256

        # MiDaS ImageNet normalization
        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        print("MiDaS depth model loaded successfully")
        print(
            "Input:",
            self.input_name
        )
        print(
            "Input shape:",
            self.session.get_inputs()[0].shape
        )
        print(
            "Output:",
            self.output_name
        )
        print(
            "Output shape:",
            self.session.get_outputs()[0].shape
        )

    def preprocess(self, frame):

        # OpenCV uses BGR.
        # Convert to RGB for MiDaS.
        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize to model input size.
        image = cv2.resize(
            image,
            (
                self.input_width,
                self.input_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to float.
        image = image.astype(
            np.float32
        ) / 255.0

        # ImageNet normalization.
        image = (
            image - self.mean
        ) / self.std

        # HWC -> CHW.
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Add batch dimension.
        image = np.expand_dims(
            image,
            axis=0
        )

        return image.astype(
            np.float32
        )

    def estimate(self, frame):

        input_tensor = self.preprocess(
            frame
        )

        depth = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            }
        )[0]

        # [1, 256, 256] -> [256, 256]
        depth = np.squeeze(
            depth
        )

        # Normalize relative depth to 0-1.
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max - depth_min > 1e-6:

            depth_normalized = (
                depth - depth_min
            ) / (
                depth_max - depth_min
            )

        else:

            depth_normalized = np.zeros_like(
                depth,
                dtype=np.float32
            )

        return depth_normalized.astype(
            np.float32
        )

    def estimate_at_point(
        self,
        depth_map,
        x,
        y,
        frame_width,
        frame_height
    ):

        depth_height, depth_width = depth_map.shape

        depth_x = int(
            x / frame_width * depth_width
        )

        depth_y = int(
            y / frame_height * depth_height
        )

        depth_x = np.clip(
            depth_x,
            0,
            depth_width - 1
        )

        depth_y = np.clip(
            depth_y,
            0,
            depth_height - 1
        )

        return float(
            depth_map[
                depth_y,
                depth_x
            ]
        )

    def estimate_bbox_depth(
        self,
        depth_map,
        bbox,
        frame_width,
        frame_height
    ):
        """
        Estimate relative depth for an object.

        Uses the median depth from the inner 60%
        of the bounding box instead of one pixel.

        Higher value generally indicates closer
        relative depth for this normalized map.
        """

        x1, y1, x2, y2 = bbox

        depth_height, depth_width = depth_map.shape

        # Convert original image coordinates
        # to depth-map coordinates.
        dx1 = int(
            x1 / frame_width * depth_width
        )

        dy1 = int(
            y1 / frame_height * depth_height
        )

        dx2 = int(
            x2 / frame_width * depth_width
        )

        dy2 = int(
            y2 / frame_height * depth_height
        )

        # Clip coordinates.
        dx1 = int(
            np.clip(
                dx1,
                0,
                depth_width - 1
            )
        )

        dy1 = int(
            np.clip(
                dy1,
                0,
                depth_height - 1
            )
        )

        dx2 = int(
            np.clip(
                dx2,
                0,
                depth_width
            )
        )

        dy2 = int(
            np.clip(
                dy2,
                0,
                depth_height
            )
        )

        # Invalid bounding box.
        if dx2 <= dx1 or dy2 <= dy1:
            return 0.0

        # Bounding-box dimensions.
        box_width = dx2 - dx1
        box_height = dy2 - dy1

        # Ignore outer 20% of the box.
        inner_x1 = dx1 + int(
            box_width * 0.20
        )

        inner_x2 = dx2 - int(
            box_width * 0.20
        )

        inner_y1 = dy1 + int(
            box_height * 0.20
        )

        inner_y2 = dy2 - int(
            box_height * 0.20
        )

        # If the inner region is invalid,
        # use the complete bounding box.
        if (
            inner_x2 <= inner_x1
            or inner_y2 <= inner_y1
        ):

            region = depth_map[
                dy1:dy2,
                dx1:dx2
            ]

        else:

            region = depth_map[
                inner_y1:inner_y2,
                inner_x1:inner_x2
            ]

        if region.size == 0:
            return 0.0

        # Median is more robust than a single pixel.
        return float(
            np.median(region)
        )