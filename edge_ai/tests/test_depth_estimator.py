import cv2

from src.depth_estimator import DepthEstimator


print("=" * 70)
print("DEPTH ESTIMATOR TEST")
print("=" * 70)

image_path = "tests/test.jpg"

frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

print(
    "Image shape:",
    frame.shape
)

depth_estimator = DepthEstimator(
    model_path="models/midas_small.onnx"
)

depth_map = depth_estimator.estimate(
    frame
)

print()
print("Depth map shape:", depth_map.shape)
print("Depth map dtype:", depth_map.dtype)
print("Minimum depth:", depth_map.min())
print("Maximum depth:", depth_map.max())
print("Mean depth:", depth_map.mean())

print()
print("=" * 70)
print("DEPTH ESTIMATOR TEST COMPLETE")
print("=" * 70)