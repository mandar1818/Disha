import time
import urllib.request
import json
import numpy as np
import cv2

API_URL = "http://127.0.0.1:8000/detect"

def send_frame(img: np.ndarray) -> dict:
    _, buf = cv2.imencode('.jpg', img)
    img_bytes = buf.tobytes()

    boundary = "----WebKitFormBoundaryStreamTest"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_stream_test():
    print("=" * 60)
    print("LIVE API MULTI-FRAME OBSTACLE DISTANCE SMOOTHING TEST")
    print("=" * 60)

    # Frame with prominent dark obstacle in center foreground
    # This will trigger depth contrast and produce an obstacle
    for frame_idx in range(1, 6):
        img = np.full((480, 640, 3), 180, dtype=np.uint8) # light background
        # Large dark obstacle in bottom center
        cv2.rectangle(img, (220, 180), (420, 420), (30, 30, 30), -1)

        res = send_frame(img)
        primary = res.get("primary_obstacle")
        unk_objs = res.get("unknown_objects", [])
        objs = res.get("objects", [])

        if primary:
            print(f"Frame {frame_idx}: Primary={primary.get('class_name')} | dist={primary.get('distance_m')}m | cat={primary.get('distance_category')} | conf={primary.get('distance_confidence')} | stab={res.get('obstacle_stability')} | approaching={res.get('approaching')}")
        else:
            print(f"Frame {frame_idx}: No primary obstacle (objs={len(objs)}, unk={len(unk_objs)})")

    print("=" * 60)
    print("LIVE STREAM OBSTACLE TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_stream_test()
