import urllib.request
import json
import os

def test_real_image():
    print("=" * 65)
    print("PHASE 6 REAL IMAGE TEST (backend/test.jpg)")
    print("=" * 65)

    img_path = r'D:\SmartVisionAI_New\backend\test.jpg'
    data = open(img_path, 'rb').read()
    b = '----WebKitFormBoundaryPhase6RealTest'
    body = (
        f'--{b}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + data + f'\r\n--{b}--\r\n'.encode('utf-8')

    req = urllib.request.Request(
        'http://127.0.0.1:8000/detect',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={b}'}
    )

    with urllib.request.urlopen(req, timeout=10) as res:
        assert res.status == 200, f"Expected 200, got {res.status}"
        resp = json.loads(res.read().decode('utf-8'))

    mo = resp.get("multiple_objects", {})
    objs = resp.get("objects", [])
    unk_objs = resp.get("unknown_objects", [])
    primary = resp.get("primary_obstacle", {})

    print(f"HTTP Status           : 200")
    print(f"Success               : {resp.get('success')}")
    print(f"Known YOLO Objects    : {len(objs)} {[o['class_name'] for o in objs]}")
    print(f"Unknown Obstacles     : {len(unk_objs)} {[u['id'] for u in unk_objs]}")
    print(f"Total Objects (Count) : {mo.get('object_count')}")
    print(f"Left Obstacles        : {mo.get('left_obstacles')}")
    print(f"Center Obstacles      : {mo.get('center_obstacles')}")
    print(f"Right Obstacles       : {mo.get('right_obstacles')}")
    print(f"Human Detected        : {mo.get('human_detected')}")
    print(f"Vehicle Detected      : {mo.get('vehicle_detected')}")
    print(f"Closest Object ID     : {mo.get('closest_object_id')}")
    print(f"Highest Risk ID       : {mo.get('highest_risk_object_id')}")
    print(f"Primary Obstacle      : class={primary.get('class_name')}, dist={primary.get('distance_m')}m, cat={primary.get('distance_category')}")
    print(f"Voice Instruction     : {resp.get('voice_instruction')}")
    print("=" * 65)

    # Verification assertions
    assert mo.get("object_count") == len(objs) + len(unk_objs)
    assert mo.get("left_obstacles") + mo.get("center_obstacles") + mo.get("right_obstacles") == mo.get("object_count")
    assert mo.get("human_detected") is True # person in test.jpg
    assert mo.get("closest_object_id") is not None
    assert mo.get("highest_risk_object_id") is not None
    assert primary.get("class_name") is not None
    print("REAL IMAGE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_real_image()
