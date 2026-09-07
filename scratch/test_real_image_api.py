import urllib.request
import json
import time

data = open(r'D:\SmartVisionAI_New\backend\test.jpg', 'rb').read()
b = '----WebKitFormBoundaryRealTest'
body = (
    f'--{b}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + data + f'\r\n--{b}--\r\n'.encode('utf-8')

print("=" * 65)
print("CONSECUTIVE REAL-IMAGE STREAM TO LIVE /detect API")
print("=" * 65)

for frame_idx in range(1, 6):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        'http://127.0.0.1:8000/detect',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={b}'}
    )

    with urllib.request.urlopen(req, timeout=10) as res:
        elapsed = (time.perf_counter() - t0) * 1000.0
        resp = json.loads(res.read().decode('utf-8'))
        p = resp.get("primary_obstacle", {})
        objs = resp.get("objects", [])
        unk_objs = resp.get("unknown_objects", [])
        print(f"Frame {frame_idx} ({elapsed:5.1f}ms): Primary={p.get('class_name')} | dist={p.get('distance_m')}m | cat={p.get('distance_category')} | stab={resp.get('obstacle_stability')} | approaching={resp.get('approaching')} | objects={len(objs)} | unknowns={len(unk_objs)}")

print("=" * 65)
