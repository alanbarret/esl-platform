"""
Render a test animation video for HELLO sign.
Shows avatar skeleton + sign overlay.
"""
import sys, json, math, subprocess, tempfile
sys.path.insert(0, '/root/.openclaw/workspace/esl-platform/backend')

import cv2
import numpy as np
import urllib.request

# ── Get animation from API ─────────────────────────────────────────────────
req = urllib.request.Request(
    "http://localhost:8001/api/v1/translate",
    data=json.dumps({"text": "Hello how are you", "output_format": "gltf"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read())

gloss_tokens = data["gloss_tokens"]
anim = data["gltf_animation"]
duration = anim["duration"]
fps = 30
total_frames = int(duration * fps)

print(f"Gloss: {gloss_tokens}")
print(f"Duration: {duration:.1f}s, Frames: {total_frames}")

# ── Build bone animation lookup ──────────────────────────────────────────────
def sample_rotation(times, values, t):
    if not times: return [0,0,0,1]
    if t <= times[0]: return values[:4]
    if t >= times[-1]: return values[-4:]
    for i in range(len(times)-1):
        if times[i] <= t < times[i+1]:
            alpha = (t - times[i]) / (times[i+1] - times[i])
            a = values[i*4:(i+1)*4]
            b = values[(i+1)*4:(i+2)*4]
            return [a[j] + alpha*(b[j]-a[j]) for j in range(4)]
    return [0,0,0,1]

def quat_to_euler(q):
    x,y,z,w = q
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    pitch = math.asin(max(-1, min(1, sinp)))
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw

# Build channel lookup: boneName -> (times, values)
channels = {}
for ch in anim["channels"]:
    s = anim["samplers"][ch["sampler"]]
    bone = ch["target"]["node"]
    if ch["target"]["path"] == "rotation":
        channels[bone] = (s["input"], s["output"])

# ── Skeleton layout (screen coords for 720x1280 portrait) ──────────────────
W, H = 480, 854

# Base bone positions (normalized 0-1)
BASE = {
    "Head":          (0.50, 0.10),
    "Neck":          (0.50, 0.17),
    "Spine2":        (0.50, 0.26),
    "Spine1":        (0.50, 0.32),
    "Spine":         (0.50, 0.38),
    "Hips":          (0.50, 0.44),
    "LeftShoulder":  (0.36, 0.22),
    "RightShoulder": (0.64, 0.22),
    "LeftArm":       (0.28, 0.30),
    "RightArm":      (0.72, 0.30),
    "LeftForeArm":   (0.20, 0.42),
    "RightForeArm":  (0.80, 0.42),
    "LeftHand":      (0.13, 0.53),
    "RightHand":     (0.87, 0.53),
    "LeftUpLeg":     (0.43, 0.52),
    "RightUpLeg":    (0.57, 0.52),
    "LeftLeg":       (0.42, 0.68),
    "RightLeg":      (0.58, 0.68),
    "LeftFoot":      (0.41, 0.82),
    "RightFoot":     (0.59, 0.82),
}

CONNECTIONS = [
    ("Head","Neck"),("Neck","Spine2"),("Spine2","Spine1"),("Spine1","Spine"),("Spine","Hips"),
    ("Neck","LeftShoulder"),("Neck","RightShoulder"),
    ("LeftShoulder","LeftArm"),("LeftArm","LeftForeArm"),("LeftForeArm","LeftHand"),
    ("RightShoulder","RightArm"),("RightArm","RightForeArm"),("RightForeArm","RightHand"),
    ("Hips","LeftUpLeg"),("LeftUpLeg","LeftLeg"),("LeftLeg","LeftFoot"),
    ("Hips","RightUpLeg"),("RightUpLeg","RightLeg"),("RightLeg","RightFoot"),
]

def get_bone_pos(bone, t, scale=1.0):
    base = BASE.get(bone, (0.5,0.5))
    bx, by = int(base[0]*W), int(base[1]*H)

    if bone not in channels:
        return (bx, by)

    times, values = channels[bone]
    q = sample_rotation(times, values, t)
    roll, pitch, yaw = quat_to_euler(q)

    # Apply offset based on parent chain (simplified FK)
    offset_scale = 60 * scale
    dx = int(math.sin(yaw) * offset_scale * 0.5)
    dy = int(-math.sin(pitch) * offset_scale * 0.3)

    return (bx + dx, by + dy)

# ── Render frames ────────────────────────────────────────────────────────────
frames = []
sign_dur = duration / max(len(gloss_tokens), 1)

for frame_idx in range(total_frames):
    t = frame_idx / fps
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (10, 10, 20)  # dark bg

    # Gradient background
    for y in range(H):
        alpha = y / H
        color = (int(10+alpha*15), int(10+alpha*20), int(20+alpha*40))
        img[y, :] = color

    # Get current bone positions with animation
    positions = {bone: get_bone_pos(bone, t) for bone in BASE}

    # Draw connections
    for a, b in CONNECTIONS:
        if a in positions and b in positions:
            cv2.line(img, positions[a], positions[b], (100, 60, 200), 3, cv2.LINE_AA)

    # Draw joints
    for bone, pos in positions.items():
        if "Hand" in bone:
            cv2.circle(img, pos, 7, (168, 255, 75), -1, cv2.LINE_AA)
        elif bone in ["Head"]:
            cv2.circle(img, pos, 18, (200, 180, 160), -1, cv2.LINE_AA)
            cv2.circle(img, pos, 18, (140, 100, 60), 2, cv2.LINE_AA)
        elif "Arm" in bone or "Shoulder" in bone:
            cv2.circle(img, pos, 8, (124, 58, 237), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, pos, 5, (80, 80, 160), -1, cv2.LINE_AA)

    # Current gloss label
    sign_idx = min(int(t / sign_dur), len(gloss_tokens)-1)
    gloss = gloss_tokens[sign_idx]
    progress_in_sign = (t - sign_idx * sign_dur) / sign_dur

    # Gloss text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(gloss, font, 1.2, 3)[0]
    tx = (W - text_size[0]) // 2
    cv2.putText(img, gloss, (tx+2, H-42), font, 1.2, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, gloss, (tx, H-44), font, 1.2, (168,255,75), 3, cv2.LINE_AA)

    # Progress bar
    bar_w = int(W * 0.8)
    bar_x = int(W * 0.1)
    cv2.rectangle(img, (bar_x, H-18), (bar_x+bar_w, H-8), (40,40,60), -1)
    prog = int(bar_w * t / duration)
    cv2.rectangle(img, (bar_x, H-18), (bar_x+prog, H-8), (124,58,237), -1)

    # Timer
    cv2.putText(img, f"{t:.1f}s", (8, 24), font, 0.5, (100,100,150), 1)

    frames.append(img)

# ── Save video ──────────────────────────────────────────────────────────────
out_path = "/root/.openclaw/workspace/esl_test.mp4"
with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
    tmp_path = tmp.name
    for f in frames:
        tmp.write(f.tobytes())

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(fps),
    "-i", tmp_path,
    "-c:v", "libx264", "-crf", "22", "-preset", "fast", "-pix_fmt", "yuv420p",
    out_path
]
result = subprocess.run(cmd, capture_output=True)
import os; os.unlink(tmp_path)

if result.returncode == 0:
    size = os.path.getsize(out_path)
    print(f"Video saved: {out_path} ({size//1024}KB, {total_frames} frames)")
else:
    print("FFmpeg error:", result.stderr.decode()[:200])
