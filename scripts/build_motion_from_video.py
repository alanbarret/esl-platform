"""
Build Motion Database from UAE Sign Language Videos
====================================================
Pipeline:
1. Download sign video from Vimeo
2. Extract MediaPipe Holistic pose (33 body + 21x2 hands + face)
3. Convert landmarks to bone rotations via IK
4. Save as JSON motion clip for each sign
5. Stitch avatar render frames into MP4

Usage:
  python3 scripts/build_motion_from_video.py --sign "Hello" --output data/motion_db/
  python3 scripts/build_motion_from_video.py --all --limit 10
"""
import argparse, json, math, struct, os, sys, urllib.request, tempfile
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SIGNS_PATH = DATA / "raw" / "uae_signs_raw.json"
MOTION_DB = DATA / "motion_db"
MOTION_DB.mkdir(parents=True, exist_ok=True)

# Load signs
signs_data = json.loads(SIGNS_PATH.read_text())
signs_index = {s["english"].upper(): s for s in signs_data}
signs_index.update({s["english"].upper().replace(" ","_"): s for s in signs_data})

# MediaPipe
mp_holistic = mp.solutions.holistic

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Body landmarks (33 total)
POSE = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# ── Geometry helpers ──────────────────────────────────────────────────────────

def lm_to_vec(lm):
    return np.array([lm.x, lm.y, lm.z])

def vec_to_quat(v1, v2):
    """Quaternion that rotates v1 to v2."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    cross = np.cross(v1, v2)
    dot = float(np.dot(v1, v2))
    w = 1.0 + dot
    if w < 1e-6:
        return [0, 0, 1, 0]
    q = [cross[0], cross[1], cross[2], w]
    n = math.sqrt(sum(x*x for x in q))
    return [x/n for x in q]

def angle_between(a, b, c):
    """Angle at joint b between a-b-c."""
    ba = a - b; bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.acos(max(-1, min(1, float(cos_a))))

def compute_bone_rotation(parent_pos, bone_pos, child_pos, bind_dir):
    """
    Compute bone rotation quaternion from pose landmarks.
    bind_dir: the direction the bone points in rest/bind pose
    """
    if child_pos is not None:
        current_dir = child_pos - bone_pos
    else:
        current_dir = bone_pos - parent_pos
    current_dir = current_dir / (np.linalg.norm(current_dir) + 1e-8)
    return vec_to_quat(np.array(bind_dir), current_dir)

# ── Pose → Bone rotations ─────────────────────────────────────────────────────

def pose_to_bone_rotations(pose_lms, left_hand_lms, right_hand_lms):
    """
    Convert MediaPipe landmarks to GLB bone rotation quaternions.
    Returns dict: boneName -> [x,y,z,w]
    """
    rotations = {}

    if pose_lms is None:
        return rotations

    lms = pose_lms.landmark

    def get(name):
        idx = POSE[name]
        lm = lms[idx]
        return np.array([lm.x, -lm.y, -lm.z])  # flip Y for 3D space

    try:
        ls = get("left_shoulder")
        rs = get("right_shoulder")
        le = get("left_elbow")
        re = get("right_elbow")
        lw = get("left_wrist")
        rw = get("right_wrist")
        lh = get("left_hip")
        rh = get("right_hip")
        nose = get("nose")

        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2

        # Spine: hip → shoulder direction
        spine_dir = shoulder_mid - hip_mid
        rotations["Spine"] = vec_to_quat([0,1,0], spine_dir)
        rotations["Spine1"] = vec_to_quat([0,1,0], spine_dir)
        rotations["Spine2"] = vec_to_quat([0,1,0], spine_dir)

        # Head: shoulder_mid → nose direction
        head_dir = nose - shoulder_mid
        rotations["Head"] = vec_to_quat([0,1,0], head_dir)
        rotations["Neck"] = vec_to_quat([0,1,0], head_dir * 0.5 + np.array([0,1,0]) * 0.5)

        # Right arm: shoulder → elbow
        r_upper = re - rs
        rotations["RightArm"] = vec_to_quat([0,-1,0], r_upper)

        # Right forearm: elbow → wrist
        r_lower = rw - re
        rotations["RightForeArm"] = vec_to_quat([0,-1,0], r_lower)

        # Left arm
        l_upper = le - ls
        rotations["LeftArm"] = vec_to_quat([0,-1,0], l_upper)

        # Left forearm
        l_lower = lw - le
        rotations["LeftForeArm"] = vec_to_quat([0,-1,0], l_lower)

    except Exception as e:
        pass

    # Hand rotations from hand landmarks
    def hand_rotations(hand_lms, side):
        if hand_lms is None:
            return
        h = hand_lms.landmark
        wrist = np.array([h[0].x, -h[0].y, -h[0].z])

        finger_bones = {
            f"{side}HandThumb1":  (1, 2),
            f"{side}HandThumb2":  (2, 3),
            f"{side}HandThumb3":  (3, 4),
            f"{side}HandIndex1":  (5, 6),
            f"{side}HandIndex2":  (6, 7),
            f"{side}HandIndex3":  (7, 8),
            f"{side}HandMiddle1": (9, 10),
            f"{side}HandMiddle2": (10, 11),
            f"{side}HandMiddle3": (11, 12),
            f"{side}HandRing1":   (13, 14),
            f"{side}HandRing2":   (14, 15),
            f"{side}HandRing3":   (15, 16),
            f"{side}HandPinky1":  (17, 18),
            f"{side}HandPinky2":  (18, 19),
            f"{side}HandPinky3":  (19, 20),
        }

        for bone, (p_idx, c_idx) in finger_bones.items():
            parent = np.array([h[p_idx].x, -h[p_idx].y, -h[p_idx].z])
            child  = np.array([h[c_idx].x, -h[c_idx].y, -h[c_idx].z])
            direction = child - parent
            rotations[bone] = vec_to_quat([0, -1, 0], direction)

    hand_rotations(left_hand_lms, "Left")
    hand_rotations(right_hand_lms, "Right")

    return rotations


# ── Download + extract pose ───────────────────────────────────────────────────

def download_video(url, dest):
    """Download video from URL to dest path."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.za.gov.ae/",
    })
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, 'wb') as f:
        f.write(r.read())


def extract_pose_from_video(video_path):
    """
    Extract per-frame bone rotations from video using MediaPipe Holistic.
    Returns list of {time, bones: {boneName: [x,y,z,w]}}
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            t = frame_idx / fps
            bones = pose_to_bone_rotations(
                result.pose_landmarks,
                result.left_hand_landmarks,
                result.right_hand_landmarks,
            )
            frames.append({"time": round(t, 4), "bones": bones})
            frame_idx += 1

    cap.release()
    return frames, fps


def frames_to_gltf_animation(frames, fps, gloss_name):
    """Convert extracted frames to GLTF animation format."""
    if not frames:
        return None

    # Collect all bone names
    all_bones = set()
    for f in frames:
        all_bones.update(f["bones"].keys())

    channels = []
    samplers = []

    for bone in sorted(all_bones):
        times = []
        rotations = []
        for f in frames:
            if bone in f["bones"]:
                times.append(f["time"])
                rotations.extend(f["bones"][bone])

        if len(times) < 2:
            continue

        samplers.append({"input": times, "interpolation": "LINEAR", "output": rotations})
        channels.append({"sampler": len(samplers)-1, "target": {"node": bone, "path": "rotation"}})

    duration = frames[-1]["time"] if frames else 0

    return {
        "name": gloss_name,
        "channels": channels,
        "samplers": samplers,
        "duration": duration,
        "fps": int(fps),
        "frame_count": len(frames),
    }


# ── Render avatar video ───────────────────────────────────────────────────────

def render_avatar_video(frames, fps, gloss_name, output_path, W=480, H=640):
    """Render skeleton animation to video file."""
    BASE = {
        'Head':(0.50,0.09),'Neck':(0.50,0.16),'Spine2':(0.50,0.25),
        'LeftShoulder':(0.36,0.20),'RightShoulder':(0.64,0.20),
        'LeftArm':(0.27,0.29),'RightArm':(0.73,0.29),
        'LeftForeArm':(0.18,0.42),'RightForeArm':(0.82,0.42),
        'LeftHand':(0.11,0.54),'RightHand':(0.89,0.54),
        'Hips':(0.50,0.43),'LeftUpLeg':(0.43,0.52),'RightUpLeg':(0.57,0.52),
        'LeftLeg':(0.42,0.67),'RightLeg':(0.58,0.67),
    }
    CONN=[('Head','Neck'),('Neck','Spine2'),('Spine2','Hips'),
          ('Neck','LeftShoulder'),('Neck','RightShoulder'),
          ('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),
          ('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),
          ('Hips','LeftUpLeg'),('LeftUpLeg','LeftLeg'),
          ('Hips','RightUpLeg'),('RightUpLeg','RightLeg')]

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for fi, frame_data in enumerate(frames):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            img[y,:] = (int(8+y/H*15), int(8+y/H*12), int(18+y/H*25))

        # Compute bone positions from rotations
        positions = {}
        bones = frame_data["bones"]
        for bone,(bxn,byn) in BASE.items():
            bx, by = int(bxn*W), int(byn*H)
            if bone in bones:
                q = bones[bone]
                x,y2,z,w = q
                try:
                    pitch = math.asin(max(-1,min(1,2*(w*y2-z*x))))
                    yaw = math.atan2(2*(w*z+x*y2),1-2*(y2*y2+z*z))
                    bx += int(math.sin(yaw)*55)
                    by += int(-math.sin(pitch)*45)
                except:
                    pass
            positions[bone] = (bx,by)

        for a,b in CONN:
            if a in positions and b in positions:
                cv2.line(img,positions[a],positions[b],(90,55,190),3,cv2.LINE_AA)
        for bone,pos in positions.items():
            col=(168,255,75) if 'Hand' in bone else (210,185,155) if bone=='Head' else (124,58,237)
            r=20 if bone=='Head' else 8 if 'Hand' in bone else 6
            cv2.circle(img,pos,r,col,-1,cv2.LINE_AA)

        ts = cv2.getTextSize(gloss_name,font,1.0,3)[0]
        cv2.putText(img,gloss_name,((W-ts[0])//2+1,H-30),font,1.0,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(img,gloss_name,((W-ts[0])//2,H-31),font,1.0,(168,255,75),3,cv2.LINE_AA)

        t = frame_data["time"]
        dur = frames[-1]["time"] if frames else 1
        bw=int(W*0.8); bx2=int(W*0.1)
        cv2.rectangle(img,(bx2,H-14),(bx2+bw,H-6),(30,30,50),-1)
        cv2.rectangle(img,(bx2,H-14),(bx2+int(bw*t/max(dur,1)),H-6),(124,58,237),-1)
        out.write(img)

    out.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def process_sign(sign_name, output_dir=None):
    """Download, extract pose, save motion JSON and render video for one sign."""
    key = sign_name.upper().replace(" ", "_")
    sign = signs_index.get(key) or signs_index.get(sign_name.upper())
    if not sign:
        print(f"Sign not found: {sign_name}")
        return None

    output_dir = Path(output_dir or MOTION_DB)
    output_dir.mkdir(parents=True, exist_ok=True)

    gloss = sign["english"].upper().replace(" ", "_")
    motion_path = output_dir / f"{gloss}.json"
    video_path = output_dir / f"{gloss}.avi"

    if motion_path.exists():
        print(f"Already processed: {gloss}")
        return json.loads(motion_path.read_text())

    print(f"Processing: {gloss}")
    print(f"  Video URL: {sign['video_url']}")

    # Download
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        download_video(sign["video_url"], tmp_path)
        size = os.path.getsize(tmp_path)
        print(f"  Downloaded: {size//1024}KB")

        # Extract pose
        frames, fps = extract_pose_from_video(tmp_path)
        print(f"  Extracted {len(frames)} frames at {fps:.0f}fps")

        # Convert to GLTF animation
        animation = frames_to_gltf_animation(frames, fps, gloss)
        if animation:
            motion_path.write_text(json.dumps(animation, indent=2))
            print(f"  Saved motion: {motion_path}")

        # Render video
        render_avatar_video(frames, fps, gloss, video_path)
        print(f"  Rendered: {video_path} ({os.path.getsize(video_path)//1024}KB)")

        return animation

    except Exception as e:
        print(f"  Error: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def stitch_videos(gloss_tokens, output_path, W=480, H=640):
    """Stitch multiple sign videos into one continuous video."""
    fps = 30
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))

    for gloss in gloss_tokens:
        vid_path = MOTION_DB / f"{gloss}.avi"
        if not vid_path.exists():
            print(f"Video not found for {gloss}, generating placeholder")
            # Generate 1s placeholder
            for _ in range(fps):
                img = np.zeros((H, W, 3), dtype=np.uint8)
                img[:] = (10, 10, 20)
                font = cv2.FONT_HERSHEY_SIMPLEX
                ts = cv2.getTextSize(gloss,font,0.8,2)[0]
                cv2.putText(img,gloss,((W-ts[0])//2,(H+ts[1])//2),font,0.8,(168,255,75),2)
                out.write(img)
            continue

        cap = cv2.VideoCapture(str(vid_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            out.write(frame)
        cap.release()

        # Add 0.3s transition pause
        for _ in range(int(fps * 0.3)):
            out.write(np.zeros((H, W, 3), dtype=np.uint8))

    out.release()
    print(f"Stitched {len(gloss_tokens)} signs → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", help="Single sign to process")
    parser.add_argument("--signs", nargs="+", help="Multiple signs to process and stitch")
    parser.add_argument("--all", action="store_true", help="Process all signs")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default=str(MOTION_DB))
    parser.add_argument("--stitch", help="Output path for stitched video")
    args = parser.parse_args()

    if args.sign:
        process_sign(args.sign, args.output)

    elif args.signs:
        gloss_tokens = []
        for s in args.signs:
            result = process_sign(s, args.output)
            if result:
                gloss_tokens.append(result["name"])
        if args.stitch and gloss_tokens:
            stitch_videos(gloss_tokens, args.stitch)

    elif args.all:
        to_process = signs_data[:args.limit]
        gloss_tokens = []
        for sign in to_process:
            result = process_sign(sign["english"], args.output)
            if result:
                gloss_tokens.append(result["name"])
        if gloss_tokens:
            out = Path(args.output) / "demo_all.avi"
            stitch_videos(gloss_tokens, out)
            print(f"Final video: {out}")
