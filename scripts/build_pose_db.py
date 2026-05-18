"""
Build Pose Database from Real UAE Sign Language Videos
Downloads and extracts MediaPipe pose for key signs,
then generates the SIGN_POSES dict for demo_server.py
"""
import urllib.request, cv2, numpy as np, json, math, struct
from pathlib import Path
import mediapipe as mp

ROOT = Path('/root/.openclaw/workspace/esl-platform')
MOTION_DB = ROOT / 'data' / 'motion_db'
MOTION_DB.mkdir(exist_ok=True)
mp_holistic = mp.solutions.holistic

# Load GLB rest quaternions
with open(ROOT / 'frontend/public/avatar/arab-man.glb', 'rb') as f:
    f.read(12); cl=struct.unpack('<I',f.read(4))[0]; f.read(4)
    gltf = json.loads(f.read(cl))
REST = {n['name']: n.get('rotation',[0,0,0,1]) for n in gltf['nodes'] if 'name' in n}

def norm(v): n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v] if n>1e-6 else v
def mul_q(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz]
def inv_q(q): x,y,z,w=q; n2=sum(c*c for c in q)+1e-12; return [-x/n2,-y/n2,-z/n2,w/n2]
def vec_q(v1,v2):
    v1=np.array(v1); v2=np.array(v2)
    n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return [0,0,0,1]
    v1/=n1; v2/=n2
    c=np.cross(v1,v2); d=float(np.dot(v1,v2)); w=1+d
    if w<1e-6: return [0,0,1,0]
    return norm([float(c[0]),float(c[1]),float(c[2]),w])

def extract_peak_pose(video_path, VW=640, VH=480):
    """
    Extract the 'peak' frame from a sign video.
    The peak is where the signing arm is highest/most extended.
    Returns bone rotation dict for that frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    VW_r = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH_r = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_frames = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as ho:
        while cap.isOpened():
            ret, fr = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            res = ho.process(rgb)
            if res.pose_landmarks:
                all_frames.append(res.pose_landmarks.landmark)
    cap.release()
    
    if not all_frames:
        return None
    
    # Find peak: frame where right wrist is highest (lowest y value)
    best_frame = None
    best_score = float('inf')
    
    # Use middle 60% of frames (avoid start/end padding)
    start = len(all_frames) // 5
    end = len(all_frames) * 4 // 5
    
    for i, lm in enumerate(all_frames[start:end], start):
        # Score = right wrist y (lower = higher on screen = higher arm)
        rw_y = lm[16].y  # right wrist
        re_y = lm[14].y  # right elbow
        score = (rw_y + re_y) / 2
        if score < best_score:
            best_score = score
            best_frame = lm
    
    if best_frame is None:
        best_frame = all_frames[len(all_frames)//2]
    
    # Convert to bone rotations
    lm = best_frame
    def p(i): return np.array([lm[i].x*VW_r, (1-lm[i].y)*VH_r, -lm[i].z*VW_r])
    
    bones = {}
    try:
        ls=p(11); rs=p(12); le=p(13); re=p(14)
        lw=p(15); rw=p(16); lh=p(23); rh=p(24); nose=p(0)
        
        def dq(d, bone):
            n=np.linalg.norm(d)
            if n<1e-6: return [0,0,0,1]
            wq = vec_q([0,-1,0], (d/n).tolist())
            rq = REST.get(bone, [0,0,0,1])
            result = mul_q(inv_q(rq), wq)
            # Clamp to reasonable range (euler -1.5 to 1.5 rad)
            mag = math.sqrt(sum(x*x for x in result[:3]))
            if mag > 0.95:  # quaternion with >~70 degree rotation
                scale = 0.85/mag
                result = norm([result[0]*scale, result[1]*scale, result[2]*scale, result[3]])
            return result
        
        bones['RightArm']     = dq(re-rs, 'RightArm')
        bones['RightForeArm'] = dq(rw-re, 'RightForeArm')
        bones['LeftArm']      = dq(le-ls, 'LeftArm')
        bones['LeftForeArm']  = dq(lw-le, 'LeftForeArm')
        bones['Head']         = dq(nose-(ls+rs)/2, 'Head')
        bones['Spine2']       = dq((ls+rs)/2-(lh+rh)/2, 'Spine2')
    except Exception as e:
        print(f'  Bone extraction error: {e}')
    
    return bones

def to_euler_repr(bones):
    """Convert bone rotation quaternions to compact euler-like representation for SIGN_POSES."""
    result = {}
    for bone, q in bones.items():
        x,y,z,w = q
        try:
            roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
            pitch = math.asin(max(-1,min(1,2*(w*y-z*x))))
            yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            # Only include if significant movement (> 0.05 rad)
            if max(abs(roll),abs(pitch),abs(yaw)) > 0.05:
                result[bone] = (round(roll,3), round(pitch,3), round(yaw,3))
        except:
            pass
    return result

# ── Target signs to process ───────────────────────────────────────────────────
data = json.loads((ROOT/'data/raw/uae_signs_full.json').read_text())
signs_idx = {s['english'].upper().replace(' ','_'): s for s in data}
signs_idx.update({s['english'].upper(): s for s in data})

TARGET_SIGNS = {
    'HELLO':      'HOW_ARE_YOU',   # use HOW_ARE_YOU as proxy
    'HOW':        'HOW_ARE_YOU',
    'YOU':        'HOW_ARE_YOU',
    'THANK_YOU':  None,  # not in dataset, use heuristic
    'YES':        None,
    'NO':         None,
    'GOOD':       None,
    'PLEASE':     None,
    'HELP':       'helps',
    'DOCTOR':     'Doctor',
    'WORK':       'Work',
    'FAMILY':     'Family',
    'SCHOOL':     'School',
    'MORNING':    'Morning',
    'WELCOME':    None,
    'SORRY':      None,
    'WHERE':      None,
    'WHAT':       None,
    'NAME':       None,
    'HOME':       'Home Lawn',
    'WATER':      'watering',
}

print('Extracting real peak poses from sign videos...')
real_poses = {}

for gloss, video_name in TARGET_SIGNS.items():
    if video_name is None:
        continue
    
    # Check if already downloaded
    mp4 = MOTION_DB / f'{video_name.upper().replace(" ","_")}.mp4'
    if not mp4.exists():
        mp4 = MOTION_DB / f'{video_name}.mp4'
    
    if not mp4.exists():
        # Try to download
        sign_key = video_name.upper().replace(' ','_')
        sign_data = signs_idx.get(sign_key) or signs_idx.get(video_name.upper())
        if not sign_data or not sign_data.get('video_url'):
            print(f'  {gloss}: no video found')
            continue
        
        print(f'  Downloading {gloss} ({video_name})...')
        try:
            req = urllib.request.Request(sign_data['video_url'], headers={
                'User-Agent':'Mozilla/5.0','Referer':'https://www.za.gov.ae/'
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                mp4.write_bytes(r.read())
            print(f'    Downloaded: {mp4.stat().st_size//1024}KB')
        except Exception as e:
            print(f'    Download failed: {e}')
            continue
    
    print(f'  Extracting {gloss} from {mp4.name}...')
    bones = extract_peak_pose(mp4)
    if bones:
        euler = to_euler_repr(bones)
        real_poses[gloss] = euler
        print(f'    Got {len(euler)} bone rotations')
    else:
        print(f'    No pose extracted')

# ── Generate SIGN_POSES dict for demo_server.py ───────────────────────────────
print('\n\nGenerated SIGN_POSES:')
print('SIGN_POSES = {')
for gloss, pose in sorted(real_poses.items()):
    print(f'    "{gloss}": {{')
    for bone, angles in pose.items():
        print(f'        "{bone}": {angles},')
    print('    },')
print('}')

# Save to JSON for use in demo_server
out = ROOT / 'data' / 'processed' / 'sign_poses_real.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(real_poses, indent=2))
print(f'\nSaved to {out}')
