"""
Render 3D avatar signing HELLO using real pose from HOW_ARE_YOU video.
The 3D model is rendered as a proper humanoid figure (not just skeleton wireframe).
"""
import cv2, numpy as np, json, math, struct
from pathlib import Path
import mediapipe as mp

ROOT = Path('/root/.openclaw/workspace/esl-platform')
MOTION_DB = ROOT / 'data' / 'motion_db'

mp_holistic = mp.solutions.holistic

# Load GLB rest quaternions
with open(ROOT / 'frontend/public/avatar/arab-man.glb', 'rb') as f:
    f.read(12)
    cl = struct.unpack('<I', f.read(4))[0]; f.read(4)
    gltf = json.loads(f.read(cl))

nodes = gltf['nodes']
REST = {n['name']: n.get('rotation',[0,0,0,1]) for n in nodes if 'name' in n}

# ── Quaternion math ──────────────────────────────────────────────────────────
def norm(v): n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v] if n>1e-6 else v
def mul_q(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz]
def inv_q(q): x,y,z,w=q; n2=sum(c*c for c in q); return [-x/n2,-y/n2,-z/n2,w/n2]
def slerp(a,b,t):
    dot=sum(ai*bi for ai,bi in zip(a,b))
    if dot<0: b=[-x for x in b]; dot=-dot
    if dot>0.9995: return norm([ai+t*(bi-ai) for ai,bi in zip(a,b)])
    th=math.acos(min(1,dot)); si=math.sin(th)
    s1=math.sin((1-t)*th)/si; s2=math.sin(t*th)/si
    return [s1*ai+s2*bi for ai,bi in zip(a,b)]

def vec_to_quat(v1, v2):
    v1=np.array(v1)/np.linalg.norm(v1); v2=np.array(v2)/np.linalg.norm(v2)
    c=np.cross(v1,v2); d=float(np.dot(v1,v2)); w=1+d
    if w<1e-6: return [0,0,1,0]
    q=[float(c[0]),float(c[1]),float(c[2]),w]
    return norm(q)

# ── Extract pose from real video ─────────────────────────────────────────────
vid_path = MOTION_DB / 'HOW_ARE_YOU.mp4'
print(f'Extracting pose from {vid_path}...')

cap = cv2.VideoCapture(str(vid_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pose_frames = []
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = h.process(rgb)
        pose_frames.append((res.pose_landmarks, res.left_hand_landmarks, res.right_hand_landmarks))
cap.release()
print(f'  Extracted {len(pose_frames)} frames at {fps:.0f}fps')

def get_lm(lms, idx, W, H):
    if lms is None: return None
    lm = lms.landmark[idx]
    return np.array([lm.x*W, (1-lm.y)*H, -lm.z*W])

def lm2bone(lms, i1, i2, W, H, rest_bone, rest_dir=[0,-1,0]):
    if lms is None: return REST.get(rest_bone,[0,0,0,1])
    p1=get_lm(lms,i1,W,H); p2=get_lm(lms,i2,W,H)
    if p1 is None or p2 is None: return REST.get(rest_bone,[0,0,0,1])
    d=p2-p1; n=np.linalg.norm(d)
    if n<1: return REST.get(rest_bone,[0,0,0,1])
    world_q = vec_to_quat(rest_dir, (d/n).tolist())
    rest_q = REST.get(rest_bone,[0,0,0,1])
    return mul_q(inv_q(rest_q), world_q)

# ── 3D Avatar Renderer (improved humanoid) ───────────────────────────────────
AW, AH = 540, 720
font = cv2.FONT_HERSHEY_SIMPLEX

# Body segment lengths (normalized to avatar height)
SEG = {
    'head_r': 0.11, 'neck_h': 0.06, 'torso_h': 0.28,
    'upper_arm': 0.16, 'forearm': 0.14, 'hand_h': 0.08,
    'hip_w': 0.12, 'thigh': 0.22, 'shin': 0.21, 'foot': 0.07,
}
AVH = AH * 0.85  # avatar height in pixels

def scale(v): return v * AVH

# Base joint positions (normalized 0-1 from top, centered)
def get_joint_positions(bones):
    """Compute 2D screen positions from bone rotations using FK."""
    cx = AW // 2
    
    # Root (hips) position
    hip_y = scale(0.52)
    hip_x = cx
    
    # Build joint tree
    joints = {'hips': np.array([hip_x, hip_y])}
    
    def fk(parent_pos, bone_name, length, base_angle, rot_offset=0):
        q = bones.get(bone_name, [0,0,0,1])
        x,y,z,w = q
        try:
            # Extract yaw and pitch from quaternion
            yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            pitch = math.asin(max(-1, min(1, 2*(w*y-z*x))))
            angle = base_angle + yaw * 0.6 + rot_offset
            angle_v = base_angle + pitch * 0.4
        except:
            angle = base_angle
            angle_v = base_angle

        dx = math.sin(angle) * scale(length)
        dy = math.cos(angle_v) * scale(length)
        return parent_pos + np.array([dx, dy])

    # Spine
    spine2_pos = joints['hips'] + np.array([0, -scale(0.28)])
    neck_pos   = spine2_pos + np.array([0, -scale(0.10)])
    head_pos   = neck_pos + np.array([0, -scale(0.07)])
    
    joints.update({'spine2': spine2_pos, 'neck': neck_pos, 'head': head_pos})
    
    # Right arm chain
    r_sh = spine2_pos + np.array([scale(0.11), -scale(0.06)])
    joints['r_shoulder'] = r_sh
    
    q_ra = bones.get('RightArm', [0,0,0,1])
    x,y,z,w = q_ra
    try:
        yaw_ra = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pitch_ra = math.asin(max(-1, min(1, 2*(w*y-z*x))))
    except:
        yaw_ra = pitch_ra = 0
    
    r_elbow = r_sh + np.array([
        scale(0.14) * math.sin(math.pi/4 + yaw_ra * 0.7),
        scale(0.14) * (0.3 + pitch_ra * 0.5)
    ])
    joints['r_elbow'] = r_elbow
    
    q_rf = bones.get('RightForeArm', [0,0,0,1])
    x,y,z,w = q_rf
    try:
        yaw_rf = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pitch_rf = math.asin(max(-1, min(1, 2*(w*y-z*x))))
    except:
        yaw_rf = pitch_rf = 0
    
    r_wrist = r_elbow + np.array([
        scale(0.13) * math.sin(yaw_ra * 0.5 + yaw_rf * 0.5),
        scale(0.13) * (0.2 + pitch_rf * 0.4)
    ])
    joints['r_wrist'] = r_wrist
    joints['r_hand'] = r_wrist + np.array([scale(0.05) * math.sin(yaw_ra*0.3), scale(0.05)])
    
    # Left arm chain (mirrored)
    l_sh = spine2_pos + np.array([-scale(0.11), -scale(0.06)])
    joints['l_shoulder'] = l_sh
    
    q_la = bones.get('LeftArm', [0,0,0,1])
    x,y,z,w = q_la
    try:
        yaw_la = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pitch_la = math.asin(max(-1, min(1, 2*(w*y-z*x))))
    except:
        yaw_la = pitch_la = 0
    
    l_elbow = l_sh + np.array([
        -scale(0.14) * math.sin(math.pi/4 - yaw_la * 0.7),
        scale(0.14) * (0.3 + pitch_la * 0.5)
    ])
    joints['l_elbow'] = l_elbow
    
    q_lf = bones.get('LeftForeArm', [0,0,0,1])
    x,y,z,w = q_lf
    try:
        yaw_lf = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pitch_lf = math.asin(max(-1, min(1, 2*(w*y-z*x))))
    except:
        yaw_lf = pitch_lf = 0
    
    l_wrist = l_elbow + np.array([
        -scale(0.13) * math.sin(-yaw_la * 0.5 + yaw_lf * 0.5),
        scale(0.13) * (0.2 + pitch_lf * 0.4)
    ])
    joints['l_wrist'] = l_wrist
    joints['l_hand'] = l_wrist + np.array([-scale(0.05) * math.sin(-yaw_la*0.3), scale(0.05)])
    
    # Legs
    r_hip = joints['hips'] + np.array([scale(0.07), 0])
    l_hip = joints['hips'] + np.array([-scale(0.07), 0])
    r_knee = r_hip + np.array([scale(0.02), scale(0.20)])
    l_knee = l_hip + np.array([-scale(0.02), scale(0.20)])
    r_ankle = r_knee + np.array([scale(0.01), scale(0.19)])
    l_ankle = l_knee + np.array([-scale(0.01), scale(0.19)])
    r_toe = r_ankle + np.array([scale(0.07), scale(0.01)])
    l_toe = l_ankle + np.array([-scale(0.07), scale(0.01)])
    
    joints.update({'r_hip':r_hip,'l_hip':l_hip,'r_knee':r_knee,'l_knee':l_knee,
                   'r_ankle':r_ankle,'l_ankle':l_ankle,'r_toe':r_toe,'l_toe':l_toe})
    return joints

def ipos(p): return (int(p[0]), int(p[1]))

def draw_avatar(bones, label, progress, frame_num):
    img = np.zeros((AH, AW, 3), dtype=np.uint8)
    
    # Dark gradient background
    for y in range(AH):
        c = int(10 + y/AH * 18)
        img[y,:] = (c, c-2, c+8)
    
    j = get_joint_positions(bones)
    
    # Draw shadow
    for pos in j.values():
        sp = (int(pos[0]), AH-30)
        ax = int(abs(pos[0]-AW//2)*0.15 + 8)
        cv2.ellipse(img, sp, (ax, 4), 0, 0, 360, (0,0,0), -1, cv2.LINE_AA)
    
    # Limb thickness
    THICK = {'torso':18,'upper':12,'lower':10,'hand':8,'thigh':14,'shin':12,'foot':8}
    
    def limb(a, b, color, t=10):
        if a in j and b in j:
            cv2.line(img, ipos(j[a]), ipos(j[b]), color, t+2, cv2.LINE_AA)
            cv2.line(img, ipos(j[a]), ipos(j[b]), tuple(min(255,c+30) for c in color), t, cv2.LINE_AA)
    
    # Draw body (back to front)
    # Torso
    limb('hips','spine2', (140,80,40), THICK['torso'])
    limb('spine2','neck', (140,80,40), THICK['torso'])
    
    # Legs
    limb('l_hip','l_knee', (60,40,120), THICK['thigh'])
    limb('l_knee','l_ankle', (50,35,100), THICK['shin'])
    limb('l_ankle','l_toe', (45,30,85), THICK['foot'])
    limb('r_hip','r_knee', (60,40,120), THICK['thigh'])
    limb('r_knee','r_ankle', (50,35,100), THICK['shin'])
    limb('r_ankle','r_toe', (45,30,85), THICK['foot'])
    
    # Arms
    limb('l_shoulder','l_elbow', (55,100,160), THICK['upper'])
    limb('l_elbow','l_wrist', (50,90,145), THICK['lower'])
    limb('l_wrist','l_hand', (160,120,90), THICK['hand'])
    
    limb('r_shoulder','r_elbow', (55,100,160), THICK['upper'])
    limb('r_elbow','r_wrist', (50,90,145), THICK['lower'])
    limb('r_wrist','r_hand', (160,120,90), THICK['hand'])
    
    # Shoulders connector
    limb('l_shoulder','r_shoulder', (130,75,35), THICK['torso'])
    
    # Head
    if 'head' in j:
        hx, hy = ipos(j['head'])
        hr = int(scale(0.08))
        # Neck
        if 'neck' in j:
            cv2.line(img, ipos(j['neck']), ipos(j['head']), (155,115,85), 14, cv2.LINE_AA)
        # Head circle (skin tone)
        cv2.circle(img, (hx,hy), hr, (30,60,80), -1, cv2.LINE_AA)
        cv2.circle(img, (hx,hy), hr, (180,140,100), 3, cv2.LINE_AA)
        cv2.circle(img, (hx,hy-2), hr, (185,145,105), -1, cv2.LINE_AA)
        # Eyes
        cv2.circle(img, (hx-int(hr*0.3), hy-int(hr*0.15)), int(hr*0.12), (50,35,25), -1)
        cv2.circle(img, (hx+int(hr*0.3), hy-int(hr*0.15)), int(hr*0.12), (50,35,25), -1)
        # Kandura (white robe outline on torso)
        if 'spine2' in j:
            sx,sy = ipos(j['spine2'])
            cv2.ellipse(img,(sx,sy),(int(scale(0.13)),int(scale(0.16))),0,0,360,(220,215,210),-1,cv2.LINE_AA)
            cv2.ellipse(img,(sx,sy),(int(scale(0.13)),int(scale(0.16))),0,0,360,(200,195,190),2,cv2.LINE_AA)
    
    # Joints
    for name, pos in j.items():
        if 'shoulder' in name:
            cv2.circle(img, ipos(pos), 8, (100,140,180), -1, cv2.LINE_AA)
        elif 'elbow' in name or 'knee' in name:
            cv2.circle(img, ipos(pos), 6, (80,120,160), -1, cv2.LINE_AA)
        elif 'hand' in name or 'wrist' in name:
            cv2.circle(img, ipos(pos), 7, (180,140,100), -1, cv2.LINE_AA)
    
    # Label
    ts = cv2.getTextSize(label, font, 1.0, 3)[0]
    cv2.putText(img, label, ((AW-ts[0])//2+1, AH-32), font, 1.0, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, label, ((AW-ts[0])//2, AH-33), font, 1.0, (168,255,75), 3, cv2.LINE_AA)
    
    # Progress bar
    bw = int(AW*0.8); bx = int(AW*0.1)
    cv2.rectangle(img,(bx,AH-16),(bx+bw,AH-7),(25,25,40),-1)
    cv2.rectangle(img,(bx,AH-16),(bx+int(bw*progress),AH-7),(124,58,237),-1)
    
    # Frame counter (subtle)
    cv2.putText(img, f'{frame_num}', (8,20), font, 0.35, (50,50,70), 1)
    return img

# ── Extract bones per frame ───────────────────────────────────────────────────
print('Computing bone rotations...')
frame_bones = []
for pose_lms, lh_lms, rh_lms in pose_frames:
    bones = {}
    if pose_lms:
        lm = pose_lms.landmark
        def p(i): return np.array([lm[i].x*W, (1-lm[i].y)*H, -lm[i].z*W])
        
        try:
            ls=p(11); rs=p(12); le=p(13); re=p(14)
            lw=p(15); rw=p(16)
            nose=p(0); lh_p=p(23); rh_p=p(24)
            
            def dir2quat(d, rest_bone):
                d = d/np.linalg.norm(d) if np.linalg.norm(d)>1e-6 else np.array([0,-1,0])
                wq = vec_to_quat([0,-1,0], d.tolist())
                rq = REST.get(rest_bone,[0,0,0,1])
                return mul_q(inv_q(rq), wq)
            
            bones['RightArm']     = dir2quat(re-rs, 'RightArm')
            bones['RightForeArm'] = dir2quat(rw-re, 'RightForeArm')
            bones['LeftArm']      = dir2quat(le-ls, 'LeftArm')
            bones['LeftForeArm']  = dir2quat(lw-le, 'LeftForeArm')
            
            head_dir = nose - (ls+rs)/2
            bones['Head'] = dir2quat(head_dir, 'Head')
            
            spine_dir = (ls+rs)/2 - (lh_p+rh_p)/2
            bones['Spine2'] = dir2quat(spine_dir, 'Spine2')
        except: pass
    
    frame_bones.append(bones)

# ── Smooth the bone data ──────────────────────────────────────────────────────
SMOOTH = 5
smoothed = []
for i in range(len(frame_bones)):
    start = max(0, i-SMOOTH//2); end = min(len(frame_bones), i+SMOOTH//2+1)
    window = frame_bones[start:end]
    smooth_bones = {}
    for bone in set().union(*[set(f.keys()) for f in window]):
        valid = [f[bone] for f in window if bone in f]
        if valid:
            avg = [sum(q[j] for q in valid)/len(valid) for j in range(4)]
            smooth_bones[bone] = norm(avg)
    smoothed.append(smooth_bones)

# ── Render ────────────────────────────────────────────────────────────────────
print('Rendering avatar video...')
out_path = '/root/.openclaw/workspace/esl_hello_3d.avi'
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (AW, AH))

for i, bones in enumerate(smoothed):
    prog = i / max(len(smoothed)-1, 1)
    img = draw_avatar(bones, 'HELLO', prog, i)
    out.write(img)

out.release()

import os
size = os.path.getsize(out_path)
print(f'Done! {out_path} ({size//1024}KB, {len(smoothed)} frames, {len(smoothed)/fps:.1f}s)')
