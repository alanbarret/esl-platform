"""
Full Avatar Renderer with Hand + Finger Animation
Uses MediaPipe Holistic: body pose + both hands (21 landmarks each)
Renders Arab man stylized avatar with all 5 fingers per hand.
"""
import cv2, numpy as np, math, struct, json
from pathlib import Path
import mediapipe as mp

ROOT = Path('/root/.openclaw/workspace/esl-platform')
MOTION_DB = ROOT / 'data' / 'motion_db'
mp_holistic = mp.solutions.holistic

# ── Extract pose + hands ──────────────────────────────────────────────────────
print('Extracting pose + hand landmarks...')
vid = MOTION_DB / 'HOW_ARE_YOU.mp4'
cap = cv2.VideoCapture(str(vid))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as ho:
    while cap.isOpened():
        ret, fr = cap.read()
        if not ret: break
        frames.append(ho.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
cap.release()
print(f'  {len(frames)} frames')

# ── Body landmark helpers ─────────────────────────────────────────────────────
AW, AH = 540, 720

def body_pt(lms, idx):
    if lms is None: return None
    lm = lms.landmark[idx]
    return (lm.x * VW, (1 - lm.y) * VH)

def hand_pt(lms, idx):
    if lms is None: return None
    lm = lms.landmark[idx]
    return (lm.x * VW, (1 - lm.y) * VH)

# Normalize body to screen coords
def norm_body(pt, body_pts):
    """Normalize body landmark to avatar screen space."""
    if pt is None or body_pts['lsh'] is None or body_pts['rsh'] is None:
        return None
    # Use shoulder width as scale reference
    lsh, rsh = body_pts['lsh'], body_pts['rsh']
    sh_w = abs(rsh[0] - lsh[0])
    if sh_w < 1: return None
    sh_mid_x = (lsh[0] + rsh[0]) / 2
    sh_mid_y = (lsh[1] + rsh[1]) / 2

    # Map to avatar space
    scale = (AW * 0.38) / sh_w
    ax = AW//2 + (pt[0] - sh_mid_x) * scale
    ay = AH * 0.32 + (pt[1] - sh_mid_y) * scale
    return (int(ax), int(ay))

def norm_hand(pt, body_pts, wrist_screen):
    """Normalize hand landmark relative to wrist position."""
    if pt is None or wrist_screen is None or body_pts['lsh'] is None:
        return None
    lsh, rsh = body_pts['lsh'], body_pts['rsh']
    sh_w = abs(rsh[0] - lsh[0])
    if sh_w < 1: return None
    sh_mid_x = (lsh[0] + rsh[0]) / 2
    sh_mid_y = (lsh[1] + rsh[1]) / 2
    scale = (AW * 0.38) / sh_w
    ax = AW//2 + (pt[0] - sh_mid_x) * scale
    ay = AH * 0.32 + (pt[1] - sh_mid_y) * scale
    return (int(ax), int(ay))

# MediaPipe hand connections (finger chains)
FINGERS = [
    # Thumb
    [(0,1),(1,2),(2,3),(3,4)],
    # Index
    [(0,5),(5,6),(6,7),(7,8)],
    # Middle
    [(0,9),(9,10),(10,11),(11,12)],
    # Ring
    [(0,13),(13,14),(14,15),(15,16)],
    # Pinky
    [(0,17),(17,18),(18,19),(19,20)],
]
PALM_CONN = [(0,1),(1,5),(5,9),(9,13),(13,17),(17,0),(0,5),(0,9),(0,13)]

# Body connections
BODY_CONN = [
    (11,12),  # shoulders
    (11,13),(13,15),  # left arm
    (12,14),(14,16),  # right arm
    (11,23),(12,24),  # torso sides
    (23,24),          # hips
    (23,25),(25,27),  # left leg
    (24,26),(26,28),  # right leg
]

# ── Smoothing ─────────────────────────────────────────────────────────────────
def smooth_lms(frames_lms, window=3):
    """Smooth landmark positions over time."""
    n = len(frames_lms)
    smoothed = []
    for i in range(n):
        s, e = max(0, i-window//2), min(n, i+window//2+1)
        buf = [frames_lms[j] for j in range(s, e) if frames_lms[j] is not None]
        if not buf:
            smoothed.append(frames_lms[i])
            continue
        avg = []
        for li in range(len(buf[0])):
            xs = [b[li][0] for b in buf]; ys = [b[li][1] for b in buf]
            avg.append((sum(xs)/len(xs), sum(ys)/len(ys)))
        smoothed.append(avg)
    return smoothed

# Extract raw landmark lists
def extract_lms(res, which='pose'):
    if which == 'pose':
        if res.pose_landmarks is None: return None
        return [(lm.x*VW, (1-lm.y)*VH) for lm in res.pose_landmarks.landmark]
    elif which == 'lhand':
        if res.left_hand_landmarks is None: return None
        return [(lm.x*VW, (1-lm.y)*VH) for lm in res.left_hand_landmarks.landmark]
    elif which == 'rhand':
        if res.right_hand_landmarks is None: return None
        return [(lm.x*VW, (1-lm.y)*VH) for lm in res.right_hand_landmarks.landmark]

raw_pose  = [extract_lms(r, 'pose')  for r in frames]
raw_lhand = [extract_lms(r, 'lhand') for r in frames]
raw_rhand = [extract_lms(r, 'rhand') for r in frames]

# Smooth
sm_pose  = smooth_lms(raw_pose,  window=5)
sm_lhand = smooth_lms(raw_lhand, window=3)
sm_rhand = smooth_lms(raw_rhand, window=3)

# ── Renderer ──────────────────────────────────────────────────────────────────
font = cv2.FONT_HERSHEY_SIMPLEX

SKIN = (185, 148, 115)
SKIN_D = (165, 128, 95)
ROBE = (232, 230, 226)
ROBE_D = (208, 206, 202)
SLEEVE = (70, 110, 165)
SLEEVE_D = (50, 85, 135)
TROUSER = (65, 48, 115)
TROUSER_D = (50, 36, 90)

def draw_frame(pose_lms, lhand_lms, rhand_lms, label, prog):
    img = np.zeros((AH, AW, 3), dtype=np.uint8)
    for y in range(AH):
        t = y / AH
        img[y,:] = (int(8+t*18), int(8+t*14), int(18+t*32))

    if pose_lms is None:
        return img

    # Compute scale from shoulder width
    lsh = pose_lms[11]; rsh = pose_lms[12]
    sh_w = abs(rsh[0] - lsh[0])
    if sh_w < 5: return img
    sc = (AW * 0.38) / sh_w
    sh_mid_x = (lsh[0] + rsh[0]) / 2
    sh_mid_y = (lsh[1] + rsh[1]) / 2

    def to_screen(pt):
        return (int(AW//2 + (pt[0]-sh_mid_x)*sc),
                int(AH*0.32 + (pt[1]-sh_mid_y)*sc))

    # Get key points
    pts = {i: to_screen(pose_lms[i]) for i in range(33)}

    # ── Floor shadow ──────────────────────────────────────────────────────────
    foot_y = max(pts.get(27,(0,AH-80))[1], pts.get(28,(0,AH-80))[1])
    shadow_y = min(foot_y + 60, AH-20)
    cv2.ellipse(img,(AW//2,shadow_y),(80,14),0,0,360,(0,0,0),-1,cv2.LINE_AA)

    # ── Kandura (robe body) ───────────────────────────────────────────────────
    lhip = pts[23]; rhip = pts[24]
    lsh2 = pts[11]; rsh2 = pts[12]
    nose = pts[0]

    # Full robe from shoulders to feet
    robe_pts = np.array([
        (lsh2[0]-30, lsh2[1]),
        (rsh2[0]+30, rsh2[1]),
        (rhip[0]+50, rhip[1]+30),
        (rhip[0]+45, shadow_y-5),
        (lhip[0]-45, shadow_y-5),
        (lhip[0]-50, rhip[1]+30),
    ], np.int32)
    cv2.fillPoly(img, [robe_pts], ROBE, cv2.LINE_AA)
    cv2.polylines(img, [robe_pts], True, ROBE_D, 2, cv2.LINE_AA)
    # Center line of robe
    neck_mid = ((lsh2[0]+rsh2[0])//2, (lsh2[1]+rsh2[1])//2)
    hip_mid = ((lhip[0]+rhip[0])//2, (lhip[1]+rhip[1])//2)
    cv2.line(img, neck_mid, hip_mid, ROBE_D, 2, cv2.LINE_AA)

    # ── Left arm + sleeve ─────────────────────────────────────────────────────
    for (a,b), thk, col in [(( 11,13),14,SLEEVE),(( 13,15),12,SLEEVE_D)]:
        cv2.line(img, pts[a], pts[b], tuple(max(0,c-20) for c in col), thk+2, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], col, thk, cv2.LINE_AA)

    # ── Legs ──────────────────────────────────────────────────────────────────
    for (a,b) in [(23,25),(25,27),(24,26),(26,28)]:
        cv2.line(img, pts[a], pts[b], TROUSER_D, 16, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], TROUSER, 12, cv2.LINE_AA)
    # Feet
    for ankle,toe_x in [(pts[27],pts[27][0]+30),(pts[28],pts[28][0]-30)]:
        cv2.line(img, ankle, (toe_x,ankle[1]+8), (45,35,70), 10, cv2.LINE_AA)

    # ── Right arm + sleeve ────────────────────────────────────────────────────
    for (a,b), thk, col in [(( 12,14),14,SLEEVE),(( 14,16),12,SLEEVE_D)]:
        cv2.line(img, pts[a], pts[b], tuple(max(0,c-20) for c in col), thk+2, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], col, thk, cv2.LINE_AA)

    # ── Hands + fingers ───────────────────────────────────────────────────────
    def draw_hand(hand_lms, wrist_pose_idx, color_base, color_tip):
        if hand_lms is None: return
        hpts = [to_screen(hand_lms[i]) for i in range(21)]
        # Palm fill
        palm_idx = [0,1,5,9,13,17]
        palm_pts = np.array([hpts[i] for i in palm_idx], np.int32)
        cv2.fillPoly(img, [palm_pts], color_base, cv2.LINE_AA)
        # Draw each finger
        for finger in FINGERS:
            for (a,b) in finger:
                thickness = 5 if b in [4,8,12,16,20] else 6  # tips thinner
                col = color_tip if b in [4,8,12,16,20] else color_base
                cv2.line(img, hpts[a], hpts[b], tuple(max(0,c-15) for c in col), thickness+2, cv2.LINE_AA)
                cv2.line(img, hpts[a], hpts[b], col, thickness, cv2.LINE_AA)
        # Fingertip dots
        for tip in [4,8,12,16,20]:
            cv2.circle(img, hpts[tip], 5, color_tip, -1, cv2.LINE_AA)
        # Knuckle dots
        for kn in [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]:
            cv2.circle(img, hpts[kn], 3, color_base, -1, cv2.LINE_AA)

    draw_hand(lhand_lms, 15, SKIN, SKIN_D)
    draw_hand(rhand_lms, 16, SKIN, SKIN_D)

    # ── Neck + Head ───────────────────────────────────────────────────────────
    neck = ((pts[11][0]+pts[12][0])//2, (pts[11][1]+pts[12][1])//2)
    cv2.line(img, neck, pts[0], SKIN, 16, cv2.LINE_AA)
    cv2.line(img, neck, pts[0], tuple(min(255,c+10) for c in SKIN), 10, cv2.LINE_AA)

    # Head
    hx, hy = pts[0]; hr = int(sh_w * sc * 0.52)
    hr = max(28, min(hr, 60))
    cv2.ellipse(img,(hx,hy),(hr,int(hr*1.18)),0,0,360,SKIN_D,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy-4),(hr,int(hr*1.1)),0,0,360,SKIN,-1,cv2.LINE_AA)
    # Eyes
    ex = int(hr*0.3); ey = int(hr*0.2)
    for ex2 in [-ex,ex]:
        cv2.circle(img,(hx+ex2,hy-ey),int(hr*0.12),(40,28,18),-1,cv2.LINE_AA)
        cv2.circle(img,(hx+ex2+2,hy-ey-2),int(hr*0.05),(220,220,225),-1)
    # Eyebrows
    cv2.line(img,(hx-ex-int(hr*0.12),hy-ey-int(hr*0.17)),(hx-ex+int(hr*0.12),hy-ey-int(hr*0.13)),(42,28,18),3,cv2.LINE_AA)
    cv2.line(img,(hx+ex-int(hr*0.12),hy-ey-int(hr*0.13)),(hx+ex+int(hr*0.12),hy-ey-int(hr*0.17)),(42,28,18),3,cv2.LINE_AA)
    # Nose
    cv2.ellipse(img,(hx,hy+int(hr*0.12)),(int(hr*0.13),int(hr*0.2)),0,0,360,SKIN_D,-1,cv2.LINE_AA)
    # Mouth
    cv2.ellipse(img,(hx,hy+int(hr*0.38)),(int(hr*0.22),int(hr*0.08)),0,0,180,(145,105,75),2,cv2.LINE_AA)
    # Beard
    cv2.ellipse(img,(hx,hy+int(hr*0.42)),(int(hr*0.42),int(hr*0.22)),0,0,180,(35,25,18),-1,cv2.LINE_AA)
    # Ghutrah (headdress)
    gpts = np.array([
        (hx-hr-10, hy-int(hr*0.15)),
        (hx, hy-int(hr*1.38)),
        (hx+hr+10, hy-int(hr*0.15)),
        (hx+hr-4, hy+int(hr*0.38)),
        (hx-hr+4, hy+int(hr*0.38)),
    ], np.int32)
    cv2.fillPoly(img,[gpts],ROBE,cv2.LINE_AA)
    cv2.polylines(img,[gpts],True,ROBE_D,2,cv2.LINE_AA)
    # Agal
    cv2.ellipse(img,(hx,hy-int(hr*0.88)),(hr-5,int(hr*0.23)),0,0,360,(18,15,12),-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy-int(hr*0.88)),(hr-5,int(hr*0.23)),0,0,360,(35,30,25),2,cv2.LINE_AA)

    # ── Label + progress bar ──────────────────────────────────────────────────
    ts = cv2.getTextSize(label,font,1.0,3)[0]
    lx = (AW-ts[0])//2
    cv2.rectangle(img,(lx-8,AH-48),(lx+ts[0]+8,AH-22),(8,8,18),-1)
    cv2.putText(img,label,(lx+1,AH-28),font,1.0,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(img,label,(lx,AH-29),font,1.0,(168,255,75),3,cv2.LINE_AA)
    bw=int(AW*0.8); bx=int(AW*0.1)
    cv2.rectangle(img,(bx,AH-14),(bx+bw,AH-6),(18,16,32),-1)
    cv2.rectangle(img,(bx,AH-14),(bx+int(bw*prog),AH-6),(124,58,237),-1)

    return img

# ── Render video ──────────────────────────────────────────────────────────────
print(f'Rendering {len(sm_pose)} frames with hands+fingers...')
OUT = '/root/.openclaw/workspace/esl_hello_hands.avi'
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'MJPG'), fps, (AW,AH))

for i in range(len(sm_pose)):
    prog = i / max(len(sm_pose)-1, 1)
    img = draw_frame(sm_pose[i], sm_lhand[i], sm_rhand[i], 'HELLO', prog)
    out.write(img)

out.release()
import os
print(f'Done! {OUT} ({os.path.getsize(OUT)//1024}KB, {len(sm_pose)/fps:.1f}s)')
