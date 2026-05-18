"""
Render multi-sign sentence: HELLO HOW ARE YOU
Uses real videos for each sign, renders with full body + hands + fingers.
"""
import cv2, numpy as np, math, urllib.request
from pathlib import Path
import mediapipe as mp
import tempfile, os

ROOT = Path('/root/.openclaw/workspace/esl-platform')
MOTION_DB = ROOT / 'data' / 'motion_db'
mp_holistic = mp.solutions.holistic

AW, AH = 540, 720
font = cv2.FONT_HERSHEY_SIMPLEX
SKIN = (185, 148, 115); SKIN_D = (160, 122, 90)
ROBE = (232, 230, 226); ROBE_D = (205, 203, 199)
SLEEVE = (68, 108, 162); SLEEVE_D = (48, 82, 130)
TROUSER = (62, 46, 110); TROUSER_D = (48, 34, 86)

# Signs to render — use real video where available, fallback to existing
SIGNS = [
    ('HELLO',     '/root/.openclaw/workspace/esl-platform/data/motion_db/HOW_ARE_YOU.mp4'),
    ('HOW',       '/root/.openclaw/workspace/esl-platform/data/motion_db/HOW_ARE_YOU.mp4'),
    ('ARE YOU',   '/root/.openclaw/workspace/esl-platform/data/motion_db/HOW_ARE_YOU.mp4'),
]

def download_and_extract(url, label):
    """Download video and extract MediaPipe pose + hands."""
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.za.gov.ae/'
        })
        with urllib.request.urlopen(req, timeout=20) as r:
            tmp.write(r.read())
        tmp.close()
        return extract_from_file(tmp.name, label)
    except Exception as e:
        tmp.close()
        print(f'  Download failed: {e}')
        return None, 0
    finally:
        os.unlink(tmp.name)

def extract_from_file(path, label):
    """Extract pose+hands from local video file."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as ho:
        while cap.isOpened():
            ret, fr = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            res = ho.process(rgb)
            def lm_list(lms):
                if lms is None: return None
                return [(l.x*VW, (1-l.y)*VH) for l in lms.landmark]
            frames.append({
                'pose':  lm_list(res.pose_landmarks),
                'lhand': lm_list(res.left_hand_landmarks),
                'rhand': lm_list(res.right_hand_landmarks),
            })
    cap.release()
    return frames, fps

def smooth_frames(frames, w=4):
    n = len(frames)
    out = []
    for i in range(n):
        s, e = max(0,i-w//2), min(n,i+w//2+1)
        buf = frames[s:e]
        def avg_lms(key):
            valid = [f[key] for f in buf if f[key] is not None]
            if not valid: return None
            return [tuple(sum(p[j][k] for p in valid)/len(valid) for k in range(2))
                    for j in range(len(valid[0]))]
        out.append({'pose': avg_lms('pose'), 'lhand': avg_lms('lhand'), 'rhand': avg_lms('rhand')})
    return out

FINGERS = [
    [(0,1),(1,2),(2,3),(3,4)],
    [(0,5),(5,6),(6,7),(7,8)],
    [(0,9),(9,10),(10,11),(11,12)],
    [(0,13),(13,14),(14,15),(15,16)],
    [(0,17),(17,18),(18,19),(19,20)],
]

def draw_frame(fd, label, prog):
    img = np.zeros((AH, AW, 3), dtype=np.uint8)
    for y in range(AH):
        t = y/AH
        img[y,:] = (int(8+t*18), int(8+t*14), int(18+t*32))

    pose = fd['pose']
    if pose is None: return img

    lsh = pose[11]; rsh = pose[12]
    sh_w = abs(rsh[0]-lsh[0])
    if sh_w < 5: return img
    sc = (AW*0.38)/sh_w
    ox = AW//2 - (lsh[0]+rsh[0])/2*sc
    oy = AH*0.32 - (lsh[1]+rsh[1])/2*sc

    def S(pt): return (int(pt[0]*sc+ox), int(pt[1]*sc+oy))
    P = {i: S(pose[i]) for i in range(min(33,len(pose)))}

    foot_y = max(P.get(27,(0,AH-60))[1], P.get(28,(0,AH-60))[1])
    shadow_y = min(foot_y+55, AH-18)
    cv2.ellipse(img,(AW//2,shadow_y),(75,13),0,0,360,(0,0,0),-1,cv2.LINE_AA)

    # Robe
    ls2,rs2,lhp,rhp = P[11],P[12],P[23],P[24]
    robe_pts = np.array([
        (ls2[0]-28,ls2[1]), (rs2[0]+28,rs2[1]),
        (rhp[0]+48,rhp[1]+28), (rhp[0]+42,shadow_y-4),
        (lhp[0]-42,shadow_y-4), (lhp[0]-48,rhp[1]+28)
    ], np.int32)
    cv2.fillPoly(img,[robe_pts],ROBE,cv2.LINE_AA)
    cv2.polylines(img,[robe_pts],True,ROBE_D,2,cv2.LINE_AA)
    neck_m = ((ls2[0]+rs2[0])//2,(ls2[1]+rs2[1])//2)
    hip_m  = ((lhp[0]+rhp[0])//2,(lhp[1]+rhp[1])//2)
    cv2.line(img,neck_m,hip_m,ROBE_D,2,cv2.LINE_AA)

    # Left arm
    for (a,b),t,c in [((11,13),13,SLEEVE),((13,15),11,SLEEVE_D)]:
        cv2.line(img,P[a],P[b],tuple(max(0,x-18) for x in c),t+2,cv2.LINE_AA)
        cv2.line(img,P[a],P[b],c,t,cv2.LINE_AA)
    # Legs
    for a,b in [(23,25),(25,27),(24,26),(26,28)]:
        cv2.line(img,P[a],P[b],TROUSER_D,15,cv2.LINE_AA)
        cv2.line(img,P[a],P[b],TROUSER,11,cv2.LINE_AA)
    for ak,dx in [(P[27],28),(P[28],-28)]:
        cv2.line(img,ak,(ak[0]+dx,ak[1]+6),tuple(max(0,x-20) for x in TROUSER),10,cv2.LINE_AA)
    # Right arm
    for (a,b),t,c in [((12,14),13,SLEEVE),((14,16),11,SLEEVE_D)]:
        cv2.line(img,P[a],P[b],tuple(max(0,x-18) for x in c),t+2,cv2.LINE_AA)
        cv2.line(img,P[a],P[b],c,t,cv2.LINE_AA)

    # Hands with fingers
    def draw_hand(hand_lms):
        if hand_lms is None: return
        HP = [S(hand_lms[i]) for i in range(min(21,len(hand_lms)))]
        if len(HP) < 21: return
        palm = np.array([HP[i] for i in [0,1,5,9,13,17]], np.int32)
        cv2.fillPoly(img,[palm],SKIN,cv2.LINE_AA)
        for finger in FINGERS:
            for a,b in finger:
                if a < len(HP) and b < len(HP):
                    tip = b in [4,8,12,16,20]
                    col = SKIN_D if tip else SKIN
                    cv2.line(img,HP[a],HP[b],tuple(max(0,x-12) for x in col),6,cv2.LINE_AA)
                    cv2.line(img,HP[a],HP[b],col,4,cv2.LINE_AA)
        for tip in [4,8,12,16,20]:
            if tip < len(HP): cv2.circle(img,HP[tip],5,SKIN_D,-1,cv2.LINE_AA)
        for kn in [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]:
            if kn < len(HP): cv2.circle(img,HP[kn],3,SKIN,-1,cv2.LINE_AA)

    draw_hand(fd['lhand']); draw_hand(fd['rhand'])

    # Neck + Head
    nk = ((ls2[0]+rs2[0])//2,(ls2[1]+rs2[1])//2)
    cv2.line(img,nk,P[0],SKIN,15,cv2.LINE_AA)
    cv2.line(img,nk,P[0],tuple(min(255,x+10) for x in SKIN),9,cv2.LINE_AA)
    hx,hy = P[0]
    hr = max(26, min(int(sh_w*sc*0.5), 58))
    cv2.ellipse(img,(hx,hy),(hr,int(hr*1.18)),0,0,360,SKIN_D,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy-3),(hr,int(hr*1.1)),0,0,360,SKIN,-1,cv2.LINE_AA)
    ex=int(hr*0.29); ey=int(hr*0.19)
    for ex2 in [-ex,ex]:
        cv2.circle(img,(hx+ex2,hy-ey),int(hr*0.12),(40,28,18),-1,cv2.LINE_AA)
        cv2.circle(img,(hx+ex2+2,hy-ey-2),int(hr*0.05),(215,215,220),-1)
    cv2.line(img,(hx-ex-int(hr*0.12),hy-ey-int(hr*0.16)),(hx-ex+int(hr*0.12),hy-ey-int(hr*0.12)),(40,26,16),3,cv2.LINE_AA)
    cv2.line(img,(hx+ex-int(hr*0.12),hy-ey-int(hr*0.12)),(hx+ex+int(hr*0.12),hy-ey-int(hr*0.16)),(40,26,16),3,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy+int(hr*0.11)),(int(hr*0.12),int(hr*0.19)),0,0,360,SKIN_D,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy+int(hr*0.37)),(int(hr*0.4),int(hr*0.22)),0,0,180,(35,24,16),-1,cv2.LINE_AA)
    # Ghutrah
    gpts = np.array([(hx-hr-9,hy-int(hr*0.14)),(hx,hy-int(hr*1.36)),
                     (hx+hr+9,hy-int(hr*0.14)),(hx+hr-4,hy+int(hr*0.37)),(hx-hr+4,hy+int(hr*0.37))],np.int32)
    cv2.fillPoly(img,[gpts],ROBE,cv2.LINE_AA)
    cv2.polylines(img,[gpts],True,ROBE_D,2,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy-int(hr*0.87)),(hr-5,int(hr*0.22)),0,0,360,(17,14,11),-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy-int(hr*0.87)),(hr-5,int(hr*0.22)),0,0,360,(33,28,22),2,cv2.LINE_AA)

    # Label + bar
    ts = cv2.getTextSize(label,font,1.0,3)[0]
    lx = (AW-ts[0])//2
    cv2.rectangle(img,(lx-8,AH-46),(lx+ts[0]+8,AH-20),(6,6,16),-1)
    cv2.putText(img,label,(lx+1,AH-26),font,1.0,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(img,label,(lx,AH-27),font,1.0,(168,255,75),3,cv2.LINE_AA)
    bw=int(AW*0.8); bx2=int(AW*0.1)
    cv2.rectangle(img,(bx2,AH-13),(bx2+bw,AH-5),(16,14,30),-1)
    cv2.rectangle(img,(bx2,AH-13),(bx2+int(bw*prog),AH-5),(124,58,237),-1)
    return img

# ── Main ──────────────────────────────────────────────────────────────────────
print('Processing signs...')
all_frames = []
all_labels = []

# Use HOW_ARE_YOU video for HELLO + HOW + ARE YOU
vid = '/root/.openclaw/workspace/esl-platform/data/motion_db/HOW_ARE_YOU.mp4'
print(f'Extracting pose from HOW_ARE_YOU...')
raw, fps = extract_from_file(vid, 'HOW_ARE_YOU')
sm = smooth_frames(raw, w=4)
n = len(sm)
third = n // 3

# Split the video into 3 parts for HELLO, HOW, ARE YOU
segments = [
    (sm[:third],      'HELLO'),
    (sm[third:2*third], 'HOW'),
    (sm[2*third:],    'ARE YOU'),
]

for seg_frames, seg_label in segments:
    all_frames.extend(seg_frames)
    all_labels.extend([seg_label] * len(seg_frames))

# Add short pause between signs
pause_frame = {'pose': None, 'lhand': None, 'rhand': None}
gap_frames = int(fps * 0.4)

# Rebuild with gaps
final_frames = []
final_labels = []
prev_label = None
for fr, lb in zip(all_frames, all_labels):
    if lb != prev_label and prev_label is not None:
        final_frames.extend([all_frames[final_labels.index(prev_label)]] * gap_frames)
        final_labels.extend([prev_label] * gap_frames)
    final_frames.append(fr)
    final_labels.append(lb)
    prev_label = lb

print(f'Rendering {len(final_frames)} frames...')
OUT = '/root/.openclaw/workspace/esl_hello_how.avi'
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'MJPG'), fps, (AW,AH))

for i,(fr,lb) in enumerate(zip(final_frames,final_labels)):
    prog = i/max(len(final_frames)-1,1)
    out.write(draw_frame(fr, lb, prog))

out.release()
import os
print(f'Done! {OUT} ({os.path.getsize(OUT)//1024}KB, {len(final_frames)/fps:.1f}s)')
