"""
Download all UAE sign videos, run MediaPipe hand detection,
clip to frames where hands are visible, save as skeleton video.
Optimised: frame skip, parallel download, ultrafast encoding.
"""
import json, os, subprocess, sys, time, cv2, numpy as np
from pathlib import Path
import mediapipe as mp

BASE       = Path('/root/.openclaw/workspace/esl-platform')
SIGNS_JSON = BASE / 'data/raw/uae_signs_full.json'
SKEL_DIR   = BASE / 'data/skeleton_videos'
TMP_DIR    = Path('/tmp/esl_dl')

SKEL_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,25),(24,26),(25,27),(26,28)]
HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
W, H = 640, 360

def download(url, dst):
    if dst.exists() and dst.stat().st_size > 10000: return True
    try:
        r = subprocess.run(['curl','-sL','--max-time','15','-o',str(dst),url],
                           capture_output=True, timeout=20)
        return r.returncode==0 and dst.exists() and dst.stat().st_size>10000
    except: return False

def process_video(video_path, sign_name, holistic):
    out_path = SKEL_DIR / f"{sign_name}.mp4"
    if out_path.exists() and out_path.stat().st_size > 5000: return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5: cap.release(); return False

    frames_data = []; has_hand = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (320, 180))
        res = holistic.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        fd = {}
        if res.pose_landmarks:
            fd['pose'] = [[lm.x,lm.y,lm.z,lm.visibility] for lm in res.pose_landmarks.landmark]
        if res.right_hand_landmarks:
            fd['rhand'] = [[lm.x,lm.y,lm.z] for lm in res.right_hand_landmarks.landmark]
        if res.left_hand_landmarks:
            fd['lhand'] = [[lm.x,lm.y,lm.z] for lm in res.left_hand_landmarks.landmark]
        frames_data.append(fd)
        rh=fd.get('rhand'); lh=fd.get('lhand')
        def valid(h): return bool(h) and not(abs(h[0][0])<0.001 and abs(h[0][1])<0.001)
        has_hand.append(valid(rh) or valid(lh))
    cap.release()

    if not frames_data: return False

    # Find hand-visible frames ±3 buffer
    expanded = list(has_hand)
    for i,v in enumerate(has_hand):
        if v:
            for j in range(max(0,i-3), min(len(has_hand),i+4)): expanded[j]=True
    keep = [i for i,v in enumerate(expanded) if v] or list(range(len(frames_data)))
    if not keep: return False

    # Free memory: only keep needed frames, drop the rest
    frames_data = [frames_data[i] for i in keep]
    keep = list(range(len(frames_data)))
    has_hand = None  # free memory
    expanded = None

    # Render skeleton
    tmp = TMP_DIR / f"{sign_name}.avi"
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W,H))
    def to_px(lm): return int(lm[0]*W), int(lm[1]*H)

    for idx in keep:
        fd = frames_data[idx]
        img = np.zeros((H,W,3), dtype=np.uint8)
        for y in range(H):
            t=y/H; img[y,:]=(int(8+t*18),int(8+t*14),int(18+t*32))
        pose=fd.get('pose')
        if pose:
            for a,b in POSE_CONN:
                if a<len(pose) and b<len(pose) and pose[a][3]>0.3 and pose[b][3]>0.3:
                    cv2.line(img,to_px(pose[a]),to_px(pose[b]),(80,60,160),2,cv2.LINE_AA)
                    cv2.line(img,to_px(pose[a]),to_px(pose[b]),(124,58,237),1,cv2.LINE_AA)
            for lm in pose:
                if len(lm)>3 and lm[3]>0.3: cv2.circle(img,to_px(lm),3,(160,130,255),-1,cv2.LINE_AA)
        for hand,c1,c2 in [(fd.get('rhand'),(160,100,40),(255,165,75)),(fd.get('lhand'),(80,160,40),(168,255,75))]:
            if hand:
                for a,b in HAND_CONN:
                    if a<len(hand) and b<len(hand):
                        cv2.line(img,to_px(hand[a]),to_px(hand[b]),c1,2,cv2.LINE_AA)
                        cv2.line(img,to_px(hand[a]),to_px(hand[b]),c2,1,cv2.LINE_AA)
                for lm in hand: cv2.circle(img,to_px(lm),3,c2,-1,cv2.LINE_AA)
        out.write(img)
    out.release()

    subprocess.run(['ffmpeg','-y','-i',str(tmp),'-c:v','libx264','-crf','23',
                    '-preset','ultrafast','-pix_fmt','yuv420p',str(out_path)],
                   capture_output=True, timeout=10)
    if tmp.exists(): os.unlink(tmp)
    return out_path.exists() and out_path.stat().st_size>5000

# ── Main ──────────────────────────────────────────────────────────────────────
with open(SIGNS_JSON) as f: signs = json.load(f)
already = {p.stem.upper() for p in SKEL_DIR.glob('*.mp4')}
todo = [s for s in signs if s.get('video_url') and
        s.get('english','').upper().replace(' ','_').replace('/','_') not in already]

print(f"Total:{len(signs)} Done:{len(already)} Todo:{len(todo)}", flush=True)

ok=fail=0; t0=time.time()
mp_hol = mp.solutions.holistic

with mp_hol.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4,
                     model_complexity=0, static_image_mode=False) as holistic:
    for i, sign in enumerate(todo):
        name = sign.get('english','').upper().replace(' ','_').replace('/','_')
        url  = sign.get('video_url','')
        if not name or not url: continue

        t1 = time.time()
        tmp_vid = TMP_DIR / f"{name}.mp4"

        if not download(url, tmp_vid):
            fail+=1; continue

        if process_video(tmp_vid, name, holistic): ok+=1
        else: fail+=1

        if tmp_vid.exists(): os.unlink(tmp_vid)

        if (i+1) % 20 == 0:
            elapsed = time.time()-t0
            rate = (i+1)/elapsed
            eta = (len(todo)-(i+1))/rate/60
            print(f"[{i+1}/{len(todo)}] {name} | {time.time()-t1:.1f}s | "
                  f"{rate:.2f}/s | ETA {eta:.0f}min | total={len(already)+ok}", flush=True)

print(f"\nDone: {ok} OK {fail} fail | Total: {len(list(SKEL_DIR.glob('*.mp4')))}")
