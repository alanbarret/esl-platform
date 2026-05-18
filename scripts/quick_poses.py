"""Quick pose extraction - uses already-downloaded videos only, fast mode."""
import cv2, numpy as np, json, math, struct
from pathlib import Path
import mediapipe as mp

ROOT = Path('/root/.openclaw/workspace/esl-platform')
MOTION_DB = ROOT / 'data' / 'motion_db'
mp_holistic = mp.solutions.holistic

with open(ROOT / 'frontend/public/avatar/arab-man.glb', 'rb') as f:
    f.read(12); cl=struct.unpack('<I',f.read(4))[0]; f.read(4)
    gltf = json.loads(f.read(cl))
REST = {n['name']: n.get('rotation',[0,0,0,1]) for n in gltf['nodes'] if 'name' in n}

def norm(v): n=math.sqrt(sum(x*x for x in v)+1e-12); return [x/n for x in v]
def mul_q(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return norm([aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
                 aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz])
def inv_q(q): x,y,z,w=q; n2=x*x+y*y+z*z+w*w+1e-12; return [-x/n2,-y/n2,-z/n2,w/n2]
def vec_q(v1,v2):
    v1=np.array(v1,float); v2=np.array(v2,float)
    n1=np.linalg.norm(v1)+1e-12; n2=np.linalg.norm(v2)+1e-12
    v1/=n1; v2/=n2
    c=np.cross(v1,v2); d=float(np.dot(v1,v2))
    w=1+d
    if w<1e-6: return [0,0,1,0]
    return norm([float(c[0]),float(c[1]),float(c[2]),w])

def get_peak_bones(mp4_path):
    cap=cv2.VideoCapture(str(mp4_path))
    W=int(cap.get(3)); H=int(cap.get(4))
    best=None; best_score=float('inf')
    n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample 10 frames from middle 60%
    start=n_frames//5; end=n_frames*4//5
    sample_frames=[start+i*(end-start)//9 for i in range(10)]
    
    with mp_holistic.Holistic(
        static_image_mode=True,  # faster for single frames
        min_detection_confidence=0.5
    ) as ho:
        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret,fr=cap.read()
            if not ret: continue
            res=ho.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks: continue
            lm=res.pose_landmarks.landmark
            score=(lm[16].y+lm[14].y)/2  # lower wrist+elbow = higher signing
            if score<best_score:
                best_score=score; best=lm
    cap.release()
    if best is None: return {}
    
    lm=best
    def p(i): return np.array([lm[i].x*W,(1-lm[i].y)*H,-lm[i].z*W])
    bones={}
    try:
        ls,rs,le,re,lw,rw=p(11),p(12),p(13),p(14),p(15),p(16)
        lh,rh,nose=p(23),p(24),p(0)
        def dq(d,bone):
            n=np.linalg.norm(d)
            if n<1: return [0,0,0,1]
            wq=vec_q([0,-1,0],(d/n).tolist()); rq=REST.get(bone,[0,0,0,1])
            return mul_q(inv_q(rq),wq)
        bones['RightArm']=dq(re-rs,'RightArm')
        bones['RightForeArm']=dq(rw-re,'RightForeArm')
        bones['LeftArm']=dq(le-ls,'LeftArm')
        bones['LeftForeArm']=dq(lw-le,'LeftForeArm')
        bones['Head']=dq(nose-(ls+rs)/2,'Head')
        bones['Spine2']=dq((ls+rs)/2-(lh+rh)/2,'Spine2')
    except: pass
    return bones

def q2euler(q):
    x,y,z,w=q
    try:
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=math.asin(max(-1,min(1,2*(w*y-z*x))))
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        return round(roll,3),round(pitch,3),round(yaw,3)
    except: return 0,0,0

# Map gloss -> video file
GLOSS_TO_VIDEO = {
    'HELLO':     'HOW_ARE_YOU.mp4',
    'HOW':       'HOW_ARE_YOU.mp4',
    'YOU':       'HOW_ARE_YOU.mp4',
    'DOCTOR':    'DOCTOR.mp4',
    'WORK':      'WORK.mp4',
    'FAMILY':    'FAMILY.mp4',
    'SCHOOL':    'SCHOOL.mp4',
    'HELP':      'HELPS.mp4',
    'SLEEP':     'SLEEP.mp4',
    'OPEN':      'OPEN.mp4',
    'OUT':       'OUT.mp4',
    'PLAYS':     'PLAYS.mp4',
    'SELL':      'SELL.mp4',
    'PUSH':      'PUSH.mp4',
    'REMOVE':    'REMOVE.mp4',
    'RELAX':     'RELAX.mp4',
    'RUSH':      'RUSH.mp4',
    'SEW':       'SEW.mp4',
    'SHOUTS':    'SHOUTS.mp4',
    'RECOMMENDED':'RECOMMENDED.mp4',
}

print('Extracting real peak poses...')
sign_poses = {}
for gloss, vid_file in GLOSS_TO_VIDEO.items():
    mp4 = MOTION_DB / vid_file
    if not mp4.exists():
        print(f'  {gloss}: missing {vid_file}')
        continue
    print(f'  {gloss}...', end='', flush=True)
    bones = get_peak_bones(mp4)
    if bones:
        euler = {bone: q2euler(q) for bone,q in bones.items()
                 if max(abs(a) for a in q2euler(q)) > 0.05}
        sign_poses[gloss] = euler
        print(f' {len(euler)} bones')
    else:
        print(' no pose')

# Save
out = ROOT / 'data' / 'processed' / 'sign_poses_real.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(sign_poses, indent=2))
print(f'\nSaved {len(sign_poses)} poses to {out}')
print('\n=== SIGN_POSES dict ===')
print('SIGN_POSES = {')
for g,p in sign_poses.items():
    items = ', '.join(f'"{b}":({v[0]},{v[1]},{v[2]})' for b,v in p.items())
    print(f'    "{g}": {{{items}}},')
print('}')
