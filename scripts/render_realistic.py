"""
Realistic Arab Man Avatar Renderer
Uses OpenCV to draw a stylized humanoid avatar that closely resembles
the Arab Man GLB model — with proper proportions, shading, kandura.
Animated using real pose data from UAE Sign Language video.
"""
import cv2, numpy as np, json, math, struct
from pathlib import Path
import mediapipe as mp

ROOT = Path('/root/.openclaw/workspace/esl-platform')
MOTION_DB = ROOT / 'data' / 'motion_db'
mp_holistic = mp.solutions.holistic

# ── Load GLB rest quaternions ─────────────────────────────────────────────────
with open(ROOT / 'frontend/public/avatar/arab-man.glb', 'rb') as f:
    f.read(12); cl=struct.unpack('<I',f.read(4))[0]; f.read(4)
    gltf = json.loads(f.read(cl))
REST = {n['name']:n.get('rotation',[0,0,0,1]) for n in gltf['nodes'] if 'name' in n}

def norm(v): n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v] if n>1e-6 else v
def mul_q(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz]
def inv_q(q): x,y,z,w=q; n2=sum(c*c for c in q); return [-x/n2,-y/n2,-z/n2,w/n2]
def vec_q(v1,v2):
    v1=np.array(v1)/np.linalg.norm(v1); v2=np.array(v2)/np.linalg.norm(v2)
    c=np.cross(v1,v2); d=float(np.dot(v1,v2)); w=1+d
    if w<1e-6: return [0,0,1,0]
    return norm([float(c[0]),float(c[1]),float(c[2]),w])

# ── Extract pose from HOW_ARE_YOU video ──────────────────────────────────────
print('Extracting real pose...')
vid = MOTION_DB / 'HOW_ARE_YOU.mp4'
cap = cv2.VideoCapture(str(vid))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

raw_frames = []
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as ho:
    while cap.isOpened():
        ret,fr=cap.read()
        if not ret: break
        raw_frames.append(ho.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
cap.release()
print(f'  {len(raw_frames)} frames at {fps:.0f}fps')

def extract_bones(res):
    bones={}
    if not res.pose_landmarks: return bones
    lm=res.pose_landmarks.landmark
    def p(i): return np.array([lm[i].x*VW,(1-lm[i].y)*VH,-lm[i].z*VW])
    try:
        ls=p(11);rs=p(12);le=p(13);re=p(14);lw=p(15);rw=p(16)
        lh=p(23);rh=p(24);nose=p(0)
        def dq(d,bone):
            d=d/np.linalg.norm(d); wq=vec_q([0,-1,0],d.tolist()); rq=REST.get(bone,[0,0,0,1])
            return mul_q(inv_q(rq),wq)
        bones['RightArm']=dq(re-rs,'RightArm'); bones['RightForeArm']=dq(rw-re,'RightForeArm')
        bones['LeftArm']=dq(le-ls,'LeftArm'); bones['LeftForeArm']=dq(lw-le,'LeftForeArm')
        bones['Head']=dq(nose-(ls+rs)/2,'Head')
        bones['Spine2']=dq((ls+rs)/2-(lh+rh)/2,'Spine2')
    except: pass
    return bones

all_bones=[extract_bones(r) for r in raw_frames]
# Smooth
W5=5
smoothed=[]
for i in range(len(all_bones)):
    s=max(0,i-W5//2); e=min(len(all_bones),i+W5//2+1)
    wnd=[all_bones[j] for j in range(s,e)]
    sb={}
    for bone in set().union(*[set(f.keys()) for f in wnd]):
        v=[f[bone] for f in wnd if bone in f]
        if v: sb[bone]=norm([sum(q[j] for q in v)/len(v) for j in range(4)])
    smoothed.append(sb)

# ── Rendering ─────────────────────────────────────────────────────────────────
AW,AH=540,720
font=cv2.FONT_HERSHEY_SIMPLEX

def q2angles(q):
    x,y,z,w=q
    try:
        pitch=math.asin(max(-1,min(1,2*(w*y-z*x))))
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        return roll,pitch,yaw
    except: return 0,0,0

def scale(v,h): return v*h*0.82

def get_joints(bones, h=AH, cx=None):
    if cx is None: cx=AW//2
    J={}
    # Root
    hip_y=h*0.56; J['hips']=np.array([cx,hip_y])
    # Spine
    sp2=J['hips']+np.array([0,-scale(0.30,h)]); J['spine2']=sp2
    nk=sp2+np.array([0,-scale(0.09,h)]); J['neck']=nk
    hd=nk+np.array([0,-scale(0.06,h)]); J['head']=hd

    def arm_chain(side, sh_off):
        sh=sp2+np.array([sh_off,-scale(0.05,h)]); J[f'{side}_sh']=sh
        q_a=bones.get('RightArm' if side=='r' else 'LeftArm',[0,0,0,1])
        roll_a,pitch_a,yaw_a=q2angles(q_a)
        sign=-1 if side=='r' else 1
        # Upper arm
        ua_dx=sign*scale(0.13,h)*math.cos(yaw_a*0.6)+scale(0.02,h)*math.sin(yaw_a*0.3)
        ua_dy=scale(0.14,h)*(0.25+pitch_a*0.55)
        el=sh+np.array([ua_dx,ua_dy]); J[f'{side}_el']=el
        # Forearm
        q_f=bones.get('RightForeArm' if side=='r' else 'LeftForeArm',[0,0,0,1])
        roll_f,pitch_f,yaw_f=q2angles(q_f)
        fa_dx=sign*scale(0.11,h)*math.cos((yaw_a+yaw_f)*0.5)+scale(0.015,h)*math.sin(yaw_f*0.4)
        fa_dy=scale(0.12,h)*(0.2+pitch_f*0.45)
        wr=el+np.array([fa_dx,fa_dy]); J[f'{side}_wr']=wr
        hnd=wr+np.array([sign*scale(0.04,h)*math.cos(yaw_a*0.2),scale(0.04,h)]); J[f'{side}_hnd']=hnd

    arm_chain('r', scale(0.12,h))
    arm_chain('l',-scale(0.12,h))

    # Legs
    for side,sign in [('r',1),('l',-1)]:
        hp=J['hips']+np.array([sign*scale(0.07,h),0]); J[f'{side}_hp']=hp
        kn=hp+np.array([sign*scale(0.01,h),scale(0.21,h)]); J[f'{side}_kn']=kn
        ak=kn+np.array([0,scale(0.19,h)]); J[f'{side}_ak']=ak
        toe=ak+np.array([sign*scale(0.08,h),scale(0.01,h)]); J[f'{side}_toe']=toe
    return J

def ipt(p): return (int(p[0]),int(p[1]))

def draw_frame(bones, label, prog, fi):
    img=np.zeros((AH,AW,3),dtype=np.uint8)

    # Background gradient
    for y in range(AH):
        t=y/AH
        img[y,:]=tuple(int(a+(b-a)*t) for a,b in [(8,18),(8,14),(18,32)])

    # Subtle floor reflection
    cv2.ellipse(img,(AW//2,AH-45),(90,15),0,0,360,(20,20,35),-1,cv2.LINE_AA)
    cv2.ellipse(img,(AW//2,AH-45),(90,15),0,0,360,(30,28,48),1,cv2.LINE_AA)

    J=get_joints(bones)

    # ── Draw back layers first ────────────────────────────────────────────────
    # Left arm (back)
    if 'l_sh' in J and 'l_el' in J:
        cv2.line(img,ipt(J['l_sh']),ipt(J['l_el']),(55,95,150),13,cv2.LINE_AA)
        cv2.line(img,ipt(J['l_sh']),ipt(J['l_el']),(70,115,175),9,cv2.LINE_AA)
    if 'l_el' in J and 'l_wr' in J:
        cv2.line(img,ipt(J['l_el']),ipt(J['l_wr']),(50,88,140),11,cv2.LINE_AA)
        cv2.line(img,ipt(J['l_el']),ipt(J['l_wr']),(65,108,163),7,cv2.LINE_AA)

    # Legs
    LEG_COLS=[(55,40,100),(48,35,90),(40,30,75)]
    for side in ['l','r']:
        if f'{side}_hp' in J and f'{side}_kn' in J:
            cv2.line(img,ipt(J[f'{side}_hp']),ipt(J[f'{side}_kn']),(55,40,100),16,cv2.LINE_AA)
            cv2.line(img,ipt(J[f'{side}_hp']),ipt(J[f'{side}_kn']),(75,58,125),11,cv2.LINE_AA)
        if f'{side}_kn' in J and f'{side}_ak' in J:
            cv2.line(img,ipt(J[f'{side}_kn']),ipt(J[f'{side}_ak']),(48,35,90),14,cv2.LINE_AA)
            cv2.line(img,ipt(J[f'{side}_kn']),ipt(J[f'{side}_ak']),(65,50,110),10,cv2.LINE_AA)
        if f'{side}_ak' in J and f'{side}_toe' in J:
            cv2.line(img,ipt(J[f'{side}_ak']),ipt(J[f'{side}_toe']),(40,30,70),12,cv2.LINE_AA)
            cv2.line(img,ipt(J[f'{side}_ak']),ipt(J[f'{side}_toe']),(55,42,88),8,cv2.LINE_AA)

    # ── Kandura (white robe - torso) ──────────────────────────────────────────
    if 'hips' in J and 'spine2' in J:
        hx,hy=ipt(J['hips']); sx,sy=ipt(J['spine2'])
        # Wide robe shape
        pts=np.array([[hx-55,hy+12],[hx+55,hy+12],[sx+38,sy-8],[sx,sy-15],[sx-38,sy-8]],np.int32)
        cv2.fillPoly(img,[pts],(230,228,225),cv2.LINE_AA)
        cv2.polylines(img,[pts],True,(200,198,195),2,cv2.LINE_AA)
        # Robe detail lines
        cv2.line(img,(sx,sy-15),(hx,hy+8),(210,208,205),2,cv2.LINE_AA)

    # Shoulder area (kandura top)
    if 'l_sh' in J and 'r_sh' in J:
        ls2,rs2=J['l_sh'],J['r_sh']; nk2=J.get('neck',J['spine2'])
        pts=np.array([ipt(ls2),ipt(rs2),ipt(nk2+np.array([0,12]))],np.int32)
        cv2.fillPoly(img,[pts],(232,230,227),cv2.LINE_AA)

    # ── Neck ──────────────────────────────────────────────────────────────────
    if 'neck' in J and 'head' in J:
        cv2.line(img,ipt(J['neck']),ipt(J['head']),(185,148,112),14,cv2.LINE_AA)
        cv2.line(img,ipt(J['neck']),ipt(J['head']),(195,158,122),9,cv2.LINE_AA)

    # ── Head ──────────────────────────────────────────────────────────────────
    if 'head' in J:
        hx2,hy2=ipt(J['head']); hr=int(AH*0.075)
        # Head shape (oval)
        cv2.ellipse(img,(hx2,hy2),(hr,int(hr*1.15)),0,0,360,(32,55,75),-1,cv2.LINE_AA)
        cv2.ellipse(img,(hx2,hy2),(hr,int(hr*1.15)),0,0,360,(175,138,105),3,cv2.LINE_AA)
        cv2.ellipse(img,(hx2,hy2),(hr,int(hr*1.15)),0,0,360,(185,148,115),-1,cv2.LINE_AA)
        # Face shading
        cv2.ellipse(img,(hx2+int(hr*0.2),hy2),(int(hr*0.7),int(hr*0.9)),0,0,360,(175,138,105),-1,cv2.LINE_AA)
        # Eyes
        ex=int(hr*0.28); ey=int(hr*0.18); er=int(hr*0.11)
        cv2.circle(img,(hx2-ex,hy2-ey),er,(45,32,22),-1,cv2.LINE_AA)
        cv2.circle(img,(hx2+ex,hy2-ey),er,(45,32,22),-1,cv2.LINE_AA)
        cv2.circle(img,(hx2-ex+2,hy2-ey-2),int(er*0.4),(200,200,210),-1)
        cv2.circle(img,(hx2+ex+2,hy2-ey-2),int(er*0.4),(200,200,210),-1)
        # Eyebrows
        cv2.line(img,(hx2-ex-er,hy2-ey-er-3),(hx2-ex+er,hy2-ey-er),(50,35,25),3,cv2.LINE_AA)
        cv2.line(img,(hx2+ex-er,hy2-ey-er),(hx2+ex+er,hy2-ey-er-3),(50,35,25),3,cv2.LINE_AA)
        # Nose
        cv2.ellipse(img,(hx2,hy2+int(hr*0.1)),(int(hr*0.12),int(hr*0.18)),0,0,360,(165,128,95),-1,cv2.LINE_AA)
        # Beard (subtle)
        cv2.ellipse(img,(hx2,hy2+int(hr*0.38)),(int(hr*0.45),int(hr*0.25)),0,0,180,(155,118,85),-1,cv2.LINE_AA)
        cv2.ellipse(img,(hx2,hy2+int(hr*0.38)),(int(hr*0.45),int(hr*0.25)),0,0,180,(40,30,25),2,cv2.LINE_AA)
        # Ghutrah (headdress) - white cloth
        pts_g=np.array([[hx2-hr-8,hy2-int(hr*0.2)],[hx2,hy2-int(hr*1.35)],
                         [hx2+hr+8,hy2-int(hr*0.2)],[hx2+hr-5,hy2+int(hr*0.4)],
                         [hx2-hr+5,hy2+int(hr*0.4)]],np.int32)
        cv2.fillPoly(img,[pts_g],(235,232,228),cv2.LINE_AA)
        cv2.polylines(img,[pts_g],True,(210,207,203),2,cv2.LINE_AA)
        # Agal (black rope on top of headdress)
        cv2.ellipse(img,(hx2,hy2-int(hr*0.85)),(hr-4,int(hr*0.22)),0,0,360,(20,18,16),-1,cv2.LINE_AA)
        cv2.ellipse(img,(hx2,hy2-int(hr*0.85)),(hr-4,int(hr*0.22)),0,0,360,(40,36,32),2,cv2.LINE_AA)

    # ── Right arm (front) ─────────────────────────────────────────────────────
    if 'r_sh' in J and 'r_el' in J:
        cv2.line(img,ipt(J['r_sh']),ipt(J['r_el']),(55,95,150),13,cv2.LINE_AA)
        cv2.line(img,ipt(J['r_sh']),ipt(J['r_el']),(75,118,178),9,cv2.LINE_AA)
    if 'r_el' in J and 'r_wr' in J:
        cv2.line(img,ipt(J['r_el']),ipt(J['r_wr']),(50,88,140),11,cv2.LINE_AA)
        cv2.line(img,ipt(J['r_el']),ipt(J['r_wr']),(68,110,165),7,cv2.LINE_AA)
    if 'r_wr' in J and 'r_hnd' in J:
        cv2.line(img,ipt(J['r_wr']),ipt(J['r_hnd']),(185,148,112),10,cv2.LINE_AA)
        cv2.ellipse(img,ipt(J['r_hnd']),(10,13),0,0,360,(185,148,112),-1,cv2.LINE_AA)
    if 'l_wr' in J and 'l_hnd' in J:
        cv2.line(img,ipt(J['l_wr']),ipt(J['l_hnd']),(185,148,112),10,cv2.LINE_AA)
        cv2.ellipse(img,ipt(J['l_hnd']),(10,13),0,0,360,(185,148,112),-1,cv2.LINE_AA)

    # Joint highlights
    for key,col,r in [('r_el',(90,140,200),7),('l_el',(90,140,200),7),
                       ('r_sh',(100,145,195),8),('l_sh',(100,145,195),8)]:
        if key in J: cv2.circle(img,ipt(J[key]),r,col,-1,cv2.LINE_AA)

    # ── Label + progress ──────────────────────────────────────────────────────
    # Gloss label with background
    ts=cv2.getTextSize(label,font,1.1,3)[0]
    lx=(AW-ts[0])//2; ly=AH-28
    cv2.rectangle(img,(lx-10,ly-ts[1]-8),(lx+ts[0]+10,ly+8),(10,10,20),-1)
    cv2.putText(img,label,(lx+1,ly),font,1.1,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(img,label,(lx,ly-1),font,1.1,(168,255,75),3,cv2.LINE_AA)

    bw=int(AW*0.8); bx=int(AW*0.1)
    cv2.rectangle(img,(bx,AH-12),(bx+bw,AH-5),(20,18,35),-1)
    cv2.rectangle(img,(bx,AH-12),(bx+int(bw*prog),AH-5),(124,58,237),-1)

    return img

# ── Render video ──────────────────────────────────────────────────────────────
print(f'Rendering {len(smoothed)} frames...')
OUT='/root/.openclaw/workspace/esl_hello_realistic.avi'
out=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'MJPG'),fps,(AW,AH))

for i,bones in enumerate(smoothed):
    prog=i/max(len(smoothed)-1,1)
    out.write(draw_frame(bones,'HELLO',prog,i))

out.release()
import os
print(f'Done! {OUT} ({os.path.getsize(OUT)//1024}KB, {len(smoothed)//fps:.0f}s)')
