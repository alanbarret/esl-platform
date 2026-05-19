"""
Arab Avatar Renderer
Takes mocap JSON landmark data and renders an Arab figure video.
Skeleton videos remain unchanged — this is a separate render pipeline.
"""
import cv2, numpy as np, json, os, subprocess, hashlib
from pathlib import Path
import math

MOCAP_DIR   = Path(__file__).parent.parent / "data" / "processed" / "mocap"
AVATAR_DIR  = Path(__file__).parent.parent / "data" / "avatar_videos"
AVATAR_DIR.mkdir(exist_ok=True)

W, H = 640, 480

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

def lerp(a, b, t): return int(a + (b-a)*t)
def lp(a, b, t): return (lerp(a[0],b[0],t), lerp(a[1],b[1],t))
def to_px(lm): return int(lm[0]*W), int(lm[1]*H)

def draw_arab(img, pose, rhand, lhand):
    if not pose: return

    def lm(i): return to_px(pose[i])
    def vis(i): return pose[i][3] if len(pose[i])>3 else 1.0

    L_SH=11; R_SH=12; L_EL=13; R_EL=14; L_WR=15; R_WR=16
    L_HP=23; R_HP=24; NOSE=0

    ls=lm(L_SH); rs=lm(R_SH); le=lm(L_EL); re=lm(R_EL)
    lw=lm(L_WR); rw=lm(R_WR); lhp=lm(L_HP); rhp=lm(R_HP)
    nose=lm(NOSE)

    sh_mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hp_mid = ((lhp[0]+rhp[0])//2, (lhp[1]+rhp[1])//2)
    sh_w   = max(40, abs(rs[0]-ls[0]))
    arm_t  = max(8, sh_w//8)

    C_ROBE  = (235,233,230); C_ROBED = (198,196,193)
    C_SKIN  = (185,148,115); C_SKIND = (148,116,84)
    C_SLEEVE= (68,108,162)
    C_GHUTRA= (242,240,237)
    C_AGAL  = (18,14,8)

    # Shadow
    cv2.ellipse(img,(sh_mid[0],H-12),(sh_w//2+18,8),0,0,360,(0,0,0),-1,cv2.LINE_AA)

    # Robe body
    rp=np.array([(ls[0]-arm_t,ls[1]),(rs[0]+arm_t,rs[1]),
                  (hp_mid[0]+sh_w//2+12,H),(hp_mid[0]-sh_w//2-12,H)],np.int32)
    cv2.fillPoly(img,[rp],C_ROBE,cv2.LINE_AA)
    cv2.line(img,sh_mid,(sh_mid[0],H),C_ROBED,2,cv2.LINE_AA)
    collar=np.array([(sh_mid[0]-sh_w//6,sh_mid[1]),(sh_mid[0],sh_mid[1]-sh_w//4),(sh_mid[0]+sh_w//6,sh_mid[1])],np.int32)
    cv2.polylines(img,[collar],False,C_ROBED,2,cv2.LINE_AA)
    cv2.line(img,ls,rs,C_ROBE,arm_t*2+2,cv2.LINE_AA)

    # Arms
    for sh,el,wr,v1,v2,v3 in [(ls,le,lw,vis(L_SH),vis(L_EL),vis(L_WR)),
                                (rs,re,rw,vis(R_SH),vis(R_EL),vis(R_WR))]:
        if v1>0.2 and v2>0.2:
            cv2.line(img,sh,el,C_ROBED,arm_t*2+2,cv2.LINE_AA)
            cv2.line(img,sh,el,C_ROBE,arm_t*2,cv2.LINE_AA)
        if v2>0.2 and v3>0.2:
            cv2.line(img,el,wr,C_ROBED,arm_t*2,cv2.LINE_AA)
            cv2.line(img,el,wr,C_ROBE,arm_t*2-2,cv2.LINE_AA)
            cuff=lp(el,wr,0.82)
            cv2.line(img,cuff,wr,C_SLEEVE,arm_t*2,cv2.LINE_AA)

    # Hands
    for hand,wr_pt in [(rhand,rw),(lhand,lw)]:
        if not hand: continue
        hp=lambda i: to_px(hand[i])
        palm=np.array([hp(0),hp(5),hp(9),hp(13),hp(17)],np.int32)
        cv2.fillConvexPoly(img,palm,C_SKIN,cv2.LINE_AA)
        for a,b in HAND_CONN:
            if a<len(hand) and b<len(hand):
                cv2.line(img,hp(a),hp(b),C_SKIND,max(2,arm_t-4),cv2.LINE_AA)
                cv2.line(img,hp(a),hp(b),C_SKIN,max(1,arm_t-5),cv2.LINE_AA)
        for i in range(len(hand)):
            cv2.circle(img,hp(i),max(2,arm_t-6),C_SKIN,-1,cv2.LINE_AA)

    # Neck
    neck_top=(sh_mid[0], sh_mid[1]-int(abs(nose[1]-sh_mid[1])*0.35))
    cv2.line(img,sh_mid,neck_top,C_SKIND,arm_t+2,cv2.LINE_AA)
    cv2.line(img,sh_mid,neck_top,C_SKIN,arm_t,cv2.LINE_AA)

    # Head
    hr=max(18, sh_w//3)
    hx,hy = nose[0], nose[1]-int(hr*0.3)

    cv2.ellipse(img,(hx,hy),(hr,int(hr*1.15)),0,0,360,C_SKIND,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy),(hr,int(hr*1.15)),0,0,360,C_SKIN,-1,cv2.LINE_AA)

    # Beard
    brd=np.array([(hx-hr+4,hy+int(hr*0.2)),(hx+hr-4,hy+int(hr*0.2)),
                   (hx+hr-8,hy+int(hr*1.1)),(hx,hy+int(hr*1.18)),(hx-hr+8,hy+int(hr*1.1))],np.int32)
    cv2.fillPoly(img,[brd],(145,110,78),cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy+int(hr*0.32)),(int(hr*0.4),int(hr*0.15)),0,0,180,(120,88,55),-1,cv2.LINE_AA)

    # Eyes
    eo=int(hr*0.28)
    for ex in [-eo,eo]:
        cv2.circle(img,(hx+ex,hy-int(hr*0.15)),int(hr*0.13),(35,22,12),-1,cv2.LINE_AA)
        cv2.circle(img,(hx+ex+2,hy-int(hr*0.18)),int(hr*0.05),(210,210,220),-1)
    bw=int(hr*0.15)
    for ex in [-eo,eo]:
        cv2.line(img,(hx+ex-bw,hy-int(hr*0.28)),(hx+ex+bw,hy-int(hr*0.24)),(60,38,18),max(1,hr//14),cv2.LINE_AA)

    # Ghutrah
    gw=int(hr*1.32)
    gp=np.array([(hx-gw,hy-int(hr*0.08)),(hx,hy-int(hr*1.42)),(hx+gw,hy-int(hr*0.08)),
                  (hx+gw-8,hy+int(hr*0.42)),(hx-gw+8,hy+int(hr*0.42))],np.int32)
    cv2.fillPoly(img,[gp],C_GHUTRA,cv2.LINE_AA)
    cv2.polylines(img,[gp],True,C_ROBED,1,cv2.LINE_AA)
    for i in range(3):
        yy=hy-int(hr*0.08)+int(hr*(0.1+i*0.15))
        cv2.line(img,(hx-gw+12,yy),(hx-gw+12+int(gw*0.6),yy),(180,30,30),1,cv2.LINE_AA)

    # Agal
    ay=hy-int(hr*0.92)
    cv2.ellipse(img,(hx,ay),(hr-2,int(hr*0.22)),0,0,360,C_AGAL,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,ay+int(hr*0.12)),(hr-6,int(hr*0.16)),0,0,360,C_AGAL,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,ay),(hr-2,int(hr*0.22)),0,0,360,(45,35,25),2,cv2.LINE_AA)

    # Face redraw on top
    cv2.ellipse(img,(hx,hy),(hr,int(hr*1.15)),0,-20,200,C_SKIN,-1,cv2.LINE_AA)
    for ex in [-eo,eo]:
        cv2.circle(img,(hx+ex,hy-int(hr*0.15)),int(hr*0.13),(35,22,12),-1,cv2.LINE_AA)
        cv2.circle(img,(hx+ex+2,hy-int(hr*0.18)),int(hr*0.05),(210,210,220),-1)
    for ex in [-eo,eo]:
        cv2.line(img,(hx+ex-bw,hy-int(hr*0.28)),(hx+ex+bw,hy-int(hr*0.24)),(60,38,18),max(1,hr//14),cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy+int(hr*0.32)),(int(hr*0.4),int(hr*0.15)),0,0,180,(120,88,55),-1,cv2.LINE_AA)
    cv2.fillPoly(img,[brd],(145,110,78),cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy+int(hr*0.32)),(int(hr*0.4),int(hr*0.15)),0,0,180,(120,88,55),-1,cv2.LINE_AA)


def render_avatar_video(sign: str) -> str | None:
    """Render Arab avatar video for a sign. Returns path or None."""
    out_path = AVATAR_DIR / f"{sign.upper()}.mp4"
    if out_path.exists() and out_path.stat().st_size > 5000:
        return str(out_path)

    mocap = MOCAP_DIR / f"{sign.upper()}.json"
    if not mocap.exists():
        return None

    with open(mocap) as f: data = json.load(f)
    fps = data.get('fps', 25)
    frames = data['frames']

    # Keep only hand-visible frames
    def valid(fd):
        for k in ['rhand','lhand']:
            h = fd.get(k)
            if h and not (abs(h[0][0])<0.001 and abs(h[0][1])<0.001): return True
        return False

    has_hand = [valid(fd) for fd in frames]
    expanded = list(has_hand)
    for i,v in enumerate(has_hand):
        if v:
            for j in range(max(0,i-4), min(len(has_hand),i+5)): expanded[j]=True
    keep = [i for i,v in enumerate(expanded) if v] or list(range(len(frames)))

    tmp = f'/tmp/{sign}_avatar.avi'
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W,H))

    for fi in keep:
        fd = frames[fi]
        img = np.zeros((H,W,3), dtype=np.uint8)
        for y in range(H):
            t=y/H; img[y,:]=(int(10+t*12),int(8+t*10),int(20+t*20))
        draw_arab(img, fd.get('pose'), fd.get('rhand'), fd.get('lhand'))
        out.write(img)
    out.release()

    subprocess.run(['ffmpeg','-y','-i',tmp,'-c:v','libx264','-crf','18',
                    '-preset','fast','-pix_fmt','yuv420p',str(out_path)],
                   capture_output=True, timeout=30)
    if os.path.exists(tmp): os.unlink(tmp)

    if out_path.exists() and out_path.stat().st_size > 5000:
        print(f"[Avatar] {sign} → {out_path.stat().st_size//1024}KB")
        return str(out_path)
    return None


def get_or_render_avatar(sign: str) -> str | None:
    out_path = AVATAR_DIR / f"{sign.upper()}.mp4"
    if out_path.exists() and out_path.stat().st_size > 5000:
        return str(out_path)
    return render_avatar_video(sign)


if __name__ == "__main__":
    signs = sorted(p.stem for p in MOCAP_DIR.glob("*.json"))
    print(f"Rendering {len(signs)} Arab avatar videos...")
    for s in signs:
        render_avatar_video(s)
    print("Done.")
