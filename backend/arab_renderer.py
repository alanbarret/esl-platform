"""
Realistic Arab Avatar Renderer
Renders a photorealistic-style Arab man (waist-up) driven by MediaPipe landmarks.
Features: proper face anatomy, detailed hands with fingers, kandura, ghutrah.
"""
import cv2, numpy as np, json, os, subprocess, math
from pathlib import Path

MOCAP_DIR  = Path(__file__).parent.parent / "data" / "processed" / "mocap"
AVATAR_DIR = Path(__file__).parent.parent / "data" / "avatar_videos"
AVATAR_DIR.mkdir(exist_ok=True)
(AVATAR_DIR / "stitched").mkdir(exist_ok=True)

W, H = 640, 540

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def px(lm): return int(lm[0]*W), int(lm[1]*H)
def lp(a,b,t): return (int(a[0]+(b[0]-a[0])*t), int(a[1]+(b[1]-a[1])*t))
def dist(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def draw_gradient_ellipse(img, center, axes, angle, color1, color2):
    """Draw an ellipse with a subtle gradient effect."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    for r in range(max(axes), 0, -max(axes)//10 or -1):
        t = 1 - r/max(axes)
        c = tuple(int(color1[i]*(1-t)+color2[i]*t) for i in range(3))
        a = tuple(max(1, int(x*r/max(axes))) for x in axes)
        cv2.ellipse(img, center, a, angle, 0, 360, c, -1)

def draw_realistic_hand(img, hand, color_skin, color_shadow, arm_t):
    """Draw a detailed realistic hand with finger segments."""
    if not hand: return
    def hp(i): return px(hand[i])

    # Finger thickness varies by segment
    seg_thick = [max(3,arm_t-3), max(2,arm_t-5), max(2,arm_t-6)]

    # Palm fill (convex hull of base joints)
    palm_pts = np.array([hp(0), hp(1), hp(5), hp(9), hp(13), hp(17)], np.int32)
    cv2.fillConvexPoly(img, palm_pts, color_skin, cv2.LINE_AA)

    # Draw each finger with tapered segments
    fingers = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
    for chain in fingers:
        pts = [hp(i) for i in chain]
        for seg_idx, (a, b) in enumerate(zip(pts, pts[1:])):
            t_val = min(seg_idx, len(seg_thick)-1)
            thick = seg_thick[t_val]
            # Shadow
            cv2.line(img, (a[0]+1,a[1]+1), (b[0]+1,b[1]+1), color_shadow, thick+2, cv2.LINE_AA)
            # Main
            cv2.line(img, a, b, color_skin, thick, cv2.LINE_AA)
            # Knuckle highlights
            if seg_idx < 2:
                cv2.circle(img, a, max(2, thick//2+1), color_shadow, -1, cv2.LINE_AA)
                cv2.circle(img, a, max(1, thick//2), (min(255,color_skin[0]+20),min(255,color_skin[1]+15),min(255,color_skin[2]+10)), -1, cv2.LINE_AA)

    # Fingertips
    for tip_idx in [4, 8, 12, 16, 20]:
        cv2.circle(img, hp(tip_idx), max(3, arm_t-6), color_shadow, -1, cv2.LINE_AA)
        cv2.circle(img, hp(tip_idx), max(2, arm_t-7), color_skin, -1, cv2.LINE_AA)
        # Nail
        tip = hp(tip_idx)
        nail_r = max(2, arm_t-8)
        cv2.ellipse(img, tip, (nail_r, max(1,nail_r-1)), 0, 0, 360, (210,190,175), -1, cv2.LINE_AA)

def draw_face(img, nose, sh_w):
    """Draw a realistic Arab face. BGR colour format."""
    hr = max(22, sh_w // 2)
    hx, hy = nose[0], nose[1] - int(hr * 0.25)

    # BGR skin tones — warm natural Arab skin
    SK   = (105, 152, 198)   # base skin      RGB≈(198,152,105)
    SKD  = (85,  128, 172)   # shadow skin
    SKL  = (125, 168, 212)   # highlight skin
    BEAR = (38,  55,  85)    # beard dark brown
    BEAR2= (48,  68,  105)   # beard lighter
    LIP  = (62,  82,  148)   # lips
    LIP2 = (70,  92,  158)   # lower lip
    EYE  = (22,  30,  50)    # iris dark brown
    WHT  = (245, 242, 238)   # eye white
    BROW = (25,  35,  58)    # eyebrow

    # ── Neck ──────────────────────────────────────────────────────────────────
    neck_w = max(14, hr//3)
    neck_h = int(hr*1.6)
    neck_pts = np.array([
        (hx-neck_w,     hy+int(hr*0.95)),
        (hx+neck_w,     hy+int(hr*0.95)),
        (hx+neck_w-3,   hy+neck_h),
        (hx-neck_w+3,   hy+neck_h),
    ], np.int32)
    cv2.fillPoly(img, [neck_pts], SKD, cv2.LINE_AA)
    cv2.fillPoly(img, [neck_pts[0:2]], SKL, cv2.LINE_AA)  # front highlight

    # ── Head base ─────────────────────────────────────────────────────────────
    cv2.ellipse(img, (hx,hy), (hr+2, int(hr*1.18)), 0, 0, 360, SKD, -1, cv2.LINE_AA)

    # Face gradient (lighter centre)
    for r in range(hr, 0, -5):
        t = 1.0 - r/hr
        c = (int(SK[0]+t*12), int(SK[1]+t*10), int(SK[2]+t*8))
        cv2.ellipse(img, (hx,hy), (r, int(r*1.18)), 0, 0, 360, c, -1)

    # ── Jaw shadow ────────────────────────────────────────────────────────────
    jaw = np.array([
        (hx-int(hr*0.75), hy+int(hr*0.3)),
        (hx-int(hr*0.65), hy+int(hr*0.82)),
        (hx-int(hr*0.3),  hy+int(hr*1.14)),
        (hx,              hy+int(hr*1.22)),
        (hx+int(hr*0.3),  hy+int(hr*1.14)),
        (hx+int(hr*0.65), hy+int(hr*0.82)),
        (hx+int(hr*0.75), hy+int(hr*0.3)),
    ], np.int32)
    cv2.polylines(img, [jaw], False, SKD, max(1,hr//16), cv2.LINE_AA)

    # ── Beard ─────────────────────────────────────────────────────────────────
    beard_pts = np.array([
        (hx-int(hr*0.65), hy+int(hr*0.25)),
        (hx+int(hr*0.65), hy+int(hr*0.25)),
        (hx+int(hr*0.60), hy+int(hr*0.78)),
        (hx+int(hr*0.30), hy+int(hr*1.15)),
        (hx,              hy+int(hr*1.22)),
        (hx-int(hr*0.30), hy+int(hr*1.15)),
        (hx-int(hr*0.60), hy+int(hr*0.78)),
    ], np.int32)
    cv2.fillPoly(img, [beard_pts], BEAR, cv2.LINE_AA)
    for i in range(5):
        xo = int((i-2)*hr*0.11)
        cv2.line(img,(hx+xo,hy+int(hr*0.28)),(hx+xo,hy+int(hr*1.1)),BEAR2,max(1,hr//22),cv2.LINE_AA)

    # Moustache
    cv2.ellipse(img,(hx-int(hr*0.2),hy+int(hr*0.26)),(int(hr*0.2),int(hr*0.09)),0,0,180,BEAR,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx+int(hr*0.2),hy+int(hr*0.26)),(int(hr*0.2),int(hr*0.09)),0,0,180,BEAR,-1,cv2.LINE_AA)

    # ── Nose ──────────────────────────────────────────────────────────────────
    nose_bridge = [(hx, hy-int(hr*0.2)), (hx-int(hr*0.05), hy+int(hr*0.12))]
    cv2.line(img, nose_bridge[0], nose_bridge[1], SKD, max(1,hr//12), cv2.LINE_AA)
    # Nose tip
    cv2.ellipse(img,(hx,hy+int(hr*0.16)),(int(hr*0.14),int(hr*0.1)),0,0,360,SKD,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,hy+int(hr*0.14)),(int(hr*0.12),int(hr*0.08)),0,0,360,SK,-1,cv2.LINE_AA)
    # Nostrils
    cv2.ellipse(img,(hx-int(hr*0.14),hy+int(hr*0.18)),(int(hr*0.07),int(hr*0.05)),15,0,360,SKD,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx+int(hr*0.14),hy+int(hr*0.18)),(int(hr*0.07),int(hr*0.05)),-15,0,360,SKD,-1,cv2.LINE_AA)

    # ── Eyes ──────────────────────────────────────────────────────────────────
    eo = int(hr*0.30); ey_off = int(hr*0.08)
    for side, ex in [(-1,-eo),(1,eo)]:
        cx, cy = hx+ex, hy-int(hr*0.06)
        # Socket
        cv2.ellipse(img,(cx,cy),(int(hr*0.21),int(hr*0.14)),0,0,360,SKD,-1,cv2.LINE_AA)
        # White
        cv2.ellipse(img,(cx,cy),(int(hr*0.17),int(hr*0.11)),0,0,360,WHT,-1,cv2.LINE_AA)
        # Iris
        cv2.circle(img,(cx+side*2,cy),int(hr*0.09),(55,40,28),-1,cv2.LINE_AA)
        # Pupil
        cv2.circle(img,(cx+side*2,cy),int(hr*0.05),(12,8,6),-1,cv2.LINE_AA)
        # Catchlight
        cv2.circle(img,(cx+side*3,cy-int(hr*0.03)),int(hr*0.02),(255,255,255),-1)
        # Upper lid
        cv2.ellipse(img,(cx,cy),(int(hr*0.17),int(hr*0.11)),0,200,340,EYE,max(1,hr//16),cv2.LINE_AA)
        # Lower lid
        cv2.ellipse(img,(cx,cy),(int(hr*0.17),int(hr*0.11)),0,20,160,SKD,max(1,hr//20),cv2.LINE_AA)

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    bw = int(hr*0.20)
    for ex in [-eo, eo]:
        bx,by = hx+ex, hy-int(hr*0.22)
        pts = np.array([
            (bx-bw,    by+int(hr*0.05)),
            (bx-bw+int(hr*0.07), by-int(hr*0.04)),
            (bx+bw-int(hr*0.05), by-int(hr*0.03)),
            (bx+bw,    by+int(hr*0.05)),
            (bx+bw-int(hr*0.04), by+int(hr*0.09)),
            (bx-bw,    by+int(hr*0.09)),
        ], np.int32)
        cv2.fillPoly(img,[pts],BROW,cv2.LINE_AA)

    # ── Lips ──────────────────────────────────────────────────────────────────
    lx,ly = hx, hy+int(hr*0.38)
    upper_lip = np.array([
        (lx-int(hr*0.28),ly), (lx-int(hr*0.1),ly-int(hr*0.06)),
        (lx,ly-int(hr*0.02)), (lx+int(hr*0.1),ly-int(hr*0.06)),
        (lx+int(hr*0.28),ly), (lx,ly+int(hr*0.04)),
    ],np.int32)
    lower_lip = np.array([
        (lx-int(hr*0.28),ly),(lx,ly+int(hr*0.13)),(lx+int(hr*0.28),ly),
    ],np.int32)
    cv2.fillPoly(img,[upper_lip],LIP,cv2.LINE_AA)
    cv2.fillPoly(img,[lower_lip],LIP2,cv2.LINE_AA)
    cv2.line(img,(lx-int(hr*0.28),ly),(lx+int(hr*0.28),ly),(45,55,110),max(1,hr//18),cv2.LINE_AA)

    # ── Ear ───────────────────────────────────────────────────────────────────
    ex2 = hx + int(hr*0.95)
    ey2 = hy - int(hr*0.05)
    cv2.ellipse(img,(ex2,ey2),(int(hr*0.13),int(hr*0.21)),0,0,360,SKD,-1,cv2.LINE_AA)
    cv2.ellipse(img,(ex2-int(hr*0.04),ey2),(int(hr*0.08),int(hr*0.15)),0,0,360,SK,-1,cv2.LINE_AA)

    # ── Ghutrah (headscarf — sides and top only, face fully open) ─────────────
    gw = int(hr*1.35)
    # Only draw sides and top, not over face
    side_l = np.array([
        (hx-int(hr*0.85), hy-int(hr*0.15)),
        (hx-gw,           hy-int(hr*0.1)),
        (hx-gw+6,         hy+int(hr*0.55)),
        (hx-int(hr*0.7),  hy+int(hr*0.55)),
    ], np.int32)
    side_r = np.array([
        (hx+int(hr*0.85), hy-int(hr*0.15)),
        (hx+gw,           hy-int(hr*0.1)),
        (hx+gw-6,         hy+int(hr*0.55)),
        (hx+int(hr*0.7),  hy+int(hr*0.55)),
    ], np.int32)
    top = np.array([
        (hx-int(hr*0.85), hy-int(hr*0.15)),
        (hx-gw,           hy-int(hr*0.1)),
        (hx,              hy-int(hr*1.5)),
        (hx+gw,           hy-int(hr*0.1)),
        (hx+int(hr*0.85), hy-int(hr*0.15)),
    ], np.int32)
    ghutra_color = (245,243,240)
    cv2.fillPoly(img,[top],    ghutra_color, cv2.LINE_AA)
    cv2.fillPoly(img,[side_l], ghutra_color, cv2.LINE_AA)
    cv2.fillPoly(img,[side_r], ghutra_color, cv2.LINE_AA)
    # Red pattern on sides
    for i in range(3):
        yy = hy-int(hr*0.1)+int(i*hr*0.15)
        cv2.line(img,(hx-gw+10,yy),(hx-gw+10+int(gw*0.4),yy),(40,40,190),2,cv2.LINE_AA)

    # ── Agal ──────────────────────────────────────────────────────────────────
    ay = hy - int(hr*0.95)
    cv2.ellipse(img,(hx,ay),(hr+2,int(hr*0.23)),0,0,360,(12,10,6),-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,ay+int(hr*0.15)),(hr-2,int(hr*0.17)),0,0,360,(12,10,6),-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx,ay),(hr+2,int(hr*0.23)),0,0,360,(60,130,180),2,cv2.LINE_AA)  # gold trim BGR

    # ── Final face redraw on top of everything ────────────────────────────────
    for r in range(hr, 0, -5):
        t = 1.0 - r/hr
        c = (int(SK[0]+t*12), int(SK[1]+t*10), int(SK[2]+t*8))
        cv2.ellipse(img,(hx,hy),(r,int(r*1.18)),0,-25,205,c,-1)
    cv2.fillPoly(img,[beard_pts],BEAR,cv2.LINE_AA)
    cv2.ellipse(img,(hx-int(hr*0.2),hy+int(hr*0.26)),(int(hr*0.2),int(hr*0.09)),0,0,180,BEAR,-1)
    cv2.ellipse(img,(hx+int(hr*0.2),hy+int(hr*0.26)),(int(hr*0.2),int(hr*0.09)),0,0,180,BEAR,-1)
    cv2.ellipse(img,(hx,hy+int(hr*0.16)),(int(hr*0.14),int(hr*0.1)),0,0,360,SKD,-1)
    cv2.ellipse(img,(hx,hy+int(hr*0.14)),(int(hr*0.12),int(hr*0.08)),0,0,360,SK,-1)
    cv2.ellipse(img,(hx-int(hr*0.14),hy+int(hr*0.18)),(int(hr*0.07),int(hr*0.05)),15,0,360,SKD,-1)
    cv2.ellipse(img,(hx+int(hr*0.14),hy+int(hr*0.18)),(int(hr*0.07),int(hr*0.05)),-15,0,360,SKD,-1)
    for side,ex in [(-1,-eo),(1,eo)]:
        cx,cy=hx+ex,hy-int(hr*0.06)
        cv2.ellipse(img,(cx,cy),(int(hr*0.17),int(hr*0.11)),0,0,360,WHT,-1)
        cv2.circle(img,(cx+side*2,cy),int(hr*0.09),(55,40,28),-1)
        cv2.circle(img,(cx+side*2,cy),int(hr*0.05),(12,8,6),-1)
        cv2.circle(img,(cx+side*3,cy-int(hr*0.03)),int(hr*0.02),(255,255,255),-1)
        cv2.ellipse(img,(cx,cy),(int(hr*0.17),int(hr*0.11)),0,200,340,EYE,max(1,hr//16))
    for ex in [-eo,eo]:
        bx,by=hx+ex,hy-int(hr*0.22)
        pts=np.array([(bx-bw,by+int(hr*0.05)),(bx-bw+int(hr*0.07),by-int(hr*0.04)),
                      (bx+bw-int(hr*0.05),by-int(hr*0.03)),(bx+bw,by+int(hr*0.05)),
                      (bx+bw-int(hr*0.04),by+int(hr*0.09)),(bx-bw,by+int(hr*0.09))],np.int32)
        cv2.fillPoly(img,[pts],BROW)
    cv2.fillPoly(img,[upper_lip],LIP)
    cv2.fillPoly(img,[lower_lip],LIP2)
    cv2.line(img,(lx-int(hr*0.28),ly),(lx+int(hr*0.28),ly),(45,55,110),max(1,hr//18))


def draw_avatar(img, pose, rhand, lhand):
    if not pose: return

    def lm(i): return px(pose[i])
    def vis(i): return pose[i][3] if len(pose[i])>3 else 1.0

    L_SH=11; R_SH=12; L_EL=13; R_EL=14; L_WR=15; R_WR=16
    L_HP=23; R_HP=24; NOSE=0

    ls_raw=lm(L_SH); rs_raw=lm(R_SH); le_raw=lm(L_EL); re_raw=lm(R_EL)
    lw_raw=lm(L_WR); rw_raw=lm(R_WR); lhp=lm(L_HP); rhp=lm(R_HP); nose=lm(NOSE)

    # MediaPipe is mirrored — sort by screen X so left=smaller X, right=larger X
    if ls_raw[0] < rs_raw[0]:
        ls, rs = ls_raw, rs_raw
        le, re = le_raw, re_raw
        lw, rw = lw_raw, rw_raw
        lhand_use, rhand_use = lhand, rhand
    else:
        ls, rs = rs_raw, ls_raw
        le, re = re_raw, le_raw
        lw, rw = rw_raw, lw_raw
        lhand_use, rhand_use = rhand, lhand

    sh_mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hp_mid = ((lhp[0]+rhp[0])//2, (lhp[1]+rhp[1])//2)
    sh_w   = max(50, abs(rs[0]-ls[0]))
    arm_t  = max(9, sh_w//7)

    # Colours
    C_ROBE   = (238,236,233); C_ROBED  = (205,202,198); C_ROBES  = (180,178,175)
    # BGR: (B,G,R) — natural warm Arab skin tone
    C_SKIN   = (125,170,210); C_SKIND  = (98,142,182); C_SKINS  = (75,115,158)
    C_SLEEVE = (65,105,158);  C_SLVD   = (48,82,128)

    # ── Floor shadow ──────────────────────────────────────────────────────────
    cv2.ellipse(img, (sh_mid[0], H-12), (sh_w//2+20,8), 0, 0, 360, (0,0,0), -1, cv2.LINE_AA)

    # ── Robe body ─────────────────────────────────────────────────────────────
    rp = np.array([
        (ls[0]-arm_t-2, ls[1]),
        (rs[0]+arm_t+2, rs[1]),
        (hp_mid[0]+sh_w//2+15, H),
        (hp_mid[0]-sh_w//2-15, H),
    ], np.int32)
    # Robe shadow
    rp_s = rp.copy(); rp_s[:,0] += 3; rp_s[:,1] += 3
    cv2.fillPoly(img, [rp_s], C_ROBES, cv2.LINE_AA)
    cv2.fillPoly(img, [rp],   C_ROBE,  cv2.LINE_AA)
    # Subtle fabric folds
    for i in range(1, 5):
        fold_x = sh_mid[0] + int((i-2.5)*sh_w*0.18)
        cv2.line(img, (fold_x, sh_mid[1]+10), (fold_x+int((i-2.5)*8), H),
                 C_ROBED, 1, cv2.LINE_AA)
    # Collar V-neck
    col = np.array([
        (sh_mid[0]-sh_w//5, sh_mid[1]),
        (sh_mid[0], sh_mid[1]-sh_w//3),
        (sh_mid[0]+sh_w//5, sh_mid[1]),
    ], np.int32)
    cv2.polylines(img, [col], False, C_ROBED, 2, cv2.LINE_AA)
    # Shoulder width
    cv2.line(img, ls, rs, C_ROBE, arm_t*2+4, cv2.LINE_AA)
    cv2.line(img, ls, rs, C_ROBED, 2, cv2.LINE_AA)

    # ── Arms ─────────────────────────────────────────────────────────────────
    for sh, el, wr, v1, v2, v3 in [
        (ls, le, lw, vis(L_SH) if ls==ls_raw else vis(R_SH), vis(L_EL) if le==le_raw else vis(R_EL), vis(L_WR) if lw==lw_raw else vis(R_WR)),
        (rs, re, rw, vis(R_SH) if rs==rs_raw else vis(L_SH), vis(R_EL) if re==re_raw else vis(L_EL), vis(R_WR) if rw==rw_raw else vis(L_WR)),
    ]:
        if v1>0.2 and v2>0.2:
            cv2.line(img, (sh[0]+2,sh[1]+2), (el[0]+2,el[1]+2), C_ROBES, arm_t*2+2, cv2.LINE_AA)
            cv2.line(img, sh, el, C_ROBED, arm_t*2+2, cv2.LINE_AA)
            cv2.line(img, sh, el, C_ROBE,  arm_t*2,   cv2.LINE_AA)
        if v2>0.2 and v3>0.2:
            cv2.line(img, (el[0]+2,el[1]+2), (wr[0]+2,wr[1]+2), C_ROBES, arm_t*2, cv2.LINE_AA)
            cv2.line(img, el, wr, C_ROBED, arm_t*2,   cv2.LINE_AA)
            cv2.line(img, el, wr, C_ROBE,  arm_t*2-2, cv2.LINE_AA)
            # Sleeve cuff
            cuff = lp(el, wr, 0.80)
            cv2.line(img, cuff, wr, C_SLVD,   arm_t*2+2, cv2.LINE_AA)
            cv2.line(img, cuff, wr, C_SLEEVE, arm_t*2,   cv2.LINE_AA)
            # Cuff trim
            cv2.line(img, cuff, (cuff[0]+int((wr[0]-cuff[0])*0.25), cuff[1]+int((wr[1]-cuff[1])*0.25)),
                     (200,190,160), max(1,arm_t//3), cv2.LINE_AA)
        if v2>0.2:
            cv2.circle(img, el, arm_t+1, C_ROBED, -1, cv2.LINE_AA)
            cv2.circle(img, el, arm_t-1, C_ROBE,  -1, cv2.LINE_AA)

    # ── Wrist / hand base ────────────────────────────────────────────────────
    for hand, wr_pt, v_wr in [(rhand_use, rw, vis(R_WR)), (lhand_use, lw, vis(L_WR))]:
        if v_wr > 0.2:
            cv2.circle(img, wr_pt, arm_t+1, C_SKIND, -1, cv2.LINE_AA)
            cv2.circle(img, wr_pt, arm_t-1, C_SKIN,  -1, cv2.LINE_AA)
        draw_realistic_hand(img, hand, C_SKIN, C_SKIND, arm_t)

    # ── Face ─────────────────────────────────────────────────────────────────
    draw_face(img, nose, sh_w)


def render_avatar_video(sign: str) -> str | None:
    out_path = AVATAR_DIR / f"{sign.upper()}.mp4"
    mocap = MOCAP_DIR / f"{sign.upper()}.json"
    if not mocap.exists(): return None

    with open(mocap) as f: data = json.load(f)
    fps = data.get('fps', 25)
    frames = data['frames']

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
        # Rich dark background gradient
        for y in range(H):
            t=y/H
            img[y,:] = (int(14+t*8), int(12+t*6), int(26+t*14))
        # Subtle vignette
        draw_avatar(img, fd.get('pose'), fd.get('rhand'), fd.get('lhand'))
        out.write(img)
    out.release()

    subprocess.run(['ffmpeg','-y','-i',tmp,'-c:v','libx264','-crf','17',
                    '-preset','fast','-pix_fmt','yuv420p',str(out_path)],
                   capture_output=True, timeout=30)
    if os.path.exists(tmp): os.unlink(tmp)
    if out_path.exists() and out_path.stat().st_size>5000:
        print(f"[Avatar] {sign} → {out_path.stat().st_size//1024}KB")
        return str(out_path)
    return None


def get_or_render_avatar(sign: str) -> str | None:
    out_path = AVATAR_DIR / f"{sign.upper()}.mp4"
    if out_path.exists() and out_path.stat().st_size > 5000:
        return str(out_path)
    return render_avatar_video(sign)


if __name__ == "__main__":
    import sys
    signs = sys.argv[1:] if len(sys.argv)>1 else sorted(p.stem for p in MOCAP_DIR.glob("*.json"))
    print(f"Rendering {len(signs)} realistic Arab avatar videos...")
    for s in signs:
        render_avatar_video(s)
    print("Done.")
