"""
Map MediaPipe pose landmarks to Arab Sheik GLB bone rotations
and render a 2D avatar video that follows the motion.
"""
import cv2, numpy as np, json, math, struct

# ── Load GLB rest pose ────────────────────────────────────────────────────────
with open('/root/.openclaw/workspace/esl-platform/frontend/public/avatar/arab-man.glb','rb') as f:
    f.read(12); cl=struct.unpack('<I',f.read(4))[0]; f.read(4)
    gltf=json.loads(f.read(cl))
nodes=gltf['nodes']
REST={n['name']:n.get('rotation',[0,0,0,1]) for n in nodes if 'name' in n}

# ── Load motion capture data ──────────────────────────────────────────────────
with open('/tmp/motion_capture.json') as f:
    mc = json.load(f)
fps = mc['fps']
frames = mc['frames']
TOTAL = len(frames)
print(f'Loaded {TOTAL} frames @ {fps}fps')

# ── Quaternion helpers ────────────────────────────────────────────────────────
def norm(v):
    n=math.sqrt(sum(x*x for x in v)+1e-12); return [x/n for x in v]

def mul_q(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return norm([aw*bx+ax*bw+ay*bz-az*by,
                 aw*by-ax*bz+ay*bw+az*bx,
                 aw*bz+ax*by-ay*bx+az*bw,
                 aw*bw-ax*bx-ay*by-az*bz])

def e2q(rx,ry,rz):
    cx,cy,cz=math.cos(rx/2),math.cos(ry/2),math.cos(rz/2)
    sx,sy,sz=math.sin(rx/2),math.sin(ry/2),math.sin(rz/2)
    return norm([sx*cy*cz+cx*sy*sz, cx*sy*cz-sx*cy*sz,
                 cx*cy*sz+sx*sy*cz, cx*cy*cz-sx*sy*sz])

def q2euler(q):
    x,y,z,w=q
    pitch=math.asin(max(-1,min(1,2*(w*y-z*x))))
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    return pitch,yaw,roll

def slerp(q1,q2,t):
    dot=sum(a*b for a,b in zip(q1,q2))
    if dot<0: q2=[-x for x in q2]; dot=-dot
    dot=min(1,dot)
    theta=math.acos(dot)
    if theta<0.001: return norm([a+t*(b-a) for a,b in zip(q1,q2)])
    return norm([(math.sin((1-t)*theta)*a+math.sin(t*theta)*b)/math.sin(theta)
                 for a,b in zip(q1,q2)])

# ── MediaPipe pose landmark indices ──────────────────────────────────────────
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
NOSE=0; L_SHOULDER=11; R_SHOULDER=12
L_ELBOW=13; R_ELBOW=14; L_WRIST=15; R_WRIST=16
L_HIP=23; R_HIP=24; L_KNEE=25; R_KNEE=26; L_ANKLE=27; R_ANKLE=28

# Hand landmarks (21 each)
WRIST_H=0
THUMB_CMC=1;THUMB_MCP=2;THUMB_IP=3;THUMB_TIP=4
INDEX_MCP=5;INDEX_PIP=6;INDEX_DIP=7;INDEX_TIP=8
MIDDLE_MCP=9;MIDDLE_PIP=10;MIDDLE_DIP=11;MIDDLE_TIP=12
RING_MCP=13;RING_PIP=14;RING_DIP=15;RING_TIP=16
PINKY_MCP=17;PINKY_PIP=18;PINKY_DIP=19;PINKY_TIP=20

def v3(a,b): return [b[0]-a[0],b[1]-a[1],b[2]-a[2]]
def len3(v): return math.sqrt(sum(x*x for x in v)+1e-12)
def norm3(v): l=len3(v); return [x/l for x in v]
def dot3(a,b): return sum(x*y for x,y in zip(a,b))
def angle3(a,o,b):
    va=norm3(v3(o,a)); vb=norm3(v3(o,b))
    return math.acos(max(-1,min(1,dot3(va,vb))))

def cross3(a,b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

# ── Compute bone offset per frame ─────────────────────────────────────────────
def compute_bone_offsets(fd):
    offsets = {}
    pose = fd.get('pose')
    rhand = fd.get('rhand')
    lhand = fd.get('lhand')

    if pose:
        lsh=pose[L_SHOULDER]; rsh=pose[R_SHOULDER]
        lel=pose[L_ELBOW];    rel=pose[R_ELBOW]
        lwr=pose[L_WRIST];    rwr=pose[R_WRIST]
        lhp=pose[L_HIP];      rhp=pose[R_HIP]

        # ── Torso tilt (Spine2) ──────────────────────────────────────────────
        sh_mid = [(lsh[i]+rsh[i])/2 for i in range(3)]
        hp_mid = [(lhp[i]+rhp[i])/2 for i in range(3)]
        spine_v = norm3(v3(hp_mid, sh_mid))
        torso_tilt_x = math.atan2(spine_v[2], spine_v[1]) * 0.4
        torso_side_z = math.atan2(sh_mid[0]-hp_mid[0], abs(sh_mid[1]-hp_mid[1])) * 0.5
        offsets['Spine1'] = (torso_tilt_x*0.3, 0, torso_side_z*0.3)
        offsets['Spine2'] = (torso_tilt_x*0.4, 0, torso_side_z*0.4)

        # ── Right arm ────────────────────────────────────────────────────────
        if rsh[3]>0.3 and rel[3]>0.3:
            arm_v = norm3(v3(rsh, rel))
            # Forward-back (rz in GLB space)
            rz = -math.atan2(-arm_v[2], arm_v[1]) * 0.9
            # Up-down
            ry = math.atan2(arm_v[0], arm_v[1]) * 0.5
            offsets['RightArm'] = (0, ry, rz)

            if rwr[3]>0.3:
                fore_v = norm3(v3(rel, rwr))
                elbow_angle = angle3(rsh, rel, rwr)
                rz_f = -(math.pi - elbow_angle) * 0.6
                offsets['RightForeArm'] = (0, 0, rz_f)

        # ── Left arm ─────────────────────────────────────────────────────────
        if lsh[3]>0.3 and lel[3]>0.3:
            arm_v = norm3(v3(lsh, lel))
            rz = math.atan2(-arm_v[2], arm_v[1]) * 0.9
            ry = -math.atan2(arm_v[0], arm_v[1]) * 0.5
            offsets['LeftArm'] = (0, ry, rz)

            if lwr[3]>0.3:
                elbow_angle = angle3(lsh, lel, lwr)
                rz_f = -(math.pi - elbow_angle) * 0.6
                offsets['LeftForeArm'] = (0, 0, rz_f)

    # ── Right hand fingers ────────────────────────────────────────────────────
    if rhand:
        def finger_curl(mcp,pip,dip,tip):
            a1=angle3(rhand[mcp],rhand[pip],rhand[dip])
            a2=angle3(rhand[pip],rhand[dip],rhand[tip])
            curl1=max(0,(math.pi-a1)*0.9)
            curl2=max(0,(math.pi-a2)*0.7)
            return curl1,curl2

        # Wrist orientation
        wrist_v = norm3(v3(rhand[WRIST_H], rhand[MIDDLE_MCP]))
        offsets['RightHand'] = (0, -math.atan2(wrist_v[0],abs(wrist_v[1]))*0.3, 0)

        for bone_pre, mcp, pip, dip, tip in [
            ('RightHandIndex', INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            ('RightHandMiddle', MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            ('RightHandRing', RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            ('RightHandPinky', PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ]:
            c1,c2 = finger_curl(mcp,pip,dip,tip)
            offsets[f'{bone_pre}1'] = (c1, 0, 0)
            offsets[f'{bone_pre}2'] = (c2, 0, 0)
            offsets[f'{bone_pre}3'] = (c2*0.6, 0, 0)

        # Thumb
        ta = angle3(rhand[THUMB_CMC], rhand[THUMB_MCP], rhand[THUMB_IP])
        tb = angle3(rhand[THUMB_MCP], rhand[THUMB_IP], rhand[THUMB_TIP])
        offsets['RightHandThumb1'] = (max(0,(math.pi-ta)*0.6), 0, -0.3)
        offsets['RightHandThumb2'] = (max(0,(math.pi-tb)*0.5), 0, 0)

    # ── Left hand fingers ─────────────────────────────────────────────────────
    if lhand:
        def finger_curl_l(mcp,pip,dip,tip):
            a1=angle3(lhand[mcp],lhand[pip],lhand[dip])
            a2=angle3(lhand[pip],lhand[dip],lhand[tip])
            return max(0,(math.pi-a1)*0.9), max(0,(math.pi-a2)*0.7)

        offsets['LeftHand'] = (0, math.atan2(
            norm3(v3(lhand[WRIST_H],lhand[MIDDLE_MCP]))[0],
            abs(norm3(v3(lhand[WRIST_H],lhand[MIDDLE_MCP]))[1]))*0.3, 0)

        for bone_pre, mcp, pip, dip, tip in [
            ('LeftHandIndex', INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            ('LeftHandMiddle', MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            ('LeftHandRing', RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            ('LeftHandPinky', PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ]:
            c1,c2 = finger_curl_l(mcp,pip,dip,tip)
            offsets[f'{bone_pre}1'] = (c1, 0, 0)
            offsets[f'{bone_pre}2'] = (c2, 0, 0)
            offsets[f'{bone_pre}3'] = (c2*0.6, 0, 0)

        ta = angle3(lhand[THUMB_CMC],lhand[THUMB_MCP],lhand[THUMB_IP])
        tb = angle3(lhand[THUMB_MCP],lhand[THUMB_IP],lhand[THUMB_TIP])
        offsets['LeftHandThumb1'] = (max(0,(math.pi-ta)*0.6), 0, 0.3)
        offsets['LeftHandThumb2'] = (max(0,(math.pi-tb)*0.5), 0, 0)

    return offsets

# ── Smooth offsets across frames (temporal filter) ────────────────────────────
def smooth_frames(all_offsets, alpha=0.5):
    smoothed = [all_offsets[0]]
    for i in range(1, len(all_offsets)):
        cur = all_offsets[i]; prev = smoothed[-1]
        merged = {}
        all_keys = set(cur) | set(prev)
        for k in all_keys:
            c = cur.get(k,(0,0,0)); p = prev.get(k,(0,0,0))
            merged[k] = tuple(alpha*c[j]+(1-alpha)*p[j] for j in range(3))
        smoothed.append(merged)
    return smoothed

# ── Pre-compute all bone offsets ──────────────────────────────────────────────
print('Computing bone offsets for all frames...')
all_offsets = [compute_bone_offsets(fd) for fd in frames]
all_offsets = smooth_frames(all_offsets, alpha=0.6)

# ── Renderer ─────────────────────────────────────────────────────────────────
AW, AH = 640, 720
SKIN=(185,148,115); SKIN_D=(155,120,88)
ROBE=(232,230,226); ROBE_D=(200,198,194)
SLEEVE=(68,108,162); SLEEVE_D=(48,82,130)
TROUSER=(62,46,110)
font = cv2.FONT_HERSHEY_SIMPLEX

tmp_avi = '/tmp/avatar_motion.avi'
out = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*'MJPG'), fps, (AW, AH))

CX = AW//2
BASE_SH_Y = int(AH*0.36)
HIPS_Y = int(AH*0.56)

def apply_bone(bone, rx, ry, rz):
    return mul_q(REST.get(bone,[0,0,0,1]), e2q(rx,ry,rz))

def get_screen_arm(shoulder_pos, bone_name, fore_name, offsets, len1=85, len2=78, flip=1):
    q_a = apply_bone(bone_name, *offsets.get(bone_name,(0,0,0)))
    p_a,y_a,r_a = q2euler(q_a)
    el = (shoulder_pos[0] + int(math.sin(y_a)*len1*flip),
          shoulder_pos[1] + int(math.cos(p_a)*len1*0.65))
    q_f = apply_bone(fore_name, *offsets.get(fore_name,(0,0,0)))
    p_f,y_f,r_f = q2euler(q_f)
    wr = (el[0] + int(math.sin((y_a+y_f*0.5))*len2*flip),
          el[1] + int(math.cos((p_a+p_f*0.5))*len2*0.6))
    return el, wr

def draw_finger_3d(img, base, angle, segs, curls, color=SKIN, thick_base=6):
    x,y=base
    for i,(seg,curl) in enumerate(zip(segs,curls)):
        nx=int(x+math.sin(angle+curl)*seg)
        ny=int(y-math.cos(angle+curl)*seg)
        t = max(2, thick_base-i*2)
        cv2.line(img,(x,y),(nx,ny),(max(0,color[0]-20),max(0,color[1]-20),max(0,color[2]-20)),t+1,cv2.LINE_AA)
        cv2.line(img,(x,y),(nx,ny),color,t,cv2.LINE_AA)
        cv2.circle(img,(x,y),t//2+1,color,-1,cv2.LINE_AA)
        x,y=nx,ny
    cv2.circle(img,(x,y),3,SKIN_D,-1,cv2.LINE_AA)

print('Rendering avatar frames...')
for fi, (fd, offsets) in enumerate(zip(frames, all_offsets)):
    img = np.zeros((AH,AW,3),dtype=np.uint8)
    # Gradient bg
    for y in range(AH): img[y,:] = (int(8+y/AH*20), int(8+y/AH*15), int(20+y/AH*35))

    # Torso shift from spine
    sx_off = int(offsets.get('Spine2',(0,0,0))[2] * 30)
    sy_off = int(offsets.get('Spine1',(0,0,0))[0] * 20)
    cx = CX + sx_off

    sh_y = BASE_SH_Y + sy_off
    rs = (cx+72, sh_y); ls = (cx-72, sh_y)
    nk = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hips_y = HIPS_Y + sy_off//2

    # Shadow
    cv2.ellipse(img,(CX,AH-22),(70,10),0,0,360,(0,0,0),-1,cv2.LINE_AA)

    # Robe
    rp = np.array([(ls[0]-24,ls[1]),(rs[0]+24,rs[1]),
                   (rs[0]+42,hips_y+22),(rs[0]+34,AH-48),
                   (ls[0]-34,AH-48),(ls[0]-42,hips_y+22)],np.int32)
    cv2.fillPoly(img,[rp],ROBE,cv2.LINE_AA)
    cv2.polylines(img,[rp],True,ROBE_D,2,cv2.LINE_AA)
    cv2.line(img,nk,((ls[0]+rs[0])//2,hips_y+8),ROBE_D,2,cv2.LINE_AA)
    cv2.line(img,ls,rs,ROBE,16,cv2.LINE_AA)

    # Legs
    for lx,sign in [(cx-16,1),(cx+16,-1)]:
        kn=(lx-sign*2,hips_y+int(AH*0.16)); ak=(lx-sign*1,hips_y+int(AH*0.31))
        cv2.line(img,(lx,hips_y),kn,TROUSER,15,cv2.LINE_AA)
        cv2.line(img,kn,ak,TROUSER,13,cv2.LINE_AA)
        toe=(ak[0]-sign*20,ak[1]+7)
        cv2.line(img,ak,toe,TROUSER,11,cv2.LINE_AA)

    # Right arm
    r_el, r_wr = get_screen_arm(rs,'RightArm','RightForeArm',offsets,flip=1)
    cv2.line(img,rs,r_el,SLEEVE_D,14,cv2.LINE_AA); cv2.line(img,rs,r_el,SLEEVE,10,cv2.LINE_AA)
    cv2.line(img,r_el,r_wr,SLEEVE_D,12,cv2.LINE_AA); cv2.line(img,r_el,r_wr,SLEEVE,8,cv2.LINE_AA)

    # Right hand + fingers
    q_ra=apply_bone('RightArm',*offsets.get('RightArm',(0,0,0)))
    _,y_ra,_=q2euler(q_ra); arm_a=-y_ra+0.15
    pp=np.array([(r_wr[0]-11,r_wr[1]+2),(r_wr[0]+11,r_wr[1]+2),
                 (r_wr[0]+13,r_wr[1]-14),(r_wr[0],r_wr[1]-22),
                 (r_wr[0]-13,r_wr[1]-14)],np.int32)
    cv2.fillPoly(img,[pp],SKIN,cv2.LINE_AA)
    for (bx,by),a_off,segs,bone_pre in [
        (( r_wr[0]-7, r_wr[1]-16), -0.18,[22,18,14],'RightHandIndex'),
        (( r_wr[0]+0, r_wr[1]-17),  0.0, [24,19,15],'RightHandMiddle'),
        (( r_wr[0]+8, r_wr[1]-14),  0.16,[21,17,13],'RightHandRing'),
        (( r_wr[0]+14,r_wr[1]-10),  0.32,[18,14,10],'RightHandPinky'),
    ]:
        c1=offsets.get(f'{bone_pre}1',(0,0,0))[0]
        c2=offsets.get(f'{bone_pre}2',(0,0,0))[0]
        draw_finger_3d(img,(bx,by),arm_a+a_off,segs,[c1,c2,c2*0.6])
    # Thumb
    tc=offsets.get('RightHandThumb1',(0,0,0))[0]
    draw_finger_3d(img,(r_wr[0]-10,r_wr[1]-6),arm_a-0.6,[15,12],[tc,tc*0.5])

    # Left arm
    l_el, l_wr = get_screen_arm(ls,'LeftArm','LeftForeArm',offsets,flip=-1)
    cv2.line(img,ls,l_el,SLEEVE_D,14,cv2.LINE_AA); cv2.line(img,ls,l_el,SLEEVE,10,cv2.LINE_AA)
    cv2.line(img,l_el,l_wr,SLEEVE_D,12,cv2.LINE_AA); cv2.line(img,l_el,l_wr,SLEEVE,8,cv2.LINE_AA)

    q_la=apply_bone('LeftArm',*offsets.get('LeftArm',(0,0,0)))
    _,y_la,_=q2euler(q_la); arm_la=y_la-0.15
    lpp=np.array([(l_wr[0]-11,l_wr[1]+2),(l_wr[0]+11,l_wr[1]+2),
                  (l_wr[0]+13,l_wr[1]-14),(l_wr[0],l_wr[1]-22),
                  (l_wr[0]-13,l_wr[1]-14)],np.int32)
    cv2.fillPoly(img,[lpp],SKIN,cv2.LINE_AA)
    for (bx,by),a_off,segs,bone_pre in [
        ((l_wr[0]-14,l_wr[1]-10), -0.32,[18,14,10],'LeftHandPinky'),
        ((l_wr[0]-8, l_wr[1]-14), -0.16,[21,17,13],'LeftHandRing'),
        ((l_wr[0]+0, l_wr[1]-17),  0.0, [24,19,15],'LeftHandMiddle'),
        ((l_wr[0]+7, l_wr[1]-16),  0.18,[22,18,14],'LeftHandIndex'),
    ]:
        c1=offsets.get(f'{bone_pre}1',(0,0,0))[0]
        c2=offsets.get(f'{bone_pre}2',(0,0,0))[0]
        draw_finger_3d(img,(bx,by),arm_la+a_off,segs,[c1,c2,c2*0.6])
    tc=offsets.get('LeftHandThumb1',(0,0,0))[0]
    draw_finger_3d(img,(l_wr[0]+10,l_wr[1]-6),arm_la+0.6,[15,12],[tc,tc*0.5])

    # Neck + Head
    cv2.line(img,nk,(cx,sh_y-36),SKIN,13,cv2.LINE_AA)
    hx2,hy2=cx,sh_y-74; hr=38
    cv2.ellipse(img,(hx2,hy2),(hr,int(hr*1.18)),0,0,360,SKIN_D,-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx2,hy2-3),(hr,int(hr*1.1)),0,0,360,SKIN,-1,cv2.LINE_AA)
    ex=int(hr*0.29); ey=int(hr*0.19)
    for ex2 in [-ex,ex]:
        cv2.circle(img,(hx2+ex2,hy2-ey),int(hr*0.12),(40,28,18),-1,cv2.LINE_AA)
        cv2.circle(img,(hx2+ex2+2,hy2-ey-2),int(hr*0.05),(215,215,220),-1)
    cv2.line(img,(hx2-ex-4,hy2-ey-5),(hx2-ex+4,hy2-ey-4),(40,26,16),2,cv2.LINE_AA)
    cv2.line(img,(hx2+ex-4,hy2-ey-4),(hx2+ex+4,hy2-ey-5),(40,26,16),2,cv2.LINE_AA)
    cv2.ellipse(img,(hx2,hy2+int(hr*0.38)),(int(hr*0.4),int(hr*0.21)),0,0,180,(35,24,16),-1,cv2.LINE_AA)
    # Ghutrah
    gp=np.array([(hx2-hr-8,hy2-int(hr*0.14)),(hx2,hy2-int(hr*1.34)),
                 (hx2+hr+8,hy2-int(hr*0.14)),(hx2+hr-4,hy2+int(hr*0.36)),
                 (hx2-hr+4,hy2+int(hr*0.36))],np.int32)
    cv2.fillPoly(img,[gp],ROBE,cv2.LINE_AA)
    cv2.polylines(img,[gp],True,ROBE_D,2,cv2.LINE_AA)
    cv2.ellipse(img,(hx2,hy2-int(hr*0.86)),(hr-5,int(hr*0.22)),0,0,360,(17,14,11),-1,cv2.LINE_AA)
    cv2.ellipse(img,(hx2,hy2-int(hr*0.86)),(hr-5,int(hr*0.22)),0,0,360,(33,28,22),2,cv2.LINE_AA)

    # Frame counter
    cv2.putText(img,f'ESL Avatar | {fi+1}/{TOTAL}',(8,18),font,0.38,(80,80,100),1,cv2.LINE_AA)
    # Progress
    prog=int(AW*0.8*(fi+1)/TOTAL)
    cv2.rectangle(img,(int(AW*0.1),AH-8),(int(AW*0.1)+int(AW*0.8),AH-3),(30,28,50),-1)
    cv2.rectangle(img,(int(AW*0.1),AH-8),(int(AW*0.1)+prog,AH-3),(124,58,237),-1)

    out.write(img)
    if (fi+1)%50==0: print(f'  Rendered {fi+1}/{TOTAL}')

out.release()
print('Converting to MP4...')
import subprocess, os
subprocess.run(['ffmpeg','-y','-i',tmp_avi,'-c:v','libx264','-crf','18',
                '-preset','fast','-pix_fmt','yuv420p',
                '/root/.openclaw/workspace/avatar_motion.mp4'],
               capture_output=True)
os.unlink(tmp_avi)
sz=os.path.getsize('/root/.openclaw/workspace/avatar_motion.mp4')//1024
print(f'Done! avatar_motion.mp4 ({sz}KB)')
