"""
Automated GLB Rigger for MediaPipe World Landmarks
====================================================
Computes per-frame bone quaternions from MediaPipe pose_world_landmarks + hand landmarks.
Uses the proven 0.0000° error formula:
  offset = conj(boneWorldQ) * worldCorrection * boneWorldQ
  final  = rest_q * offset

Coordinate mapping (verified):
  MediaPipe world: x=signer-right, y=UP, z=signer-back
  GLB model:       x=model-right,  y=UP, z=model-back
  → GLB_x = -MP_x  (signer faces camera = mirror X)
  → GLB_y = +MP_y
  → GLB_z = +MP_z

Usage:
  from auto_rigger import AutoRigger
  rigger = AutoRigger('/path/to/model.glb')
  bone_quats = rigger.rig_frame(pose_world_lms, right_hand_lms, left_hand_lms)
  # bone_quats: {bone_name: [x,y,z,w] quaternion}
"""
import struct, json, math
import numpy as np
from pathlib import Path


# ── Quaternion helpers ─────────────────────────────────────────────────────────
def qn(q):
    n = np.linalg.norm(q)
    return q/n if n > 1e-9 else np.array([0,0,0,1.0])

def qmul(a, b):
    ax,ay,az,aw = a; bx,by,bz,bw = b
    return qn(np.array([aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx,
                         aw*bz+ax*by-ay*bx+az*bw,
                         aw*bw-ax*bx-ay*by-az*bz]))

def q2mat(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def qinv(q):
    return np.array([-q[0],-q[1],-q[2],q[3]])

def rot_from_to(a, b):
    """Quaternion rotating unit vector a to unit vector b."""
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    c = np.clip(np.dot(a, b), -1, 1)
    if c > 0.9999: return np.array([0,0,0,1.0])
    if c < -0.9999:
        perp = np.array([0,0,1.0]) if abs(a[0]) < 0.9 else np.array([0,1,0.0])
        ax = np.cross(a, perp); ax /= np.linalg.norm(ax)
        return np.array([ax[0],ax[1],ax[2],0.0])
    ax = np.cross(a, b); s = math.sqrt((1+c)*2)
    return qn(np.array([ax[0]/s, ax[1]/s, ax[2]/s, s*0.5]))

def retarget(target_dir, bone_world_q):
    """Compute LOCAL offset quaternion to make bone point toward target_dir.
    Formula: offset = conj(boneWorldQ) * worldCorrection * boneWorldQ
    Proven 0.0000° error.
    """
    cur = q2mat(bone_world_q) @ np.array([0,1,0])
    world_corr = rot_from_to(cur, target_dir)
    return qmul(qmul(qinv(bone_world_q), world_corr), bone_world_q)


# ── Coordinate conversion ──────────────────────────────────────────────────────
def mp2glb(lm):
    """MediaPipe world landmark → GLB coordinate space.
    MP: x=signer-right, y=UP, z=signer-back
    GLB: x=model-right(=viewer-right), y=UP, z=model-back
    Signer faces camera → flip X (signer-right = viewer-left = model-left = -GLB_x)
    """
    return np.array([-lm[0], lm[1], lm[2]])


class AutoRigger:
    """Automatically rigs a GLB model to MediaPipe world landmarks."""

    def __init__(self, glb_path: str):
        self.glb_path = glb_path
        self._load_skeleton()

    def _load_skeleton(self):
        """Load GLB skeleton and compute rest world quaternions for all bones."""
        with open(self.glb_path, 'rb') as f:
            f.read(12)
            cl = struct.unpack('<I', f.read(4))[0]; f.read(4)
            gltf = json.loads(f.read(cl))

        nodes = gltf['nodes']
        parent_map = {}
        for i, n in enumerate(nodes):
            for c in n.get('children', []): parent_map[c] = i

        # Build rest quaternions (local and world-accumulated)
        self.rest_local = {}   # bone_name → local rest quaternion
        self.rest_world = {}   # bone_name → accumulated world quaternion
        self.node_idx   = {}   # bone_name → node index

        world_cache = {}
        def get_world(ni):
            if ni in world_cache: return world_cache[ni]
            q = np.array(nodes[ni].get('rotation', [0,0,0,1]))
            if ni in parent_map:
                q = qmul(get_world(parent_map[ni]), q)
            world_cache[ni] = q; return q

        for i, n in enumerate(nodes):
            name = n.get('name')
            if not name: continue
            self.node_idx[name]   = i
            self.rest_local[name] = np.array(n.get('rotation', [0,0,0,1]))
            self.rest_world[name] = get_world(i)

        # Pre-compute bone world quaternions needed for retargeting
        # These are the accumulated world Q at each bone's REST pose
        self._bwq = {}
        for name in self.rest_world:
            self._bwq[name] = self.rest_world[name]

        print(f"[AutoRigger] Loaded {len(self.rest_world)} bones from {Path(self.glb_path).name}")

    def _arm_chain(self, sh_world_q, arm_rest_q, fore_rest_q,
                   sh_lm, el_lm, wr_lm, side='right'):
        """Rig one arm: shoulder→elbow→wrist."""
        offs = {}
        arm_dir  = mp2glb(el_lm) - mp2glb(sh_lm)
        arm_dir /= np.linalg.norm(arm_dir) + 1e-9

        arm_world_q   = qmul(sh_world_q, arm_rest_q)
        arm_off       = retarget(arm_dir, arm_world_q)
        arm_final_q   = qmul(arm_rest_q, arm_off)
        offs['arm']   = arm_off

        if wr_lm is not None:
            fore_dir  = mp2glb(wr_lm) - mp2glb(el_lm)
            fore_dir /= np.linalg.norm(fore_dir) + 1e-9
            fore_world_q  = qmul(qmul(sh_world_q, arm_final_q), fore_rest_q)
            offs['fore']  = retarget(fore_dir, fore_world_q)

        return offs

    def _finger_chain(self, hand_lms, bone_world_q, mcp_idx, pip_idx, dip_idx):
        """Rig one finger from hand landmarks."""
        offs = {}
        # Each segment: direction from joint A to joint B
        pairs = [(0, mcp_idx, 'mcp'), (mcp_idx, pip_idx, 'pip'), (pip_idx, dip_idx, 'dip')]
        current_wq = bone_world_q
        for a, b, seg in pairs:
            if a >= len(hand_lms) or b >= len(hand_lms): break
            seg_dir  = np.array(hand_lms[b]) - np.array(hand_lms[a])
            seg_dir /= np.linalg.norm(seg_dir) + 1e-9
            # Convert hand landmarks (image-space) to GLB
            seg_glb  = np.array([-seg_dir[0], -seg_dir[1], seg_dir[2]])
            off      = retarget(seg_glb, current_wq)
            offs[seg] = off
            # Accumulate for next joint
            current_wq = qmul(current_wq, off)
        return offs

    def rig_frame(self, pose_lms, right_hand_lms=None, left_hand_lms=None,
                  smoothing=0.0):
        """
        Compute bone quaternion offsets for one frame.

        Args:
            pose_lms: list of 33 landmarks, each [x,y,z] or [x,y,z,vis]
                      from pose_world_landmarks
            right_hand_lms: list of 21 hand landmarks [x,y,z] or None
            left_hand_lms:  list of 21 hand landmarks [x,y,z] or None
            smoothing: 0=no smoothing, 0.8=heavy smoothing (for animation)

        Returns:
            dict: {bone_name: np.array([x,y,z,w])} — offset quaternions
                  Apply as: final_q = rest_q * offset_q
        """
        offs = {}
        P = pose_lms  # shorthand
        vis = lambda i: (P[i][3] if len(P[i]) > 3 else 1.0) > 0.3

        def mid(a, b): return [(P[a][j]+P[b][j])/2 for j in range(3)]

        # ── Spine ──────────────────────────────────────────────────────────────
        if vis(11) and vis(12) and vis(23) and vis(24):
            sh = mp2glb(mid(11,12)); hp = mp2glb(mid(23,24))
            spine_dir = sh - hp; spine_dir /= np.linalg.norm(spine_dir) + 1e-9
            spine_off = retarget(spine_dir, self._bwq.get('Spine2', np.array([0,0,0,1])))
            # Apply 25% correction (spine doesn't need full retarget)
            identity = np.array([0,0,0,1.0])
            for bone in ['Spine', 'Spine1', 'Spine2']:
                if bone in self._bwq:
                    blended = qn(identity*(1-0.25) + spine_off*0.25)
                    offs[bone] = blended

        # ── Right arm ──────────────────────────────────────────────────────────
        if vis(12) and vis(14):
            sh_wq = self._bwq.get('RightShoulder', np.array([0,0,0,1]))
            arm_r  = self.rest_local.get('RightArm',    np.array([0,0,0,1]))
            fore_r = self.rest_local.get('RightForeArm',np.array([0,0,0,1]))
            wr = P[16] if vis(16) else None
            arm_offs = self._arm_chain(sh_wq, arm_r, fore_r, P[12], P[14], wr, 'right')
            if 'arm'  in arm_offs: offs['RightArm']    = arm_offs['arm']
            if 'fore' in arm_offs: offs['RightForeArm']= arm_offs['fore']

        # ── Left arm ───────────────────────────────────────────────────────────
        if vis(11) and vis(13):
            sh_wq = self._bwq.get('LeftShoulder', np.array([0,0,0,1]))
            arm_r  = self.rest_local.get('LeftArm',    np.array([0,0,0,1]))
            fore_r = self.rest_local.get('LeftForeArm',np.array([0,0,0,1]))
            wr = P[15] if vis(15) else None
            arm_offs = self._arm_chain(sh_wq, arm_r, fore_r, P[11], P[13], wr, 'left')
            if 'arm'  in arm_offs: offs['LeftArm']    = arm_offs['arm']
            if 'fore' in arm_offs: offs['LeftForeArm']= arm_offs['fore']

        # ── Fingers ────────────────────────────────────────────────────────────
        FINGER_MAP = [
            ('Thumb',  1, 2, 3),    # mcp=1,pip=2,dip=3
            ('Index',  5, 6, 7),
            ('Middle', 9,10,11),
            ('Ring',  13,14,15),
            ('Pinky', 17,18,19),
        ]

        def rig_hand(hand_lms, side):
            if not hand_lms: return
            # Check if hand is valid (not all zeros)
            if abs(hand_lms[0][0]) < 0.001 and abs(hand_lms[0][1]) < 0.001: return

            for fname, mcp, pip, dip in FINGER_MAP:
                bone1 = f'{side}Hand{fname}1'
                bone2 = f'{side}Hand{fname}2'
                bone3 = f'{side}Hand{fname}3'

                if mcp >= len(hand_lms) or dip >= len(hand_lms): continue

                # Compute joint angles (curl = deviation from straight)
                def lm2np(i): return np.array(hand_lms[i][:3])
                def joint_angle(a, o, b):
                    va = lm2np(a)-lm2np(o); vb = lm2np(b)-lm2np(o)
                    la, lb = np.linalg.norm(va), np.linalg.norm(vb)
                    if la < 1e-6 or lb < 1e-6: return 0
                    return math.acos(np.clip(np.dot(va/la, vb/lb), -1, 1))

                # Curl = pi - angle (0=straight, pi=fully bent)
                curl1 = max(0, joint_angle(mcp-1 if mcp>0 else 0, mcp, pip))
                curl2 = max(0, joint_angle(mcp, pip, dip))

                # Apply as Euler rotation around local X axis
                sign = -1 if side == 'Right' else 1
                def euler_quat(angle):
                    return np.array([math.sin(angle/2)*sign, 0, 0, math.cos(angle/2)])

                offs[bone1] = euler_quat(curl1 * 0.8)
                offs[bone2] = euler_quat(curl2 * 0.75)
                offs[bone3] = euler_quat(curl2 * 0.4)

        rig_hand(right_hand_lms, 'Right')
        rig_hand(left_hand_lms, 'Left')

        return offs

    def get_rest_quaternion(self, bone_name):
        """Get the rest quaternion for a bone (apply offset ON TOP of this)."""
        return self.rest_local.get(bone_name, np.array([0,0,0,1]))


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    glb = '/root/.openclaw/workspace/esl-platform/frontend/public/avatar/arab-man.glb'
    rigger = AutoRigger(glb)

    # Load a real mocap frame and test
    import json as js
    with open('/root/.openclaw/workspace/esl-platform/data/processed/mocap/DOCTOR.json') as f:
        d = js.load(f)

    frame = d['frames'][21]
    pose  = frame.get('pose', [])
    rhand = frame.get('rhand')
    lhand = frame.get('lhand')

    print(f"\nRigging DOCTOR frame 21...")
    offs = rigger.rig_frame(pose, rhand, lhand)
    print(f"Bones driven: {len(offs)}")
    for bone, q in sorted(offs.items()):
        if np.linalg.norm(np.array(q) - np.array([0,0,0,1])) > 0.01:
            print(f"  {bone}: [{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}]")

    # Verify arm error
    import numpy as np
    sh_q  = rigger._bwq.get('RightShoulder', np.array([0,0,0,1]))
    arm_r = rigger.rest_local.get('RightArm', np.array([0,0,0,1]))
    arm_off = offs.get('RightArm', np.array([0,0,0,1]))
    arm_fin = qmul(arm_r, arm_off)
    arm_wq  = qmul(sh_q, arm_fin)
    result  = q2mat(arm_wq) @ np.array([0,1,0])
    target  = mp2glb(pose[14]) - mp2glb(pose[12])
    target /= np.linalg.norm(target)
    err = math.degrees(math.acos(np.clip(np.dot(result, target), -1, 1)))
    print(f"\nArm retarget error: {err:.4f}° (target: 0°)")
