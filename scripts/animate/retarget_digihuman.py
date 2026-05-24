#!/usr/bin/env python3
"""
DigiHuman-style retargeting for the ESL avatar.

Current scope:
  - Body: shoulders, upper arms, forearms (DigiHuman LookRotation algorithm)
  - Wrists: NOT IMPLEMENTED (stub)
  - Fingers: NOT IMPLEMENTED (stub)

Algorithm:
  1. SETUP: For each bone, cache InverseRotation = Inverse(LookRotation(rest_bone - rest_child, rest_up)) * rest_WORLD_rotation
  2. PER FRAME: world_rotation = LookRotation(bone_lm - child_lm, fv) * InverseRotation
     where fv is the elbow plane normal (cross of upper arm and forearm directions)
  3. Convert WORLD to LOCAL: local = inverse(parent_animated_world) * world_rotation

Usage:
  python3 retarget_digihuman.py <avatar.glb> <holistic.json> <output_anim.glb>
        [--smooth K] [--trim-trailing]
"""
import sys, json, struct, argparse, math
from pathlib import Path
import numpy as np
from pygltflib import GLTF2


# MediaPipe Pose landmark indices (BlazePose 33)
class LM:
    NOSE = 0
    L_SHOULDER = 11; R_SHOULDER = 12
    L_ELBOW = 13;    R_ELBOW = 14
    L_WRIST = 15;    R_WRIST = 16
    L_HIP = 23;      R_HIP = 24
    L_KNEE = 25;     R_KNEE = 26
    L_ANKLE = 27;    R_ANKLE = 28


# ============================================================================
# Quaternion helpers (Unity-compatible)
# ============================================================================

def quat_identity():
    return np.array([0.0, 0.0, 0.0, 1.0])

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_inverse(q):
    return quat_conj(q)

def quat_rotate_vec(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12: return quat_identity()
    return q / n

def slerp(q1, q2, t):
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2; dot = -dot
    if dot > 0.9995:
        return quat_normalize(q1 + t * (q2 - q1))
    theta_0 = math.acos(dot); sin_0 = math.sin(theta_0)
    theta = theta_0 * t; sin_t = math.sin(theta)
    s1 = math.cos(theta) - dot * sin_t / sin_0
    s2 = sin_t / sin_0
    return s1 * q1 + s2 * q2


def look_rotation(forward, upwards):
    """Unity-compatible LookRotation. Aligns +Z with 'forward', +Y as close to 'upwards' as possible.
    Returns quaternion (x, y, z, w)."""
    f = np.asarray(forward, dtype=np.float64)
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-9:
        return quat_identity()
    z_axis = f / f_norm
    u = np.asarray(upwards, dtype=np.float64)
    x_axis = np.cross(u, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        if abs(z_axis[1]) < 0.9:
            x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        else:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            return quat_identity()
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    m = np.column_stack([x_axis, y_axis, z_axis])
    return mat_to_quat(m)


def mat_to_quat(m):
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        w = (m[2,1] - m[1,2]) / s
        x = 0.25 * s
        y = (m[0,1] + m[1,0]) / s
        z = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        w = (m[0,2] - m[2,0]) / s
        x = (m[0,1] + m[1,0]) / s
        y = 0.25 * s
        z = (m[1,2] + m[2,1]) / s
    else:
        s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        w = (m[1,0] - m[0,1]) / s
        x = (m[0,2] + m[2,0]) / s
        y = (m[1,2] + m[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-9: return np.array([0.0, 0.0, 1.0])
    return n / norm


def mp_to_world(p):
    """MediaPipe pose_world (+X subject-left, +Y down, +Z away from camera)
    -> our world (+X subject-left, +Y up, +Z toward camera). Map: (x, -y, -z)."""
    return np.array([p[0], -p[1], -p[2]])


# ============================================================================
# Avatar rig
# ============================================================================

class AvatarRig:
    """Loads avatar GLB and provides bone hierarchy + rest pose info."""

    BODY_JOINTS = {
        'LeftArm':       (LM.L_SHOULDER,   'LeftArm'),
        'LeftForeArm':   (LM.L_ELBOW,      'LeftForeArm'),
        'LeftHand':      (LM.L_WRIST,      'LeftHand'),
        'RightArm':      (LM.R_SHOULDER,   'RightArm'),
        'RightForeArm':  (LM.R_ELBOW,      'RightForeArm'),
        'RightHand':     (LM.R_WRIST,      'RightHand'),
    }
    BODY_CHILDREN = {
        'LeftArm':       'LeftForeArm',
        'LeftForeArm':   'LeftHand',
        'RightArm':      'RightForeArm',
        'RightForeArm':  'RightHand',
    }
    BODY_PARENTS = {
        'LeftForeArm':   'LeftArm',
        'RightForeArm':  'RightArm',
    }

    def __init__(self, glb_path):
        self.g = GLTF2().load(glb_path)
        self.name_to_idx = {n.name: i for i, n in enumerate(self.g.nodes) if n.name}
        self.node_parent = {}
        for i, n in enumerate(self.g.nodes):
            for c in (n.children or []):
                self.node_parent[c] = i
        self.rest_local_q = {i: np.array(n.rotation or [0, 0, 0, 1.0]) for i, n in enumerate(self.g.nodes)}
        self.rest_world_q = {i: self._compute_world_q(i) for i in range(len(self.g.nodes))}
        self.rest_world_pos = {i: self._compute_world_pos(i) for i in range(len(self.g.nodes))}

        # Body forward direction at rest.
        # Use (spine, rhip, lhip) order so the normal points TOWARD the camera (+Z),
        # matching the avatar's chest direction.
        spine_pos = self.world_pos('Spine')
        lhip_pos = self.world_pos('LeftUpLeg')
        rhip_pos = self.world_pos('RightUpLeg')
        self.body_forward_rest = triangle_normal(spine_pos, rhip_pos, lhip_pos)

        # Cache InverseRotation per arm bone using the elbow axis as 'up' reference.
        self.bone_inverse_rot = {}
        for joint_name, (lm_idx, bone_name) in self.BODY_JOINTS.items():
            if bone_name not in self.name_to_idx:
                continue
            child_joint = self.BODY_CHILDREN.get(joint_name)
            if child_joint is None or self.BODY_JOINTS[child_joint][1] not in self.name_to_idx:
                continue
            bone_pos = self.world_pos(bone_name)
            child_pos = self.world_pos(self.BODY_JOINTS[child_joint][1])
            forward = bone_pos - child_pos
            up = self._compute_elbow_axis(joint_name)
            if up is None or np.linalg.norm(up) < 1e-6:
                up = self.body_forward_rest
            inv_rot = quat_inverse(look_rotation(forward, up))
            bone_idx = self.name_to_idx[bone_name]
            self.bone_inverse_rot[bone_name] = quat_mul(inv_rot, self.rest_world_q[bone_idx])

    def _compute_elbow_axis(self, joint_name):
        """Return the rest elbow plane normal (cross of upper arm and forearm)."""
        if 'Left' in joint_name:
            sh = self.world_pos('LeftArm')
            el = self.world_pos('LeftForeArm')
            wr = self.world_pos('LeftHand')
        elif 'Right' in joint_name:
            sh = self.world_pos('RightArm')
            el = self.world_pos('RightForeArm')
            wr = self.world_pos('RightHand')
        else:
            return None
        if sh is None or el is None or wr is None:
            return None
        axis = np.cross(el - sh, wr - el)
        n = np.linalg.norm(axis)
        if n < 1e-6:
            return None
        return axis / n

    def world_pos(self, bone_name):
        if bone_name not in self.name_to_idx: return None
        return self.rest_world_pos[self.name_to_idx[bone_name]]

    def _compute_world_q(self, idx):
        chain = []; cur = idx
        while cur is not None: chain.append(cur); cur = self.node_parent.get(cur)
        chain.reverse()
        q = quat_identity()
        for c in chain:
            q = quat_mul(q, self.rest_local_q[c])
        return q

    def _compute_world_pos(self, idx):
        chain = []; cur = idx
        while cur is not None: chain.append(cur); cur = self.node_parent.get(cur)
        chain.reverse()
        pos = np.zeros(3); R = quat_identity()
        for c in chain:
            t = np.array(self.g.nodes[c].translation or [0, 0, 0])
            pos = pos + quat_rotate_vec(R, t)
            R = quat_mul(R, self.rest_local_q[c])
        return pos


# ============================================================================
# Body solver
# ============================================================================

def get_body_landmark(pose, joint_name):
    """Get a 3D world position for a body joint from MP pose_world landmarks."""
    pw = np.array(pose)
    arr_w = np.array([mp_to_world(pw[i, :3]) for i in range(len(pw))])

    if joint_name == 'LeftArm':       return arr_w[LM.L_SHOULDER]
    if joint_name == 'LeftForeArm':   return arr_w[LM.L_ELBOW]
    if joint_name == 'LeftHand':      return arr_w[LM.L_WRIST]
    if joint_name == 'RightArm':      return arr_w[LM.R_SHOULDER]
    if joint_name == 'RightForeArm':  return arr_w[LM.R_ELBOW]
    if joint_name == 'RightHand':     return arr_w[LM.R_WRIST]
    return None


def solve_body_frame(pose, rig, vis_threshold=0.5):
    """Solve LOCAL rotations for body bones (arms + forearms).
    
    Skips per-side retargeting when the side's elbow/wrist visibility is below threshold.
    For skipped bones, returns the rest LOCAL rotation (so the arm stays in A-pose / resting).
    
    Returns (local_rotations_by_bone, animated_world_quats_by_joint).
    """
    out = {}
    animated_world_q = {}
    if pose is None:
        return out, animated_world_q

    pw = np.array(pose)
    arr_w = np.array([mp_to_world(pw[i, :3]) for i in range(len(pw))])

    # Check per-side visibility: only retarget the arm if the elbow AND wrist are visible.
    # Visibility is the 4th column when present.
    has_vis = pw.shape[1] >= 4
    def side_visible(side):
        if not has_vis: return True
        if side == 'L':
            return pw[LM.L_ELBOW, 3] > vis_threshold and pw[LM.L_WRIST, 3] > vis_threshold
        else:
            return pw[LM.R_ELBOW, 3] > vis_threshold and pw[LM.R_WRIST, 3] > vis_threshold
    left_visible = side_visible('L')
    right_visible = side_visible('R')

    # Compute elbow axes for each side (cross of upper arm and forearm)
    def elbow_axis_for(side):
        if side == 'L':
            sh = arr_w[LM.L_SHOULDER]; el = arr_w[LM.L_ELBOW]; wr = arr_w[LM.L_WRIST]
        else:
            sh = arr_w[LM.R_SHOULDER]; el = arr_w[LM.R_ELBOW]; wr = arr_w[LM.R_WRIST]
        ax = np.cross(el - sh, wr - el)
        n = np.linalg.norm(ax)
        if n < 1e-5: return None
        return ax / n

    left_elbow_axis = elbow_axis_for('L')
    right_elbow_axis = elbow_axis_for('R')

    BODY_ORDER = ['LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm']

    for joint_name in BODY_ORDER:
        bone_name = rig.BODY_JOINTS[joint_name][1]
        if bone_name not in rig.name_to_idx:
            continue
        # Skip this side's arm bones if the elbow/wrist isn't visible — keep them in rest pose.
        if 'Left' in joint_name and not left_visible: continue
        if 'Right' in joint_name and not right_visible: continue

        child_joint = rig.BODY_CHILDREN.get(joint_name)
        child_bone = rig.BODY_JOINTS[child_joint][1]
        if child_bone not in rig.name_to_idx:
            continue

        bone_lm = get_body_landmark(pose, joint_name)
        child_lm = get_body_landmark(pose, child_joint)
        if bone_lm is None or child_lm is None:
            continue

        if 'Left' in joint_name:
            fv = left_elbow_axis
        else:
            fv = right_elbow_axis
        if fv is None or np.linalg.norm(fv) < 1e-6:
            fv = rig.body_forward_rest

        forward = bone_lm - child_lm
        if np.linalg.norm(forward) < 1e-6:
            continue

        inv_rot = rig.bone_inverse_rot.get(bone_name)
        if inv_rot is None:
            continue

        anim_world_q = quat_mul(look_rotation(forward, fv), inv_rot)
        animated_world_q[joint_name] = anim_world_q

        # Convert to LOCAL for glTF animation track
        bone_idx = rig.name_to_idx[bone_name]
        par_idx = rig.node_parent.get(bone_idx)
        parent_anim_world = quat_identity()
        if par_idx is not None:
            par_name = rig.g.nodes[par_idx].name
            for jn, (_, bn) in rig.BODY_JOINTS.items():
                if bn == par_name and jn in animated_world_q:
                    parent_anim_world = animated_world_q[jn]
                    break
            else:
                parent_anim_world = rig.rest_world_q.get(par_idx, quat_identity())

        local_q = quat_mul(quat_inverse(parent_anim_world), anim_world_q)
        out[bone_name] = quat_normalize(local_q)

    return out, animated_world_q


# ============================================================================
# Hand solver — wrist only (DigiHuman-style)
# ============================================================================

# MediaPipe hand landmark indices
class HM:
    WRIST = 0
    THUMB1 = 1;  THUMB2 = 2;  THUMB3 = 3;  THUMB4 = 4
    INDEX1 = 5;  INDEX2 = 6;  INDEX3 = 7;  INDEX4 = 8
    MIDDLE1 = 9; MIDDLE2 = 10; MIDDLE3 = 11; MIDDLE4 = 12
    RING1 = 13;  RING2 = 14;  RING3 = 15;  RING4 = 16
    PINKY1 = 17; PINKY2 = 18; PINKY3 = 19; PINKY4 = 20


def solve_hand_frame(pose, hand_lm, side, rig, forearm_anim_world_q=None, hand_lm_img=None):
    """Solve the wrist (Hand bone) rotation + finger bones.
    
    Wrist orientation: from hand_world_landmarks (reliable MCP positions).
    Finger directions: from hand_lm_img (image-relative 2D), projected through the
    solved wrist orientation onto the palm plane. The 2D image positions are MORE
    reliable than hand_world Z-depth for finger curl.
    """
    out = {}
    if hand_lm is None or pose is None:
        return out
    h = np.asarray(hand_lm)
    if h.shape[0] < 21:
        return out

    # Convert hand world landmarks via axis flip (same as point cloud renderer).
    h_world = np.array([mp_to_world(h[k]) for k in range(len(h))])

    # Build OBSERVED hand frame in world space:
    #   forward = wrist -> middle_MCP (palm direction)
    #   palm normal = cross(wrist->index_MCP, wrist->pinky_MCP)
    obs_forward = h_world[HM.MIDDLE1] - h_world[HM.WRIST]
    if np.linalg.norm(obs_forward) < 1e-6: return out
    obs_palm_normal = triangle_normal(h_world[HM.WRIST], h_world[HM.INDEX1], h_world[HM.PINKY1])

    # Build REST hand frame from avatar rest world positions:
    hand_bone = f'{side}Hand'
    if hand_bone not in rig.name_to_idx: return out
    wrist_rest = rig.world_pos(hand_bone)
    index_rest = rig.world_pos(f'{side}HandIndex1')
    pinky_rest = rig.world_pos(f'{side}HandPinky1')
    middle_rest = rig.world_pos(f'{side}HandMiddle1')
    if any(p is None for p in [wrist_rest, index_rest, pinky_rest, middle_rest]):
        return out
    rest_forward = middle_rest - wrist_rest
    if np.linalg.norm(rest_forward) < 1e-6: return out
    rest_palm_normal = triangle_normal(wrist_rest, index_rest, pinky_rest)

    # DigiHuman-style wrist rotation:
    #   world_rotation = LookRotation(obs_forward, obs_palm_normal) * InverseRotation
    # where InverseRotation = Inverse(LookRotation(rest_forward, rest_palm_normal)) * rest_WORLD_rotation
    inv_rot = quat_mul(quat_inverse(look_rotation(rest_forward, rest_palm_normal)),
                       rig.rest_world_q[rig.name_to_idx[hand_bone]])
    wrist_world_q = quat_mul(look_rotation(obs_forward, obs_palm_normal), inv_rot)

    # Convert to LOCAL using the animated ForeArm's world quaternion (passed in).
    forearm_bone = f'{side}ForeArm'
    if forearm_anim_world_q is not None:
        parent_world = forearm_anim_world_q
    elif forearm_bone in rig.name_to_idx:
        parent_world = rig.rest_world_q[rig.name_to_idx[forearm_bone]]
    else:
        parent_world = quat_identity()
    wrist_local_q = quat_mul(quat_inverse(parent_world), wrist_world_q)
    out[hand_bone] = quat_normalize(wrist_local_q)

    # ============================================================
    # FINGERS — same DigiHuman algorithm, per joint, hierarchical
    # ============================================================
    # Track animated world quaternions for the hierarchy walk
    animated_world = {'Wrist': wrist_world_q}

    finger_chains = [
        ['Thumb1',  'Thumb2',  'Thumb3'],
        ['Index1',  'Index2',  'Index3'],
        ['Middle1', 'Middle2', 'Middle3'],
        ['Ring1',   'Ring2',   'Ring3'],
        ['Pinky1',  'Pinky2',  'Pinky3'],
    ]
    finger_lm_idx = {
        'Thumb1': HM.THUMB1, 'Thumb2': HM.THUMB2, 'Thumb3': HM.THUMB3, 'Thumb4': HM.THUMB4,
        'Index1': HM.INDEX1, 'Index2': HM.INDEX2, 'Index3': HM.INDEX3, 'Index4': HM.INDEX4,
        'Middle1': HM.MIDDLE1, 'Middle2': HM.MIDDLE2, 'Middle3': HM.MIDDLE3, 'Middle4': HM.MIDDLE4,
        'Ring1': HM.RING1, 'Ring2': HM.RING2, 'Ring3': HM.RING3, 'Ring4': HM.RING4,
        'Pinky1': HM.PINKY1, 'Pinky2': HM.PINKY2, 'Pinky3': HM.PINKY3, 'Pinky4': HM.PINKY4,
    }
    tip_lm_idx = {
        'Thumb3': HM.THUMB4, 'Index3': HM.INDEX4, 'Middle3': HM.MIDDLE4,
        'Ring3': HM.RING4, 'Pinky3': HM.PINKY4,
    }

    for chain in finger_chains:
        for i, finger_name in enumerate(chain):
            bone_name = f'{side}Hand{finger_name}'
            if bone_name not in rig.name_to_idx: continue

            # Child landmark index in hand_world_landmarks
            if i < len(chain) - 1:
                child_lm = finger_lm_idx[chain[i + 1]]
            else:
                tip = tip_lm_idx.get(finger_name)
                if tip is None: continue
                child_lm = tip

            this_lm = finger_lm_idx[finger_name]
            obs_forward = h_world[this_lm] - h_world[child_lm]
            if np.linalg.norm(obs_forward) < 1e-6: continue

            # 'up' reference: same palm normal as wrist (stable across all fingers)
            obs_up = obs_palm_normal

            # Rest positions for THIS finger bone and its child
            bone_rest_pos = rig.world_pos(bone_name)
            if bone_rest_pos is None: continue
            if i < len(chain) - 1:
                child_rest_pos = rig.world_pos(f'{side}Hand{chain[i+1]}')
            else:
                tip_name = {'Thumb3': 'Thumb4', 'Index3': 'Index4', 'Middle3': 'Middle4',
                            'Ring3': 'Ring4', 'Pinky3': 'Pinky4'}.get(finger_name)
                child_rest_pos = rig.world_pos(f'{side}Hand{tip_name}') if tip_name else None
            if child_rest_pos is None: continue

            rest_forward = bone_rest_pos - child_rest_pos
            if np.linalg.norm(rest_forward) < 1e-6: continue
            rest_up = rest_palm_normal  # same palm normal as wrist at rest

            # DigiHuman: world_rotation = LookRotation(obs_fwd, obs_up) * Inverse(LookRotation(rest_fwd, rest_up)) * rest_world
            finger_inv_rot = quat_mul(quat_inverse(look_rotation(rest_forward, rest_up)),
                                       rig.rest_world_q[rig.name_to_idx[bone_name]])
            finger_world_q = quat_mul(look_rotation(obs_forward, obs_up), finger_inv_rot)
            animated_world[finger_name] = finger_world_q

            # Convert to LOCAL using the animated parent
            if i > 0:
                parent_anim_world = animated_world.get(chain[i - 1], quat_identity())
            else:
                parent_anim_world = animated_world.get('Wrist', quat_identity())

            local_q = quat_mul(quat_inverse(parent_anim_world), finger_world_q)
            out[bone_name] = quat_normalize(local_q)

    return out


# ============================================================================
# Smoothing
# ============================================================================

def smooth_quats(seq, window=5):
    if window <= 1: return seq
    out = []
    n = len(seq)
    for i in range(n):
        neighbors = [seq[j] for j in range(max(0, i - window // 2), min(n, i + window // 2 + 1)) if seq[j] is not None]
        if not neighbors:
            out.append(None); continue
        acc = neighbors[0]
        for k, q in enumerate(neighbors[1:], start=2):
            acc = slerp(acc, q, 1.0 / k)
        out.append(acc)
    return out


def forward_fill(seq):
    last = None; out = []
    for q in seq:
        out.append(q if q is not None else (last if last is not None else quat_identity()))
        if q is not None: last = q
    return out


def has_signal(pose, wrist_thresh=0.5):
    if pose is None: return False
    p = np.asarray(pose)
    if p.shape[1] >= 4:
        return bool(p[LM.L_WRIST, 3] > wrist_thresh or p[LM.R_WRIST, 3] > wrist_thresh)
    return True


# ============================================================================
# GLB writer
# ============================================================================

def write_anim_glb(bone_tracks, fps, out_path):
    n_frames = max(len(v) for v in bone_tracks.values())
    times = np.arange(n_frames, dtype=np.float32) / fps

    binary = bytearray()
    def append_array(arr):
        nonlocal binary
        if (len(binary) % 4) != 0:
            binary += b'\x00' * (4 - (len(binary) % 4))
        offset = len(binary)
        data = arr.astype(np.float32).tobytes()
        binary += data
        return offset, len(data)

    accessors, buffer_views = [], []
    t_off, t_len = append_array(times)
    buffer_views.append({'buffer': 0, 'byteOffset': t_off, 'byteLength': t_len})
    accessors.append({'bufferView': 0, 'componentType': 5126, 'count': n_frames,
                      'type': 'SCALAR', 'min': [0.0], 'max': [float(times[-1])]})

    nodes, bone_to_idx = [], {}
    for i, bone in enumerate(bone_tracks.keys()):
        nodes.append({'name': bone}); bone_to_idx[bone] = i
    nodes.append({'name': 'Root', 'children': list(range(len(bone_tracks)))})
    scenes = [{'name': 'Scene', 'nodes': [len(nodes) - 1]}]

    channels, samplers = [], []
    for bone, quats in bone_tracks.items():
        arr = np.array(quats, dtype=np.float32)
        offset, length = append_array(arr)
        bv = len(buffer_views)
        buffer_views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': length})
        acc = len(accessors)
        accessors.append({'bufferView': bv, 'componentType': 5126, 'count': n_frames, 'type': 'VEC4'})
        s = len(samplers)
        samplers.append({'input': 0, 'interpolation': 'LINEAR', 'output': acc})
        channels.append({'sampler': s, 'target': {'node': bone_to_idx[bone], 'path': 'rotation'}})

    animations = [{'name': 'ESL_DigiHuman', 'channels': channels, 'samplers': samplers}]
    while len(binary) % 4 != 0: binary += b'\x00'

    g = {
        'asset': {'version': '2.0', 'generator': 'retarget_digihuman.py'},
        'scene': 0, 'scenes': scenes, 'nodes': nodes,
        'accessors': accessors, 'bufferViews': buffer_views,
        'buffers': [{'byteLength': len(binary)}], 'animations': animations,
    }
    json_bytes = json.dumps(g, separators=(',', ':')).encode('utf-8')
    while len(json_bytes) % 4 != 0: json_bytes += b' '
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    out = bytearray()
    out += b'glTF' + struct.pack('<I', 2) + struct.pack('<I', total)
    out += struct.pack('<I', len(json_bytes)) + b'JSON' + json_bytes
    out += struct.pack('<I', len(binary)) + b'BIN\x00' + binary
    Path(out_path).write_bytes(out)
    print(f"✅ Wrote {out_path} ({total} bytes, {len(bone_tracks)} bones, {n_frames} frames)")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('avatar')
    ap.add_argument('mocap')
    ap.add_argument('output_glb')
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--trim-trailing', action='store_true')
    ap.add_argument('--skip-body', action='store_true')
    args = ap.parse_args()

    print(f"Loading avatar: {args.avatar}")
    rig = AvatarRig(args.avatar)

    print(f"Loading mocap: {args.mocap}")
    with open(args.mocap) as f:
        data = json.load(f)
    fps = float(data['fps'])
    raw_frames = data['frames']

    poses = [np.array(f['pose']) if f.get('pose') else None for f in raw_frames]

    if args.trim_trailing:
        end = len(poses)
        while end > 0 and (poses[end-1] is None or not has_signal(poses[end-1])):
            end -= 1
        start = 0
        while start < end and (poses[start] is None or not has_signal(poses[start])):
            start += 1
        start = max(0, start - 3)
        end = min(len(poses), end + 3)
        poses = poses[start:end]
        raw_frames = raw_frames[start:end]
        print(f"  Trimmed to {len(poses)} active frames ({start}..{end})")

    lh = [np.array(f['lh']) if f.get('lh') else None for f in raw_frames]
    rh = [np.array(f['rh']) if f.get('rh') else None for f in raw_frames]
    lh_img_list = [np.array(f['lh_img']) if f.get('lh_img') else None for f in raw_frames]
    rh_img_list = [np.array(f['rh_img']) if f.get('rh_img') else None for f in raw_frames]

    if args.skip_body:
        solved_body = [{} for _ in poses]
        body_anim_worlds = [{} for _ in poses]
    else:
        body_results = [solve_body_frame(p, rig) if p is not None and has_signal(p) else ({}, {}) for p in poses]
        solved_body = [r[0] for r in body_results]
        body_anim_worlds = [r[1] for r in body_results]

    solved_lh, solved_rh = [], []
    for i, (p, l, r) in enumerate(zip(poses, lh, rh)):
        l_fa = body_anim_worlds[i].get('LeftForeArm') if i < len(body_anim_worlds) else None
        r_fa = body_anim_worlds[i].get('RightForeArm') if i < len(body_anim_worlds) else None
        l_img = lh_img_list[i] if i < len(lh_img_list) else None
        r_img = rh_img_list[i] if i < len(rh_img_list) else None
        solved_lh.append(solve_hand_frame(p, l, 'Left', rig, l_fa, l_img) if p is not None and has_signal(p) else {})
        solved_rh.append(solve_hand_frame(p, r, 'Right', rig, r_fa, r_img) if p is not None and has_signal(p) else {})

    n = len(solved_body)
    all_bones = set()
    for fr in solved_body + solved_lh + solved_rh:
        all_bones.update(fr.keys())
    tracks_raw = {b: [None] * n for b in all_bones}
    for src in (solved_body, solved_lh, solved_rh):
        for i, fr in enumerate(src):
            for b, q in fr.items(): tracks_raw[b][i] = q

    # Fill None slots with the bone's REST LOCAL rotation so the avatar returns to
    # its natural resting pose (A-pose for RPM models) when MediaPipe can't see the limb.
    tracks = {}
    for b, seq in tracks_raw.items():
        if b in rig.name_to_idx:
            rest_local = rig.rest_local_q[rig.name_to_idx[b]]
        else:
            rest_local = quat_identity()
        seq = [s if s is not None else rest_local for s in seq]
        seq = smooth_quats(seq, args.smooth)
        seq = [s if s is not None else rest_local for s in seq]
        tracks[b] = seq

    print(f"Animated bones ({len(tracks)}): {sorted(tracks.keys())}")
    write_anim_glb(tracks, fps, args.output_glb)


if __name__ == '__main__':
    main()
