#!/usr/bin/env python3
"""
Retarget v2 — Hierarchical, avatar-aware.

Key improvements over v1:
  1. Reads the AVATAR'S actual rest pose to determine each bone's rest world direction.
     (Critical for A-pose rigs like Ready Player Me, where arms aren't sideways +X.)
  2. Computes WORLD rotations for each bone, then converts to LOCAL by
     accounting for the parent's accumulated world rotation.
  3. Skips Spine retargeting by default (user noted spine shouldn't move).
  4. Properly solves LeftHand / RightHand orientation from the forearm->wrist direction
     plus a perpendicular constraint (palm normal from hand landmarks).

Usage:
  python3 retarget_v2.py <avatar.glb> <holistic.json> <output_anim.glb>
        [--smooth K] [--trim-trailing] [--include-spine] [--include-legs]
"""

import sys, json, struct, argparse, math
from pathlib import Path
import numpy as np
from pygltflib import GLTF2


# MediaPipe Pose landmark indices
LM = {
    'nose': 0,
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24,
    'l_knee': 25, 'r_knee': 26,
    'l_ankle': 27, 'r_ankle': 28,
}

# MediaPipe Hands landmark indices
H = {
    'wrist': 0,
    'thumb1': 1, 'thumb2': 2, 'thumb3': 3, 'thumb4': 4,
    'index1': 5, 'index2': 6, 'index3': 7, 'index4': 8,
    'middle1': 9, 'middle2': 10, 'middle3': 11, 'middle4': 12,
    'ring1': 13, 'ring2': 14, 'ring3': 15, 'ring4': 16,
    'pinky1': 17, 'pinky2': 18, 'pinky3': 19, 'pinky4': 20,
}

# Bone -> (head_landmark, tail_landmark) for body
# Shoulder bones (clavicles) intentionally left in rest pose -- forcing them to
# match mid_shoulder -> shoulder produces a horizontal arm splay that ruins the
# downstream arm/forearm placement.
BODY_BONES_UPPER = [
    ('LeftArm',       'l_shoulder',   'l_elbow'),
    ('LeftForeArm',   'l_elbow',      'l_wrist'),
    ('RightArm',      'r_shoulder',   'r_elbow'),
    ('RightForeArm',  'r_elbow',      'r_wrist'),
]
BODY_BONES_NECK = [
    ('Neck',          'mid_shoulder', 'head_top'),
]
BODY_BONES_SPINE = [
    ('Spine', 'mid_hip', 'mid_shoulder'),
]
BODY_BONES_LEGS = [
    ('LeftUpLeg',  'l_hip',  'l_knee'),
    ('LeftLeg',    'l_knee', 'l_ankle'),
    ('RightUpLeg', 'r_hip',  'r_knee'),
    ('RightLeg',   'r_knee', 'r_ankle'),
]

# Finger bones: (short, head_idx_in_hand, tail_idx_in_hand)
FINGER_BONES = [
    ('Thumb1',  H['wrist'],   H['thumb1']),
    ('Thumb2',  H['thumb1'],  H['thumb2']),
    ('Thumb3',  H['thumb2'],  H['thumb3']),
    ('Index1',  H['wrist'],   H['index1']),
    ('Index2',  H['index1'],  H['index2']),
    ('Index3',  H['index2'],  H['index3']),
    ('Middle1', H['wrist'],   H['middle1']),
    ('Middle2', H['middle1'], H['middle2']),
    ('Middle3', H['middle2'], H['middle3']),
    ('Ring1',   H['wrist'],   H['ring1']),
    ('Ring2',   H['ring1'],   H['ring2']),
    ('Ring3',   H['ring2'],   H['ring3']),
    ('Pinky1',  H['wrist'],   H['pinky1']),
    ('Pinky2',  H['pinky1'],  H['pinky2']),
    ('Pinky3',  H['pinky2'],  H['pinky3']),
]


# ----------------------------- Quaternion math ------------------------------

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


def quat_rotate_vec(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def quat_from_two_vectors(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return quat_identity()
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    s = math.sqrt((1.0 + dot) * 2.0)
    inv = 1.0 / s
    return np.array([axis[0] * inv, axis[1] * inv, axis[2] * inv, s * 0.5])


def quat_from_axis_angle(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle*0.5)])


def slerp(q1, q2, t):
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2; dot = -dot
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = math.acos(dot); sin_0 = math.sin(theta_0)
    theta = theta_0 * t; sin_t = math.sin(theta)
    s1 = math.cos(theta) - dot * sin_t / sin_0
    s2 = sin_t / sin_0
    return s1 * q1 + s2 * q2


# ----------------------------- Avatar rest pose -----------------------------

class AvatarRest:
    """Parse a GLB avatar, expose:
       - bone parent chain
       - bone rest LOCAL rotation (quaternion)
       - bone rest WORLD rotation (quaternion)
       - bone rest WORLD direction (where +Y_local points in world)
    """
    def __init__(self, gltf_path: str):
        self.g = GLTF2().load(gltf_path)
        self.name_to_idx = {n.name: i for i, n in enumerate(self.g.nodes) if n.name}
        self.idx_to_name = {i: n.name for n, i in zip(self.g.nodes, range(len(self.g.nodes))) if n.name}
        # Build parent map
        self.parent_of = {}
        for i, n in enumerate(self.g.nodes):
            for c in (n.children or []):
                self.parent_of[c] = i
        # Cache rest local rotation per node
        self.rest_local_quat = {}
        for i, n in enumerate(self.g.nodes):
            self.rest_local_quat[i] = np.array(n.rotation or [0, 0, 0, 1], dtype=np.float64)
        # Cache rest WORLD rotation per node (root-down accumulation)
        self.rest_world_quat = {}
        for i in range(len(self.g.nodes)):
            self.rest_world_quat[i] = self._compute_world_quat(i)

    def _compute_world_quat(self, idx):
        # World = R_root * R_p1 * ... * R_self.
        # Build chain root->self
        chain = []
        cur = idx
        while cur is not None:
            chain.append(cur)
            cur = self.parent_of.get(cur)
        chain.reverse()
        q = quat_identity()
        for c in chain:
            q = quat_mul(q, self.rest_local_quat[c])
        return q

    def world_dir_local_y(self, bone_name: str) -> np.ndarray:
        """Where does this bone's local +Y axis point in world space at rest?"""
        idx = self.name_to_idx[bone_name]
        q = self.rest_world_quat[idx]
        return quat_rotate_vec(q, np.array([0.0, 1.0, 0.0]))

    def parent_world_quat(self, bone_name: str) -> np.ndarray:
        idx = self.name_to_idx[bone_name]
        p = self.parent_of.get(idx)
        if p is None:
            return quat_identity()
        return self.rest_world_quat[p]


# ----------------------------- Landmark helpers ------------------------------

def mp_to_gltf(p: np.ndarray) -> np.ndarray:
    """MediaPipe pose_world (+X subject-left, +Y down, +Z away) -> glTF (+Y up, +Z toward camera)."""
    return np.array([p[0], -p[1], -p[2]])


def get_landmark(pose: np.ndarray, key: str) -> np.ndarray:
    if key in LM:
        return mp_to_gltf(pose[LM[key]][:3])
    if key == 'mid_hip':
        return 0.5 * (mp_to_gltf(pose[LM['l_hip']][:3]) + mp_to_gltf(pose[LM['r_hip']][:3]))
    if key == 'mid_shoulder':
        return 0.5 * (mp_to_gltf(pose[LM['l_shoulder']][:3]) + mp_to_gltf(pose[LM['r_shoulder']][:3]))
    if key == 'head_top':
        nose = mp_to_gltf(pose[LM['nose']][:3])
        ls = mp_to_gltf(pose[LM['l_shoulder']][:3])
        rs = mp_to_gltf(pose[LM['r_shoulder']][:3])
        mid_sh = 0.5 * (ls + rs)
        return nose + 0.5 * (nose - mid_sh)
    raise KeyError(key)


def has_signal(pose, wrist_vis_threshold=0.5) -> bool:
    """True if at least one wrist is reliably tracked (i.e. visible in the source frame).
    Without a visible wrist, MediaPipe hallucinates a low resting position and we get garbage."""
    if pose is None: return False
    p = np.asarray(pose)
    if p.shape[1] >= 4:
        return bool(p[LM['l_wrist'], 3] > wrist_vis_threshold or p[LM['r_wrist'], 3] > wrist_vis_threshold)
    return True


# ----------------------------- The actual retargeter ------------------------

def solve_bone_world_rot(rest_dir_world: np.ndarray, target_dir_world: np.ndarray) -> np.ndarray:
    """Return the WORLD quaternion that rotates rest_dir_world to target_dir_world."""
    return quat_from_two_vectors(rest_dir_world, target_dir_world)


def world_to_local(world_q: np.ndarray, parent_world_q: np.ndarray, rest_local_q: np.ndarray) -> np.ndarray:
    """Given a desired world rotation for a bone, compute the local rotation it should
    have, such that parent_world * local = world_q * rest_world.
    
    The bone's rest world = parent_world * rest_local.
    The bone's animated world = world_q * rest_world  (apply delta in world).
    We want: parent_world * local = world_q * parent_world * rest_local
        => local = parent_world^-1 * world_q * parent_world * rest_local
    """
    pwq_inv = quat_conj(parent_world_q)
    return quat_mul(pwq_inv, quat_mul(world_q, quat_mul(parent_world_q, rest_local_q)))


def solve_body_frame(pose, avatar: AvatarRest, include_spine: bool, include_legs: bool, include_neck: bool = False) -> dict:
    out = {}
    bones = list(BODY_BONES_UPPER)
    if include_spine: bones = BODY_BONES_SPINE + bones
    if include_neck: bones = bones + BODY_BONES_NECK
    if include_legs: bones = bones + BODY_BONES_LEGS

    for bone, head_key, tail_key in bones:
        if bone not in avatar.name_to_idx:
            continue
        try:
            head = get_landmark(pose, head_key)
            tail = get_landmark(pose, tail_key)
        except Exception:
            continue
        target_dir = tail - head
        if np.linalg.norm(target_dir) < 1e-5:
            continue
        # Bone's rest world direction = +Y_local rotated by rest world quat
        rest_dir = avatar.world_dir_local_y(bone)
        # World delta
        world_delta = solve_bone_world_rot(rest_dir, target_dir)
        # Convert to local
        idx = avatar.name_to_idx[bone]
        rest_local = avatar.rest_local_quat[idx]
        parent_world = avatar.parent_world_quat(bone)
        local = world_to_local(world_delta, parent_world, rest_local)
        out[bone] = local / (np.linalg.norm(local) + 1e-12)
    return out


def solve_hand_orientation(pose, side: str, avatar: AvatarRest) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (palm_x, palm_y, palm_z) world-space basis of the hand based on the pose:
        palm_y = elbow->wrist direction (where the hand points)
        palm_x = perpendicular (across the palm, thumb side)
        palm_z = palm normal
    
    We can't get palm orientation from pose alone — we'd need hand landmarks.
    Caller will use hand landmarks separately.
    Returns None if not enough info.
    """
    return None  # not used; see solve_hand_frame_v2 below


def solve_hand_frame_v2(pose, hand_lm, side: str, avatar: AvatarRest) -> dict:
    """Solve LeftHand/RightHand orientation + finger bones.

    Strategy (revised after diagnosing depth-z noise in MediaPipe hand landmarks):
      1. WRIST ORIENTATION:
         - Forward axis (Y_obs): elbow -> wrist  (from pose_world, reliable depth)
         - Palm normal: estimated from forearm direction + anatomical default
           (palm faces the body midline by default). When hand image landmarks are
           available, refine palm normal by 2D image vector index_mcp->pinky_mcp
           projected to the plane perpendicular to the forearm.
         - 3-axis rotation match (Kabsch-style) maps avatar rest hand frame to
           this observed hand frame.
      2. FINGERS:
         - Use 2D image landmarks (X, Y only; ignore unreliable Z).
         - Express each finger bone's direction in the hand-local 2D plane that
           is perpendicular to the forearm in image space.
         - This produces correct in-plane finger flexion (curl) without relying
           on the noisy depth axis.
    """
    out = {}
    if hand_lm is None: return out
    h = np.asarray(hand_lm)
    if h.shape[0] < 21: return out

    # ---- Compute observed hand frame in WORLD (glTF) space ----
    # Forward axis = elbow -> wrist from pose_world (reliable).
    side_key = 'l' if side == 'Left' else 'r'
    try:
        elbow_w = get_landmark(pose, f'{side_key}_elbow')
        wrist_w = get_landmark(pose, f'{side_key}_wrist')
        mid_hip = get_landmark(pose, 'mid_hip')
        mid_sh = get_landmark(pose, 'mid_shoulder')
    except Exception:
        return out
    y_obs = wrist_w - elbow_w
    if np.linalg.norm(y_obs) < 1e-6: return out
    y_obs /= np.linalg.norm(y_obs)

    # Palm normal default: palm faces the body / away from the camera.
    # The signer's chest faces the camera, so palms-flat-against-chest means palm
    # normal points AWAY from the viewer (-Z_gltf).
    palm_normal_default = np.array([0.0, 0.0, -1.0])
    x_obs = np.cross(y_obs, palm_normal_default)
    if np.linalg.norm(x_obs) < 1e-6:
        x_obs = np.cross(y_obs, np.array([0.0, 1.0, 0.0]))
    x_obs /= np.linalg.norm(x_obs)
    # For the LEFT hand with palm facing body:
    #   Anatomically thumb is on the INNER side (toward body midline).
    #   When palm faces body, looking at the back of the hand: thumb stays on inner side.
    #   For LEFT hand: inner side = body-RIGHT = -X_gltf direction.
    #   So pinky_mcp -> index_mcp = -X direction.
    #   We want x_obs to be in -X direction for LEFT hand.
    # cross(y_obs, -Z) with y_obs mostly +Y gives +X, so we NEGATE for LEFT.
    if side == 'Left':
        x_obs = -x_obs
    z_obs = np.cross(x_obs, y_obs)
    z_obs /= (np.linalg.norm(z_obs) + 1e-9)

    # Optionally refine palm twist using 2D hand landmarks (image x, y only).
    # Compute the angle of (index_mcp - pinky_mcp) in the image plane relative to
    # the forearm direction, and rotate the (x_obs, z_obs) basis by that angle around y_obs.
    # This captures the visible "hand rotated about forearm" motion that the body-up
    # default would miss.
    try:
        i_lm = h[H['index1']]
        p_lm = h[H['pinky1']]
        # 2D vector in image (use X right, Y up to match world after flip)
        v2 = np.array([i_lm[0] - p_lm[0], -(i_lm[1] - p_lm[1]), 0.0])
        # Project x_obs and z_obs to the image plane (just take X and Y components, since
        # in this video setup the camera is roughly aligned with -Z_glTF axis -> image plane is XY).
        x_img = np.array([x_obs[0], x_obs[1], 0.0])
        z_img = np.array([z_obs[0], z_obs[1], 0.0])
        if np.linalg.norm(x_img) > 1e-4 and np.linalg.norm(v2) > 1e-4:
            x_img /= np.linalg.norm(x_img); z_img /= np.linalg.norm(z_img)
            v2 /= np.linalg.norm(v2)
            # Angle of v2 in (x_img, z_img) basis
            cos_a = float(np.dot(v2, x_img))
            sin_a = float(np.dot(v2, z_img))
            angle = math.atan2(sin_a, cos_a)
            # Rotate x_obs and z_obs around y_obs by angle
            twist = quat_from_axis_angle(y_obs, angle)
            x_obs = quat_rotate_vec(twist, x_obs)
            z_obs = quat_rotate_vec(twist, z_obs)
    except Exception:
        pass

    # ---- Build the avatar's REST hand frame in WORLD ----
    # Same construction as observed frame: y = wrist->middle_mcp, x = pinky->index, z = x cross y.
    hand_bone = f"{side}Hand"
    if hand_bone not in avatar.name_to_idx: return out
    hand_idx = avatar.name_to_idx[hand_bone]

    def avatar_world_pos(bone_name):
        idx = avatar.name_to_idx[bone_name]
        chain = []; cur = idx
        while cur is not None:
            chain.append(cur); cur = avatar.parent_of.get(cur)
        chain.reverse()
        pos = np.zeros(3); R = quat_identity()
        for c in chain:
            t = np.array(avatar.g.nodes[c].translation or [0, 0, 0])
            pos = pos + quat_rotate_vec(R, t)
            R = quat_mul(R, avatar.rest_local_quat[c])
        return pos

    wrist_rest = avatar_world_pos(hand_bone)
    middle_rest = avatar_world_pos(f'{side}HandMiddle1')
    index_rest = avatar_world_pos(f'{side}HandIndex1')
    pinky_rest = avatar_world_pos(f'{side}HandPinky1')

    y_rest = middle_rest - wrist_rest
    if np.linalg.norm(y_rest) < 1e-6: return out
    y_rest /= np.linalg.norm(y_rest)
    x_rest_raw = index_rest - pinky_rest
    x_rest = x_rest_raw - np.dot(x_rest_raw, y_rest) * y_rest
    if np.linalg.norm(x_rest) < 1e-6: return out
    x_rest /= np.linalg.norm(x_rest)
    z_rest = np.cross(x_rest, y_rest)

    # ---- Solve: rotation R such that R * rest_frame = obs_frame ----
    # Build rotation matrices from frames (columns are basis vectors), then R = obs * rest^T.
    # No additional twist needed now that palm_normal_default = -Z (faces body).
    # The Kabsch fit below will solve the full hand orientation directly.
    x_obs_img_plane = x_obs.copy()
    z_obs_img_plane = z_obs.copy()
    M_rest = np.column_stack([x_rest, y_rest, z_rest])
    M_obs  = np.column_stack([x_obs,  y_obs,  z_obs])
    R = M_obs @ M_rest.T

    # Convert R to quaternion
    def mat_to_quat(m):
        tr = m[0,0]+m[1,1]+m[2,2]
        if tr > 0:
            s = math.sqrt(tr+1.0)*2
            w = 0.25*s
            x = (m[2,1]-m[1,2])/s
            y = (m[0,2]-m[2,0])/s
            z = (m[1,0]-m[0,1])/s
        elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
            s = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2
            w = (m[2,1]-m[1,2])/s
            x = 0.25*s
            y = (m[0,1]+m[1,0])/s
            z = (m[0,2]+m[2,0])/s
        elif m[1,1]>m[2,2]:
            s = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2
            w = (m[0,2]-m[2,0])/s
            x = (m[0,1]+m[1,0])/s
            y = 0.25*s
            z = (m[1,2]+m[2,1])/s
        else:
            s = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2
            w = (m[1,0]-m[0,1])/s
            x = (m[0,2]+m[2,0])/s
            y = (m[1,2]+m[2,1])/s
            z = 0.25*s
        return np.array([x, y, z, w])

    world_delta = mat_to_quat(R)
    world_delta = world_delta / (np.linalg.norm(world_delta) + 1e-12)

    # Apply: the world rotation should rotate the rest hand frame to the observed hand frame.
    # Bone's animated world = world_delta * hand_rest_world_q  (delta applied in world)
    # local = parent_world^-1 * world_delta * parent_world * rest_local
    rest_local = avatar.rest_local_quat[hand_idx]
    parent_world = avatar.parent_world_quat(hand_bone)
    local = world_to_local(world_delta, parent_world, rest_local)
    out[hand_bone] = local / (np.linalg.norm(local) + 1e-12)

    # ---- Finger bones (hierarchical top-down solve) ----
    # CRITICAL: each finger bone's parent (in the avatar hierarchy) is the PREVIOUS
    # joint in the same finger chain (or LeftHand for the root). When the parent's
    # rotation changes, the child's reference frame changes too. We need to walk
    # each finger top-down, accumulating world quaternions, and solving each joint
    # against its parent's CURRENT (animated) world quaternion -- not the rest one.

    # Build hand_world_q: the hand bone's animated world rotation
    hand_world_q = quat_mul(world_delta, avatar.rest_world_quat[hand_idx])

    # 2D image landmarks (image-relative). +X = right of image, +Y_image_flipped = up.
    # For finger projection use the IMAGE-PLANE x/z basis (pre-twist), not the twisted one,
    # because the actual finger image landmarks are recorded in the camera's image plane.
    wrist_lm_2d = np.array([h[H['wrist']][0], -h[H['wrist']][1]])
    x_obs_img = np.array([x_obs_img_plane[0], x_obs_img_plane[1]])
    y_obs_img = np.array([y_obs[0], y_obs[1]])
    n_x = np.linalg.norm(x_obs_img); n_y = np.linalg.norm(y_obs_img)
    if n_x < 1e-4 or n_y < 1e-4:
        return out
    x_obs_img /= n_x; y_obs_img /= n_y

    def project_finger_to_world(lm_2d_relative):
        # The camera looks along -Z_gltf. So image (X_right, Y_up_flipped) maps directly
        # to world (X, Y). We DELIBERATELY ignore z entirely and treat finger directions
        # as 2D in the world XY plane. This avoids the y_obs forearm direction contaminating
        # the projection with spurious +Z movement when the forearm points toward camera.
        return np.array([lm_2d_relative[0], lm_2d_relative[1], 0.0])

    # Track animated world quaternions per bone (start with the hand)
    animated_world_q = {hand_idx: hand_world_q}

    # Walk each finger from root joint outward
    FINGER_CHAINS = [
        ['Thumb1',  'Thumb2',  'Thumb3'],
        ['Index1',  'Index2',  'Index3'],
        ['Middle1', 'Middle2', 'Middle3'],
        ['Ring1',   'Ring2',   'Ring3'],
        ['Pinky1',  'Pinky2',  'Pinky3'],
    ]
    # Map short name -> landmark indices for head/tail (head is the LM at this bone's start)
    finger_lm_for_short = {short: (head, tail) for short, head, tail in FINGER_BONES}

    for chain in FINGER_CHAINS:
        for short in chain:
            bone_name = f"{side}Hand{short}"
            if bone_name not in avatar.name_to_idx: continue
            bone_idx = avatar.name_to_idx[bone_name]
            head_lm, tail_lm = finger_lm_for_short[short]

            # Observed direction in WORLD from image-plane projection
            head_2d = np.array([h[head_lm][0], -h[head_lm][1]]) - wrist_lm_2d
            tail_2d = np.array([h[tail_lm][0], -h[tail_lm][1]]) - wrist_lm_2d
            head_w = project_finger_to_world(head_2d)
            tail_w = project_finger_to_world(tail_2d)
            d_world = tail_w - head_w
            if np.linalg.norm(d_world) < 1e-5: continue
            d_world = d_world / np.linalg.norm(d_world)

            # Parent's CURRENT animated world quaternion
            parent_idx = avatar.parent_of.get(bone_idx)
            if parent_idx is None or parent_idx not in animated_world_q:
                # Parent hasn't been solved yet; fall back to rest
                parent_anim_q = avatar.rest_world_quat.get(parent_idx, quat_identity())
            else:
                parent_anim_q = animated_world_q[parent_idx]

            # Bone's CURRENT rest direction = parent_anim_q applied to bone's rest_local +Y axis,
            # then propagated through bone's rest_local rotation.
            # bone_rest_world_q (in animated frame) = parent_anim_q * bone_rest_local_q
            bone_rest_local_q = avatar.rest_local_quat[bone_idx]
            bone_rest_anim_world_q = quat_mul(parent_anim_q, bone_rest_local_q)
            # +Y_local in this animated world frame:
            current_rest_dir = quat_rotate_vec(bone_rest_anim_world_q, np.array([0.0, 1.0, 0.0]))

            # World delta to rotate current_rest_dir to d_world
            bone_world_delta = solve_bone_world_rot(current_rest_dir, d_world)

            # Convert to LOCAL: parent_anim_q^-1 * world_delta * parent_anim_q * rest_local
            local_q = world_to_local(bone_world_delta, parent_anim_q, bone_rest_local_q)
            local_q = local_q / (np.linalg.norm(local_q) + 1e-12)
            out[bone_name] = local_q

            # Update animated_world_q for descendants
            animated_world_q[bone_idx] = quat_mul(bone_world_delta, bone_rest_anim_world_q)

    return out


# ----------------------------- Smoothing + filling ---------------------------

def smooth_quats(seq, window=5):
    if window <= 1: return seq
    out = []
    n = len(seq)
    for i in range(n):
        neighbors = [seq[j] for j in range(max(0, i - window//2), min(n, i + window//2 + 1)) if seq[j] is not None]
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


# ----------------------------- GLB writer ------------------------------------

def write_anim_glb(bone_tracks: dict, fps: float, out_path: str):
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

    animations = [{'name': 'ESL_v2', 'channels': channels, 'samplers': samplers}]
    while len(binary) % 4 != 0: binary += b'\x00'

    g = {
        'asset': {'version': '2.0', 'generator': 'retarget_v2.py'},
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


# ----------------------------- Main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('avatar')
    ap.add_argument('mocap')
    ap.add_argument('output_glb')
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--trim-trailing', action='store_true')
    ap.add_argument('--include-spine', action='store_true')
    ap.add_argument('--include-neck', action='store_true')
    ap.add_argument('--include-legs', action='store_true')
    args = ap.parse_args()

    print(f"Loading avatar rest pose: {args.avatar}")
    avatar = AvatarRest(args.avatar)
    print(f"  {len(avatar.g.nodes)} nodes; mapped {len(avatar.name_to_idx)} named bones")

    with open(args.mocap) as f:
        data = json.load(f)
    fps = float(data['fps'])
    raw = data['frames']
    print(f"Loaded {len(raw)} frames @ {fps}fps")

    poses = [np.array(f['pose']) if f.get('pose') else None for f in raw]

    if args.trim_trailing:
        # Trim by wrist visibility — the actual signing window
        end = len(poses)
        while end > 0 and (poses[end-1] is None or not has_signal(poses[end-1])):
            end -= 1
        start = 0
        while start < end and (poses[start] is None or not has_signal(poses[start])):
            start += 1
        # Add a small padding for natural lead-in/out
        start = max(0, start - 3)
        end = min(len(poses), end + 3)
        poses = poses[start:end]
        raw = raw[start:end]
        print(f"  Trimmed to {len(poses)} active frames ({start}..{end}) based on wrist visibility")

    lh = [np.array(f['lh']) if f.get('lh') else None for f in raw]
    rh = [np.array(f['rh']) if f.get('rh') else None for f in raw]

    solved_body = [solve_body_frame(p, avatar, args.include_spine, args.include_legs, include_neck=args.include_neck)
                   if p is not None and has_signal(p) else {} for p in poses]
    solved_lh = [solve_hand_frame_v2(p, l, 'Left', avatar)
                 if p is not None and has_signal(p) else {} for p, l in zip(poses, lh)]
    solved_rh = [solve_hand_frame_v2(p, r, 'Right', avatar)
                 if p is not None and has_signal(p) else {} for p, r in zip(poses, rh)]

    n = len(solved_body)
    all_bones = set()
    for fr in solved_body + solved_lh + solved_rh: all_bones.update(fr.keys())
    tracks_raw = {b: [None] * n for b in all_bones}
    for src in (solved_body, solved_lh, solved_rh):
        for i, fr in enumerate(src):
            for b, q in fr.items(): tracks_raw[b][i] = q

    tracks = {}
    for b, seq in tracks_raw.items():
        seq = forward_fill(seq)
        seq = smooth_quats(seq, args.smooth)
        seq = [s if s is not None else quat_identity() for s in seq]
        tracks[b] = seq

    print(f"Animated bones ({len(tracks)}): {sorted(tracks.keys())}")
    write_anim_glb(tracks, fps, args.output_glb)


if __name__ == '__main__':
    main()
