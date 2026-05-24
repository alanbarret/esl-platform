#!/usr/bin/env python3
"""
Retarget pre-extracted MediaPipe Pose mocap (JSON) to a Mixamo-style skeleton,
producing a GLB animation that can be merged into a Ready Player Me avatar.

Input JSON format:
{
  "fps": 25.0,
  "frames": [{"pose": [[x,y,z,visibility], ... 33 items]}, ...],
  "world_landmarks": true
}

Usage:
  python3 retarget_from_mocap.py <mocap.json> <output_anim.glb> [--smooth K] [--trim-trailing N]
"""

import sys
import argparse
import json
import struct
import math
from pathlib import Path

import numpy as np


# MediaPipe Pose landmark indices (BlazePose 33)
LM = {
    'nose': 0,
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24,
    'l_knee': 25, 'r_knee': 26,
    'l_ankle': 27, 'r_ankle': 28,
}

# Body bones with their head/tail landmark keys.
BODY_BONES_UPPER = [
    ('Spine',         'mid_hip',      'mid_shoulder'),
    ('Neck',          'mid_shoulder', 'head_top'),
    ('LeftShoulder',  'mid_shoulder', 'l_shoulder'),
    ('LeftArm',       'l_shoulder',   'l_elbow'),
    ('LeftForeArm',   'l_elbow',      'l_wrist'),
    ('RightShoulder', 'mid_shoulder', 'r_shoulder'),
    ('RightArm',      'r_shoulder',   'r_elbow'),
    ('RightForeArm',  'r_elbow',      'r_wrist'),
]
BODY_BONES_LEGS = [
    ('LeftUpLeg',     'l_hip',        'l_knee'),
    ('LeftLeg',       'l_knee',       'l_ankle'),
    ('RightUpLeg',    'r_hip',        'r_knee'),
    ('RightLeg',      'r_knee',       'r_ankle'),
]

# MediaPipe Hands landmark indices (21 per hand)
H = {
    'wrist': 0,
    'thumb1': 1, 'thumb2': 2, 'thumb3': 3, 'thumb4': 4,
    'index1': 5, 'index2': 6, 'index3': 7, 'index4': 8,
    'middle1': 9, 'middle2': 10, 'middle3': 11, 'middle4': 12,
    'ring1': 13, 'ring2': 14, 'ring3': 15, 'ring4': 16,
    'pinky1': 17, 'pinky2': 18, 'pinky3': 19, 'pinky4': 20,
}
# Finger bone chains: (short_name, head_landmark, tail_landmark)
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

# Mixamo bone rest direction in MODEL world space (T-pose, +Y up, character facing -Z).
# IMPORTANT: In MediaPipe pose_world coords (subject's view): +X to subject's left.
# After axis fix (mp_to_gltf below): +X stays to character's left, matching Mixamo.
REST_DIR_WORLD = {
    'Spine':         np.array([0.0, 1.0, 0.0]),
    'Neck':          np.array([0.0, 1.0, 0.0]),
    'Head':          np.array([0.0, 1.0, 0.0]),
    'LeftShoulder':  np.array([1.0, 0.0, 0.0]),
    'LeftArm':       np.array([1.0, 0.0, 0.0]),
    'LeftForeArm':   np.array([1.0, 0.0, 0.0]),
    'LeftHand':      np.array([1.0, 0.0, 0.0]),
    'RightShoulder': np.array([-1.0, 0.0, 0.0]),
    'RightArm':      np.array([-1.0, 0.0, 0.0]),
    'RightForeArm':  np.array([-1.0, 0.0, 0.0]),
    'RightHand':     np.array([-1.0, 0.0, 0.0]),
    'LeftUpLeg':     np.array([0.0, -1.0, 0.0]),
    'LeftLeg':       np.array([0.0, -1.0, 0.0]),
    'RightUpLeg':    np.array([0.0, -1.0, 0.0]),
    'RightLeg':      np.array([0.0, -1.0, 0.0]),
}


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
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


def slerp(q1, q2, t):
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2; dot = -dot
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(dot); sin_0 = math.sin(theta_0)
    theta = theta_0 * t; sin_t = math.sin(theta)
    s1 = math.cos(theta) - dot * sin_t / sin_0
    s2 = sin_t / sin_0
    return s1 * q1 + s2 * q2


def mp_to_gltf(p: np.ndarray) -> np.ndarray:
    """MediaPipe pose_world: +X subject-left, +Y down, +Z away.
    glTF: +X character-left, +Y up, +Z toward viewer (out of screen).
    Mapping: x' = x, y' = -y, z' = -z
    """
    return np.array([p[0], -p[1], -p[2]])


def get_landmark(pose: np.ndarray, key: str) -> np.ndarray | None:
    """Resolve named landmarks (real or virtual) from a 33x4 pose array."""
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
        up = nose - mid_sh
        return nose + 0.5 * up
    raise KeyError(key)


def has_signal(pose: np.ndarray) -> bool:
    """Detect if a frame is non-trivial (i.e. not all zeros / identity)."""
    if pose is None:
        return False
    # MediaPipe pads idle frames with very low visibility or zeros.
    # Treat a frame as valid if at least one shoulder visibility > 0.5.
    if pose.shape[1] >= 4:
        return bool(pose[LM['l_shoulder'], 3] > 0.5 or pose[LM['r_shoulder'], 3] > 0.5)
    return True


def solve_body_frame(pose: np.ndarray, include_legs: bool = False) -> dict:
    out = {}
    bones = BODY_BONES_UPPER + (BODY_BONES_LEGS if include_legs else [])
    for bone, head_key, tail_key in bones:
        try:
            head = get_landmark(pose, head_key)
            tail = get_landmark(pose, tail_key)
        except Exception:
            continue
        d = tail - head
        if np.linalg.norm(d) < 1e-5:
            continue
        rest = REST_DIR_WORLD[bone]
        out[bone] = quat_from_two_vectors(rest, d)
    return out


def solve_hand_frame(hand_lm: np.ndarray | None, side: str) -> dict:
    """Compute per-finger-bone quaternions from a 21x3 hand landmark array.
    The hand frame is built locally: y_axis = wrist->middle_mcp, x_axis = across palm.
    Bones in Mixamo's local hand frame point +Y. We resolve each finger bone's
    direction in this hand-local frame and produce a quaternion rotating +Y to it.
    """
    out = {}
    if hand_lm is None:
        return out
    h = np.asarray(hand_lm)
    if h.shape[0] < 21:
        return out
    wrist = h[H['wrist']]
    middle_mcp = h[H['middle1']]
    index_mcp = h[H['index1']]
    pinky_mcp = h[H['pinky1']]

    y_axis = middle_mcp - wrist
    n = np.linalg.norm(y_axis)
    if n < 1e-6:
        return out
    y_axis /= n
    x_axis = index_mcp - pinky_mcp
    x_axis -= np.dot(x_axis, y_axis) * y_axis
    if np.linalg.norm(x_axis) < 1e-6:
        return out
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)

    def to_local(p):
        v = p - wrist
        return np.array([np.dot(v, x_axis), np.dot(v, y_axis), np.dot(v, z_axis)])

    local = np.array([to_local(p) for p in h])
    REST = np.array([0.0, 1.0, 0.0])
    for short, head_idx, tail_idx in FINGER_BONES:
        d = local[tail_idx] - local[head_idx]
        if np.linalg.norm(d) < 1e-5:
            continue
        bone_name = f"{side}Hand{short}"
        out[bone_name] = quat_from_two_vectors(REST, d)
    return out


def smooth_quats(seq, window=5):
    if window <= 1:
        return seq
    out = []
    n = len(seq)
    for i in range(n):
        neighbors = []
        for j in range(max(0, i - window // 2), min(n, i + window // 2 + 1)):
            if seq[j] is not None:
                neighbors.append(seq[j])
        if not neighbors:
            out.append(None); continue
        acc = neighbors[0]
        for k, q in enumerate(neighbors[1:], start=2):
            acc = slerp(acc, q, 1.0 / k)
        out.append(acc)
    return out


def forward_fill(seq):
    last = None
    out = []
    for q in seq:
        if q is None:
            out.append(last if last is not None else np.array([0.0, 0.0, 0.0, 1.0]))
        else:
            out.append(q); last = q
    return out


def build_glb_animation(bone_tracks, fps, out_path):
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

    accessors = []
    buffer_views = []

    t_offset, t_len = append_array(times)
    buffer_views.append({'buffer': 0, 'byteOffset': t_offset, 'byteLength': t_len})
    accessors.append({
        'bufferView': 0, 'componentType': 5126, 'count': n_frames,
        'type': 'SCALAR', 'min': [0.0], 'max': [float(times[-1])],
    })

    nodes = []
    bone_to_idx = {}
    for i, bone in enumerate(bone_tracks.keys()):
        nodes.append({'name': bone}); bone_to_idx[bone] = i
    nodes.append({'name': 'Root', 'children': list(range(len(bone_tracks)))})
    scenes = [{'name': 'Scene', 'nodes': [len(nodes) - 1]}]

    channels = []; samplers = []
    for bone, quats in bone_tracks.items():
        arr = np.array(quats, dtype=np.float32)
        offset, length = append_array(arr)
        bv_idx = len(buffer_views)
        buffer_views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': length})
        acc_idx = len(accessors)
        accessors.append({'bufferView': bv_idx, 'componentType': 5126, 'count': n_frames, 'type': 'VEC4'})
        sampler_idx = len(samplers)
        samplers.append({'input': 0, 'interpolation': 'LINEAR', 'output': acc_idx})
        channels.append({'sampler': sampler_idx, 'target': {'node': bone_to_idx[bone], 'path': 'rotation'}})

    animations = [{'name': 'ESLRetarget', 'channels': channels, 'samplers': samplers}]

    while len(binary) % 4 != 0:
        binary += b'\x00'

    gltf_json = {
        'asset': {'version': '2.0', 'generator': 'retarget_from_mocap.py'},
        'scene': 0, 'scenes': scenes, 'nodes': nodes,
        'accessors': accessors, 'bufferViews': buffer_views,
        'buffers': [{'byteLength': len(binary)}],
        'animations': animations,
    }
    json_bytes = json.dumps(gltf_json, separators=(',', ':')).encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '

    glb_total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    out = bytearray()
    out += b'glTF'
    out += struct.pack('<I', 2)
    out += struct.pack('<I', glb_total)
    out += struct.pack('<I', len(json_bytes))
    out += b'JSON'
    out += json_bytes
    out += struct.pack('<I', len(binary))
    out += b'BIN\x00'
    out += binary

    Path(out_path).write_bytes(out)
    print(f"✅ Wrote {out_path} ({glb_total} bytes, {len(bone_tracks)} bones, {n_frames} frames @ {fps}fps = {n_frames/fps:.2f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mocap')
    ap.add_argument('output_glb')
    ap.add_argument('--smooth', type=int, default=5)
    ap.add_argument('--trim-trailing', action='store_true',
                    help='Trim trailing zero/idle frames (visibility-based).')
    ap.add_argument('--legs', action='store_true', help='Include leg retargeting (off by default; ESL footage rarely has clean leg data).')
    args = ap.parse_args()
    start, end = 0, 0  # for trim trailing reference in hand frames


    with open(args.mocap) as f:
        data = json.load(f)
    fps = float(data['fps'])
    raw_frames = data['frames']
    poses = []
    for f in raw_frames:
        if f.get('pose'):
            poses.append(np.array(f['pose']))
        else:
            poses.append(None)
    print(f"Loaded {len(poses)} frames @ {fps}fps from {args.mocap}")

    if args.trim_trailing:
        # Trim trailing frames where no valid pose was detected
        end = len(poses)
        while end > 0 and (poses[end - 1] is None or not has_signal(poses[end - 1])):
            end -= 1
        # Also trim leading
        start = 0
        while start < end and (poses[start] is None or not has_signal(poses[start])):
            start += 1
        poses = poses[start:end]
        # Also trim raw_frames so hand data lines up
        raw_frames = raw_frames[start:end]
        print(f"  Trimmed to {len(poses)} active frames ({start}..{end})")
    else:
        end = len(poses)

    # Optional hand data
    lh_frames, rh_frames = [], []
    has_hands = data.get('has_hands', False)
    for f in raw_frames:
        lh_frames.append(np.array(f['lh']) if f.get('lh') else None)
        rh_frames.append(np.array(f['rh']) if f.get('rh') else None)
    if args.trim_trailing:
        lh_frames = lh_frames[start:end]
        rh_frames = rh_frames[start:end]

    # Solve per-frame quaternions
    solved_body = [solve_body_frame(p, include_legs=args.legs) if p is not None and has_signal(p) else {} for p in poses]
    solved_lh = [solve_hand_frame(lh, 'Left') for lh in lh_frames] if has_hands else [{}] * len(poses)
    solved_rh = [solve_hand_frame(rh, 'Right') for rh in rh_frames] if has_hands else [{}] * len(poses)

    n = len(solved_body)

    # Reorganize: bone -> list of quat-or-None per frame
    all_bones = set()
    for fr in solved_body + solved_lh + solved_rh:
        all_bones.update(fr.keys())
    bone_tracks_raw = {b: [None] * n for b in all_bones}
    for i, fr in enumerate(solved_body):
        for b, q in fr.items(): bone_tracks_raw[b][i] = q
    for i, fr in enumerate(solved_lh):
        for b, q in fr.items(): bone_tracks_raw[b][i] = q
    for i, fr in enumerate(solved_rh):
        for b, q in fr.items(): bone_tracks_raw[b][i] = q

    # Smooth + forward-fill
    bone_tracks = {}
    for b, seq in bone_tracks_raw.items():
        seq = forward_fill(seq)
        seq = smooth_quats(seq, args.smooth)
        seq = [s if s is not None else np.array([0.0, 0.0, 0.0, 1.0]) for s in seq]
        bone_tracks[b] = seq

    print(f"Animated bones ({len(bone_tracks)}): {sorted(bone_tracks.keys())}")
    build_glb_animation(bone_tracks, fps, args.output_glb)


if __name__ == '__main__':
    main()
