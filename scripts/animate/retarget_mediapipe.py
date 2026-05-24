#!/usr/bin/env python3
"""
Retarget MediaPipe Holistic landmarks from a video to a Mixamo-style skeleton,
producing a GLB animation that can be merged into a Ready Player Me avatar.

Pipeline:
  video.mp4
    -> MediaPipe Holistic per frame -> (pose_world_landmarks, lh, rh)
    -> Build a synthetic "MediaPipe skeleton" with target world positions per bone
    -> Solve per-bone local quaternions to point each bone at its child target
    -> Write quaternions as glTF animation tracks targeting Mixamo bone names
    -> Merge into avatar with merge_animation.py

Usage:
  python3 retarget_mediapipe.py <input_video> <output_anim.glb> [--max-seconds N] [--smooth K]
"""

import sys
import argparse
import json
import struct
import math
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp


# -----------------------------------------------------------------------------
# Mixamo bone hierarchy with rest-pose primary child directions.
# We define each chain as (bone_name, parent_name, mediapipe_landmark_source).
# The "direction" is computed from landmarks at runtime.
# -----------------------------------------------------------------------------

# MediaPipe Pose landmark indices (BlazePose 33)
LM = {
    'nose': 0,
    'l_eye_in': 1, 'l_eye': 2, 'l_eye_out': 3,
    'r_eye_in': 4, 'r_eye': 5, 'r_eye_out': 6,
    'l_ear': 7, 'r_ear': 8,
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24,
    'l_knee': 25, 'r_knee': 26,
    'l_ankle': 27, 'r_ankle': 28,
    'l_foot_idx': 31, 'r_foot_idx': 32,
}

# MediaPipe Hands landmark indices (21 per hand)
H = {
    'wrist': 0,
    'thumb1': 1, 'thumb2': 2, 'thumb3': 3, 'thumb4': 4,
    'index1': 5, 'index2': 6, 'index3': 7, 'index4': 8,
    'middle1': 9, 'middle2': 10, 'middle3': 11, 'middle4': 12,
    'ring1': 13, 'ring2': 14, 'ring3': 15, 'ring4': 16,
    'pinky1': 17, 'pinky2': 18, 'pinky3': 19, 'pinky4': 20,
}


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the quaternion (x,y,z,w) that rotates unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        # 180-degree rotation around any axis orthogonal to a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    s = math.sqrt((1.0 + dot) * 2.0)
    inv = 1.0 / s
    return np.array([axis[0] * inv, axis[1] * inv, axis[2] * inv, s * 0.5])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[:3]


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# Frame extraction
# -----------------------------------------------------------------------------

def extract_landmarks(video_path: str, max_seconds: float | None = None):
    """Run MediaPipe Holistic over the video; return list of frame landmark dicts."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps) if max_seconds else None

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

    frames = []
    i = 0
    while True:
        ok, img_bgr = cap.read()
        if not ok:
            break
        if max_frames and i >= max_frames:
            break
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        f = {'i': i, 'pose_w': None, 'lh': None, 'rh': None}
        if result.pose_world_landmarks:
            f['pose_w'] = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks.landmark])
        if result.left_hand_landmarks:
            f['lh'] = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark])
        if result.right_hand_landmarks:
            f['rh'] = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark])
        frames.append(f)
        i += 1
        if i % 30 == 0:
            print(f"  ...processed {i} frames", flush=True)

    cap.release()
    holistic.close()
    print(f"Extracted {len(frames)} frames at {fps:.1f}fps")
    return frames, fps


# -----------------------------------------------------------------------------
# Coordinate spaces.
# MediaPipe pose_world: meters, +X is to subject's left, +Y down, +Z away from camera.
# glTF/Three.js: +Y up, +X to character's left (mirror of viewer), +Z toward camera.
# So we need: x' = x, y' = -y, z' = -z  (right-handed -> right-handed with axis flip)
# -----------------------------------------------------------------------------

def mp_to_gltf(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], -p[1], -p[2]])


# -----------------------------------------------------------------------------
# Per-frame target direction vectors for each retargetable Mixamo bone.
# Each "bone direction" is the unit vector from the bone's head to its primary child,
# in WORLD space (after axis fix).
# -----------------------------------------------------------------------------

# For body, we point each bone at the next joint.
BODY_BONE_TARGETS = [
    # (bone_name, head_landmark_key, tail_landmark_key, rest_dir_in_parent_space)
    # rest_dir is the direction in glTF (+Y up) the bone naturally points in T-pose.
    # In Mixamo rest pose, all body bones point +Y (up the spine, out along arms, down legs).
    # The "head→tail" landmarks are used at runtime to estimate the WORLD direction the bone should point.
    ('Spine',     'mid_hip',      'mid_shoulder'),
    ('Neck',      'mid_shoulder', 'head_top'),
    ('LeftShoulder', 'mid_shoulder', 'l_shoulder'),
    ('LeftArm',     'l_shoulder', 'l_elbow'),
    ('LeftForeArm', 'l_elbow',    'l_wrist'),
    ('RightShoulder', 'mid_shoulder', 'r_shoulder'),
    ('RightArm',     'r_shoulder', 'r_elbow'),
    ('RightForeArm', 'r_elbow',    'r_wrist'),
    ('LeftUpLeg', 'l_hip', 'l_knee'),
    ('LeftLeg',   'l_knee', 'l_ankle'),
    ('RightUpLeg', 'r_hip', 'r_knee'),
    ('RightLeg',   'r_knee', 'r_ankle'),
]

# Finger chains: (bone_name, head_idx, tail_idx) within the hand landmarks.
FINGER_TARGETS = [
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


def get_body_landmark(pose: np.ndarray, key: str) -> np.ndarray | None:
    """Return a body landmark in glTF coords, or a derived virtual landmark."""
    if pose is None:
        return None
    if key in LM:
        return mp_to_gltf(pose[LM[key]])
    if key == 'mid_hip':
        return 0.5 * (mp_to_gltf(pose[LM['l_hip']]) + mp_to_gltf(pose[LM['r_hip']]))
    if key == 'mid_shoulder':
        return 0.5 * (mp_to_gltf(pose[LM['l_shoulder']]) + mp_to_gltf(pose[LM['r_shoulder']]))
    if key == 'head_top':
        # Approximate top of head: nose + offset upward + small forward
        nose = mp_to_gltf(pose[LM['nose']])
        ls = mp_to_gltf(pose[LM['l_shoulder']])
        rs = mp_to_gltf(pose[LM['r_shoulder']])
        mid_sh = 0.5 * (ls + rs)
        up = nose - mid_sh
        return nose + 0.5 * up
    raise KeyError(key)


# -----------------------------------------------------------------------------
# Solve per-frame, per-bone LOCAL quaternions.
# Approach (simple/robust):
#   For each bone, compute the WORLD direction it should point (head -> child landmark).
#   The bone's rest local-to-parent rotation already exists in the avatar.
#   For our generated animation, we override LOCAL rotation directly.
#   We assume rest local rotation in parent's frame makes the bone point along its
#   rest direction; we replace it with the rotation that makes it point along the
#   *current* direction (transformed to the parent's current world rotation).
#
# For an MVP we IGNORE parent's accumulated rotation and just emit a "look at child"
# quaternion in WORLD space; this works decently when the parent is mostly upright.
# True hierarchical solving would compose parent world transforms recursively.
# -----------------------------------------------------------------------------

# Mixamo bone rest direction in MODEL world space (T-pose):
#   - Spine, Neck, Head: +Y
#   - LeftShoulder, LeftArm, LeftForeArm: along character's left arm (=+X in model)
#   - Right side: -X
#   - Legs: -Y (down)
REST_DIR_WORLD = {
    'Spine':         np.array([0.0, 1.0, 0.0]),
    'Spine1':        np.array([0.0, 1.0, 0.0]),
    'Spine2':        np.array([0.0, 1.0, 0.0]),
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
# Finger rest direction: along character's local +Y from each finger root in hand frame.
# (Mixamo finger bones point +Y in their local space.)


def solve_body_frame(pose_w: np.ndarray) -> dict:
    """Return {bone_name: quaternion(x,y,z,w)} for body bones for one frame."""
    out = {}
    if pose_w is None:
        return out
    for bone, head_key, tail_key in BODY_BONE_TARGETS:
        head = get_body_landmark(pose_w, head_key)
        tail = get_body_landmark(pose_w, tail_key)
        if head is None or tail is None:
            continue
        cur_dir = tail - head
        if np.linalg.norm(cur_dir) < 1e-5:
            continue
        rest = REST_DIR_WORLD[bone]
        out[bone] = quat_from_two_vectors(rest, cur_dir)
    return out


def solve_hand_frame(hand: np.ndarray, side: str) -> dict:
    """Return {bone_name: quat} for one hand. side ∈ {'Left','Right'}.
    We treat the wrist as world origin for the hand and produce local-ish rotations
    against the wrist coordinate frame.
    Hand landmark coords are in normalized image-relative space (MediaPipe hand uses
    a hand-local frame: wrist at origin, index_mcp roughly along +X, middle_mcp +Y).
    """
    out = {}
    if hand is None:
        return out
    # Build a hand-local frame: x' along wrist->middle_mcp, y' along wrist->index_mcp cross product
    wrist = hand[H['wrist']]
    middle_mcp = hand[H['middle1']]
    index_mcp = hand[H['index1']]
    pinky_mcp = hand[H['pinky1']]
    y_axis = middle_mcp - wrist
    if np.linalg.norm(y_axis) < 1e-5:
        return out
    y_axis /= np.linalg.norm(y_axis)
    # Across-the-palm vector
    x_axis = index_mcp - pinky_mcp
    x_axis -= np.dot(x_axis, y_axis) * y_axis
    if np.linalg.norm(x_axis) < 1e-5:
        return out
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)

    # Transform every hand landmark into the hand-local frame
    def to_local(p):
        v = p - wrist
        return np.array([np.dot(v, x_axis), np.dot(v, y_axis), np.dot(v, z_axis)])

    local = np.array([to_local(p) for p in hand])

    # For each finger bone, compute rotation that maps +Y to (tail-head) in local frame.
    # (Mixamo finger bones point +Y in their local space.)
    REST = np.array([0.0, 1.0, 0.0])
    for short_name, head_idx, tail_idx in FINGER_TARGETS:
        d = local[tail_idx] - local[head_idx]
        if np.linalg.norm(d) < 1e-5:
            continue
        bone_name = f"{side}Hand{short_name}"
        out[bone_name] = quat_from_two_vectors(REST, d)
    return out


# -----------------------------------------------------------------------------
# Temporal smoothing (slerp over a window)
# -----------------------------------------------------------------------------

def smooth_quats(seq: list[np.ndarray | None], window: int = 5) -> list[np.ndarray | None]:
    if window <= 1:
        return seq
    out = []
    n = len(seq)
    for i in range(n):
        # Collect valid neighbors
        neighbors = []
        for j in range(max(0, i - window // 2), min(n, i + window // 2 + 1)):
            if seq[j] is not None:
                neighbors.append(seq[j])
        if not neighbors:
            out.append(None); continue
        # Iterative slerp toward the center
        acc = neighbors[0]
        for k, q in enumerate(neighbors[1:], start=2):
            acc = slerp(acc, q, 1.0 / k)
        out.append(acc)
    return out


# -----------------------------------------------------------------------------
# Forward-fill missing frames so the animation never freezes badly
# -----------------------------------------------------------------------------

def forward_fill(seq):
    last = None
    out = []
    for q in seq:
        if q is None:
            out.append(last if last is not None else np.array([0,0,0,1.0]))
        else:
            out.append(q); last = q
    return out


# -----------------------------------------------------------------------------
# Build glTF animation from per-frame per-bone quaternions
# -----------------------------------------------------------------------------

def build_glb_animation(
    bone_tracks: dict[str, list[np.ndarray]],
    fps: float,
    bone_name_to_node: dict[str, int],
    avatar_node_count: int,
    out_path: str,
):
    """Write a standalone GLB containing nodes named like Mixamo bones and one animation.
    The output is meant to be merged via merge_animation.py into the real avatar.
    """
    n_frames = max(len(v) for v in bone_tracks.values())
    times = np.arange(n_frames, dtype=np.float32) / fps

    # Build binary buffer
    # Layout:
    #   [time accessor (N floats)]
    #   for each bone: [N quaternions (N*4 floats)]
    binary = bytearray()

    def append_array(arr: np.ndarray) -> tuple[int, int]:
        """Append float32 array; return (byteOffset, byteLength). Pads to 4."""
        nonlocal binary
        if (len(binary) % 4) != 0:
            binary += b'\x00' * (4 - (len(binary) % 4))
        offset = len(binary)
        data = arr.astype(np.float32).tobytes()
        binary += data
        return offset, len(data)

    accessors = []
    buffer_views = []

    # Time accessor
    t_offset, t_len = append_array(times)
    buffer_views.append({'buffer': 0, 'byteOffset': t_offset, 'byteLength': t_len})
    accessors.append({
        'bufferView': 0, 'componentType': 5126, 'count': n_frames,
        'type': 'SCALAR', 'min': [0.0], 'max': [float(times[-1])],
    })
    time_accessor_idx = 0

    # Build nodes: one node per bone, named to match Mixamo names
    nodes = []
    bone_to_idx = {}
    for i, bone in enumerate(bone_tracks.keys()):
        nodes.append({'name': bone})
        bone_to_idx[bone] = i

    # Scene with single root containing all bones flat (the merger only cares about names)
    nodes.append({'name': 'Root', 'children': list(range(len(bone_tracks)))})
    root_idx = len(nodes) - 1
    scenes = [{'name': 'Scene', 'nodes': [root_idx]}]

    # Quaternion accessors per bone
    channels = []
    samplers = []
    for bone, quats in bone_tracks.items():
        arr = np.array(quats, dtype=np.float32)  # (N, 4)
        offset, length = append_array(arr)
        bv_idx = len(buffer_views)
        buffer_views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': length})
        acc_idx = len(accessors)
        accessors.append({
            'bufferView': bv_idx, 'componentType': 5126, 'count': n_frames,
            'type': 'VEC4',
        })
        sampler_idx = len(samplers)
        samplers.append({'input': time_accessor_idx, 'interpolation': 'LINEAR', 'output': acc_idx})
        channels.append({
            'sampler': sampler_idx,
            'target': {'node': bone_to_idx[bone], 'path': 'rotation'},
        })

    animations = [{'name': 'MediaPipeRetarget', 'channels': channels, 'samplers': samplers}]

    # Pad binary to 4 bytes
    while len(binary) % 4 != 0:
        binary += b'\x00'

    gltf_json = {
        'asset': {'version': '2.0', 'generator': 'mediapipe_retarget.py'},
        'scene': 0, 'scenes': scenes,
        'nodes': nodes,
        'accessors': accessors,
        'bufferViews': buffer_views,
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
    print(f"✅ Wrote {out_path} ({glb_total} bytes, {len(bone_tracks)} bones, {n_frames} frames)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('output_glb')
    ap.add_argument('--max-seconds', type=float, default=None)
    ap.add_argument('--smooth', type=int, default=5)
    args = ap.parse_args()

    print(f"Processing {args.video}...")
    frames, fps = extract_landmarks(args.video, args.max_seconds)

    # Solve per-frame quaternions per bone
    print("Solving body quaternions...")
    body_solved = [solve_body_frame(f['pose_w']) for f in frames]
    print("Solving left hand quaternions...")
    lh_solved = [solve_hand_frame(f['lh'], 'Left') for f in frames]
    print("Solving right hand quaternions...")
    rh_solved = [solve_hand_frame(f['rh'], 'Right') for f in frames]

    # Reorganize into per-bone time series
    all_bones = set()
    for fr in body_solved + lh_solved + rh_solved:
        all_bones.update(fr.keys())

    bone_tracks_raw: dict[str, list] = {b: [None] * len(frames) for b in all_bones}
    for i, fr in enumerate(body_solved):
        for b, q in fr.items(): bone_tracks_raw[b][i] = q
    for i, fr in enumerate(lh_solved):
        for b, q in fr.items(): bone_tracks_raw[b][i] = q
    for i, fr in enumerate(rh_solved):
        for b, q in fr.items(): bone_tracks_raw[b][i] = q

    # Smooth + forward-fill
    bone_tracks = {}
    for b, seq in bone_tracks_raw.items():
        seq = forward_fill(seq)
        seq = smooth_quats(seq, args.smooth)
        seq = [s if s is not None else np.array([0,0,0,1.0]) for s in seq]
        bone_tracks[b] = seq

    print(f"Total animated bones: {len(bone_tracks)}")
    print(f"  Body bones: {sorted(b for b in bone_tracks if 'Hand' not in b)}")
    print(f"  Finger bones: {len([b for b in bone_tracks if 'Hand' in b and ('Thumb' in b or 'Index' in b or 'Middle' in b or 'Ring' in b or 'Pinky' in b)])}")

    build_glb_animation(
        bone_tracks=bone_tracks,
        fps=fps,
        bone_name_to_node={},  # unused now
        avatar_node_count=0,
        out_path=args.output_glb,
    )


if __name__ == '__main__':
    main()
