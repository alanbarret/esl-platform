#!/usr/bin/env python3
"""
Overlay the SAME 3D MediaPipe landmark skeleton on both the source video and
the avatar render, using a camera fitted to the source frame so the skeleton
projects exactly onto the human body.

For each frame:
  1. Fit a camera (similarity transform: scale + 2D translation) that projects the
     3D pose_world_landmarks onto MediaPipe's image-normalized pose_landmarks.
     This gives us a perfect overlay on the source video.
  2. Use the same fitted camera to draw the 3D skeleton on the avatar panel
     (treating the avatar panel as a separate canvas at the same fitted scale).

Since the source camera is approximately orthographic in image space (subject is
mostly facing the camera at a known distance), the fit is:
    img_x = scale_x * world_x + tx
    img_y = -scale_y * world_y + ty   (image y is down)
This is solved per-frame from the shoulder + hip landmarks.

Usage:
  python3 overlay_landmarks_fit.py <avatar.mp4> <source.mp4> <holistic.json> <out.mp4>
                                   [--trim-start N]
"""
import sys, argparse, json
from pathlib import Path
import cv2
import numpy as np


POSE_EDGES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
POSE_LABELS = {
    0: 'nose', 11: 'L_sh', 12: 'R_sh', 13: 'L_el', 14: 'R_el',
    15: 'L_wr', 16: 'R_wr', 23: 'L_hip', 24: 'R_hip',
    25: 'L_kn', 26: 'R_kn',
}
HAND_LABELS = {0: 'wr', 4: 'th', 8: 'in', 12: 'mi', 16: 'ri', 20: 'pi'}


def fit_camera(pose_world, pose_img):
    """Solve for (sx, sy, tx, ty) such that:
        pose_img.x ≈ sx * pose_world.x + tx
        pose_img.y ≈ sy * (-pose_world.y) + ty   (note y flip)
    using the most reliable visible landmarks (shoulders, hips, nose).

    Returns dict with sx, sy, tx, ty so that fn(world_xyz) -> (img_x, img_y) in 0..1 image coords.
    Z is ignored for the source-fit camera (orthographic approximation).
    """
    pw = np.array(pose_world)  # (33, 4)
    pi = np.array(pose_img)    # (33, 4)

    # Use highly visible landmarks for solve
    use_indices = []
    for i in [11, 12, 23, 24, 13, 14, 0]:
        vis_w = pw[i, 3] if pw.shape[1] >= 4 else 1.0
        vis_i = pi[i, 3] if pi.shape[1] >= 4 else 1.0
        if vis_w > 0.5 and vis_i > 0.5:
            use_indices.append(i)
    if len(use_indices) < 4:
        return None

    # Build linear system: for each landmark i,
    #   pi[i,0] = sx * pw[i,0] + tx
    #   pi[i,1] = -sy * pw[i,1] + ty   (image y down, world y down too in MP — same sign)
    # Note: MediaPipe pose_world y is also DOWN (origin at hips, +y down). So no flip.
    A_x = np.stack([pw[use_indices, 0], np.ones(len(use_indices))], axis=1)
    b_x = pi[use_indices, 0]
    sol_x, *_ = np.linalg.lstsq(A_x, b_x, rcond=None)
    sx, tx = sol_x

    A_y = np.stack([pw[use_indices, 1], np.ones(len(use_indices))], axis=1)
    b_y = pi[use_indices, 1]
    sol_y, *_ = np.linalg.lstsq(A_y, b_y, rcond=None)
    sy, ty = sol_y

    # Manual vertical offset to nudge skeleton up/down on the source frame.
    # Negative value shifts UP (image-normalized y is 0=top, 1=bottom).
    ty -= 0.05

    return {'sx': float(sx), 'sy': float(sy), 'tx': float(tx), 'ty': float(ty)}


def project_world_to_image_norm(world_xyz, cam):
    """Project a single 3D point using the fitted source camera. Returns (x_norm, y_norm)."""
    return (cam['sx'] * world_xyz[0] + cam['tx'],
            cam['sy'] * world_xyz[1] + cam['ty'])


def hand_in_world_avatar_anchored(hand_lm, pose_world, av_bones, side: str, av_cam):
    """Like hand_in_world, but the hand size is anchored to the AVATAR's forearm length
    (in pose_world-equivalent units, derived by inverting the avatar camera scale).
    Keeps the in-image hand size consistent with avatar's actual arm proportions.
    """
    if hand_lm is None or av_bones is None or av_cam is None: return None
    h = np.asarray(hand_lm)
    if h.shape[0] < 21: return None
    pw = np.array(pose_world)
    if side == 'lh':
        wrist_w = pw[15, :3]; elbow_w = pw[13, :3]
        bone_wrist = 'LeftHand'; bone_elbow = 'LeftForeArm'
    else:
        wrist_w = pw[16, :3]; elbow_w = pw[14, :3]
        bone_wrist = 'RightHand'; bone_elbow = 'RightForeArm'
    if bone_wrist not in av_bones or bone_elbow not in av_bones:
        return None
    # Forearm length in NDC for the avatar
    av_wr = np.array(av_bones[bone_wrist][:2])
    av_el = np.array(av_bones[bone_elbow][:2])
    av_fa_ndc = float(np.linalg.norm(av_wr - av_el))
    # Equivalent forearm length in pose_world after avatar camera scale:
    # NDC = sqrt((sx*dx)^2 + (sy*dy)^2). Approximate with mean(|sx|,|sy|).
    avg_scale = (abs(av_cam['sx']) + abs(av_cam['sy'])) * 0.5
    if avg_scale < 1e-9: return None
    fa_len_equiv = av_fa_ndc / avg_scale  # in pose_world units
    target_hand_length = fa_len_equiv * 0.2  # match the source 0.2x factor

    mid_tip_dist_img = float(np.linalg.norm(h[12, :2] - h[0, :2]))
    if mid_tip_dist_img < 1e-6: return None
    scale = target_hand_length / mid_tip_dist_img

    out = []
    for k in range(len(h)):
        dx = (h[k, 0] - h[0, 0]) * scale
        dy = (h[k, 1] - h[0, 1]) * scale
        out.append([wrist_w[0] + dx, wrist_w[1] + dy, wrist_w[2]])
    return np.array(out)


def hand_in_world(hand_lm, pose_world, side: str):
    """Place hand landmarks in MediaPipe pose_world space so they can be projected with the
    same camera as the pose. We anchor the hand's wrist at pose_world's wrist position
    and scale the hand to be a realistic fraction of the forearm length.
    """
    if hand_lm is None: return None
    h = np.asarray(hand_lm)
    if h.shape[0] < 21: return None
    pw = np.array(pose_world)
    if side == 'lh':
        wrist_w = pw[15, :3]; elbow_w = pw[13, :3]
    else:
        wrist_w = pw[16, :3]; elbow_w = pw[14, :3]
    forearm = wrist_w - elbow_w
    fa_len = float(np.linalg.norm(forearm))
    if fa_len < 1e-6: return None

    # Anatomical hand-length as a fraction of forearm length (wrist to middle-finger tip).
    # Empirically tuned for the source video framing.
    target_hand_length = fa_len * 0.2

    # The hand landmark length in image-normalized coords: wrist (0) to middle_tip (12)
    mid_tip_dist_img = float(np.linalg.norm(h[12, :2] - h[0, :2]))
    if mid_tip_dist_img < 1e-6: return None
    scale = target_hand_length / mid_tip_dist_img

    out = []
    for k in range(len(h)):
        dx = (h[k, 0] - h[0, 0]) * scale
        dy = (h[k, 1] - h[0, 1]) * scale
        out.append([wrist_w[0] + dx, wrist_w[1] + dy, wrist_w[2]])
    return np.array(out)


AVATAR_EDGES = [
    ('Hips', 'Spine'), ('Spine', 'Spine2'), ('Spine2', 'Neck'), ('Neck', 'Head'),
    ('Spine2', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'),
    ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
    ('Spine2', 'RightShoulder'), ('RightShoulder', 'RightArm'),
    ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
    ('LeftHand', 'LeftHandThumb1'), ('LeftHandThumb1', 'LeftHandThumb2'), ('LeftHandThumb2', 'LeftHandThumb3'),
    ('LeftHand', 'LeftHandIndex1'), ('LeftHandIndex1', 'LeftHandIndex2'), ('LeftHandIndex2', 'LeftHandIndex3'),
    ('LeftHand', 'LeftHandMiddle1'), ('LeftHandMiddle1', 'LeftHandMiddle2'), ('LeftHandMiddle2', 'LeftHandMiddle3'),
    ('LeftHand', 'LeftHandRing1'), ('LeftHandRing1', 'LeftHandRing2'), ('LeftHandRing2', 'LeftHandRing3'),
    ('LeftHand', 'LeftHandPinky1'), ('LeftHandPinky1', 'LeftHandPinky2'), ('LeftHandPinky2', 'LeftHandPinky3'),
    ('RightHand', 'RightHandThumb1'), ('RightHandThumb1', 'RightHandThumb2'), ('RightHandThumb2', 'RightHandThumb3'),
    ('RightHand', 'RightHandIndex1'), ('RightHandIndex1', 'RightHandIndex2'), ('RightHandIndex2', 'RightHandIndex3'),
    ('RightHand', 'RightHandMiddle1'), ('RightHandMiddle1', 'RightHandMiddle2'), ('RightHandMiddle2', 'RightHandMiddle3'),
    ('RightHand', 'RightHandRing1'), ('RightHandRing1', 'RightHandRing2'), ('RightHandRing2', 'RightHandRing3'),
    ('RightHand', 'RightHandPinky1'), ('RightHandPinky1', 'RightHandPinky2'), ('RightHandPinky2', 'RightHandPinky3'),
]

AVATAR_LABELS_SHORT = {
    'Head': 'head', 'Neck': 'neck', 'Spine': 'spine', 'Spine2': 'sp2', 'Hips': 'hip',
    'LeftShoulder': 'L_sh', 'LeftArm': 'L_el', 'LeftForeArm': 'L_fa', 'LeftHand': 'L_wr',
    'RightShoulder': 'R_sh', 'RightArm': 'R_el', 'RightForeArm': 'R_fa', 'RightHand': 'R_wr',
}


def draw_retargeted_overlay(canvas, frame_data, av_bones, panel_x, panel_y, panel_w, panel_h):
    """Draw the source MediaPipe skeleton, but with each segment (left arm, right arm,
    torso) snapped to the avatar's corresponding bone positions. This produces an
    overlay where shoulders/elbows/wrists land exactly on the avatar's bones, while
    fingers preserve the source's hand shape.
    """
    pose = frame_data.get('pose')
    if not pose: return
    pw = np.array(pose)

    # Avatar bone positions in NDC (0..1) -> pixel coords
    def av_to_px(name):
        b = av_bones.get(name)
        if b is None: return None
        return int(panel_x + b[0] * panel_w), int(panel_y + b[1] * panel_h)

    # ---- Torso (shoulders + hips connecting line) ----
    if av_to_px('LeftShoulder') and av_to_px('RightShoulder'):
        ls = av_to_px('LeftShoulder'); rs = av_to_px('RightShoulder')
        cv2.line(canvas, ls, rs, (80, 80, 220), 1)
        cv2.circle(canvas, ls, 3, (80, 80, 220), -1)
        cv2.circle(canvas, rs, 3, (80, 80, 220), -1)

    # ---- Per-arm segments ----
    for side, pose_sh, pose_el, pose_wr, bone_sh, bone_el, bone_wr, hand_key in [
        ('L', 11, 13, 15, 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'lh'),
        ('R', 12, 14, 16, 'RightShoulder', 'RightArm', 'RightForeArm', 'rh'),
    ]:
        sh_px = av_to_px(bone_sh)
        el_px = av_to_px(bone_el)
        wr_px = av_to_px(bone_wr)
        if not (sh_px and el_px and wr_px): continue
        # Draw the source skeleton's arm by snapping each joint to the avatar.
        cv2.line(canvas, sh_px, el_px, (80, 80, 220), 1)
        cv2.line(canvas, el_px, wr_px, (80, 80, 220), 1)
        cv2.circle(canvas, sh_px, 3, (80, 80, 220), -1)
        cv2.circle(canvas, el_px, 3, (80, 80, 220), -1)
        cv2.circle(canvas, wr_px, 3, (80, 80, 220), -1)

        # ---- Hand: anchor at avatar wrist, orient using avatar's forearm direction,
        #            scale fingers from MediaPipe hand landmarks. ----
        hand_lm = frame_data.get(hand_key)
        if not hand_lm: continue
        h = np.asarray(hand_lm)
        if h.shape[0] < 21: continue
        # Build a 2D image-plane frame for the avatar's forearm:
        #   y_axis = elbow -> wrist (pointing along the hand direction)
        #   x_axis = perpendicular, rotated by the SOURCE's observed wrist twist.
        av_wr = np.array(av_bones[bone_wr][:2])
        av_el = np.array(av_bones[bone_el][:2])
        y_axis = av_wr - av_el
        fa_len_ndc = float(np.linalg.norm(y_axis))
        if fa_len_ndc < 1e-4: continue
        y_axis /= fa_len_ndc
        # Default x_axis perpendicular to forearm
        x_axis_default = np.array([y_axis[1], -y_axis[0]])
        if side == 'R': x_axis_default = -x_axis_default

        # Compute the SOURCE's wrist twist angle. The source forearm direction is
        # (pose_elbow -> pose_wrist) in pose_img (2D). The source palm direction is
        # (pinky_mcp -> index_mcp) in hand_lm (2D, image-relative). The angle between
        # source x_axis (perpendicular to source forearm) and the actual palm tells
        # us how the wrist is twisted relative to the forearm.
        pose_img = frame_data.get('pose_img')
        wrist_twist_angle = 0.0
        hand_lm_full = frame_data.get(hand_key)
        if pose_img and hand_lm_full:
            pi = np.array(pose_img)
            src_el = pi[pose_el, :2]; src_wr = pi[pose_wr, :2]
            src_fa = src_wr - src_el
            src_fa_len = float(np.linalg.norm(src_fa))
            if src_fa_len > 1e-6:
                src_fa_n = src_fa / src_fa_len
                src_x_default = np.array([src_fa_n[1], -src_fa_n[0]])
                if side == 'R': src_x_default = -src_x_default
                # Actual palm direction in image
                idx_pt = np.array([h[5, 0], h[5, 1]])
                pky_pt = np.array([h[17, 0], h[17, 1]])
                palm_dir = idx_pt - pky_pt
                palm_len = float(np.linalg.norm(palm_dir))
                if palm_len > 1e-6:
                    palm_n = palm_dir / palm_len
                    # Angle from src_x_default to palm_n in image plane
                    import math
                    cos_a = float(np.dot(palm_n, src_x_default))
                    sin_a = float(palm_n[0] * src_x_default[1] - palm_n[1] * src_x_default[0])
                    wrist_twist_angle = math.atan2(sin_a, cos_a)

        # Rotate the default x_axis by the observed twist angle in the 2D image plane.
        # In 2D, rotation by `angle` of vector v with perpendicular w (= y_axis):
        #   v' = cos(angle) * v + sin(angle) * w
        import math
        ca, sa = math.cos(wrist_twist_angle), math.sin(wrist_twist_angle)
        x_axis = np.array([ca * x_axis_default[0] + sa * y_axis[0],
                           ca * x_axis_default[1] + sa * y_axis[1]])

        # MediaPipe hand 2D landmark frame (image-normalized, y down).
        # Convert to a 2D frame at the wrist with image y flipped (so +y_image_flipped = up).
        wrist_lm_2d = np.array([h[0, 0], -h[0, 1]])
        middle_lm_2d = np.array([h[9, 0], -h[9, 1]])
        hand_y_img = middle_lm_2d - wrist_lm_2d
        hand_y_len = float(np.linalg.norm(hand_y_img))
        if hand_y_len < 1e-6: continue
        # Scale: avatar hand size = ~80% of avatar forearm length (NDC).
        target_hand_ndc = fa_len_ndc * 0.8
        scale = target_hand_ndc / (np.linalg.norm(np.array([h[12,0],-h[12,1]]) - wrist_lm_2d) + 1e-9)

        # Build the in-image basis of the source hand: y along wrist->middle, x perpendicular.
        hy = hand_y_img / hand_y_len
        hx = np.array([hy[1], -hy[0]])
        # x sign: source x is pinky->index? Check from index/pinky MCP positions.
        index_dir = np.array([h[5,0], -h[5,1]]) - wrist_lm_2d
        pinky_dir = np.array([h[17,0], -h[17,1]]) - wrist_lm_2d
        # Sign so hx points from pinky toward index in image:
        if np.dot(index_dir - pinky_dir, hx) < 0:
            hx = -hx

        # Project each landmark to (hx, hy) basis, then express in avatar (x_axis, y_axis).
        # Avatar coords use y-down (image), so when we go back to NDC we have to flip y_axis sign.
        # av_wr (in NDC) is in image coords (y down); convert y_axis to image coords by flipping y.
        y_axis_img = y_axis
        x_axis_img = x_axis

        HAND_EDGES_DRAW = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        color = (80, 200, 80) if side == 'L' else (180, 100, 60)
        pts_ndc = []
        for k in range(len(h)):
            local = np.array([h[k,0], -h[k,1]]) - wrist_lm_2d
            a = float(np.dot(local, hx))
            b = float(np.dot(local, hy))
            pt_ndc = av_wr + scale * (a * x_axis_img + b * y_axis_img)
            pts_ndc.append(pt_ndc)
        pts_px = [(int(panel_x + p[0] * panel_w), int(panel_y + p[1] * panel_h)) for p in pts_ndc]
        for a, c in HAND_EDGES_DRAW:
            cv2.line(canvas, pts_px[a], pts_px[c], color, 1)
        for pt in pts_px:
            cv2.circle(canvas, pt, 3, color, -1)


def draw_avatar_bones(canvas, bones, panel_x, panel_y, panel_w, panel_h):
    """Draw the avatar's actual bone positions in cyan/green/orange with labels."""
    def to_px(b):
        return int(panel_x + b[0] * panel_w), int(panel_y + b[1] * panel_h)
    for a, c in AVATAR_EDGES:
        if a not in bones or c not in bones: continue
        if a.startswith('LeftHand') and len(a) > 8:
            col = (60, 255, 60)
        elif a.startswith('RightHand') and len(a) > 9:
            col = (60, 180, 255)
        else:
            col = (100, 220, 255)
        cv2.line(canvas, to_px(bones[a]), to_px(bones[c]), col, 1)
    for name, b in bones.items():
        p = to_px(b)
        if name.startswith('LeftHand') and len(name) > 8:
            col = (60, 255, 60)
        elif name.startswith('RightHand') and len(name) > 9:
            col = (60, 180, 255)
        else:
            col = (100, 220, 255)
        cv2.circle(canvas, p, 3, col, -1)
        cv2.circle(canvas, p, 3, (0, 0, 0), 1)
        if name in AVATAR_LABELS_SHORT:
            cv2.putText(canvas, AVATAR_LABELS_SHORT[name], (p[0]+4, p[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton_from_world(canvas, pose_world, lh_world, rh_world, cam,
                              panel_x, panel_y, panel_w, panel_h):
    """Draw skeleton on panel using the fitted camera projection."""
    if pose_world is None or cam is None: return

    def proj(w):
        x_norm, y_norm = project_world_to_image_norm(w, cam)
        return int(panel_x + x_norm * panel_w), int(panel_y + y_norm * panel_h)

    pw = np.array(pose_world)
    # Pose
    for a, b in POSE_EDGES:
        pa = proj(pw[a, :3]); pb = proj(pw[b, :3])
        cv2.line(canvas, pa, pb, (80, 80, 220), 1)
    for i in range(len(pw)):
        if i in (1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32): continue
        p = proj(pw[i, :3])
        cv2.circle(canvas, p, 3, (80, 80, 220), -1)
        cv2.circle(canvas, p, 3, (255, 255, 255), 1)
        if i in POSE_LABELS:
            cv2.putText(canvas, POSE_LABELS[i], (p[0]+4, p[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Hands
    for pts, color in [(lh_world, (80, 200, 80)), (rh_world, (180, 100, 60))]:
        if pts is None: continue
        for a, b in HAND_EDGES:
            pa = proj(pts[a]); pb = proj(pts[b])
            cv2.line(canvas, pa, pb, color, 1)
        for i in range(len(pts)):
            p = proj(pts[i])
            cv2.circle(canvas, p, 3, color, -1)
            if i in HAND_LABELS:
                cv2.putText(canvas, HAND_LABELS[i], (p[0]+4, p[1]+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def fit_camera_to_avatar(bones_for_frame, pose_world):
    """Fit a camera to the AVATAR panel so the same pose_world skeleton stretches to
    match the avatar's actual body extent in the render.

    Anchors:
      X: shoulder pair (consistent with avatar width).
      Y: Head -> Hips (the full visible torso). This makes the projected skeleton
         span the avatar's body height, not just the shoulder->elbow segment.
    """
    if not bones_for_frame: return None
    pw = np.array(pose_world)
    if 'LeftShoulder' not in bones_for_frame or 'RightShoulder' not in bones_for_frame:
        return None
    av_ls = bones_for_frame['LeftShoulder']
    av_rs = bones_for_frame['RightShoulder']

    # X scale + translation from shoulder pair.
    dpw_x = pw[11, 0] - pw[12, 0]
    dav_x = av_ls[0] - av_rs[0]
    if abs(dpw_x) < 1e-6: return None
    sx = dav_x / dpw_x
    tx = av_ls[0] - sx * pw[11, 0]

    # Y scale stretched to match avatar body height: use nose -> mid_hip in pose_world
    # vs Head -> Hips in avatar NDC.
    if 'Head' in bones_for_frame and 'Hips' in bones_for_frame:
        av_head_y = bones_for_frame['Head'][1]
        av_hips_y = bones_for_frame['Hips'][1]
        pw_nose_y = pw[0, 1]
        pw_mid_hip_y = (pw[23, 1] + pw[24, 1]) * 0.5
        dpw_y = pw_mid_hip_y - pw_nose_y
        dav_y = av_hips_y - av_head_y
        if abs(dpw_y) > 1e-6:
            sy = dav_y / dpw_y
        else:
            sy = sx
    else:
        sy = sx

    # Translation anchored to shoulder midpoint Y (so shoulders line up correctly).
    av_sh_mid_y = (av_ls[1] + av_rs[1]) * 0.5
    pw_sh_mid_y = (pw[11, 1] + pw[12, 1]) * 0.5
    ty = av_sh_mid_y - sy * pw_sh_mid_y

    # Apply the same -5% UP nudge as the source camera so both overlays are consistent.
    ty -= 0.05

    return {'sx': float(sx), 'sy': float(sy), 'tx': float(tx), 'ty': float(ty)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('avatar_mp4')
    ap.add_argument('source_mp4')
    ap.add_argument('holistic_json')
    ap.add_argument('bones_json')
    ap.add_argument('out_mp4')
    ap.add_argument('--trim-start', type=int, default=0)
    args = ap.parse_args()

    with open(args.holistic_json) as f: h = json.load(f)
    with open(args.bones_json) as f: b = json.load(f)

    cap_av = cv2.VideoCapture(args.avatar_mp4)
    cap_src = cv2.VideoCapture(args.source_mp4)
    av_w = int(cap_av.get(cv2.CAP_PROP_FRAME_WIDTH))
    av_h = int(cap_av.get(cv2.CAP_PROP_FRAME_HEIGHT))
    av_fps = cap_av.get(cv2.CAP_PROP_FPS) or 25.0
    n_av = int(cap_av.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_w = av_w; panel_h = av_h
    out_w = panel_w * 2
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp = str(Path(args.out_mp4).with_suffix('.tmp.mp4'))
    writer = cv2.VideoWriter(tmp, fourcc, av_fps, (out_w, out_h))

    src_frames = h['frames'][args.trim_start:]
    print(f"Avatar frames: {n_av}, Source frames (post-trim): {len(src_frames)}")

    i = 0
    while True:
        ok_av, av_img = cap_av.read()
        if not ok_av: break
        si = min(len(src_frames) - 1, i)
        fdata = src_frames[si] if si < len(src_frames) else None

        cap_src.set(cv2.CAP_PROP_POS_FRAMES, args.trim_start + si)
        ok_src, src_img = cap_src.read()
        if not ok_src or src_img is None:
            src_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        else:
            src_img = cv2.resize(src_img, (panel_w, panel_h))

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, 0:panel_w] = src_img
        canvas[:, panel_w:2*panel_w] = av_img

        if fdata and fdata.get('pose') and fdata.get('pose_img'):
            # Source camera: fit pose_world -> pose_img (NDC)
            src_cam = fit_camera(fdata['pose'], fdata['pose_img'])

            lh_world = hand_in_world(fdata.get('lh'), fdata['pose'], 'lh')
            rh_world = hand_in_world(fdata.get('rh'), fdata['pose'], 'rh')

            # Draw on source using source camera
            if src_cam:
                draw_skeleton_from_world(canvas, fdata['pose'], lh_world, rh_world,
                                         src_cam, 0, 0, panel_w, panel_h)

                # Draw only the AVATAR's own bones on the avatar panel (no source overlay).
                av_bones = b['frames'][i] if i < len(b['frames']) else None
                if av_bones:
                    draw_avatar_bones(canvas, av_bones, panel_w, 0, panel_w, panel_h)

        cv2.putText(canvas, 'SOURCE + 3D landmarks', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(canvas, 'AVATAR + 3D landmarks', (panel_w+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(canvas, f"f{i}", (out_w-60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        writer.write(canvas)
        i += 1

    cap_av.release(); cap_src.release(); writer.release()

    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', tmp,
        '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        args.out_mp4,
    ], check=True)
    Path(tmp).unlink(missing_ok=True)
    print(f"✅ Wrote {args.out_mp4}")


if __name__ == '__main__':
    main()
