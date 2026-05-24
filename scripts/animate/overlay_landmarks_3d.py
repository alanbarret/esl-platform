#!/usr/bin/env python3
"""
Overlay MediaPipe 3D landmarks (pose_world + hands) onto both the source video
and the rendered avatar, projected through the SAME camera as the avatar render.

This shows where the 3D source skeleton actually is in 3D space, alongside the
avatar's 3D skeleton, so divergence between them is immediately visible.

Usage:
  python3 overlay_landmarks_3d.py <avatar.mp4> <source.mp4> <holistic.json>
                                  <bones.json> <out.mp4>
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
    25: 'L_kn', 26: 'R_kn', 27: 'L_an', 28: 'R_an',
}
HAND_LABELS = {0: 'wr',
               1: 'th1', 2: 'th2', 3: 'th3', 4: 'th4',
               5: 'in1', 6: 'in2', 7: 'in3', 8: 'in4',
               9: 'mi1', 10: 'mi2', 11: 'mi3', 12: 'mi4',
               13: 'ri1', 14: 'ri2', 15: 'ri3', 16: 'ri4',
               17: 'pi1', 18: 'pi2', 19: 'pi3', 20: 'pi4'}


def mp_to_gltf(p):
    """MediaPipe pose_world: +X subject-left, +Y down, +Z away.
    glTF: +X char-left, +Y up, +Z toward viewer.
    Map: (x, -y, -z)"""
    return np.array([p[0], -p[1], -p[2]], dtype=np.float64)


def project_world_to_screen(world_xyz, projection_4x4, view_4x4, width, height):
    """Project a world-space 3D point to screen pixels using Three.js camera matrices.
    
    Three.js camera.projectionMatrix is stored column-major as length-16 array.
    The combined projection*view * (x,y,z,1) gives clip coords; divide by w for NDC;
    NDC to pixel: x_px = (ndc.x*0.5 + 0.5) * width, y_px = (-ndc.y*0.5 + 0.5) * height.
    """
    # Three.js arrays are column-major. Convert to 4x4 row-major numpy.
    P = np.array(projection_4x4, dtype=np.float64).reshape(4, 4, order='F')
    V = np.array(view_4x4, dtype=np.float64).reshape(4, 4, order='F')
    p4 = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    clip = P @ V @ p4
    if clip[3] == 0: return None
    ndc = clip[:3] / clip[3]
    x_px = int((ndc[0] * 0.5 + 0.5) * width)
    y_px = int((-ndc[1] * 0.5 + 0.5) * height)
    return (x_px, y_px, float(ndc[2]))


def estimate_subject_height_meters(pose_world):
    """Estimate subject height in meters from MediaPipe pose_world landmarks."""
    # nose to ankle Y delta
    nose_y = -pose_world[0][1]  # mp_to_gltf y
    l_ank_y = -pose_world[27][1] if pose_world[27][3] > 0.3 else None
    r_ank_y = -pose_world[28][1] if pose_world[28][3] > 0.3 else None
    if l_ank_y is None and r_ank_y is None:
        # Fallback: use shoulder->hip distance and assume ratio
        l_sh = mp_to_gltf(pose_world[11][:3]); l_hip = mp_to_gltf(pose_world[23][:3])
        return float(np.linalg.norm(l_sh - l_hip)) * 4.0
    ank_y = min(filter(lambda x: x is not None, [l_ank_y, r_ank_y]))
    return float(nose_y - ank_y) * 1.07  # nose to top of head extra


def world_anchor_for_avatar(bones_for_frame, avatar_camera):
    """Compute the world position the avatar uses for its 'mid_shoulder' anchor.
    
    Since bones.json gives us screen-space (NDC 0..1) positions and the camera, we
    can back-project a fixed Z (avatar mid_shoulder Z in world). But easier: we'll
    place the source skeleton at a fixed world position matching where the avatar
    renders, anchored using shoulder midpoint == avatar's shoulder midpoint in world.
    """
    return None  # we'll use a known fixed anchor instead


def transform_pose_world_to_avatar_world(pose_world_landmarks, target_mid_shoulder=np.array([0.0, 1.5, 0.0]),
                                          target_height=1.75):
    """Transform MediaPipe pose_world (origin at hips, meters) to the avatar's world
    coordinates so that the skeleton overlaps the avatar in 3D.
    
    Strategy:
      - Convert MP pose_world via mp_to_gltf (axis flip).
      - Translate so mid_shoulder is at target_mid_shoulder.
      - Scale so subject height matches target_height.
    """
    arr = np.array([mp_to_gltf(p[:3]) for p in pose_world_landmarks])
    mid_sh = (arr[11] + arr[12]) * 0.5
    # Translate
    arr -= mid_sh
    # Estimate height in mp space (nose y - ankle y)
    nose_y = arr[0][1]
    ank_y_l = arr[27][1] if pose_world_landmarks[27][3] > 0.3 else None
    ank_y_r = arr[28][1] if pose_world_landmarks[28][3] > 0.3 else None
    if ank_y_l is not None and ank_y_r is not None:
        ank_y = min(ank_y_l, ank_y_r)
    elif ank_y_l is not None: ank_y = ank_y_l
    elif ank_y_r is not None: ank_y = ank_y_r
    else: ank_y = -0.9  # fallback
    mp_height = float(nose_y - ank_y) * 1.07
    if mp_height < 0.1: mp_height = 1.7  # fallback
    scale = target_height / mp_height
    arr *= scale
    arr += target_mid_shoulder
    return arr


def transform_hand_to_avatar_world(hand_lm, pose_avatar_world, side: str, hand_scale_factor=1.0):
    """Map MediaPipe hand landmarks (image-normalized) into avatar world space.
    
    Anchor the wrist at the pose's wrist (in avatar world) and orient the hand
    using the pose's forearm direction. Hand landmark Z is ignored.
    """
    h = np.asarray(hand_lm)
    if h.shape[0] < 21:
        return None
    side_idx = 15 if side == 'lh' else 16   # left wrist / right wrist
    elbow_idx = 13 if side == 'lh' else 14
    if side_idx >= len(pose_avatar_world): return None
    wrist_world = pose_avatar_world[side_idx]
    elbow_world = pose_avatar_world[elbow_idx]
    forearm_dir = wrist_world - elbow_world
    fa_len = float(np.linalg.norm(forearm_dir))
    if fa_len < 1e-6: return None
    forearm_dir = forearm_dir / fa_len
    # Hand size = ~70% of forearm length
    hand_scale = fa_len * 0.7 * hand_scale_factor
    
    # Build a perpendicular frame using world up
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(forearm_dir, world_up))) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    x_world = np.cross(world_up, forearm_dir); x_world /= np.linalg.norm(x_world)
    z_world = np.cross(x_world, forearm_dir)

    # Build hand-local basis from hand landmarks (image coords, y flipped)
    def to_local(p):
        return np.array([p[0] - h[0, 0], -(p[1] - h[0, 1]), 0.0])

    yL = to_local(h[9]); yL /= (np.linalg.norm(yL) + 1e-9)
    xL_raw = to_local(h[5]) - to_local(h[17])
    xL = xL_raw - np.dot(xL_raw, yL) * yL
    if np.linalg.norm(xL) < 1e-6:
        return None
    xL /= np.linalg.norm(xL)

    # Project each landmark to (xL, yL), then express in world via (x_world, forearm_dir).
    positions = []
    for k in range(len(h)):
        local = to_local(h[k])
        a = float(np.dot(local, xL))
        b = float(np.dot(local, yL))
        positions.append(wrist_world + x_world * a * hand_scale * 5 + forearm_dir * b * hand_scale * 5)
    return np.array(positions)


def draw_source_skeleton_2d(canvas, frame_data, panel_x, panel_y, panel_w, panel_h):
    """Draw skeleton on the source panel using image-normalized landmarks.
    These are the source camera's actual 2D projections so they align with the video."""
    if not frame_data: return

    def to_px(nx, ny):
        return int(panel_x + nx * panel_w), int(panel_y + ny * panel_h)

    pose_img = frame_data.get('pose_img')
    if pose_img:
        arr = np.array(pose_img)
        for a, b in POSE_EDGES:
            cv2.line(canvas, to_px(arr[a,0], arr[a,1]), to_px(arr[b,0], arr[b,1]),
                     (80, 80, 220), 1)
        for i in range(len(arr)):
            if i in (1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32): continue
            p = to_px(arr[i,0], arr[i,1])
            cv2.circle(canvas, p, 3, (80, 80, 220), -1)
            cv2.circle(canvas, p, 3, (255, 255, 255), 1)
            if i in POSE_LABELS:
                cv2.putText(canvas, POSE_LABELS[i], (p[0]+4, p[1]-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    for hand_key, color in [('lh', (80, 200, 80)), ('rh', (180, 100, 60))]:
        hand = frame_data.get(hand_key)
        if not hand: continue
        harr = np.array(hand)
        pts = [to_px(harr[k,0], harr[k,1]) for k in range(len(harr))]
        for a, b in HAND_EDGES:
            cv2.line(canvas, pts[a], pts[b], color, 1)
        for i, pt in enumerate(pts):
            cv2.circle(canvas, pt, 3, color, -1)
            if i in HAND_LABELS and i in (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20):
                cv2.putText(canvas, HAND_LABELS[i], (pt[0]+4, pt[1]+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def draw_3d_skeleton(canvas, pose_3d_pts, lh_3d_pts, rh_3d_pts,
                     camera, panel_x, panel_y, panel_w, panel_h):
    """Project 3D landmark points and draw."""
    if camera is None: return

    proj = camera['projection']; view = camera['matrixWorldInverse']
    width, height = camera['width'], camera['height']

    def proj_pt(p3):
        s = project_world_to_screen(p3, proj, view, width, height)
        if s is None: return None
        # Scale screen coords from camera (width, height) to (panel_w, panel_h)
        x_panel = int(panel_x + s[0] * panel_w / width)
        y_panel = int(panel_y + s[1] * panel_h / height)
        return (x_panel, y_panel)

    # Pose
    if pose_3d_pts is not None:
        # Edges
        for a, b in POSE_EDGES:
            pa = proj_pt(pose_3d_pts[a]); pb = proj_pt(pose_3d_pts[b])
            if pa and pb: cv2.line(canvas, pa, pb, (80, 80, 220), 1)
        # Points + labels
        for i in range(len(pose_3d_pts)):
            if i in (1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,29,30,31,32): continue
            p = proj_pt(pose_3d_pts[i])
            if not p: continue
            cv2.circle(canvas, p, 3, (80, 80, 220), -1)
            cv2.circle(canvas, p, 3, (255, 255, 255), 1)
            if i in POSE_LABELS:
                cv2.putText(canvas, POSE_LABELS[i], (p[0]+4, p[1]-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Hands
    for pts, color in [(lh_3d_pts, (80, 200, 80)), (rh_3d_pts, (180, 100, 60))]:
        if pts is None: continue
        for a, b in HAND_EDGES:
            pa = proj_pt(pts[a]); pb = proj_pt(pts[b])
            if pa and pb: cv2.line(canvas, pa, pb, color, 1)
        for i, p3 in enumerate(pts):
            p = proj_pt(p3)
            if not p: continue
            cv2.circle(canvas, p, 3, color, -1)
            if i in HAND_LABELS and i in (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20):
                cv2.putText(canvas, HAND_LABELS[i], (p[0]+4, p[1]+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


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
    print(f"Holistic frames: {len(h['frames'])}, Bones frames: {len(b['frames'])}")

    camera = b.get('camera')
    if not camera:
        print("WARNING: no camera info in bones.json — re-render avatar first.")
        return

    # Compute avatar's target mid_shoulder world position by averaging bone positions
    # We need a fixed anchor for the source skeleton placement that overlaps the avatar.
    # We'll use the first frame's avatar LeftShoulder/RightShoulder bone positions in NDC
    # to derive their world positions. Actually we don't have world positions — only NDC.
    # Simpler: use a static anchor that matches the avatar's known T-pose mid_shoulder.
    # For RPM model, hips at (0, ~1.0, 0) and mid_shoulder at (0, ~1.4, 0).
    # Just use a hard-coded anchor.
    target_mid_shoulder = np.array([0.0, 1.45, 0.0])
    target_height = 1.75

    cap_av = cv2.VideoCapture(args.avatar_mp4)
    cap_src = cv2.VideoCapture(args.source_mp4)
    av_w = int(cap_av.get(cv2.CAP_PROP_FRAME_WIDTH))
    av_h = int(cap_av.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    av_fps = cap_av.get(cv2.CAP_PROP_FPS) or 25.0
    n_av = int(cap_av.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_w = av_w; panel_h = av_h
    out_w = panel_w * 2  # source-with-3d-overlay | avatar-with-3d-overlay
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp = str(Path(args.out_mp4).with_suffix('.tmp.mp4'))
    writer = cv2.VideoWriter(tmp, fourcc, av_fps, (out_w, out_h))

    src_frames = h['frames'][args.trim_start:]
    print(f"Avatar frames: {n_av}, Source (after trim): {len(src_frames)}")

    i = 0
    while True:
        ok_av, av_img = cap_av.read()
        if not ok_av: break
        # Source frame index matched 1:1 (since trim was already applied to source list)
        si = min(len(src_frames) - 1, i)
        fdata = src_frames[si] if si < len(src_frames) else None

        # Source image: same trim offset
        cap_src.set(cv2.CAP_PROP_POS_FRAMES, args.trim_start + si)
        ok_src, src_img = cap_src.read()
        if not ok_src or src_img is None:
            src_img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        else:
            src_img = cv2.resize(src_img, (panel_w, panel_h))

        # Build 3D landmark positions in avatar world space
        pose_3d = None; lh_3d = None; rh_3d = None
        if fdata and fdata.get('pose'):
            pose_3d = transform_pose_world_to_avatar_world(fdata['pose'],
                                                          target_mid_shoulder=target_mid_shoulder,
                                                          target_height=target_height)
            if fdata.get('lh'):
                lh_3d = transform_hand_to_avatar_world(fdata['lh'], pose_3d, 'lh')
            if fdata.get('rh'):
                rh_3d = transform_hand_to_avatar_world(fdata['rh'], pose_3d, 'rh')

        # Canvas: [source-with-overlay | avatar-with-overlay]
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, 0:panel_w] = src_img
        canvas[:, panel_w:2*panel_w] = av_img

        # Overlay on AVATAR panel using its camera projection (real 3D landmarks)
        draw_3d_skeleton(canvas, pose_3d, lh_3d, rh_3d, camera, panel_w, 0, panel_w, panel_h)

        # For the SOURCE panel, use the SOURCE's own image-normalized landmarks,
        # which ARE the source camera's projection of the skeleton.
        # These align perfectly with the source video by definition.
        draw_source_skeleton_2d(canvas, fdata, 0, 0, panel_w, panel_h)

        # Labels
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
