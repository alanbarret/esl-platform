"""
Convert input video to skeleton overlay video using MediaPipe Holistic.
Draws body, hands (with all 5 fingers), and face landmarks.
"""
import cv2, numpy as np, sys
from pathlib import Path
import mediapipe as mp

INPUT  = sys.argv[1] if len(sys.argv)>1 else '/root/.openclaw/media/inbound/file_15---cde4e4da-5f90-409f-971e-911fe5ac93b7.mp4'
OUTPUT = sys.argv[2] if len(sys.argv)>2 else '/root/.openclaw/workspace/skeleton_output.avi'

mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Write to temp AVI then convert to MP4
tmp_avi = OUTPUT.replace('.mp4','.tmp.avi').replace('.avi','.tmp.avi')
out = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))

# Custom drawing specs
BODY_SPEC = mp_draw.DrawingSpec(color=(124,58,237), thickness=3, circle_radius=4)
HAND_SPEC  = mp_draw.DrawingSpec(color=(168,255,75), thickness=2, circle_radius=3)
CONN_SPEC  = mp_draw.DrawingSpec(color=(100,80,200), thickness=2)
HCONN_SPEC = mp_draw.DrawingSpec(color=(100,200,80), thickness=2)
FACE_SPEC  = mp_draw.DrawingSpec(color=(60,60,80),   thickness=1, circle_radius=1)
FCONN_SPEC = mp_draw.DrawingSpec(color=(40,40,60),   thickness=1)

print(f'Processing {TOTAL} frames at {fps:.0f}fps...')

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
) as holistic:
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Dark background
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            t = y/H
            canvas[y,:] = (int(8+t*18), int(8+t*14), int(18+t*32))

        # Process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        # Face mesh (subtle)
        if res.face_landmarks:
            mp_draw.draw_landmarks(canvas, res.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION, FACE_SPEC, FCONN_SPEC)
            mp_draw.draw_landmarks(canvas, res.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_draw.DrawingSpec(color=(80,80,120), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(60,60,100), thickness=1))

        # Body pose
        if res.pose_landmarks:
            mp_draw.draw_landmarks(canvas, res.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS, BODY_SPEC, CONN_SPEC)

        # Left hand - full finger detail
        if res.left_hand_landmarks:
            mp_draw.draw_landmarks(canvas, res.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, HAND_SPEC, HCONN_SPEC)

        # Right hand - full finger detail
        if res.right_hand_landmarks:
            mp_draw.draw_landmarks(canvas, res.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,165,75), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(200,130,60), thickness=2))

        # Frame info
        cv2.putText(canvas, f'Frame {fi+1}/{TOTAL}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,60,80), 1)

        # Progress bar
        prog = int(W * 0.8 * (fi+1) / TOTAL)
        cv2.rectangle(canvas, (int(W*0.1), H-10), (int(W*0.1)+int(W*0.8), H-4), (30,28,50), -1)
        cv2.rectangle(canvas, (int(W*0.1), H-10), (int(W*0.1)+prog, H-4), (124,58,237), -1)

        out.write(canvas)
        fi += 1
        if fi % 50 == 0:
            print(f'  {fi}/{TOTAL} frames done')

cap.release()
out.release()

import os, subprocess
if OUTPUT.endswith('.mp4'):
    subprocess.run(['ffmpeg','-y','-i',tmp_avi,'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',OUTPUT],capture_output=True)
    os.unlink(tmp_avi)
import os
print(f'Done! {OUTPUT} ({os.path.getsize(OUTPUT)//1024}KB)')
