"""
Re-extract all mocap using pose_world_landmarks (real 3D metric space)
instead of pose_landmarks (image space with weak Z).

pose_world_landmarks:
- Coordinates in meters
- Origin at hip center
- Y-up coordinate system
- Z = depth (negative = toward camera)
- ~30cm upper arm length (realistic)
- Z contribution ~32% vs 10% for image landmarks
"""
import cv2, mediapipe as mp, json, os, sys
from pathlib import Path

MOTION_DB = Path('/root/.openclaw/workspace/esl-platform/data/motion_db')
MOCAP_DIR = Path('/root/.openclaw/workspace/esl-platform/data/processed/mocap')
MOCAP_DIR.mkdir(exist_ok=True)

def extract_world(video_path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    frames = []
    mp_pose = mp.solutions.pose
    mp_hol  = mp.solutions.holistic

    with mp_hol.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hol:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            small = cv2.resize(frame, (320, 180))
            res = hol.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            
            fd = {}
            
            # Use pose_world_landmarks for BODY (true 3D metric space)
            if res.pose_world_landmarks:
                fd['pose'] = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in res.pose_world_landmarks.landmark
                ]
            elif res.pose_landmarks:
                # Fallback to image landmarks if world not available
                fd['pose'] = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in res.pose_landmarks.landmark
                ]
            
            # Hands: MediaPipe hand landmarks are already in normalized 3D
            if res.right_hand_landmarks:
                fd['rhand'] = [[lm.x,lm.y,lm.z] for lm in res.right_hand_landmarks.landmark]
            if res.left_hand_landmarks:
                fd['lhand'] = [[lm.x,lm.y,lm.z] for lm in res.left_hand_landmarks.landmark]
            
            frames.append(fd)
    
    cap.release()
    return {'fps': fps, 'frames': frames, 'world_landmarks': True}


if __name__ == '__main__':
    videos = sorted(MOTION_DB.glob('*.mp4'))
    print(f"Re-extracting {len(videos)} videos with world landmarks...")
    
    for vid in videos:
        name = vid.stem.upper().replace(' ','_')
        out = MOCAP_DIR / f"{name}.json"
        
        print(f"  {name}...", end='', flush=True)
        try:
            data = extract_world(vid)
            if data:
                with open(out,'w') as f:
                    json.dump(data, f, separators=(',',':'))
                print(f" {len(data['frames'])}fr ✓")
            else:
                print(" FAILED")
        except Exception as e:
            print(f" ERROR: {e}")
    
    print("Done.")
