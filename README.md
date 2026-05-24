# ESL Platform — Emirati Sign Language Avatar

Converts Arabic/English text into Emirati Sign Language as a 3D avatar video
(or live in-browser Three.js playback).

## Pipeline

```
TEXT (ar/en)
  ↓ OpenAI gloss generation
GLOSS TOKENS  (sign names + Arabic words to finger-spell)
  ↓ token resolution (direct match / Arabic→English / translate-to-Arabic / letter-spell)
SIGN LIST  (one or more renderable sign names per token)
  ↓ for each sign:
SOURCE VIDEO  →  MediaPipe Holistic (3D pose + 2 hands)
              →  DigiHuman-style retargeting onto Mixamo-rigged GLB
              →  Merged animation GLB
              →  MP4 (headless Three.js render)
```

The backend exposes the merged GLB so the frontend can play it live with
`AnimationMixer` — same data as the MP4, pixel-identical playback.

## Components

| Path | Purpose |
|------|---------|
| `backend/server.py` | HTTP API. Endpoints: `/health`, `/api/v1/gloss`, `/api/v1/translate`, `/api/v1/avatar-glb/{TOKEN}`, `/api/v1/avatar-video/{KEY}`. ThreadingHTTPServer, single file. |
| `backend/avatar_3d_renderer.py` | Subprocess orchestrator for the 3D pipeline (extract → retarget → merge → render) with on-disk caching. |
| `scripts/animate/extract_v2.py` | MediaPipe Tasks API (PoseLandmarker + HandLandmarker) — metric 3D landmarks. |
| `scripts/animate/retarget_digihuman.py` | DigiHuman-style LookRotation retargeting onto the Arab sheikh GLB. |
| `scripts/animate/merge_animation.py` | Bone-name remapped glTF animation merge. |
| `scripts/animate/render.js` | Headless Chromium + Three.js renderer → MP4 via FFmpeg. |
| `scripts/animate/build.sh` | Pipeline runner for a single token. |
| `scripts/scrape.py` | Downloads source videos from the manifest into `data/motion_db/`. |
| `scripts/batch_render.py` | Runs the full pipeline over every downloaded sign. |
| `frontend/` | React + Three.js dashboard (Vite). |
| `wordpress-plugin/` | WP plugin: select text on any article → translate + play avatar. |

## Setup

### Backend deps

```bash
sudo apt install ffmpeg python3 nodejs npm chromium-browser curl
pip install mediapipe opencv-python numpy pygltflib openai
```

### MediaPipe models

```bash
mkdir -p data/mediapipe_models
cd data/mediapipe_models
curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### Avatar GLB

The Arab sheikh model lives at
`data/avatars/arab-man/source/ready player me arab sheik.glb`
(committed in the repo).

### Renderer deps (Three.js + Puppeteer)

```bash
cd scripts/animate
npm install
```

### Frontend deps

```bash
cd frontend
npm install
```

### OpenAI

Export your key so the server can do English-to-Arabic translation + gloss generation:

```bash
export OPENAI_API_KEY=sk-...
```

(Without it the server falls back to a small rule-based map.)

## Run order

### 1. Scrape the source videos

```bash
python3 scripts/scrape.py
```

This downloads all signs listed in `data/raw/uae_signs_full.json` into
`data/motion_db/<TOKEN>.mp4`. Already-cached files are skipped, so it's safe
to re-run. Common flags:

```bash
python3 scripts/scrape.py --limit 50          # just the first 50
python3 scripts/scrape.py --filter sport      # only sport-related entries
python3 scripts/scrape.py --workers 8         # bump parallelism (default 4)
python3 scripts/scrape.py --force             # re-download everything
```

### 2. Build the 3D avatar renders

```bash
python3 scripts/batch_render.py
```

For each downloaded sign, this runs `scripts/animate/build.sh` which executes
the full pipeline (MediaPipe extract → retarget → merge → render). Outputs:

- `data/processed/mocap_holistic_v2/<TOKEN>.json` — landmark cache
- `data/avatars/arab-man/_<TOKEN>_anim.glb` — animation-only GLB
- `data/avatars/arab-man/arab_sheik_<TOKEN>.glb` — merged avatar + animation
- `data/avatar_videos_3d/arab_sheik_<TOKEN>.mp4` — rendered MP4

Already-rendered tokens are skipped. Use `--limit 100` or `--tokens SCHOOL DOCTOR FAMILY`
to render a subset. ~15-30 s per token; full library ≈ 6-10 hours.

### 3. Run the backend

```bash
cd backend
ESL_PORT=8001 python3 server.py
```

The server logs how many signs are available on startup and serves the API on
the port you pick.

### 4. Run the frontend

Development:

```bash
cd frontend
npm run dev
```

Production build:

```bash
cd frontend
npm run build
# serve dist/ via any static server, proxy /api/* to the backend
```

### 5. WordPress plugin (optional)

Zip `wordpress-plugin/` and upload via WP Admin → Plugins → Add New → Upload.
Configure the API URL in *Settings → ESL Sign Plugin*.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + sign count |
| GET | `/api/v1/signs` | List of every renderable sign |
| POST | `/api/v1/gloss` | `{"text":"..."}` → `{"gloss_tokens":[...]}` |
| POST | `/api/v1/translate` | `{"text":"..."}` → tokens + `avatar_video_url` |
| GET | `/api/v1/avatar-glb/{TOKEN}` | Merged GLB for live playback |
| GET | `/api/v1/avatar-glb/{TOKEN}?list=1` | JSON: which signs this token expands into |
| GET | `/api/v1/avatar-video/{KEY}` | Stitched MP4 (lazy-built and cached) |

## How a token gets rendered

`backend/server.py:resolve_renderable(token)` walks this chain:

1. **Direct match.** If `data/motion_db/<TOKEN>.mp4` exists, use it.
2. **Arabic→English mapping.** A small static dictionary covers common
   words (`دكتور` → `DOCTOR`, etc.).
3. **English → Arabic translation.** If the token is an English word without
   a sign, OpenAI translates it to Arabic and the system finger-spells the
   Arabic letters (e.g. `WAR` → `حرب` → `HAA RAA BAA`). Results cached in
   `data/processed/en2ar_cache.json`.
4. **Letter-spell fallback.** Map each character through `ARABIC_CHAR_MAP`
   or `ENGLISH_LETTER_MAP` and keep ones that have source videos.

## Cleaning the caches

Everything under these paths is regenerated on demand:

```
data/motion_db/*.mp4                  # source videos
data/processed/mocap_holistic_v2/*    # MediaPipe landmark caches
data/avatars/arab-man/_*.glb          # per-token animation tracks
data/avatars/arab-man/arab_sheik_*.glb # merged avatar + animation
data/avatar_videos_3d/*.mp4           # rendered MP4s
data/processed/en2ar_cache.json       # English→Arabic translation cache
```

They are all listed in `.gitignore`. Delete to force a rebuild.
