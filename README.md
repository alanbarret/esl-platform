# ESL Platform — Emirati Sign Language Avatar Generation

A production-grade AI platform that converts Arabic/English text into realistic GLTF avatar signing Emirati Sign Language, exported as MP4 video.

## Architecture

```
TEXT INPUT (Arabic/English)
        ↓
AI GLOSS GENERATION (AraT5/mT5)
        ↓
MOTION GENERATION (Retrieval + Transformer)
        ↓
GLTF AVATAR ANIMATION (Three.js / React Three Fiber)
        ↓
MP4 VIDEO EXPORT (FFmpeg)
```

## Stack

| Layer | Technology |
|---|---|
| Frontend | React + React Three Fiber + Three.js + TailwindCSS |
| Backend | Python FastAPI + async |
| AI Models | PyTorch + HuggingFace (AraT5/mT5) |
| Pose | MediaPipe Holistic + OpenCV |
| Video | FFmpeg + OpenCV VideoWriter |
| 3D | GLTF/GLB + VRM rig |
| Infra | Docker Compose + GPU support |

## Quick Start

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Project Structure

```
esl-platform/
├── backend/          # FastAPI + AI pipeline
│   └── app/
│       ├── api/      # Route handlers
│       ├── core/     # Config, logging, GPU
│       ├── models/   # AI model wrappers
│       ├── services/ # Business logic
│       └── utils/    # Helpers
├── frontend/         # React + Three.js
│   └── src/
│       ├── components/  # UI + 3D components
│       ├── hooks/       # Custom hooks
│       ├── pages/       # Page components
│       ├── store/       # State management
│       └── types/       # TypeScript types
├── scripts/          # Training + data pipelines
├── docker/           # Dockerfiles
├── data/             # Datasets + models
└── docs/             # Documentation
```
