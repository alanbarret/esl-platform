"""
FastAPI Route Handlers
======================
POST /api/v1/translate        — Text → Gloss → Motion → Video/GLTF
POST /api/v1/generate-video   — MotionSequence → MP4
POST /api/v1/extract-pose     — Video file → PoseSequence JSON
POST /api/v1/train-gloss-model — Trigger fine-tuning job
GET  /api/v1/health           — Health check
GET  /api/v1/models/status    — AI model loading status
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.models.gloss_model import get_gloss_model
from app.models.motion_engine import get_motion_engine
from app.models.pose_extractor import PoseExtractor
from app.services.translation_service import (
    TranslationRequest,
    TranslationResult,
    get_translation_service,
)

logger = get_logger(__name__)
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Arabic or English input")
    language: str = Field("auto", pattern="^(auto|ar|en)$")
    output_format: str = Field("mp4", pattern="^(mp4|gltf|json)$")
    fps: int = Field(30, ge=15, le=60)
    width: int = Field(1920, ge=320, le=3840)
    height: int = Field(1080, ge=240, le=2160)
    transparent_bg: bool = False


class GlossOnlyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field("auto", pattern="^(auto|ar|en)$")


class TrainRequest(BaseModel):
    dataset_path: str
    epochs: int = Field(10, ge=1, le=100)
    batch_size: int = Field(8, ge=1, le=64)
    learning_rate: float = Field(5e-5, gt=0)
    output_dir: str = "models/gloss-finetuned"


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ESL Platform API"}


@router.get("/models/status")
async def models_status() -> dict:
    gloss = get_gloss_model()
    return {
        "gloss_model": {
            "loaded": gloss.is_loaded,
            "device": str(gloss._device) if gloss.is_loaded else "not_loaded",
        },
    }


# ── Core Endpoints ─────────────────────────────────────────────────────────────

@router.post("/translate", response_model=None)
async def translate(req: TranslateRequest) -> dict:
    """
    Full pipeline: Arabic/English text → ESL gloss → motion → MP4/GLTF.

    Returns video file path or GLTF animation JSON depending on output_format.
    """
    service = get_translation_service()
    result: TranslationResult = await service.translate(
        TranslationRequest(
            text=req.text,
            language=req.language,
            output_format=req.output_format,
            fps=req.fps,
            width=req.width,
            height=req.height,
            transparent_bg=req.transparent_bg,
        )
    )

    if result.status == "failed":
        raise HTTPException(status_code=500, detail=result.error)

    response = {
        "request_id": result.request_id,
        "input_text": result.input_text,
        "detected_language": result.detected_language,
        "gloss_tokens": result.gloss_tokens,
        "total_duration": result.total_duration,
        "status": result.status,
    }

    if result.output_path:
        response["video_url"] = f"/api/v1/video/{Path(result.output_path).name}"
    if result.gltf_animation:
        response["gltf_animation"] = result.gltf_animation

    return response


@router.post("/gloss")
async def generate_gloss_only(req: GlossOnlyRequest) -> dict:
    """Generate only the ESL gloss sequence without rendering video."""
    model = get_gloss_model()
    if not model.is_loaded:
        model.load()
    gloss_tokens = model.generate(req.text, language=req.language)
    return {
        "input_text": req.text,
        "language": req.language,
        "gloss_tokens": gloss_tokens,
        "gloss_string": " ".join(gloss_tokens),
    }


@router.post("/generate-video")
async def generate_video_from_gloss(
    gloss_tokens: list[str],
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict:
    """Generate video from pre-computed gloss tokens."""
    from app.services.video_renderer import VideoRenderer, RenderConfig
    engine = get_motion_engine()
    motion = engine.generate(gloss_tokens, fps=fps)

    renderer = VideoRenderer(RenderConfig(width=width, height=height, fps=fps))
    result = await renderer.render(motion)

    return {
        "gloss_tokens": gloss_tokens,
        "total_duration": result.duration_seconds,
        "frame_count": result.frame_count,
        "video_url": f"/api/v1/video/{result.output_path.name}",
        "file_size_bytes": result.file_size_bytes,
    }


@router.post("/extract-pose")
async def extract_pose(
    video: UploadFile = File(..., description="Video file to extract pose from"),
) -> dict:
    """Extract MediaPipe Holistic pose sequence from uploaded video."""
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        content = await video.read()
        tmp.write(content)

    try:
        extractor = PoseExtractor()
        sequence = extractor.extract_from_video(tmp_path)
        return {
            "source": video.filename,
            "fps": sequence.fps,
            "total_frames": sequence.total_frames,
            "pose_data": {
                "frames": len(sequence.frames),
                "landmarks_per_frame": {
                    "body": 33,
                    "left_hand": 21,
                    "right_hand": 21,
                    "face": 468,
                },
            },
            "sequence_json": sequence.to_json()[:500] + "...",  # truncated preview
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/train-gloss-model")
async def train_gloss_model(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Trigger AraT5 fine-tuning job in background.
    Returns job ID to poll for status.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(
        _run_training_job,
        job_id=job_id,
        dataset_path=req.dataset_path,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        output_dir=req.output_dir,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Training job started in background. Poll /training-status/{job_id} for updates.",
    }


# ── Video File Serving ─────────────────────────────────────────────────────────

@router.get("/video/{filename}")
async def serve_video(filename: str) -> FileResponse:
    """Serve a generated MP4 video file."""
    from app.core.config import get_settings
    settings = get_settings()
    video_path = settings.VIDEO_OUTPUT_DIR / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename,
    )


# ── Background Tasks ───────────────────────────────────────────────────────────

async def _run_training_job(
    job_id: str,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str,
) -> None:
    """Background: fine-tune AraT5 on ESL gloss dataset."""
    logger.info("training_job_start", job_id=job_id, dataset=dataset_path)
    try:
        from scripts.train_gloss_model import train
        await train(
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        logger.info("training_job_complete", job_id=job_id)
    except Exception as e:
        logger.error("training_job_failed", job_id=job_id, error=str(e))
