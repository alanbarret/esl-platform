"""
Translation Service
===================
Orchestrates the full TEXT → GLOSS → MOTION → VIDEO pipeline.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.gloss_model import get_gloss_model
from app.models.motion_engine import get_motion_engine
from app.services.video_renderer import VideoRenderer, RenderConfig

logger = get_logger(__name__)


@dataclass
class TranslationRequest:
    text: str
    language: str = "auto"          # "ar" | "en" | "auto"
    output_format: str = "mp4"      # "mp4" | "json" | "gltf"
    fps: int = 30
    width: int = 1920
    height: int = 1080
    transparent_bg: bool = False


@dataclass
class TranslationResult:
    request_id: str
    input_text: str
    detected_language: str
    gloss_tokens: list[str]
    total_duration: float
    output_path: Optional[str]
    gltf_animation: Optional[dict]
    status: str = "completed"
    error: Optional[str] = None


class TranslationService:
    """
    End-to-end pipeline service.

    text → [GlossModel] → gloss_tokens
    gloss_tokens → [MotionEngine] → MotionSequence
    MotionSequence → [VideoRenderer] → MP4
    MotionSequence → [GLTF Export] → animation JSON
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._gloss_model = None
        self._motion_engine = None

    def _ensure_loaded(self) -> None:
        if self._gloss_model is None:
            self._gloss_model = get_gloss_model()
            if not self._gloss_model.is_loaded:
                self._gloss_model.load()
        if self._motion_engine is None:
            self._motion_engine = get_motion_engine()

    async def translate(self, req: TranslationRequest) -> TranslationResult:
        """Full pipeline: text → video/gltf."""
        request_id = str(uuid.uuid4())[:8]
        logger.info("translation_start", id=request_id, text=req.text[:80])

        try:
            self._ensure_loaded()

            # Step 1: Text → Gloss
            gloss_tokens = self._gloss_model.generate(req.text, language=req.language)
            logger.info("gloss_generated", id=request_id, glosses=gloss_tokens)

            # Step 2: Gloss → Motion
            motion = self._motion_engine.generate(gloss_tokens, fps=req.fps)

            # Step 3: Export
            output_path = None
            gltf_animation = None

            if req.output_format == "mp4":
                renderer = VideoRenderer(RenderConfig(
                    width=req.width,
                    height=req.height,
                    fps=req.fps,
                    transparent_bg=req.transparent_bg,
                ))
                out_file = self.settings.VIDEO_OUTPUT_DIR / f"{request_id}.mp4"
                render_result = await renderer.render(motion, out_file)
                output_path = str(render_result.output_path)

            elif req.output_format == "gltf":
                gltf_animation = motion.to_gltf_animation()

            detected_lang = (
                "ar" if self._gloss_model._is_arabic(req.text) else "en"
            ) if req.language == "auto" else req.language

            return TranslationResult(
                request_id=request_id,
                input_text=req.text,
                detected_language=detected_lang,
                gloss_tokens=gloss_tokens,
                total_duration=motion.total_duration,
                output_path=output_path,
                gltf_animation=gltf_animation,
                status="completed",
            )

        except Exception as e:
            logger.error("translation_failed", id=request_id, error=str(e))
            return TranslationResult(
                request_id=request_id,
                input_text=req.text,
                detected_language="unknown",
                gloss_tokens=[],
                total_duration=0.0,
                output_path=None,
                gltf_animation=None,
                status="failed",
                error=str(e),
            )


# ── Singleton ──────────────────────────────────────────────────────────────────
_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    global _service
    if _service is None:
        _service = TranslationService()
    return _service
