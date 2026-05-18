"""
Gloss Generation Model
======================
AraT5 / mT5 fine-tuned for Arabic/English → ESL gloss sequence generation.

Architecture: Encoder-Decoder (seq2seq)
Input:  Arabic or English sentence
Output: ESL gloss sequence (space-separated gloss tokens)
"""
from __future__ import annotations

import re
import torch
from pathlib import Path
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Arabic preprocessing ─────────────────────────────────────────────────────

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")
TATWEEL = re.compile(r"\u0640")

def preprocess_arabic(text: str) -> str:
    """Remove diacritics, tatweel, normalize whitespace."""
    text = ARABIC_DIACRITICS.sub("", text)
    text = TATWEEL.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Gloss postprocessing ─────────────────────────────────────────────────────

def postprocess_gloss(gloss: str) -> list[str]:
    """Clean and tokenize generated gloss string into list of gloss tokens."""
    gloss = gloss.upper().strip()
    # Remove any non-ASCII non-Arabic chars that slip through
    gloss = re.sub(r"[^\w\s]", "", gloss)
    tokens = [t for t in gloss.split() if t]
    return tokens


# ── Model ────────────────────────────────────────────────────────────────────

class GlossGenerationModel:
    """
    Wraps AraT5 / mT5 for Arabic/English → ESL gloss generation.

    Fine-tuning target:
      Input:  "مرحبا كيف حالك"
      Output: "HELLO YOU HOW"

    Usage:
      model = GlossGenerationModel()
      model.load()
      glosses = model.generate("مرحبا")  # → ["HELLO"]
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._device = self._resolve_device()
        self._loaded = False

    def _resolve_device(self) -> torch.device:
        cfg = self.settings.DEVICE
        if cfg == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if cfg == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self, checkpoint: Optional[str] = None) -> None:
        """Load tokenizer + model from HuggingFace hub or local checkpoint."""
        model_id = checkpoint or self.settings.GLOSS_MODEL_PATH or self.settings.GLOSS_MODEL_NAME
        logger.info("loading_gloss_model", model_id=model_id, device=str(self._device))

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()
        self._loaded = True
        logger.info("gloss_model_loaded")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        language: str = "auto",
        num_beams: int | None = None,
    ) -> list[str]:
        """
        Generate ESL gloss tokens from input text.

        Args:
            text:      Input Arabic or English sentence.
            language:  "ar" | "en" | "auto"
            num_beams: Override beam search width.

        Returns:
            List of uppercase gloss tokens, e.g. ["HELLO", "YOU", "HOW"]
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Preprocess
        if language == "ar" or (language == "auto" and self._is_arabic(text)):
            text = preprocess_arabic(text)
            prompt = f"translate Arabic to ESL gloss: {text}"
        else:
            prompt = f"translate English to ESL gloss: {text}"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.settings.GLOSS_MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
        ).to(self._device)

        gen_config = GenerationConfig(
            max_new_tokens=self.settings.GLOSS_MAX_TARGET_LENGTH,
            num_beams=num_beams or self.settings.GLOSS_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        output_ids = self._model.generate(**inputs, generation_config=gen_config)
        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return postprocess_gloss(decoded)

    @staticmethod
    def _is_arabic(text: str) -> bool:
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return arabic_chars / max(len(text), 1) > 0.3

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Singleton ─────────────────────────────────────────────────────────────────

_gloss_model: Optional[GlossGenerationModel] = None


def get_gloss_model() -> GlossGenerationModel:
    global _gloss_model
    if _gloss_model is None:
        _gloss_model = GlossGenerationModel()
    return _gloss_model
