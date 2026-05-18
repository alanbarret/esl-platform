"""Application configuration — environment-driven settings."""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    APP_NAME: str = "ESL Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR.parent / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    MOTION_DB_DIR: Path = DATA_DIR / "motion_db"
    TEMP_DIR: Path = BASE_DIR / "tmp"

    # AI Models
    GLOSS_MODEL_NAME: str = "UBC-NLP/AraT5v2-base-1024"
    GLOSS_MODEL_PATH: str = ""          # Override with fine-tuned checkpoint
    GLOSS_MAX_INPUT_LENGTH: int = 512
    GLOSS_MAX_TARGET_LENGTH: int = 256
    GLOSS_NUM_BEAMS: int = 4

    # GPU
    DEVICE: str = "cuda"                # "cuda" | "cpu" | "mps"
    GPU_MEMORY_FRACTION: float = 0.85

    # Video
    VIDEO_FPS: int = 30
    VIDEO_WIDTH: int = 1920
    VIDEO_HEIGHT: int = 1080
    VIDEO_CODEC: str = "libx264"
    VIDEO_OUTPUT_DIR: Path = BASE_DIR / "tmp" / "videos"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER: str = "redis://redis:6379/0"
    CELERY_BACKEND: str = "redis://redis:6379/1"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    def ensure_dirs(self) -> None:
        for d in [self.DATA_DIR, self.MODELS_DIR, self.MOTION_DB_DIR,
                  self.TEMP_DIR, self.VIDEO_OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
