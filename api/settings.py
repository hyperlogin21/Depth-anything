"""Environment-driven configuration for the API.

On AutoDL the system disk is small and ephemeral-ish, while a large
persistent data disk is mounted at /root/autodl-tmp. When that path exists
we default storage and the Hugging Face model cache there; everywhere else
we fall back to a `storage/` directory next to the repo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_AUTODL_DATA_DIR = Path("/root/autodl-tmp")


def _default_storage_dir() -> Path:
    if _AUTODL_DATA_DIR.is_dir():
        return _AUTODL_DATA_DIR / "depth-anything-storage"
    return Path(__file__).resolve().parent.parent / "storage"


def _default_cache_dir() -> Optional[str]:
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    if _AUTODL_DATA_DIR.is_dir():
        return str(_AUTODL_DATA_DIR / "hf-cache")
    return None


class Settings:
    MODEL_ID: str = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")

    STORAGE_DIR: Path = Path(os.environ.get("STORAGE_DIR", str(_default_storage_dir())))
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    OUTPUT_DIR: Path = STORAGE_DIR / "outputs"

    MODEL_CACHE_DIR: Optional[str] = _default_cache_dir()

    MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "500"))
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "4"))
    MAX_SIDE: Optional[int] = int(os.environ["MAX_SIDE"]) if os.environ.get("MAX_SIDE") else None
    INVERT_DEFAULT: bool = os.environ.get("INVERT_DEFAULT", "false").lower() == "true"
    SMOOTHING: float = float(os.environ.get("SMOOTHING", "0.9"))

    JOB_RETENTION_HOURS: float = float(os.environ.get("JOB_RETENTION_HOURS", "24"))
    MAX_CONCURRENT_JOBS: int = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))

    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8000"))


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
