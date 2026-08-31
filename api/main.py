"""FastAPI service: upload a video, get a black-and-white depth video back.

    POST   /videos              upload a video, returns a job id
    GET    /videos               list jobs
    GET    /videos/{id}          poll job status / progress
    GET    /videos/{id}/download download the finished depth video
    DELETE /videos/{id}          delete a job and its files
    GET    /health                liveness + model/device info
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from core.depth_video import SUPPORTED_EXTENSIONS

from .jobs import Job, job_manager
from .settings import settings

CLEANUP_INTERVAL_SECONDS = 3600


def _cleanup_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(CLEANUP_INTERVAL_SECONDS):
        job_manager.cleanup_expired()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = threading.Event()
    threading.Thread(target=_cleanup_loop, args=(stop_event,), daemon=True).start()
    yield
    stop_event.set()


app = FastAPI(
    title="Depth-Anything Video API",
    description="Upload a video, get back a black-and-white depth video for depth-conditioned "
    "video models (e.g. Seedance).",
    lifespan=lifespan,
)


def _job_to_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "processed_frames": job.processed_frames,
        "total_frames": job.total_frames,
        "original_filename": job.original_filename,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "download_url": f"/videos/{job.id}/download" if job.status == "completed" else None,
    }


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "status": "ok",
        "model": settings.MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_concurrent_jobs": settings.MAX_CONCURRENT_JOBS,
    }


@app.post("/videos")
async def upload_video(
    file: UploadFile = File(...),
    invert: bool = Query(settings.INVERT_DEFAULT, description="near=black/far=white instead of the default"),
    max_side: Optional[int] = Query(None, description="Downscale longest side before inference for speed"),
    smoothing: float = Query(settings.SMOOTHING, description="Temporal EMA smoothing factor in [0, 1)"),
) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    job = job_manager.create_job(file.filename or "video", ext)

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(job.input_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")
                out.write(chunk)
    except Exception:
        job.input_path.unlink(missing_ok=True)
        job_manager.delete(job.id)
        raise

    job_manager.submit(job, invert=invert, max_side=max_side, smoothing=smoothing)
    return _job_to_dict(job)


@app.get("/videos")
def list_jobs() -> list:
    return [_job_to_dict(j) for j in job_manager.list()]


@app.get("/videos/{job_id}")
def get_job(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_to_dict(job)


@app.get("/videos/{job_id}/download")
def download_video(job_id: str) -> FileResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(409, f"Job is '{job.status}', not ready for download")
    if not job.output_path.exists():
        raise HTTPException(410, "Output file no longer available")
    return FileResponse(
        job.output_path, media_type="video/mp4", filename=f"{job_id}_depth.mp4"
    )


@app.delete("/videos/{job_id}")
def delete_job(job_id: str) -> dict:
    if not job_manager.delete(job_id):
        raise HTTPException(404, "Job not found")
    return {"deleted": job_id}
