"""In-process job queue for video -> depth video conversion.

Kept deliberately simple (a thread pool + an in-memory dict, no external
queue/broker) since this runs as a single process on a single GPU box.
Jobs and their files do not survive a process restart, which is an
acceptable trade-off for a self-hosted rental-GPU tool.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.depth_engine import DepthEngine
from core.depth_video import VideoDepthConfig, process_video

from .settings import settings


@dataclass
class Job:
    id: str
    input_path: Path
    output_path: Path
    original_filename: str
    status: str = "pending"  # pending | processing | completed | failed
    error: Optional[str] = None
    processed_frames: int = 0
    total_frames: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def progress(self) -> Optional[float]:
        if self.total_frames:
            return round(min(self.processed_frames / self.total_frames, 1.0), 4)
        return None


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # A single worker (by default) processes one video at a time, which
        # keeps GPU memory usage predictable on a single-GPU rental box.
        self._executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_JOBS)
        self._engine: Optional[DepthEngine] = None
        self._engine_lock = threading.Lock()

    def get_engine(self) -> DepthEngine:
        # Loaded lazily on first job and reused afterwards; loading a
        # Depth-Anything checkpoint per request would be far too slow.
        with self._engine_lock:
            if self._engine is None:
                self._engine = DepthEngine(settings.MODEL_ID, cache_dir=settings.MODEL_CACHE_DIR)
            return self._engine

    def create_job(self, original_filename: str, ext: str) -> Job:
        job_id = uuid.uuid4().hex
        input_path = settings.UPLOAD_DIR / f"{job_id}{ext}"
        output_path = settings.OUTPUT_DIR / f"{job_id}.mp4"
        job = Job(
            id=job_id,
            input_path=input_path,
            output_path=output_path,
            original_filename=original_filename,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def submit(self, job: Job, invert: bool, max_side: Optional[int], smoothing: float) -> None:
        self._executor.submit(self._run, job, invert, max_side, smoothing)

    def _run(self, job: Job, invert: bool, max_side: Optional[int], smoothing: float) -> None:
        job.status = "processing"
        job.updated_at = time.time()
        try:
            engine = self.get_engine()
            config = VideoDepthConfig(
                model_id=settings.MODEL_ID,
                batch_size=settings.BATCH_SIZE,
                max_side=max_side if max_side is not None else settings.MAX_SIDE,
                invert=invert,
                smoothing=smoothing,
            )

            def progress_cb(done: int, total: Optional[int]) -> None:
                job.processed_frames = done
                job.total_frames = total
                job.updated_at = time.time()

            process_video(
                str(job.input_path),
                str(job.output_path),
                config,
                engine=engine,
                progress_cb=progress_cb,
            )
            job.status = "completed"
        except Exception as exc:  # noqa: BLE001 - reported on the job, not raised
            job.status = "failed"
            job.error = str(exc)
            traceback.print_exc()
        finally:
            job.updated_at = time.time()
            job.input_path.unlink(missing_ok=True)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        job.input_path.unlink(missing_ok=True)
        job.output_path.unlink(missing_ok=True)
        return True

    def cleanup_expired(self) -> None:
        cutoff = time.time() - settings.JOB_RETENTION_HOURS * 3600
        with self._lock:
            expired = [j for j in self._jobs.values() if j.created_at < cutoff]
            for job in expired:
                del self._jobs[job.id]
        for job in expired:
            job.input_path.unlink(missing_ok=True)
            job.output_path.unlink(missing_ok=True)


job_manager = JobManager()
