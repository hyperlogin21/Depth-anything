"""Video -> black-and-white depth video pipeline built on DepthEngine.

Used by both the standalone CLI script (video_to_depth.py) and the API
(api/jobs.py) so there is exactly one place that implements the conversion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np

from .depth_engine import DepthEngine

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Near-visually-lossless x264 quality. Depth maps are used as conditioning
# input for downstream models, so we avoid the banding a low-bitrate encode
# would introduce in smooth gradients.
DEFAULT_CRF = "16"


@dataclass
class VideoDepthConfig:
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    batch_size: int = 4
    max_side: Optional[int] = None
    invert: bool = False
    # EMA factor (0-1) used to smooth the per-frame min/max normalization
    # window across time. 0 disables smoothing (pure per-frame normalization,
    # which flickers); closer to 1 is smoother but reacts more slowly to
    # scene/cut changes.
    smoothing: float = 0.9
    cache_dir: Optional[str] = None
    device: Optional[str] = None


def _resize_for_inference(frame: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class _TemporalNormalizer:
    """Smooths per-frame min/max with an EMA so the depth video doesn't flicker."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.min_v: Optional[float] = None
        self.max_v: Optional[float] = None

    def normalize(self, depth: np.ndarray, invert: bool) -> np.ndarray:
        frame_min, frame_max = float(depth.min()), float(depth.max())
        if self.min_v is None or self.alpha <= 0:
            self.min_v, self.max_v = frame_min, frame_max
        else:
            self.min_v = self.alpha * self.min_v + (1 - self.alpha) * frame_min
            self.max_v = self.alpha * self.max_v + (1 - self.alpha) * frame_max
        span = max(self.max_v - self.min_v, 1e-6)
        norm = np.clip((depth - self.min_v) / span, 0.0, 1.0)
        if invert:
            norm = 1.0 - norm
        gray = (norm * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)


def process_video(
    input_path: str,
    output_path: str,
    config: VideoDepthConfig,
    engine: Optional[DepthEngine] = None,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> None:
    """Reads `input_path`, writes a grayscale depth video to `output_path`.

    `progress_cb(processed_frames, total_frames_or_None)` is called after
    every processed batch, so callers (e.g. the API job manager) can report
    progress.
    """
    owns_engine = engine is None
    if engine is None:
        engine = DepthEngine(config.model_id, device=config.device, cache_dir=config.cache_dir)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=None,
        pixelformat="yuv420p",
        output_params=["-crf", DEFAULT_CRF, "-preset", "medium"],
    )
    normalizer = _TemporalNormalizer(config.smoothing)

    processed = 0
    batch: List[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            batch.append(frame_rgb)
            if len(batch) >= config.batch_size:
                processed = _run_batch(
                    batch, engine, normalizer, config, writer, processed, progress_cb, total_frames
                )
                batch = []
        if batch:
            processed = _run_batch(
                batch, engine, normalizer, config, writer, processed, progress_cb, total_frames
            )
        if processed == 0:
            raise ValueError(f"No frames could be read from: {input_path}")
    finally:
        cap.release()
        writer.close()
        if owns_engine:
            del engine


def _run_batch(
    batch: List[np.ndarray],
    engine: DepthEngine,
    normalizer: _TemporalNormalizer,
    config: VideoDepthConfig,
    writer,
    processed: int,
    progress_cb: Optional[Callable[[int, Optional[int]], None]],
    total_frames: Optional[int],
) -> int:
    infer_frames = [_resize_for_inference(f, config.max_side) for f in batch]
    target_sizes = [f.shape[:2] for f in batch]
    depths = engine.infer_batch(infer_frames, target_sizes=target_sizes)

    for depth in depths:
        gray = normalizer.normalize(depth, config.invert)
        writer.append_data(gray)
        processed += 1

    if progress_cb:
        progress_cb(processed, total_frames)
    return processed
