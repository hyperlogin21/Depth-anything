"""Thin wrapper around a Depth-Anything V2 checkpoint from Hugging Face."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEngine:
    """Loads a Depth-Anything V2 model once and runs batched depth inference."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 roughly halves VRAM and time on GPU; CPU always runs in fp32.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id, cache_dir=cache_dir)
            .to(self.device, dtype=self.dtype)
            .eval()
        )

    @torch.inference_mode()
    def infer_batch(
        self,
        frames: List[np.ndarray],
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[np.ndarray]:
        """Runs depth estimation on a batch of RGB frames (H, W, 3 uint8 arrays).

        Returns one float32 raw (unnormalized) depth map per frame, resized to
        `target_sizes[i]` (or the input frame's own size if not given).
        """
        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        predicted = self.model(**inputs).predicted_depth  # (B, h', w')

        sizes = target_sizes or [f.shape[:2] for f in frames]
        depths = []
        for i, (h, w) in enumerate(sizes):
            depth = torch.nn.functional.interpolate(
                predicted[i].float().unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            depths.append(depth.squeeze(0).squeeze(0).cpu().numpy())
        return depths
