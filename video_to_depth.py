#!/usr/bin/env python3
"""Convert a video into a black-and-white depth video using Depth-Anything V2.

The output is a grayscale (equal RGB channels) mp4 where pixel brightness
encodes relative distance to the camera, matching the depth-conditioning
input format expected by controlnet-style / depth-conditioned video models
such as Seedance. Use --invert if your target model expects the opposite
convention (near=black instead of near=white).

Examples:
  python video_to_depth.py input.mp4 output_depth.mp4
  python video_to_depth.py input.mp4 output_depth.mp4 --model base --max-side 896
  python video_to_depth.py input.mp4 output_depth.mp4 --invert --smoothing 0.85
"""

from __future__ import annotations

import argparse
import sys
import time

from core.depth_video import VideoDepthConfig, process_video

MODEL_ALIASES = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Path to the source video")
    parser.add_argument("output", help="Path to write the depth video (.mp4)")
    parser.add_argument(
        "--model",
        default="small",
        help="small|base|large, or a full Hugging Face model id (default: small)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Frames per inference batch")
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Downscale the longest side to this many pixels before inference for speed; "
        "the output is always resized back to the source resolution",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert so far=white/near=black instead of the default near=white/far=black",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="Temporal EMA smoothing factor in [0, 1) for the normalization window "
        "(0 disables smoothing and flickers more; default 0.9)",
    )
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto-detect)")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face model cache directory")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model_id = MODEL_ALIASES.get(args.model, args.model)

    config = VideoDepthConfig(
        model_id=model_id,
        batch_size=args.batch_size,
        max_side=args.max_side,
        invert=args.invert,
        smoothing=args.smoothing,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    def progress(done: int, total) -> None:
        if total:
            pct = 100 * done / total
            print(f"\r  {done}/{total} frames ({pct:.1f}%)", end="", file=sys.stderr, flush=True)
        else:
            print(f"\r  {done} frames", end="", file=sys.stderr, flush=True)

    print(f"Loading {model_id} ...", file=sys.stderr)
    start = time.time()
    try:
        process_video(args.input, args.output, config, progress_cb=progress)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFailed: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone in {time.time() - start:.1f}s -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
