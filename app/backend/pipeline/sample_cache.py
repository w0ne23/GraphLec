"""
Build a lightweight sampled-frame cache from an input video.

The cache is the shared coordinate system for later passes:
  1. practice/video/slide region classification
  2. scene/base detection
  3. annotation detection

It stores resized sampled frames plus a manifest that maps every sample back to
the original frame number.

Usage:
    python -m pipeline.sample_cache --input lecture.mp4 --output sample_cache/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imagehash
import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


MANIFEST_FILENAME = "sampled_manifest.json"
VIDEO_FILENAME = "sampled_frames.avi"
SCHEMA_VERSION = 1


@dataclass
class SampleCacheConfig:
    sample_every: int = 2
    resize_width: int = 768
    jpeg_quality: int = 95


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def to_decision_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32) if frame_a.ndim == 2 else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = frame_b.astype(np.float32) if frame_b.ndim == 2 else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash_int(frame: np.ndarray) -> int:
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return int(str(imagehash.phash(pil_img)), 16)


def phash_distance_int(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def read_video_metadata(input_path: str) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video metadata: {input_path}")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if frame_count > 0 else 0.0,
    }


def create_sample_cache(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig | None = None,
) -> dict:
    cfg = cfg or SampleCacheConfig()
    cfg.sample_every = max(1, int(cfg.sample_every))
    cfg.resize_width = max(160, int(cfg.resize_width))

    video_meta = read_video_metadata(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_path = output_path / VIDEO_FILENAME
    manifest_path = output_path / MANIFEST_FILENAME
    video_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)

    cached_height = int(video_meta["height"] * (cfg.resize_width / video_meta["width"]))
    sampled_fps = max(1.0, video_meta["fps"] / cfg.sample_every)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        sampled_fps,
        (cfg.resize_width, cached_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open sample cache writer: {video_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        writer.release()
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    frames: list[dict] = []
    prev_decision = None
    prev_phash = None
    frame_no = 0
    sample_index = 0
    progress_interval = 2000

    log.info(
        "sample cache start: input=%s fps=%.2f frames=%s sample_every=%s size=%sx%s",
        input_path,
        video_meta["fps"],
        video_meta["frame_count"],
        cfg.sample_every,
        cfg.resize_width,
        cached_height,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if frame_no % cfg.sample_every != 0:
                continue

            sample_index += 1
            small = resize_frame(frame, cfg.resize_width)
            if small.shape[1] != cfg.resize_width or small.shape[0] != cached_height:
                small = cv2.resize(small, (cfg.resize_width, cached_height), interpolation=cv2.INTER_AREA)

            decision = to_decision_frame(small)
            phash_int = compute_phash_int(decision)
            prev_mse = compute_mse(prev_decision, decision) if prev_decision is not None else None
            prev_hash_dist = (
                phash_distance_int(prev_phash, phash_int)
                if prev_phash is not None
                else None
            )

            writer.write(small)
            frames.append({
                "sample_index": sample_index,
                "frame_no": frame_no,
                "timestamp_sec": round(frame_no / video_meta["fps"], 6),
                "phash_int": phash_int,
                "prev_mse": round(prev_mse, 6) if prev_mse is not None else None,
                "prev_hash_dist": prev_hash_dist,
            })

            prev_decision = decision
            prev_phash = phash_int

            if sample_index % progress_interval == 0:
                pct = (frame_no / video_meta["frame_count"] * 100.0) if video_meta["frame_count"] > 0 else 0.0
                log.info("sample cache progress: samples=%s frame=%s %.1f%%", sample_index, frame_no, pct)
    finally:
        cap.release()
        writer.release()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_path": input_path,
        "video_filename": VIDEO_FILENAME,
        "config": asdict(cfg),
        "source": video_meta,
        "cache": {
            "sampled_fps": sampled_fps,
            "width": cfg.resize_width,
            "height": cached_height,
            "sample_count": sample_index,
        },
        "frames": frames,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("sample cache done: samples=%s output=%s", sample_index, output_path)
    return manifest


def load_sample_cache(cache_dir: str | Path) -> dict:
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sample cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_sample_cache(cache_dir: str | Path) -> Iterator[tuple[dict, np.ndarray]]:
    cache_path = Path(cache_dir)
    manifest = load_sample_cache(cache_path)
    video_path = cache_path / manifest["video_filename"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open sampled cache video: {video_path}")

    try:
        for frame_info in manifest.get("frames", []):
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Sample cache video ended early: {video_path}")
            yield frame_info, frame
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled frame cache for video analysis passes.")
    parser.add_argument("--input", "-i", required=True, help="Input .mp4 path")
    parser.add_argument("--output", "-o", required=True, help="Output cache directory")
    parser.add_argument("--sample-every", type=int, default=SampleCacheConfig.sample_every)
    parser.add_argument("--resize-width", type=int, default=SampleCacheConfig.resize_width)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = SampleCacheConfig(
        sample_every=args.sample_every,
        resize_width=args.resize_width,
    )
    create_sample_cache(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
