"""
Detect stabilized scene/base frames from a sampled frame cache.

Input is a cache directory produced by:
    python -m pipeline.sample_cache --input lecture.mp4 --output sample_cache/

This pass reads sampled_frames.avi + sampled_manifest.json, detects scene
transitions on cached frames, and writes small preview base images plus
scene_transitions.json containing original frame_no/timestamp mappings.

If --regions is given, only segments with type=slide are processed. Video and
other non-slide regions are hard boundaries: pending transitions are discarded
and the scene detector state is reset when the next slide region starts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2

try:
    from .sample_cache import iter_sample_cache, load_sample_cache
    from .scene_transition_probe import (
        ProbeConfig,
        compute_mse,
        compute_phash,
        is_duplicate_scene,
        save_scene,
        to_decision_frame,
        transition_reason,
    )
except ImportError:  # Allows direct script execution during local debugging.
    from sample_cache import iter_sample_cache, load_sample_cache
    from scene_transition_probe import (
        ProbeConfig,
        compute_mse,
        compute_phash,
        is_duplicate_scene,
        save_scene,
        to_decision_frame,
        transition_reason,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_slide_regions(regions_path: str | None, guard_samples: int = 0) -> list[dict]:
    if not regions_path:
        return []
    path = Path(regions_path)
    if not path.exists():
        raise FileNotFoundError(f"Region timeline not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = sorted(payload.get("segments", []), key=lambda item: int(item["start_sample_index"]))
    regions = []
    for i, seg in enumerate(segments):
        if seg.get("type") != "slide":
            continue
        start_sample_index = int(seg["start_sample_index"])
        end_sample_index = int(seg["end_sample_index"])
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        if guard_samples > 0 and prev_seg is not None and prev_seg.get("type") != "slide":
            start_sample_index += guard_samples
        if guard_samples > 0 and next_seg is not None and next_seg.get("type") != "slide":
            end_sample_index -= guard_samples
        if start_sample_index > end_sample_index:
            continue
        regions.append({
            "segment_index": int(seg["segment_index"]),
            "type": seg.get("type", ""),
            "start_sample_index": start_sample_index,
            "end_sample_index": end_sample_index,
            "original_start_sample_index": int(seg["start_sample_index"]),
            "original_end_sample_index": int(seg["end_sample_index"]),
            "start_frame_no": int(seg["start_frame_no"]),
            "end_frame_no": int(seg["end_frame_no"]),
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
        })
    return sorted(regions, key=lambda item: item["start_sample_index"])


def _region_for_sample(
    sample_index: int,
    regions: list[dict],
    current_pos: int,
) -> tuple[dict | None, int]:
    if not regions:
        return None, current_pos
    pos = current_pos
    while pos < len(regions) and sample_index > regions[pos]["end_sample_index"]:
        pos += 1
    if pos >= len(regions):
        return None, pos
    region = regions[pos]
    if region["start_sample_index"] <= sample_index <= region["end_sample_index"]:
        return region, pos
    return None, pos


def _save_cache_scene(
    out_dir: Path,
    scene_index: int,
    frame,
    frame_info: dict,
    reason: str,
    details: dict,
) -> dict:
    record = save_scene(
        out_dir,
        scene_index,
        frame,
        int(frame_info["frame_no"]),
        float(frame_info["timestamp_sec"]),
        reason,
        details,
    )
    record["sample_index"] = int(frame_info["sample_index"])
    return record


def _scene_time(record: dict) -> float:
    return float(
        record.get(
            "scene_start_sec",
            record.get("base_timestamp_sec", record.get("timestamp_sec", 0.0)),
        )
        or 0.0
    )


def _same_region(a: dict, b: dict) -> bool:
    return a.get("region_segment_index") == b.get("region_segment_index")


def _remove_pruned_scene_previews(out_dir: Path, pruned_records: list[dict]) -> None:
    for record in pruned_records:
        filename = record.get("filename")
        if filename:
            (out_dir / str(filename)).unlink(missing_ok=True)


def prune_transition_middle_frames(
    records: list[dict],
    out_dir: Path,
    max_gap_sec: float = 3.0,
    min_cluster_scenes: int = 3,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Remove only the middle candidates from rapid transition clusters.

    A real fast slide change often looks like:

      clean slide A -> transition frames -> clean slide B

    The first and last candidates are therefore kept. Only candidates between
    them are pruned. This is deliberately conservative: a lone quick pair is
    left intact, and the final candidate in a burst is never removed here.
    """
    if len(records) < min_cluster_scenes:
        return records, [], []

    kept: list[dict] = []
    pruned: list[dict] = []
    review_candidates: list[dict] = []
    i = 0

    while i < len(records):
        cluster = [records[i]]
        j = i + 1

        while j < len(records):
            current = records[j]
            prev = cluster[-1]
            if not _same_region(prev, current):
                break
            gap = _scene_time(current) - _scene_time(prev)
            if gap < 0 or gap > max_gap_sec:
                break
            cluster.append(current)
            j += 1

        if len(cluster) >= min_cluster_scenes:
            kept.append(cluster[0])
            kept.append(cluster[-1])
            middle = cluster[1:-1]
            review_candidates.append({
                "reason": "transition_cluster",
                "cluster_scene_indices": [int(x["scene_index"]) for x in cluster],
                "context_scene_indices": [
                    int(cluster[0]["scene_index"]),
                    int(cluster[-1]["scene_index"]),
                ],
                "middle_scene_indices": [int(x["scene_index"]) for x in middle],
                "cluster_start_sec": _scene_time(cluster[0]),
                "cluster_end_sec": _scene_time(cluster[-1]),
                "max_adjacent_gap_sec": max_gap_sec,
                "candidate_records": [
                    {
                        "scene_index": item.get("scene_index"),
                        "filename": item.get("filename"),
                        "scene_start_sec": item.get("scene_start_sec"),
                        "base_timestamp_sec": item.get("base_timestamp_sec"),
                        "frame_no": item.get("frame_no"),
                        "base_frame_no": item.get("base_frame_no"),
                        "reason": item.get("reason"),
                    }
                    for item in cluster
                ],
            })
            for item in middle:
                pruned.append({
                    "scene_index": item.get("scene_index"),
                    "filename": item.get("filename"),
                    "scene_start_sec": item.get("scene_start_sec"),
                    "base_timestamp_sec": item.get("base_timestamp_sec"),
                    "reason": item.get("reason"),
                    "prune_reason": "transition_middle_frame",
                    "cluster_start_scene_index": cluster[0].get("scene_index"),
                    "cluster_end_scene_index": cluster[-1].get("scene_index"),
                    "cluster_gap_sec": max_gap_sec,
                })
            log.info(
                "[prune] transition cluster %s-%s: removed %s middle candidates (%s)",
                cluster[0].get("scene_index"),
                cluster[-1].get("scene_index"),
                len(middle),
                ", ".join(str(x.get("scene_index")) for x in middle),
            )
            i = j
        else:
            kept.append(cluster[0])
            i += 1

    if pruned:
        _remove_pruned_scene_previews(out_dir, pruned)
    return kept, pruned, review_candidates


def run_cache_probe(
    cache_dir: str,
    output_dir: str,
    cfg: ProbeConfig,
    regions_path: str | None = None,
    region_guard_sec: float = 1.0,
    prune_bursts: bool = True,
    transient_burst_gap_sec: float = 3.0,
    transient_burst_min_extra_scenes: int = 2,
) -> list[dict]:
    manifest = load_sample_cache(cache_dir)
    sampled_fps = float(manifest["cache"]["sampled_fps"])
    sample_count = int(manifest["cache"]["sample_count"])
    guard_samples = max(0, int(round(region_guard_sec * sampled_fps))) if regions_path else 0
    slide_regions = _load_slide_regions(regions_path, guard_samples=guard_samples)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("scene_*.jpg"):
        stale.unlink(missing_ok=True)

    stable_frames_required = max(2, int(cfg.delay_sec * sampled_fps))
    pending_max_frames = max(stable_frames_required, int(cfg.max_pending_sec * sampled_fps))

    records: list[dict] = []
    scene_index = 0
    processed = 0
    skipped = 0
    base_decision = None
    last_saved_base_decision = None
    prev_decision = None
    prev_hash = None
    pending = None
    active_region = None
    region_pos = 0

    log.info(
        "cache scene probe start: cache=%s samples=%s sampled_fps=%.3f slide_regions=%s",
        cache_dir,
        sample_count,
        sampled_fps,
        len(slide_regions) if slide_regions else "all",
    )

    for frame_info, frame in iter_sample_cache(cache_dir):
        processed += 1
        sample_index = int(frame_info["sample_index"])
        region = None
        if slide_regions:
            region, region_pos = _region_for_sample(sample_index, slide_regions, region_pos)
            if region is None:
                skipped += 1
                if active_region is not None:
                    log.info(
                        "[region %03d] leave slide region @ %.3fs frame=%s",
                        int(active_region["segment_index"]),
                        float(frame_info["timestamp_sec"]),
                        int(frame_info["frame_no"]),
                    )
                active_region = None
                base_decision = None
                prev_decision = None
                prev_hash = None
                pending = None
                continue
            if active_region is None or active_region["segment_index"] != region["segment_index"]:
                active_region = region
                base_decision = None
                prev_decision = None
                prev_hash = None
                pending = None
                log.info(
                    "[region %03d] enter slide region %.3f-%.3fs",
                    int(region["segment_index"]),
                    float(region["start_sec"]),
                    float(region["end_sec"]),
                )

        decision = to_decision_frame(frame, cfg.resize_width)
        decision_hash = compute_phash(decision)

        if base_decision is None:
            base_decision = decision.copy()
            prev_decision = decision.copy()
            prev_hash = decision_hash
            if (
                last_saved_base_decision is not None
                and is_duplicate_scene(last_saved_base_decision, decision, cfg)
            ):
                log.info(
                    "[suppress] duplicate region first frame @ %.3fs frame=%s",
                    float(frame_info["timestamp_sec"]),
                    int(frame_info["frame_no"]),
                )
            else:
                scene_index += 1
                reason = "region_first_frame" if slide_regions else "first_frame"
                record = _save_cache_scene(out_dir, scene_index, frame, frame_info, reason, {})
                record["scene_start_frame_no"] = int(frame_info["frame_no"])
                record["scene_start_sec"] = float(frame_info["timestamp_sec"])
                record["base_frame_no"] = int(frame_info["frame_no"])
                record["base_timestamp_sec"] = float(frame_info["timestamp_sec"])
                if active_region is not None:
                    record["region_segment_index"] = int(active_region["segment_index"])
                    record["region_start_sec"] = float(active_region["start_sec"])
                    record["region_end_sec"] = float(active_region["end_sec"])
                records.append(record)
                last_saved_base_decision = decision.copy()
            continue

        if pending is not None:
            anchor_mse = compute_mse(pending["anchor_decision"], decision)
            anchor_hash_dist = int(pending["anchor_hash"] - decision_hash)
            prev_pending_mse = compute_mse(pending["last_decision"], decision)
            prev_pending_hash_dist = int(pending["last_hash"] - decision_hash)
            pending["observed"] += 1

            if (
                anchor_mse <= cfg.stable_mse
                and anchor_hash_dist <= cfg.stable_hash
                and prev_pending_mse <= cfg.stable_prev_mse
                and prev_pending_hash_dist <= cfg.stable_prev_hash
            ):
                pending["stable"] += 1
                pending.update({
                    "frame": frame.copy(),
                    "decision": decision.copy(),
                    "frame_info": dict(frame_info),
                    "hash": decision_hash,
                })
            else:
                pending.update({
                    "anchor_decision": decision.copy(),
                    "anchor_hash": decision_hash,
                    "frame": frame.copy(),
                    "decision": decision.copy(),
                    "frame_info": dict(frame_info),
                    "hash": decision_hash,
                    "stable": 1,
                })

            pending["last_decision"] = decision.copy()
            pending["last_hash"] = decision_hash

            if pending["stable"] >= stable_frames_required or pending["observed"] >= pending_max_frames:
                if base_decision is not None and is_duplicate_scene(base_decision, pending["decision"], cfg):
                    log.info(
                        "[suppress] duplicate pending scene @ %.3fs frame=%s",
                        float(pending["frame_info"]["timestamp_sec"]),
                        int(pending["frame_info"]["frame_no"]),
                    )
                else:
                    scene_index += 1
                    start_info = dict(pending["start_frame_info"])
                    save_info = dict(pending["frame_info"])
                    record = _save_cache_scene(
                        out_dir,
                        scene_index,
                        pending["frame"],
                        save_info,
                        pending["reason"] + "_stabilized",
                        pending["details"],
                    )
                    record["scene_start_frame_no"] = int(start_info["frame_no"])
                    record["scene_start_sec"] = float(start_info["timestamp_sec"])
                    record["base_frame_no"] = int(save_info["frame_no"])
                    record["base_timestamp_sec"] = float(save_info["timestamp_sec"])
                    if active_region is not None:
                        record["region_segment_index"] = int(active_region["segment_index"])
                        record["region_start_sec"] = float(active_region["start_sec"])
                        record["region_end_sec"] = float(active_region["end_sec"])
                    records.append(record)
                    base_decision = pending["decision"].copy()
                    last_saved_base_decision = pending["decision"].copy()

                prev_decision = decision.copy()
                prev_hash = decision_hash
                pending = None
            continue

        assert base_decision is not None and prev_decision is not None and prev_hash is not None
        reason, details = transition_reason(base_decision, prev_decision, decision, prev_hash, decision_hash, cfg)
        if reason is not None:
            pending = {
                "start_frame_info": dict(frame_info),
                "anchor_decision": decision.copy(),
                "anchor_hash": decision_hash,
                "last_decision": decision.copy(),
                "last_hash": decision_hash,
                "frame": frame.copy(),
                "decision": decision.copy(),
                "frame_info": dict(frame_info),
                "hash": decision_hash,
                "stable": 1,
                "observed": 1,
                "reason": reason,
                "details": details,
            }
            log.info(
                "[pending] %s @ %.3fs frame=%s",
                reason,
                float(frame_info["timestamp_sec"]),
                int(frame_info["frame_no"]),
            )
            continue

        prev_decision = decision.copy()
        prev_hash = decision_hash

        if processed % 1000 == 0:
            pct = (processed / sample_count * 100.0) if sample_count > 0 else 0.0
            log.info("processed=%s/%s %.1f%%", processed, sample_count, pct)

    if pending is not None:
        if base_decision is None or not is_duplicate_scene(base_decision, pending["decision"], cfg):
            scene_index += 1
            record = _save_cache_scene(
                out_dir,
                scene_index,
                pending["frame"],
                pending["frame_info"],
                pending["reason"] + "_flush",
                pending["details"],
            )
            record["scene_start_frame_no"] = int(pending["start_frame_info"]["frame_no"])
            record["scene_start_sec"] = float(pending["start_frame_info"]["timestamp_sec"])
            record["base_frame_no"] = int(pending["frame_info"]["frame_no"])
            record["base_timestamp_sec"] = float(pending["frame_info"]["timestamp_sec"])
            records.append(record)
            last_saved_base_decision = pending["decision"].copy()

    pruned_records: list[dict] = []
    review_candidates: list[dict] = []
    if prune_bursts:
        records, pruned_records, review_candidates = prune_transition_middle_frames(
            records,
            out_dir,
            max_gap_sec=max(0.0, transient_burst_gap_sec),
            min_cluster_scenes=max(3, transient_burst_min_extra_scenes + 1),
        )

    payload = {
        "cache_dir": str(cache_dir),
        "source_input": manifest.get("input_path"),
        "regions_path": str(regions_path) if regions_path else None,
        "region_guard_sec": region_guard_sec if regions_path else 0.0,
        "postprocess": {
            "prune_transition_middle_frames": prune_bursts,
            "transient_burst_gap_sec": transient_burst_gap_sec,
            "transition_min_cluster_scenes": max(3, transient_burst_min_extra_scenes + 1),
            "pruned_count": len(pruned_records),
            "pruned_records": pruned_records,
            "review_candidate_count": len(review_candidates),
            "review_candidates": review_candidates,
        },
        "config": asdict(cfg),
        "cache": manifest.get("cache"),
        "source": manifest.get("source"),
        "slide_regions": slide_regions,
        "processed_samples": processed,
        "skipped_samples": skipped,
        "scene_count": len(records),
        "scenes": records,
    }
    with open(out_dir / "scene_transitions.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info("cache scene probe done: scenes=%s output=%s", len(records), out_dir)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect scene transitions from a sampled frame cache.")
    parser.add_argument("--cache", required=True, help="Sample cache directory")
    parser.add_argument("--output", "-o", required=True, help="Output scene probe directory")
    parser.add_argument("--regions", help="timeline_segments.json from Step 1; only type=slide regions are processed")
    parser.add_argument("--region-guard-sec", type=float, default=1.0, help="Shrink slide regions next to non-slide regions by this many seconds")
    parser.add_argument("--no-prune-transient-bursts", action="store_true", help="Disable rapid transition-cluster middle-frame pruning")
    parser.add_argument("--transient-burst-gap-sec", type=float, default=3.0, help="Max gap between adjacent scene candidates in one transition cluster")
    parser.add_argument("--transient-burst-min-extra-scenes", type=int, default=2, help="Legacy option: default 2 means prune only clusters with 3+ candidates")
    parser.add_argument("--resize-width", type=int, default=ProbeConfig.resize_width)
    parser.add_argument("--delay-sec", type=float, default=ProbeConfig.delay_sec)
    parser.add_argument("--max-pending-sec", type=float, default=ProbeConfig.max_pending_sec)
    parser.add_argument("--stable-mse", type=float, default=ProbeConfig.stable_mse)
    parser.add_argument("--stable-prev-mse", type=float, default=ProbeConfig.stable_prev_mse)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = ProbeConfig(
        resize_width=max(160, args.resize_width),
        delay_sec=max(0.0, args.delay_sec),
        max_pending_sec=max(args.delay_sec, args.max_pending_sec),
        stable_mse=max(0.0, args.stable_mse),
        stable_prev_mse=max(0.0, args.stable_prev_mse),
    )
    run_cache_probe(
        args.cache,
        args.output,
        cfg,
        regions_path=args.regions,
        region_guard_sec=max(0.0, args.region_guard_sec),
        prune_bursts=not args.no_prune_transient_bursts,
        transient_burst_gap_sec=max(0.0, args.transient_burst_gap_sec),
        transient_burst_min_extra_scenes=max(1, args.transient_burst_min_extra_scenes),
    )


if __name__ == "__main__":
    main()
