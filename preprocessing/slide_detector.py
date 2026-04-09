#!/usr/bin/env python3
"""
슬라이드 전환 감지 및 추출기

- 4fps 샘플링으로 빠른 분석 (전체 프레임 디코딩 불필요)
- 3.0초 이하 구간 자동 스킵
- pHash 기반 중복 슬라이드 제거 (뒤로 돌아가는 경우 포함)
- 각 슬라이드마다 시작 프레임(깨끗한) + 종료 프레임(변경분 포함) 저장

사용법:
  python slide_detector.py <video_path>
  python slide_detector.py <video_path> -o ./output --min-duration 3.0
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# ── 파라미터 ──────────────────────────────────────────────
SAMPLE_FPS       = 4      # 분석용 샘플링 FPS
MIN_DURATION     = float(os.getenv("SLIDE_MIN_DURATION_SEC", "3.0"))    # 이 초 이하 구간은 스킵
TRANSITION_THR   = 5      # pHash 해밍 거리: 슬라이드 전환 판정
DUPLICATE_THR    = 4      # pHash 해밍 거리: 중복 슬라이드 판정
START_OFFSET     = 0.5    # 전환 직후 이 초 뒤 프레임을 "시작"으로 저장
END_OFFSET       = 0.5    # 다음 전환 이 초 전 프레임을 "종료"으로 저장
END_ALIGN_MAX_IMG_DIST = int(os.getenv("SLIDE_END_ALIGN_MAX_IMG_DIST", "16"))
END_ALIGN_BACKTRACK_SEC = float(os.getenv("SLIDE_END_ALIGN_BACKTRACK_SEC", "2.0"))
END_ALIGN_STEP_SEC = float(os.getenv("SLIDE_END_ALIGN_STEP_SEC", "0.25"))
RELAXED_TRANSITION_THR = 3
RELAXED_DUPLICATE_THR = 1
RELAX_MIN_VIDEO_SEC = 300
RELAX_MAX_PERIODS = 5
TIGHT_TRANSITION_CANDIDATES = (8, 10)
OVERSEG_MIN_PERIODS = 40
OVERSEG_DENSITY_PER_MIN = 2.5
OVERSEG_SHORT_RATIO = 0.30
OVERSEG_SHORT_SEC = 4.0
MIN_PERIODS_AFTER_TIGHTEN = 8
OCR_REFINE_ENABLED = os.getenv("SLIDE_OCR_REFINE", "1") != "0"
OCR_LANG = os.getenv("SLIDE_OCR_LANG", "kor+eng")
OCR_PSM = os.getenv("SLIDE_OCR_PSM", "6")
OCR_MAX_PHASH_DIST = int(os.getenv("SLIDE_OCR_MAX_PHASH_DIST", "16"))
OCR_BOUNDARY_MERGE_MAX_IMG_DIST = int(os.getenv("SLIDE_OCR_BOUNDARY_MERGE_MAX_IMG_DIST", "12"))
OCR_EMPTY_VISUAL_MERGE_MAX_IMG_DIST = int(os.getenv("SLIDE_OCR_EMPTY_VISUAL_MERGE_MAX_IMG_DIST", "3"))
OCR_MIN_TEXT_CHARS = int(os.getenv("SLIDE_OCR_MIN_TEXT_CHARS", "12"))
OCR_MIN_TOKENS = int(os.getenv("SLIDE_OCR_MIN_TOKENS", "3"))
OCR_CONTAINMENT_THR = float(os.getenv("SLIDE_OCR_CONTAINMENT_THR", "0.84"))
OCR_JACCARD_THR = float(os.getenv("SLIDE_OCR_JACCARD_THR", "0.55"))
OCR_LEN_RATIO_MIN = float(os.getenv("SLIDE_OCR_LEN_RATIO_MIN", "0.45"))
OCR_EXACT_ANCHOR_ENABLED = os.getenv("SLIDE_OCR_EXACT_ANCHOR", "1") != "0"
OCR_ANCHOR_MIN_CHARS = int(os.getenv("SLIDE_OCR_ANCHOR_MIN_CHARS", "14"))
OCR_ANCHOR_MIN_TOKENS = int(os.getenv("SLIDE_OCR_ANCHOR_MIN_TOKENS", "4"))
OCR_ANCHOR_TOP_K = int(os.getenv("SLIDE_OCR_ANCHOR_TOP_K", "3"))
OCR_INTERIOR_SPLIT_ENABLED = os.getenv("SLIDE_OCR_INTERIOR_SPLIT", "1") != "0"
OCR_INTERIOR_SPLIT_MIN_SEC = float(os.getenv("SLIDE_OCR_INTERIOR_SPLIT_MIN_SEC", "60"))
OCR_INTERIOR_SPLIT_STEP_SEC = float(os.getenv("SLIDE_OCR_INTERIOR_SPLIT_STEP_SEC", "8"))
OCR_INTERIOR_SPLIT_MIN_GAP_SEC = float(os.getenv("SLIDE_OCR_INTERIOR_SPLIT_MIN_GAP_SEC", "8"))
OCR_INTERIOR_SPLIT_MAX_PER_PERIOD = int(os.getenv("SLIDE_OCR_INTERIOR_SPLIT_MAX_PER_PERIOD", "6"))
OCR_INTERIOR_SPLIT_HASH_DRIFT = int(os.getenv("SLIDE_OCR_INTERIOR_SPLIT_HASH_DRIFT", "18"))
OCR_INTERIOR_SPLIT_HASH_TARGET_RATIO = float(os.getenv("SLIDE_OCR_INTERIOR_SPLIT_HASH_TARGET_RATIO", "0.55"))
DUP_FALLBACK_OCR_ENABLED = os.getenv("SLIDE_DUP_FALLBACK_OCR", "1") != "0"
DUP_FALLBACK_MAX_END_DIST = int(os.getenv("SLIDE_DUP_FALLBACK_MAX_END_DIST", "8"))
DUP_FALLBACK_MAX_START_DIST = int(os.getenv("SLIDE_DUP_FALLBACK_MAX_START_DIST", "12"))
DUP_FALLBACK_MAX_GAP_SEC = float(os.getenv("SLIDE_DUP_FALLBACK_MAX_GAP_SEC", "120"))
INTERIOR_SPLIT_ENABLED = os.getenv("SLIDE_INTERIOR_SPLIT", "1") != "0"
INTERIOR_SPLIT_MIN_SEC = float(os.getenv("SLIDE_INTERIOR_SPLIT_MIN_SEC", "45"))
INTERIOR_SPLIT_MIN_DRIFT = int(os.getenv("SLIDE_INTERIOR_SPLIT_MIN_DRIFT", "12"))
INTERIOR_SPLIT_DELTA_THR = int(os.getenv("SLIDE_INTERIOR_SPLIT_DELTA_THR", "2"))
INTERIOR_SPLIT_MIN_SEG_SEC = float(os.getenv("SLIDE_INTERIOR_SPLIT_MIN_SEG_SEC", "2.0"))
INTERIOR_SPLIT_MAX_NEW_SEGS = int(os.getenv("SLIDE_INTERIOR_SPLIT_MAX_NEW_SEGS", "10"))
INTERIOR_SPLIT_HOLD_FRAMES = int(os.getenv("SLIDE_INTERIOR_SPLIT_HOLD_FRAMES", "3"))


# ── 영상 메타데이터 ───────────────────────────────────────

def get_video_meta(v_path: str) -> tuple[int, int, float]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", v_path]
    meta = json.loads(subprocess.check_output(cmd))
    v_s = next(s for s in meta["streams"] if s["codec_type"] == "video")
    w, h = int(v_s["width"]), int(v_s["height"])
    num, den = map(int, v_s["avg_frame_rate"].split("/"))
    if den == 0:
        raise ValueError("avg_frame_rate 분모가 0입니다.")
    return w, h, num / den


# ── pHash 시퀀스 추출 ─────────────────────────────────────

def sample_phashes(v_path: str, w: int, h: int) -> list:
    """4fps 샘플링으로 pHash 배열 반환"""
    proc = subprocess.Popen(
        ["ffmpeg", "-i", v_path,
         "-vf", f"fps={SAMPLE_FPS}",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    hashes = []
    sz = w * h
    try:
        while True:
            data = proc.stdout.read(sz)
            if not data or len(data) < sz:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
            small = Image.fromarray(frame).resize((w // 4, h // 4))
            hashes.append(imagehash.phash(small))
            print(f"\r  pHash 계산: {len(hashes)}프레임 "
                  f"({len(hashes) / SAMPLE_FPS:.1f}초)", end="", flush=True)
    finally:
        proc.terminate()
        proc.wait()
    print()
    return hashes


# ── 안정 구간 탐지 ────────────────────────────────────────

def detect_stable_periods(hashes: list, transition_thr: int = TRANSITION_THR) -> list[tuple[int, int]]:
    """
    연속된 pHash 시퀀스에서 안정 구간을 탐지.
    반환: [(start_idx, end_idx), ...] (샘플 프레임 인덱스 기준)
    """
    if len(hashes) < 2:
        return [(0, len(hashes) - 1)]

    periods = []
    start = 0

    for i in range(1, len(hashes)):
        dist = hashes[i] - hashes[i - 1]
        if dist > transition_thr:
            periods.append((start, i - 1))
            start = i

    periods.append((start, len(hashes) - 1))
    return periods


def refine_periods_by_interior_split(
    periods: list[tuple[int, int]],
    hashes: list,
    transition_thr: int,
) -> tuple[list[tuple[int, int]], dict]:
    """
    긴 구간의 시작/끝 프레임이 크게 다른 경우, 해당 구간만 국소적으로 재탐지해 분할.
    전역 임계값을 낮추지 않고 under-segmentation을 보완한다.
    """
    stats = {
        "enabled": INTERIOR_SPLIT_ENABLED,
        "checked": 0,
        "split_periods": 0,
        "added_segments": 0,
    }
    if not INTERIOR_SPLIT_ENABLED or not periods:
        return periods, stats

    local_thr = max(1, transition_thr - max(1, INTERIOR_SPLIT_DELTA_THR))
    refined: list[tuple[int, int]] = []
    min_seg_frames = max(1, int(round(INTERIOR_SPLIT_MIN_SEG_SEC * SAMPLE_FPS)))
    hold_frames = max(1, INTERIOR_SPLIT_HOLD_FRAMES)

    def _representative_drift(start_idx: int, end_idx: int) -> int:
        # 실제 저장 프레임(start+offset, end-offset)에 가까운 위치로 drift를 본다.
        start_off = int(round(START_OFFSET * SAMPLE_FPS))
        end_off = int(round(END_OFFSET * SAMPLE_FPS))
        rep_start = min(end_idx, start_idx + start_off)
        rep_end = max(start_idx, end_idx - end_off)
        return hashes[rep_end] - hashes[rep_start]

    def _convert_local_periods(start_idx: int, local_periods: list[tuple[int, int]]) -> list[tuple[int, int]]:
        converted = []
        for ls, le in local_periods:
            gs = start_idx + ls
            ge = start_idx + le
            if ge < gs:
                continue
            if (ge - gs + 1) >= min_seg_frames:
                converted.append((gs, ge))
        return converted

    def _fallback_drift_split(start_idx: int, end_idx: int) -> list[tuple[int, int]]:
        # 전환이 완만해서 인접 frame diff가 낮을 때, anchor 대비 누적 drift로 분할 후보 생성.
        split_points = []
        anchor = start_idx
        i = start_idx + min_seg_frames
        max_split_points = max(0, INTERIOR_SPLIT_MAX_NEW_SEGS - 1)

        while i <= end_idx - min_seg_frames and len(split_points) < max_split_points:
            drift = hashes[i] - hashes[anchor]
            if drift >= INTERIOR_SPLIT_MIN_DRIFT:
                hold_ok = True
                for j in range(i, min(i + hold_frames, end_idx + 1)):
                    if (hashes[j] - hashes[anchor]) < INTERIOR_SPLIT_MIN_DRIFT:
                        hold_ok = False
                        break
                if hold_ok:
                    split_points.append(i)
                    anchor = i
                    i += min_seg_frames
                    continue
            i += 1

        if not split_points:
            return []

        bounds = [start_idx] + split_points + [end_idx + 1]
        segments = []
        for a, b in zip(bounds, bounds[1:]):
            seg_start = a
            seg_end = b - 1
            if seg_end >= seg_start and (seg_end - seg_start + 1) >= min_seg_frames:
                segments.append((seg_start, seg_end))
        return segments

    for start_idx, end_idx in periods:
        duration = (end_idx - start_idx + 1) / SAMPLE_FPS
        if duration < INTERIOR_SPLIT_MIN_SEC:
            refined.append((start_idx, end_idx))
            continue

        start_end_drift = _representative_drift(start_idx, end_idx)
        if start_end_drift < INTERIOR_SPLIT_MIN_DRIFT:
            refined.append((start_idx, end_idx))
            continue

        stats["checked"] += 1
        local_hashes = hashes[start_idx:end_idx + 1]
        local_periods = detect_stable_periods(local_hashes, transition_thr=local_thr)
        converted = _convert_local_periods(start_idx, local_periods)
        if len(converted) <= 1 or len(converted) > INTERIOR_SPLIT_MAX_NEW_SEGS:
            converted = _fallback_drift_split(start_idx, end_idx)

        if len(converted) <= 1:
            refined.append((start_idx, end_idx))
            continue

        refined.extend(converted)
        stats["split_periods"] += 1
        stats["added_segments"] += len(converted) - 1

    return refined, stats


# ── OCR 기반 경계 보정 ────────────────────────────────────

def _normalize_ocr_text(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_ocr_text(text: str) -> list[str]:
    norm = _normalize_ocr_text(text)
    if not norm:
        return []
    return [tok for tok in norm.split(" ") if tok]


def _looks_like_code_line(raw: str) -> bool:
    return any(ch in (raw or "") for ch in "{}()[]=:.<>$")


def _is_strong_anchor_line(raw: str, norm: str, toks: list[str]) -> bool:
    if _looks_like_code_line(raw):
        return True
    if any(ch.isdigit() for ch in (raw or "")):
        return True
    if any("가" <= ch <= "힣" for ch in (raw or "")) and len(norm) >= OCR_ANCHOR_MIN_CHARS:
        return True
    if len(toks) >= max(5, OCR_ANCHOR_MIN_TOKENS + 1):
        return True
    return False


def _extract_ocr_anchor_meta(text: str) -> list[tuple[str, bool]]:
    scored: list[tuple[tuple[int, int], str, bool]] = []
    seen = set()

    for raw in (text or "").splitlines():
        norm = _normalize_ocr_text(raw)
        if not norm or norm in seen:
            continue
        toks = _tokenize_ocr_text(norm)
        if not toks:
            continue

        keep = (
            len(norm) >= OCR_ANCHOR_MIN_CHARS
            or len(toks) >= OCR_ANCHOR_MIN_TOKENS
            or (_looks_like_code_line(raw) and len(toks) >= 2)
        )
        if not keep:
            continue

        seen.add(norm)
        strong = _is_strong_anchor_line(raw, norm, toks)
        scored.append(((len(toks), len(norm)), norm, strong))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(line, strong) for _, line, strong in scored[: max(1, OCR_ANCHOR_TOP_K)]]


def _extract_ocr_anchor_lines(text: str) -> list[str]:
    return [line for line, _ in _extract_ocr_anchor_meta(text)]


def _has_exact_anchor_match(prev_text: str, next_text: str) -> bool:
    if not OCR_EXACT_ANCHOR_ENABLED:
        return False

    prev_norm = _normalize_ocr_text(prev_text)
    next_norm = _normalize_ocr_text(next_text)
    if not prev_norm or not next_norm:
        return False

    prev_anchors = _extract_ocr_anchor_meta(prev_text)
    next_anchors = _extract_ocr_anchor_meta(next_text)
    if not prev_anchors or not next_anchors:
        return False

    shorter, longer_text = (
        (prev_anchors, next_norm) if len(prev_anchors) <= len(next_anchors) else (next_anchors, prev_norm)
    )
    shared = [(anchor, strong) for anchor, strong in shorter if anchor == longer_text or anchor in longer_text]
    if not shared:
        return False
    if len(shared) >= 2:
        return True
    return bool(shared[0][1])


def _should_merge_by_ocr_text(prev_text: str, next_text: str) -> bool:
    a = _normalize_ocr_text(prev_text)
    b = _normalize_ocr_text(next_text)
    if not a or not b:
        return False

    if _has_exact_anchor_match(prev_text, next_text):
        return True

    if a == b and max(len(a), len(b)) >= OCR_MIN_TEXT_CHARS:
        return True

    ta = set(_tokenize_ocr_text(a))
    tb = set(_tokenize_ocr_text(b))
    if min(len(ta), len(tb)) < OCR_MIN_TOKENS:
        return False

    inter = len(ta & tb)
    union = len(ta | tb)
    if inter == 0 or union == 0:
        return False

    containment = inter / min(len(ta), len(tb))
    jaccard = inter / union
    len_ratio = min(len(a), len(b)) / max(len(a), len(b))

    return (
        containment >= OCR_CONTAINMENT_THR
        and jaccard >= OCR_JACCARD_THR
        and len_ratio >= OCR_LEN_RATIO_MIN
    )


def _is_low_info_ocr_text(text: str) -> bool:
    norm = _normalize_ocr_text(text)
    if not norm:
        return True
    toks = _tokenize_ocr_text(norm)
    if not toks:
        return True
    total_len = sum(len(t) for t in toks)
    if total_len <= 2:
        return True
    if len(toks) == 1 and len(toks[0]) <= 2:
        return True
    return False


class _TesseractOCREngine:
    def __init__(self, lang: str, psm: str):
        self.backend = "tesseract"
        self.device = "cpu"
        self.lang = lang
        self.psm = psm

    def read_text(self, image_path: Path) -> str:
        cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            self.lang,
            "--oem",
            "1",
            "--psm",
            self.psm,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return ""
        if proc.returncode != 0:
            return ""
        return proc.stdout or ""


def _is_duplicate_by_phash(
    h_start,
    h_end,
    ref_start,
    ref_end,
    duplicate_thr: int,
):
    start_dist = h_start - ref_start
    end_dist = h_end - ref_end
    return (start_dist <= duplicate_thr and end_dist <= duplicate_thr), start_dist, end_dist


def refine_periods_with_ocr(
    v_path: str,
    out_dir: Path,
    periods: list[tuple[int, int]],
    hashes: list,
) -> tuple[list[tuple[int, int]], dict]:
    """
    pHash로 잡은 경계 후보를 OCR로 재검증해, 애니메이션으로 인한 과분할을 병합.
    """
    stats = {
        "enabled": True,
        "backend": "tesseract",
        "device": "cpu",
        "boundaries": max(0, len(periods) - 1),
        "interior_candidates": 0,
        "interior_split_periods": 0,
        "interior_added_segments": 0,
        "interior_hash_fallback_splits": 0,
        "checked": 0,
        "merged": 0,
        "blocked_by_img_dist": 0,
        "merged_by_empty_visual": 0,
        "ocr_calls": 0,
        "elapsed_sec": 0.0,
        "error": "",
    }
    if len(periods) < 2:
        stats["enabled"] = False
        return periods, stats

    engine = _TesseractOCREngine(lang=OCR_LANG, psm=OCR_PSM)

    t0 = time.perf_counter()
    tmp_dir = out_dir / ".ocr_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) 긴 구간에서 OCR 텍스트 스냅샷 기반 분할 (start/end mismatch 완화)
    expanded_periods: list[tuple[int, int]] = []
    interior_split_boundaries: set[int] = set()
    min_gap_frames = max(1, int(round(OCR_INTERIOR_SPLIT_MIN_GAP_SEC * SAMPLE_FPS)))
    step_frames = max(1, int(round(OCR_INTERIOR_SPLIT_STEP_SEC * SAMPLE_FPS)))
    start_offset_frames = int(round(START_OFFSET * SAMPLE_FPS))
    end_offset_frames = int(round(END_OFFSET * SAMPLE_FPS))
    interior_idx = 0

    for start_idx, end_idx in periods:
        duration = (end_idx - start_idx + 1) / SAMPLE_FPS
        if not OCR_INTERIOR_SPLIT_ENABLED or duration < OCR_INTERIOR_SPLIT_MIN_SEC:
            expanded_periods.append((start_idx, end_idx))
            continue

        rep_start = min(end_idx, start_idx + start_offset_frames)
        rep_end = max(start_idx, end_idx - end_offset_frames)

        stats["interior_candidates"] += 1
        anchor_frame = rep_start
        anchor_img = tmp_dir / f"i{interior_idx:05d}_anchor.jpg"
        interior_idx += 1
        save_frame(v_path, anchor_frame / SAMPLE_FPS, str(anchor_img))
        anchor_text = engine.read_text(anchor_img)
        stats["ocr_calls"] += 1
        if not _normalize_ocr_text(anchor_text):
            expanded_periods.append((start_idx, end_idx))
            continue

        end_img = tmp_dir / f"i{interior_idx:05d}_end.jpg"
        interior_idx += 1
        save_frame(v_path, rep_end / SAMPLE_FPS, str(end_img))
        end_text = engine.read_text(end_img)
        stats["ocr_calls"] += 1
        start_end_hash_drift = hashes[rep_end] - hashes[rep_start]
        try:
            start_end_visual_drift = imagehash.phash(Image.open(anchor_img)) - imagehash.phash(Image.open(end_img))
        except Exception:
            start_end_visual_drift = start_end_hash_drift
        effective_drift = max(start_end_hash_drift, start_end_visual_drift)
        # 시작/끝 OCR 텍스트가 사실상 같다면, 내부 분할 시도를 생략한다.
        if _normalize_ocr_text(end_text) and _should_merge_by_ocr_text(anchor_text, end_text):
            expanded_periods.append((start_idx, end_idx))
            continue

        split_points: list[int] = []
        probe = min(end_idx - min_gap_frames, anchor_frame + step_frames)
        while probe <= end_idx - min_gap_frames and len(split_points) < OCR_INTERIOR_SPLIT_MAX_PER_PERIOD:
            probe_img = tmp_dir / f"i{interior_idx:05d}_probe.jpg"
            interior_idx += 1
            save_frame(v_path, probe / SAMPLE_FPS, str(probe_img))
            probe_text = engine.read_text(probe_img)
            stats["ocr_calls"] += 1

            if _normalize_ocr_text(probe_text) and not _should_merge_by_ocr_text(anchor_text, probe_text):
                # 이전 probe(같음)와 현재 probe(다름) 사이를 정밀 탐색해 실제 전환점을 찾는다
                fine_step = max(1, SAMPLE_FPS)  # 1초 간격
                prev_probe = probe - step_frames  # 마지막 "같음" 위치
                refined_split = probe
                if prev_probe >= anchor_frame:
                    fp = prev_probe + fine_step
                    while fp < probe:
                        fp_img = tmp_dir / f"i{interior_idx:05d}_fsplit.jpg"
                        interior_idx += 1
                        save_frame(v_path, fp / SAMPLE_FPS, str(fp_img))
                        fp_text = engine.read_text(fp_img)
                        stats["ocr_calls"] += 1
                        if _normalize_ocr_text(fp_text) and not _should_merge_by_ocr_text(anchor_text, fp_text):
                            refined_split = fp
                            break
                        fp += fine_step
                split_points.append(refined_split)
                anchor_frame = refined_split
                anchor_text = probe_text  # 현재 probe의 텍스트 사용
                probe = min(end_idx - min_gap_frames, anchor_frame + min_gap_frames)
                continue
            probe += step_frames

        # probing이 끝까지 못 가서 split을 놓친 경우: end 근처 역방향 탐색
        if not split_points and _normalize_ocr_text(end_text):
            # anchor_text는 마지막 probe까지 갱신된 상태 (= 시작 슬라이드 텍스트)
            # end_text와 다르다면, end에서 역방향으로 전환점을 찾는다
            if not _should_merge_by_ocr_text(anchor_text, end_text):
                reverse_probe = rep_end - step_frames
                best_split = rep_end  # 최소한 rep_end에서는 다르다
                same_boundary = None  # "같음" 경계
                while reverse_probe > anchor_frame + min_gap_frames:
                    rp_img = tmp_dir / f"i{interior_idx:05d}_rprobe.jpg"
                    interior_idx += 1
                    save_frame(v_path, reverse_probe / SAMPLE_FPS, str(rp_img))
                    rp_text = engine.read_text(rp_img)
                    stats["ocr_calls"] += 1
                    if _normalize_ocr_text(rp_text) and _should_merge_by_ocr_text(anchor_text, rp_text):
                        same_boundary = reverse_probe
                        break
                    best_split = reverse_probe
                    reverse_probe -= step_frames
                # "같음" 경계를 찾았고, best_split과의 간격이 step보다 크면 정밀 탐색
                if same_boundary is not None and best_split - same_boundary > SAMPLE_FPS:
                    fine_step = max(1, SAMPLE_FPS)  # 1초 간격
                    fp = same_boundary + fine_step
                    while fp < best_split:
                        fp_img = tmp_dir / f"i{interior_idx:05d}_fine.jpg"
                        interior_idx += 1
                        save_frame(v_path, fp / SAMPLE_FPS, str(fp_img))
                        fp_text = engine.read_text(fp_img)
                        stats["ocr_calls"] += 1
                        if _normalize_ocr_text(fp_text) and not _should_merge_by_ocr_text(anchor_text, fp_text):
                            best_split = fp
                            break
                        fp += fine_step
                # best_split이 양쪽 최소 길이를 만족하는지 확인
                # 역방향 탐색은 끝 근처 전환을 찾는 것이므로 오른쪽은 MIN_DURATION만 요구
                min_right = max(1, int(round(MIN_DURATION * SAMPLE_FPS)))
                if (best_split - start_idx) >= min_gap_frames and (end_idx - best_split) >= min_right:
                    split_points.append(best_split)

        if (
            not split_points
            and _normalize_ocr_text(end_text)
            and effective_drift >= OCR_INTERIOR_SPLIT_HASH_DRIFT
            and start_end_hash_drift >= 4
        ):
            target_drift = max(2, int(round(start_end_hash_drift * OCR_INTERIOR_SPLIT_HASH_TARGET_RATIO)))
            search_start = rep_start + min_gap_frames
            search_end = rep_end - min_gap_frames
            for cand in range(search_start, search_end + 1):
                if (hashes[cand] - hashes[rep_start]) >= target_drift:
                    split_points.append(cand)
                    stats["interior_hash_fallback_splits"] += 1
                    break

        if not split_points:
            expanded_periods.append((start_idx, end_idx))
            continue

        bounds = [start_idx] + split_points + [end_idx + 1]
        converted = []
        # 최소 세그먼트 길이: 기본은 min_gap이지만, 마지막 세그먼트는 MIN_DURATION 허용
        min_seg_default = min_gap_frames
        min_seg_tail = max(1, int(round(MIN_DURATION * SAMPLE_FPS)))
        seg_pairs = list(zip(bounds, bounds[1:]))
        for idx_seg, (a, b) in enumerate(seg_pairs):
            seg_start = a
            seg_end = b - 1
            is_last = (idx_seg == len(seg_pairs) - 1)
            min_seg = min_seg_tail if is_last else min_seg_default
            if seg_end >= seg_start and (seg_end - seg_start + 1) >= min_seg:
                converted.append((seg_start, seg_end))

        if len(converted) <= 1:
            expanded_periods.append((start_idx, end_idx))
            continue

        base_idx = len(expanded_periods)
        expanded_periods.extend(converted)
        # interior split으로 생성된 경계 인덱스를 기록 (boundary merge에서 보호)
        for ci in range(base_idx, base_idx + len(converted) - 1):
            interior_split_boundaries.add(ci)
        stats["interior_split_periods"] += 1
        stats["interior_added_segments"] += len(converted) - 1

    periods = expanded_periods
    stats["boundaries"] = max(0, len(periods) - 1)

    # 2) 경계 OCR 비교로 과분할 병합
    merged_periods: list[tuple[int, int]] = []
    i = 0
    boundary_idx = 0
    while i < len(periods):
        cur_start, cur_end = periods[i]
        while i + 1 < len(periods):
            # interior split으로 생성된 경계는 병합하지 않음
            if i in interior_split_boundaries:
                break
            nxt_start, nxt_end = periods[i + 1]
            boundary_idx += 1
            boundary_dist = hashes[nxt_start] - hashes[cur_end]
            if boundary_dist > OCR_MAX_PHASH_DIST:
                break

            prev_ts = max(cur_start / SAMPLE_FPS, cur_end / SAMPLE_FPS - 0.15)
            next_ts = min(nxt_end / SAMPLE_FPS, nxt_start / SAMPLE_FPS + 0.15)
            prev_img = tmp_dir / f"b{boundary_idx:05d}_prev.jpg"
            next_img = tmp_dir / f"b{boundary_idx:05d}_next.jpg"
            save_frame(v_path, prev_ts, str(prev_img))
            save_frame(v_path, next_ts, str(next_img))

            try:
                boundary_img_dist = imagehash.phash(Image.open(prev_img)) - imagehash.phash(Image.open(next_img))
            except Exception:
                boundary_img_dist = boundary_dist

            if boundary_img_dist > OCR_BOUNDARY_MERGE_MAX_IMG_DIST:
                stats["blocked_by_img_dist"] += 1
                break

            prev_text = engine.read_text(prev_img)
            next_text = engine.read_text(next_img)
            stats["checked"] += 1
            stats["ocr_calls"] += 2

            if _should_merge_by_ocr_text(prev_text, next_text):
                cur_end = nxt_end
                i += 1
                stats["merged"] += 1
                continue

            prev_norm = _normalize_ocr_text(prev_text)
            next_norm = _normalize_ocr_text(next_text)
            if (
                _is_low_info_ocr_text(prev_norm)
                and _is_low_info_ocr_text(next_norm)
                and boundary_img_dist <= OCR_EMPTY_VISUAL_MERGE_MAX_IMG_DIST
            ):
                cur_end = nxt_end
                i += 1
                stats["merged"] += 1
                stats["merged_by_empty_visual"] += 1
                continue
            break

        merged_periods.append((cur_start, cur_end))
        i += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return merged_periods, stats


# ── 프레임 저장 ───────────────────────────────────────────

def save_frame(v_path: str, timestamp: float, out_path: str) -> None:
    """특정 타임스탬프의 프레임을 고품질 JPEG로 저장"""
    timestamp = max(0.0, timestamp)
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", f"{timestamp:.3f}",
         "-i", v_path,
         "-frames:v", "1",
         "-q:v", "2",
         out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _image_phash_distance(path_a: Path, path_b: Path) -> int:
    try:
        return imagehash.phash(Image.open(path_a)) - imagehash.phash(Image.open(path_b))
    except Exception:
        return 0


def _align_end_frame_to_start(
    v_path: str,
    start_img: Path,
    end_img: Path,
    start_ts: float,
    end_ts: float,
) -> tuple[float, bool]:
    """
    저장된 start/end 이미지가 과도하게 다르면 end 프레임을 뒤로 당겨 같은 슬라이드로 정렬.
    반환: (aligned_end_ts, shifted)
    """
    if END_ALIGN_BACKTRACK_SEC <= 0 or END_ALIGN_STEP_SEC <= 0:
        return end_ts, False

    base_dist = _image_phash_distance(start_img, end_img)
    if base_dist <= END_ALIGN_MAX_IMG_DIST:
        return end_ts, False

    max_steps = max(1, int(round(END_ALIGN_BACKTRACK_SEC / END_ALIGN_STEP_SEC)))
    tmp_img = end_img.with_name(f"{end_img.stem}_align_tmp{end_img.suffix}")

    for i in range(1, max_steps + 1):
        cand_ts = max(start_ts, end_ts - i * END_ALIGN_STEP_SEC)
        if cand_ts >= end_ts:
            continue
        save_frame(v_path, cand_ts, str(tmp_img))
        if not tmp_img.exists():
            continue
        cand_dist = _image_phash_distance(start_img, tmp_img)
        if cand_dist <= END_ALIGN_MAX_IMG_DIST:
            shutil.move(str(tmp_img), str(end_img))
            return cand_ts, True

    if tmp_img.exists():
        tmp_img.unlink()
    return end_ts, False


# ── 메인 ─────────────────────────────────────────────────

def run(v_path: str, output_dir: str, min_duration: float = MIN_DURATION,
        transition_thr: int = TRANSITION_THR, duplicate_thr: int = DUPLICATE_THR) -> None:
    v_path = str(Path(v_path).resolve())
    v_name = Path(v_path).stem
    out_dir = Path(output_dir) / v_name

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # [1] 메타데이터
    print(f"\n[1/3] 영상 정보 분석: {v_name}")
    w, h, fps = get_video_meta(v_path)
    print(f"  해상도: {w}x{h}, FPS: {fps:.2f}")

    # [2] pHash 샘플링
    print(f"\n[2/3] {SAMPLE_FPS}fps 샘플링 + pHash 계산")
    hashes = sample_phashes(v_path, w, h)
    total_sec = len(hashes) / SAMPLE_FPS
    print(f"  총 {len(hashes)}샘플 완료 (영상 길이: {int(total_sec)//60}분 {int(total_sec)%60}초)")

    # [3] 구간 탐지 + 저장
    print(f"\n[3/3] 슬라이드 구간 탐지 (전환 기준: pHash>{transition_thr}, "
          f"중복 기준: pHash≤{duplicate_thr}, 최소 {min_duration}초)")
    periods = detect_stable_periods(hashes, transition_thr)

    # 장시간 영상에서 슬라이드 수가 비정상적으로 적게 잡히면 임계값을 자동 완화한다.
    # (예: 기본값 5/4에서 2장으로 과병합되는 케이스)
    if (
        total_sec >= RELAX_MIN_VIDEO_SEC
        and len(periods) <= RELAX_MAX_PERIODS
        and transition_thr > RELAXED_TRANSITION_THR
    ):
        relaxed_periods = detect_stable_periods(hashes, RELAXED_TRANSITION_THR)
        if len(relaxed_periods) > len(periods):
            print(
                f"  ⚙️ 자동 보정: 전환 임계값 {transition_thr}→{RELAXED_TRANSITION_THR}, "
                f"중복 임계값 {duplicate_thr}→{RELAXED_DUPLICATE_THR}"
            )
            transition_thr = RELAXED_TRANSITION_THR
            duplicate_thr = min(duplicate_thr, RELAXED_DUPLICATE_THR)
            periods = relaxed_periods

    # 과검출 보정: 애니메이션/포인터 이동으로 전환이 과다하게 잘린 경우
    minutes = max(total_sec / 60.0, 1e-6)
    short_cutoff = max(OVERSEG_SHORT_SEC, min_duration * 2.5)
    short_count = sum(1 for s, e in periods if ((e - s + 1) / SAMPLE_FPS) <= short_cutoff)
    period_density = len(periods) / minutes
    short_ratio = (short_count / len(periods)) if periods else 0.0

    if (
        len(periods) >= OVERSEG_MIN_PERIODS
        and period_density >= OVERSEG_DENSITY_PER_MIN
        and short_ratio >= OVERSEG_SHORT_RATIO
        and transition_thr < max(TIGHT_TRANSITION_CANDIDATES)
    ):
        chosen_thr = transition_thr
        tightened_periods = periods

        for cand_thr in TIGHT_TRANSITION_CANDIDATES:
            if cand_thr <= transition_thr:
                continue
            cand_periods = detect_stable_periods(hashes, cand_thr)
            if len(cand_periods) < MIN_PERIODS_AFTER_TIGHTEN:
                continue
            chosen_thr = cand_thr
            tightened_periods = cand_periods
            cand_density = len(cand_periods) / minutes
            if cand_density <= OVERSEG_DENSITY_PER_MIN:
                break

        if len(tightened_periods) < len(periods):
            print(
                "  ⚙️ 자동 보정: 애니메이션 과검출 완화 "
                f"(전환 임계값 {transition_thr}→{chosen_thr})"
            )
            transition_thr = chosen_thr
            periods = tightened_periods

    if INTERIOR_SPLIT_ENABLED:
        before = len(periods)
        periods, split_stats = refine_periods_by_interior_split(periods, hashes, transition_thr)
        if split_stats.get("split_periods", 0) > 0:
            print(
                "  ⚙️ 내부 분할 보정: "
                f"긴 구간 {split_stats['checked']}개 검사, "
                f"{split_stats['split_periods']}개 분할, "
                f"세그먼트 +{split_stats['added_segments']} "
                f"(전환 임계값 {transition_thr}→국소 {max(1, transition_thr - max(1, INTERIOR_SPLIT_DELTA_THR))})"
            )
            if len(periods) != before:
                print(f"     구간 수: {before} → {len(periods)}")

    if OCR_REFINE_ENABLED:
        before = len(periods)
        periods, ocr_stats = refine_periods_with_ocr(
            v_path=v_path,
            out_dir=out_dir,
            periods=periods,
            hashes=hashes,
        )
        if ocr_stats.get("enabled"):
            print(
                "  ⚙️ OCR 보정: "
                f"검사 경계 {ocr_stats['checked']}/{ocr_stats['boundaries']}, "
                f"병합 {ocr_stats['merged']}개, "
                f"빈텍스트 시각병합 {ocr_stats.get('merged_by_empty_visual', 0)}개, "
                f"시각 게이트 차단 {ocr_stats.get('blocked_by_img_dist', 0)}개, "
                f"{ocr_stats['elapsed_sec']:.2f}s"
            )
            if ocr_stats.get("interior_split_periods", 0) > 0:
                print(
                    "     내부 분할: "
                    f"후보 {ocr_stats['interior_candidates']}개, "
                    f"분할 {ocr_stats['interior_split_periods']}개, "
                    f"세그먼트 +{ocr_stats['interior_added_segments']}, "
                    f"hash 보조 {ocr_stats.get('interior_hash_fallback_splits', 0)}개"
                )
            if len(periods) != before:
                print(f"     구간 수: {before} → {len(periods)}")
        else:
            print(f"  ⚠️ OCR 보정 비활성화: {ocr_stats.get('error') or '조건 불충족'}")

    print(f"  탐지된 구간: {len(periods)}개\n")

    seen: dict[int, dict] = {}
    timeline = []   # 강의 흐름 순서 기록 (짧음 스킵 제외)
    log_rows = []
    slide_no = 0
    dup_ocr_engine = _TesseractOCREngine(lang=OCR_LANG, psm=OCR_PSM) if DUP_FALLBACK_OCR_ENABLED else None
    dup_ocr_cache: dict[str, str] = {}
    dup_tmp_dir = out_dir / ".dup_fallback_ocr"
    if dup_ocr_engine is not None:
        dup_tmp_dir.mkdir(parents=True, exist_ok=True)
    start_offset_frames = int(round(START_OFFSET * SAMPLE_FPS))
    end_offset_frames = int(round(END_OFFSET * SAMPLE_FPS))

    for start_idx, end_idx in periods:
        duration  = (end_idx - start_idx + 1) / SAMPLE_FPS
        start_ts  = start_idx / SAMPLE_FPS
        end_ts    = end_idx   / SAMPLE_FPS
        rep_start_idx = min(end_idx, start_idx + start_offset_frames)
        rep_end_idx = max(start_idx, end_idx - end_offset_frames)

        # ── 최소 길이 이하 스킵
        if duration <= min_duration:
            log_rows.append({
                "status":    "SKIP_SHORT",
                "start_sec": round(start_ts, 2),
                "end_sec":   round(end_ts,   2),
                "duration":  round(duration,  2),
            })
            print(f"  [{_fmt(start_ts)} ~ {_fmt(end_ts)}] ({duration:.1f}s) → 짧음, 스킵")
            continue

        # ── 중복 체크 (엄격: start+end 둘 다 / 보조: start 유사 + OCR 확인)
        h_start = hashes[rep_start_idx]
        h_end   = hashes[rep_end_idx]
        dup_no = None
        dup_method = None
        current_start_text = None
        current_start_img = None

        def _get_current_start_text() -> str:
            nonlocal current_start_text, current_start_img
            if current_start_text is not None:
                return current_start_text
            if dup_ocr_engine is None:
                current_start_text = ""
                return current_start_text
            if current_start_img is None:
                current_start_img = dup_tmp_dir / f"period_{start_idx:06d}_start.jpg"
                save_frame(v_path, rep_start_idx / SAMPLE_FPS, str(current_start_img))
            current_start_text = dup_ocr_engine.read_text(current_start_img)
            return current_start_text

        for no, info in seen.items():
            strict_dup, start_dist, end_dist = _is_duplicate_by_phash(
                h_start, h_end, info["hash_start"], info["hash_end"], duplicate_thr
            )
            if strict_dup:
                # pHash가 중복이라도, OCR로 텍스트가 실제로 같은지 확인
                # (동일 템플릿 슬라이드는 pHash가 비슷하지만 텍스트가 다를 수 있음)
                # start 이미지만 비교 (end 이미지는 다음 슬라이드 프레임일 수 있음)
                if dup_ocr_engine is not None:
                    cur_text = _get_current_start_text()
                    if _normalize_ocr_text(cur_text):
                        seen_start_img = out_dir / f"slide_{no:03d}_start.jpg"
                        key = str(seen_start_img)
                        seen_text = dup_ocr_cache.get(key)
                        if seen_text is None:
                            seen_text = dup_ocr_engine.read_text(seen_start_img) if seen_start_img.exists() else ""
                            dup_ocr_cache[key] = seen_text
                        # pHash strict duplicate이므로 OCR 임계값을 관대하게 적용
                        # (Tesseract 노이즈로 같은 슬라이드도 미세 차이 발생)
                        if _normalize_ocr_text(seen_text):
                            ta = set(_tokenize_ocr_text(cur_text))
                            tb = set(_tokenize_ocr_text(seen_text))
                            inter = len(ta & tb)
                            containment = inter / min(len(ta), len(tb)) if min(len(ta), len(tb)) > 0 else 0
                            jaccard = inter / (len(ta | tb)) if len(ta | tb) > 0 else 0
                            if containment < 0.55 or jaccard < 0.35:
                                continue  # pHash는 중복이지만 OCR이 확실히 다름 → 새 슬라이드
                dup_no = no
                dup_method = "phash_strict"
                break

            # 강사가 잠깐 넘겼다가 되돌아온 경우:
            # 시작 프레임은 동일하지만 종료 프레임이 애니메이션/포인터로 달라질 수 있다.
            if dup_ocr_engine is None:
                continue
            if start_dist > DUP_FALLBACK_MAX_START_DIST or end_dist > DUP_FALLBACK_MAX_END_DIST:
                continue
            # OCR fallback은 "잠깐 넘겼다가 돌아온" 경우만 허용한다.
            # 한참 뒤에 옛 슬라이드와 텍스트가 얼핏 비슷하게 읽히는 경우까지
            # duplicate로 흡수되면 중간 슬라이드가 통째로 사라질 수 있다.
            occs = info.get("occurrences") or []
            if occs:
                last_end = float(occs[-1].get("end_sec", 0) or 0)
                if start_ts - last_end > DUP_FALLBACK_MAX_GAP_SEC:
                    continue

            cur_text = _get_current_start_text()
            if not _normalize_ocr_text(cur_text):
                continue

            matched_by_text = False
            seen_start_img = out_dir / f"slide_{no:03d}_start.jpg"
            key = str(seen_start_img)
            seen_text = dup_ocr_cache.get(key)
            if seen_text is None:
                seen_text = dup_ocr_engine.read_text(seen_start_img) if seen_start_img.exists() else ""
                dup_ocr_cache[key] = seen_text
            if _should_merge_by_ocr_text(cur_text, seen_text):
                matched_by_text = True

            if matched_by_text:
                dup_no = no
                dup_method = "ocr_fallback"
                break

        if dup_no is not None:
            # 누적 시간 합산 + 이번 등장 기록
            seen[dup_no]["total_duration"] += duration
            seen[dup_no]["occurrences"].append({
                "start_sec": round(start_ts, 2),
                "end_sec":   round(end_ts,   2),
                "duration":  round(duration,  2),
            })

            # end 프레임 갱신: 이번 구간이 더 길면 덮어쓰기
            end_updated = False
            if duration > seen[dup_no]["best_duration"]:
                seen[dup_no]["best_duration"] = duration
                seen[dup_no]["hash_end"] = h_end
                seen[dup_no]["rep_end_idx"] = rep_end_idx
                save_end_ts = rep_end_idx / SAMPLE_FPS
                f_end = str(out_dir / f"slide_{dup_no:03d}_end.jpg")
                save_frame(v_path, save_end_ts, f_end)
                start_img_path = out_dir / f"slide_{dup_no:03d}_start.jpg"
                aligned_end_ts, shifted = _align_end_frame_to_start(
                    v_path=v_path,
                    start_img=start_img_path,
                    end_img=Path(f_end),
                    start_ts=start_ts,
                    end_ts=save_end_ts,
                )
                if shifted:
                    save_end_ts = aligned_end_ts
                    aligned_idx = max(start_idx, int(round(aligned_end_ts * SAMPLE_FPS)))
                    seen[dup_no]["rep_end_idx"] = aligned_idx
                    seen[dup_no]["hash_end"] = hashes[aligned_idx]
                end_updated = True

            # 타임라인 기록
            timeline.append({
                "slide_no":  dup_no,
                "start_sec": round(start_ts, 2),
                "end_sec":   round(end_ts,   2),
                "duration":  round(duration,  2),
                "is_dup":    True,
                "dup_method": dup_method or "unknown",
            })

            suffix = " ★end 갱신" if end_updated else ""
            print(f"  [{_fmt(start_ts)} ~ {_fmt(end_ts)}] ({duration:.1f}s) "
                  f"→ 중복 (슬라이드 {dup_no:03d}, {dup_method or 'unknown'}){suffix}")

            log_rows.append({
                "status":         "DUP",
                "slide_no":       dup_no,
                "dup_method":     dup_method or "unknown",
                "start_sec":      round(start_ts, 2),
                "end_sec":        round(end_ts,   2),
                "duration":       round(duration,  2),
                "total_duration": round(seen[dup_no]["total_duration"], 2),
                "end_updated":    end_updated,
            })
            continue

        # ── 새 슬라이드 저장
        slide_no += 1
        seen[slide_no] = {
            "hash_start":     h_start,
            "hash_end":       h_end,
            "rep_start_idx":  rep_start_idx,
            "rep_end_idx":    rep_end_idx,
            "total_duration": duration,
            "best_duration":  duration,
            "occurrences":    [{"start_sec": round(start_ts, 2),
                                "end_sec":   round(end_ts,   2),
                                "duration":  round(duration,  2)}],
        }

        save_start_ts = rep_start_idx / SAMPLE_FPS
        save_end_ts   = rep_end_idx   / SAMPLE_FPS

        f_start = str(out_dir / f"slide_{slide_no:03d}_start.jpg")
        f_end   = str(out_dir / f"slide_{slide_no:03d}_end.jpg")
        save_frame(v_path, save_start_ts, f_start)
        save_frame(v_path, save_end_ts,   f_end)
        aligned_end_ts, shifted = _align_end_frame_to_start(
            v_path=v_path,
            start_img=Path(f_start),
            end_img=Path(f_end),
            start_ts=start_ts,
            end_ts=save_end_ts,
        )
        if shifted:
            save_end_ts = aligned_end_ts
            aligned_idx = max(start_idx, int(round(aligned_end_ts * SAMPLE_FPS)))
            seen[slide_no]["rep_end_idx"] = aligned_idx
            seen[slide_no]["hash_end"] = hashes[aligned_idx]

        # 타임라인 기록
        timeline.append({
            "slide_no":  slide_no,
            "start_sec": round(start_ts, 2),
            "end_sec":   round(end_ts,   2),
            "duration":  round(duration,  2),
            "is_dup":    False,
        })

        log_rows.append({
            "status":          "SAVED",
            "slide_no":        slide_no,
            "start_sec":       round(start_ts,      2),
            "end_sec":         round(end_ts,         2),
            "duration":        round(duration,       2),
            "total_duration":  round(duration,       2),
            "saved_start_sec": round(save_start_ts, 2),
            "saved_end_sec":   round(save_end_ts,   2),
        })
        print(f"  [{_fmt(start_ts)} ~ {_fmt(end_ts)}] ({duration:.1f}s) "
              f"→ 슬라이드 {slide_no:03d} 저장")

    # ── 슬라이드별 occurrences를 log_rows(SAVED)에 병합
    for row in log_rows:
        if row["status"] == "SAVED":
            no = row["slide_no"]
            row["total_duration"] = round(seen[no]["total_duration"], 2)
            row["occurrences"]    = seen[no]["occurrences"]

    # timeline의 시간 순서를 명시적으로 남긴다.
    for idx, item in enumerate(timeline, start=1):
        item["timeline_idx"] = idx

    # ── 로그 저장
    log_path = out_dir / f"{v_name}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "slides":   log_rows,
            "timeline": timeline,
        }, f, ensure_ascii=False, indent=2)

    if dup_ocr_engine is not None:
        shutil.rmtree(dup_tmp_dir, ignore_errors=True)

    skipped_short = sum(1 for r in log_rows if r["status"] == "SKIP_SHORT")
    skipped_dup   = sum(1 for r in log_rows if r["status"] == "DUP")

    print(f"\n{'='*50}")
    print(f"  저장된 슬라이드 : {slide_no}개")
    print(f"  스킵 (짧음)     : {skipped_short}개")
    print(f"  중복 등장       : {skipped_dup}개")
    print(f"  출력 경로       : {out_dir}")
    print(f"  로그            : {log_path}")
    print(f"{'='*50}")
    print(f"\n강의 흐름 (timeline):")
    for t in timeline:
        dup_mark = " [재등장]" if t["is_dup"] else ""
        print(f"  [{_fmt(t['start_sec'])} ~ {_fmt(t['end_sec'])}] "
              f"슬라이드 {t['slide_no']:03d}{dup_mark} ({t['duration']:.0f}s)")


def _fmt(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m:02d}:{s:02d}"


# ── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="슬라이드 전환 감지 및 추출기")
    parser.add_argument("video", help="영상 파일 경로")
    parser.add_argument("-o", "--output", default="./output", help="출력 디렉토리 (기본: ./output)")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION,
                        help=f"최소 슬라이드 지속 시간 초 (기본: {MIN_DURATION})")
    parser.add_argument("--transition-thr", type=int, default=TRANSITION_THR,
                        help=f"전환 판정 pHash 해밍 거리 (기본: {TRANSITION_THR})")
    parser.add_argument("--duplicate-thr", type=int, default=DUPLICATE_THR,
                        help=f"중복 판정 pHash 해밍 거리 (기본: {DUPLICATE_THR})")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    run(args.video, args.output, args.min_duration, args.transition_thr, args.duplicate_thr)
