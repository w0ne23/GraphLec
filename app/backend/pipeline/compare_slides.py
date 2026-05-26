"""
compare_slides.py
=================
이미 추출된 슬라이드 이미지를 읽어서 phash 전체 쌍 비교를 실행한다.
slide_extractor.py를 재실행하지 않고 DUPLICATE_HASH_THRESHOLD 튜닝에 사용.

Usage:
    python compare_slides.py --slides output_slides/
    python compare_slides.py --slides output_slides/ --threshold 50
    python compare_slides.py --slides output_slides/ --threshold 50 --update-metadata
"""

import cv2
import json
import argparse
import logging
import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict

try:
    from .slide_extractor import (
        Config,
        duplicate_frame_features,
        duplicate_pair_decision,
    )
except ImportError:
    from slide_extractor import (
        Config,
        duplicate_frame_features,
        duplicate_pair_decision,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

DEFAULT_THRESHOLD = 30
RESIZE_WIDTH      = 960
HASH_SIZE         = 16   # 256비트


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def resize_frame(frame, width: int):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def compute_phash_hires(frame) -> imagehash.ImageHash:
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img, hash_size=HASH_SIZE)


# ──────────────────────────────────────────────
# 메인 비교 로직
# ──────────────────────────────────────────────
def compare_slides(slides_dir: str, threshold: int, update_metadata: bool):
    out_path  = Path(slides_dir)
    meta_path = out_path / "metadata.json"

    if not meta_path.exists():
        log.error(f"metadata.json 없음: {meta_path}")
        return

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    cfg = Config()
    cfg.DUPLICATE_HASH_THRESHOLD = int(threshold)

    # scene_index별 그룹화
    groups: dict[int, list] = defaultdict(list)
    for m in metadata:
        groups[m["scene_index"]].append(m)

    # scene별 clean representative 구성
    representatives: dict[int, dict] = {}
    for idx in sorted(groups.keys()):
        frames     = groups[idx]
        base_list  = [f for f in frames if f["capture_type"] == "base"]
        clean_list = [f for f in frames if f.get("is_clean_final")]

        rep = clean_list[-1] if clean_list else base_list[0] if base_list else None
        if not rep:
            log.warning(f"  representative 없음: scene {idx}")
            continue

        fname = rep["filename"]
        img = cv2.imread(str(out_path / fname))
        if img is None:
            log.warning(f"  이미지 로드 실패: {fname}")
            continue
        representatives[idx] = duplicate_frame_features(img, cfg) | {"filename": fname}
        log.info(f"  representative 계산: scene {idx} ({fname})")

    scene_indices = sorted(representatives)
    duplicate_map: dict[int, set[int]] = defaultdict(set)
    duplicate_edges: set[frozenset[int]] = set()

    print("\n" + "═" * 70)
    print(f"  슬라이드 clean representative 중복 비교  |  phash threshold={threshold}")
    print("═" * 70)
    print(
        f"  {'scene pair':<17} {'p':>3} {'d':>3} {'cp':>3} "
        f"{'cchg':>6} {'cedge':>6} {'hist':>6}  {'판정'}"
    )
    print(f"  {'-'*17} {'-'*3} {'-'*3} {'-'*3} {'-'*6} {'-'*6} {'-'*6}  {'-'*12}")

    results = []
    for i in range(len(scene_indices)):
        for j in range(i + 1, len(scene_indices)):
            idx_a = scene_indices[i]
            idx_b = scene_indices[j]
            is_dup, metrics = duplicate_pair_decision(representatives[idx_a], representatives[idx_b], cfg)
            if (
                is_dup
                or metrics["phash"] <= cfg.DUPLICATE_HASH_THRESHOLD
                or metrics["dhash"] <= cfg.DUPLICATE_DHASH_THRESHOLD
                or metrics["content_phash"] <= cfg.DUPLICATE_CONTENT_HASH_THRESHOLD
            ):
                flag = f"★ 중복 후보/{metrics['reason']}" if is_dup else ""
                print(
                    f"  {idx_a:03d} ↔ {idx_b:03d}       "
                    f"{metrics['phash']:>3} {metrics['dhash']:>3} {metrics['content_phash']:>3} "
                    f"{metrics['content_changed']:>6.4f} {metrics['content_edge']:>6.4f} "
                    f"{metrics['hist']:>6.4f}  {flag}"
                )
            results.append({"scene_a": idx_a, "scene_b": idx_b, **metrics, "duplicate": is_dup})

            if is_dup:
                duplicate_map[idx_a].add(idx_b)
                duplicate_map[idx_b].add(idx_a)
                duplicate_edges.add(frozenset((idx_a, idx_b)))

    print("═" * 70)

    # 중복 관계 요약
    if duplicate_map:
        print(f"\n  중복 관계 요약:")
        for idx in sorted(duplicate_map.keys()):
            print(f"    scene_{idx:03d}  →  duplicate_of {sorted(duplicate_map[idx])}")
    else:
        print(f"\n  threshold={threshold} 기준 중복 후보 없음")

    # 거리 분포 요약
    dists = [r["phash"] for r in results]
    if dists:
        dup_dists    = [r["phash"] for r in results if r["duplicate"]]
        nondup_dists = [r["phash"] for r in results if not r["duplicate"]]
        print(f"\n  [거리 분포]")
        print(f"    전체   min={min(dists)}  max={max(dists)}  avg={sum(dists)/len(dists):.1f}")
        if dup_dists:
            print(f"    중복   min={min(dup_dists)}  max={max(dup_dists)}")
        if nondup_dists:
            print(f"    비중복 min={min(nondup_dists)}  max={max(nondup_dists)}")
        print(f"\n  → 적정 threshold 범위: {max(dup_dists) if dup_dists else '?'} < threshold < {min(nondup_dists) if nondup_dists else '?'}")

    # metadata.json 업데이트 (--update-metadata 옵션)
    if update_metadata:
        grouped: list[set[int]] = []
        for idx in sorted(groups):
            for group in grouped:
                if all(frozenset((idx, member)) in duplicate_edges for member in group):
                    group.add(idx)
                    break
            else:
                grouped.append({idx})
        group_of = {idx: group for group in grouped for idx in group}
        for m in metadata:
            idx = m["scene_index"]
            members = sorted(group_of.get(idx, {idx}))
            m["duplicate_of"] = [other for other in members if other != idx]
            m["same_slide_group"] = members
            m["same_slide_canonical"] = members[0]
            m["same_slide_group_size"] = len(members)
            m["slide_group"] = members
            m["slide_canonical_index"] = members[0]
            m["slide_group_size"] = len(members)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        log.info(f"\n  metadata.json 업데이트 완료: {meta_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="슬라이드 중복 phash 비교 (튜닝용)")
    parser.add_argument("--slides",    "-s", default="output_slides/", help="슬라이드 출력 디렉토리")
    parser.add_argument("--threshold", "-t", type=int, default=DEFAULT_THRESHOLD, help="중복 판정 임계값 (기본 30)")
    parser.add_argument("--update-metadata", action="store_true", help="비교 결과로 metadata.json의 is_visual_duplicate 업데이트")
    args = parser.parse_args()

    compare_slides(args.slides, args.threshold, args.update_metadata)
