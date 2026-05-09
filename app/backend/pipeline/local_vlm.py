"""
Local VLM review for slide duplicate/build candidates.

The VLM stage is intentionally optional. By default it writes review results
without changing metadata, so CPU-only environments can test model quality
before enabling automatic application.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DECISIONS = {
    "same_slide_duplicate",
    "same_slide_build",
    "transition_noise",
    "different_slide",
    "uncertain",
}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def local_vlm_enabled() -> bool:
    return env_bool("GRAPHLEC_VLM_ENABLED", False)


def local_vlm_apply_enabled() -> bool:
    return env_bool("GRAPHLEC_VLM_APPLY", False)


def local_vlm_worker_count(candidate_count: int) -> int:
    if candidate_count <= 1:
        return 1
    workers = env_int("GRAPHLEC_VLM_WORKERS", 2)
    return max(1, min(workers, candidate_count))


def _image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _candidate_scene_indices(candidate: dict[str, Any]) -> list[int]:
    indices = []
    for value in candidate.get("scene_indices") or []:
        try:
            indices.append(int(value))
        except (TypeError, ValueError):
            continue
    return indices


def _limited_candidate_filenames(candidate: dict[str, Any]) -> list[str]:
    filenames = list(candidate.get("filenames") or [])
    scene_indices = list(candidate.get("scene_indices") or [])
    if len(filenames) <= 2 or len(filenames) != len(scene_indices):
        return filenames

    candidate_type = candidate.get("candidate_type")
    if candidate_type == "transition_noise":
        middle_indices = list(candidate.get("middle_scene_indices") or [])
        positions = []
        for middle in middle_indices[:1]:
            if middle in scene_indices:
                mid_pos = scene_indices.index(middle)
                positions = [
                    max(0, mid_pos - 1),
                    mid_pos,
                    min(len(scene_indices) - 1, mid_pos + 1),
                ]
                break
        if not positions:
            positions = list(range(min(3, len(filenames))))
    elif candidate_type == "same_slide_build":
        positions = [0, len(filenames) - 1]
    elif candidate_type == "same_slide_duplicate":
        if len(filenames) <= 3:
            return filenames
        positions = sorted({0, len(filenames) // 2, len(filenames) - 1})
    else:
        return filenames

    positions = sorted(dict.fromkeys(pos for pos in positions if 0 <= pos < len(filenames)))
    return [filenames[pos] for pos in positions]


def _normalize_result(candidate: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    scene_indices = _candidate_scene_indices(candidate)
    context_scene_indices = _candidate_scene_indices({"scene_indices": candidate.get("context_scene_indices")})
    middle_scene_indices = _candidate_scene_indices({"scene_indices": candidate.get("middle_scene_indices")})
    decision = str(raw.get("decision", "uncertain")).strip()
    if decision not in DECISIONS:
        decision = "uncertain"

    if decision == "same_slide_build":
        fallback_representative = scene_indices[-1] if scene_indices else None
    else:
        fallback_representative = scene_indices[0] if scene_indices else None

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    representative = raw.get("representative_scene_index", fallback_representative)
    try:
        representative = int(representative) if representative is not None else fallback_representative
    except (TypeError, ValueError):
        representative = fallback_representative
    if representative not in scene_indices:
        representative = fallback_representative

    if decision in {"same_slide_duplicate", "same_slide_build"}:
        should_merge_slide_group = True
        should_drop_scene = False
    elif decision == "transition_noise":
        should_merge_slide_group = False
        should_drop_scene = True
    else:
        should_merge_slide_group = bool(raw.get("should_merge_slide_group", False))
        should_drop_scene = bool(raw.get("should_drop_scene", False))

    return {
        "candidate_type": candidate.get("candidate_type"),
        "scene_indices": scene_indices,
        "context_scene_indices": context_scene_indices,
        "middle_scene_indices": middle_scene_indices,
        "filenames": candidate.get("filenames", []),
        "decision": decision,
        "confidence": confidence,
        "representative_scene_index": representative,
        "should_merge_slide_group": should_merge_slide_group,
        "should_drop_scene": should_drop_scene,
        "reason": str(raw.get("reason", "")).strip(),
        "raw_response": raw,
    }


def _prompt(candidate: dict[str, Any]) -> str:
    metrics = candidate.get("metrics", {})
    return (
        "You are reviewing lecture slide extraction results.\n"
        "Compare the provided images in order. Decide whether they are the same lecture slide, "
        "a build/animation step of the same slide, a transition/noisy intermediate frame, "
        "or genuinely different slides.\n\n"
        "Definitions:\n"
        "- same_slide_duplicate: same completed slide or same slide revisited; minor handwriting/toolbars may differ.\n"
        "- same_slide_build: second image preserves the first slide structure and adds/reveals content from the same slide.\n"
        "- transition_noise: image is captured during slide movement/animation and should not be used as a representative scene.\n"
        "- different_slide: images are different lecture-material slides.\n"
        "- uncertain: not enough evidence.\n\n"
        f"Candidate type: {candidate.get('candidate_type')}\n"
        f"Scene indices: {candidate.get('scene_indices')}\n"
        f"Context scene indices: {candidate.get('context_scene_indices')}\n"
        f"Middle scene indices: {candidate.get('middle_scene_indices')}\n"
        f"Filenames: {candidate.get('filenames')}\n"
        f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        "For transition_noise candidates, images are ordered as surrounding context and middle candidates; "
        "set should_drop_scene=true only when the middle image(s) are transitional/noisy captures.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "decision": "same_slide_duplicate|same_slide_build|transition_noise|different_slide|uncertain",\n'
        '  "confidence": 0.0,\n'
        '  "representative_scene_index": 0,\n'
        '  "should_merge_slide_group": false,\n'
        '  "should_drop_scene": false,\n'
        '  "reason": "short reason"\n'
        "}\n"
    )


class OllamaVLMProvider:
    def __init__(self):
        self.base_url = os.getenv("GRAPHLEC_OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
        self.model = os.getenv("GRAPHLEC_OLLAMA_MODEL", "gemma3:4b")

    def review(self, candidate: dict[str, Any], slides_dir: Path) -> dict[str, Any]:
        images = []
        for filename in _limited_candidate_filenames(candidate):
            path = slides_dir / filename
            if path.exists():
                images.append(_image_b64(path))
        if not images:
            raise FileNotFoundError(f"candidate images not found: {candidate.get('filenames')}")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _prompt(candidate),
                    "images": images,
                }
            ],
            "stream": False,
            "options": {
                "temperature": env_float("GRAPHLEC_VLM_TEMPERATURE", 0.0),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "")
        return _normalize_result(candidate, _extract_json_object(content))


def load_provider():
    provider = os.getenv("GRAPHLEC_VLM_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        raise ValueError(f"unsupported GRAPHLEC_VLM_PROVIDER={provider!r}; only 'ollama' is implemented")
    return OllamaVLMProvider()


def run_local_vlm_review(slides_dir: str | Path) -> dict[str, Any]:
    slides_dir = Path(slides_dir)
    candidates_path = slides_dir / "llm_review_candidates.json"
    results_path = slides_dir / "llm_review_results.json"

    if not candidates_path.exists():
        payload = {"status": "skipped", "reason": "candidate file missing", "results": []}
        results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = candidates_payload.get("candidates", [])
    worker_count = local_vlm_worker_count(len(candidates))

    results = []
    errors = []
    started = time.time()

    def review_one(idx: int, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        item_started = time.time()
        try:
            provider = load_provider()
            result = provider.review(candidate, slides_dir)
            result["candidate_index"] = idx
            result["elapsed_sec"] = round(time.time() - item_started, 3)
            return "result", result
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            return "error", {
                "candidate_index": idx,
                "scene_indices": candidate.get("scene_indices"),
                "filenames": candidate.get("filenames"),
                "elapsed_sec": round(time.time() - item_started, 3),
                "error": str(exc),
            }

    if worker_count <= 1:
        iterator = (
            review_one(idx, candidate)
            for idx, candidate in enumerate(candidates, start=1)
        )
        for kind, payload_item in iterator:
            if kind == "result":
                results.append(payload_item)
            else:
                errors.append(payload_item)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(review_one, idx, candidate): idx
                for idx, candidate in enumerate(candidates, start=1)
            }
            for future in as_completed(futures):
                kind, payload_item = future.result()
                if kind == "result":
                    results.append(payload_item)
                else:
                    errors.append(payload_item)

    results.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))
    errors.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))

    payload = {
        "status": "ok" if not errors else "partial",
        "provider": os.getenv("GRAPHLEC_VLM_PROVIDER", "ollama"),
        "model": os.getenv("GRAPHLEC_OLLAMA_MODEL", "gemma3:4b"),
        "candidate_count": len(candidates),
        "worker_count": worker_count,
        "processed_count": len(results),
        "error_count": len(errors),
        "elapsed_sec": round(time.time() - started, 3),
        "apply_enabled": local_vlm_apply_enabled(),
        "results": results,
        "errors": errors,
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def apply_vlm_slide_decisions(metadata: list[dict[str, Any]], review_payload: dict[str, Any]) -> list[dict[str, Any]]:
    min_confidence = env_float("GRAPHLEC_VLM_APPLY_MIN_CONFIDENCE", 0.65)
    results = review_payload.get("results", [])

    scenes = sorted({int(item.get("scene_index")) for item in metadata if item.get("scene_index") is not None})
    parent = {scene: scene for scene in scenes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def known_scene_indices(values: Any) -> list[int]:
        indices = []
        for value in values or []:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx in parent:
                indices.append(idx)
        return indices

    for item in metadata:
        idx = int(item.get("scene_index", 0) or 0)
        for other in item.get("same_slide_group") or []:
            other_idx = known_scene_indices([other])
            if idx in parent and other_idx:
                union(idx, other_idx[0])

    result_by_scene: dict[int, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    dropped_scenes: set[int] = set()
    build_representatives: set[int] = set()
    for result in results:
        scene_indices = known_scene_indices(result.get("scene_indices", []))
        decision = result.get("decision")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        for scene in scene_indices:
            result_by_scene.setdefault(scene, []).append(result)

        if confidence < min_confidence:
            continue
        if decision in {"same_slide_duplicate", "same_slide_build"}:
            if len(scene_indices) >= 2:
                first = scene_indices[0]
                for other in scene_indices[1:]:
                    union(first, other)
            if decision == "same_slide_build":
                representative = known_scene_indices([result.get("representative_scene_index")])
                if not representative and scene_indices:
                    representative = [scene_indices[-1]]
                if representative:
                    build_representatives.add(representative[0])
        elif decision == "transition_noise" and result.get("should_drop_scene", True):
            drop_indices = known_scene_indices(result.get("middle_scene_indices") or scene_indices)
            dropped_scenes.update(drop_indices)

    groups: dict[int, set[int]] = {}
    for scene in scenes:
        groups.setdefault(find(scene), set()).add(scene)
    group_of = {scene: sorted(members) for members in groups.values() for scene in members}
    canonical_by_root: dict[int, int] = {}
    for root, members in groups.items():
        preferred = sorted(scene for scene in members if scene in build_representatives)
        canonical_by_root[root] = preferred[0] if preferred else min(members)

    for item in metadata:
        idx = int(item.get("scene_index", 0) or 0)
        members = group_of.get(idx, [idx])
        canonical = canonical_by_root.get(find(idx), members[0]) if idx in parent else members[0]
        item["duplicate_of"] = [x for x in members if x != idx]
        item["same_slide_group"] = members
        item["same_slide_canonical"] = canonical
        item["same_slide_group_size"] = len(members)
        item["slide_group"] = members
        item["slide_canonical_index"] = canonical
        item["slide_group_size"] = len(members)
        scene_results = result_by_scene.get(idx, [])
        if scene_results:
            item["vlm_review_decisions"] = [
                {
                    "decision": r.get("decision"),
                    "confidence": r.get("confidence"),
                    "scene_indices": r.get("scene_indices"),
                    "middle_scene_indices": r.get("middle_scene_indices"),
                    "representative_scene_index": r.get("representative_scene_index"),
                    "reason": r.get("reason"),
                }
                for r in scene_results
            ]
        if idx == canonical:
            item["vlm_preferred_representative"] = idx in build_representatives
        if idx in dropped_scenes:
            item["is_transition_noise"] = True
            item["manual_review"] = False
        elif any(r.get("decision") == "uncertain" for r in scene_results):
            item["manual_review"] = True

    if dropped_scenes:
        return [
            item
            for item in metadata
            if int(item.get("scene_index", 0) or 0) not in dropped_scenes
        ]

    return metadata
