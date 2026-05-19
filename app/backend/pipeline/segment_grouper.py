"""
전사 세그먼트를 맥락/문장·문단 단위로 묶기

1차: 시간/문장부호 기반으로 "호흡 단위" 그룹 생성
2차: LLM을 이용해 claim 검증에 필요한 설명 단위로 병합/분리

- 쉼/침묵: 인접 세그먼트 간 간격이 짧으면 같은 호흡으로 묶음
- 문장 끝: 마침표·물음표·느낌표 등에서 문장 경계로 분리
- 문단: 긴 침묵에서 문단 경계로 분리
- 최대 길이: 한 묶음이 너무 길어지지 않도록 상한
- 의미 기반 병합: LLM이 "같은 주제/맥락의 설명"이라고 판단하면 인접 그룹 병합
"""

import json
import os
import re

from google.genai import types

from .config import (
    GEMINI_GENERATIVE_MODEL,
    gemini_client_2,
    get_anthropic_client,
    resolve_anthropic_model,
)
from .utils import api_call_with_retry


# 문장 종결로 볼 문장부호 (한국어·영어·공백 제거 후 끝)
SENTENCE_END_PATTERN = re.compile(r".*[.!?\u3002\uFF01\uFF1F]\s*$")

# 1차 그룹 상한: 이 시간(초)을 넘으면 새 그룹 시작
MAX_GROUP_DURATION = 30.0
# 최종 context 상한: LLM이 과하게 병합해도 이 한도를 넘기지 않는다.
MAX_CONTEXT_DURATION = 90.0
MAX_CONTEXT_SEGMENTS = 10
# 이 간격(초) 이상이면 새 그룹 (문단/호흡 끊김)
PAUSE_PARAGRAPH = 1.2
# 이 간격(초) 이상이면 문장 경계 후보 (같은 문장이 아닐 가능성)
PAUSE_SENTENCE = 0.6
# 문장 끝 부호가 있으면 이 간격 이하여도 새 문장으로 분리 가능
PAUSE_AFTER_SENTENCE_END = 0.4


def _context_grouper_model() -> str:
    return (
        os.getenv("SEGMENT_GROUPER_MODEL", "").strip()
        or os.getenv("CONTEXT_GROUPER_MODEL", "").strip()
        or os.getenv("VERIFIER_CONTEXT_GROUPER_MODEL", "").strip()
        or "claude-haiku-4.5"
    )


def _is_anthropic_model(model: str) -> bool:
    spec = str(model or "").strip().lower()
    return (
        spec.startswith("claude")
        or spec in {"haiku-4.5", "haiku4.5", "claude-haiku-4.5", "claude-haiku4.5", "claude-haiku-4-5"}
        or "haiku" in spec
        or "sonnet" in spec
        or "opus" in spec
    )


def _call_context_grouper_llm(prompt: str, *, max_tokens: int = 2048) -> str:
    model = _context_grouper_model()
    if _is_anthropic_model(model):
        client = get_anthropic_client()
        if client is None:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않아 segment grouper에서 Claude를 사용할 수 없습니다.")
        resolved_model = resolve_anthropic_model(model)

        def call_api():
            return client.messages.create(
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            )

        response = api_call_with_retry(call_api)
        return "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", "") == "text"
        ).strip()

    def call_api():
        return gemini_client_2.models.generate_content(
            model=model or GEMINI_GENERATIVE_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
            ),
        )

    response = api_call_with_retry(call_api)
    return (response.text or "").strip()


def _build_group_list_for_prompt(groups: list[dict], max_chars: int = 220) -> str:
    """LLM 프롬프트용 그룹 요약 문자열 생성"""
    lines = []
    for i, g in enumerate(groups):
        text = (g.get("text") or "").replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars].rstrip() + "..."
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def decide_semantic_merges(groups: list[dict]) -> set[int]:
    """
    인접 그룹이 같은 주제/맥락인지 LLM으로 판단하여
    병합할 인덱스(i, i+1 병합)를 반환.
    """
    if len(groups) <= 1:
        return set()

    group_list = _build_group_list_for_prompt(groups)

    prompt = f"""당신은 대학 강의 전사를 claim 검증용 context로 나누는 도우미입니다.
아래는 시간과 문장부호 기준으로 1차로 나눈 '맥락 그룹' 목록입니다.
각 그룹은 동일한 강의의 연속된 구간입니다.

목표: 인접한 두 그룹이 하나의 claim 또는 그 claim을 바로 보충하는 설명이면 하나로 다시 합칩니다.
여기서 context는 "넓은 강의 주제"가 아니라, downstream claim extraction/verifier가 함께 봐야 하는 최소 설명 단위입니다.

특히, 다음과 같은 경우는 합쳐야 합니다.
- 한 문장이 시간/문장부호 때문에 잘린 경우
- 같은 claim의 주어, 조건, 이유, 예시가 바로 이어져 단독으로 떼면 의미가 불완전한 경우
- 앞 그룹의 지시어/생략된 주어/대상을 다음 그룹이 직접 완성하는 경우
- 앞 그룹의 도입 질문/목표를 다음 그룹이 같은 질문의 범위를 좁혀 다시 세우는 경우
- 흔한 오해를 소개한 뒤 바로 정정하고, 실제 정의/설명 예시를 보기 직전까지 준비 설명이 이어지는 경우

다음과 같은 경우는 합치지 마세요.
- 같은 넓은 주제 안이라도 설명 역할이 바뀌는 경우
  예: 주제 도입/질문 제시 → 답변 본문, 흔한 오해 소개 → 정정, 본문 설명 → 학습 안내/전환
- 서로 다른 정의, 서로 다른 예시, 서로 다른 claim을 순서대로 나열하는 경우
- 앞 설명이 끝나고 새 설명 단계나 학습 안내로 넘어가는 경우
- 문제 풀이에서 완전히 다른 문제로 넘어가는 경우
- 병합 결과가 대략 10개 발화 또는 90초를 넘는 경우. 단, 한 문장이 실제로 이어진 경우만 예외입니다.

단, 여러 정의/설명 예시를 보여준 뒤 강의자가 그 예시들을 종합해 자신의 최종 정의나 판단 기준을 바로 제시하는 경우는
같은 검증 context로 유지하세요. 이 경우 학생이 비교해서 이해해야 하는 단위가 하나입니다.

아래 형식의 목록을 보고, i번째 그룹과 i+1번째 그룹을 합칠지 여부만 판단하세요.

그룹 목록:
{group_list}

출력은 다음 JSON 형식으로만 작성하세요:

```json
{{ "merge_after": [0, 1, 5] }}
```

여기서 merge_after의 각 숫자 i는 "그룹 i와 그룹 i+1을 병합하라"는 의미입니다.
설명은 쓰지 말고 JSON만 출력하세요.
"""

    try:
        text = _call_context_grouper_llm(prompt, max_tokens=2048)

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        merge_after = data.get("merge_after", [])
        return {int(i) for i in merge_after if 0 <= int(i) < len(groups) - 1}
    except Exception:
        # 실패 시 의미 기반 병합 없이 진행
        return set()


def decide_context_breaks(groups: list[dict]) -> set[int]:
    """
    같은 scene 안의 연속 segment들을 의미/맥락 단위 context로 나누기 위해
    context가 끝나는 지점(i, i+1 사이 break)을 LLM으로 판단한다.

    기본 전제는 "인접 segment는 이어지는 강의 흐름"이며, LLM은 끊을 지점만 고른다.
    """
    if len(groups) <= 1:
        return set()

    group_list = _build_group_list_for_prompt(groups)

    prompt = f"""당신은 대학 강의 전사를 claim 검증용 context로 나누는 도우미입니다.
아래는 같은 scene/slide 안에서 시간 순서대로 이어지는 발화 segment 목록입니다.

목표: segment들을 downstream claim extraction/verifier가 함께 봐야 하는 최소 설명 단위로 나누세요.
context는 "같은 넓은 주제"가 아니라 "하나의 claim 또는 바로 붙은 보충 설명" 단위입니다.

중요 원칙:
- 기본적으로 인접 segment는 이어지는 강의 흐름이지만, 넓은 주제가 같다는 이유만으로 모두 붙이지 마세요.
- scene 밖으로 넘어가는 병합은 이미 금지되어 있으므로, 아래 목록 내부에서만 판단하세요.
- 새 개념, 새 소주제, 새 설명 단계, 새 정의 예시, 학습 안내/전환으로 넘어가는 지점에서 끊으세요.
- 다음과 같은 설명 역할 전환은 같은 큰 주제 안이어도 끊으세요:
  주제 도입/질문 제시 → 답변 본문, 흔한 오해 소개 → 정정/정의, 본문 설명 → 학습 안내/전환.
- 도입 질문/목표를 세운 뒤, 바로 다음 발화가 같은 질문을 더 전공적/기술적 관점으로 좁혀 다시 세우면
  그 재정의 문장까지 도입 context에 포함하고, 그 다음 실제 답변/오해/정의 본문이 시작되는 지점에서 끊으세요.
- 흔한 오해를 소개한 뒤 바로 정정하고 실제 정의/설명 예시를 보기 전까지는 하나의 준비 context로 유지하세요.
- 실제 정의/설명 예시가 문장 형태로 나열되기 시작하는 지점에서는 새 context를 시작하세요.
- 여러 정의/설명 예시를 먼저 제시하고 곧바로 강의자의 최종 정의나 판단 기준을 제시하는 경우는
  비교 검증에 필요한 하나의 context로 유지하세요.
- 최종 정의나 판단 기준이 끝난 뒤, 학습 안내나 다음 설명 예고로 넘어가면 끊으세요.
- 수업 운영 멘트, 감사 인사, 녹음/마이크 안내, 쉬는 시간 안내, 다음 장/다음 주제로 넘어간다는 전환 멘트는
  본 설명 context와 섞지 말고 별도 context가 되도록 앞뒤를 끊으세요.
- 가능하면 한 context는 1~10개 발화, 90초 이내가 되게 하세요. 단, 실제 한 문장이 이어진 경우는 예외입니다.

끊지 마세요:
- 같은 개념을 계속 설명하는 경우
- 앞 segment의 주어/조건/이유/예시가 바로 이어져 단독으로 떼면 claim이 불완전한 경우
- "그러니까", "즉", "예를 들어", "이러한", "그런데", "그리고"처럼 이어지는 설명인 경우
- 여러 정의/설명 예시와 강의자가 고른 최종 정의가 비교 블록 안에서 바로 이어지는 경우

segment 목록:
{group_list}

출력은 다음 JSON 형식으로만 작성하세요:

```json
{{ "break_after": [3, 8, 14] }}
```

break_after의 각 숫자 i는 "segment i까지 현재 context로 묶고, segment i+1부터 새 context를 시작하라"는 의미입니다.
끊을 지점이 없으면 빈 배열을 반환하세요. 설명은 쓰지 말고 JSON만 출력하세요.
"""

    try:
        text = _call_context_grouper_llm(prompt, max_tokens=2048)

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        break_after = data.get("break_after", [])
        return {int(i) for i in break_after if 0 <= int(i) < len(groups) - 1}
    except Exception:
        # 실패 시 scene 내부 발화 흐름을 최대한 보존한다.
        return set()


def _split_oversized_groups(
    groups: list[dict],
    segments: list[dict],
    *,
    max_context_segments: int = MAX_CONTEXT_SEGMENTS,
    max_context_duration: float = MAX_CONTEXT_DURATION,
) -> list[dict]:
    """LLM 과병합 방지용 안전장치. 원본 segment 순서를 보존해 큰 context만 나눈다."""
    if not groups:
        return []

    split_groups: list[dict] = []
    for group in groups:
        indices = [
            int(idx)
            for idx in (group.get("segment_indices") or [])
            if isinstance(idx, int) and 0 <= int(idx) < len(segments)
        ]
        if not indices:
            split_groups.append(group)
            continue

        duration = float(group.get("end", 0.0) or 0.0) - float(group.get("start", 0.0) or 0.0)
        if len(indices) <= max_context_segments and duration <= max_context_duration:
            split_groups.append(group)
            continue

        chunk: list[int] = []
        chunk_start = None
        for idx in indices:
            seg = segments[idx]
            seg_start = float(seg.get("start", 0.0) or 0.0)
            seg_end = float(seg.get("end", seg_start) or seg_start)
            if chunk:
                next_count = len(chunk) + 1
                next_duration = seg_end - float(chunk_start if chunk_start is not None else seg_start)
                if next_count > max_context_segments or next_duration > max_context_duration:
                    text = " ".join(str(segments[i].get("text", "") or "").strip() for i in chunk).strip()
                    split_groups.append({
                        **{k: v for k, v in group.items() if k not in {"start", "end", "text", "segment_indices"}},
                        "start": float(segments[chunk[0]].get("start", 0.0) or 0.0),
                        "end": float(segments[chunk[-1]].get("end", segments[chunk[-1]].get("start", 0.0)) or 0.0),
                        "text": text,
                        "segment_indices": chunk.copy(),
                    })
                    chunk = []
                    chunk_start = None
            if not chunk:
                chunk_start = seg_start
            chunk.append(idx)

        if chunk:
            text = " ".join(str(segments[i].get("text", "") or "").strip() for i in chunk).strip()
            split_groups.append({
                **{k: v for k, v in group.items() if k not in {"start", "end", "text", "segment_indices"}},
                "start": float(segments[chunk[0]].get("start", 0.0) or 0.0),
                "end": float(segments[chunk[-1]].get("end", segments[chunk[-1]].get("start", 0.0)) or 0.0),
                "text": text,
                "segment_indices": chunk.copy(),
            })

    return split_groups


def decide_merge_pair(current_text: str, next_text: str, max_chars: int = 400) -> bool:
    """
    두 인접 구간(current_text, next_text)이 같은 맥락/설명의 연속인지 LLM으로 판단.
    - True: 같은 맥락으로 보고 병합
    - False: 다른 맥락으로 보고 분리
    """
    cur = (current_text or "").replace("\n", " ").strip()
    nxt = (next_text or "").replace("\n", " ").strip()
    if not cur or not nxt:
        return False
    if len(cur) > max_chars:
        cur = cur[:max_chars].rstrip() + "..."
    if len(nxt) > max_chars:
        nxt = nxt[:max_chars].rstrip() + "..."

    prompt = f"""당신은 대학 강의 전사를 분석하는 도우미입니다.

아래에는 같은 슬라이드 안에서 연속된 두 구간의 전사 텍스트가 주어집니다.

[이전 구간] (이미 하나의 맥락으로 묶여 있는 설명 전체)
---
{cur}
---

[다음 구간] (바로 이어지는 설명)
---
{nxt}
---

판단 기준:

1. 다음과 같은 경우라면 "같은 맥락의 설명"으로 보고 합쳐야 합니다.
   - 이전 구간에서 소개한 개념/정의/기능에 대한 설명이 다음 구간에서 계속 이어지는 경우
   - 다음 구간이 이전 구간의 내용을 다시 말하거나 정리하는 경우
     (예: "다시 말해서", "다시 말하면", "정리하면", "즉", "한마디로" 등으로 시작)
   - 다음 구간이 이전 구간에서 소개한 개념의 예시/응용/비유를 설명하는 경우
     (예: "예를 들어", "예를 들면", "하나의 예로" 등)
   - 이전 구간이 "~에 대해 알아보겠습니다", "~의 정의는" 처럼 앞으로 설명할 것을 예고하고,
     다음 구간이 그 실제 설명인 경우

2. 다음과 같은 경우라면 "다른 맥락"으로 보고 합치지 말아야 합니다.
   - 이전 구간과 완전히 다른 개념/챕터/슬라이드의 내용을 시작하는 경우
   - 강의의 주요 주제와 관계없는 짧은 잡담, 수업 운영 멘트, 안내만으로 이루어진 경우
   - 이전 구간의 설명이 사실상 끝났고, 새로운 주제를 소개하는 경우
     (예: "다음으로", "이제 다른 주제로", "이제 ~를 보겠습니다" 등)

출력 형식:
- 두 구간이 같은 맥락의 하나의 설명으로 이어져 있다고 판단되면
  MERGE
  라고만 출력하세요.
- 서로 다른 맥락이라고 판단되면
  SPLIT
  라고만 출력하세요.

추가 설명이나 다른 문장은 쓰지 말고, "MERGE" 또는 "SPLIT" 중 하나만 출력하세요."""

    try:
        text = _call_context_grouper_llm(prompt, max_tokens=256).strip().upper()
        if "MERGE" in text and "SPLIT" not in text:
            return True
        if "SPLIT" in text and "MERGE" not in text:
            return False
        # 애매하면 보수적으로 분리
        return False
    except Exception:
        # 실패 시 보수적으로 분리
        return False


def group_segments_by_context(
    segments: list[dict],
    *,
    max_group_duration: float = MAX_GROUP_DURATION,
    pause_paragraph: float = PAUSE_PARAGRAPH,
    pause_sentence: float = PAUSE_SENTENCE,
    pause_after_sentence_end: float = PAUSE_AFTER_SENTENCE_END,
    use_semantic_merge: bool = True,
) -> list[dict]:
    """
    인접 세그먼트를 맥락/문장·문단 단위로 묶는다.

    각 세그먼트는 { 'start', 'end', 'text' } 형태.
    반환되는 각 그룹은:
      - start, end: 구간 시간
      - text: 묶인 텍스트 (공백으로 연결)
      - segment_indices: 원본 segments 인덱스 리스트
    """
    if not segments:
        return []

    # 1차: 시간/문장부호 기반 그룹화
    groups: list[dict] = []
    current_indices = [0]
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    current_texts = [segments[0]["text"].strip()]

    for i in range(1, len(segments)):
        seg = segments[i]
        prev_end = segments[i - 1]["end"]
        gap = seg["start"] - prev_end
        prev_text = (segments[i - 1]["text"] or "").strip()
        prev_ends_sentence = bool(SENTENCE_END_PATTERN.match(prev_text))

        # 문단 수준 침묵 -> 무조건 새 그룹
        if gap >= pause_paragraph:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 그룹 길이 상한 초과 -> 새 그룹
        if (seg["end"] - current_start) > max_group_duration:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 문장 끝 + (짧은 침묵이거나 다음으로 이어짐) -> 문장 경계로 새 그룹
        if prev_ends_sentence and gap >= pause_after_sentence_end:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 문장 수준 침묵 (문장 끝 부호 없음) -> 새 그룹
        if gap >= pause_sentence:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 같은 호흡/문맥으로 묶음 (1차 기준)
        current_indices.append(i)
        current_end = seg["end"]
        current_texts.append(seg["text"].strip())

    if current_indices:
        text = " ".join(t for t in current_texts if t)
        groups.append({
            "start": current_start,
            "end": current_end,
            "text": text,
            "segment_indices": current_indices.copy(),
        })

    # 2차: 의미/주제 기반 병합 (LLM)
    if not use_semantic_merge or len(groups) <= 1:
        return groups

    merge_after = decide_semantic_merges(groups)
    if not merge_after:
        return groups

    merged: list[dict] = []
    current = {
        "start": groups[0]["start"],
        "end": groups[0]["end"],
        "text": groups[0]["text"],
        "segment_indices": list(groups[0]["segment_indices"]),
    }

    for i in range(len(groups) - 1):
        g_next = groups[i + 1]

        if i in merge_after:
            # 의미적으로 같은 맥락 -> 병합
            current["end"] = g_next["end"]
            current["text"] = f"{current['text']} {g_next['text']}".strip()
            current["segment_indices"].extend(g_next["segment_indices"])
        else:
            merged.append(current)
            current = {
                "start": g_next["start"],
                "end": g_next["end"],
                "text": g_next["text"],
                "segment_indices": list(g_next["segment_indices"]),
            }

    if current:
        merged.append(current)

    return _split_oversized_groups(merged, segments)


def expand_group_annotations_to_segments(
    annotated_groups: list[dict],
    segments: list[dict],
    groups: list[dict],
) -> list[dict]:
    """
    그룹 단위 강조 결과를 원본 세그먼트 단위로 펼친다.

    - annotated_groups: 그룹별 강조 정보가 붙은 리스트 (각 항목에 'start'로 그룹 식별)
    - segments: 원본 전사 세그먼트 리스트
    - groups: group_segments_by_context() 반환값 (각 그룹에 segment_indices 있음)

    반환: segments와 같은 길이·순서의 리스트. 각 세그먼트에 해당 그룹의
    audio_emphasis, 점수 산출용 필드 등이 복사됨.
    """
    group_by_start = {g["start"]: g for g in groups}
    annotated_by_start = {s["start"]: s for s in annotated_groups}

    # segment index -> annotated group (that contains this segment)
    index_to_annotated = {}
    for g in groups:
        start = g["start"]
        ann = annotated_by_start.get(start)
        if ann is None:
            continue
        for idx in g["segment_indices"]:
            index_to_annotated[idx] = ann

    result = []
    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        ann = index_to_annotated.get(i)
        if ann:
            for key in (
                "emphasis_methods",
                "emphasis_reasons",
                "detection_count",
                "confidence",
                "emphasis_signals",
                "emphasis_keywords",
                "emphasis_keywords_by_method",
                "emphasis_detail",
                "audio_emphasis",
            ):
                if key in ann:
                    seg_copy[key] = ann[key]
        else:
            seg_copy["detection_count"] = 0
        result.append(seg_copy)

    return result


# ---------------------------------------------------------------------------
# scene 기반 그룹화 (metadata.json 사용)
# ---------------------------------------------------------------------------

def load_slide_ranges(metadata_path: str, duration_sec: float) -> list[dict]:
    """
    metadata.json에서 scene occurrence별 시간 구간 계산.
    같은 scene_index의 base(annot_index=0)가 한 scene 시작.
    반환: [ {"scene_index": 1, "slide_number": 1, "start_sec": 0.07, "end_sec": 46.73}, ... ]
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    bases = [x for x in items if x.get("annot_index") == 0 or x.get("capture_type") == "base"]
    def _scene_index(item: dict):
        return item.get("scene_index")

    bases = sorted(bases, key=lambda x: (_scene_index(x), x["timestamp_sec"]))
    seen = set()
    unique_bases = []
    for b in bases:
        scene_idx = _scene_index(b)
        if scene_idx in seen:
            continue
        seen.add(scene_idx)
        unique_bases.append({
            "scene_index": scene_idx,
            "slide_number": b.get("slide_number"),
            "slide_canonical_index": b.get("slide_canonical_index", b.get("same_slide_canonical")),
            "slide_visit_order": b.get("slide_visit_order", b.get("same_slide_visit_order", 1)),
            "slide_is_revisit": bool(b.get("slide_is_revisit", b.get("same_slide_is_revisit", False))),
            "timestamp_sec": b["timestamp_sec"],
        })
    unique_bases.sort(key=lambda x: x["timestamp_sec"])
    ranges = []
    for i, b in enumerate(unique_bases):
        start = b["timestamp_sec"]
        end = unique_bases[i + 1]["timestamp_sec"] if i + 1 < len(unique_bases) else duration_sec
        ranges.append({
            "scene_index": b["scene_index"],
            "slide_number": b.get("slide_number"),
            "slide_canonical_index": b.get("slide_canonical_index"),
            "slide_visit_order": b.get("slide_visit_order", 1),
            "slide_is_revisit": b.get("slide_is_revisit", False),
            "start_sec": start,
            "end_sec": end,
        })
    return ranges


def group_segments_by_scene_and_context(
    segments: list[dict],
    scene_ranges: list[dict],
    duration_sec: float,
    *,
    use_llm_merge: bool = True,
    use_pause_sentence: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    먼저 scene occurrence별로 세그먼트를 나누고, 각 scene 내부에서 의미 전환점 기준으로 컨텍스트 구성.
    use_pause_sentence: True면 침묵/문장끝 기준 분할 추가 (나중에 사용할 옵션).

    반환: (groups_flat, scenes_structure)
    - groups_flat: 강조 분석용 그룹 리스트 (start, end, text, segment_indices, scene_index, context_index_in_scene)
    - scenes_structure: 최종 JSON용 [ { scene_index, slide_number, start_sec, end_sec, text, contexts: [...] } ]
    """
    if not segments or not scene_ranges:
        return [], []

    seg_to_scene = []
    for i, seg in enumerate(segments):
        t = seg["start"]
        scene_idx = None
        for r in scene_ranges:
            if r["start_sec"] <= t < r["end_sec"]:
                scene_idx = r["scene_index"]
                break
        if scene_idx is None and scene_ranges:
            if t < scene_ranges[0]["start_sec"]:
                scene_idx = scene_ranges[0]["scene_index"]
            else:
                scene_idx = scene_ranges[-1]["scene_index"]
        seg_to_scene.append(scene_idx)

    scenes_structure = []
    groups_flat = []

    for r in scene_ranges:
        sidx = r["scene_index"]
        start_sec = r["start_sec"]
        end_sec = r["end_sec"]
        indices_in_scene = [i for i in range(len(segments)) if seg_to_scene[i] == sidx]
        if not indices_in_scene:
            scenes_structure.append({
                "scene_id": f"scene/{int(sidx):04d}",
                "scene_index": sidx,
                "slide_number": r.get("slide_number"),
                "slide_canonical_index": r.get("slide_canonical_index"),
                "slide_visit_order": r.get("slide_visit_order", 1),
                "slide_is_revisit": r.get("slide_is_revisit", False),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": "",
                "contexts": [],
            })
            continue

        scene_segments = [segments[i] for i in indices_in_scene]
        initial_groups = []
        for k, i in enumerate(indices_in_scene):
            seg = segments[i]
            initial_groups.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": (seg.get("text") or "").strip(),
                "segment_indices": [i],
            })

        break_after: set[int] = set()
        if use_llm_merge and len(initial_groups) > 1:
            break_after = decide_context_breaks(initial_groups)

        context_groups = []
        cur = {
            "start": initial_groups[0]["start"],
            "end": initial_groups[0]["end"],
            "text": initial_groups[0]["text"],
            "segment_indices": list(initial_groups[0]["segment_indices"]),
        }
        for j in range(len(initial_groups) - 1):
            g_next = initial_groups[j + 1]
            if j in break_after:
                context_groups.append(cur)
                cur = {
                    "start": g_next["start"],
                    "end": g_next["end"],
                    "text": g_next["text"],
                    "segment_indices": list(g_next["segment_indices"]),
                }
            else:
                cur["end"] = g_next["end"]
                cur["text"] = f"{cur['text']} {g_next['text']}".strip()
                cur["segment_indices"].extend(g_next["segment_indices"])
        context_groups.append(cur)
        context_groups = _split_oversized_groups(context_groups, segments)

        scene_text = " ".join((g["text"] for g in context_groups if g["text"])).strip()
        contexts_for_scene = []
        for cix, g in enumerate(context_groups):
            g_flat = {
                "start": g["start"],
                "end": g["end"],
                "text": g["text"],
                "segment_indices": g["segment_indices"],
                "scene_index": sidx,
                "slide_number": r.get("slide_number"),
                "context_index_in_scene": cix,
            }
            groups_flat.append(g_flat)
            segs_in_context = [
                {"start": segments[i]["start"], "end": segments[i]["end"], "text": (segments[i].get("text") or "").strip()}
                for i in g["segment_indices"]
            ]
            contexts_for_scene.append({
                "context_index": cix,
                "start": g["start"],
                "end": g["end"],
                "text": g["text"],
                "segment_indices": g["segment_indices"],
                "segments": segs_in_context,
            })
        scenes_structure.append({
            "scene_id": f"scene/{int(sidx):04d}",
            "scene_index": sidx,
            "slide_number": r.get("slide_number"),
            "slide_canonical_index": r.get("slide_canonical_index"),
            "slide_visit_order": r.get("slide_visit_order", 1),
            "slide_is_revisit": r.get("slide_is_revisit", False),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": scene_text,
            "contexts": contexts_for_scene,
        })

    return groups_flat, scenes_structure
