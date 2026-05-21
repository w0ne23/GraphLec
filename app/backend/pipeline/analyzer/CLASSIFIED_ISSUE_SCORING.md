# Context-Based Classified Issue Scoring

이 문서는 현재 기본 analyzer 실행 경로의 context 기반 issue 점수 산정 방식을 설명한다.
legacy verifier 진입점과 호환 필드가 제거된 현재 코드 흐름 기준 문서다.

## 한 줄 요약

최종 severity judge는 각 모델에게 issue별로 세 가지 점수를 받는다.

- `is_valid_issue`
- `category_severity`
- `context_resolution`

모델별 최종 점수는 다음 산식으로 코드가 직접 계산한다.

```text
final_model_score =
  is_valid_issue
* category_severity
* (1.0 - context_resolution)
```

여러 모델을 쓰는 경우 모델별 `final_model_score`에 모델 가중치를 곱해 합산한 값이
issue의 `final_severity_score`가 된다.

## 현재 실행 경로

백엔드 verifier 실행은 `app/backend/pipeline/main.py`에서
`pipeline.analyzer.run_all`을 실행하며, `run_all.py`의 기본 경로는
`run_classified_issue_pipeline()`이다.

현재 흐름:

1. claim extraction
2. first issue judge
3. issue type classifier
4. classified issue severity judge
5. web-friendly `content_verification.v2` 변환

## 입력 파일과 context 단위

기본 입력은 analyzer용 `*_merged_clean.json`이다.
이 파일의 `slides[].contexts[]`가 context lookup으로 들어간다.

issue type classifier가 만든 다음 단계 입력은 `classified_issue_input.v1`이고,
severity judge는 이 입력과 `*_merged_clean.json`의 context 정보를 함께 사용한다.

각 issue는 보통 다음 정보를 가진다.

- `claim_id`
- `claim_text`
- `resolved_claim`
- `category`
- `location.slide_number`
- `context.context_id`
- `context.context_ids`

context window는 기본 `2`다.
즉, 대상 context 주변 앞뒤 context를 함께 judge prompt에 넣어 문맥 해소 여부를 보게 한다.

## Severity Judge Prompt 출력

`classified_issue_verifier.py`의 `_build_prompt()`는 모델에게
`final_model_score`를 직접 쓰라고 하지 않는다.
모델은 아래 세 점수만 출력한다.

```json
{
  "judgments": [
    {
      "id": "입력 id",
      "judgment": "valid_issue",
      "is_valid_issue": 0.0,
      "category_severity": 0.0,
      "context_resolution": 0.0,
      "reason": "판단 근거",
      "minimal_fix": "수정안"
    }
  ]
}
```

각 점수 의미:

| Field | 의미 |
| --- | --- |
| `is_valid_issue` | 이 분류 기준으로 실제 issue일 가능성 |
| `category_severity` | 해당 분류 안에서 오류가 얼마나 심각한지 |
| `context_resolution` | 제공된 context/슬라이드 문맥이 issue를 얼마나 해소하는지 |

`context_resolution`은 반대로 작동한다.
문맥이 문제를 많이 해소할수록 값이 커지고,
최종 점수에서는 `1.0 - context_resolution`로 감점된다.

## 모델별 점수 계산

모델 응답은 `_normalize_judgment_row()`에서 정규화된다.
모든 점수는 `0.0~1.0`으로 clamp된다.

```text
is_valid_issue     = clamp01(model_output.is_valid_issue)
category_severity  = clamp01(model_output.category_severity)
context_resolution = clamp01(model_output.context_resolution)

final_model_score =
  clamp01(is_valid_issue * category_severity * (1.0 - context_resolution))
```

예시:

```text
is_valid_issue     = 0.90
category_severity  = 0.80
context_resolution = 0.25

final_model_score =
  0.90 * 0.80 * (1.0 - 0.25)
= 0.54
```

이 산식 때문에 하나의 축이라도 낮으면 최종 점수가 크게 내려간다.

- issue 자체가 애매하면 `is_valid_issue`가 낮아져 점수가 내려간다.
- 분류 안에서 심각도가 낮으면 `category_severity`가 낮아져 점수가 내려간다.
- 주변 context와 슬라이드가 문제를 해소하면 `context_resolution`이 높아져 점수가 내려간다.

## 여러 모델 결합

여러 모델 결과는 `_weighted_final_score()`에서 합쳐진다.

```text
final_severity_score =
sum(final_model_score * model_weight)
```

`model_weight`는 `_parse_model_weights()`에서 전체 합이 1.0이 되도록 정규화된다.
다만 특정 모델 응답이 parse 실패 등으로 `status != "ok"`이면 그 모델은 합산에서 제외되고,
남은 모델 weight를 다시 1.0으로 재정규화하지는 않는다.
제외된 weight는 `missing_model_weight`로 기록된다.

기본 모델과 기본 weight:

```text
DEFAULT_MODELS = ("gpt", "claude", "grok")
DEFAULT_MODEL_WEIGHTS = "gpt=0.4,claude=0.4,grok=0.2"
```

모델 weight는 다음 인자로 override할 수 있다.

- `verifier_model_weights`
- standalone severity judge CLI의 `--model-weights`
- standalone CLI 기본값으로 쓰이는 `CLASSIFIED_ISSUE_VERIFIER_MODEL_WEIGHTS`

모델 응답 파싱 실패 등으로 `status != "ok"`이면 해당 모델의 점수는 합산하지 않고,
그 모델 weight는 `missing_model_weight`로 기록된다.

## 상태 판정

`build_content_verification_view()`는 `final_severity_score`를 UI용 status로 바꾼다.

기본 threshold:

```text
CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD = 0.50
CLASSIFIED_ISSUE_VERIFIER_REJECTED_THRESHOLD  = 0.20
```

상태:

| Score range | Status |
| --- | --- |
| `>= 0.50` | `confirmed` |
| `> 0.20` and `< 0.50` | `professor_check` |
| `<= 0.20` | `rejected` |

환경변수로 threshold를 바꿀 수 있다.

## Web 결과 필드 매핑

최종 web 결과는 `content_verification.v2` 형태로 변환되며, severity 이름을 canonical 필드로 사용한다.

| Web field | 실제 source |
| --- | --- |
| `feedback_items[].severity_score` | `final_severity_score` |
| `feedback_items[].severity_score_percent` | `final_severity_score * 100` |
| `feedback_items[].severity_status` | `_status_from_severity(final_severity_score)` |
| `checks.severity.model_results[].confidence` | 각 모델의 `final_model_score` |
| `checks.severity.model_results[].vote_score` | 각 모델의 `final_model_score` |
| `checks.severity.model_results[].score` | 각 모델의 `final_model_score` |

별도 상세 필드도 함께 남는다.

- `classified_issue_verifier.final_severity_score`
- `classified_issue_verifier.average_is_valid_issue`
- `classified_issue_verifier.average_category_severity`
- `classified_issue_verifier.average_context_resolution`
- `classified_issue_verifier.model_disagreement`
- `classified_issue_verifier.needs_manual_review`

## Legacy Verifier와의 차이

legacy verifier scoring 진입점과 모듈은 제거되어 현재 기본 verifier 실행 경로에 남아 있지 않다.
따라서 현재 결과값을 설명할 때 기준이 되는 산식은 아래다.

```text
final_model_score =
  is_valid_issue
* category_severity
* (1.0 - context_resolution)
```
