# Crosscheck Scoring Design

이 문서는 analyzer의 현재 crosscheck 점수화 방식을 정리한다.

## 목적

crosscheck의 목표는 모델에게 단순히 `맞다/아니다`를 고르게 하는 것이 아니라,
하나의 issue가 교수에게 보여줄 만큼 실제로 문제가 남아 있는지를 0~1 운영 점수로 만드는 것이다.

이 점수는 LLM이 직접 예측한 "오류 확률"이 아니다.
LLM이 같은 기준의 세부 항목을 채점하면, 코드가 그 항목을 가중합해서 만든 구조화된 점수다.

따라서 `final_score`의 의미는 다음과 같다.

```text
제공된 강의 문맥에서 해당 issue를 교수에게 올릴 근거가 얼마나 남아 있는가
```

## 공통 Issue Type

Verifier와 crosscheck는 같은 4개 issue type을 사용한다.

| Code | type | 표시명 | 기준 |
| --- | --- | --- | --- |
| A | `factual_error` | 발언 자체 오류 | 발화 자체의 객관 사실, 정의, 분류, 수치, 순서, 원인-결과, 작동 방식이 강의 문맥을 함께 봐도 틀린 경우 |
| B | `temporal_error` | 시간적 오류 | 현재성, 최신성, 지원 여부, 사용 여부, 시점 의존 수치나 상태가 기준 시점에서 틀리거나 확인이 필요한 경우 |
| C | `scope_overclaim` | 범위 과잉 단정 | 특정 조건에서는 맞지만 모든 경우에 맞는 것처럼 범위, 조건, 예외, 다른 가능성을 닫아 말한 경우 |
| D | `confusing_explanation` | 혼동 가능 설명 | 명백한 사실 오류라고 단정되지는 않더라도 학생이 핵심 개념, 주체, 과정, 원인, 조건을 잘못 연결해 외울 가능성이 큰 경우 |

A-D 유형 점수는 합이 1인 확률 분포가 아니다.
각 유형에 독립적으로 해당하는 정도를 0~1로 채점한다.

예를 들어 한 issue가 다음처럼 동시에 여러 성격을 가질 수 있다.

```json
{
  "issue_type_scores": {
    "factual_error": 0.85,
    "temporal_error": 0.0,
    "scope_overclaim": 0.7,
    "confusing_explanation": 0.6
  }
}
```

대표 유형은 가장 높은 점수의 유형이고, 기준값 이상인 나머지는 보조 유형으로 남긴다.

## 입력 단위

기본 실행 경로는 `judge_claim_batch()`다.

같은 슬라이드에 걸린 issue들을 하나의 batch로 묶어 crosscheck 모델에 보낸다.
각 issue는 `i0001`, `i0002` 같은 `issue_id`로 분리된다.

프롬프트에 들어가는 공통 문맥은 다음과 같다.

- 강의 도메인
- 대상 슬라이드 제목과 시간 범위
- 이전 슬라이드 제목과 시간 범위
- 이전 슬라이드와 현재 슬라이드의 슬라이드 텍스트
- 이전 슬라이드와 현재 슬라이드에서 발화된 전체 utterance
- 각 대상 발화 전후 +/-5개 utterance를 병합한 문맥
- issue 목록

## 모델 출력

모델은 `verdict`나 `confidence`를 직접 출력하지 않는다.
모델은 issue마다 다음 구조를 출력한다.

```json
{
  "issue_id": "i0001",
  "issue_type": "scope_overclaim",
  "issue_type_code": "C",
  "issue_type_scores": {
    "A": 0.25,
    "B": 0.0,
    "C": 0.9,
    "D": 0.6
  },
  "issue_type_rationale": "닫힌 표현과 문맥 후에도 남는 반례 가능성이 가장 강하므로 C로 분류함",
  "criteria_scores": {
    "issue_presence": 1.0,
    "context_unresolved": 0.75,
    "evidence_strength": 0.75
  },
  "criteria_evidence": {
    "issue_presence": "학생이 잘못 외울 명제가 남음",
    "context_unresolved": "제공된 문맥이 조건을 충분히 보완하지 못함",
    "evidence_strength": "강의 수준에서 설명 가능한 반례나 조건 차이가 있음"
  },
  "reason": "전체 판단 이유"
}
```

## Criteria

현재 criteria와 가중치는 다음과 같다.

| Field | Weight | 의미 |
| --- | ---: | --- |
| `issue_presence` | 0.40 | 원문 발화와 강의 흐름 기준으로 실제 문제가 남는가 |
| `context_unresolved` | 0.35 | 주변 발화와 슬라이드 문맥을 함께 봐도 문제가 해소되지 않는가 |
| `evidence_strength` | 0.25 | 반례, 정의 차이, 수치 오류, 현행성 오류, 직접 충돌, 조건/범위 차이 같은 구체 근거가 있는가 |

가중치 합은 1.0이다.

기존의 `local_context_unresolved`와 `slide_context_unresolved`는 `context_unresolved`로 합쳤다.
crosscheck 입력에서 발화 문맥과 슬라이드 문맥은 함께 제공되므로, 둘을 분리하면 같은 근거를 두 번 점수화할 위험이 크다.

기존의 `teaching_priority`는 검증 점수에서 제거했다.
이 값은 "문제가 실제로 남는가"보다 "얼마나 우선적으로 보여줄 것인가"에 가까워 주관성이 크다.
필요하면 추후 severity 또는 정렬 보조값으로만 다룬다.

## 단일 모델 점수 계산

issue를 `i`, crosscheck 모델을 `m`이라고 두면, 모델 `m`은 issue `i`에 대해 세 개의 criteria 점수를 낸다.

```text
c_m,i =
[
  issue_presence_m,i,
  context_unresolved_m,i,
  evidence_strength_m,i
]
```

criteria 가중치는 다음 벡터로 고정한다.

```text
a = [0.40, 0.35, 0.25]
```

모델 `m`의 gate 적용 전 원점수는 다음과 같다.

```text
raw_score_m,i =
  issue_presence     * 0.40
+ context_unresolved * 0.35
+ evidence_strength  * 0.25
```

벡터로 쓰면 다음과 같다.

```text
raw_score_m,i = a · c_m,i
```

## Gate

단순 가중합만 쓰면 핵심 조건이 부족한데도 중간 이상의 점수가 나올 수 있다.
이를 막기 위해 세 가지 상한 gate를 둔다.

- `issue_presence < 0.5`이면 최종 모델 점수는 최대 0.39
- `context_unresolved < 0.5`이면 최종 모델 점수는 최대 0.39
- `evidence_strength < 0.5`이면 최종 모델 점수는 최대 0.79

의미는 다음과 같다.

- issue 자체가 실제 문제라는 점이 확인되지 않으면 낮은 점수 구간으로 제한한다.
- 제공된 문맥이 문제를 해소하면 낮은 점수 구간으로 제한한다.
- 구체 근거가 약하면 확정 구간까지 올라가지 못하게 제한한다.

수식은 다음과 같다.

```text
cap_m,i = 1.00

if issue_presence_m,i < 0.5:
  cap_m,i = min(cap_m,i, 0.39)

if context_unresolved_m,i < 0.5:
  cap_m,i = min(cap_m,i, 0.39)

if evidence_strength_m,i < 0.5:
  cap_m,i = min(cap_m,i, 0.79)

model_score_m,i = min(raw_score_m,i, cap_m,i)
```

## 여러 모델의 Issue 점수 결합

각 모델이 낸 `model_score_m,i`는 모델별 weight를 곱해 최종 점수로 합친다.

모델 weight를 `w_m`이라고 할 때, issue `i`의 최종 점수는 다음과 같다.

```text
final_score_i =
sum_m(model_score_m,i * w_m) / sum_m(w_m)
```

전체 흐름을 한 줄로 쓰면 다음과 같다.

```text
final_score_i =
sum_m( min(a · c_m,i, cap_m,i) * w_m ) / sum_m(w_m)
```

## 여러 모델의 A-D 유형 점수 결합

유형 점수는 issue 점수와 분리한다.

모델 `m`이 낸 유형 `t`의 점수를 다음처럼 둔다.

```text
type_score_m,i,t
```

이때 `t`는 A/B/C/D 중 하나다.

유형 점수를 결합할 때는 모델 weight만 쓰지 않는다.
해당 모델이 issue 자체를 얼마나 강하게 유지했는지도 반영한다.

```text
effective_weight_m,i = w_m * model_score_m,i
```

최종 유형 점수는 다음과 같다.

```text
final_type_score_i,t =
sum_m(effective_weight_m,i * type_score_m,i,t)
/ sum_m(effective_weight_m,i)
```

이 방식의 의미는 다음과 같다.

- 어떤 모델이 "이 issue는 거의 문제가 아니다"라고 낮게 본 경우, 그 모델의 A-D 유형 판단도 최종 유형 결정에 작게 반영된다.
- A-D는 합이 1이 아니므로, 여러 유형이 동시에 높게 남을 수 있다.
- 대표 유형은 `max_t(final_type_score_i,t)`로 결정한다.
- 보조 유형은 기본적으로 `0.45` 이상인 나머지 유형을 남긴다.

## 최종 상태

최종 상태는 `final_score`로만 결정한다.

| final_score | status | 의미 |
| ---: | --- | --- |
| `>= 0.80` | `confirmed` | 확정 |
| `>= 0.40` and `< 0.80` | `professor_check` | 교수 확인 |
| `< 0.40` | `rejected` | 기각 |

모델은 최종 `status`를 직접 정하지 않는다.
모델은 criteria 점수와 A-D 유형 점수만 낸다.
최종 점수, 최종 상태, 대표 유형은 서버 코드가 계산한다.

## 최종 JSON 핵심 필드

```json
{
  "crosscheck_score": 0.84,
  "crosscheck_weighted_status": "confirmed",
  "issue_type": "factual_error",
  "issue_type_code": "A",
  "issue_type_scores": {
    "factual_error": 0.82,
    "temporal_error": 0.03,
    "scope_overclaim": 0.67,
    "confusing_explanation": 0.58
  },
  "primary_issue_type": {
    "type": "factual_error",
    "code": "A",
    "label": "발언 자체 오류",
    "score": 0.82
  },
  "secondary_issue_types": [
    {
      "type": "scope_overclaim",
      "code": "C",
      "label": "범위 과잉 단정",
      "score": 0.67
    },
    {
      "type": "confusing_explanation",
      "code": "D",
      "label": "혼동 가능 설명",
      "score": 0.58
    }
  ]
}
```

## 해석상 주의

- `crosscheck_score`는 오류 확률이 아니라 교수에게 올릴 근거가 남은 정도다.
- `issue_type_scores`는 합이 1인 확률 분포가 아니다.
- `issue_type_scores`는 issue의 성격을 나타내고, `crosscheck_score`는 issue를 유지할지 판단한다.
- 최종 판정은 특정 모델 하나가 아니라 weighted ensemble aggregator가 계산한다.
