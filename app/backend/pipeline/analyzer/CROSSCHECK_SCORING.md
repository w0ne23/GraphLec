# Crosscheck Scoring Design

이 문서는 analyzer의 현재 crosscheck 점수화 방식을 정리한다.

## 목적

crosscheck의 목표는 모델에게 단순히 `맞다/아니다`를 고르게 하는 것이 아니라,
하나의 issue가 교수에게 보여줄 만큼 실제로 문제가 남아 있는지를 0~1 점수로 안정적으로 만드는 것이다.

현재 방식에서 0~1 점수는 LLM이 자기 확신도를 직접 예측한 값이 아니다.
LLM이 같은 기준의 세부 항목을 채점하면, 코드가 그 항목을 가중합해서 만든 구조화된 점수다.

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

대략적인 입력 형태는 다음과 같다.

```text
## 도메인
운영체제

## 대상 슬라이드
슬라이드 7 제목 (03:10 ~ 03:40)

## 이전 슬라이드
슬라이드 6 제목 (02:40 ~ 03:10)

## 이전+현재 슬라이드 내용 (슬라이드 텍스트 + 강의자 발화)
[슬라이드 6]
[슬라이드 텍스트]
...
[강의자 발화]
  U0024 [180.0s] ...

[슬라이드 7]
[슬라이드 텍스트]
...
[강의자 발화]
  U0029 [237.1s] ...

## 대상 발화 전후 +/-5개 병합 문맥
   U0024 [180.0s, slide 6] ...
>> i0001 U0029 [237.1s, slide 7] ...
   U0030 [240.2s, slide 7] ...
>> i0002 U0033 [258.4s, slide 7] ...

## 지적 목록
### i0001
- utterance_id: U0029
- 유형: 범위 과잉 단정 (scope_overclaim)
- claim: ...
- 문제: ...
```

batch 응답에서 특정 issue가 빠지거나 JSON 파싱이 실패하면, 누락된 issue만 다시 1건짜리 batch로 재시도한다.
이 fallback도 같은 prompt builder, 같은 5개 criteria, 같은 `results` 배열 출력 구조를 사용한다.

## 모델 출력

모델은 `verdict`나 `confidence`를 직접 출력하지 않는다.
모델은 issue마다 아래 JSON을 출력한다.

```json
{
  "issue_id": "i0001",
  "criteria_scores": {
    "misinformation_risk": 1.0,
    "nearby_context_unresolved": 0.5,
    "slide_context_unresolved": 0.25,
    "concrete_basis": 0.75,
    "student_impact": 0.5
  },
  "criteria_evidence": {
    "misinformation_risk": "학생이 잘못 외울 수 있는 명제와 그 이유",
    "nearby_context_unresolved": "+/-5개 발화에서 해소되는지 여부",
    "slide_context_unresolved": "슬라이드 텍스트/구조에서 해소되는지 여부",
    "concrete_basis": "반례, 충돌, 조건 누락 등 구체 근거",
    "student_impact": "후속 개념에서 생길 학생 오해"
  },
  "reason": "전체 판단 이유"
}
```

점수는 0.0~1.0 사이 값이다.
프롬프트는 기본적으로 `0.0 / 0.5 / 1.0`을 권장하고, 필요한 경우 `0.25 / 0.75`를 허용한다.
즉 정밀한 확률이 아니라 등급형 점수에 가깝다.

## Criteria

현재 criteria와 가중치는 다음과 같다.

| Field | Weight | 의미 |
| --- | ---: | --- |
| `misinformation_risk` | 0.15 | 해당 claim이 학생에게 잘못된 정보를 주는 명제가 되는가 |
| `nearby_context_unresolved` | 0.25 | 대상 발화 전후 +/-5개 발화에서 문제가 해소되지 않는가 |
| `slide_context_unresolved` | 0.25 | 해당 슬라이드의 텍스트/그림/구조에서 문제가 해소되지 않는가 |
| `concrete_basis` | 0.20 | 반례, 정답 충돌, 현재성 오류, 범위 과잉, 주체/과정 혼동 같은 구체 근거가 있는가 |
| `student_impact` | 0.15 | 학생이 실제로 잘못 외우거나 후속 개념을 혼동할 위험이 구체적인가 |

가중치 합은 1.0이다.

### 범위 과잉 단정 완화 원칙

`모든`, `오직`, `~만`, `독점` 같은 닫힌 표현은 단독으로 높은 점수의 근거가 되지 않는다.
범위 과잉 issue를 유지하려면 원문, 전후 +/-5개 발화, 이전+현재 슬라이드를 함께 본 뒤에도
학생이 아래와 같은 닫힌 명제를 실제로 외울 가능성이 남아야 한다.

- 다른 가능성은 불가능하다.
- 다른 주체는 관여하지 않는다.
- 이 조건에서만 성립한다.
- 대표 경로나 관리 주체 설명이 아니라 배타적 사실 명제로 제시되었다.

반례가 강의 범위 밖의 더 상위/하위 계층, 예외적 구현, 고급 세부사항에만 의존하면
`concrete_basis`와 `student_impact`를 낮게 준다. 역할, 책임, 관리 주체, 대표 경로를 설명하는 표준적 표현은
문맥상 핵심 명제가 맞으면 낮은 점수로 처리한다.

## 단일 모델 점수 계산

하나의 모델이 하나의 issue에 대해 낸 점수는 다음 산식으로 계산한다.

```text
issue_score =
  misinformation_risk       * 0.15
+ nearby_context_unresolved * 0.25
+ slide_context_unresolved  * 0.25
+ concrete_basis            * 0.20
+ student_impact            * 0.15
```

예시:

```text
misinformation_risk       = 1.00
nearby_context_unresolved = 0.50
slide_context_unresolved  = 0.25
concrete_basis            = 0.75
student_impact            = 0.50

issue_score =
  1.00 * 0.15
+ 0.50 * 0.25
+ 0.25 * 0.25
+ 0.75 * 0.20
+ 0.50 * 0.15
= 0.5625
```

## Gate

단순 가중합만 쓰면 핵심 조건이 부족한데도 중간 이상의 점수가 나올 수 있다.
이를 막기 위해 현재는 세 가지 상한 gate를 둔다.

- `misinformation_risk < 0.5`이면 최종 모델 점수는 최대 0.44
- `nearby_context_unresolved < 0.5`이고 `slide_context_unresolved < 0.5`이면 최종 모델 점수는 최대 0.44
- `concrete_basis < 0.5`이고 `student_impact < 0.5`이면 최종 모델 점수는 최대 0.79

의미는 다음과 같다.

- 학생에게 잘못된 정보를 준다는 점이 확인되지 않으면 rejected 쪽으로 제한한다.
- 주변 발화와 슬라이드가 둘 다 문제를 해소하면 rejected 쪽으로 제한한다.
- 구체 근거와 학생 영향이 둘 다 약하면 confirmed까지 올라가지 못하게 제한한다.

## 여러 모델 결합

각 모델이 낸 `issue_score`는 모델별 weight를 곱해 최종 점수로 합친다.

```text
final_score =
sum(model_issue_score * model_weight) / sum(model_weight)
```

예시:

```text
gpt-5.4              score=0.80, weight=0.38
claude-sonnet-4.5    score=0.70, weight=0.34
gpt-5.4-mini         score=0.55, weight=0.18
grok-4.3             score=0.40, weight=0.10

final_score =
(0.80*0.38 + 0.70*0.34 + 0.55*0.18 + 0.40*0.10) / (0.38 + 0.34 + 0.18 + 0.10)
= 0.6810
```

최종 상태는 threshold로 결정한다.

| Range | Status | 의미 |
| --- | --- | --- |
| `0.80 ~ 1.00` | `confirmed` | 교수에게 확정 이슈로 보여줌 |
| `0.45 ~ 0.79` | `professor_check` | 교수 확인 대상으로 보여줌 |
| `0.00 ~ 0.44` | `rejected` | 최종 화면에는 보여주지 않음 |

## 단순 1/0 방식과의 차이

초기 대안은 모델이 문제라고 생각하면 `1`, 아니면 `0`을 주는 방식이었다.

```text
model_score = 1 if issue_valid else 0
final_score = weighted average(model_score)
```

이 방식은 단순하고 해석이 쉽다.
하지만 다음 정보가 사라진다.

- issue 자체는 그럴듯하지만 슬라이드가 상당 부분 해소한 경우
- 주변 발화 때문에 확정 오류는 아니지만 교수 확인은 필요한 경우
- 사실 오류는 아니지만 학생 오해 위험이 남는 경우
- 모델 간 의견 차이가 단순 yes/no보다 연속적으로 나타나는 경우

현재 채택한 방식은 단순 1/0 판정보다 복잡하지만, 왜 점수가 낮아졌는지 criteria별로 확인할 수 있다.
따라서 1차 구현은 현재의 criteria 기반 0~1 점수화를 사용한다.

## 해석 주의

이 점수는 통계적으로 보정된 확률이 아니다.
LLM이 직접 낸 confidence도 아니다.

현재 점수는 다음을 위한 운영 지표다.

- 여러 모델 결과를 같은 좌표계에서 비교
- issue를 `confirmed`, `professor_check`, `rejected`로 나누기
- 점수가 낮아진 이유를 criteria별로 추적
- 특정 모델이 모두 agree하는 경향이 있는지 리포트에서 확인

따라서 `0.63`을 “63% 확률로 오류”라고 해석하면 안 된다.
정확한 해석은 “현재 criteria와 model weight 기준으로 교수 확인 구간에 들어간 issue”다.
