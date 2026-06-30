# GraphLec QnA Evaluation

이 폴더는 `d92fe381-188f-4e5e-b613-7aa8d4b09c42` 운영체제 강의의 QnA 평가셋과 실행 결과를 저장한다.

## 구성

- `d92fe381_os_eval_set.jsonl`: 내용형 40개, 구조형 30개 평가 질문과 기준 답변
- `runs/`: QnA 호출 결과와 평가 결과 저장 위치

## Docker 기준 실행 순서

1. Docker Compose 서비스를 실행한다.

```bash
docker compose up -d db neo4j query_service backend
```

2. 평가셋 폴더가 컨테이너에 마운트되도록 Compose를 반영한다.

`docker-compose.yml`의 `backend` 서비스에는 아래 볼륨이 필요하다.

```yaml
- ./evaluations:/evaluations
```

이미 컨테이너가 떠 있었다면 backend를 다시 생성한다.

```bash
docker compose up -d --force-recreate backend
```

3. QnA API로 평가 질문을 일괄 실행한다.

백엔드 API를 통해 평가하면 QnA 전에 그래프 적재가 자동으로 실행된다.

```bash
docker compose exec backend python /app/scripts/eval/run_qna_eval.py \
  --dataset /evaluations/d92fe381_os_eval_set.jsonl \
  --out /evaluations/runs/d92_qna_results.jsonl \
  --stem d92fe381-188f-4e5e-b613-7aa8d4b09c42 \
  --backend-url http://localhost:8000
```

먼저 1개만 테스트하려면 `--limit 1`을 붙인다.

```bash
docker compose exec backend python /app/scripts/eval/run_qna_eval.py \
  --dataset /evaluations/d92fe381_os_eval_set.jsonl \
  --out /evaluations/runs/d92_qna_results_test.jsonl \
  --stem d92fe381-188f-4e5e-b613-7aa8d4b09c42 \
  --backend-url http://localhost:8000 \
  --limit 1
```

`query_service`의 `/internal/query`를 직접 호출하는 방식은 기본 실행 방식으로 권장하지 않는다. 백엔드의 `/results/{lecture_id}/query`를 호출해야 QnA 전에 해당 강의 그래프가 Neo4j에 자동 적재된다.

4. 내용형 질문은 RAGAS로 평가한다.

RAGAS는 `app/backend/requirements.txt`에 포함되어 있으므로 backend 이미지를 새로 빌드하면 함께 설치된다.
만약 `ModuleNotFoundError: No module named 'ragas'`가 나오면 아래처럼 backend 이미지를 다시 빌드한다.

```bash
docker compose build backend
docker compose up -d backend
```

```bash
docker compose exec backend python /app/scripts/eval/run_ragas_content_eval.py \
  --input /evaluations/runs/d92_qna_results.jsonl \
  --out /evaluations/runs/d92_ragas_content_scores.csv
```

5. 구조형 질문은 위치/구간 정확도로 평가한다.

```bash
docker compose exec backend python /app/scripts/eval/evaluate_structural.py \
  --input /evaluations/runs/d92_qna_results.jsonl \
  --out /evaluations/runs/d92_structural_scores.csv
```

## 평가 분리 기준

내용형 질문은 답변의 근거 충실성, 질문 관련성, 검색 근거 품질을 평가하므로 RAGAS를 사용한다.
구조형 질문은 슬라이드 번호, 장면 번호, 시간 구간 정답 여부가 핵심이므로 별도 정확도 지표를 사용한다.
