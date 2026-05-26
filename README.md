# GraphLec: 통합 강의 분석 및 지식그래프 생성 파이프라인

GraphLec은 강의 영상에서 시각적 정보(슬라이드)와 청각적 정보(음성)를 결합하여, 고정밀 전사, 강조 구간 분석, **지식 그래프(Parquet)**·**벡터 검색(LanceDB)**·**웹 질의(Django + FastAPI)**까지 이어지는 멀티모달 파이프라인입니다.

---

## 1. 주요 기능

### 영상 및 슬라이드 분석

- 슬라이드 변화 감지: MSE(Mean Squared Error) 기반으로 영상 내 슬라이드 전환 시점을 추적하여 이미지를 추출합니다.
- 시각 텍스트 추출: Gemini Vision을 사용하여 각 슬라이드의 텍스트와 관련 정보를 추출합니다.

### 음성 전사 및 강조 분석

- 슬라이드별 맞춤 전사: Groq Whisper(whisper-large-v3-turbo) 등을 사용하여 슬라이드 구간에 맞춰 오디오를 잘라 전사합니다.
- 2단계 텍스트 교정: 전문용어·오타 교정 후, 슬라이드 컨텍스트를 반영해 자연스러운 문장으로 정리합니다.
- 강조 구간 탐지: 오디오 통계·키워드·LLM을 결합해 핵심 구간을 탐지합니다.

### 지식그래프·검색·Q&A

- 멀티모달 통합: 슬라이드 텍스트와 오디오 전사를 타임스탬프 기준으로 병합(`fused.json`)합니다.
- 지식 그래프: 개념·관계 트리플을 수집해 `{stem}_graph_triples.parquet`, `{stem}_nodes.parquet`, `{stem}_edges.parquet`로 저장합니다.
- 벡터 검색: 청크 임베딩 후 LanceDB(`data/lancedb` 등)에 적재하여 stem 단위 검색합니다.
- 질의응답: FastAPI 질의 서비스 + Django 웹 UI에서 자연어 질의에 답합니다.

---

## 2. 파이프라인 구조

`main.py` 기준 단계 요약입니다.

```text
강의 영상 (.mp4)
    ├─ [병렬] Stage 1A 슬라이드 추출 · Stage 1B 오디오 품질
    ├─ Stage 2 슬라이드 텍스트/강조
    ├─ [병렬] Stage 3A 필기 강조 · Stage 3B 오디오 전사·교정·강조
    ├─ [병렬] Stage 4A 슬라이드 분류 · Stage 4B by_slide 저장
    ├─ Stage 5 퓨전 → {stem}_fused.json
    ├─ Stage 6 그래프 트리플 → Parquet (triples / nodes / edges)
    ├─ Neo4j 적재 (옵션) — nodes/edges Parquet → Neo4j (`--load-neo4j`로 활성화)
    └─ Stage 7 Lance 인덱스 → Parquet 백업 + LanceDB

질의: Neo4j(구조/내용) + Lance 보조·재순위(내용형) + Gemini (query_service) ← Django 프록시
```

---

## 3. 설치 및 설정

### 시스템

- **ffmpeg** / **ffprobe** (영상·오디오 처리). `requirements.txt`로 설치되지 않으므로 OS 패키지로 설치합니다.

### 패키지 설치

```bash
pip install -r requirements.txt
```

GPU용 PyTorch가 필요하면 `requirements.txt` 상단 주석의 CUDA 인덱스 URL을 참고합니다.

### 환경 변수 (프로젝트 루트 `.env`)

파이프라인(`config.py`)에서 필요한 예:

- `GROQ_API_KEY`: Whisper 전사
- `GOOGLE_API_KEY_1` 또는 `GOOGLE_API_KEY`: Gemini(비디오/슬라이드 등)
- `GOOGLE_API_KEY_2`: 오디오 파이프라인·그래프(Stage 6) 등(미설정 시 1번 키로 대체 가능)

질의 서비스(`query_service`)는 `GOOGLE_API_KEY_2`·`GOOGLE_API_KEY`·`GEMINI_API_KEY` 등으로 Gemini를 찾습니다. LanceDB 경로는 `GRAPHLEC_LANCE_ROOT`(미설정 시 저장소 루트의 `data/lancedb`).

### LocalVLM / Ollama 설정 (선택)

슬라이드 추출 단계에서 자동 규칙만으로 애매한 후보를 LocalVLM에 넘겨 최종 판정할 수 있습니다. 현재 용도는 다음 세 가지입니다.

- `same_slide_duplicate`: 같은 슬라이드가 다시 등장했는지 확인
- `same_slide_build`: 같은 슬라이드에 내용이 빌드업된 것인지 확인
- `transition_noise`: 슬라이드 전환 중간 프레임이라 제거해야 하는지 확인

LocalVLM은 기본 비활성화되어 있습니다. 사용하려면 먼저 호스트 머신에 Ollama를 설치하고 모델을 받아야 합니다.

```bash
ollama serve
ollama pull gemma3:4b
ollama list
```

`ollama serve`는 파이프라인 실행 중 켜져 있어야 합니다. Docker Compose로 백엔드를 띄우는 경우 백엔드 컨테이너는 호스트의 Ollama 서버를 `http://host.docker.internal:11434`로 호출합니다.

#### CPU 로컬 환경 권장값

```env
GRAPHLEC_VLM_ENABLED=1
GRAPHLEC_VLM_PROVIDER=ollama
GRAPHLEC_OLLAMA_BASE_URL=http://host.docker.internal:11434
GRAPHLEC_OLLAMA_MODEL=gemma3:4b
GRAPHLEC_VLM_WORKERS=2
GRAPHLEC_VLM_TEMPERATURE=0.0
GRAPHLEC_VLM_APPLY=0
GRAPHLEC_VLM_APPLY_MIN_CONFIDENCE=0.65
```

- `GRAPHLEC_VLM_ENABLED=1`: LocalVLM 후보 판정을 실행합니다.
- `GRAPHLEC_VLM_APPLY=0`: 판정 결과만 `slides/llm_review_results.json`에 저장하고 metadata에는 반영하지 않습니다. 정확도를 확인한 뒤 `1`로 바꿉니다.
- `GRAPHLEC_VLM_APPLY_MIN_CONFIDENCE`: `APPLY=1`일 때 이 confidence 이상인 결정만 metadata에 반영합니다.
- `GRAPHLEC_VLM_WORKERS`: Ollama 동시 요청 수입니다. CPU 로컬은 `2` 정도가 무난합니다.

#### GPU 서버 환경 예시

GPU 서버에서는 더 큰 VLM을 쓸 수 있고 동시 요청 수도 늘릴 수 있습니다. 설치된 Ollama 모델 태그는 서버에서 `ollama list`로 확인하세요.

```env
GRAPHLEC_VLM_ENABLED=1
GRAPHLEC_VLM_PROVIDER=ollama
GRAPHLEC_OLLAMA_BASE_URL=http://host.docker.internal:11434
GRAPHLEC_OLLAMA_MODEL=qwen2.5vl:7b
GRAPHLEC_VLM_WORKERS=4
GRAPHLEC_VLM_TEMPERATURE=0.0
GRAPHLEC_VLM_APPLY=1
GRAPHLEC_VLM_APPLY_MIN_CONFIDENCE=0.65
```

GPU 서버에서도 Ollama는 백엔드 컨테이너 밖의 호스트 또는 별도 서버에서 실행할 수 있습니다. 별도 서버라면 `GRAPHLEC_OLLAMA_BASE_URL`을 해당 서버 주소로 바꿉니다.

#### 결과 확인

슬라이드 추출이 끝나면 아래 파일을 확인합니다.

- `slides/llm_review_candidates.json`: LocalVLM에 넘긴 후보 목록
- `slides/llm_review_results.json`: LocalVLM 판정 결과
- `slides/metadata.json`: `GRAPHLEC_VLM_APPLY=1`인 경우 transition 제거·중복 그룹 판정이 반영된 최종 metadata

Ollama가 꺼져 있거나 모델이 없으면 LocalVLM 단계에서 실패하므로, 먼저 `ollama list`와 `ollama serve` 상태를 확인합니다.

**Neo4j 적재(Stage 6 직후, 기본 비활성화)**  
`main.py`는 기본적으로 Stage 6 이후 Neo4j 적재를 건너뜁니다. Neo4j까지 올리고 싶다면 실행 시 **`--load-neo4j`** 를 지정하세요 (Parquet 생성은 그대로).  
`--load-neo4j`를 사용했는데 연결 실패하면 파이프라인은 **오류로 중단**됩니다.

#### Neo4j 설치·실행 (로컬, 둘 중 하나면 됨)

**방법 A — Neo4j Desktop (GUI, 처음 쓸 때 무난함)**  

1. [Neo4j Desktop 다운로드](https://neo4j.com/download/) 후 OS에 맞게 설치한다.  
2. 앱을 연 뒤 **New project** → **Add** → **Local DB** 로 로컬 데이터베이스를 만든다.  
3. DB를 선택하고 비밀번호를 정한다(또는 첫 실행 시 안내에 따라 설정).  
4. **Start** 로 DB를 켠다.  
5. **Open** 을 눌러 Neo4j Browser가 열리면 서버가 준비된 것이다.  
6. Bolt 연결 정보는 보통 다음과 같다(Desktop 하단·설정에서도 확인 가능).  
   - 주소: `bolt://localhost:7687` 또는 `bolt://127.0.0.1:7687`  
   - 사용자: `neo4j`  
   - 비밀번호: 3번에서 설정한 값  

**방법 B — Docker (명령으로만 띄울 때)**  

Docker가 설치되어 있다면 예시는 다음과 같다. `YOUR_PASSWORD` 를 본인 비밀번호로 바꾼다.

```bash
docker run -d --name graphlec-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/YOUR_PASSWORD \
  neo4j:5
```

- 웹 UI(선택): 브라우저에서 `http://127.0.0.1:7474`  
- Bolt: `bolt://127.0.0.1:7687`, 사용자 `neo4j`, 비밀번호는 `YOUR_PASSWORD` 와 동일  

컨테이너를 끄려면: `docker stop graphlec-neo4j` — 다시 켤 때: `docker start graphlec-neo4j`  

#### GraphLec `.env` 예시 (프로젝트 루트)

서버가 떠 있는 상태에서 아래를 맞춘다(비밀번호는 위에서 설정한 것과 동일).

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=여기에_비밀번호
```

자세한 설치·업그레이드·운영은 [Neo4j 공식 문서](https://neo4j.com/docs/)를 보면 된다.

**자주 나오는 문제**  

- `Connection refused`: Neo4j가 **Start**/컨테이너가 **실행 중**인지 확인한다. 방화벽이 **7687** 포트를 막지 않는지 본다.  
- 비밀번호를 잊었으면: Desktop은 DB 설정에서 재설정, Docker는 컨테이너·볼륨을 지우고 `NEO4J_AUTH` 로 다시 만드는 편이 단순하다.

---

## 4. 사용법

### 전체 실행

```bash
python main.py --input input/lecture.mp4
python main.py --input input/lecture.mp4 --output output --slides output_slides
```

### 주요 옵션

- `--output`, `--slides`: 산출물·슬라이드 디렉터리
- `--skip-extract`: 이미 슬라이드가 있을 때 Stage 1A 추출 생략
- `--force`: 기존 출력이 있어도 강제 재실행
- `--skip-graph-triples`: Stage 6(그래프 Parquet) 생략
- `--skip-neo4j`: Stage 6 직후 Neo4j 적재 생략 (기본값)
- `--load-neo4j`: Stage 6 직후 Neo4j 적재 활성화 (`NEO4J_*` 필요)
- `--skip-lance-index`: Stage 7(Lance 인덱스) 생략
- `--lance-root`: LanceDB 경로(기본: 환경변수 또는 `data/lancedb`)
- `--debug`, `--masks`: Stage 1 디버그 / Stage 3A 마스크 저장

---

## 5. 결과물 파일 안내 (`output/` 등)

| 구분 | 예시 | 설명 |
|------|------|------|
| 퓨전 | `{stem}_fused.json` | 슬라이드·세그먼트 통합 본문 |
| GraphRAG 입력 | `{stem}_graphrag.txt` | fused.json을 슬라이드 블록 단위 자연어 문서로 변환한 단일 txt |
| 그래프 | `{stem}_graph_triples.parquet`, `{stem}_nodes.parquet`, `{stem}_edges.parquet` | 트리플·정규화 노드/엣지 |
| 검색 | `{stem}_chunks_lance.parquet`, `data/lancedb/` | 청크 백업·LanceDB 테이블 `chunks`(stem 컬럼) |
| 기타 | `{stem}_segments.json`, `{stem}_by_slide.json`, … | 전사·분류·강조 등 (`config.output_paths` 참고) |
| 슬라이드 이미지 | `output_slides/` 등 | 추출 프레임·메타데이터 |

### Microsoft GraphRAG 입력 txt 생성

`fused.json`에서 제목, 슬라이드 원문, `contexts[].text`, 점수가 0보다 큰 `emphasized_keywords`, `annotations_summary`를 모아 강의 전체를 하나의 자연어 txt로 변환합니다. 슬라이드별 블록에는 `slide_id`, 시간 범위, 역할도 함께 남겨 GraphRAG로 만든 개념 그래프를 기존 구조 그래프와 다시 연결할 수 있게 합니다.

```bash
python -m app.backend.pipeline.fused_to_graphrag_text --stem os1-1 --output-dir output
python -m app.backend.pipeline.fused_to_graphrag_text --fused-path output/os1-1_fused.json
```

---

## 6. 요구 사항 (`requirements.txt` 요약)

opencv-python, numpy, Pillow, google-generativeai, google-genai, groq, torch, transformers, librosa, **neo4j**, lancedb, pyarrow, pandas, fastapi, uvicorn, django, httpx 등 (전체 목록은 파일 참고).

---

## 7. 웹·질의 서비스 (Django + FastAPI)

브라우저는 **Django만** 호출하고, Django가 **FastAPI** 질의 서비스로 프록시합니다.

### 구성

- **Django** (`web/`): 강의 메타(`Lecture`), 질의 페이지, `/api/query/`·`/api/graph/full/` 등
- **FastAPI** (`query_service/`): `POST /internal/query` — `stem` + `question`(질문 유형별 Neo4j·Lance·Gemini; 아래 **질의응답: 이전 버전과 현재 버전의 차이** 참고)

### 실행 (터미널 2개, 프로젝트 루트에서 가상환경 활성화 후)

**1) FastAPI 질의 서비스 (포트 8001)**

```bash
uvicorn query_service.main:app --host 127.0.0.1 --port 8001
```

**2) Django (포트 8000)**

```bash
cd web
python manage.py migrate
python manage.py runserver 8000
```

브라우저: `http://127.0.0.1:8000/`  
`main.py` 실행이 완료되면 해당 영상의 `stem`이 자동으로 `Lecture`에 등록됩니다.

### 환경 변수 (웹·질의 연동)

- `QUERY_SERVICE_URL` (기본 `http://127.0.0.1:8001`)
- `GRAPHLEC_LANCE_ROOT` — LanceDB 디렉터리 (미설정 시 프로젝트 루트 `data/lancedb`)
- `GRAPHLEC_OUTPUT_DIR` — 전체 그래프 Parquet 위치 (미설정 시 프로젝트 루트 `output/`)
- `GOOGLE_API_KEY` 등 — Gemini(질의 서비스·파이프라인과 공통 키 이름 사용 가능)

질의 서비스는 **프로젝트 루트**에서 실행하는 것을 권장합니다(`python main.py`로 만든 LanceDB 경로와 맞추기 쉬움).  
Django는 로컬 테스트 기준으로 SQLite(`web/db.sqlite3`)를 사용합니다.

### 질의응답: 이전 버전과 현재 버전의 차이

구현 기준은 **이전**: `query_service/archive/main_first_ver.py`(FastAPI 앱 버전 **0.2.0**), **현재**: `query_service/main.py`(**0.3.0**)입니다. 공통으로 Django는 `lecture_id`→`stem` 변환 후 `POST /internal/query`에 `stem`·`question`만 넘기고, 질문은 키워드 휴리스틱으로 **구조형(structural)** 과 **내용형(content)** 으로 나뉩니다. 차이는 주로 **내용형 파이프라인**과 **구조형 Cypher 생성 보조**, **응답에 실리는 보조 메타데이터**입니다.

---

#### 공통 동작(두 버전 모두)

- **구조형 질문**: 슬라이드 번호·시간·구간·장면 등 `STRUCTURAL_KEYWORDS`에 걸리면 LLM이 읽기 전용 Cypher를 생성하고 Neo4j에 `$stem` 필터로 조회합니다. 그래프 결과가 비면 답을 내지 않습니다.
- **Neo4j 내용 조회 골격**: 키워드마다 고정 Cypher 템플릿으로 서브개념·세그먼트·슬라이드(개념/본문)를 `CONTAINS` 매칭해 모읍니다. (현재는 `query_service/neo4j_content_queries.py`로 분리·한도 설정 가능.)
- **최종 답변**: 근거 문자열 + 사용자 질문을 Gemini에 넘기며, 시스템 프롬프트로 환각을 줄입니다.

---

#### 1) 내용형(content) 질문 — 이전 버전(0.2.x)

| 항목 | 동작 |
|------|------|
| 키워드 | 질문에서 따옴표·토큰 분리 등으로만 추출 (`extract_keywords_from_question`). |
| Neo4j | 키워드로 위 고정 템플릿을 돌려 슬라이드·세그먼트·개념 관계를 모음. 행 수 상한은 쿼리에 박힌 고정 `LIMIT`(예: 세그먼트 15, 슬라이드 5 등). |
| 근거 문자열 | `_build_content_context`: 질문 복사 + `[지식 그래프 조회 결과 — 내용 질문]` 아래에 구성 개념·음성 구간·슬라이드 본문을 **고정 섹션 순서**로 나열. 키워드 출현 빈도로 슬라이드·세그먼트 정렬. |
| Lance(벡터) | 그래프에서 나온 노드 `id` 집합(`allowed_ids`)이 있을 때만, 질문 문장으로 `lance_search` **1회**, 상위 `TOP_K` 후 **`linked_node_id`가 `allowed_ids`에 있는 청크만 최대 4개** 선택. |
| LLM 컨텍스트 | 그래프 텍스트 뒤에 `[Lance 보강 텍스트]` 블록을 **붙인 하이브리드** (`_build_hybrid_context_base`). Lance는 그래프와 **노드 id로만** 연결된 경우에 한해 보강. |
| 답변 프롬프트 | 근거에 `[Lance 보강 텍스트]` 규칙만 명시. 복합 질문(정의+예시 동시)에 대한 별도 지시는 없음. |

**요약**: 내용형은 “키워드 → 고정 그래프 조회 → 단순 정렬된 긴 컨텍스트 → (선택) id 일치 Lance 소량 보강” 구조입니다. 의도 분류·임베딩 재순위는 없습니다.

---

#### 2) 내용형(content) 질문 — 현재 버전(0.3.x)

| 항목 | 동작 |
|------|------|
| 의도 + 키워드 | Gemini로 **JSON 의도 분류**(`infer_intents_json`: definition / example / explanation / comparison / temporal / location / general 및 가중치)와 **`keywords_hint`** 를 받아, 추출 키워드 목록에 **추가 검색어**로 합칩니다. |
| Neo4j | 동일한 템플릿 계열이나 `neo4j_content_queries.py`로 이전했고, `intent_config.json`의 `neo4j_limits`·환경변수 `GRAPHLEC_CONTENT_*` 로 키워드당 행 수 상한을 조정할 수 있습니다. |
| 그래프+Lance 후보 | 그래프 행을 `EvidenceItem`으로 올린 뒤, Lance는 **2-pass**: 1차 넓게 검색해 `linked_node_id ∈ allowed_ids` 인 **strict** 를 우선 채우고, 부족하면 2차로 더 가져와 **soft**(id 불일치 보조)를 섞습니다. |
| 재순위 | 후보 전체에 대해 **질문·근거 임베딩 코사인 유사도**, **의도 가중치·출처 종류별 사전(intent_source_prior)**, **키워드 겹침**을 합산한 점수로 정렬한 뒤 **MMR**로 중복이 큰 근거를 줄여 `mmr_pick_k`개만 선택합니다. |
| 근거 문자열 | `build_sectioned_context`: 상단에 **`[질문 의도(모델 추론)]`**, 아래에 출처 종류별 태그(개념 관계, 슬라이드 본문, 음성 구간, 의미 검색 strict/soft 등). Lance soft에는 **“(그래프 id 미일치 보조)”** 메타가 붙을 수 있습니다. |
| LLM 컨텍스트 | 내용형은 하이브리드 블록 대신 **위 한 덩어리 sectioned context만** Gemini에 넘깁니다(Lance가 이미 재순위·섹션에 녹아 있음). |
| 답변 프롬프트 | 의도 블록은 참고용, 실제 사실은 아래 근거에만 의존하라는 규칙·**복합 질문** 처리·의미 검색 보조 사용 조건이 **ANSWER_SYSTEM_PROMPT**에 추가됨. |
| API `retrieved_chunks` | 내용형 응답에서 청크 목록은 **파이프라인에 포함된 Lance 계열 근거**만 옮깁니다(`_evidence_to_retrieved_chunks`). 그래프-only 근거는 청크 필드에 안 실릴 수 있습니다. |

**요약**: 현재 버전은 “의도 추론 → 더 많은 후보(그래프+Lance 2-pass) → 임베딩+의도+키워드+MMR으로 압축 → 한 근거 문자열”로, **검색 품질과 다양성**을 올리는 쪽으로 바뀌었습니다.

---

#### 3) 구조형(structural) 질문 — 차이

| 항목 | 이전(0.2.x) | 현재(0.3.x) |
|------|-------------|-------------|
| Cypher 생성 입력 | 질문·스키마만으로 `generate_cypher` 호출. | 답 생성 전에 같은 **의도 JSON**을 한 번 돌려, 가중치가 일정 이상인 의도만 요약한 **`intent_hint` 문자열**을 Cypher 프롬프트에 **추가**합니다. |
| 목적 | 슬라이드/시간 등 구조 질의에 맞는 쿼리만 생성. | “몇 번 슬라이드”류에서도 질문 속 **의도(예: 예시 위주)** 를 힌트로 넣어 생성 품질을 보완. |
| Lance | 구조형은 조회 후 `allowed_ids`로 Lance 1회 검색·id 필터 **최대 4개** 보강 후, 그래프+Lance 하이브리드로 답변. | 동일하게 **구조형 경로만** `_build_hybrid_context_base`를 탑니다(내용형과 대칭). |

---

#### 4) 코드·설정 측면

- **모듈 분리**: 현재는 `content_retrieval.py`(의도·재순위·MMR·컨텍스트), `neo4j_content_queries.py`(고정 Cypher), `graph_constants.py`(개념 간 관계 타입)로 나뉘어 단일 파일보다 역할이 분리되어 있습니다.
- **튜닝**: `query_service/intent_config.json`에서 Lance 상한·MMR·의도-출처 가중 등을 조정할 수 있습니다(이전은 코드 상수 위주).

---

## 8. 데이터베이스

로컬 테스트는 SQLite 단일 DB(`web/db.sqlite3`)로 동작합니다.
