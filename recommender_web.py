"""
recommender_web.py
──────────────────
FastAPI 기반 강의 추천 웹 서버

실행:
  uvicorn recommender_web:app --port 8002 --reload

엔드포인트:
  POST /recommend   { "query": "스레드 자세히 설명하는 강의", "top_k": 3 }
  GET  /health
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from recommender import Recommender, RecommenderConfig


# ============================================================================
#  앱 초기화
# ============================================================================

METADATA_DIR = os.getenv("METADATA_DIR", "metadata/")
_recommender: Optional[Recommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender
    print(f"[시작] 메타데이터 로드: {METADATA_DIR}")
    _recommender = Recommender(metadata_dir=METADATA_DIR, config=RecommenderConfig())
    yield
    print("[종료]")


app = FastAPI(
    title="GraphLEC Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
#  요청 / 응답 스키마
# ============================================================================

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 3


class ScoreDetail(BaseModel):
    content_pct:       float
    vec_score:         float
    dm_score:          float
    sim_title:         float
    sim_keyword:       float
    sim_summary:       float
    dm_keyword:        float
    domain_score:      float
    difficulty_match:  float
    depth_score:       float
    combined_boost:    float
    duration_score:    float
    duration_mismatch: bool
    frag_penalty:      float


class LectureResult(BaseModel):
    video_id:          str
    title:             str
    domain:            str
    instructor:        str
    score:             float
    duration_sec:      float
    reason:            str
    summary:           str
    score_detail:      ScoreDetail


class RecommendResponse(BaseModel):
    query:   str
    results: list[LectureResult]


# ============================================================================
#  엔드포인트
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_HTML)


@app.get("/health")
def health():
    loaded = _recommender is not None
    count  = len(_recommender.collection.lectures) if loaded else 0
    return {"status": "ok", "lectures_loaded": count}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not _recommender:
        raise HTTPException(status_code=503, detail="추천 엔진 초기화 중")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다")

    results = _recommender.recommend_from_query(req.query, top_k=req.top_k)
    top     = [r for r in results if r.score > 0][: req.top_k]

    return RecommendResponse(
        query=req.query,
        results=[
            LectureResult(
                video_id     = r.video_id,
                title        = r.title,
                domain       = r.domain,
                instructor   = r.instructor,
                score        = r.score,
                duration_sec = r.score_detail.get("duration_sec", 0.0),
                reason       = r.reason,
                summary      = r.summary,
                score_detail = ScoreDetail(
                    content_pct       = r.score_detail.get("content_pct",       0.0),
                    vec_score         = r.score_detail.get("vec_score",          0.0),
                    dm_score          = r.score_detail.get("dm_score",           0.0),
                    sim_title         = r.score_detail.get("sim_title",          0.0),
                    sim_keyword       = r.score_detail.get("sim_keyword",        0.0),
                    sim_summary       = r.score_detail.get("sim_summary",        0.0),
                    dm_keyword        = r.score_detail.get("dm_keyword",         0.0),
                    domain_score      = r.score_detail.get("domain_score",       0.0),
                    difficulty_match  = r.score_detail.get("difficulty_match",   0.0),
                    depth_score       = r.score_detail.get("depth_score",        0.0),
                    combined_boost    = r.score_detail.get("combined_boost",     1.0),
                    duration_score    = r.score_detail.get("duration_score",     1.0),
                    duration_mismatch = r.score_detail.get("duration_mismatch",  False),
                    frag_penalty      = r.score_detail.get("frag_penalty",       0.0),
                ),
            )
            for r in top
        ],
    )


# ============================================================================
#  임베디드 UI
# ============================================================================

_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphLEC 강의 추천</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d0f14;
    --surface: #151820;
    --surface2: #1c2030;
    --border: #252a3a;
    --accent: #4f8cff;
    --accent2: #7b5ea7;
    --green: #3dd68c;
    --yellow: #f0c040;
    --red: #ff5f57;
    --text: #e2e8f0;
    --text-muted: #6b7a99;
    --text-dim: #3a4258;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Noto Sans KR', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* 배경 그리드 */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.3;
    pointer-events: none;
    z-index: 0;
  }

  .wrap {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 780px;
    padding: 60px 24px 100px;
  }

  /* 헤더 */
  header {
    margin-bottom: 48px;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }

  .logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }

  .logo-text {
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.5px;
  }

  .logo-text span { color: var(--accent); }

  .subtitle {
    font-size: 13px;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.3px;
  }

  /* 검색 영역 */
  .search-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
  }

  .search-box:focus-within {
    border-color: var(--accent);
  }

  .search-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  textarea {
    width: 100%;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    font-weight: 400;
    resize: none;
    line-height: 1.6;
    min-height: 60px;
  }

  textarea::placeholder { color: var(--text-dim); }

  .search-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
  }

  .topk-control {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-muted);
  }

  .topk-control input[type=range] {
    -webkit-appearance: none;
    width: 80px; height: 3px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }

  .topk-control input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    background: var(--accent);
    border-radius: 50%;
    cursor: pointer;
  }

  .topk-val {
    color: var(--accent);
    min-width: 16px;
    text-align: center;
  }

  .btn-search {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 9px 22px;
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .btn-search:hover { opacity: 0.88; }
  .btn-search:active { transform: scale(0.97); }
  .btn-search:disabled { opacity: 0.4; cursor: not-allowed; }

  /* 예시 질의 */
  .examples {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 36px;
  }

  .ex-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .ex-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  /* 분석 결과 뱃지 */
  .analysis-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 24px;
    animation: fadeIn 0.3s ease;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
  }

  .badge .key { color: var(--text-dim); margin-right: 2px; }
  .badge .val { color: var(--text); }
  .badge.type .val { color: var(--accent); }
  .badge.focus .val { color: var(--yellow); }
  .badge.domain .val { color: var(--green); }

  .badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255,95,87,0.08);
    border: 1px solid rgba(255,95,87,0.25);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--red);
    margin-top: 5px;
  }

  /* 결과 카드 */
  .results { display: flex; flex-direction: column; gap: 14px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    animation: slideUp 0.3s ease both;
    transition: border-color 0.2s;
  }

  .card:hover { border-color: #2e3550; }

  .card:nth-child(1) { animation-delay: 0.05s; }
  .card:nth-child(2) { animation-delay: 0.10s; }
  .card:nth-child(3) { animation-delay: 0.15s; }

  .card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }

  .card-rank {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 3px;
    min-width: 20px;
  }

  .card-title-wrap { flex: 1; }

  .card-title {
    font-size: 15px;
    font-weight: 700;
    line-height: 1.4;
    margin-bottom: 4px;
  }

  .card-meta {
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .card-meta .dot { color: var(--text-dim); }

  .card-score {
    text-align: right;
    flex-shrink: 0;
  }

  .score-num {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
  }

  .score-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 2px;
  }

  .card-reason {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .card-summary {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.6;
    border-left: 2px solid var(--border);
    padding-left: 12px;
    margin-bottom: 14px;
  }

  /* 스코어 바 */
  .score-bars {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .bar-item { }

  .bar-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
  }

  .bar-label span:last-child { color: var(--text-muted); }

  .bar-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--accent);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
  }

  .bar-fill.green { background: var(--green); }
  .bar-fill.yellow { background: var(--yellow); }
  .bar-fill.purple { background: var(--accent2); }

  .depth-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(240,192,64,0.1);
    border: 1px solid rgba(240,192,64,0.25);
    border-radius: 4px;
    padding: 2px 7px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--yellow);
    margin-left: 6px;
  }

  /* 빈 상태 */
  .empty {
    text-align: center;
    padding: 60px 0;
    color: var(--text-dim);
    font-size: 14px;
  }

  /* 에러 */
  .error-box {
    background: rgba(255,95,87,0.08);
    border: 1px solid rgba(255,95,87,0.2);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 13px;
    color: #ff8a85;
    animation: fadeIn 0.2s ease;
  }

  /* 로딩 */
  .loading {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 40px 0;
    color: var(--text-muted);
    font-size: 13px;
    font-family: var(--mono);
  }

  .spinner {
    width: 16px; height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* 구분선 */
  .section-title {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo">
      <div class="logo-icon">◈</div>
      <div class="logo-text">Graph<span>LEC</span></div>
    </div>
    <div class="subtitle">지식 그래프 기반 강의 추천 시스템</div>
  </header>

  <div class="search-box">
    <div class="search-label">질의 입력</div>
    <textarea id="queryInput" rows="2"
      placeholder="예: 스레드를 자세히 설명해주는 강의 추천해줘"></textarea>
    <div class="search-footer">
      <div class="topk-control">
        <span>결과 수</span>
        <input type="range" id="topkSlider" min="1" max="10" value="3">
        <span class="topk-val" id="topkVal">3</span>
      </div>
      <button class="btn-search" id="searchBtn" onclick="doSearch()">
        <span>추천받기</span>
        <span>→</span>
      </button>
    </div>
  </div>

  <div class="examples" id="examples">
    <span class="ex-chip" onclick="setQuery(this)">스레드 자세히 설명하는 강의</span>
    <span class="ex-chip" onclick="setQuery(this)">딥러닝 원리 강의 찾아줘</span>
    <span class="ex-chip" onclick="setQuery(this)">파이썬 기초 다 들었는데 다음은?</span>
    <span class="ex-chip" onclick="setQuery(this)">백엔드 개발자 취업 준비 커리큘럼</span>
    <span class="ex-chip" onclick="setQuery(this)">비동기 처리할 때 자꾸 막혀</span>
    <span class="ex-chip" onclick="setQuery(this)">손익계산서 읽는 법 강의</span>
  </div>

  <div id="output"></div>

</div>

<script>
  const API = '';

  const slider = document.getElementById('topkSlider');
  const topkVal = document.getElementById('topkVal');
  slider.addEventListener('input', () => { topkVal.textContent = slider.value; });

  const input = document.getElementById('queryInput');
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSearch(); }
  });

  function setQuery(el) {
    input.value = el.textContent;
    input.focus();
  }

  function pct(v) { return Math.round((v || 0) * 100); }

  function queryTypLabel(t) {
    return { direct: 'Direct', followup: 'Follow-up', career: 'Career', unknown: 'Unknown' }[t] || t;
  }

  async function doSearch() {
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('searchBtn');
    btn.disabled = true;

    const out = document.getElementById('output');
    out.innerHTML = `<div class="loading"><div class="spinner"></div><span>분석 중...</span></div>`;

    try {
      const res = await fetch(`${API}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: parseInt(slider.value) }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `서버 오류 (${res.status})`);
      }

      const data = await res.json();
      renderResults(data);
    } catch (e) {
      out.innerHTML = `<div class="error-box">⚠ ${e.message}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderResults(data) {
    const out = document.getElementById('output');
    if (!data.results || data.results.length === 0) {
      out.innerHTML = `<div class="empty">일치하는 강의를 찾지 못했습니다.</div>`;
      return;
    }

    const d = data.results[0].score_detail;

    const analysisBar = `
      <div class="analysis-bar">
        ${d.domain_score > 0 ? `<div class="badge domain"><span class="key">도메인</span><span class="val">일치</span></div>` : ''}
        ${d.difficulty_match > 0 ? `<div class="badge"><span class="key">난이도</span><span class="val">일치</span></div>` : ''}
        ${d.depth_score > 0 ? `<div class="badge focus"><span class="key">깊이</span><span class="val">${pct(d.depth_score)}%</span></div>` : ''}
        ${d.combined_boost > 1.0 ? `<div class="badge"><span class="key">boost</span><span class="val">+${Math.round((d.combined_boost-1)*100)}%</span></div>` : ''}
      </div>`;

    const cards = data.results.map((r, i) => {
      const sd = r.score_detail;
      const depthTag = sd.depth_score > 0.1
        ? `<span class="depth-tag">◈ 깊이 +${pct(sd.depth_score)}%</span>` : '';

      const durationMin = r.duration_sec ? Math.round(r.duration_sec / 60) : null;
      const durationBadge = sd.duration_mismatch
        ? `<span class="badge-warn">⚠ 요청 시간 초과 (${durationMin}분)</span>` : '';

      const durationInfo = durationMin
        ? `<span>· ${durationMin}분</span>` : '';

      return `
      <div class="card">
        <div class="card-header">
          <div class="card-rank">#${i + 1}</div>
          <div class="card-title-wrap">
            <div class="card-title">${r.title}${depthTag}</div>
            <div class="card-meta">
              <span>${r.domain}</span>
              <span class="dot">·</span>
              <span>${r.instructor || '강사 미상'}</span>
              <span class="dot">·</span>
              <span>${r.video_id}</span>
              ${durationInfo}
            </div>
            ${durationBadge}
          </div>
          <div class="card-score">
            <div class="score-num">${pct(r.score)}</div>
            <div class="score-label">/ 100</div>
          </div>
        </div>

        <div class="card-reason">💡 ${r.reason}</div>

        <div class="card-summary">${r.summary ? r.summary.slice(0, 120) + (r.summary.length > 120 ? '…' : '') : '요약 없음'}</div>

        <div class="score-bars">
          <div class="bar-item">
            <div class="bar-label"><span>내용 관련도</span><span>${sd.content_pct}%</span></div>
            <div class="bar-track"><div class="bar-fill" style="width:${sd.content_pct}%"></div></div>
          </div>
          <div class="bar-item">
            <div class="bar-label"><span>벡터 유사도</span><span>${Math.round(sd.vec_score*100)}%</span></div>
            <div class="bar-track"><div class="bar-fill green" style="width:${Math.round(sd.vec_score*100)}%"></div></div>
          </div>
          <div class="bar-item">
            <div class="bar-label"><span>직접 매칭</span><span>${Math.round(sd.dm_keyword*100)}%</span></div>
            <div class="bar-track"><div class="bar-fill yellow" style="width:${Math.round(sd.dm_keyword*100)}%"></div></div>
          </div>
        </div>
      </div>`;
    }).join('');

    out.innerHTML = `
      <div class="section-title">추천 결과 ${data.results.length}건</div>
      ${analysisBar}
      <div class="results">${cards}</div>`;
  }
</script>
</body>
</html>

"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recommender_web:app", host="0.0.0.0", port=8002, reload=True)