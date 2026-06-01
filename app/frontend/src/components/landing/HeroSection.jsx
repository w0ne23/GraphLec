import { useEffect, useRef, useState } from 'react'

export default function HeroSection({ current, onNext }) {
  const canvasRef = useRef(null)
  const currentRef = useRef(current)
  const [activeCard, setActiveCard] = useState(1)
  const [cardPaused, setCardPaused] = useState(false)

  useEffect(() => {
    currentRef.current = current
  }, [current])

  useEffect(() => {
    if (current !== 0 || cardPaused) return undefined

    const timerId = window.setInterval(() => {
      setActiveCard(card => (card === 3 ? 1 : card + 1))
    }, 3000)

    return () => window.clearInterval(timerId)
  }, [cardPaused, current])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const P = { r: 102, g: 103, b: 171 }
    const N = 66
    const getBounds = () => ({
      top: 64,
      bottom: Math.max(120, window.innerHeight - 58),
      width: window.innerWidth,
      height: Math.max(120, window.innerHeight - 122),
    })
    const createNodes = () => {
      const bounds = getBounds()
      const cols = Math.ceil(Math.sqrt(N * (bounds.width / bounds.height)))
      const rows = Math.ceil(N / cols)

      return Array.from({ length: N }, (_, i) => {
        const isHub = i < 10
        const col = i % cols
        const row = Math.floor(i / cols)
        return {
          x: ((col + Math.random() * 0.8 + 0.1) / cols) * bounds.width,
          y: bounds.top + ((row + Math.random() * 0.8 + 0.1) / rows) * bounds.height,
          vx: (Math.random() - 0.5) * (isHub ? 0.25 : 0.4),
          vy: (Math.random() - 0.5) * (isHub ? 0.25 : 0.4),
          r: isHub ? Math.random() * 2.4 + 5.2 : Math.random() * 1.8 + 2,
          op: isHub ? 0.72 : Math.random() * 0.32 + 0.24,
          ph: Math.random() * Math.PI * 2,
          isHub,
        }
      })
    }

    let bounds = getBounds()
    let nodes = createNodes()

    const resize = () => {
      const prev = bounds
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      bounds = getBounds()

      nodes = nodes.map(node => ({
        ...node,
        x: Math.min(bounds.width, Math.max(0, node.x * (bounds.width / prev.width))),
        y: Math.min(
          bounds.bottom,
          Math.max(bounds.top, bounds.top + ((node.y - prev.top) / prev.height) * bounds.height)
        ),
      }))
    }
    resize()
    window.addEventListener('resize', resize)

    let rafId

    const draw = () => {
      rafId = requestAnimationFrame(draw)
      if (currentRef.current !== 0) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      nodes.forEach(n => {
        n.ph += 0.012
        n.x += n.vx
        n.y += n.vy
        if (n.x < 0 || n.x > bounds.width) n.vx *= -1
        if (n.y < bounds.top || n.y > bounds.bottom) n.vy *= -1
      })

      const MAX_DIST = 200
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const a = nodes[i], b = nodes[j]
          const d = Math.hypot(a.x - b.x, a.y - b.y)
          if (d > MAX_DIST) continue
          const alpha = (1 - d / MAX_DIST) * 0.24
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(b.x, b.y)
          ctx.strokeStyle = `rgba(${P.r},${P.g},${P.b},${alpha})`
          ctx.lineWidth = a.isHub || b.isHub ? 1 : 0.58
          ctx.stroke()
        }
      }

      nodes.forEach(n => {
        const pr = n.r * (1 + 0.2 * Math.sin(n.ph))
        const al = n.op * (0.8 + 0.2 * Math.sin(n.ph))
        if (n.isHub) {
          ctx.shadowColor = `rgba(${P.r},${P.g},${P.b},${al * 0.32})`
          ctx.shadowBlur = pr * 3.2
          ctx.beginPath()
          ctx.arc(n.x, n.y, pr * 4.8, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(${P.r},${P.g},${P.b},${al * 0.055})`
          ctx.fill()
          ctx.shadowBlur = pr * 1.8
          ctx.beginPath()
          ctx.arc(n.x, n.y, pr * 2.4, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(${P.r},${P.g},${P.b},${al * 0.11})`
          ctx.fill()
        }
        ctx.shadowColor = `rgba(${P.r},${P.g},${P.b},${al * 0.28})`
        ctx.shadowBlur = n.isHub ? pr * 1.2 : pr * 0.7
        ctx.beginPath()
        ctx.arc(n.x, n.y, pr, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${P.r},${P.g},${P.b},${al})`
        ctx.fill()
        ctx.shadowBlur = 0
      })
    }

    draw()

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <>
      <canvas ref={canvasRef} id="bg-canvas" />
      <div className="hero-content">
        <div
          className={`hero-card-stage hero-card-stage-paused hero-card-stage-show-${activeCard}`}
          onMouseEnter={() => setCardPaused(true)}
          onMouseLeave={() => setCardPaused(false)}
        >
          <div className="hero-orbit-card hero-orbit-card-1">
            <div className="hero-card-title-row">
              <span className="hero-card-num">1</span>
              <div className="hero-card-title">Verification</div>
            </div>
            <div className="hero-card-sub">
              강의자가 영상을 전달하면 Verifier가 Multi-LLM으로 판단하고 피드백을 돌려줍니다.
            </div>

            <div className="hero-verifier-diagram">
              <div className="hero-flow-node hero-flow-node-lecturer">
                <div className="hero-flow-person">
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4.5 20c1.4-4.1 4-6 7.5-6s6.1 1.9 7.5 6" />
                  </svg>
                  <strong>강의자</strong>
                </div>
                <em>강의 영상 업로드</em>
              </div>

              <div className="hero-flow-connector">
                <div className="hero-flow-arrow hero-flow-arrow-upload">
                  <span>강의 영상</span>
                </div>
                <div className="hero-flow-arrow hero-flow-arrow-feedback">
                  <span>피드백</span>
                </div>
              </div>

              <div className="hero-flow-node hero-flow-node-verifier">
                <strong>Verifier</strong>
                <em>Multi-LLM 판단 후 피드백 제공</em>
              </div>

              <div className="hero-feedback-panel">
                <div className="hero-feedback-head">
                  <span className="hero-feedback-dot" />
                  피드백 리포트
                </div>
                <div className="hero-feedback-tags">
                  <span className="hero-feedback-tag-fact">사실 오류</span>
                  <span className="hero-feedback-tag-era">오래된 내용</span>
                  <span className="hero-feedback-tag-confusion">혼동 가능 설명</span>
                  <span className="hero-feedback-tag-scope">과도한 일반화</span>
                  <span className="hero-feedback-tag-slide">슬라이드 오타</span>
                </div>
                <ul className="hero-feedback-issues">
                  <li>
                    <span className="hero-issue-pill hero-issue-fact">사실 오류</span>
                    <strong>
                      페이지 테이블은 <del>CPU</del> 안에 저장됩니다.
                    </strong>
                    <em>→ 메인 메모리</em>
                  </li>
                  <li>
                    <span className="hero-issue-pill hero-issue-era">오래된 내용</span>
                    <strong>
                      현대 OS는 <del>세그멘테이션만</del> 사용합니다.
                    </strong>
                    <em>→ 페이징 기반 가상 메모리</em>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <div className="hero-orbit-card hero-orbit-card-2">
            <div className="hero-card-title-row">
              <span className="hero-card-num">2</span>
              <div className="hero-card-title">Recommendation</div>
            </div>
            <div className="hero-card-sub">
              학습자가 원하는 강의를 검색하면, 분석된 강의 메타데이터와 일치도를 계산해 강의를 추천합니다.
            </div>

            <div className="hero-recommend-diagram">
              <div className="hero-match-strip">
                <span>영상 분석 그래프</span>
                <i><b>추출</b></i>
                <span>메타데이터</span>
                <i className="hero-match-both"><em aria-hidden="true" /><b>매칭</b></i>
                <span className="hero-query-node">질의 분석</span>
              </div>

              <div className="hero-recommend-panel">
                <div className="hero-learner-row">
                  <div className="hero-learner-profile">
                    <div className="hero-learner-icon">
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <circle cx="12" cy="8" r="4" />
                        <path d="M4.5 20c1.4-4.1 4-6 7.5-6s6.1 1.9 7.5 6" />
                      </svg>
                    </div>
                    <strong>학습자</strong>
                  </div>
                  <div className="hero-learner-query">
                    <span>프로세스와 CPU 관계를 설명하는 강의 추천해줘.</span>
                  </div>
                </div>

                <div className="hero-score-list">
                  <div className="hero-score-item hero-score-item-top">
                    <div className="hero-score-copy">
                      <strong>프로세스 관리와 상태 전이</strong>
                      <span className="hero-score-tags">#프로세스 #CPU</span>
                    </div>
                    <div className="hero-score-points">
                      <span className="hero-score-pill hero-score-pill-content">내용 32</span>
                      <span className="hero-score-pill hero-score-pill-meaning">의미 28</span>
                      <span className="hero-score-pill hero-score-pill-condition">조건 15</span>
                      <em>75점</em>
                    </div>
                  </div>
                  <div className="hero-score-item">
                    <div className="hero-score-copy">
                      <strong>운영체제 개론</strong>
                      <span className="hero-score-tags">#운영체제 #자원관리</span>
                    </div>
                    <div className="hero-score-points">
                      <span className="hero-score-pill hero-score-pill-content">내용 27</span>
                      <span className="hero-score-pill hero-score-pill-meaning">의미 24</span>
                      <span className="hero-score-pill hero-score-pill-condition">조건 13</span>
                      <em>64점</em>
                    </div>
                  </div>
                  <div className="hero-score-item">
                    <div className="hero-score-copy">
                      <strong>CPU 스케줄링 기초</strong>
                      <span className="hero-score-tags">#스케줄링 #성능</span>
                    </div>
                    <div className="hero-score-points">
                      <span className="hero-score-pill hero-score-pill-content">내용 23</span>
                      <span className="hero-score-pill hero-score-pill-meaning">의미 25</span>
                      <span className="hero-score-pill hero-score-pill-condition">조건 10</span>
                      <em>58점</em>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="hero-orbit-card hero-orbit-card-3">
            <div className="hero-card-title-row">
              <span className="hero-card-num">3</span>
              <div className="hero-card-title">QnA</div>
            </div>
            <div className="hero-card-sub">
              학습자가 강의 시청 중 궁금한 점을 질문하면, 답변과 함께 출처 구간과 근거 그래프를 제공합니다.
            </div>

            <div className="hero-qna-panel">
              <div className="hero-learner-row hero-qna-question-row">
                <div className="hero-learner-profile">
                  <div className="hero-learner-icon">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <circle cx="12" cy="8" r="4" />
                      <path d="M4.5 20c1.4-4.1 4-6 7.5-6s6.1 1.9 7.5 6" />
                    </svg>
                  </div>
                  <strong>학습자</strong>
                </div>
                <div className="hero-learner-query">
                  <span>문맥 전환은 왜 필요한가요?</span>
                </div>
              </div>

              <div className="hero-answer-bubble">
                <div className="hero-qna-answer-grid">
                  <div className="hero-answer-copy">
                    <p>
                      문맥 전환은 CPU가 실행 중인 프로세스의 상태를 저장하고,
                      다음 프로세스의 상태를 복원해 여러 작업을 번갈아 실행하기 위해 필요합니다.
                    </p>

                    <div className="hero-answer-divider" />

                    <div className="hero-source-row">
                      <span>강의 출처</span>
                      <strong>Slide 12 · 14:20</strong>
                    </div>
                  </div>

                  <div className="hero-mini-graph">
                    <svg className="hero-graph-edges" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                      <line x1="50" y1="50" x2="22" y2="24" />
                      <line x1="50" y1="50" x2="78" y2="24" />
                      <line x1="50" y1="50" x2="62" y2="78" />
                      <line x1="50" y1="50" x2="22" y2="76" />
                    </svg>
                    <span className="hero-graph-node hero-graph-node-main">문맥 전환</span>
                    <span className="hero-graph-node hero-graph-node-a">CPU 상태</span>
                    <span className="hero-graph-node hero-graph-node-b">Slide 12</span>
                    <span className="hero-graph-node hero-graph-node-c">스케줄링</span>
                    <span className="hero-graph-node hero-graph-node-d">프로세스</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <button
            type="button"
            className="hero-card-nav hero-card-nav-prev"
            aria-label="이전 카드"
            onClick={() => setActiveCard(card => (card === 1 ? 3 : card - 1))}
          >
            ‹
          </button>
          <button
            type="button"
            className="hero-card-nav hero-card-nav-next"
            aria-label="다음 카드"
            onClick={() => setActiveCard(card => (card === 3 ? 1 : card + 1))}
          >
            ›
          </button>
          <div className="hero-card-dots" aria-label="카드 순서">
            {[1, 2, 3].map(card => (
              <button
                key={card}
                type="button"
                className={card === activeCard ? 'active' : ''}
                aria-label={`${card}번 카드 보기`}
                aria-current={card === activeCard ? 'true' : undefined}
                onClick={() => setActiveCard(card)}
              />
            ))}
          </div>
        </div>
        <div className="hero-copy">
          <h1 className="hero-title">
            <span className="hero-title-top">강의 영상을</span>
            <span className="hero-title-linked">
              <span className="peri">지식으로</span>
              <span className="dim">연결하다</span>
            </span>
          </h1>
          <p className="hero-desc">
            <strong>GraphLec</strong>은 슬라이드, 음성, 필기를 동시에 분석해<br />
            강의 콘텐츠를 구조화하고 학습을 연결합니다.
          </p>
          <button type="button" className="hero-cta" onClick={onNext}>
            시작하기
          </button>
        </div>
      </div>
    </>
  )
}
