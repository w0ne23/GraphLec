import { useEffect, useRef } from 'react'

export default function HeroSection({ current, onNext }) {
  const canvasRef = useRef(null)
  const currentRef = useRef(current)

  useEffect(() => {
    currentRef.current = current
  }, [current])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const P = { r: 102, g: 103, b: 171 }
    const N = 50
    const nodes = Array.from({ length: N }, (_, i) => {
      const isHub = i < 8
      return {
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * (isHub ? 0.25 : 0.4),
        vy: (Math.random() - 0.5) * (isHub ? 0.25 : 0.4),
        r: isHub ? Math.random() * 2 + 4 : Math.random() * 1.5 + 1.5,
        op: isHub ? 0.75 : Math.random() * 0.35 + 0.25,
        ph: Math.random() * Math.PI * 2,
        isHub,
      }
    })

    let rafId

    const draw = () => {
      rafId = requestAnimationFrame(draw)
      if (currentRef.current !== 0) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      nodes.forEach(n => {
        n.ph += 0.012
        n.x += n.vx
        n.y += n.vy
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1
        if (n.y < 0 || n.y > canvas.height)  n.vy *= -1
      })

      const MAX_DIST = 180
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const a = nodes[i], b = nodes[j]
          const d = Math.hypot(a.x - b.x, a.y - b.y)
          if (d > MAX_DIST) continue
          const alpha = (1 - d / MAX_DIST) * 0.28
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(b.x, b.y)
          ctx.strokeStyle = `rgba(${P.r},${P.g},${P.b},${alpha})`
          ctx.lineWidth = a.isHub || b.isHub ? 0.8 : 0.5
          ctx.stroke()
        }
      }

      nodes.forEach(n => {
        const pr = n.r * (1 + 0.2 * Math.sin(n.ph))
        const al = n.op * (0.8 + 0.2 * Math.sin(n.ph))
        if (n.isHub) {
          ctx.beginPath()
          ctx.arc(n.x, n.y, pr * 4, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(${P.r},${P.g},${P.b},${al * 0.06})`
          ctx.fill()
          ctx.beginPath()
          ctx.arc(n.x, n.y, pr * 2, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(${P.r},${P.g},${P.b},${al * 0.12})`
          ctx.fill()
        }
        ctx.beginPath()
        ctx.arc(n.x, n.y, pr, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${P.r},${P.g},${P.b},${al})`
        ctx.fill()
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
        <div className="hero-card-stage">
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
                  <span>사실 오류</span>
                  <span>시대적 오류</span>
                  <span>혼동 오류</span>
                  <span>범위 오류</span>
                  <span>슬라이드 오류</span>
                </div>
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
                <span>질의 분석</span>
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
        </div>
        <div className="hero-copy">
          <h1 className="hero-title">
            강의 영상을<br />
            <span className="peri">지식으로</span><br />
            <span className="dim">연결하다</span>
          </h1>
          <p className="hero-desc">
            <strong>GraphLEC</strong>은 슬라이드, 음성, 필기를 동시에 분석해<br />
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
