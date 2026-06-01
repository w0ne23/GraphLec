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
        <div className="hero-card-stage hero-card-stage-paused">
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
          <div className="hero-orbit-card hero-orbit-card-2" />
          <div className="hero-orbit-card hero-orbit-card-3" />
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
