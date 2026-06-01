import { useEffect, useState } from 'react'

const PIPELINES = {
  verify: {
    steps: [
      { label: '강의 업로드',         desc: '강의자가 영상을 업로드하면 검증 파이프라인 시작' },
      { label: '멀티모달 분석',        desc: '슬라이드·음성·필기를 동시에 처리해 구조화' },
      { label: 'Multi-LLM 강의 검증', desc: 'Multi-LLM이 여러 단계에 걸쳐 강의 정확성·완결성 검증' },
      { label: '피드백 제공',          desc: '강의자에게 개선 사항을 피드백 형태로 제공' },
    ],
    title: '강의 검증 파이프라인',
    sub: 'Multi-LLM이 강의 내용을 분석하고 강의자에게 피드백을 제공합니다.',
  },
  upload: {
    steps: [
      { label: '강의 업로드', desc: '강의자가 영상을 업로드하면 분석 파이프라인 시작' },
      { label: '멀티모달 분석', desc: '슬라이드·음성·필기를 동시에 처리해 구조화' },
      { label: '그래프 구축', desc: '강의 내용의 개념과 관계로 가중치 지식 그래프 구축' },
      { label: '추천·검색', desc: '학습자 질의에 맞는 강의 추천 및 QnA 제공' },
    ],
    title: '업로드부터 추천까지, 자동으로',
    sub: '영상 업로드 한 번으로 전체 파이프라인이 실행됩니다.',
  },
}

export default function WorkflowSection({ active }) {
  const [pipeline, setPipeline] = useState('verify')
  const [activeStep, setActiveStep] = useState(-1)
  const [allActive, setAllActive] = useState(false)

  useEffect(() => {
    if (!active) return

    const steps = PIPELINES[pipeline].steps
    let step = 1
    setAllActive(false)
    setActiveStep(0)

    const interval = setInterval(() => {
      setActiveStep(step)
      step++
      if (step >= steps.length) {
        clearInterval(interval)
        setTimeout(() => setAllActive(true), 900)
      }
    }, 900)

    return () => clearInterval(interval)
  }, [active, pipeline])

  const p = PIPELINES[pipeline]
  const isStepActive = (i) => allActive || activeStep === i

  return (
    <div className="how-inner">
      <p className="section-label">작동 방식</p>
      <h2 className="section-title">{p.title}</h2>
      <p className="section-sub">{p.sub}</p>

      <div className="how-steps">
        {p.steps.map((step, i) => (
          <div key={i} className={`how-step${isStepActive(i) ? ' step-active' : ''}`}>
            <div className="step-dot">{String(i + 1).padStart(2, '0')}</div>
            <div className="step-label">{step.label}</div>
            <div className="step-desc">{step.desc}</div>
          </div>
        ))}
      </div>

      <div className="slide-nav-btn">
        <button
          className={`btn-ghost-dark${pipeline === 'verify' ? ' pipeline-active' : ''}`}
          onClick={() => setPipeline('verify')}
        >
          검증
        </button>
        <button
          className={`btn-ghost-dark${pipeline === 'upload' ? ' pipeline-active' : ''}`}
          onClick={() => setPipeline('upload')}
        >
          업로드
        </button>
      </div>
    </div>
  )
}