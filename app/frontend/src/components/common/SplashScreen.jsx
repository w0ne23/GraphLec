import { useEffect, useState } from 'react'
import graphQnaIcon from '../../assets/splash-icons/graph-qna.png'
import recommendIcon from '../../assets/splash-icons/recommend-combined.png'
import verifyIcon from '../../assets/splash-icons/verify-combined.png'

const EXIT_MS = 760

const FEATURES = [
  { key: 'verify', title: 'Verify', type: 'verify' },
  { key: 'qna', title: 'QnA', type: 'qna' },
  { key: 'recommend', title: 'recommend', type: 'recommend' },
]

export default function SplashScreen({ onDone }) {
  const [leaving, setLeaving] = useState(false)

  useEffect(() => {
    if (leaving) return

    const dismiss = () => {
      setLeaving(true)
      window.setTimeout(() => onDone?.(), EXIT_MS)
    }

    window.addEventListener('pointerdown', dismiss, { once: true })
    window.addEventListener('keydown', dismiss, { once: true })
    window.addEventListener('touchstart', dismiss, { once: true, passive: true })

    return () => {
      window.removeEventListener('pointerdown', dismiss)
      window.removeEventListener('keydown', dismiss)
      window.removeEventListener('touchstart', dismiss)
    }
  }, [leaving, onDone])

  return (
    <div
      className={`splash-screen${leaving ? ' splash-screen--leaving' : ''}`}
      aria-label="GraphLec intro"
      role="button"
      tabIndex={0}
    >
      <div className="splash-inner">
        <header className="splash-hero">
          <h1 className="splash-logo">
            Graph<span>Lec</span>
          </h1>
          <p className="splash-tagline">
            an all-in-one lecture video solution for content verification and intelligent search
          </p>
        </header>

        <div className="splash-cards">
          {FEATURES.map(feature => (
            <section className="splash-card" key={feature.key}>
              <h2>{feature.title}</h2>
              <FeatureIcon type={feature.type} />
            </section>
          ))}
        </div>

        <p className="splash-click-hint">아무곳이나 클릭하세요</p>
      </div>
    </div>
  )
}

function FeatureIcon({ type }) {
  if (type === 'qna') {
    return (
      <div className="splash-icon splash-icon--qna" aria-hidden="true">
        <img className="splash-icon-img splash-icon-img--graph" src={graphQnaIcon} alt="" />
      </div>
    )
  }

  if (type === 'verify') {
    return (
      <div className="splash-icon splash-icon--verify" aria-hidden="true">
        <img className="splash-icon-img splash-icon-img--combined splash-icon-img--verify" src={recommendIcon} alt="" />
      </div>
    )
  }

  return (
    <div className="splash-icon splash-icon--recommend" aria-hidden="true">
      <img className="splash-icon-img splash-icon-img--combined splash-icon-img--recommend" src={verifyIcon} alt="" />
    </div>
  )
}
