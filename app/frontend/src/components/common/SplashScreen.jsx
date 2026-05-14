import { useEffect, useState } from 'react'

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
      </div>
    </div>
  )
}

function FeatureIcon({ type }) {
  if (type === 'qna') {
    return (
      <div className="splash-icon splash-icon--qna" aria-hidden="true">
        <span className="qna-node qna-node--top" />
        <span className="qna-node qna-node--left" />
        <span className="qna-node qna-node--mid" />
        <span className="qna-node qna-node--right-top" />
        <span className="qna-node qna-node--right-bottom" />
        <span className="qna-link qna-link--a" />
        <span className="qna-link qna-link--b" />
        <span className="qna-link qna-link--c" />
        <span className="qna-link qna-link--d" />
        <span className="qna-mark">?!</span>
      </div>
    )
  }

  return (
    <div className={`splash-icon splash-icon--${type}`} aria-hidden="true">
      <span className="screen-base" />
      {type === 'verify' && (
        <>
          <span className="play-mark" />
          <span className="search-ring" />
          <span className="search-handle" />
          <span className="warning-triangle" />
          <span className="warning-bang">!</span>
        </>
      )}
      {type === 'recommend' && (
        <>
          <span className="recommend-tile" />
          <span className="recommend-play" />
          <span className="search-ring" />
          <span className="search-handle" />
        </>
      )}
    </div>
  )
}
