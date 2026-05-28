import { Fragment } from 'react'
import {
  statusText,
} from './verifierUtils'
import { VERIFY_STEPS } from './verifierConstants'

export default function VerifyStageTimeline({ statuses = [], activeTab, onSelectTab }) {
  return (
    <div className="vf-report-rail vf-report-rail--tabs">
      {VERIFY_STEPS.map((step, index) => {
        const status = statuses[index] || 'wait'
        const isActive = activeTab === step.key
        const linkStatus = status === 'done' ? 'done' : 'idle'
        return (
          <Fragment key={step.key}>
            {index === 4 && <span className="vf-report-step-divider" aria-hidden="true" />}
            <button
              type="button"
              className={`vf-report-step vf-report-step--${status} ${isActive ? 'vf-report-step--active' : ''}`}
              onClick={() => onSelectTab(step.key)}
              aria-pressed={isActive}
            >
              <div>
                <span className="vf-bold">{step.label}</span>
                <span className="vf-report-step-status">{statusText(status)}</span>
              </div>
            </button>
            {index < 3 && <span className={`vf-report-step-link vf-report-step-link--${linkStatus}`} aria-hidden="true" />}
          </Fragment>
        )
      })}
    </div>
  )
}
