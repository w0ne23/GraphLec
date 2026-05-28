import { Fragment } from 'react'
import {
  statusText,
} from './verifierUtils'
import { VERIFY_STEPS } from './verifierConstants'

const STEP_STATUS_CLASS = {
  done: 'vf-report-step--done',
  run: 'vf-report-step--run',
}

function cx(...classes) {
  return classes.filter(Boolean).join(' ')
}

export default function VerifyStageTimeline({ statuses = [], activeTab, onSelectTab, compact = false }) {
  return (
    <div className={cx('vf-report-rail', compact && 'vf-report-rail--compact', 'vf-report-rail--tabs')}>
      {VERIFY_STEPS.map((step, index) => {
        const status = statuses[index] || 'wait'
        const isActive = activeTab === step.key
        const linkStatus = status === 'done' ? 'done' : 'idle'
        return (
          <Fragment key={step.key}>
            {index === 4 && <span className="vf-report-step-divider" aria-hidden="true" />}
            <button
              type="button"
              className={cx('vf-report-step', STEP_STATUS_CLASS[status], isActive && 'vf-report-step--active')}
              onClick={() => onSelectTab(step.key)}
              aria-pressed={isActive}
            >
              <div className="vf-report-step-body">
                <span className="vf-bold">{step.label}</span>
                <span className="vf-report-step-status">{statusText(status)}</span>
              </div>
            </button>
            {index < 3 && <span className={cx('vf-report-step-link', linkStatus === 'done' && 'vf-report-step-link--done')} aria-hidden="true" />}
          </Fragment>
        )
      })}
    </div>
  )
}
