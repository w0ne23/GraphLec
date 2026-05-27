import {
  asArray,
  asObject,
  compactText,
  hasDataForTab,
  typeLabel,
  uniqueTexts,
} from './verifierUtils'
import {
  MetricStrip,
  MouseTooltip,
} from './VerifyReportParts'

const UNKNOWN_MODEL_LABEL = '알 수 없음'

function normalizeHeaderChip(chip) {
  if (chip && typeof chip === 'object') {
    return {
      label: compactText(chip.label, ''),
      models: asArray(chip.models).map(item => compactText(item, '')).filter(Boolean),
      tooltipTitle: compactText(chip.tooltipTitle, ''),
      tooltipVariant: compactText(chip.tooltipVariant, ''),
    }
  }
  return { label: compactText(chip, ''), models: [], tooltipTitle: '', tooltipVariant: '' }
}

function HeaderChips({ chips }) {
  const visibleChips = asArray(chips).map(normalizeHeaderChip).filter(chip => chip.label)
  if (!visibleChips.length) return null

  function headerChipTooltip(chip) {
    if (!chip.models.length) return null
    return (
      <>
        <span className="vf-bold">{chip.tooltipTitle || (chip.label === '모델' ? '사용 모델' : '사용 모델 목록')}</span>
        {chip.tooltipVariant === 'value' ? (
          <span className="vf-header-chip-tooltip-value">{chip.models[0]}</span>
        ) : (
          <span className="vf-header-chip-tooltip-list">
            {chip.models.map(item => <span key={item}>{item}</span>) || <span>알 수 없음</span>}
          </span>
        )}
      </>
    )
  }

  return (
    <div className="vf-header-chip-list">
      {visibleChips.map((chip, index) => (
        <MouseTooltip
          key={`${chip.label}-${index}`}
          className={`vf-header-chip ${chip.models.length ? 'vf-header-chip--tooltip' : ''}`}
          tabIndex={0}
          tooltip={headerChipTooltip(chip)}
          tooltipClassName="vf-header-chip-tooltip"
        >
          {chip.label}
        </MouseTooltip>
      ))}
    </div>
  )
}

function CountBoard({ items }) {
  const rows = asArray(items).filter(item => compactText(item?.label, '') || item?.value !== undefined)
  if (!rows.length) return null

  return (
    <div className="vf-stage-data-board">
      {rows.map(item => (
        <div key={item.label}>
          <span>{compactText(item.label)}</span>
          <span className="vf-bold">{compactText(item.value)}</span>
        </div>
      ))}
    </div>
  )
}

function slideReviewModelChip(model) {
  const models = uniqueTexts(model.slideReviewModels)
  return {
    label: '모델',
    models: models.length ? models : [UNKNOWN_MODEL_LABEL],
    tooltipTitle: '사용 모델',
    tooltipVariant: 'value',
  }
}

export default function SlideReviewReportPanel({ model }) {
  const isReady = hasDataForTab(model, 'slide_review')
  const summary = asObject(model.slideErrors.summary)

  return (
    <section className="vf-stage-data" aria-label="슬라이드 오류 보고서">
      <div className="vf-stage-data-head">
        <span className="vf-bold">흐름 보고서</span>
      </div>
      <div className="vf-stage-data-panel">
        <div className="vf-stage-data-panel-head">
          <div>
            <span>슬라이드</span>
            <span className="vf-bold">슬라이드 오류</span>
          </div>
          <HeaderChips chips={[isReady ? slideReviewModelChip(model) : '']} />
        </div>
        <MetricStrip items={[
          { label: '슬라이드', value: isReady ? summary.total_slide_count ?? '-' : '-' },
          { label: '오류', value: isReady ? model.slideFindings.length : '-' },
          { label: '보고 대상', value: isReady ? summary.reportable_error_count ?? '-' : '-' },
          { label: '유형', value: isReady ? Object.keys(summary.breakdown_by_type || {}).length : '-' },
        ]} />
        <CountBoard items={Object.entries(asObject(summary.breakdown_by_type)).map(([label, value]) => ({ label: typeLabel(label), value }))} />
      </div>
    </section>
  )
}
