import {
  compactText,
  formatScore,
  resultFileUrl,
  hideMissingImage,
  statusText,
  pendingText,
} from '../verifierUtils'
import {
  ChipList,
  TextBlock,
  MouseTooltip,
} from '../VerifyReportParts'

/**
 * SlideReviewStage: 슬라이드 오류 검토 단계 메인 컴포넌트
 * 메인 패널의 요약 로직에 의존하지 않고 자체적으로 칩과 리스트를 렌더링합니다.
 */
export default function SlideReviewStage({ model, status, resultId }) {
  const rows = model.slideFindings
  const models = model.slideReviewModels || []

  // 이 스테이지에 특화된 모델 정보 칩
  const renderModelChip = () => {
    if (!models.length) return null
    return (
      <MouseTooltip
        className="vf-header-chip vf-header-chip--tooltip"
        tabIndex={0}
        tooltip={
          <>
            <span className="vf-bold">사용 모델</span>
            <span className="vf-header-chip-tooltip-value">{models[0]}</span>
          </>
        }
        tooltipClassName="vf-header-chip-tooltip"
      >
        모델
      </MouseTooltip>
    )
  }

  if (!model.slideFindings.length) {
    return (
      <section className={`vf-claim-flow vf-claim-flow--${status}`}>
        <div className="vf-claim-flow-head">
          <div>
            <span>슬라이드 오류 목록</span>
            <div className="vf-claim-flow-title-row">
              <h2>슬라이드 오류</h2>
              <div className="vf-header-chip-list">
                {renderModelChip()}
              </div>
            </div>
          </div>
          <span className="vf-bold">{statusText(status)}</span>
        </div>
        <div className="vf-stage-pending">{pendingText(status, '슬라이드 오류 결과가 생성되는 중입니다.', '슬라이드 오류 결과는 아직 없습니다.')}</div>
      </section>
    )
  }

  return (
    <section className="vf-claim-flow vf-claim-flow--slide">
      <div className="vf-claim-flow-head">
        <div>
          <span>슬라이드 오류 목록</span>
          <div className="vf-claim-flow-title-row">
            <h2>슬라이드 오류</h2>
            <div className="vf-header-chip-list">
              {renderModelChip()}
            </div>
          </div>
        </div>
        <span className="vf-bold">{rows.length}</span>
      </div>
      {rows.length ? (
        <div className="vf-record-list">
          {rows.map((item, index) => {
            const imageUrl = resultFileUrl(item.slide_image_path, resultId)
            return (
              <article key={item.slide_error_id || index} className="vf-record vf-slide-report">
                {imageUrl && (
                  <div className="vf-slide-thumb">
                    <img src={imageUrl} alt={compactText(item.slide_title || '슬라이드 이미지')} onError={hideMissingImage} />
                  </div>
                )}
                <div className="vf-slide-detail">
                  <div className="vf-record-head">
                    <span className="vf-bold">{compactText(item.slide_error_id || `S${index + 1}`)}</span>
                    <ChipList items={[
                      item.slide_number ? `슬라이드 ${item.slide_number}` : '',
                      item.slide_title,
                      item.error_type_label || item.error_type,
                      `신뢰도 ${formatScore(item.confidence)}`,
                      `심각도 ${formatScore(item.severity_score)}`,
                      item.source,
                      item.model,
                    ]} />
                  </div>
                  <div className="vf-diff-row">
                    <div><span>문제 표기</span><span className="vf-bold">{compactText(item.problematic_text)}</span></div>
                    <div><span>수정 제안</span><span className="vf-bold">{compactText(item.corrected_text || item.suggested_fix)}</span></div>
                  </div>
                  <TextBlock label="근거">{item.reason}</TextBlock>
                  <TextBlock label="슬라이드 이미지">{item.slide_image_path}</TextBlock>
                </div>
              </article>
            )
          })}
        </div>
      ) : (
        <div className="vf-report-empty">
          표시할 항목이 없습니다.
        </div>
      )}
    </section>
  )
}
