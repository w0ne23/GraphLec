import {
  compactText,
  firstFilled,
  formatScore,
  resultFileUrl,
  hideMissingImage,
  statusText,
  pendingText,
} from '../verifierUtils'
import {
  ChipList,
  TextBlock,
} from '../VerifyReportParts'

function slideImageSource(item) {
  return firstFilled(
    item.slide_image_url,
    item.image_url,
    item.thumbnail_url,
    item.slide_image_path,
    item.image_path,
    item.thumbnail_path,
    item.base_path
  )
}

/**
 * SlideReviewStage: 슬라이드 오타 검토 단계 메인 컴포넌트
 * 메인 패널의 요약 로직에 의존하지 않고 자체적으로 칩과 리스트를 렌더링합니다.
 */
export default function SlideReviewStage({ model, status, resultId }) {
  const rows = model.slideFindings

  if (!model.slideFindings.length) {
    return (
      <section className={`vf-claim-flow vf-claim-flow--${status}`}>
        <div className="vf-claim-flow-head">
          <div>
            <span>슬라이드 오타 목록</span>
            <div className="vf-claim-flow-title-row">
              <h2>슬라이드 오타</h2>
            </div>
          </div>
          <span className="vf-bold">{statusText(status)}</span>
        </div>
        <div className="vf-stage-pending">{pendingText(status, '슬라이드 오타 결과가 생성되는 중입니다.', '슬라이드 오타 결과는 아직 없습니다.')}</div>
      </section>
    )
  }

  return (
    <section className={`vf-claim-flow vf-claim-flow--${status}`}>
      <div className="vf-claim-flow-head">
        <div className="vf-claim-flow-title">
          <span>슬라이드 오타 목록</span>
          <div className="vf-claim-flow-title-row">
            <h2>슬라이드 오타</h2>
          </div>
        </div>
        <div className="vf-claim-flow-head-actions">
          <span className="vf-bold vf-claim-flow-count">{rows.length}</span>
        </div>
      </div>
      {rows.length ? (
        <div className="vf-record-list">
          {rows.map((item, index) => {
            const imageSource = slideImageSource(item)
            const imageUrl = resultFileUrl(imageSource, resultId)
            return (
              <article
                key={item.slide_error_id || index}
                className="vf-record vf-slide-report"
                data-image-missing-container="true"
              >
                {imageUrl && (
                  <div className="vf-slide-thumb" data-missing-target="true">
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
                  <TextBlock label="슬라이드 이미지">{imageSource}</TextBlock>
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
