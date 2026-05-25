import { 
  asObject, 
  compactText, 
  resultFileUrl, 
  hideMissingImage, 
  formatScore, 
  EmptyFiltered,
  ChipList,
  TextBlock,
  MetricStrip
} from '../VerifyReportPanels'

/**
 * Stage 5: Slide Review
 * 슬라이드 내의 오탈자나 시각적 구성 오류를 탐지한 결과를 보여줍니다.
 */
export default function SlideReviewStage({ model, rows, resultId, activeFilter, status }) {
  const isDone = status === 'done'
  const summary = asObject(model.slideErrors.summary)

  if (!model.slideFindings.length && !isDone) {
    return (
      <div className="vf-stage-pending">
        슬라이드 오류 결과가 생성되는 중입니다.
      </div>
    )
  }

  return (
    <div className="vf-stage-slide">
      {rows.length > 0 ? (
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
        <EmptyFiltered activeFilter={activeFilter} />
      )}
    </div>
  )
}
