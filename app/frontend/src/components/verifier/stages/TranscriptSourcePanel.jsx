import { Fragment } from 'react'
import {
  asArray,
  compactText,
  resultFileUrl,
  hideMissingImage,
  idNumber,
  sceneIdFromIndex,
  claimKey,
  formatTime,
  isFinalReviewTarget,
  issueTypeTone,
  getIssueType,
  hasIssueInComparison,
  safeDomId,
  jumpToClaimRow,
  getClaimFlowSource,
  uniqueTexts,
  firstFilled
} from '../verifierUtils'
import {
  MouseTooltip,
} from '../VerifyReportParts'

/**
 * 전사 데이터에서 주장의 색상(Tone)을 결정합니다.
 */
function transcriptClaimToneForRow(row, activeTab) {
  if (!row) return ''
  if (activeTab === 'claim_extraction') return 'claim-extraction'
  if (activeTab === 'issue_judge') {
    return row.issue || hasIssueInComparison(row) ? 'issue-judge' : 'muted'
  }
  if (activeTab === 'final_verification') {
    if (isFinalReviewTarget(row.severity)) return 'final-review'
    return 'muted'
  }
  return issueTypeTone(getIssueType(row.type || row.severity || row.issue))
}

/**
 * 전사 데이터 표시를 위해 행(Row) 데이터를 변환합니다.
 */
function transcriptClaimFromRow(row, activeTab) {
  return {
    ...getClaimFlowSource(row),
    _transcriptTone: transcriptClaimToneForRow(row, activeTab),
  }
}

/**
 * 클레임과 연결된 모든 컨텍스트 ID를 추출합니다.
 */
function claimContextIds(claim) {
  return uniqueTexts([
    claim?.context_id,
    ...asArray(claim?.context_ids),
  ])
}

/**
 * 컨텍스트 ID별로 클레임을 그룹화합니다.
 */
function claimsByContextId(claims) {
  const byContext = new Map()
  asArray(claims).forEach(claim => {
    claimContextIds(claim).forEach(contextId => {
      if (!byContext.has(contextId)) byContext.set(contextId, [])
      byContext.get(contextId).push(claim)
    })
  })
  return byContext
}

/**
 * 전사 데이터에서 매칭할 클레임 텍스트 후보들을 반환합니다.
 */
function claimTranscriptCandidates(claim) {
  return uniqueTexts([
    claim?.claim_text,
    claim?.resolved_claim,
  ])
}

/**
 * 전사 엔트리의 장면 키를 생성합니다.
 */
function transcriptSceneKey(entry) {
  const fallbackKey = compactText(entry?.context_id, 'scene')
  return compactText(
    entry?.scene_id || sceneIdFromIndex(firstFilled(entry?.scene_index, entry?.scene_number, idNumber(entry?.context_id, 'SC'))),
    fallbackKey,
  )
}

/**
 * 장면 번호를 계산합니다.
 */
function transcriptSceneNumber(scene, index) {
  const first = asArray(scene?.entries)[0] || {}
  const fromId = idNumber(first.context_id, 'SC')
  const value = idNumber(scene?.scene_id, 'SC')
    || firstFilled(scene?.scene_index, first.scene_index, scene?.scene_number, first.scene_number, fromId)
  return value || index + 1
}

/**
 * 장면 타이틀을 생성합니다.
 */
function transcriptSceneTitle(scene, index) {
  return `장면 ${transcriptSceneNumber(scene, index)}`
}

/**
 * 장면의 메타 정보를 생성합니다.
 */
function transcriptSceneMeta(scene, claimCount) {
  return `발화 문맥 ${asArray(scene?.entries).length}개 · 주장 ${claimCount}개`
}

/**
 * 시간 범위를 텍스트로 변환합니다.
 */
function transcriptTimeRange(start, end) {
  const startText = formatTime(start)
  const endText = formatTime(end)
  if (startText && endText) return `${startText}~${endText}`
  return startText || endText
}

/**
 * 컨텍스트 라벨(C1, C2 등)을 생성합니다.
 */
function transcriptContextLabel(entry, index) {
  const idValue = idNumber(entry?.context_id, 'C')
  if (idValue) return idValue
  const contextIndex = Number(entry?.context_index)
  const number = Number.isFinite(contextIndex) ? contextIndex + 1 : index + 1
  return number
}

/**
 * 전사 엔트리들을 장면 단위로 그룹화합니다.
 */
function groupTranscriptEntriesByScene(entries) {
  const groups = []
  const byKey = new Map()
  asArray(entries).forEach(entry => {
    const key = transcriptSceneKey(entry)
    if (!byKey.has(key)) {
      const group = {
        key,
        scene_id: entry?.scene_id || sceneIdFromIndex(firstFilled(entry?.scene_index, entry?.scene_number, idNumber(entry?.context_id, 'SC'))),
        slide_number: entry?.slide_number,
        scene_index: entry?.scene_index ?? entry?.scene_number,
        slide_title: entry?.slide_title || '',
        slide_image_path: entry?.slide_image_path,
        slide_image_url: entry?.slide_image_url,
        start_time: entry?.start_time,
        end_time: entry?.end_time,
        entries: [],
      }
      byKey.set(key, group)
      groups.push(group)
    }
    byKey.get(key).entries.push(entry)
  })
  return groups
}

/**
 * 전사 엔트리 내에서 클레임 구간을 찾아 세그먼트로 나눕니다.
 */
function transcriptSegmentsForEntry(entry, claims) {
  const text = compactText(entry?.text, '')
  const matches = []
  asArray(claims).forEach(claim => {
    const claimText = claimTranscriptCandidates(claim).find(candidate => text.includes(candidate))
    if (!claimText) return
    const index = text.indexOf(claimText)
    if (index < 0) return
    matches.push({
      claim,
      text: claimText,
      start: index,
      end: index + claimText.length,
    })
  })
  matches.sort((left, right) => left.start - right.start || left.end - right.end)

  const segments = []
  let cursor = 0
  matches.forEach(match => {
    if (match.start < cursor) return
    const before = text.slice(cursor, match.start)
    if (before) segments.push({ type: 'text', text: before })
    segments.push({ type: 'claim', text: match.text, claim: match.claim })
    cursor = match.end
  })
  const after = text.slice(cursor)
  if (after) segments.push({ type: 'text', text: after })

  return segments.length ? segments : [{ type: 'text', text }]
}

/**
 * 장면에 포함된 클레임 개수를 계산합니다.
 */
function sceneClaimCount(scene, byContext) {
  return uniqueTexts(asArray(scene?.entries).flatMap(entry => (
    asArray(byContext.get(entry.context_id)).map(claim => claimKey(claim))
  ))).length
}

/**
 * TranscriptSourcePanel: '내용에서 보기' 기능을 담당하는 컴포넌트
 */
export default function TranscriptSourcePanel({ model, resultId, rows, activeTab, onClaimClick }) {
  const sceneGroups = asArray(model.transcriptScenes).length
    ? asArray(model.transcriptScenes)
    : groupTranscriptEntriesByScene(model.transcriptEntries)
  if (!sceneGroups.length) return null

  const byContext = claimsByContextId(asArray(rows).length ? rows.map(row => transcriptClaimFromRow(row, activeTab)) : model.claims)

  return (
    <div className="vf-record-list vf-source-transcript-list">
      {sceneGroups.map((group, groupIndex) => {
        const entries = asArray(group.entries)
        const first = entries[0] || {}
        const imageUrl = resultFileUrl(group.slide_image_url || group.slide_image_path || first.slide_image_url || first.slide_image_path, resultId)
        const claimCount = sceneClaimCount(group, byContext)
        const title = transcriptSceneTitle(group, groupIndex)
        const meta = transcriptSceneMeta(group, claimCount)
        return (
          <article key={group.key} className="vf-record vf-source-transcript-scene">
            <div className="vf-source-transcript-scene-head">
              <span className="vf-bold">{title}</span>
              <span>| {meta}</span>
            </div>
            <div className="vf-source-transcript-scene-body">
              {imageUrl && (
                <aside className="vf-source-transcript-side">
                  <div className="vf-source-transcript-thumb">
                    <img src={imageUrl} alt={compactText(group.slide_title || first.slide_title || title || 'scene thumbnail')} onError={hideMissingImage} />
                  </div>
                </aside>
              )}
              <div className="vf-source-transcript-main">
                {entries.length ? entries.map((entry, entryIndex) => {
                  const claims = byContext.get(entry.context_id) || []
                  const segments = transcriptSegmentsForEntry(entry, claims)
                  let claimSegmentIndex = 0
                  const timeRange = transcriptTimeRange(entry.start_time, entry.end_time)
                  return (
                    <div key={`${entry.context_id || 'context'}-${entryIndex}`} className="vf-source-transcript-context-block">
                      <span className="vf-bold vf-source-transcript-context-id">{transcriptContextLabel(entry, entryIndex)}</span>
                      <p className="vf-source-transcript-context">
                        {segments.map((segment, segmentIndex) => {
                          if (segment.type === 'claim') {
                            claimSegmentIndex += 1
                            const claimIndexLabel = idNumber(claimKey(segment.claim), 'CL') || claimSegmentIndex
                            const claimTooltip = compactText(segment.claim?.resolved_claim || segment.claim?.claim_text || segment.text)
                            const claimTooltipContent = (
                              <>
                                <div>추출된 주장</div>
                                <p>{claimTooltip}</p>
                              </>
                            )
                            return (
                              <MouseTooltip
                                key={`${claimKey(segment.claim)}-${segmentIndex}`}
                                className={`vf-source-transcript-claim ${segment.claim?._transcriptTone ? `vf-source-transcript-claim--${safeDomId(segment.claim._transcriptTone)}` : ''}`}
                                tooltip={claimTooltipContent}
                                tooltipClassName="vf-source-transcript-claim-tooltip"
                                tabIndex={0}
                                ariaLabel={claimTooltip}
                                onClick={() => onClaimClick ? onClaimClick(claimKey(segment.claim)) : jumpToClaimRow(claimKey(segment.claim))}
                              >
                                <span className="vf-source-transcript-claim-index">{claimIndexLabel}</span>
                                {segment.text}
                              </MouseTooltip>
                            )
                          }
                          return (
                            <Fragment key={`text-${segmentIndex}`}>{segment.text}</Fragment>
                          )
                        })}
                        {timeRange && <span className="vf-source-transcript-time"> ({timeRange})</span>}
                      </p>
                    </div>
                  )
                }) : (
                  <div className="vf-source-transcript-empty">발화 없음</div>
                )}
              </div>
            </div>
          </article>
        )
      })}
    </div>
  )
}
