import { Fragment } from 'react'
import { 
  asArray, 
  compactText, 
  resultFileUrl, 
  hideMissingImage, 
  formatTime, 
  idNumber,
  claimKey,
  sceneIdFromIndex,
  sceneValueFromId,
  contextValueFromId,
  claimValueFromId,
  safeDomId,
  jumpToClaimRow,
  MouseTooltip
} from '../VerifyReportPanels'

import ClaimFlowRecord from './shared/ClaimFlowRecord'

/**
 * Stage 1: Claim Extraction
 * 전사 데이터로부터 지식 주장을 추출한 결과를 테이블 혹은 전사 흐름으로 보여줍니다.
 */
export default function ClaimExtractionStage({ model, rows, resultId, viewMode, onClaimClick }) {
  if (viewMode === 'transcript') {
    return (
      <TranscriptSourcePanel 
        model={model} 
        resultId={resultId} 
        rows={rows} 
        activeTab="claim_extraction" 
        onClaimClick={onClaimClick} 
      />
    )
  }

  return (
    <ClaimExtractionGroupedList 
      rows={rows} 
      resultId={resultId} 
    />
  )
}

// --- Stage 1 전용 컴포넌트 및 로직 ---

export function TranscriptSourcePanel({ model, resultId, rows, activeTab, onClaimClick }) {
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

export function ClaimExtractionGroupedList({ rows, resultId }) {
  const groups = groupClaimRowsBySceneContext(rows)
  return (
    <div className="vf-claim-table">
      <div className="vf-claim-table-head">장면</div>
      <div className="vf-claim-table-head">발화 문맥</div>
      <div className="vf-claim-table-head">주장</div>
      {groups.map((scene, sceneIndex) => (
        <Fragment key={scene.id}>
          <div
            className="vf-claim-table-cell vf-claim-table-cell--scene"
            style={{ gridRow: `span ${Math.max(1, scene.contexts.reduce((total, context) => total + context.rows.length, 0))}` }}
          >
            {sceneValueFromId(scene.id, sceneIndex)}
          </div>
          {scene.contexts.map((context, contextIndex) => (
            <Fragment key={context.id}>
              <div
                className="vf-claim-table-cell vf-claim-table-cell--context"
                style={{ gridRow: `span ${Math.max(1, context.rows.length)}` }}
              >
                {contextValueFromId(context.id, contextIndex)}
              </div>
              {context.rows.map((row, rowIndex) => (
                <div key={row.claim_id} className="vf-claim-table-record">
                  <ClaimFlowRecord
                    row={row}
                    activeTab="claim_extraction"
                    resultId={resultId}
                    displayId={claimValueFromId(row.claim_id, rowIndex)}
                  />
                </div>
              ))}
            </Fragment>
            ))}
        </Fragment>
      ))}
    </div>
  )
}

// --- Stage 1 전용 헬퍼 로직 ---

function claimContextIds(claim) {
  return Array.from(new Set([
    claim?.context_id,
    ...asArray(claim?.context_ids),
  ].filter(Boolean)))
}

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

function claimTranscriptCandidates(claim) {
  return Array.from(new Set([
    claim?.claim_text,
    claim?.resolved_claim,
  ].filter(Boolean)))
}

function transcriptSceneKey(entry) {
  const fallbackKey = compactText(entry?.context_id, 'scene')
  const sceneIdxValue = (entry?.scene_index ?? entry?.scene_number) || idNumber(entry?.context_id, 'SC')
  return compactText(
    entry?.scene_id || sceneIdFromIndex(sceneIdxValue),
    fallbackKey,
  )
}

function transcriptSceneNumber(scene, index) {
  const first = asArray(scene?.entries)[0] || {}
  const fromId = idNumber(first.context_id, 'SC')
  const sceneVal = (scene?.scene_index ?? first.scene_index) ?? (scene?.scene_number ?? first.scene_number) ?? fromId
  const value = idNumber(scene?.scene_id, 'SC') || sceneVal
  return value || index + 1
}

function transcriptSceneTitle(scene, index) {
  return `장면 ${transcriptSceneNumber(scene, index)}`
}

function transcriptSceneMeta(scene, claimCount) {
  return `발화 문맥 ${asArray(scene?.entries).length}개 · 주장 ${claimCount}개`
}

function transcriptTimeRange(start, end) {
  const startText = formatTime(start)
  const endText = formatTime(end)
  if (startText && endText) return `${startText}~${endText}`
  return startText || endText
}

function transcriptContextLabel(entry, index) {
  const idValue = idNumber(entry?.context_id, 'C')
  if (idValue) return idValue
  const contextIndex = Number(entry?.context_index)
  const number = Number.isFinite(contextIndex) ? contextIndex + 1 : index + 1
  return number
}

function groupTranscriptEntriesByScene(entries) {
  const groups = []
  const byKey = new Map()
  asArray(entries).forEach(entry => {
    const key = transcriptSceneKey(entry)
    if (!byKey.has(key)) {
      const sceneIdxValue = (entry?.scene_index ?? entry?.scene_number) || idNumber(entry?.context_id, 'SC')
      const group = {
        key,
        scene_id: entry?.scene_id || sceneIdFromIndex(sceneIdxValue),
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

function sceneClaimCount(scene, byContext) {
  return Array.from(new Set(asArray(scene?.entries).flatMap(entry => (
    asArray(byContext.get(entry.context_id)).map(claim => claimKey(claim))
  )))).length
}

function transcriptClaimToneForRow(row, activeTab) {
  if (!row) return ''
  return 'claim-extraction'
}

function transcriptClaimFromRow(row, activeTab) {
  const source = row.claim || row.comparison || row.issue || row.type || row.severity || {}
  return {
    ...source,
    _transcriptTone: transcriptClaimToneForRow(row, activeTab),
  }
}

function claimFlowContextId(row) {
  const source = row.claim || row.comparison || row.issue || row.type || row.severity || {}
  return compactText(source.context_id || asArray(source.context_ids)[0], '-')
}

function claimFlowSceneId(row) {
  const source = row.claim || row.comparison || row.issue || row.type || row.severity || {}
  const explicit = compactText(source.scene_id || source.scene_key, '')
  if (explicit) return explicit
  const contextId = claimFlowContextId(row)
  return sceneIdFromContextId(contextId) || compactText(source.scene_index || source.scene_number, '-')
}

function sceneIdFromContextId(contextId) {
  const match = compactText(contextId, '').match(/(?:^|-)SC\d+/i)
  return match ? match[0].replace(/^-/, '') : ''
}

function groupClaimRowsBySceneContext(rows) {
  const sceneGroups = []
  const sceneMap = new Map()

  asArray(rows).forEach(row => {
    const sceneId = claimFlowSceneId(row)
    const contextId = claimFlowContextId(row)
    if (!sceneMap.has(sceneId)) {
      const sceneGroup = { id: sceneId, contexts: [], contextMap: new Map() }
      sceneMap.set(sceneId, sceneGroup)
      sceneGroups.push(sceneGroup)
    }
    const sceneGroup = sceneMap.get(sceneId)
    if (!sceneGroup.contextMap.has(contextId)) {
      const contextGroup = { id: contextId, rows: [] }
      sceneGroup.contextMap.set(contextId, contextGroup)
      sceneGroup.contexts.push(contextGroup)
    }
    sceneGroup.contextMap.get(contextId).rows.push(row)
  })

  return sceneGroups
}
