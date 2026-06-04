const PROFESSOR_CHECK_MIN_SCORE = 0.4

export const ISSUE_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'factual_error', code: 'A', label: '발언 자체 오류' },
  { key: 'temporal_error', code: 'B', label: '시간적 오류' },
  { key: 'scope_overclaim', code: 'C', label: '범위 과잉 단정' },
  { key: 'confusing_explanation', code: 'D', label: '혼동 가능 설명' },
]

export const ISSUE_FILTER_DESCRIPTIONS = {
  factual_error: '문장 자체의 개념, 인과관계, 용어 연결, 수치가 강의 문맥을 봐도 틀린 경우입니다.',
  temporal_error: '현재성, 최신성, 지원 여부, 사용 여부, 시점 의존 수치나 상태가 기준 시점에서 틀리거나 확인이 필요한 경우입니다.',
  scope_overclaim: '반례나 예외가 있는데도 항상, 모든, 오직, 반드시처럼 범위를 과하게 닫아 말한 경우입니다.',
  confusing_explanation: '발언 자체가 명백히 틀렸다고 단정하기보다, 학생이 핵심 개념이나 주체/과정을 잘못 외울 가능성이 큰 설명입니다.',
}

export function asArray(value) {
  return Array.isArray(value) ? value : []
}

export function formatTime(seconds) {
  const value = Number(seconds)
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0
  const m = Math.floor(safe / 60)
  const s = Math.floor(safe % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function formatPercent(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '-'
  return `${Math.round(number * 100)}%`
}

export function toNumberOrUndefined(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

export function compactText(value, fallback = '-') {
  const text = String(value ?? '').trim()
  return text || fallback
}

function firstFilled(...values) {
  return values.find(value => value !== undefined && value !== null && value !== '') ?? ''
}

export function toResultFileUrl(value, resultId = '') {
  const raw = String(value ?? '').trim()
  if (!raw) return ''
  if (/^(https?:|data:|blob:)/.test(raw)) return raw

  const normalized = raw.replace(/\\/g, '/')
  const resultMatch = normalized.match(/(?:local_storage|\/?files)\/results\/([^/]+)\/(.+)$/)
  if (resultMatch) {
    const [, pathResultId, filePath] = resultMatch
    if (!resultId || pathResultId === resultId) return `/files/results/${pathResultId}/${filePath}`
    return ''
  }

  if (normalized.startsWith('/files/')) return normalized

  const localStorageMarker = 'local_storage/'
  const localStorageMarkerIndex = normalized.indexOf(localStorageMarker)
  if (localStorageMarkerIndex >= 0) {
    return `/files/${normalized.slice(localStorageMarkerIndex + localStorageMarker.length)}`
  }

  return raw
}

export function formatModelName(value) {
  const text = compactText(value, '')
  return text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : ''
}

function normalizeDetailText(value) {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

function isSameDetailText(left, right) {
  const leftText = normalizeDetailText(left)
  const rightText = normalizeDetailText(right)
  return leftText !== '' && leftText === rightText
}

export function uniqueDetailValue(value, previousValues) {
  if (value === undefined || value === null || value === '') return ''
  return previousValues.some(previous => isSameDetailText(value, previous)) ? '' : value
}

function labelForIssueType(type) {
  const labels = {
    factual_error: '발언 자체 오류',
    temporal_error: '시간적 오류',
    scope_overclaim: '범위 과잉 단정',
    confusing_explanation: '혼동 가능 설명',
    outdated: '시간적 오류',
    simple_factual_error: '단순 사실 오류',
    scope_error: '범위 오류',
  }
  return labels[type] || compactText(type)
}

function labelForIssueSubtype(type) {
  const labels = {
    factual_error: '발언 자체 오류',
    temporal_error: '시간적 오류',
    scope_overclaim: '범위 과잉 단정',
    confusing_explanation: '혼동 가능 설명',
    simple_factual_error: '단순 사실 오류',
    scope_error: '범위 오류',
    outdated: '시간적 오류',
  }
  return labels[type] || compactText(type)
}

export function getIssueSubtype(item) {
  const direct = item.feedback_type || item.issue_type || item.type || item.issue_subtype
  if (direct === 'temporal_error') return 'temporal_error'
  if (direct === 'outdated') return 'temporal_error'
  if (direct === 'scope_overclaim') return 'scope_overclaim'
  if (direct === 'scope_error') return 'scope_overclaim'
  if (direct === 'confusing_explanation') return 'confusing_explanation'
  if (direct === 'factual_error') return 'factual_error'
  if (direct === 'simple_factual_error') return 'factual_error'
  if (item.issue_pattern === 'scope_overstatement') return 'scope_overclaim'
  return ''
}

export function countIssueFilters(items) {
  return items.reduce((acc, item) => {
    const key = getIssueSubtype(item) || 'unknown'
    acc.all += 1
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, { all: 0, factual_error: 0, temporal_error: 0, scope_overclaim: 0, confusing_explanation: 0 })
}

export function matchesIssueFilter(item, filter) {
  if (!filter || filter === 'all') return true
  return getIssueSubtype(item) === filter
}

export function claimDisplayIssueKey(claim) {
  return getIssueSubtype(claim) || claim.feedback_type || claim.issue_type || claim.type
}

export function labelForClaimIssue(claim) {
  const key = claimDisplayIssueKey(claim)
  return labelForIssueSubtype(key) || labelForIssueType(key)
}

export function formatIssueTypeScores(scores) {
  if (!scores || typeof scores !== 'object') return ''
  return ISSUE_FILTERS
    .filter(item => item.key !== 'all')
    .map(item => {
      const value = scores[item.key] ?? scores[item.code] ?? scores[item.code?.toLowerCase?.()]
      const number = Number(value)
      return Number.isFinite(number) ? `${item.label} ${Math.round(number * 100)}%` : ''
    })
    .filter(Boolean)
    .join(' / ')
}

function getSlideImageUrl(slide = {}, resultId = '') {
  return toResultFileUrl(firstFilled(
    slide.slide_image_url,
    slide.slide_image_path,
    slide.image_url,
    slide.image_path,
    slide.thumbnail_url,
    slide.thumbnail_path,
    slide.base_path
  ), resultId)
}

function getSlideTitle(slide = {}) {
  return firstFilled(slide.slide_title, slide.title)
}

function getSlideNumber(slide = {}) {
  return firstFilled(slide.slide_number, slide.slide_no, slide.slide_index)
}

function collectSourceIssues(verifier) {
  const sourceIssues = []
  asArray(verifier?.feedback_items).forEach(item => {
    asArray(item?.evidence?.source_issues).forEach(issue => {
      if (issue) sourceIssues.push(issue)
    })
  })
  return [...asArray(verifier?.issues), ...sourceIssues]
}

function addSlideMeta(slides, slide = {}, resultId = '') {
  const slideNumber = getSlideNumber(slide)
  if (slideNumber === '') return

  const key = String(slideNumber)
  const previous = slides.get(key) || { slideNumber }
  slides.set(key, {
    ...previous,
    slideNumber,
    title: previous.title || getSlideTitle(slide),
    imageUrl: previous.imageUrl || getSlideImageUrl(slide, resultId),
  })
}

function getExplicitSceneList(verifier) {
  return [
    verifier?.timeline,
    verifier?.scenes,
    verifier?.video_timeline,
    verifier?.slide_timeline,
    verifier?.views?.timeline,
    verifier?.views?.timeline?.items,
  ].find(value => asArray(value).length > 0) || []
}

function parseTimeText(value) {
  const text = String(value ?? '').trim()
  if (!text) return undefined
  const parts = text.split(':').map(Number)
  if (parts.some(Number.isNaN)) return undefined
  if (parts.length === 1) return parts[0]
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  return parts[0] * 3600 + parts[1] * 60 + parts[2]
}

function getSceneTimestamp(scene = {}, slide = {}) {
  return (
    toNumberOrUndefined(scene.timestamp) ??
    toNumberOrUndefined(scene.start_time) ??
    toNumberOrUndefined(scene.scene_start_sec) ??
    toNumberOrUndefined(scene.timestamp_sec) ??
    toNumberOrUndefined(slide.timestamp) ??
    toNumberOrUndefined(slide.start_time) ??
    parseTimeText(scene.timestamp_formatted) ??
    parseTimeText(slide.timestamp_formatted)
  )
}

function getSceneEndTime(scene = {}, slide = {}) {
  return (
    toNumberOrUndefined(scene.end_time) ??
    toNumberOrUndefined(scene.scene_end_sec) ??
    toNumberOrUndefined(slide.end_time) ??
    toNumberOrUndefined(slide.scene_end_sec)
  )
}

function sceneSortValue(scene) {
  return toNumberOrUndefined(scene.timestamp) ?? parseTimeText(scene.timestamp) ?? Number.MAX_SAFE_INTEGER
}

function addScene(scenes, scene = {}, slide = {}, resultId = '') {
  const timestamp = getSceneTimestamp(scene, slide)
  const slideNumber = firstFilled(scene.slide_number, getSlideNumber(slide))
  const sceneNumber = firstFilled(scene.scene_number, scene.scene_index, slide.scene_number, slide.scene_index)
  if (timestamp === undefined && slideNumber === '' && sceneNumber === '') return

  const key = `${sceneNumber || 'scene'}:${slideNumber || 'slide'}:${timestamp ?? scene.timestamp ?? ''}`
  if (scenes.has(key)) return

  const title = firstFilled(
    getSlideTitle(slide),
    getSlideTitle(scene),
    scene.text,
    slideNumber !== '' ? `Slide ${slideNumber}` : ''
  )
  scenes.set(key, {
    timestamp: timestamp ?? firstFilled(scene.timestamp, scene.timestamp_formatted, slide.timestamp_formatted, '00:00'),
    end_time: getSceneEndTime(scene, slide),
    type: scene.type || (slide.role === 'elaborated' ? 'emphasis' : 'slide'),
    text: title,
    image_url: toResultFileUrl(firstFilled(scene.image_url, scene.image_path, getSlideImageUrl(slide, resultId)), resultId),
    scene_number: sceneNumber,
    slide_number: slideNumber,
  })
}

export function buildVerifierSlideMap(verifier, timelineScenes = [], resultId = '') {
  const slides = new Map()

  asArray(timelineScenes).forEach(scene => addSlideMeta(slides, scene, resultId))
  asArray(getExplicitSceneList(verifier)).forEach(scene => addSlideMeta(slides, scene, resultId))
  collectSourceIssues(verifier).forEach(issue => {
    addSlideMeta(slides, issue?.judge_context?.slide, resultId)
    asArray(issue?.judge_context?.context_bundle?.target_contexts).forEach(context => addSlideMeta(slides, context, resultId))
    asArray(issue?.judge_context?.context_bundle?.neighbor_contexts).forEach(context => addSlideMeta(slides, context, resultId))
  })
  asArray(verifier?.slide_errors).forEach(error => addSlideMeta(slides, error, resultId))

  return slides
}

export function buildVerifierScenes(verifier, timelineScenes = [], resultId = '') {
  const scenes = new Map()
  asArray(timelineScenes).forEach(scene => addScene(scenes, scene, scene, resultId))

  if (scenes.size === 0) {
    asArray(getExplicitSceneList(verifier)).forEach(scene => addScene(scenes, scene, scene, resultId))
  }

  if (scenes.size === 0) {
    collectSourceIssues(verifier).forEach(issue => {
      const slide = issue?.judge_context?.slide || {}
      addScene(scenes, issue?.location || issue?.context || {}, slide, resultId)
      asArray(issue?.judge_context?.context_bundle?.target_contexts).forEach(context => addScene(scenes, context, slide, resultId))
      asArray(issue?.judge_context?.context_bundle?.neighbor_contexts).forEach(context => addScene(scenes, context, slide, resultId))
    })
  }

  return Array.from(scenes.values()).sort((a, b) => sceneSortValue(a) - sceneSortValue(b))
}

export function groupTyposBySlide(items, slideMap = new Map(), resultId = '') {
  const groups = new Map()
  asArray(items).forEach((typo, idx) => {
    const slideNumber = typo.slide_number ?? 'unknown'
    const key = String(slideNumber)
    const slide = slideMap.get(key) || {}
    const typoImageUrl = getSlideImageUrl(typo, resultId)
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        slideNumber,
        title: typo.slide_title || slide.title || '',
        imageUrl: slide.imageUrl || typoImageUrl || '',
        items: [],
      })
    }

    const group = groups.get(key)
    if (!group.imageUrl) {
      group.imageUrl = slide.imageUrl || typoImageUrl || ''
    }
    if (!group.title) {
      group.title = typo.slide_title || slide.title || ''
    }
    group.items.push({ ...typo, _typoIndex: idx })
  })

  return Array.from(groups.values()).sort((a, b) => {
    const aNumber = Number(a.slideNumber)
    const bNumber = Number(b.slideNumber)
    if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) return aNumber - bNumber
    return String(a.slideNumber).localeCompare(String(b.slideNumber))
  })
}

function uniqueStrings(values) {
  return [...new Set(asArray(values).map(value => String(value || '').trim()).filter(Boolean))]
}

function sortUtteranceIds(values) {
  return uniqueStrings(values).sort((a, b) => {
    const left = /^U\d+$/.test(a) ? Number(a.slice(1)) : Number.MAX_SAFE_INTEGER
    const right = /^U\d+$/.test(b) ? Number(b.slice(1)) : Number.MAX_SAFE_INTEGER
    return left - right || a.localeCompare(b)
  })
}

function extractUtteranceIds(text) {
  if (!text) return []
  return uniqueStrings(String(text).match(/\bU\d{4,}\b/g) || [])
}

function transcriptContextsFromText(text, targetContextIds = []) {
  const targetSet = new Set(uniqueStrings(targetContextIds))
  return String(text || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const match = line.match(/^([^:：]+)[:：]\s*(.*)$/)
      const contextId = match ? match[1].trim() : ''
      return {
        context_id: contextId || `transcript-${index + 1}`,
        text: match ? match[2].trim() : line,
        is_target: contextId ? targetSet.has(contextId) : false,
      }
    })
    .filter(context => context.text)
}

function getItemLocation(item, sourceClaim = {}) {
  return item.location || sourceClaim.location || {}
}

function getItemStartTime(item, sourceClaim = {}) {
  const location = getItemLocation(item, sourceClaim)
  const value = Number(location.start_time ?? item.start_time ?? sourceClaim.start_time)
  return Number.isFinite(value) ? value : undefined
}

function getItemSceneLabel(item, sourceClaim = {}) {
  const location = getItemLocation(item, sourceClaim)
  const evidence = item.evidence || {}
  const sourceIssue = asArray(evidence.source_issues)[0] || {}
  const context = sourceIssue.context || {}
  const targetContext = sourceIssue.judge_context?.context_bundle?.target_contexts?.[0] || {}
  const contextId =
    location.context_id ||
    evidence.context_id ||
    item.context_id ||
    sourceClaim.context_id ||
    context.context_id ||
    asArray(context.context_ids)[0] ||
    targetContext.context_id
  const sceneNumber =
    toNumberOrUndefined(location.scene_number) ??
    toNumberOrUndefined(evidence.scene_number) ??
    toNumberOrUndefined(item.scene_number) ??
    toNumberOrUndefined(sourceClaim.scene_number) ??
    toNumberOrUndefined(String(contextId || '').match(/SC0*(\d+)/)?.[1])

  return sceneNumber !== undefined ? `Scene ${String(sceneNumber).padStart(2, '0')}` : ''
}

function getItemSlideTitle(item) {
  const evidence = item.evidence || {}
  const sourceIssue = asArray(evidence.source_issues).find(issue => issue?.judge_context?.slide?.title)
  return sourceIssue?.judge_context?.slide?.title || ''
}

function getFeedbackUtteranceIds(item, sourceClaim = {}) {
  const evidence = item.evidence || {}
  const sourceIssues = asArray(evidence.source_issues)
  return sortUtteranceIds([
    ...asArray(item.utterance_ids),
    ...asArray(item.related_utterance_ids),
    ...asArray(evidence.related_utterance_ids),
    item.utterance_id,
    sourceClaim.utterance_id,
    ...sourceIssues.map(issue => issue?.utterance_id),
    ...extractUtteranceIds(evidence.evidence_in_context),
    ...extractUtteranceIds(item.confirmation_reason || evidence.confirmation_reason),
  ])
}

function claimTextCandidates(item, sourceClaim = {}) {
  return uniqueStrings([
    item.claim_text,
    item.resolved_claim,
    sourceClaim.claim_text,
    sourceClaim.resolved_claim,
    item.problem?.problematic_content,
  ])
}

function getFeedbackTranscriptContexts(item, sourceClaim = {}) {
  const evidence = item.evidence || {}
  const contexts = []

  asArray(evidence.source_issues).forEach(issue => {
    const bundle = issue?.judge_context?.context_bundle || {}
    asArray(bundle.target_contexts).forEach(context => {
      if (context?.text) contexts.push(context)
    })
    asArray(bundle.window_contexts).forEach(context => {
      if (context?.text) contexts.push(context)
    })
    contexts.push(...transcriptContextsFromText(bundle.current_slide_transcript, bundle.target_context_ids))
    if (issue?.context?.text) contexts.push(issue.context)
  })

  const seenTexts = new Set()
  const normalized = contexts
    .map(context => ({
      context_id: context.context_id,
      is_target: Boolean(context.is_target),
      slide_number: context.slide_number,
      text: String(context.text || '').trim(),
    }))
    .filter(context => {
      if (!context.text || seenTexts.has(context.text)) return false
      seenTexts.add(context.text)
      return true
    })

  const targetContexts = normalized.filter(context => context.is_target)
  if (targetContexts.length) return targetContexts

  const candidates = claimTextCandidates(item, sourceClaim)
  const matchingContexts = normalized.filter(context => (
    candidates.some(candidate => context.text.includes(candidate) || candidate.includes(context.text))
  ))
  return matchingContexts.length ? matchingContexts : normalized
}

function getRejectionReason(item) {
  return (
    item.rejection_reason ||
    item.professor_check_reason ||
    item.review_reason ||
    item.evidence?.rejection_reason ||
    item.checks?.crosscheck?.reason ||
    ''
  )
}

function getModelVerdicts(item) {
  const rows = asArray(item.checks?.crosscheck?.model_results)
  return rows.reduce((acc, row) => {
    const model = row?.model || row?.resolved_model || row?.source_model
    if (model) acc[model] = row
    return acc
  }, {})
}

function getCrosscheckScoreFromModels(modelResults) {
  const rows = asArray(modelResults)
  let weightedSum = 0
  let totalWeight = 0

  rows.forEach(row => {
    const score = toNumberOrUndefined(row?.confidence ?? row?.vote_score ?? row?.score)
    const weight = toNumberOrUndefined(row?.model_weight) ?? 1
    if (score === undefined || weight <= 0) return
    weightedSum += score * weight
    totalWeight += weight
  })

  if (totalWeight <= 0) return undefined
  return Math.max(0, Math.min(1, weightedSum / totalWeight))
}

function getCrosscheckScore(item, crosscheck = {}) {
  return (
    toNumberOrUndefined(item.severity_score) ??
    toNumberOrUndefined(item.classified_issue_verifier?.final_severity_score) ??
    toNumberOrUndefined(item.crosscheck_score) ??
    toNumberOrUndefined(item.score) ??
    toNumberOrUndefined(crosscheck.score) ??
    toNumberOrUndefined(crosscheck.scoring?.score) ??
    getCrosscheckScoreFromModels(crosscheck.model_results)
  )
}

function displayStageFromScore(score, fallbackStatus = '') {
  const status = fallbackStatus === 'review_needed' ? 'professor_check' : fallbackStatus
  if (status === 'confirmed') return 'confirmed'
  if (status) return status
  if (score !== undefined) {
    return score >= PROFESSOR_CHECK_MIN_SCORE ? 'professor_check' : 'rejected'
  }
  return 'professor_check'
}

export function scoreLabel(score) {
  return score !== undefined ? formatPercent(score) : ''
}

export function feedbackItemToClaim(item, claimById) {
  const sourceClaim = claimById.get(item.source_claim_id) || {}
  const problem = item.problem || {}
  const feedback = item.professor_feedback || {}
  const evidence = item.evidence || {}
  const crosscheck = item.checks?.crosscheck || {}
  const crosscheckScore = getCrosscheckScore(item, crosscheck)
  const location = getItemLocation(item, sourceClaim)
  const utteranceIds = getFeedbackUtteranceIds(item, sourceClaim)
  const status = displayStageFromScore(crosscheckScore, item.status)
  const title =
    problem.problematic_content ||
    item.claim_text ||
    sourceClaim.claim_text ||
    item.resolved_claim ||
    sourceClaim.resolved_claim ||
    problem.summary ||
    '-'

  return {
    ...item,
    stage: status || 'professor_check',
    utterance_id: utteranceIds.join(', ') || item.utterance_id || sourceClaim.utterance_id,
    utterance_ids: utteranceIds,
    start_time: getItemStartTime(item, sourceClaim),
    slide_number: location.slide_number ?? evidence.slide_number,
    scene_label: getItemSceneLabel(item, sourceClaim),
    slide_title: getItemSlideTitle(item),
    claim_text: title,
    resolved_claim: item.resolved_claim || sourceClaim.resolved_claim,
    issue: problem.summary || feedback.summary,
    correct_info: problem.correct_info,
    issue_type: item.feedback_type || item.issue_type || item.type,
    issue_type_code: item.issue_type_code,
    issue_type_code_label: item.issue_type_code_label,
    issue_type_scores: item.issue_type_scores || crosscheck.scoring?.issue_type_scores,
    primary_issue_type: item.primary_issue_type || crosscheck.scoring?.primary_issue_type,
    secondary_issue_types: item.secondary_issue_types || crosscheck.scoring?.secondary_issue_types,
    issue_type_rationale: item.issue_type_rationale,
    issue_category_label: item.feedback_label || item.issue_category_label,
    transcript_contexts: getFeedbackTranscriptContexts(item, sourceClaim),
    transcript_claim_text: item.claim_text || sourceClaim.claim_text,
    student_misunderstanding: feedback.student_misunderstanding,
    why_it_matters: feedback.why_it_matters,
    suggested_rephrase: feedback.suggested_rephrase,
    teaching_note: feedback.teaching_note,
    why_wrong: problem.why_wrong || feedback.why_wrong,
    issue_basis: problem.issue_basis || feedback.issue_basis,
    student_error: problem.student_error || feedback.student_error,
    counterexample_or_condition:
      problem.counterexample_or_condition ||
      evidence.counterexample_or_condition ||
      feedback.counterexample_or_condition,
    context_resolution: problem.context_resolution || evidence.context_resolution || feedback.context_resolution,
    recommendation: problem.recommendation || feedback.teaching_note,
    evidence_in_context: evidence.evidence_in_context || feedback.evidence_in_context,
    severity_score: item.severity_score ?? item.classified_issue_verifier?.final_severity_score,
    crosscheck_score: crosscheckScore,
    crosscheck_score_verdict: item.crosscheck_score_verdict ?? crosscheck.verdict,
    crosscheck_weighted_status: status,
    model_verdicts: getModelVerdicts(item),
    rejection_reason: status === 'rejected' ? getRejectionReason(item) : item.professor_check_reason || item.review_reason,
    evidence_sources: evidence.evidence_sources || item.evidence_sources,
  }
}
