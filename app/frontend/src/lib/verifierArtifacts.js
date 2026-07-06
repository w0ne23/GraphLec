// 실제 result 상세보기에서 analyzer 산출물 파일을 로드해 buildReportModel이
// 기대하는 형태로 조립한다. 백엔드 verifier 응답은 classified_issue_artifacts를
// "파일 경로" dict로만 주므로(내용 아님), 프론트가 해당 파일을 fetch해야 한다.
// preview 흐름(useVerifierPreviewFlow)이 하는 것과 동일한 로딩을 result id 기준으로 수행.

function toFilesUrl(rawPath) {
  const raw = String(rawPath ?? '').trim().replace(/\\/g, '/')
  if (!raw) return ''
  const match = raw.match(/(?:local_storage|\/?files)\/results\/(.+)$/)
  return match ? `/files/results/${match[1]}` : ''
}

// classified_issue_artifacts에 없는 산출물(issue_judge_compare, merged_clean)은
// 같은 analyzer 디렉토리의 형제 파일이므로 issue_judge 경로에서 접미사만 바꿔 유도한다.
function siblingUrl(rawPath, fromSuffix, toSuffix) {
  const raw = String(rawPath ?? '').trim().replace(/\\/g, '/')
  if (!raw || !raw.endsWith(fromSuffix)) return ''
  return toFilesUrl(raw.slice(0, -fromSuffix.length) + toSuffix)
}

async function loadJson(url) {
  if (!url) return null
  const response = await fetch(url).catch(() => null)
  if (!response?.ok) return null
  return response.json().catch(() => null)
}

async function loadJsonl(url) {
  if (!url) return []
  const response = await fetch(url).catch(() => null)
  if (!response?.ok) return []
  const text = await response.text().catch(() => '')
  return text
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      try {
        return JSON.parse(line)
      } catch {
        return null
      }
    })
    .filter(Boolean)
}

export async function loadVerifierArtifacts(verifier) {
  const paths = verifier?.classified_issue_artifacts
  if (!paths || typeof paths !== 'object' || !Object.keys(paths).length) return null

  const urls = {
    claims: toFilesUrl(paths.claims_json),
    claimsJsonl: toFilesUrl(paths.claims_jsonl),
    issueJudge: toFilesUrl(paths.issue_judge),
    issueJudgeSummary: toFilesUrl(paths.issue_judge_summary),
    issueTypes: toFilesUrl(paths.issue_types),
    classifiedIssues: toFilesUrl(paths.classified_issues),
    issueJudgeCompare: siblingUrl(paths.issue_judge, '_issue_judge.json', '_issue_judge_compare.json'),
    mergedClean: siblingUrl(paths.issue_judge, '_issue_judge.json', '_merged_clean.json'),
  }

  const [
    claims,
    claimsJsonl,
    issueJudge,
    issueJudgeSummary,
    issueTypes,
    classifiedIssues,
    issueJudgeCompare,
    mergedClean,
  ] = await Promise.all([
    loadJson(urls.claims),
    loadJsonl(urls.claimsJsonl),
    loadJson(urls.issueJudge),
    loadJson(urls.issueJudgeSummary),
    loadJson(urls.issueTypes),
    loadJson(urls.classifiedIssues),
    loadJson(urls.issueJudgeCompare),
    loadJson(urls.mergedClean),
  ])

  const loaded = {
    claims,
    claimsJsonl,
    issueJudge,
    issueJudgeSummary,
    issueTypes,
    classifiedIssues,
    issueJudgeCompare,
    mergedClean,
  }
  // 로드 실패(null)한 키는 제거해 verifierArtifactsFromResult의 동기 폴백을 덮지 않게 한다.
  return Object.fromEntries(Object.entries(loaded).filter(([, value]) => value != null))
}
