export const MOCK_LECTURE = {
  id: 'test-uuid-1234',
  title: '파이썬 기초 프로그래밍 강의',
}

export const MOCK_FILE = {
  name: '테스트 강의 영상.mp4',
  size: 428 * 1024 * 1024,
}

export const MOCK_CLAIMS = [
  {
    utterance_id: 'claim-01',
    claim_text: '파이썬은 1991년에 귀도 반 로섬에 의해 발표되었습니다.',
    resolved_claim: '파이썬의 탄생 연도와 창시자 정보',
    start_time: 15.5,
    issue: '발표 연도 오기입 가능성 조사',
    correct_info: '1991년 발표가 맞음 (정상)',
    slide_number: 2,
  },
  {
    utterance_id: 'claim-02',
    claim_text: '리스트는 대괄호 []를 사용하여 선언하며, 내부 요소는 수정이 불가능합니다.',
    resolved_claim: '리스트의 가변성(Mutability) 설명 오류',
    start_time: 120.3,
    issue: '리스트는 수정 가능(Mutable)하나 불가능하다고 설명함',
    correct_info: '리스트는 가변 객체이므로 수정이 가능함. 수정 불가능한 것은 튜플(Tuple).',
    slide_number: 15,
  },
  {
    utterance_id: 'claim-03',
    claim_text: '딕셔너리는 키와 값의 쌍으로 이루어진 데이터 구조입니다.',
    resolved_claim: '딕셔너리 구조 설명',
    start_time: 305.8,
    issue: '내용 상 이상 없음',
    correct_info: '정확한 설명임',
    slide_number: 22,
  },
]
