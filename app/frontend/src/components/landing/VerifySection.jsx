export default function VerifySection() {
  return (
    <>
      <div className="feat-header">
        <p className="section-label">강의자 기능</p>
        <h2 className="section-title">강의를 업로드하면 AI가 검증합니다</h2>
        <p className="section-sub">멀티모달 분석과 Multi-LLM이 강의 내용을 분석하고 구조화된 피드백을 제공합니다.</p>
      </div>

      <div className="verify-body">
        <div className="verify-left">
          <div className="verify-eyebrow">
            <div className="feature-icon-wrap feature-icon-wrap-inline">
              <svg className="feature-icon" viewBox="0 0 24 24">
                <path d="M9 12l2 2 4-4" />
                <circle cx="12" cy="12" r="9" />
              </svg>
            </div>
            <span className="verify-label">VERIFY</span>
          </div>
          <div className="verify-big-title">
            슬라이드<br />음성<br /><span>필기</span>까지
          </div>
          <p className="verify-desc">
            강의의 모든 신호를 동시에 분석합니다. Multi-LLM이 내용의 정확성과 완결성을 검증하고, 개선이 필요한 구간을 짚어드립니다.
          </p>
          <div className="feature-tags">
            <span className="tag">멀티모달 분석</span>
            <span className="tag">Multi-LLM</span>
            <span className="tag">강의 검증</span>
            <span className="tag">피드백 제공</span>
          </div>
        </div>

        <div className="verify-right">
          <div className="vf-top-bar">
            <span className="vf-top-label">AI 피드백 리포트</span>
            <div className="vf-file-badge">
              <span className="vf-status-dot" />
              os1-1 · 분석 완료
            </div>
          </div>

          <div className="vf-summary">
            <div className="vf-summary-item">
              <div className="vf-summary-num warn">4</div>
              <div>
                <div className="vf-summary-text">일반 오류</div>
                <div className="vf-summary-sub">사실·오래됨·혼동·일반화</div>
              </div>
            </div>
            <div className="vf-summary-item">
              <div className="vf-summary-num info">2</div>
              <div>
                <div className="vf-summary-text">슬라이드 오타</div>
                <div className="vf-summary-sub">오타 · 표기 오류</div>
              </div>
            </div>
          </div>

          <div className="vf-cards">
            {/* 일반 오류 카드 */}
            <div className="vf-card">
              <div className="vf-error-block">
                <div className="vf-error-head">
                  <span className="vf-type-badge fact">사실 오류</span>
                  <span className="vf-card-issue">페이지 테이블은 CPU 안에 저장됩니다.</span>
                </div>
                <div className="vf-card-reason">
                  페이지 테이블은 주로 <strong>메인 메모리</strong>에 저장되고, CPU는 MMU와 TLB를 통해 주소 변환을 보조합니다.
                </div>
              </div>
              <div className="vf-error-block vf-error-block-separated">
                <div className="vf-error-head">
                  <span className="vf-type-badge time">오래된 내용</span>
                  <span className="vf-card-issue">현대 OS는 세그멘테이션만 사용합니다.</span>
                </div>
                <div className="vf-card-reason">
                  최신 운영체제는 대체로 <strong>페이징 기반</strong> 가상 메모리를 사용하므로, 현재 구조를 과거 방식처럼 이해하게 됩니다.
                </div>
              </div>
              <div className="vf-error-block vf-error-block-separated">
                <div className="vf-error-head">
                  <span className="vf-type-badge confuse">혼동 가능 설명</span>
                  <span className="vf-card-issue">페이지 폴트는 메모리 부족 오류입니다.</span>
                </div>
                <div className="vf-card-reason">
                  페이지 폴트는 참조한 페이지가 없거나 권한이 맞지 않을 때 발생하며, <strong>메모리 부족</strong>과는 원인이 다릅니다.
                </div>
              </div>
              <div className="vf-error-block vf-error-block-separated">
                <div className="vf-error-head">
                  <span className="vf-type-badge scope">과도한 일반화</span>
                  <span className="vf-card-issue">세그먼트 베이스와 리미트까지 외우면 됩니다.</span>
                </div>
                <div className="vf-card-reason">
                  이번 강의는 페이징 중심의 가상 메모리 흐름을 다루므로, <strong>세그멘테이션</strong> 세부 구조는 범위를 벗어납니다.
                </div>
              </div>
            </div>

            {/* 슬라이드 오타 카드 */}
            <div className="vf-card">
              <div className="vf-typo-stack">
                <div className="vf-typo-item">
                  <div className="vf-typo-head">슬라이드 3</div>
                  <div className="vf-typo-row">
                    <div className="vf-slide-preview">
                      <div className="vf-slide-mini-head">
                        <span className="vf-slide-mini-num">3</span>
                        <span className="vf-slide-mini-line" />
                      </div>
                      <div className="vf-slide-inner">
                        <div className="vf-slide-heading">프로세스 스케줄링 실습</div>
                        <div className="vf-slide-text">· FCFS와 RR 비교</div>
                        <div className="vf-slide-text">· <span className="vf-slide-word">pytho</span> 코드 실행</div>
                        <div className="vf-slide-text">· 평균 <span className="vf-slide-word">대긱</span> 시간 분석</div>
                      </div>
                    </div>
                    <div className="vf-typo-wrap">
                      <div className="vf-typo-content">
                        <div className="vf-typo-title">오타 목록</div>
                        <div className="vf-typo-fix">
                          <span className="vf-typo-before">pytho</span>
                          <span className="vf-typo-after">→ python</span>
                        </div>
                        <div className="vf-typo-fix">
                          <span className="vf-typo-before">대긱</span>
                          <span className="vf-typo-after">→ 대기</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="vf-typo-item">
                  <div className="vf-typo-head">슬라이드 8</div>
                  <div className="vf-typo-row">
                    <div className="vf-slide-preview">
                      <div className="vf-slide-mini-head">
                        <span className="vf-slide-mini-num">8</span>
                        <span className="vf-slide-mini-line" />
                      </div>
                      <div className="vf-slide-inner">
                        <div className="vf-slide-heading">프로세스와 스레드</div>
                        <div className="vf-slide-text">· <span className="vf-slide-word">프로쎄스</span>는 실행 프로그램</div>
                        <div className="vf-slide-text">· 스레드는 실행 흐름의 단위</div>
                        <div className="vf-slide-text">· 주소 공간 공유 비교</div>
                      </div>
                    </div>
                    <div className="vf-typo-wrap">
                      <div className="vf-typo-content">
                        <div className="vf-typo-title">오타 목록</div>
                        <div className="vf-typo-fix">
                          <span className="vf-typo-before">프로쎄스</span>
                          <span className="vf-typo-after">→ 프로세스</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
