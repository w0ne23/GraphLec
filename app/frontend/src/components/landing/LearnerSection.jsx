export default function LearnerSection() {
  return (
    <>
      <div className="feat-header">
        <p className="section-label">학습자 기능</p>
        <h2 className="section-title">질문하고, 추천받고, 학습하다</h2>
        <p className="section-sub">학습자는 강의에서 궁금한 점을 질문하거나, 자신에게 맞는 강의를 추천받을 수 있습니다.</p>
      </div>

      <div className="features-grid-2">
        {/* QnA 카드 */}
        <div className="feature-card learner-card">
          <div className="feature-card-head">
            <div className="feature-icon-wrap feature-icon-wrap-inline">
              <svg className="feature-icon" viewBox="0 0 24 24">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h3 className="feature-name feature-name-inline">QnA</h3>
          </div>

          <div className="qna-demo qna-demo-compact">
            <div className="chat-bubble-wrap user">
              <div className="chat-bubble">응용소프트웨어와 운영체제 사이의 화살표는 뭘 의미해?</div>
            </div>

            <div className="chat-bubble-wrap bot">
              <div className="chat-bubble chat-bubble-full">
                <div className="chat-answer-text">
                  화살표는 <strong>두 계층 간의 상호작용</strong>을 의미합니다. 응용 소프트웨어는 운영체제에게 요청을 보내고, 운영체제는 공통 서비스와 하드웨어 자원을 제공합니다.
                </div>
                <div className="chat-evidence">
                  <div className="chat-evidence-label">영상 구간</div>
                  <div className="scene-tags">
                    <span className="scene-tag">슬라이드 4 · Scene4</span>
                    <span className="scene-tag">슬라이드 9 · Scene12</span>
                  </div>
                  <div className="chat-evidence-label chat-evidence-label-graph">
                    근거 그래프
                  </div>
                  <div className="mini-graph-wrap">
                    <svg width="100%" height="120" viewBox="0 0 360 160" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
                      <defs>
                        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                          <path d="M0,0 L6,3 L0,6 Z" fill="#8E8FC7" opacity="0.7"/>
                        </marker>
                        <marker id="arr-y" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                          <path d="M0,0 L6,3 L0,6 Z" fill="#E6A817" opacity="0.8"/>
                        </marker>
                      </defs>
                      <line x1="162" y1="77" x2="196" y2="77" stroke="#8E8FC7" strokeWidth="1.5" markerEnd="url(#arr)" opacity="0.8"/>
                      <line x1="196" y1="84" x2="162" y2="84" stroke="#8E8FC7" strokeWidth="1.5" markerEnd="url(#arr)" opacity="0.5"/>
                      <line x1="185" y1="38" x2="205" y2="58" stroke="#8E8FC7" strokeWidth="1" markerEnd="url(#arr)" opacity="0.5"/>
                      <line x1="124" y1="70" x2="92" y2="52" stroke="#8E8FC7" strokeWidth="1" markerEnd="url(#arr)" opacity="0.35"/>
                      <line x1="130" y1="100" x2="112" y2="120" stroke="#8E8FC7" strokeWidth="1" markerEnd="url(#arr)" opacity="0.4"/>
                      <line x1="222" y1="100" x2="230" y2="120" stroke="#8E8FC7" strokeWidth="1" markerEnd="url(#arr)" opacity="0.4"/>
                      <line x1="158" y1="90" x2="152" y2="118" stroke="#E6A817" strokeWidth="1.2" markerEnd="url(#arr-y)" opacity="0.7"/>
                      <line x1="204" y1="98" x2="178" y2="122" stroke="#E6A817" strokeWidth="1.2" markerEnd="url(#arr-y)" opacity="0.7"/>
                      <line x1="237" y1="72" x2="278" y2="48" stroke="#6DC8B0" strokeWidth="1" opacity="0.6" markerEnd="url(#arr)"/>
                      <line x1="238" y1="82" x2="278" y2="108" stroke="#6DC8B0" strokeWidth="1" opacity="0.6" markerEnd="url(#arr)"/>
                      <line x1="192" y1="22" x2="278" y2="20" stroke="#6DC8B0" strokeWidth="1" opacity="0.5" markerEnd="url(#arr)"/>
                      <circle cx="76" cy="42" r="16" fill="#E8647A" opacity="0.55"/>
                      <text x="76" y="39" textAnchor="middle" fontSize="6.5" fill="white" fontFamily="Noto Sans KR">응용SW</text>
                      <text x="76" y="47" textAnchor="middle" fontSize="6.5" fill="white" fontFamily="Noto Sans KR">박스</text>
                      <circle cx="178" cy="22" r="14" fill="#E8647A" opacity="0.7"/>
                      <text x="178" y="26" textAnchor="middle" fontSize="7" fill="white" fontFamily="Noto Sans KR">사용자</text>
                      <circle cx="140" cy="80" r="22" fill="#E8647A" opacity="0.9"/>
                      <text x="140" y="84" textAnchor="middle" fontSize="8" fill="white" fontFamily="Noto Sans KR">운영체제</text>
                      <circle cx="216" cy="80" r="22" fill="#E8647A" opacity="0.9"/>
                      <text x="216" y="77" textAnchor="middle" fontSize="7.5" fill="white" fontFamily="Noto Sans KR">응용</text>
                      <text x="216" y="87" textAnchor="middle" fontSize="7.5" fill="white" fontFamily="Noto Sans KR">소프트웨어</text>
                      <circle cx="166" cy="132" r="15" fill="#E6A817" opacity="0.88"/>
                      <text x="166" y="129" textAnchor="middle" fontSize="7" fill="white" fontFamily="Noto Sans KR">화살표</text>
                      <circle cx="100" cy="132" r="13" fill="#E8647A" opacity="0.6"/>
                      <text x="100" y="136" textAnchor="middle" fontSize="7" fill="white" fontFamily="Noto Sans KR">커널</text>
                      <circle cx="234" cy="132" r="13" fill="#E8647A" opacity="0.6"/>
                      <text x="234" y="136" textAnchor="middle" fontSize="7" fill="white" fontFamily="Noto Sans KR">프로세스</text>
                      <circle cx="292" cy="18" r="11" fill="#6DC8B0" opacity="0.85"/>
                      <text x="292" y="22" textAnchor="middle" fontSize="6.5" fill="white">slide9</text>
                      <circle cx="292" cy="52" r="11" fill="#6DC8B0" opacity="0.85"/>
                      <text x="292" y="56" textAnchor="middle" fontSize="6.5" fill="white">slide4</text>
                      <circle cx="292" cy="110" r="11" fill="#6DC8B0" opacity="0.85"/>
                      <text x="292" y="114" textAnchor="middle" fontSize="6.5" fill="white">slide8</text>
                    </svg>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="feature-tags feature-tags-qna">
            <span className="tag">가중치 지식그래프</span>
            <span className="tag">자연어 질의</span>
            <span className="tag">근거 영상 구간</span>
          </div>
        </div>

        {/* Recommend 카드 */}
        <div className="feature-card learner-card">
          <div className="feature-card-head">
            <div className="feature-icon-wrap feature-icon-wrap-inline">
              <svg className="feature-icon" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12" />
              </svg>
            </div>
            <h3 className="feature-name feature-name-inline">Recommend</h3>
          </div>

          <div className="rec-demo">
            <div className="rec-search-bar">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#8E8FC7" strokeWidth="2" strokeLinecap="round">
                <circle cx="11" cy="11" r="8" />
                <path d="M21 21l-4.35-4.35" />
              </svg>
              <span className="rec-search-text">프로세스와 CPU의 관계를 설명해주는 강의 추천해줘</span>
              <button className="rec-search-btn">검색</button>
            </div>

            <div className="rec-result-label">
              <strong>"프로세스와 CPU"</strong>에 대한 추천 강의 5개
            </div>

            <div className="rec-card top">
              <div className="rec-card-header">
                <div className="rec-thumb">30:00</div>
                <div className="rec-info">
                  <div className="rec-tags">
                    <span className="rec-tag">#프로세스</span>
                    <span className="rec-tag">#cpu</span>
                  </div>
                  <div className="rec-title">프로세스 관리와 상태 전이</div>
                </div>
                <div className="rec-score">75점</div>
              </div>
              <div className="rec-metrics">
                <div className="rec-metrics-title">AI 세부 분석 지표</div>
                <div className="metric-row">
                  <span className="metric-label">내용 점수</span>
                  <div className="metric-bar-bg"><div className="metric-bar content metric-w-64" /></div>
                  <span className="metric-val">64</span>
                </div>
                <div className="metric-row sub">
                  <span className="metric-label">의미 유사도</span>
                  <div className="metric-bar-bg"><div className="metric-bar sim metric-w-81" /></div>
                  <span className="metric-val">81</span>
                </div>
                <div className="metric-row sub">
                  <span className="metric-label">직접 매칭</span>
                  <div className="metric-bar-bg"><div className="metric-bar direct metric-w-17" /></div>
                  <span className="metric-val">17</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">개념 그래프</span>
                  <div className="metric-bar-bg"><div className="metric-bar graph metric-w-58" /></div>
                  <span className="metric-val">58</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">조건 부스트</span>
                  <div className="metric-bar-bg"><div className="metric-bar boost metric-w-40" /></div>
                  <span className="metric-val">40</span>
                </div>
              </div>
            </div>

            <div className="rec-card-mini">
              <div className="rec-thumb rec-thumb-mini">55:00</div>
              <div className="rec-info">
                <div className="rec-tags"><span className="rec-tag">#운영체제</span></div>
                <div className="rec-title">운영체제 개론</div>
              </div>
              <div className="rec-score">64점</div>
            </div>

            <div className="rec-more-wrap">
              <div className="rec-more-dots">• • •</div>
            </div>
          </div>

          <div className="feature-tags feature-tags-recommend">
            <span className="tag">벡터 검색</span>
            <span className="tag">자연어 질의</span>
            <span className="tag">전체 강의 점수화</span>
          </div>
        </div>
      </div>
    </>
  )
}