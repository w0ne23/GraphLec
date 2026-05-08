"""
Lecture Claim Verification Pipeline - Wireframe Slides
실행: python make_slides.py
출력: pipeline_wireframe.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree


# ============================================================
# 디자인 토큰 (가이드의 색상/타이포 시스템)
# ============================================================

# Colors
COLOR_STAGE_EXTRACT = RGBColor(0xFF, 0xFF, 0xFF)      # #ffffff
COLOR_STAGE_JUDGE = RGBColor(0xF5, 0xF5, 0xF7)        # #f5f5f7
COLOR_STAGE_CROSSCHECK = RGBColor(0x27, 0x27, 0x29)   # #272729

COLOR_PIPELINE_ACCENT = RGBColor(0x00, 0x66, 0xCC)    # #0066cc
COLOR_ACCENT_ON_DARK = RGBColor(0x29, 0x97, 0xFF)     # #2997ff

COLOR_HEADING = RGBColor(0x1D, 0x1D, 0x1F)            # #1d1d1f
COLOR_BODY = RGBColor(0x1D, 0x1D, 0x1F)
COLOR_BODY_ON_DARK = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_COMMENT = RGBColor(0x7A, 0x7A, 0x7A)            # #7a7a7a
COLOR_HAIRLINE = RGBColor(0xE0, 0xE0, 0xE0)           # #e0e0e0
COLOR_CODE_LIGHT = RGBColor(0xFA, 0xFA, 0xFC)         # #fafafc

# Fonts
FONT_DISPLAY = "SF Pro Display"
FONT_DISPLAY_FALLBACK = "Helvetica Neue"  # SF Pro 없을 때
FONT_MONO = "SF Mono"
FONT_MONO_FALLBACK = "Menlo"

# Slide size: 16:9, 1920x1080 (EMU 환산: 1 inch = 914400 EMU)
SLIDE_WIDTH = Inches(20)   # 1920px @ 96dpi
SLIDE_HEIGHT = Inches(11.25)  # 1080px


# ============================================================
# 헬퍼 함수
# ============================================================

def set_slide_background(slide, color):
    """슬라이드 배경색 설정"""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text,
                 font_name=FONT_DISPLAY, font_size=17, bold=False,
                 color=COLOR_BODY, align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP, letter_spacing=0):
    """텍스트 박스 추가. letter_spacing은 100 단위 EMU (음수 가능)"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.word_wrap = True
    tf.vertical_anchor = anchor

    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color

    # letter-spacing은 python-pptx 기본 API에 없어서 XML 직접 조작
    if letter_spacing != 0:
        rPr = run._r.get_or_add_rPr()
        rPr.set('spc', str(letter_spacing))

    return tb


def add_pill_badge(slide, left, top, text, bg_color, text_color,
                   font_size=13, padding_h=Inches(0.18), height=Inches(0.32)):
    """Pill 형태 배지 (stage badge)"""
    # 텍스트 길이 추정으로 폭 계산 (간단 휴리스틱)
    estimated_width = Inches(0.12 * len(text) + 0.4)

    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, estimated_width, height
    )
    # 완전한 pill을 위해 corner radius 최대
    pill.adjustments[0] = 0.5

    pill.fill.solid()
    pill.fill.fore_color.rgb = bg_color
    pill.line.fill.background()  # 보더 제거

    tf = pill.text_frame
    tf.margin_left = padding_h
    tf.margin_right = padding_h
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = FONT_DISPLAY
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = text_color
    rPr = run._r.get_or_add_rPr()
    rPr.set('spc', '80')  # 약간의 letter-spacing

    return pill


def add_stage_card(slide, left, top, width, height,
                   stage_num, stage_name, description,
                   active=False):
    """Stage indicator 카드 (overview 슬라이드용)"""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    # rounded.lg = 18px ≈ 0.1875 inch
    card.adjustments[0] = 0.08

    if active:
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_PIPELINE_ACCENT
        card.line.fill.background()
        # Shadow는 python-pptx에서 제한적이라 생략 (실제 PPT에서 수동 추가 가능)
        text_color = COLOR_BODY_ON_DARK
        sub_color = RGBColor(0xCC, 0xE0, 0xF5)
    else:
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CODE_LIGHT
        card.line.color.rgb = COLOR_HAIRLINE
        card.line.width = Pt(1)
        text_color = COLOR_HEADING
        sub_color = COLOR_COMMENT

    # 카드 안 텍스트 비우기
    card.text_frame.text = ""

    pad = Inches(0.42)
    inner_left = left + pad
    inner_top = top + pad
    inner_width = width - 2 * pad

    # Stage number/label (작은 라벨)
    add_text_box(
        slide, inner_left, inner_top, inner_width, Inches(0.3),
        f"STAGE {stage_num}",
        font_size=13, bold=True, color=sub_color,
        letter_spacing=120,
    )

    # Stage name (큰 타이틀)
    add_text_box(
        slide, inner_left, inner_top + Inches(0.4),
        inner_width, Inches(0.7),
        stage_name,
        font_size=32, bold=True, color=text_color,
        letter_spacing=-30,
    )

    # Description
    add_text_box(
        slide, inner_left, inner_top + Inches(1.25),
        inner_width, height - Inches(1.5),
        description,
        font_size=15, color=text_color,
    )


def add_arrow(slide, left, top, width, color=COLOR_PIPELINE_ACCENT):
    """가로 화살표 (stage 사이 연결)"""
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, left, top, width, Inches(0.3)
    )
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    return arrow


def add_slide_footer(slide, current_stage_text, page_num, total_pages,
                     on_dark=False):
    """모든 슬라이드 하단 푸터"""
    color = COLOR_BODY_ON_DARK if on_dark else COLOR_COMMENT
    # 좌측 푸터
    add_text_box(
        slide, Inches(0.83), Inches(10.6),
        Inches(8), Inches(0.3),
        current_stage_text,
        font_size=11, color=color,
    )
    # 우측 페이지 번호
    add_text_box(
        slide, Inches(18), Inches(10.6),
        Inches(1.2), Inches(0.3),
        f"{page_num} / {total_pages}",
        font_size=11, color=color, align=PP_ALIGN.RIGHT,
    )


# ============================================================
# 슬라이드 1: Cover
# ============================================================

def make_cover_slide(prs):
    blank_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_STAGE_EXTRACT)

    # 좌상단 작은 라벨
    add_text_box(
        slide, Inches(0.83), Inches(0.83),
        Inches(8), Inches(0.3),
        "LECTURE QUALITY ASSURANCE · TECHNICAL DECK",
        font_size=13, bold=True, color=COLOR_PIPELINE_ACCENT,
        letter_spacing=180,
    )

    # 메인 타이틀 (중앙 약간 위)
    add_text_box(
        slide, Inches(0.83), Inches(4.2),
        Inches(18.3), Inches(1.5),
        "Lecture Claim",
        font_size=64, bold=True, color=COLOR_HEADING,
        letter_spacing=-32,
    )
    add_text_box(
        slide, Inches(0.83), Inches(5.3),
        Inches(18.3), Inches(1.5),
        "Verification Pipeline",
        font_size=64, bold=True, color=COLOR_HEADING,
        letter_spacing=-32,
    )

    # 서브 타이틀
    add_text_box(
        slide, Inches(0.83), Inches(6.7),
        Inches(18.3), Inches(0.8),
        "강의 발화에서 검증 가능한 사실 주장을 추출하고, 3단계 게이트를 거쳐 교수에게 전달할 후보를 선별합니다.",
        font_size=22, color=COLOR_HEADING,
        letter_spacing=-15,
    )

    # 하단 화살표 장식 (단일 강조 요소)
    arrow_y = Inches(8.8)
    add_arrow(slide, Inches(0.83), arrow_y, Inches(1.2))

    add_text_box(
        slide, Inches(2.2), arrow_y - Inches(0.05),
        Inches(8), Inches(0.4),
        "Extract  →  Judge  →  Crosscheck",
        font_size=15, bold=True, color=COLOR_PIPELINE_ACCENT,
        letter_spacing=80,
    )

    # 푸터
    add_slide_footer(slide, "Lecture Claim Verifier · Cover", 1, 20)


# ============================================================
# 슬라이드 3: Pipeline Overview
# ============================================================

def make_overview_slide(prs):
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_STAGE_JUDGE)

    # Stage badge (좌상단)
    add_pill_badge(
        slide, Inches(0.83), Inches(0.83),
        "OVERVIEW",
        bg_color=COLOR_PIPELINE_ACCENT,
        text_color=COLOR_BODY_ON_DARK,
    )

    # 메인 헤드라인
    add_text_box(
        slide, Inches(0.83), Inches(1.7),
        Inches(18.3), Inches(1.0),
        "3단계 파이프라인",
        font_size=44, bold=True, color=COLOR_HEADING,
        letter_spacing=-30,
    )

    # 서브 텍스트
    add_text_box(
        slide, Inches(0.83), Inches(2.85),
        Inches(18.3), Inches(0.6),
        "각 단계는 독립적으로 동작하며, 점점 더 엄격한 필터로 후보를 좁혀갑니다.",
        font_size=20, color=COLOR_HEADING,
        letter_spacing=-15,
    )

    # 3개의 Stage Card
    card_top = Inches(4.5)
    card_height = Inches(4.5)
    card_width = Inches(5.5)
    gap_between = Inches(0.6)
    arrow_width = Inches(0.6)

    # 좌측 여백 계산: (전체 20" - 3*5.5 - 2*0.6 - 2*0.6) / 2
    total_content = 3 * 5.5 + 2 * 0.6 + 2 * 0.6  # 17.7"
    left_margin = Inches((20 - total_content) / 2)  # 1.15"

    # Stage 1: Extract
    add_stage_card(
        slide, left_margin, card_top, card_width, card_height,
        stage_num=1,
        stage_name="Extract",
        description=(
            "강의 발화에서 검증 가능한 \n"
            "사실 주장(claim)의 raw inventory를 추출합니다.\n\n"
            "• definition / numeric / causal\n"
            "• relationship / currentness\n"
            "• 인접 조각 발화 자동 병합"
        ),
        active=False,
    )

    # 화살표 1
    arrow1_left = left_margin + card_width + Inches(0.3)
    add_arrow(slide, arrow1_left, card_top + Inches(2.1), arrow_width)

    # Stage 2: Judge (활성 강조)
    stage2_left = arrow1_left + arrow_width + Inches(0.3)
    add_stage_card(
        slide, stage2_left, card_top, card_width, card_height,
        stage_num=2,
        stage_name="Judge",
        description=(
            "추출된 claim을 LLM으로 판정하여 \n"
            "이슈 후보만 좁혀냅니다.\n\n"
            "• 3-Gate 보수적 선별\n"
            "• 4가지 issue type\n"
            "• confidence ≥ 0.80 임계값"
        ),
        active=True,
    )

    # 화살표 2
    arrow2_left = stage2_left + card_width + Inches(0.3)
    add_arrow(slide, arrow2_left, card_top + Inches(2.1), arrow_width)

    # Stage 3: Crosscheck
    stage3_left = arrow2_left + arrow_width + Inches(0.3)
    add_stage_card(
        slide, stage3_left, card_top, card_width, card_height,
        stage_num=3,
        stage_name="Crosscheck",
        description=(
            "5개 채점 기준으로 독립 재평가하여 \n"
            "최종 verdict를 결정합니다.\n\n"
            "• Criteria-based scoring\n"
            "• 3개 cap gate 적용\n"
            "• Reject / Review / Confirmed"
        ),
        active=False,
    )

    # 하단 결과 라벨
    result_top = Inches(9.4)
    add_text_box(
        slide, Inches(0.83), result_top,
        Inches(18.3), Inches(0.4),
        "최종 출력: 기각 (0.00–0.44)  ·  교수 확인 (0.45–0.79)  ·  확정 (0.80–1.00)",
        font_size=15, bold=True, color=COLOR_COMMENT,
        align=PP_ALIGN.CENTER,
        letter_spacing=20,
    )

    # 푸터
    add_slide_footer(slide, "Lecture Claim Verifier · Overview", 3, 20)


# ============================================================
# 메인
# ============================================================

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT

    make_cover_slide(prs)
    # 슬라이드 2번은 와이어프레임 범위 밖이지만 페이지 번호를 위해 빈 자리만 인지
    make_overview_slide(prs)

    output_path = "pipeline_wireframe.pptx"
    prs.save(output_path)
    print(f"✓ 생성 완료: {output_path}")
    print(f"  슬라이드: 2장 (Cover + Pipeline Overview)")
    print(f"  크기: 1920 × 1080 (16:9)")


if __name__ == "__main__":
    main()
