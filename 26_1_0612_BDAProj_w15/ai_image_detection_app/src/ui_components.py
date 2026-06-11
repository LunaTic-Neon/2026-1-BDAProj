import streamlit as st


def render_workflow_card(title: str, input_text: str, action_text: str, output_text: str) -> None:
    st.markdown(f"#### {title}")
    c1, c2, c3 = st.columns(3)
    c1.info(f"입력\n\n{input_text}")
    c2.warning(f"작업\n\n{action_text}")
    c3.success(f"결과물\n\n{output_text}")


def render_project_notice() -> None:
    st.info(
        "이 프로젝트는 이미지 제작 과정에 AI가 활용되었을 가능성을 판별하는 서비스입니다. "
        "데이터 특성상 영상 기반 탐지가 아니라 실사 사진과 생성·합성·아바타 계열 이미지의 구분 문제로 해석합니다."
    )


def render_leakage_warning() -> None:
    st.warning(
        "`category`, `source`, `fake_method`, `detection_difficulty`, `domain`은 라벨과 직접 연결될 수 있는 누수 가능 컬럼입니다. "
        "이 컬럼들은 EDA와 보고서 분석용으로만 사용하고 모델 입력에는 사용하지 않습니다."
    )


def render_report_tip(text: str) -> None:
    with st.expander("보고서에 쓸 수 있는 핵심 문장", expanded=False):
        st.write(text)


def render_story_insight(title: str, body: str) -> None:
    st.success(f"**인사이트 — {title}**\n\n{body}")


def render_page_footer() -> None:
    st.caption(
        "결과 해석 시 URL 출처 편향, 이미지 품질, 사전학습 모델의 도메인 차이를 함께 고려해 주세요."
    )