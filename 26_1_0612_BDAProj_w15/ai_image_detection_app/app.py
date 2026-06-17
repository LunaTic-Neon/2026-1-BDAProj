# app.py — 프로젝트 진입점
# 5주차 영화 대시보드와 동일한 멀티페이지 구조(st.Page + st.navigation).
# 실행:  streamlit run app.py
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="AI 활용 이미지 판별",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 페이지 정의 (pages/ 폴더의 파일들)
eda = st.Page("pages/1_EDA.py", title="EDA", icon="🔎", default=True)
viz = st.Page("pages/2_시각화.py", title="시각화", icon="📈")
service = st.Page("pages/3_모델_서비스.py", title="모델·서비스", icon="🤖")

REPORT_PATH = Path(__file__).resolve().parents[1] / "보고서.md"
report_text = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "보고서.md 파일이 없습니다."
report_bytes = report_text.encode("utf-8-sig")

top_left, top_view, top_down = st.columns([0.72, 0.14, 0.14])
with top_view:
    if hasattr(st, "popover"):
        with st.popover("보고서 보기", use_container_width=True):
            st.markdown(report_text)
    else:
        with st.expander("보고서 보기"):
            st.markdown(report_text)
with top_down:
    st.download_button(
        "보고서 다운로드",
        data=report_bytes,
        file_name="보고서.md",
        mime="text/markdown",
        use_container_width=True,
    )

pg = st.navigation({
    "AI 활용 이미지 판별 프로젝트": [eda, viz, service],
})

# 사이드바 공통 영역
st.sidebar.markdown("### AI 활용 이미지 판별")
st.sidebar.caption("평가기준에 맞춘 핵심 기능 중심 구성")
st.sidebar.markdown("#### 페이지 구성")
st.sidebar.markdown(
    """
    <div style="border:1px solid rgba(120,120,120,.24);border-radius:12px;padding:.7rem .85rem;margin:.45rem 0;background:rgba(120,120,120,.07);">
        <strong>🔎 EDA</strong><br><span>데이터 분포·누수·샘플 확인</span>
    </div>
    <div style="border:1px solid rgba(120,120,120,.24);border-radius:12px;padding:.7rem .85rem;margin:.45rem 0;background:rgba(120,120,120,.07);">
        <strong>📈 시각화</strong><br><span>핵심 인사이트 그래프</span>
    </div>
    <div style="border:1px solid rgba(120,120,120,.24);border-radius:12px;padding:.7rem .85rem;margin:.45rem 0;background:rgba(120,120,120,.07);">
        <strong>🤖 모델·서비스</strong><br><span>이미지 판별·결과 확인</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.success("EDA / 시각화 / 모델 서비스 중심으로 정리됨")
with st.sidebar.expander("보고서.md 보기"):
    st.markdown(report_text)
st.sidebar.download_button(
    "보고서.md 다운로드",
    data=report_bytes,
    file_name="보고서.md",
    mime="text/markdown",
    use_container_width=True,
)
st.sidebar.markdown("---")

pg.run()
