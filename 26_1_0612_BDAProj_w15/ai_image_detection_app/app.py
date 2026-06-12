# app.py — 프로젝트 진입점
# 5주차 영화 대시보드와 동일한 멀티페이지 구조(st.Page + st.navigation).
# 실행:  streamlit run app.py
import streamlit as st

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
st.sidebar.markdown("---")

pg.run()
