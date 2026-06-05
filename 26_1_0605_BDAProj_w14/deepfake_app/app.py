# app.py — 프로젝트 진입점
# 5주차 영화 대시보드와 동일한 멀티페이지 구조(st.Page + st.navigation).
# 실행:  streamlit run app.py
import streamlit as st

st.set_page_config(
    page_title="딥페이크 이미지 판별",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 페이지 정의 (pages/ 폴더의 파일들)
eda = st.Page("pages/1_EDA.py", title="EDA", icon="📊", default=True)
viz = st.Page("pages/2_시각화.py", title="시각화", icon="📈")
service = st.Page("pages/3_모델_서비스.py", title="모델·서비스", icon="🕵️")

pg = st.navigation({
    "딥페이크 판별 프로젝트": [eda, viz, service],
})

# 사이드바 공통 영역
st.sidebar.markdown("### 🕵️ 딥페이크 이미지 판별")
st.sidebar.caption("이미지가 AI 합성(FAKE)인지 실제(REAL)인지 분류")
st.sidebar.markdown("---")
st.sidebar.caption("이름 / 학번")   # TODO: 본인 정보로 변경

pg.run()
