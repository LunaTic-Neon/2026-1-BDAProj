# app.py — 프로젝트 진입점
# 5주차 영화 대시보드와 동일한 멀티페이지 구조(st.Page + st.navigation).
# 실행:  streamlit run app.py
import streamlit as st

st.set_page_config(
    page_title="AI 합성 이미지 판별",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 페이지 정의 (pages/ 폴더의 파일들)
eda = st.Page("pages/1_EDA.py", title="EDA", icon="📊", default=True)
viz = st.Page("pages/2_시각화.py", title="시각화", icon="📈")
service = st.Page("pages/3_모델_서비스.py", title="모델·서비스", icon="🕵️")

pg = st.navigation({
    "AI 합성 이미지 판별 프로젝트": [eda, viz, service],
})

# 사이드바 공통 영역
st.sidebar.markdown("### 🕵️ AI 합성 이미지 판별")
st.sidebar.caption("얼굴 이미지가 실사 사진인지 생성/아바타 계열 이미지인지 분류")
st.sidebar.info("이 프로젝트는 영상 딥페이크 탐지보다 이미지 기반 실사/생성 이미지 구분에 가깝습니다.")
st.sidebar.markdown("---")
st.sidebar.caption("이름 / 학번")   # TODO: 본인 정보로 변경

pg.run()
