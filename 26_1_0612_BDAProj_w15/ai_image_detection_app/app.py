# app.py — 프로젝트 진입점
# 5주차 영화 대시보드와 동일한 멀티페이지 구조(st.Page + st.navigation).
# 실행:  streamlit run app.py
import streamlit as st

st.set_page_config(
    page_title="AI 활용 이미지 판별",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 페이지 정의 (pages/ 폴더의 파일들)
eda = st.Page("pages/1_EDA.py", title="EDA", icon="📊", default=True)
viz = st.Page("pages/2_시각화.py", title="시각화", icon="📈")
service = st.Page("pages/3_모델_서비스.py", title="모델·서비스", icon="🕵️")

pg = st.navigation({
    "AI 활용 이미지 판별 프로젝트": [eda, viz, service],
})

# 사이드바 공통 영역
st.sidebar.markdown("### 🕵️ AI 활용 이미지 판별")
st.sidebar.caption("이미지 제작 과정에 AI가 활용되었을 가능성을 분류")
st.sidebar.info("이 프로젝트는 영상 기반 탐지가 아니라 이미지 기반 AI 활용 가능성 판별에 가깝습니다.")
st.sidebar.markdown("#### 페이지 구성")
st.sidebar.write("1. EDA — 데이터·결측·샘플 확인")
st.sidebar.write("2. 시각화 — 인사이트·누수 진단")
st.sidebar.write("3. 모델·서비스 — 판별·평가·LLM 해설")
st.sidebar.markdown("#### 제출 체크리스트")
st.sidebar.success("EDA / 시각화 / 모델 서비스 / 보고서 자동 반영 구성 완료")
st.sidebar.markdown("---")
st.sidebar.caption("이름 / 학번")   # TODO: 본인 정보로 변경

pg.run()
