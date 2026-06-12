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
viz = st.Page("pages/2_시각화.py", title="시각화", icon="📊")
service = st.Page("pages/3_모델_서비스.py", title="모델·서비스", icon="🤖")

pg = st.navigation({
    "AI 활용 이미지 판별 프로젝트": [eda, viz, service],
})

# 사이드바 공통 영역
st.sidebar.markdown("### 🤖 AI 활용 이미지 판별")
st.sidebar.caption("이미지 제작 과정에 AI가 활용되었을 가능성을 분류")
st.sidebar.markdown("#### 페이지 구성")
st.sidebar.markdown(
    """
    <style>
    .sidebar-page-card {
        border: 1px solid rgba(120, 120, 120, 0.24);
        border-radius: 12px;
        padding: 0.7rem 0.85rem;
        margin: 0.45rem 0;
        background: rgba(120, 120, 120, 0.07);
    }
    .sidebar-page-card strong {
        font-size: 0.96rem;
    }
    .sidebar-page-card span {
        color: rgba(90, 90, 90, 0.92);
        font-size: 0.82rem;
    }
    </style>
    <div class="sidebar-page-card">
        <strong>🔎 EDA</strong><br>
        <span>데이터·결측·샘플 확인</span>
    </div>
    <div class="sidebar-page-card">
        <strong>📊 시각화</strong><br>
        <span>분포·인사이트 확인</span>
    </div>
    <div class="sidebar-page-card">
        <strong>🤖 모델·서비스</strong><br>
        <span>이미지 판별·평가</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("#### 제출 체크리스트")
st.sidebar.success("EDA / 시각화 / 모델 서비스 / 보고서 자동 반영 구성 완료")
st.sidebar.markdown("---")

pg.run()
