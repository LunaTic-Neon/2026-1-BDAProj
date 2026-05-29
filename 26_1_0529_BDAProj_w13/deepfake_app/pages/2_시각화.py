# pages/2_시각화.py — 2차 작업: 그래프로 인사이트 찾기
# 4주차에서 배운 plotly/altair 를 씁니다. 그래프마다 "그래서 무엇"을 한 줄 적으세요.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
 
import streamlit as st
import plotly.express as px
import pandas as pd

from src.data_loader import load_data
from src.features import url_domain

st.title("📈 시각화")

# 사이드바: 불러올 행 수
st.sidebar.header("데이터 옵션")
max_rows = st.sidebar.number_input("최대 불러올 행(nrows, 0=전체)", min_value=0, value=5000, step=500)

nrows = None if max_rows == 0 else int(max_rows)

df = load_data(nrows=nrows)

df = df.assign(domain=url_domain(df))   # 출처 도메인 파생 컬럼 추가

# 분석에 쓸 컬럼을 사용자가 고르게
cols = df.columns.tolist()

st.header("그래프 1 — 분포")
col1 = st.selectbox("볼 컬럼", cols, index=cols.index("label"), key="hist")
fig1 = px.histogram(df, x=col1, color="label", title=f"{col1} 분포")
st.plotly_chart(fig1, use_container_width=True)
st.text_input("해석(한 줄)", value="")

st.header("그래프 2 — 관계")
c1, c2 = st.columns(2)
x = c1.selectbox("X축", cols, index=cols.index("age_group"), key="x")
y = c2.selectbox("Y축", cols, index=cols.index("image_quality"), key="y")
fig2 = px.histogram(df, x=x, color=y, barmode="group", title=f"{x} × {y}")
st.plotly_chart(fig2, use_container_width=True)
st.text_input("해석(한 줄)", value="")

# 도메인별 FAKE 비율
st.header("도메인(출처)별 FAKE 비율")
if "domain" in df.columns:
    ratio = (df.groupby("domain")["label"].apply(lambda s: (s=="FAKE").mean()).reset_index(name="fake_ratio"))
    ratio = ratio.sort_values("fake_ratio", ascending=False)
    topn = st.slider("상위 도메인 개수", 3, 30, 10)
    st.plotly_chart(px.bar(ratio.head(topn), x="domain", y="fake_ratio", title="도메인별 FAKE 비율", labels={"fake_ratio":"FAKE 비율"}), use_container_width=True)
    st.caption("해석: 특정 도메인(예: dicebear 등)이 FAKE 비율이 높아 도메인 편향 가능성")
else:
    st.info("domain 파생이 불가능합니다 (image_url 형식 불일치).")

# confidence_score 분포
if "confidence_score" in df.columns:
    st.header("confidence_score 분포")
    st.plotly_chart(px.histogram(df, x="confidence_score", color="label", nbins=40), use_container_width=True)

# TODO: 상호작용(필터링) 추가 — age_group/quality 필터로 도메인별 비율 재계산
