# pages/2_시각화.py — 2차 작업: 그래프로 인사이트 찾기
# 4주차에서 배운 plotly/altair 를 씁니다. 그래프마다 "그래서 무엇"을 한 줄 적으세요.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.express as px

from src.data_loader import load_data
from src.features import url_domain

st.title("📈 시각화")

df = load_data()
df = df.assign(domain=url_domain(df))   # 출처 도메인 파생 컬럼 추가

# 분석에 쓸 컬럼을 사용자가 고르게
cols = df.columns.tolist()

st.header("그래프 1 — 분포")
col1 = st.selectbox("볼 컬럼", cols, index=cols.index("label"), key="hist")
fig1 = px.histogram(df, x=col1, color="label", title=f"{col1} 분포")
st.plotly_chart(fig1, use_container_width=True)
st.caption("해석: (이 그래프에서 무엇을 알 수 있나? 한 줄)")  # TODO

st.header("그래프 2 — 관계")
c1, c2 = st.columns(2)
x = c1.selectbox("X축", cols, index=cols.index("age_group"), key="x")
y = c2.selectbox("Y축", cols, index=cols.index("image_quality"), key="y")
fig2 = px.histogram(df, x=x, color=y, barmode="group", title=f"{x} × {y}")
st.plotly_chart(fig2, use_container_width=True)
st.caption("해석: (X와 Y 사이에 관계가 보이나? 한 줄)")  # TODO

# TODO: 출처 도메인(domain)별 FAKE/REAL 비율, confidence_score 분포 등을 추가하면 인사이트가 잘 보입니다.
#   px.histogram(df, x="domain", color="label")
