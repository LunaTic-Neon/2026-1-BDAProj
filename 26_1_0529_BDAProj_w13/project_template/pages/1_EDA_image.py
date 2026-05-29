# 1_EDA_image.py — 이미지 EDA
# 사용법: 이 파일 내용을 pages/1_EDA.py 에 덮어쓰세요. (이미지_데이터_쓸_때/README.md 참고)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.express as px
from src.data_loader import load_image_meta   # data_loader_image.py 를 src/data_loader.py로 교체했다고 가정

st.title("📊 이미지 EDA")

df = load_image_meta()

if len(df) == 0:
    st.warning("data/images/<클래스명>/ 에 이미지가 없습니다. 이미지_데이터_쓸_때/README.md 를 확인하세요.")
    st.stop()

# ① 개요
c1, c2, c3 = st.columns(3)
c1.metric("총 이미지", f"{len(df):,}")
c2.metric("클래스 수", df["label"].nunique())
c3.metric("손상 파일", int((df["mode"] == "손상").sum()))

# ② 클래스 분포 — 불균형 확인 (가장 중요)
st.header("1. 클래스별 장수")
st.bar_chart(df["label"].value_counts())

# ③ 샘플 이미지를 눈으로 — 이미지의 describe()
st.header("2. 클래스별 샘플")
for label in sorted(df["label"].unique()):
    st.markdown(f"**{label}**")
    cols = st.columns(5)
    for col, (_, r) in zip(cols, df[df["label"] == label].head(5).iterrows()):
        col.image(r["path"], use_container_width=True)

# ④ 이미지 크기 분포 — resize 기준 정하기
st.header("3. 이미지 크기 분포")
fig = px.scatter(df, x="width", y="height", color="label", title="width × height")
st.plotly_chart(fig, use_container_width=True)

# ⑤ 컬러 모드 — RGB/흑백이 섞였으면 모델 넣기 전 통일 필요
st.header("4. 컬러 모드")
st.dataframe(df["mode"].value_counts())

# TODO: 내가 발견한 것 (불균형? 크기 제각각? 흑백 섞임?)을 글로 적으세요 — 보고서 재료
st.info("발견 예시: 'cat이 dog의 2배 → 불균형', '이미지 크기가 제각각 → 224로 통일 필요'")
