# pages/1_EDA.py — 이미지 EDA (CSV+URL 데이터용)
# 템플릿 1_EDA_image.py(로컬 폴더 스캔)를 이 데이터(CSV+image_url)에 맞게 교체한 것.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.express as px

from src.data_loader import load_data, fetch_image, LEAKAGE_COLS

st.title("📊 이미지 EDA")

df = load_data()

# ① 개요 ----------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("총 이미지", f"{len(df):,}")
c2.metric("클래스 수", df["label"].nunique())
c3.metric("결측 합계", int(df.isnull().sum().sum()))

with st.expander("데이터 미리보기 (상위 10행)"):
    st.dataframe(df.head(10), use_container_width=True)

# ② 클래스 분포 — 불균형 확인 (가장 중요) -------------------------------
st.header("1. 클래스별 장수 (label)")
st.bar_chart(df["label"].value_counts())
st.caption("FAKE 3,767 / REAL 2,790 — 약한 불균형(57:43).")

# ③ 클래스별 샘플 이미지 — 이미지의 describe() --------------------------
st.header("2. 클래스별 샘플")
st.caption("image_url에서 직접 받아 표시합니다(원격). 느리면 새로고침을 줄이세요.")
n_sample = st.slider("클래스당 표시 장수", 1, 5, 3)
for label in sorted(df["label"].unique()):
    st.markdown(f"**{label}**")
    cols = st.columns(n_sample)
    sample = df[df["label"] == label].head(n_sample)
    for col, (_, r) in zip(cols, sample.iterrows()):
        img = fetch_image(r["image_url"])
        if img is not None:
            col.image(img, use_container_width=True)
        else:
            col.warning("로드 실패")

# ④ 결측치 --------------------------------------------------------------
st.header("3. 결측치")
na = df.isnull().sum()
na = na[na > 0]
if len(na):
    st.bar_chart(na)
    st.caption("fake_method 결측은 REAL 행(변조 기법 없음)에서 발생합니다.")
else:
    st.success("결측치 없음")

# ⑤ 데이터 누수(leakage) 컬럼 — 학습에 쓰면 안 됨 ----------------------
st.header("4. ⚠️ 데이터 누수 컬럼")
st.warning(
    "아래 컬럼들은 label과 거의 1:1로 대응합니다. "
    "**모델 학습에는 쓰지 않고** EDA 분석용으로만 사용합니다."
)
for col in LEAKAGE_COLS:
    with st.expander(f"`{col}` × label 교차표"):
        st.dataframe(
            df.groupby([col, "label"]).size().unstack(fill_value=0),
            use_container_width=True,
        )

# TODO: 내가 발견한 것(불균형? 누수? 출처 편향?)을 글로 적으세요 — 보고서 재료
st.info("발견 예시: 'category·source가 label과 100% 일치 → 메타데이터로 풀면 안 됨, 이미지로 판별해야 함'")
