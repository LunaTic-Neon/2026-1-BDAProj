# pages/1_EDA.py — 이미지 EDA (CSV+URL 데이터용)
# 템플릿 1_EDA_image.py(로컬 폴더 스캔)를 이 데이터(CSV+image_url)에 맞게 교체한 것.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
 
import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image

from src.data_loader import load_data, fetch_image, LEAKAGE_COLS
from src.features import url_domain

st.title("📊 이미지 EDA")

# 사이드바: 몇 행 불러올지, 샘플 이미지 옵션
st.sidebar.header("데이터 로드 옵션")
max_rows = st.sidebar.number_input("최대 불러올 행(nrows, 0=전체)", min_value=0, value=5000, step=500)
use_local_samples = st.sidebar.checkbox("샘플 이미지 로컬 폴더 우선 사용(있을 경우)", value=True)
sample_local_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "sample_images"))

nrows = None if max_rows == 0 else int(max_rows)

df = load_data(nrows=nrows)

# 기본 메트릭
c1, c2, c3 = st.columns(3)
c1.metric("총 이미지", f"{len(df):,}")
c2.metric("클래스 수", df["label"].nunique())
c3.metric("결측 합계", int(df.isnull().sum().sum()))

with st.expander("데이터 미리보기 (상위 10행)"):
    st.dataframe(df.head(10), use_container_width=True)

# 라벨 분포(수 + 비율)
st.header("1. 클래스별 분포 (count + percent)")
vc = df["label"].value_counts().rename_axis("label").reset_index(name="count")
vc["percent"] = (vc["count"] / vc["count"].sum() * 100).round(2)
fig = px.bar(vc, x="label", y="count", text="percent", color="label", title="Label 분포 (count)")
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=True)
st.caption(f"샘플 비율 — {', '.join([f'{r.label}:{r.percent}%' for r in vc.itertuples()])}")

# 클래스별 샘플 이미지 표시
st.header("2. 클래스별 샘플 이미지")
st.caption("image_url에서 직접 받아 표시합니다(원격). 느리면 nrows를 줄이거나 로컬 샘플을 미리 준비하세요.")
show_per_class = st.slider("클래스당 표시 장수", 1, 8, 3)

# helper: 로컬 샘플 우선 사용 함수
def _local_sample_paths(label, k):
    # 폴더명이 label과 정확히 일치하지 않으면 모든 파일에서 label 포함되는 이름으로 약식 필터
    if not os.path.isdir(sample_local_dir):
        return []
    files = []
    for fn in os.listdir(sample_local_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            if label.lower() in fn.lower() or k > 0:
                files.append(os.path.join(sample_local_dir, fn))
    return files[:k]

cols_labels = sorted(df["label"].unique())
for label in cols_labels:
    st.markdown(f"**{label}**")
    cols = st.columns(show_per_class)
    sample = df[df["label"] == label].head(show_per_class)
    # try local fallback
    local_paths = _local_sample_paths(label, show_per_class) if use_local_samples else []
    for i, (col, row) in enumerate(zip(cols, sample.iterrows())):
        _, r = row
        img = None
        if local_paths and i < len(local_paths):
            try:
                img = Image.open(local_paths[i]).convert("RGB")
            except Exception:
                img = None
        if img is None:
            img = fetch_image(r["image_url"])
        if img is not None:
            col.image(img, use_column_width=True)
        else:
            col.warning("로드 실패")

# 결측치 요약
st.header("3. 결측치")
na = df.isnull().sum()
na = na[na > 0]
if len(na):
    st.bar_chart(na)
    st.caption("fake_method 결측은 REAL 행(변조 기법 없음)에서 발생합니다.")
else:
    st.success("결측치 없음")

# confidence_score 분포 (있으면)
st.header("4. Confidence score 분포")
if "confidence_score" in df.columns:
    fig_cs = px.histogram(df, x="confidence_score", color="label", nbins=40, marginal="box", title="confidence_score 분포 (클래스별)")
    st.plotly_chart(fig_cs, use_container_width=True)
    st.write(df.groupby("label")["confidence_score"].describe()[["mean","std","50%"]])
else:
    st.info("confidence_score 컬럼이 없습니다.")

# 도메인(출처) 분석
st.header("5. 출처 도메인(domain) 분석")
df = df.assign(domain=url_domain(df))
if df["domain"].notna().any():
    domain_counts = df["domain"].value_counts().reset_index()
    domain_counts.columns = ["domain","count"]
    topn = st.slider("상위 도메인 개수", 3, 30, 10)
    st.plotly_chart(px.bar(domain_counts.head(topn), x="domain", y="count", title=f"상위 {topn} 도메인"), use_container_width=True)

    # 도메인별 FAKE 비율
    ratio = (df.groupby("domain")["label"].apply(lambda s: (s=="FAKE").mean()).reset_index(name="fake_ratio"))
    ratio = ratio.sort_values("fake_ratio", ascending=False)
    st.subheader("도메인별 FAKE 비율 (상위 20)")
    st.plotly_chart(px.bar(ratio.head(20), x="domain", y="fake_ratio", title="도메인별 FAKE 비율", labels={"fake_ratio":"FAKE 비율"}), use_container_width=True)
else:
    st.info("domain 파생이 불가능합니다 (image_url 형식 불일치).")

# 누수 컬럼(학습에 쓰지 말 것)
st.header("6. ⚠️ 데이터 누수 컬럼 (학습 금지)")
st.warning(
    "아래 컬럼들은 label과 거의 1:1로 대응합니다. 모델 학습에는 쓰지 마세요. EDA용으로만 확인하세요."
)
for col in LEAKAGE_COLS:
    with st.expander(f"`{col}` × label 교차표"):
        st.dataframe(
            df.groupby([col, "label"]).size().unstack(fill_value=0),
            use_container_width=True,
        )

st.info("다음: 시각화 페이지에서 도메인·품질·연령대·성별 간의 관계를 더 깊게 봅니다.")
