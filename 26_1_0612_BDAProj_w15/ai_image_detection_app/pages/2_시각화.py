import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import data_missing_message, load_data
from src.features import add_resolution_features, url_domain


st.title("시각화 - 핵심 패턴")


@st.cache_data
def _load_data():
    return load_data()


def _prepare(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.upper()
    for col in ["gender", "age_group", "image_quality", "source", "fake_method", "detection_difficulty"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    if "confidence_score" in df.columns:
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    if "image_url" in df.columns:
        df["domain"] = url_domain(df).fillna("Unknown")
    return add_resolution_features(df)


try:
    df = _prepare(_load_data())
except FileNotFoundError:
    st.error("데이터 파일이 없습니다.")
    st.code(data_missing_message(), language="text")
    st.stop()
except Exception as exc:
    st.error("데이터 로드 오류")
    st.exception(exc)
    st.stop()

st.markdown("---")
st.subheader("1. 성별 Unknown 편향")
if {"label", "gender"}.issubset(df.columns):
    gender_df = df.groupby(["label", "gender"]).size().reset_index(name="count")
    gender_df["ratio"] = gender_df["count"] / gender_df.groupby("label")["count"].transform("sum") * 100
    fig = px.bar(
        gender_df,
        x="label",
        y="ratio",
        color="gender",
        text=gender_df["ratio"].map(lambda value: f"{value:.1f}%"),
        title="라벨별 gender 비율",
    )
    fig.update_layout(yaxis_title="비율(%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("핵심: FAKE는 gender=Unknown 비율이 높습니다.")

st.markdown("---")
st.subheader("2. 연령대 + 성별 조합")
if {"label", "age_group", "gender"}.issubset(df.columns):
    combo = df.groupby(["age_group", "gender", "label"]).size().reset_index(name="count")
    combo["group"] = combo["age_group"] + " / " + combo["gender"]
    pivot = combo.pivot_table(index="group", columns="label", values="count", fill_value=0)
    for label in ["FAKE", "REAL"]:
        if label not in pivot.columns:
            pivot[label] = 0
    total = (pivot["FAKE"] + pivot["REAL"]).replace(0, pd.NA)
    pivot["FAKE 비율"] = (pivot["FAKE"] / total * 100).fillna(0).round(1)
    heat_df = pivot.reset_index().sort_values("FAKE 비율", ascending=False)
    fig = px.imshow(
        heat_df.set_index("group")[["FAKE 비율"]],
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, "#c53030"], [1, "#2b6cb0"]],
        title="연령대/성별 조합별 FAKE 비율(%)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("핵심: 연령대 단독보다 성별과 조합할 때 차이가 선명합니다.")

st.markdown("---")
st.subheader("3. 품질 등급별 라벨 분포")
if {"label", "image_quality"}.issubset(df.columns):
    quality_df = df.groupby(["image_quality", "label"]).size().reset_index(name="count")
    fig = px.bar(
        quality_df,
        x="image_quality",
        y="count",
        color="label",
        barmode="group",
        text="count",
        title="image_quality별 FAKE/REAL 수",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("핵심: 품질은 모델 신뢰도 설명에 쓰기 좋습니다.")

st.markdown("---")
st.subheader("4. confidence_score 분포")
if {"label", "confidence_score"}.issubset(df.columns):
    st.caption("confidence_score는 데이터셋이 각 이미지 라벨에 대해 부여한 신뢰도 점수입니다.")
    fig = px.histogram(
        df.dropna(subset=["confidence_score"]),
        x="confidence_score",
        color="label",
        barmode="overlay",
        nbins=20,
        opacity=0.72,
        title="confidence_score 히스토그램",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("핵심: REAL과 FAKE의 confidence_score 위치가 약간 다릅니다.")

st.markdown("---")
st.subheader("5. 출처 편향")
if {"label", "source"}.issubset(df.columns):
    source_df = df.groupby(["source", "label"]).size().reset_index(name="count")
    fig = px.bar(source_df, x="source", y="count", color="label", barmode="stack", text="count", title="source별 라벨 분포")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("핵심: source는 라벨과 직접 연결되어 모델 입력에서 제외해야 합니다.")
