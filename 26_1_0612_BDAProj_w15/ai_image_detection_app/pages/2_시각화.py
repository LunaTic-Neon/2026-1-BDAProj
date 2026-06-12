import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import LEAKAGE_COLS, data_missing_message, load_data
from src.features import add_resolution_features, url_domain
from src.ui_components import render_leakage_warning, render_story_insight

st.title("📈 시각화 — AI 활용 이미지 판별 인사이트")
st.caption("데이터 분포와 편향을 간단히 확인하는 페이지입니다.")


@st.cache_data
def _load_data():
    return load_data()


try:
    df = _load_data()
except FileNotFoundError:
    st.error("데이터 파일이 없어 시각화를 진행할 수 없습니다.")
    st.code(data_missing_message(), language="text")
    st.stop()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

if "image_url" in df.columns:
    df = df.assign(domain=url_domain(df))
else:
    df = df.assign(domain="UNKNOWN")
df = add_resolution_features(df)
if "label" in df.columns:
    df["label"] = df["label"].astype(str).str.upper()

# 핵심 요약
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 행 수", f"{len(df):,}")
c2.metric("전체 컬럼 수", f"{df.shape[1]:,}")
c3.metric("FAKE 수", f"{int((df['label'] == 'FAKE').sum()) if 'label' in df.columns else 0:,}")
c4.metric("REAL 수", f"{int((df['label'] == 'REAL').sum()) if 'label' in df.columns else 0:,}")

render_leakage_warning()

st.markdown("---")
st.header("1. 라벨 분포")
if "label" in df.columns:
    label_df = df["label"].value_counts().rename_axis("label").reset_index(name="count")
    fig = px.bar(label_df, x="label", y="count", color="label", text="count", title="FAKE/REAL 분포")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    render_story_insight(
        "라벨은 약간 불균형합니다",
        "FAKE가 REAL보다 조금 많아 단순 정확도만 보면 다수 클래스에 끌릴 수 있습니다. 그래서 발표에서는 정확도와 함께 오분류 가능성을 같이 설명하는 것이 좋습니다.",
    )
else:
    st.warning("label 컬럼이 없어 라벨 분포를 표시할 수 없습니다.")

st.markdown("---")
st.header("2. 이미지 품질 분포")
quality_cols = [c for c in ["image_quality", "confidence_score", "dataset_split"] if c in df.columns]
if quality_cols:
    for col_name in quality_cols:
        vc = df[col_name].fillna("UNKNOWN").value_counts().reset_index()
        vc.columns = [col_name, "count"]
        fig = px.bar(vc, x=col_name, y="count", title=f"{col_name} 분포")
        st.plotly_chart(fig, use_container_width=True)
    render_story_insight(
        "품질은 판정 근거보다 신뢰도 설명에 가깝습니다",
        "이미지 품질과 confidence_score는 모델이 어떤 입력에서 더 안정적인지 설명하는 보조 근거로 쓰는 것이 적절합니다.",
    )
else:
    st.info("품질 관련 컬럼이 없어 해당 분석을 건너뜁니다.")

st.markdown("---")
st.header("4. 출처 도메인 편향")
if {"domain", "label"}.issubset(df.columns):
    top_domains = df["domain"].fillna("UNKNOWN").value_counts().head(10).index.tolist()
    domain_df = df[df["domain"].isin(top_domains)].copy()
    domain_counts = domain_df.groupby(["domain", "label"]).size().reset_index(name="count")
    fig = px.bar(domain_counts, x="domain", y="count", color="label", barmode="stack", title="상위 도메인별 분포")
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    render_story_insight(
        "도메인 편향은 가장 중요한 해석 포인트입니다",
        "REAL은 실사 사진 플랫폼, FAKE는 생성/아바타 API에 몰려 있어 모델이 출처 차이를 학습할 가능성이 큽니다.",
    )
else:
    st.info("domain 또는 label 컬럼이 없어 도메인 편향 분석을 표시할 수 없습니다.")

st.markdown("---")
st.header("5. 누수 가능 컬럼 진단")
available_leakage_cols = [col for col in LEAKAGE_COLS if col in df.columns]
if available_leakage_cols and "label" in df.columns:
    selected_col = st.selectbox("진단할 컬럼", available_leakage_cols)
    cross = pd.crosstab(df[selected_col].fillna("UNKNOWN"), df["label"], margins=True)
    ratio = pd.crosstab(df[selected_col].fillna("UNKNOWN"), df["label"], normalize="index").fillna(0) * 100
    left, right = st.columns(2)
    with left:
        st.dataframe(cross, use_container_width=True)
    with right:
        st.dataframe(ratio.round(1), use_container_width=True)
    st.error("이 컬럼들은 label과 직접 연결될 수 있으므로 모델 입력에 사용하지 않습니다.")
else:
    st.info("진단 가능한 누수 컬럼이 없습니다.")

st.markdown("---")
st.header("6. 모델 입력 제외 컬럼 요약")
exclude_summary = pd.DataFrame(
    [
        {"컬럼": "category", "이유": "FAKE/REAL과 직접 연결"},
        {"컬럼": "source", "이유": "출처가 라벨과 직접 연결"},
        {"컬럼": "fake_method", "이유": "FAKE 전용 정보"},
        {"컬럼": "detection_difficulty", "이유": "라벨과 강하게 연결"},
        {"컬럼": "domain", "이유": "URL 출처 편향"},
        {"컬럼": "image_url", "이유": "다운로드용 정보"},
        {"컬럼": "image_id", "이유": "식별자"},
    ]
)
st.dataframe(exclude_summary, use_container_width=True)
st.caption("위 컬럼은 보고서 설명용으로는 유용하지만 모델 입력에는 넣지 않는 것이 원칙입니다.")
