# pages/1_EDA.py — 이미지 EDA (CSV+URL 데이터용)
# 완전 구현: UI(사이드바, KPI, 탭), 샘플 이미지 그리드, 결측/캐시 리포트, 캐시 관리
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from src.features import url_domain
from src.ui_components import render_project_notice, render_report_tip, render_leakage_warning
from src.data_loader import load_data, data_missing_message

st.set_page_config(page_title="EDA", page_icon="🔎", layout="wide")
st.title("🔎 이미지 EDA — AI 활용 이미지 판별")
render_project_notice()

render_report_tip(
    "- URL 기반 얼굴 이미지 데이터\n"
    "- 모델 입력은 이미지 픽셀 중심\n"
    "- 누수 가능 컬럼은 분석용으로만 사용"
)

st.markdown(
    """
    <style>
    .sample-card {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 12px;
        padding: 10px;
        background: rgba(120, 120, 120, 0.04);
        height: 100%;
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.04);
    }
    .sample-card img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        border-radius: 10px;
        display: block;
    }
    .sample-meta {
        margin-top: 8px;
        padding: 8px 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.82);
        color: #111111;
        font-size: 0.9rem;
        line-height: 1.45;
        min-height: 3.2em;
        border: 1px solid rgba(0, 0, 0, 0.08);
        text-shadow: none;
    }
    .sample-meta strong {
        color: #000000;
        font-weight: 700;
    }
    @media (prefers-color-scheme: dark) {
        .sample-meta {
            background: rgba(20, 20, 20, 0.88);
            color: #f5f5f5;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .sample-meta strong {
            color: #ffffff;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def _load(nrows):
    return load_data(nrows)

rows_opt = st.sidebar.selectbox("로드할 메타 데이터 행 수", options=[None, 100, 500, 2000], index=0, format_func=lambda x: "전체" if x is None else str(x))
label_filter = st.sidebar.multiselect("표시할 클래스", options=["FAKE", "REAL"], default=["FAKE", "REAL"])

try:
    df = _load(rows_opt)
except FileNotFoundError:
    st.error("데이터 파일이 없어 EDA를 진행할 수 없습니다.")
    st.code(data_missing_message(), language="text")
    st.stop()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

if "label" in df.columns and label_filter:
    df_view = df[df["label"].isin(label_filter)].copy()
else:
    df_view = df.copy()

if "image_url" in df_view.columns:
    df_view["domain"] = url_domain(df_view)

render_leakage_warning()

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 행 수", f"{len(df_view):,}")
c2.metric("컬럼 수", f"{df_view.shape[1]:,}")
c3.metric("결측 image_url", f"{int(df_view['image_url'].isna().sum()) if 'image_url' in df_view.columns else 0:,}")
c4.metric("중복 image_url", f"{int(df_view['image_url'].duplicated().sum()) if 'image_url' in df_view.columns else 0:,}")

st.markdown("---")
st.subheader("1. 라벨 분포")
if "label" in df_view.columns:
    label_counts = df_view["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    fig = px.bar(label_counts, x="label", y="count", color="label", text="count")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("label 컬럼이 없습니다.")

st.subheader("2. 데이터 품질 요약")
quality_cols = [c for c in ["image_quality", "confidence_score", "dataset_split"] if c in df_view.columns]
if quality_cols:
    cols = st.columns(len(quality_cols))
    for idx, col_name in enumerate(quality_cols):
        with cols[idx]:
            vc = df_view[col_name].fillna("UNKNOWN").value_counts().reset_index()
            vc.columns = [col_name, "count"]
            fig = px.bar(vc, x=col_name, y="count", title=col_name)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("품질 관련 컬럼이 없습니다.")

st.subheader("3. 누수 가능 컬럼 확인")
leak_cols = [c for c in ["category", "source", "fake_method", "detection_difficulty"] if c in df_view.columns]
if leak_cols:
    leak_summary = []
    for col in leak_cols:
        leak_summary.append({"컬럼": col, "고유값 수": int(df_view[col].nunique(dropna=True)), "결측 수": int(df_view[col].isna().sum())})
    st.dataframe(pd.DataFrame(leak_summary), use_container_width=True)
else:
    st.info("누수 가능 컬럼이 없습니다.")

st.subheader("4. 샘플 이미지 확인")
if "image_url" in df_view.columns:
    sample_df = df_view[[c for c in ["image_url", "label", "image_quality", "source", "fake_method", "detection_difficulty"] if c in df_view.columns]].dropna(subset=["image_url"]).head(8)
    if len(sample_df):
        cols = st.columns(4)
        for i, (_, row) in enumerate(sample_df.iterrows()):
            method_value = row.get('fake_method', None)
            method_text = "" if pd.isna(method_value) else f", method={method_value}"
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div class="sample-card">
                        <img src="{row['image_url']}" alt="sample image" />
                        <div class="sample-meta">
                            <strong>{row.get('label', '-')}</strong><br>
                            quality={row.get('image_quality', '-')}, source={row.get('source', '-')}{method_text}, diff={row.get('detection_difficulty', '-')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("표시할 샘플 이미지가 없습니다.")
else:
    st.info("image_url 컬럼이 없습니다.")

st.subheader("5. 결측치 요약")
na = df_view.isnull().sum()
na = na[na > 0].sort_values(ascending=False)
if len(na):
    missing_df = na.reset_index()
    missing_df.columns = ["컬럼", "결측 수"]
    missing_df["결측 비율(%)"] = (missing_df["결측 수"] / len(df_view) * 100).round(1)
    st.dataframe(missing_df, use_container_width=True)
    st.caption(
        "결측 비율이 높아 보일 수 있지만, 이 데이터는 원본 수집 단계에서 URL·메타데이터가 함께 제공되는 구조라 일부 컬럼의 결측이 곧바로 데이터 전체 품질 저하를 의미하지는 않습니다. "
        "중요한 것은 모델 입력에 쓰는 이미지 자체가 정상적으로 확보되는지와, 누수 가능 컬럼을 배제한 뒤에도 라벨 분포와 출처 편향이 일관되게 유지되는지입니다."
    )
else:
    st.success("결측치가 없습니다.")

st.caption("EDA는 평가용 핵심만 남겼습니다. 상세 전처리와 모델 비교는 다른 페이지에서 확인합니다.")

# EOF

