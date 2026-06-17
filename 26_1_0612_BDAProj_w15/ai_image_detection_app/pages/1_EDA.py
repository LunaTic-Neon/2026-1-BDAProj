import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import base64

from src.data_loader import data_missing_message, load_data
from src.features import url_domain
from src.ui_components import render_project_notice


st.title("EDA - 데이터 점검")
render_project_notice()

APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = APP_DIR / "data" / "demo_samples"

st.markdown(
    """
    <style>
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }
    .sample-tile {
        border: 1px solid rgba(127, 127, 127, 0.22);
        border-radius: 8px;
        padding: 8px;
        background: rgba(127, 127, 127, 0.05);
    }
    .sample-tile img {
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 6px;
        display: block;
    }
    .sample-caption {
        margin-top: 6px;
        font-size: 0.82rem;
        line-height: 1.25;
        text-align: center;
        word-break: break-word;
    }
    @media (max-width: 900px) {
        .sample-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


@st.cache_data
def _load(nrows):
    return load_data(nrows)


def _prepare(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.upper()
    for col in ["gender", "age_group", "image_quality", "source", "fake_method", "detection_difficulty", "dataset_split"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    if "image_url" in df.columns:
        df["domain"] = url_domain(df).fillna("Unknown")
    if "confidence_score" in df.columns:
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    return df


try:
    df = _prepare(_load(None))
except FileNotFoundError:
    st.error("데이터 파일이 없습니다.")
    st.code(data_missing_message(), language="text")
    st.stop()
except Exception as exc:
    st.error("데이터 로드 오류")
    st.exception(exc)
    st.stop()

label_counts = df["label"].value_counts() if "label" in df.columns else pd.Series(dtype=int)
fake_count = int(label_counts.get("FAKE", 0))
real_count = int(label_counts.get("REAL", 0))

c1, c2 = st.columns(2)
c1.metric("행 / 열", f"{len(df):,} / {df.shape[1]}")
c2.metric("FAKE / REAL", f"{fake_count:,} / {real_count:,}")

st.markdown("---")
st.subheader("1. 컬럼 역할 정리")
role_rows = [
    {"구분": "타깃", "컬럼": "label", "사용": "예측 정답"},
    {"구분": "이미지 입력", "컬럼": "image_url", "사용": "이미지 다운로드/데모용"},
    {"구분": "기초 메타", "컬럼": "gender, age_group, image_quality, confidence_score", "사용": "EDA/시각화"},
    {"구분": "누수 위험", "컬럼": "category, source, fake_method, detection_difficulty, domain", "사용": "라벨과 직접 연결될 수 있어 EDA/보고서 분석용으로만 사용하고 모델 입력에서는 제외"},
    {"구분": "관리 정보", "컬럼": "dataset_split, date_collected, version, year", "사용": "데이터 확인"},
]
st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("2. 라벨별 메타 요약")
summary_rows = []
if "label" in df.columns:
    for label, part in df.groupby("label"):
        row = {"label": label, "count": len(part)}
        if "gender" in part.columns:
            row["Unknown gender(%)"] = round((part["gender"].str.lower() == "unknown").mean() * 100, 1)
        if "confidence_score" in part.columns:
            row["avg confidence"] = round(float(part["confidence_score"].mean()), 3)
        if "image_quality" in part.columns:
            row["top quality"] = part["image_quality"].mode().iloc[0]
        summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty:
    g1, g2, g3 = st.columns(3)
    with g1:
        fig = px.bar(summary_df, x="label", y="count", color="label", text="count", title="라벨 수")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = px.bar(summary_df, x="label", y="Unknown gender(%)", color="label", text="Unknown gender(%)", title="Unknown gender 비율")
        fig.update_layout(showlegend=False, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
    with g3:
        fig = px.bar(summary_df, x="label", y="avg confidence", color="label", text="avg confidence", title="평균 confidence")
        fig.update_layout(showlegend=False, yaxis_range=[0.8, 1.0])
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("3. 누수 위험 확인")
leak_cols = [col for col in ["source", "fake_method", "detection_difficulty", "domain"] if col in df.columns]
if leak_cols and "label" in df.columns:
    selected_col = st.selectbox("확인할 컬럼", leak_cols)
    cross = pd.crosstab(df[selected_col], df["label"])
    st.dataframe(cross, use_container_width=True)
    st.warning("이 표에서 한쪽 라벨로만 몰리면 모델 입력에 쓰면 안 됩니다.")

st.markdown("---")
st.subheader("4. 샘플 이미지")
sample_paths = sorted((DEMO_DIR / "real").glob("*.jpg"))[:4] + sorted((DEMO_DIR / "fake").glob("*.jpg"))[:4]
if sample_paths:
    cards = ["<div class='sample-grid'>"]
    for path in sample_paths:
        label = path.parent.name.upper()
        cards.append(
            "<div class='sample-tile'>"
            f"<img src='{image_data_uri(path)}' alt='{label} sample'>"
            f"<div class='sample-caption'><b>{label}</b><br>{path.name}</div>"
            "</div>"
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)
else:
    st.info("데모 샘플 이미지가 없습니다.")