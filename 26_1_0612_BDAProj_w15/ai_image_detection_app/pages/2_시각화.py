import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import LEAKAGE_COLS, data_missing_message, load_data
from src.features import add_resolution_features, url_domain
from src.report_sync import find_project_report_path, sync_features_to_report
from src.ui_components import render_leakage_warning, render_project_notice, render_story_insight, render_workflow_card


st.title("📊 시각화 — AI 활용 이미지 판별 인사이트")
st.caption("데이터 분포, 출처 편향, 누수 가능 컬럼을 확인하여 모델 입력에 쓰면 안 되는 정보를 구분합니다.")
render_project_notice()
render_workflow_card(
    "이 페이지의 사용 흐름",
    "FINAL_DATASET.csv와 특징추출 CSV",
    "라벨·출처·품질·해상도·특징 분포를 비교합니다.",
    "보고서에 넣을 데이터 스토리와 모델 제외 근거를 얻습니다.",
)


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

st.info(
    "핵심 해석: 이 데이터는 영상 기반 탐지보다 URL 기반 얼굴 이미지의 "
    "AI 활용 가능성, 즉 실사(REAL)와 생성/아바타 계열(FAKE) 구분 문제에 가깝습니다."
)

app_dir = Path(os.path.dirname(os.path.dirname(__file__)))
feature_files = sorted((app_dir / "data").glob("features_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if feature_files:
    with st.expander("추출된 특징 파일 추가 분석", expanded=False):
        selected_feature_file = st.selectbox(
            "특징 파일 선택",
            feature_files,
            format_func=lambda p: p.name,
        )
        feat_df = pd.read_csv(selected_feature_file)
        st.write(f"선택 파일: `{selected_feature_file.name}` / {len(feat_df):,}행")
        feat_cards = st.columns(4)
        feat_cards[0].metric("특징 행 수", f"{len(feat_df):,}")
        if "label" in feat_df.columns:
            feat_cards[1].metric("라벨 종류", f"{feat_df['label'].nunique():,}")
        if "brightness" in feat_df.columns:
            feat_cards[2].metric("평균 밝기", f"{feat_df['brightness'].mean():.1f}")
        if "sharpness" in feat_df.columns:
            feat_cards[3].metric("평균 선명도", f"{feat_df['sharpness'].mean():.1f}")
        feature_plot_cols = [c for c in ["brightness", "sharpness", "face_area_ratio", "mean_pixel", "std_pixel", "avg_r", "avg_g", "avg_b"] if c in feat_df.columns]
        if feature_plot_cols:
            selected_features = st.multiselect("특징 분포 확인", feature_plot_cols, default=feature_plot_cols[:3])
            for feature in selected_features:
                left, right = st.columns(2)
                with left:
                    fig = px.histogram(
                        feat_df,
                        x=feature,
                        color=("label" if "label" in feat_df.columns else None),
                        nbins=30,
                        title=f"{feature} 특징 분포",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    if "label" in feat_df.columns:
                        fig = px.box(feat_df, x="label", y=feature, color="label", title=f"label별 {feature}")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("선택한 특징 파일에 시각화 가능한 기본 이미지 특징 컬럼이 없습니다.")
        if st.button("선택한 특징 파일을 보고서에 반영"):
            summary_path = app_dir / "reports" / "feature_pipeline_summary.json"
            try:
                report_path = sync_features_to_report(
                    selected_feature_file,
                    find_project_report_path(app_dir),
                    summary_path if summary_path.exists() else None,
                )
                st.success(f"보고서 반영 완료: {report_path}")
            except Exception as e:
                st.error("보고서 반영 중 오류가 발생했습니다.")
                st.exception(e)
else:
    st.caption("추출된 특징 파일이 없으면 EDA 페이지에서 특징추출을 실행하거나 `python -m src.feature_pipeline`을 사용해 주세요.")

total_rows = len(df)
label_counts = df["label"].value_counts() if "label" in df.columns else pd.Series(dtype=int)
fake_count = int(label_counts.get("FAKE", 0))
real_count = int(label_counts.get("REAL", 0))

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("전체 행 수", f"{total_rows:,}")
kpi2.metric("FAKE 수", f"{fake_count:,}")
kpi3.metric("REAL 수", f"{real_count:,}")
if fake_count + real_count:
    kpi4.metric("FAKE 비율", f"{fake_count / (fake_count + real_count) * 100:.1f}%")
else:
    kpi4.metric("FAKE 비율", "-")

st.subheader("핵심 인사이트 요약")
insight_cols = st.columns(4)
insight_cols[0].metric("라벨 수", f"FAKE {fake_count:,} / REAL {real_count:,}")
if {"category", "label"}.issubset(df.columns):
    category_purity = df.groupby("category")["label"].nunique().eq(1).mean() * 100
    insight_cols[1].metric("category 순도", f"{category_purity:.0f}%")
if {"source", "label"}.issubset(df.columns):
    source_purity = df.groupby("source")["label"].nunique().eq(1).mean() * 100
    insight_cols[2].metric("source 순도", f"{source_purity:.0f}%")
if {"domain", "label"}.issubset(df.columns):
    domain_purity = df.groupby("domain")["label"].nunique().eq(1).mean() * 100
    insight_cols[3].metric("domain 순도", f"{domain_purity:.0f}%")
render_leakage_warning()
render_story_insight(
    "이 데이터는 '조작 흔적'보다 '제작 생태계 차이'가 먼저 보입니다",
    "REAL은 images.unsplash.com의 1080×1080 실사 사진으로 모여 있고, FAKE는 dicebear·multiavatar·randomuser 같은 생성/아바타 API의 1024×1024 이미지로 모여 있습니다. "
    "따라서 모델이 좋은 성능을 보이더라도, 그것은 얼굴의 미세한 위조 흔적만이 아니라 실사 사진 서비스의 질감과 아바타 생성 서비스의 스타일 차이를 함께 포착한 결과일 수 있습니다.",
)

st.markdown("---")

st.header("1. 라벨 분포")
if "label" in df.columns and not label_counts.empty:
    label_df = label_counts.rename_axis("label").reset_index(name="count")
    label_df["ratio"] = label_df["count"] / label_df["count"].sum() * 100
    fig = px.bar(label_df, x="label", y="count", color="label", text="count", title="FAKE/REAL 샘플 수")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("라벨 분포 상세 표", expanded=False):
        st.dataframe(label_df, use_container_width=True)
    render_story_insight(
        "FAKE가 더 많은 구조는 서비스의 오경보 기준을 바꿉니다",
        "FAKE가 REAL보다 많기 때문에 모델이 다수 클래스에 끌릴 가능성이 있습니다. 실제 서비스에서는 단순 정확도보다 REAL 사진을 FAKE로 오해하는 오경보와 FAKE 이미지를 REAL로 놓치는 누락을 나누어 봐야 합니다.",
    )
else:
    st.warning("label 컬럼이 없어 라벨 분포를 표시할 수 없습니다.")

st.markdown("---")

st.header("2. 이미지 품질과 라벨 관계")
if {"image_quality", "label"}.issubset(df.columns):
    quality_df = (
        df.assign(image_quality=df["image_quality"].fillna("UNKNOWN"))
        .groupby(["image_quality", "label"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        quality_df,
        x="image_quality",
        y="count",
        color="label",
        barmode="group",
        title="image_quality별 FAKE/REAL 분포",
    )
    st.plotly_chart(fig, use_container_width=True)
    render_story_insight(
        "품질은 판정 근거가 아니라 신뢰도 조절 장치입니다",
        "High와 Medium 품질이 두 라벨 모두에 존재하므로 품질 하나로 AI 활용 여부를 나누기는 어렵습니다. 대신 흐림·저해상도 이미지는 얼굴 질감과 경계 정보를 약하게 만들어 모델 확률을 흔들 수 있어, 서비스에서 품질 경고를 함께 보여주는 근거가 됩니다.",
    )
else:
    st.info("image_quality 또는 label 컬럼이 없어 해당 분석을 건너뜁니다.")

st.markdown("---")

st.header("3. confidence_score 분포")
if {"confidence_score", "label"}.issubset(df.columns):
    score_df = df.copy()
    score_df["confidence_score"] = pd.to_numeric(score_df["confidence_score"], errors="coerce")
    score_df = score_df.dropna(subset=["confidence_score"])
    if len(score_df):
        tab_hist, tab_box = st.tabs(["히스토그램", "박스플롯"])
        with tab_hist:
            fig = px.histogram(
                score_df,
                x="confidence_score",
                color="label",
                nbins=25,
                barmode="overlay",
                title="label별 confidence_score 분포",
            )
            fig.update_traces(opacity=0.72)
            st.plotly_chart(fig, use_container_width=True)
        with tab_box:
            fig = px.box(score_df, x="label", y="confidence_score", color="label", title="label별 confidence_score 비교")
            st.plotly_chart(fig, use_container_width=True)
        render_story_insight(
            "REAL의 높은 confidence는 실사 출처의 일관성을 보여줍니다",
            "REAL의 confidence_score 평균이 FAKE보다 높습니다. 이는 Unsplash 실사 사진이 비교적 일정한 품질과 촬영 구도를 가지는 반면, FAKE는 여러 생성 API가 섞여 스타일과 품질 편차가 더 크기 때문으로 해석할 수 있습니다.",
        )
    else:
        st.info("confidence_score를 숫자로 변환할 수 있는 행이 없습니다.")
else:
    st.info("confidence_score 또는 label 컬럼이 없어 해당 분석을 건너뜁니다.")

st.markdown("---")

st.header("4. FAKE 생성 방식 분석")
if {"fake_method", "label"}.issubset(df.columns):
    fake_df = df[df["label"].astype(str).str.upper() == "FAKE"].copy()
    if len(fake_df):
        method_df = fake_df["fake_method"].fillna("UNKNOWN").value_counts().head(15).rename_axis("fake_method").reset_index(name="count")
        fig = px.bar(method_df, x="fake_method", y="count", text="count", title="FAKE 데이터 내 fake_method 상위 분포")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.warning("fake_method는 정답 라벨과 직접 연결될 가능성이 높으므로 모델 입력에는 사용하지 않고 EDA 분석용으로만 사용합니다.")
        render_story_insight(
            "FAKE는 하나의 기술이 아니라 여러 생성 서비스의 묶음입니다",
            "FAKE 내부에는 dicebear, multiavatar, randomuser, pravatar처럼 서로 다른 생성 방식이 섞여 있습니다. 따라서 이 프로젝트의 모델은 하나의 AI 흔적만 찾기보다 여러 생성 서비스의 스타일을 실사 사진과 구분하는 방향으로 동작할 가능성이 큽니다.",
        )
    else:
        st.info("FAKE 라벨 행이 없어 fake_method 분석을 표시할 수 없습니다.")
else:
    st.info("fake_method 또는 label 컬럼이 없어 해당 분석을 건너뜁니다.")

st.markdown("---")

st.header("5. 데이터셋 분할과 라벨 균형")
if {"dataset_split", "label"}.issubset(df.columns):
    split_df = df.groupby(["dataset_split", "label"]).size().reset_index(name="count")
    fig = px.bar(split_df, x="dataset_split", y="count", color="label", barmode="group", title="dataset_split × label 분포")
    st.plotly_chart(fig, use_container_width=True)
    split_ratio = pd.crosstab(df["dataset_split"], df["label"], normalize="index").fillna(0) * 100
    with st.expander("dataset_split 비율 상세 표", expanded=False):
        st.dataframe(split_ratio.round(1), use_container_width=True)
    render_story_insight(
        "분할별 비율이 비슷하면 평가 결과의 비교가 쉬워집니다",
        "train, val, test 모두 FAKE가 조금 더 많은 구조를 유지합니다. 따라서 특정 split만 유난히 다른 데이터가 아니라, 전체 데이터의 출처 구조가 평가셋에도 이어진다고 볼 수 있습니다.",
    )
else:
    st.info("dataset_split 또는 label 컬럼이 없어 해당 분석을 건너뜁니다.")

st.markdown("---")

st.header("6. 해상도와 라벨 관계")
if {"resolution", "label"}.issubset(df.columns):
    res_counts = df["resolution"].fillna("UNKNOWN").value_counts().head(15).rename_axis("resolution").reset_index(name="count")
    fig = px.bar(res_counts, x="resolution", y="count", text="count", title="resolution 상위 분포")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    if {"meta_total_pixels", "meta_aspect_ratio"}.issubset(df.columns):
        left, right = st.columns(2)
        with left:
            fig = px.box(df, x="label", y="meta_total_pixels", color="label", title="label별 메타 해상도 픽셀 수")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(df, x="label", y="meta_aspect_ratio", color="label", title="label별 메타 종횡비")
            st.plotly_chart(fig, use_container_width=True)
    render_story_insight(
        "해상도는 데이터 수집 방식의 흔적입니다",
        "REAL은 1080×1080, FAKE는 1024×1024로 분리되어 있습니다. 이는 실사 사진 플랫폼과 생성 API가 이미지를 제공하는 기본 규격 차이로 볼 수 있으며, 모델이 해상도 자체를 정답 힌트로 쓰지 않도록 주의해야 합니다.",
    )
else:
    st.info("resolution 또는 label 컬럼이 없어 해당 분석을 건너뜁니다.")

st.markdown("---")

st.header("7. URL 도메인/출처 편향 분석")
if {"domain", "label"}.issubset(df.columns):
    top_n = st.slider("표시할 상위 도메인 수", 5, 30, 12)
    top_domains = df["domain"].fillna("UNKNOWN").value_counts().head(top_n).index.tolist()
    domain_df = df[df["domain"].isin(top_domains)].copy()
    domain_counts = domain_df.groupby(["domain", "label"]).size().reset_index(name="count")
    fig = px.bar(domain_counts, x="domain", y="count", color="label", barmode="stack", title="상위 도메인별 FAKE/REAL 분포")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    ratio_table = pd.crosstab(domain_df["domain"], domain_df["label"], normalize="index").fillna(0) * 100
    with st.expander("도메인별 라벨 비율 상세 표", expanded=False):
        st.dataframe(ratio_table.round(1), use_container_width=True)
    render_story_insight(
        "출처 도메인은 이미지가 만들어진 배경을 보여줍니다",
        "REAL은 사진 공유 플랫폼, FAKE는 아바타·프로필 생성 API에 모여 있습니다. 이는 모델이 사람 얼굴만 보는 것이 아니라 배경, 색감, 구도, 렌더링 스타일 같은 제작 환경의 차이까지 함께 포착할 수 있음을 의미합니다.",
    )
else:
    st.info("domain 또는 label 컬럼이 없어 도메인 편향 분석을 표시할 수 없습니다.")

st.markdown("---")

st.header("8. 모델 입력 제외 컬럼 요약")
exclude_summary = pd.DataFrame(
    [
        {"컬럼": "label_numeric", "사용 범위": "모델 입력 금지", "제외 이유": "정답 라벨을 숫자로 바꾼 컬럼입니다."},
        {"컬럼": "category", "사용 범위": "EDA 전용", "제외 이유": "AI Generated는 FAKE, Authentic은 REAL과 직접 연결됩니다."},
        {"컬럼": "source", "사용 범위": "EDA 전용", "제외 이유": "MultiSource는 FAKE, Unsplash는 REAL과 직접 연결됩니다."},
        {"컬럼": "fake_method", "사용 범위": "EDA 전용", "제외 이유": "FAKE에만 값이 존재하고 REAL은 결측입니다."},
        {"컬럼": "detection_difficulty", "사용 범위": "EDA 전용", "제외 이유": "Easy는 REAL, Medium/Hard는 FAKE와 강하게 연결됩니다."},
        {"컬럼": "domain", "사용 범위": "보고서 분석용", "제외 이유": "이미지 URL 출처가 라벨과 직접 연결됩니다."},
        {"컬럼": "image_url", "사용 범위": "다운로드용", "제외 이유": "URL 문자열 자체는 이미지 픽셀 기반 판별 정보가 아닙니다."},
        {"컬럼": "image_id", "사용 범위": "식별자", "제외 이유": "샘플 식별자이며 일반화 가능한 이미지 특성이 아닙니다."},
    ]
)
with st.expander("모델 입력 제외 컬럼 상세 표", expanded=False):
    st.dataframe(exclude_summary, use_container_width=True)
st.error("위 컬럼을 모델 입력에 넣으면 성능이 높아 보여도 실제 이미지 판별 능력이 아니라 정답 힌트를 이용한 결과일 수 있습니다.")

st.markdown("---")

st.header("9. 누수 가능 컬럼 진단")
st.write("아래 컬럼은 label과 거의 직접 연결될 수 있으므로, 모델 학습/예측 입력에서는 제외하는 것이 원칙입니다.")
available_leakage_cols = [col for col in LEAKAGE_COLS if col in df.columns]
if available_leakage_cols and "label" in df.columns:
    selected_leakage_col = st.selectbox("진단할 누수 가능 컬럼", available_leakage_cols)
    cross = pd.crosstab(df[selected_leakage_col].fillna("UNKNOWN"), df["label"], margins=True)
    ratio = pd.crosstab(df[selected_leakage_col].fillna("UNKNOWN"), df["label"], normalize="index").fillna(0) * 100
    left, right = st.columns(2)
    with left:
        st.subheader("개수")
        with st.expander("교차표 개수 보기", expanded=False):
            st.dataframe(cross, use_container_width=True)
    with right:
        st.subheader("행 기준 비율(%)")
        with st.expander("교차표 비율 보기", expanded=False):
            st.dataframe(ratio.round(1), use_container_width=True)
    st.error(
        "실제 점검 결과 category/source/detection_difficulty/fake_method는 라벨과 거의 직접 연결됩니다. "
        "모델에 이 컬럼을 넣으면 AI 활용 이미지 판별 능력이 아니라 정답 힌트를 외운 결과일 수 있습니다."
    )
else:
    st.info("진단 가능한 누수 컬럼 또는 label 컬럼이 없습니다.")

st.markdown("---")

st.header("10. 추가 탐색 그래프")
exclude_vis_cols = {
    "image_id",
    "image_url",
    "label_numeric",
    "category",
    "source",
    "fake_method",
    "date_collected",
    "version",
    "year",
    "domain",
}
candidate_cols = [col for col in df.columns if col not in exclude_vis_cols]
if candidate_cols:
    with st.expander("추가 탐색 그래프", expanded=False):
        default_col = "label" if "label" in candidate_cols else candidate_cols[0]
        selected_col = st.selectbox("분포를 볼 컬럼", candidate_cols, index=candidate_cols.index(default_col))
        color_col = "label" if "label" in df.columns and selected_col != "label" else None
        fig = px.histogram(df, x=selected_col, color=color_col, title=f"{selected_col} 분포")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("추가 탐색용 그래프입니다. 보고서에는 위의 고정 인사이트 그래프를 우선 사용하는 것을 추천합니다.")
else:
    st.info("추가 탐색에 사용할 컬럼이 없습니다.")
