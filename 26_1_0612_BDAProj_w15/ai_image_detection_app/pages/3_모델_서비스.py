import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.data_loader import CACHE_DIR, _url_to_name, data_missing_message, load_data
from src.finetuned_resnet import MODEL_PATH as FINETUNED_MODEL_PATH, evaluate_finetuned_sample, predict_with_finetuned_resnet
from src.image_forensics import ai_feature_interpretations, ai_feature_signal_summary, extract_ai_detection_features
from src.model_eval import compute_eval_metrics, sample_evaluation_df
from src.ui_components import render_project_notice


APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = APP_DIR / "data" / "demo_samples"

st.title("AI 모델·서비스")
render_project_notice()

@st.cache_data
def _load_metadata():
    return load_data()


def load_demo_images() -> list[Path]:
    if not DEMO_DIR.exists():
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(DEMO_DIR.glob(f"**/{ext}"))
    return sorted(paths)


def open_image_safe(path_or_file):
    try:
        return Image.open(path_or_file).convert("RGB")
    except Exception:
        return None


def confidence_level(score: float) -> str:
    if score >= 0.8:
        return "높음"
    if score >= 0.6:
        return "보통"
    return "낮음"


def meaningful_feature_rows(features: dict) -> list[dict]:
    interpretations = ai_feature_interpretations(features)
    focus = {"해상도", "질감 복잡도", "엣지 밀도", "얼굴 영역 비율"}
    return [row for row in interpretations if row["특성"] in focus]


def feature_contribution_rows(features: dict) -> list[dict]:
    """서로 단위가 다른 특성을 0~100 비중으로 맞춰 그래프에 사용합니다."""
    megapixels = float(features.get("megapixels", 0) or 0)
    texture = float(features.get("texture_entropy", 0) or 0)
    edge = float(features.get("edge_density", 0) or 0)
    high_freq = float(features.get("high_frequency_ratio", 0) or 0)
    saturation = float(features.get("saturation_std", 0) or 0)
    face_ratio = float(features.get("face_area_ratio", 0) or 0)
    raw_rows = [
        {"특성": "해상도", "raw": min(abs(megapixels - 0.3) / 0.7, 1.0)},
        {"특성": "질감", "raw": min(abs(texture - 6.2) / 2.0, 1.0)},
        {"특성": "엣지", "raw": min(abs(edge - 0.05) / 0.15, 1.0)},
        {"특성": "고주파", "raw": min(abs(high_freq - 0.08) / 0.12, 1.0)},
        {"특성": "채도", "raw": min(abs(saturation - 0.16) / 0.2, 1.0)},
        {"특성": "얼굴", "raw": min(face_ratio / 0.35, 1.0)},
    ]
    total = sum(row["raw"] for row in raw_rows) or 1.0
    return [{"특성": row["특성"], "비중(%)": row["raw"] / total * 100} for row in raw_rows]


def feature_plot(features: dict):
    plot_df = pd.DataFrame(feature_contribution_rows(features))
    fig = px.bar(plot_df, x="특성", y="비중(%)", text=plot_df["비중(%)"].map(lambda value: f"{value:.1f}%"), title="주요 특성 참고 비중")
    fig.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(range=[0, max(35, float(plot_df["비중(%)"].max()) + 5)])
    return fig


def render_feature_matrix(features: dict):
    rows = ai_feature_interpretations(features)
    important = {row["특성"] for row in meaningful_feature_rows(features)}
    html = "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:8px 0;'>"
    for row in rows:
        is_important = row["특성"] in important
        border = "2px solid #ff8a00" if is_important else "1px solid #ddd"
        opacity = "1" if is_important else "0.62"
        font_size = "1.02rem" if is_important else "0.86rem"
        html += (
            f"<div style='border:{border};border-radius:8px;padding:8px 10px;background:rgba(127,127,127,0.06);opacity:{opacity};'>"
            f"<b><u>{row['특성']}</u></b><br>"
            f"<span style='font-size:{font_size};font-weight:700;'>{row['값']}</span><br>"
            f"<span style='font-size:0.78rem;'>{row['해석']}</span>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def brief_prediction_reason(features: dict, predicted_label: str) -> str:
    summary = ai_feature_signal_summary(features)
    signals = summary["fake_signals"] if predicted_label == "FAKE" else summary["real_signals"]
    if not signals:
        signals = summary["fake_signals"] + summary["real_signals"]
    return " ".join(signals[:2]) if signals else "유의미한 특성이 뚜렷하지 않습니다."


def render_feature_demo(image: Image.Image):
    features = extract_ai_detection_features(image)
    if not features:
        st.warning("특성 추출 실패")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("해상도", f"{features['width']}x{features['height']}")
    c2.metric("질감", f"{features['texture_entropy']:.3f}")
    c3.metric("엣지", f"{features['edge_density']:.3f}")
    c4.metric("얼굴비율", f"{features['face_area_ratio']:.3f}")

    compact_rows = pd.DataFrame(ai_feature_interpretations(features))[['특성', '값', '해석']]
    compact_rows = compact_rows.rename(columns={'해석': '판별근거'})
    st.dataframe(compact_rows, use_container_width=True, hide_index=True)

    return features


def render_feature_reason_area(features: dict):
    summary = ai_feature_signal_summary(features)
    st.info(summary["verdict"])
    left, middle, right = st.columns([1.05, 1.05, 1.9])
    with left:
        st.warning("FAKE 근거")
        for signal in summary["fake_signals"][:3] or ["뚜렷한 신호 적음"]:
            st.write(f"- {signal}")
    with middle:
        st.success("REAL 근거")
        for signal in summary["real_signals"][:3] or ["뚜렷한 신호 적음"]:
            st.write(f"- {signal}")
    with right:
        st.plotly_chart(feature_plot(features), use_container_width=True)


def render_simple_prediction(features: dict, result: dict):
    """모델 판별 결과를 카드와 확률 그래프로 표시합니다."""
    predicted_label = result["pred_label"]
    score = float(result["score"])
    score_map = result.get("score_map", {})
    rows = [{"label": label, "score": float(value) * 100} for label, value in score_map.items()]

    c1, c2, c3 = st.columns(3)
    c1.metric("판정", predicted_label)
    c2.metric("확률", f"{score * 100:.1f}%")
    c3.metric("신뢰도", confidence_level(score))

    st.info(brief_prediction_reason(features, predicted_label))

    plot_df = pd.DataFrame(rows)
    if not plot_df.empty:
        fig = px.bar(plot_df, x="score", y="label", orientation="h", text=plot_df["score"].map(lambda value: f"{value:.1f}%"), title="판별 확률")
        fig.update_layout(xaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def classification_metric_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support

    success_df = eval_df[eval_df["error"].isna() & eval_df["pred_label"].notna()].copy()
    if success_df.empty:
        return pd.DataFrame()
    labels = [label for label in ["FAKE", "REAL"] if label in set(success_df["true_label"]) or label in set(success_df["pred_label"])]
    precision, recall, f1, support = precision_recall_fscore_support(
        success_df["true_label"],
        success_df["pred_label"],
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        [
            {
                "label": label,
                "precision": round(float(precision[idx]), 3),
                "recall": round(float(recall[idx]), 3),
                "f1_score": round(float(f1[idx]), 3),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(labels)
        ]
    )


def run_prediction(image: Image.Image, features: dict):
    """저장된 MobileNetV3 모델로 이미지를 판별합니다."""
    if not FINETUNED_MODEL_PATH.exists():
        st.error("저장된 모델 파일이 없습니다.")
        return
    try:
        result = predict_with_finetuned_resnet(image, FINETUNED_MODEL_PATH)
        render_simple_prediction(features, result)
    except Exception as exc:
        st.error("파인튜닝 모델 실행 실패")
        st.exception(exc)
        return


def render_feature_demo_tab():
    """데모 이미지의 해석 가능한 특성을 보여주는 탭입니다."""
    st.subheader("특성추출")
    demo_images = load_demo_images()
    if not demo_images:
        st.info("데모 샘플이 없습니다.")
        return

    selected_demo = st.selectbox("이미지 선택", demo_images, format_func=lambda path: str(path.relative_to(APP_DIR)))
    image = open_image_safe(selected_demo)
    if image is None:
        st.error("이미지를 열 수 없습니다.")
        return

    left, right = st.columns([1, 1.45])
    with left:
        st.image(image, caption=str(selected_demo.relative_to(APP_DIR)), use_container_width=True)
    with right:
        features = render_feature_demo(image)
    if features:
        st.markdown("---")
        render_feature_reason_area(features)


def render_prediction_input(image: Image.Image | None, caption: str | None, button_key: str):
    if image is None:
        st.info("이미지를 선택해 주세요.")
        return

    left, right = st.columns([1, 1.25])
    with left:
        st.image(image, caption=caption, use_container_width=True)
        clicked = st.button("판별 시작", type="primary", key=button_key, use_container_width=True)
    with right:
        if not clicked:
            st.info("판별 시작을 누르면 특성과 결과가 표시됩니다.")
            return
        features = extract_ai_detection_features(image)
        st.subheader("특성")
        render_feature_matrix(features)
        st.subheader("판별 결과")
        run_prediction(image, features)


def render_prediction_tab():
    st.subheader("간단 판별")
    tab_file, tab_demo = st.tabs(["파일 업로드", "데모 샘플"])

    with tab_file:
        uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"])
        image = open_image_safe(uploaded_file) if uploaded_file is not None else None
        render_prediction_input(image, uploaded_file.name if uploaded_file is not None else None, "predict_uploaded")

    with tab_demo:
        demo_images = load_demo_images()
        if not demo_images:
            st.info("데모 샘플이 없습니다.")
            return
        selected_demo = st.selectbox("데모 이미지", demo_images, format_func=lambda path: str(path.relative_to(APP_DIR)))
        image = open_image_safe(selected_demo)
        render_prediction_input(image, str(selected_demo.relative_to(APP_DIR)), "predict_demo")


def render_eval_tab():
    """test split 캐시 이미지 기준으로 모델 성능을 평가합니다."""
    st.subheader("샘플 평가")
    try:
        df = _load_metadata()
    except FileNotFoundError:
        st.error("평가 데이터가 없습니다.")
        st.code(data_missing_message(), language="text")
        return
    except Exception as exc:
        st.error("메타데이터 로드 실패")
        st.exception(exc)
        return

    if not {"label", "image_url"}.issubset(df.columns):
        st.error("label, image_url 컬럼이 필요합니다.")
        return

    c1, c2, c3 = st.columns(3)
    sample_size = c1.slider("샘플 수", 10, 200, 30, step=10)
    balance_by_label = c2.checkbox("균형 샘플", value=True)
    random_state = c3.number_input("시드", min_value=0, value=42, step=1)

    if not st.button("평가 시작", type="primary"):
        return

    try:
        if FINETUNED_MODEL_PATH.exists():
            df_eval_base = df.copy()
            if "dataset_split" in df_eval_base.columns:
                df_eval_base = df_eval_base[df_eval_base["dataset_split"].astype(str).str.lower() == "test"].copy()
            df_eval_base["_cached_path"] = df_eval_base["image_url"].map(lambda url: str(CACHE_DIR / _url_to_name(str(url))))
            df_eval_base = df_eval_base[df_eval_base["_cached_path"].map(lambda path: Path(path).exists())].copy()
            df_sample = sample_evaluation_df(df_eval_base, sample_size=sample_size, balance_by_label=balance_by_label, random_state=int(random_state))
            st.caption("평가 모델: 파인튜닝 MobileNetV3 Small")
            eval_df = evaluate_finetuned_sample(df_sample, max_workers=4)
        else:
            st.error("저장된 모델 파일이 없습니다.")
            return
        metrics = compute_eval_metrics(eval_df)
    except Exception as exc:
        st.error("평가 실패")
        st.exception(exc)
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("시도", f"{metrics['attempted']:,}")
    m2.metric("성공", f"{metrics['success']:,}")
    m3.metric("실패", f"{metrics['failed']:,}")
    m4.metric("정확도", "-" if metrics["accuracy"] is None else f"{metrics['accuracy'] * 100:.1f}%")
    metric_df = classification_metric_table(eval_df)
    if not metric_df.empty:
        st.subheader("분류 지표")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
    if not metrics["confusion_matrix"].empty:
        st.plotly_chart(px.imshow(metrics["confusion_matrix"], text_auto=True, aspect="auto", title="Confusion Matrix"), use_container_width=True)


tab_features, tab_predict, tab_eval = st.tabs(["특성추출", "이미지 판별", "샘플 평가"])
with tab_features:
    render_feature_demo_tab()
with tab_predict:
    render_prediction_tab()
with tab_eval:
    render_eval_tab()