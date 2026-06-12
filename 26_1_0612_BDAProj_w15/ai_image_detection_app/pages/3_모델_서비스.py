import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.data_loader import data_missing_message, fetch_image, load_data
from src.image_quality import avg_brightness, sharpness_score
from src.lightweight_inference import find_available_model, load_lightweight_bundle, predict_with_lightweight_model
from src.model_eval import compute_eval_metrics, evaluate_image_sample, format_prediction_results, sample_evaluation_df
from src.ui_components import render_project_notice

MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = APP_DIR / "data" / "demo_samples"

st.set_page_config(page_title="모델·서비스", page_icon="🤖", layout="wide")
st.title("AI 모델·서비스 — 이미지 판별")
render_project_notice()
st.caption("사전학습 CNN 임베딩에 우리 데이터로 추가학습한 경량 분류기를 연결해 판별합니다.")

@st.cache_resource(show_spinner="모델 로드 중입니다.")
def load_img_model():
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers 또는 huggingface-hub 설치 상태를 확인해 주세요.") from exc
    device = 0 if st.session_state.get("use_gpu", False) else -1
    return pipeline("image-classification", model=MODEL_NAME, device=device)

@st.cache_resource
def load_local_bundle():
    model_path = find_available_model()
    if model_path is None:
        return None
    return load_lightweight_bundle(model_path)

@st.cache_data
def _load_metadata():
    return load_data()


def confidence_level(score: float):
    if score >= 0.8:
        return "높음", "결과가 비교적 안정적입니다."
    if score >= 0.6:
        return "보통", "이미지 품질과 함께 확인해 주세요."
    return "낮음", "재확인이 필요합니다."


def inspect_image_quality(image: Image.Image) -> dict:
    width, height = image.size
    brightness = avg_brightness(image)
    sharpness = sharpness_score(image)
    aspect_ratio = width / height if height else 0
    warnings = []
    if width < 128 or height < 128:
        warnings.append("해상도가 낮습니다.")
    if brightness < 30:
        warnings.append("이미지가 어둡습니다.")
    if brightness > 230:
        warnings.append("이미지가 너무 밝습니다.")
    if sharpness < 20:
        warnings.append("이미지가 흐립니다.")
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        warnings.append("종횡비가 일반적이지 않습니다.")
    return {"width": width, "height": height, "aspect_ratio": aspect_ratio, "brightness": brightness, "sharpness": sharpness, "warnings": warnings}


def load_demo_images() -> list:
    if not DEMO_DIR.exists():
        return []
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(DEMO_DIR.glob(f"**/{ext}"))
    return sorted(image_paths)


def render_prediction_results(results: list):
    results_df = format_prediction_results(results)
    top = results_df.iloc[0]
    predicted_label = top["정규화 라벨"]
    score = float(top["확률"])
    level, level_message = confidence_level(score)

    c1, c2, c3 = st.columns(3)
    c1.metric("최종 판정", predicted_label)
    c2.metric("예측 확률", f"{score * 100:.1f}%")
    c3.metric("신뢰도", level)

    if level == "높음":
        st.success(level_message)
    elif level == "보통":
        st.warning(level_message)
    else:
        st.error(level_message)

    if predicted_label == "FAKE":
        st.warning("AI 활용/생성 계열일 가능성이 높습니다. 다만 실제 제작 과정을 보장하는 판정은 아닙니다.")
    else:
        st.success("실사 사진 계열일 가능성이 높습니다. 다만 고품질 AI 이미지를 완전히 배제하는 것은 아닙니다.")

    fig = px.bar(
        results_df.assign(표시=results_df["정규화 라벨"] + " (" + results_df["원본 라벨"].astype(str) + ")"),
        x="확률(%)",
        y="표시",
        orientation="h",
        color="정규화 라벨",
        text=results_df["확률(%)"].map(lambda v: f"{v:.1f}%"),
        title="후보별 확률",
    )
    fig.update_layout(xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results_df, use_container_width=True)


def render_local_prediction_result(local_result: dict):
    predicted_label = local_result["pred_label"]
    score = float(local_result["score"])
    level, level_message = confidence_level(score)
    metrics = local_result.get("metrics", {})
    per_class = metrics.get("per_class", {})
    fake_f1 = per_class.get("FAKE", {}).get("f1")
    real_f1 = per_class.get("REAL", {}).get("f1")

    c1, c2, c3 = st.columns(3)
    c1.metric("최종 판정", predicted_label)
    c2.metric("예측 확률", f"{score * 100:.1f}%")
    c3.metric("추가학습 모델", "사용 중")

    st.info(
        f"현재 판별은 사전학습 CNN({local_result.get('backbone', 'resnet18')})에서 임베딩을 추출한 뒤, 우리 데이터로 추가학습한 경량 분류기로 예측합니다."
    )

    if level == "높음":
        st.success(level_message)
    elif level == "보통":
        st.warning(level_message)
    else:
        st.error(level_message)

    score_map = local_result.get("score_map", {})
    plot_df = pd.DataFrame(
        [{"label": k, "score": float(v) * 100} for k, v in score_map.items()]
    ).sort_values("score", ascending=True)
    fig = px.bar(plot_df, x="score", y="label", orientation="h", text=plot_df["score"].map(lambda v: f"{v:.1f}%"), title="추가학습 모델 확률")
    fig.update_layout(xaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    summary_rows = []
    if fake_f1 is not None:
        summary_rows.append({"지표": "FAKE F1", "값": round(float(fake_f1), 3)})
    if real_f1 is not None:
        summary_rows.append({"지표": "REAL F1", "값": round(float(real_f1), 3)})
    if metrics.get("accuracy") is not None:
        summary_rows.append({"지표": "Accuracy", "값": round(float(metrics["accuracy"]), 3)})
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.caption("이 방식은 외부 사전학습 모델을 그대로 쓰는 것보다 우리 데이터 특성을 일부 반영할 수 있지만, 학습 데이터 규모와 출처 편향의 한계는 남아 있습니다.")


def render_single_prediction_tab():
    st.info("파일 업로드 또는 데모 샘플 중 하나를 선택해 판별할 수 있습니다.")
    tab_file, tab_demo = st.tabs(["파일 업로드", "데모 샘플"])
    image = None
    input_source = None

    with tab_file:
        uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                input_source = uploaded_file.name
            except Exception:
                st.error("이미지를 열 수 없습니다.")

    with tab_demo:
        st.caption("데모 샘플은 프로젝트 폴더의 `data/demo_samples/` 안에 미리 넣어둔 로컬 이미지입니다. 발표나 테스트 때 같은 예시를 안정적으로 다시 불러올 때 사용합니다.")
        demo_images = load_demo_images()
        if demo_images:
            selected_demo = st.selectbox("데모 샘플", demo_images, format_func=lambda p: str(p.relative_to(APP_DIR)))
            if selected_demo:
                try:
                    image = Image.open(selected_demo).convert("RGB")
                    input_source = str(selected_demo.relative_to(APP_DIR))
                except Exception:
                    st.error("데모 이미지를 열 수 없습니다.")
        else:
            st.info("`data/demo_samples/real`, `data/demo_samples/fake` 폴더에 샘플을 넣으면 데모가 활성화됩니다.")

    if image is None:
        st.warning("이미지를 입력하면 판별을 시작할 수 있습니다.")
        return

    st.markdown("---")
    st.subheader("1. 입력 이미지")
    st.caption(input_source)
    left, right = st.columns([1.2, 1])
    with left:
        st.image(image, caption="입력 이미지", use_container_width=True)
    quality = inspect_image_quality(image)
    with right:
        st.metric("해상도", f"{quality['width']} × {quality['height']}")
        st.metric("밝기", f"{quality['brightness']:.1f}")
        st.metric("선명도", f"{quality['sharpness']:.1f}")
        if quality["warnings"]:
            st.warning("품질 경고")
            for warning in quality["warnings"]:
                st.write(f"- {warning}")
        else:
            st.success("품질 경고가 없습니다.")

    st.markdown("---")
    st.subheader("2. 판별 결과")
    if not st.button("판별 시작", type="primary"):
        st.info("`판별 시작` 버튼을 누르면 결과가 표시됩니다.")
        return

    local_bundle = load_local_bundle()
    if local_bundle is not None:
        try:
            local_result = predict_with_lightweight_model(image, local_bundle, model_name="resnet18")
            render_local_prediction_result(local_result)
            return
        except Exception as e:
            st.warning("추가학습 모델 예측에 실패하여 기본 사전학습 모델로 전환합니다.")
            with st.expander("오류 상세"):
                st.exception(e)

    try:
        classifier = load_img_model()
        results = classifier(image)
    except Exception as e:
        st.error("모델 예측 중 오류가 발생했습니다.")
        st.exception(e)
        return

    if not results:
        st.error("예측 결과가 없습니다.")
        return
    render_prediction_results(results)


def render_eval_tab():
    st.subheader("샘플 성능 평가")
    st.caption("CSV의 `image_url`과 실제 `label`을 이용해 일부 샘플의 예측 성능을 점검합니다.")
    try:
        df = _load_metadata()
    except FileNotFoundError:
        st.error("평가에 필요한 데이터 파일이 없습니다.")
        st.code(data_missing_message(), language="text")
        return
    except Exception as e:
        st.error("메타데이터를 불러오는 중 오류가 발생했습니다.")
        st.exception(e)
        return

    if not {"label", "image_url"}.issubset(df.columns):
        st.error("샘플 평가에 필요한 컬럼이 없습니다.")
        return

    c1, c2, c3 = st.columns(3)
    sample_size = c1.slider("평가 샘플 수", 10, 100, 30, step=10)
    balance_by_label = c2.checkbox("FAKE/REAL 균형 샘플링", value=True)
    random_state = c3.number_input("랜덤 시드", min_value=0, value=42, step=1)
    max_workers = st.slider("이미지 다운로드 워커 수", 1, 8, 4)
    st.caption("워커 수는 이미지를 동시에 몇 개까지 내려받을지 정하는 값입니다. 높을수록 빠를 수 있지만 네트워크 부담도 커집니다.")

    if not st.button("샘플 평가 시작", type="primary"):
        st.info("옵션을 확인한 뒤 `샘플 평가 시작`을 눌러 주세요.")
        return

    try:
        df_sample = sample_evaluation_df(df, sample_size=sample_size, balance_by_label=balance_by_label, random_state=int(random_state))
        classifier = load_img_model()
        eval_df = evaluate_image_sample(df_sample, classifier, max_workers=max_workers)
        metrics = compute_eval_metrics(eval_df)
    except Exception as e:
        st.error("샘플 평가 중 오류가 발생했습니다.")
        st.exception(e)
        return

    st.success("샘플 평가가 완료되었습니다.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("평가 시도", f"{metrics['attempted']:,}")
    m2.metric("성공 평가", f"{metrics['success']:,}")
    m3.metric("실패", f"{metrics['failed']:,}")
    m4.metric("정확도", "-" if metrics["accuracy"] is None else f"{metrics['accuracy'] * 100:.1f}%")

    if not metrics["confusion_matrix"].empty:
        fig = px.imshow(metrics["confusion_matrix"], text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics["confusion_matrix"], use_container_width=True)

    if not metrics["class_summary"].empty:
        st.dataframe(metrics["class_summary"], use_container_width=True)

    wrong_df = eval_df[(eval_df["error"].isna()) & (eval_df["is_correct"] == False)]
    st.subheader("오분류 사례")
    if len(wrong_df):
        st.dataframe(wrong_df[["image_id", "true_label", "pred_label", "score", "image_url"]], use_container_width=True)
    else:
        st.success("오분류 사례가 없습니다.")

    st.download_button(
        "평가 결과 CSV 다운로드",
        data=eval_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="eval_results_sample.csv",
        mime="text/csv",
    )

main_tab1, main_tab2 = st.tabs(["단일 이미지 판별", "샘플 성능 평가"])
with main_tab1:
    render_single_prediction_tab()
with main_tab2:
    render_eval_tab()