import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import json
from PIL import Image
from transformers import pipeline

from src.data_loader import data_missing_message, fetch_image, load_data
from src.image_quality import avg_brightness, sharpness_score
from src.llm_explainer import build_explanation_prompt, check_ollama_status, generate_ollama_explanation
from src.model_eval import (
    compute_eval_metrics,
    evaluate_image_sample,
    format_prediction_results,
    normalize_prediction_label,
    sample_evaluation_df,
)
from src.report_sync import find_project_report_path, sync_eval_to_report
from src.ui_components import render_project_notice


MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = APP_DIR / "data" / "demo_samples"


@st.cache_resource(show_spinner="모델 로드 중입니다. 최초 실행에는 시간이 걸릴 수 있습니다.")
def load_img_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=MODEL_NAME, device=device)


@st.cache_data
def _load_metadata():
    return load_data()


def confidence_level(score: float):
    if score >= 0.8:
        return "높음", "결과가 비교적 안정적입니다."
    if score >= 0.6:
        return "보통", "이미지 품질과 후보 확률을 함께 확인해 주세요."
    return "낮음", "더 선명한 이미지나 다른 샘플로 재확인하는 것을 권장합니다."


def inspect_image_quality(image: Image.Image) -> dict:
    width, height = image.size
    brightness = avg_brightness(image)
    sharpness = sharpness_score(image)
    aspect_ratio = width / height if height else 0
    warnings = []

    if width < 128 or height < 128:
        warnings.append("이미지 해상도가 낮아 예측 신뢰도가 떨어질 수 있습니다.")
    if brightness < 30:
        warnings.append("이미지가 어두워 모델이 얼굴 특징을 충분히 보기 어려울 수 있습니다.")
    if brightness > 230:
        warnings.append("이미지가 지나치게 밝아 세부 질감 정보가 약할 수 있습니다.")
    if sharpness < 20:
        warnings.append("이미지가 흐릿해 예측 결과가 불안정할 수 있습니다.")
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        warnings.append("종횡비가 일반적인 얼굴 이미지와 달라 결과가 흔들릴 수 있습니다.")

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "brightness": brightness,
        "sharpness": sharpness,
        "warnings": warnings,
    }


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

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("최종 판정", predicted_label)
    result_col2.metric("예측 확률", f"{score * 100:.1f}%")
    result_col3.metric("신뢰도", level)

    if level == "높음":
        st.success(level_message)
    elif level == "보통":
        st.warning(level_message)
    else:
        st.error(level_message)

    st.subheader("후보별 확률")
    plot_df = results_df.copy()
    plot_df["표시 라벨"] = plot_df["정규화 라벨"] + " (" + plot_df["원본 라벨"].astype(str) + ")"
    fig = px.bar(
        plot_df,
        x="확률(%)",
        y="표시 라벨",
        orientation="h",
        color="정규화 라벨",
        text=plot_df["확률(%)"].map(lambda value: f"{value:.1f}%"),
        title="모델 후보별 예측 확률",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results_df.assign(**{"확률(%)": results_df["확률(%)"].round(2)}), use_container_width=True)

    with st.expander("원본 모델 출력 확인"):
        st.json(results)

    with st.expander("발표/보고서용 캡처 포인트", expanded=False):
        st.write("- 입력 이미지와 품질 경고")
        st.write("- 최종 판정, 예측 확률, 신뢰도")
        st.write("- 후보별 확률 그래프")
        st.write("- Ollama 해설을 사용하는 경우 해설 문장")

    st.subheader("LLM 결과 해설")
    use_ollama = st.checkbox("Ollama 해설 사용", value=False)
    if use_ollama:
        status = check_ollama_status()
        if not status["available"]:
            st.warning("Ollama 서버에 연결할 수 없습니다. 터미널에서 `ollama serve`를 실행하고 `ollama pull llama3.2`로 모델을 설치해 주세요.")
            st.caption(status.get("error"))
        else:
            default_model = "llama3.2"
            model_options = status["models"] or [default_model]
            model_name = st.selectbox("Ollama 모델", model_options, index=0)
            with st.expander("Ollama 프롬프트 미리보기", expanded=False):
                preview_prompt = build_explanation_prompt(
                    pred_label=predicted_label,
                    score=score,
                    confidence=level,
                    quality_warnings=st.session_state.get("last_quality_warnings", []),
                    candidate_results=results_df.to_dict("records"),
                )
                st.code(preview_prompt, language="text")
            if st.button("해설 생성"):
                with st.spinner("Ollama가 해설을 생성하는 중입니다."):
                    explanation = generate_ollama_explanation(preview_prompt, model_name=model_name)
                if explanation["ok"]:
                    st.info(explanation["text"])
                else:
                    st.error("Ollama 해설 생성에 실패했습니다.")
                    st.caption(explanation["error"])


def render_single_prediction_tab():
    tab_file, tab_url, tab_demo = st.tabs(["📁 파일 업로드", "🔗 URL 입력", "🧪 데모 샘플"])
    image = None
    input_source = None

    with tab_file:
        uploaded_file = st.file_uploader("얼굴 이미지 파일을 업로드해 주세요.", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                input_source = f"업로드 파일: {uploaded_file.name}"
            except Exception:
                st.error("업로드한 이미지를 열 수 없습니다. 다른 이미지 파일을 사용해 주세요.")

    with tab_url:
        url = st.text_input("이미지 URL을 입력해 주세요.")
        if url:
            with st.spinner("URL에서 이미지를 불러오는 중입니다."):
                image_from_url = fetch_image(url)
            if image_from_url is None:
                st.error("이미지를 불러오지 못했습니다. URL 만료, 외부 서버 차단, 네트워크 문제를 확인해 주세요.")
            else:
                image = image_from_url
                input_source = f"URL: {url}"

    with tab_demo:
        demo_images = load_demo_images()
        if demo_images:
            selected_demo = st.selectbox("로컬 데모 샘플 선택", demo_images, format_func=lambda p: str(p.relative_to(APP_DIR)))
            if selected_demo:
                try:
                    image = Image.open(selected_demo).convert("RGB")
                    input_source = f"데모 샘플: {selected_demo.relative_to(APP_DIR)}"
                except Exception:
                    st.error("선택한 데모 이미지를 열 수 없습니다.")
        else:
            st.info("`data/demo_samples/real`, `data/demo_samples/fake` 폴더에 이미지를 넣으면 발표용 샘플 선택 UI가 활성화됩니다.")

    if image is None:
        st.warning("이미지를 업로드하거나 URL을 입력하면 판별을 시작할 수 있습니다.")
        return

    st.markdown("---")
    st.subheader("1. 입력 이미지 확인")
    st.caption(input_source)

    image_col, quality_col = st.columns([1.2, 1])
    with image_col:
        st.image(image, caption="입력 이미지", use_container_width=True)

    quality = inspect_image_quality(image)
    st.session_state["last_quality_warnings"] = quality["warnings"]
    with quality_col:
        q1, q2 = st.columns(2)
        q1.metric("해상도", f"{quality['width']} × {quality['height']}")
        q2.metric("종횡비", f"{quality['aspect_ratio']:.2f}")
        q3, q4 = st.columns(2)
        q3.metric("밝기", f"{quality['brightness']:.1f}")
        q4.metric("선명도", f"{quality['sharpness']:.1f}")

        if quality["warnings"]:
            st.warning("입력 이미지 품질 경고")
            for warning in quality["warnings"]:
                st.write(f"- {warning}")
        else:
            st.success("입력 이미지 품질에 큰 경고가 없습니다.")

    st.markdown("---")
    st.subheader("2. 모델 판별")
    if not st.button("판별 시작", type="primary"):
        st.info("이미지를 확인하신 뒤 `판별 시작` 버튼을 눌러 주세요.")
        return

    try:
        classifier = load_img_model()
    except Exception as e:
        st.error("모델을 불러오지 못했습니다.")
        st.write("인터넷 연결, HuggingFace 접근 가능 여부, `torch`/`transformers` 설치 상태를 확인해 주세요.")
        with st.expander("오류 상세"):
            st.exception(e)
        return

    try:
        results = classifier(image)
    except Exception as e:
        st.error("이미지 예측 중 오류가 발생했습니다.")
        with st.expander("오류 상세"):
            st.exception(e)
        return

    if not results:
        st.error("모델이 예측 결과를 반환하지 않았습니다.")
        return

    render_prediction_results(results)


def render_eval_tab():
    st.subheader("샘플 성능 평가")
    st.caption("CSV의 `image_url`과 실제 `label`을 이용해 일부 샘플만 평가합니다. 네트워크와 CPU 성능에 따라 시간이 걸릴 수 있습니다.")

    try:
        df = _load_metadata()
    except FileNotFoundError:
        st.error("평가에 필요한 데이터 파일이 없습니다.")
        st.code(data_missing_message(), language="text")
        return
    except Exception as e:
        st.error("메타데이터를 불러오는 중 오류가 발생했습니다.")
        with st.expander("오류 상세"):
            st.exception(e)
        return

    missing_required = {"label", "image_url"} - set(df.columns)
    if missing_required:
        st.error(f"샘플 평가에 필요한 컬럼이 없습니다: {', '.join(sorted(missing_required))}")
        return

    opt1, opt2, opt3 = st.columns(3)
    sample_size = opt1.slider("평가 샘플 수", 10, 200, 30, step=10)
    balance_by_label = opt2.checkbox("FAKE/REAL 균형 샘플링", value=True)
    random_state = opt3.number_input("랜덤 시드", min_value=0, value=42, step=1)
    if "dataset_split" in df.columns:
        split_options = ["전체"] + sorted(df["dataset_split"].dropna().astype(str).unique().tolist())
        selected_split = st.selectbox("평가에 사용할 dataset_split", split_options)
        if selected_split != "전체":
            df = df[df["dataset_split"].astype(str) == selected_split].copy()
    max_workers = st.slider("이미지 다운로드 워커 수", 1, 16, 4)

    with st.expander("평가 방식 안내", expanded=False):
        st.write("- 전체 데이터가 아니라 선택한 샘플 수만 평가합니다.")
        st.write("- URL 다운로드 실패나 모델 예측 실패는 `error` 컬럼에 기록하고 전체 평가는 계속 진행합니다.")
        st.write("- 결과는 데이터셋 전체 성능이 아니라 발표/보고서용 소규모 점검 결과로 해석해야 합니다.")

    eval_clicked = st.button("샘플 평가 시작", type="primary", key="eval_start")
    if not eval_clicked:
        if "last_eval_results" in st.session_state:
            st.info("이전 평가 결과가 아래에 표시됩니다. 새로 실행하려면 `샘플 평가 시작`을 누르세요.")
        else:
            st.info("옵션을 확인한 뒤 `샘플 평가 시작`을 눌러 주세요.")

    if eval_clicked:
        try:
            df_sample = sample_evaluation_df(
                df,
                sample_size=sample_size,
                balance_by_label=balance_by_label,
                random_state=int(random_state),
            )
        except Exception as e:
            st.error(str(e))
            return

        try:
            classifier = load_img_model()
        except Exception as e:
            st.error("모델을 불러오지 못했습니다.")
            st.write("인터넷 연결, HuggingFace 접근 가능 여부, `torch`/`transformers` 설치 상태를 확인해 주세요.")
            with st.expander("오류 상세"):
                st.exception(e)
            return

        progress = st.progress(0)

        def update_progress(value):
            progress.progress(min(100, int(value * 100)))

        with st.spinner("샘플 이미지를 다운로드하고 모델 예측을 수행하는 중입니다."):
            eval_df = evaluate_image_sample(
                df_sample,
                classifier,
                max_workers=max_workers,
                progress_callback=update_progress,
            )
        metrics = compute_eval_metrics(eval_df)
        st.session_state["last_eval_results"] = eval_df
        st.session_state["last_eval_metrics"] = metrics
        st.success("샘플 평가가 완료되었습니다.")

    eval_df = st.session_state.get("last_eval_results")
    metrics = st.session_state.get("last_eval_metrics")
    if eval_df is None or metrics is None:
        return

    st.markdown("---")
    st.subheader("평가 결과 요약")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("평가 시도", f"{metrics['attempted']:,}")
    m2.metric("성공 평가", f"{metrics['success']:,}")
    m3.metric("실패", f"{metrics['failed']:,}")
    m4.metric("정확도", "-" if metrics["accuracy"] is None else f"{metrics['accuracy'] * 100:.1f}%")
    m5.metric("평균 확률", "-" if metrics["avg_score"] is None else f"{metrics['avg_score'] * 100:.1f}%")

    if metrics["success"] == 0:
        st.error("성공적으로 평가된 이미지가 없습니다. URL 상태와 네트워크 연결을 확인해 주세요.")
        st.dataframe(eval_df, use_container_width=True)
        return

    confusion = metrics["confusion_matrix"]
    st.subheader("Confusion Matrix")
    if not confusion.empty:
        fig = px.imshow(confusion, text_auto=True, aspect="auto", title="실제 라벨 × 예측 라벨")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(confusion, use_container_width=True)

    class_summary = metrics["class_summary"]
    if not class_summary.empty:
        st.subheader("클래스별 요약")
        show_summary = class_summary.copy()
        show_summary["accuracy"] = (show_summary["accuracy"] * 100).round(1)
        show_summary["avg_score"] = (show_summary["avg_score"] * 100).round(1)
        st.dataframe(show_summary, use_container_width=True)

    st.subheader("오분류 사례")
    wrong_df = eval_df[(eval_df["error"].isna()) & (eval_df["is_correct"] == False)]
    if len(wrong_df):
        st.dataframe(
            wrong_df[["image_id", "true_label", "pred_label", "raw_pred_label", "score", "image_url", "image_path"]],
            use_container_width=True,
        )
    else:
        st.success("성공 평가 샘플 중 오분류 사례가 없습니다.")

    if len(wrong_df):
        with st.expander("오분류 이미지 미리보기", expanded=False):
            preview_rows = wrong_df.head(6).reset_index(drop=True)
            preview_cols = st.columns(3)
            for idx, row in preview_rows.iterrows():
                with preview_cols[idx % 3]:
                    if pd.notna(row.get("image_path")):
                        st.image(str(row.get("image_path")), use_container_width=True)
                    st.caption(f"실제: {row.get('true_label')} / 예측: {row.get('pred_label')} / 확률: {float(row.get('score', 0)) * 100:.1f}%")

    with st.expander("전체 평가 결과 보기", expanded=False):
        st.dataframe(eval_df, use_container_width=True)

    st.download_button(
        "평가 결과 CSV 다운로드",
        data=eval_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="eval_results_sample.csv",
        mime="text/csv",
    )

    report_col1, report_col2 = st.columns(2)
    with report_col1:
        if st.button("보고서용 평가 결과 저장"):
            reports_dir = APP_DIR / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            eval_path = reports_dir / "eval_results_sample.csv"
            summary_path = reports_dir / "eval_summary.json"
            eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
            summary_payload = {
                "attempted": int(metrics["attempted"]),
                "success": int(metrics["success"]),
                "failed": int(metrics["failed"]),
                "accuracy": None if metrics["accuracy"] is None else float(metrics["accuracy"]),
                "avg_score": None if metrics["avg_score"] is None else float(metrics["avg_score"]),
                "wrong_count": int(((eval_df["error"].isna()) & (eval_df["is_correct"] == False)).sum()),
                "status": "saved",
            }
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"저장 완료: {eval_path}")
    with report_col2:
        if st.button("평가 결과를 보고서에 반영"):
            eval_path = APP_DIR / "reports" / "eval_results_sample.csv"
            summary_path = APP_DIR / "reports" / "eval_summary.json"
            if not eval_path.exists():
                st.warning("먼저 `보고서용 평가 결과 저장`을 눌러 평가 결과 파일을 생성해 주세요.")
            else:
                try:
                    report_path = sync_eval_to_report(eval_path, find_project_report_path(APP_DIR), summary_path)
                    st.success(f"보고서 반영 완료: {report_path}")
                except Exception as e:
                    st.error("보고서 반영 중 오류가 발생했습니다.")
                    st.exception(e)


st.title("🕵️ 모델 · 서비스 (AI 활용 이미지 판별)")
st.caption("이미지 제작 과정에 AI가 활용되었을 가능성을 판별하고, 일부 샘플 기준 성능을 점검합니다.")
render_project_notice()

st.info(
    "이 서비스는 영상 기반 탐지기가 아니라, 현재 데이터 특성에 맞춘 이미지 기반 "
    "AI 활용 가능성 판별 MVP입니다. 결과는 보조 판단용으로만 사용해야 합니다."
)

with st.expander("사용 방법", expanded=False):
    st.markdown(
        """
        1. `단일 이미지 판별` 탭에서 파일 업로드, URL 입력, 또는 로컬 데모 샘플로 이미지를 선택합니다.
        2. 입력 이미지의 해상도, 밝기, 선명도 품질 지표를 확인합니다.
        3. `판별 시작` 버튼을 누르면 사전학습 이미지 분류 모델이 결과를 반환합니다.
        4. `샘플 성능 평가` 탭에서 CSV의 일부 샘플을 대상으로 실제 라벨과 모델 예측을 비교할 수 있습니다.
        """
    )

single_tab, eval_tab = st.tabs(["🖼️ 단일 이미지 판별", "📊 샘플 성능 평가"])
with single_tab:
    render_single_prediction_tab()

with eval_tab:
    render_eval_tab()

st.markdown("---")
st.warning(
    "주의: 이 결과는 사전학습 모델의 추론 결과이며 실제 AI 활용 여부, 법적 판단, 악의적 조작 여부를 보장하지 않습니다. "
    "URL 출처 편향, 데이터 도메인 차이, 이미지 품질에 따라 일반화 성능이 제한될 수 있습니다."
)