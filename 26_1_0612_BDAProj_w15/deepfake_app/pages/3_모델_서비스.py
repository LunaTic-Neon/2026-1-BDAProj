import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

from src.data_loader import fetch_image
from src.image_quality import avg_brightness, sharpness_score


MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = APP_DIR / "data" / "demo_samples"


@st.cache_resource(show_spinner="모델 로드 중입니다. 최초 실행에는 시간이 걸릴 수 있습니다.")
def load_img_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=MODEL_NAME, device=device)


def normalize_prediction_label(label: str) -> str:
    text = str(label).lower()
    fake_keywords = ["fake", "synthetic", "generated", "ai", "gan", "deepfake"]
    real_keywords = ["real", "authentic", "original", "human", "photo"]
    if any(keyword in text for keyword in fake_keywords):
        return "FAKE"
    if any(keyword in text for keyword in real_keywords):
        return "REAL"
    return str(label).upper()


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


def format_results(results: list) -> pd.DataFrame:
    records = []
    for result in results:
        raw_label = result.get("label", "UNKNOWN")
        score = float(result.get("score", 0.0))
        records.append(
            {
                "원본 라벨": raw_label,
                "정규화 라벨": normalize_prediction_label(raw_label),
                "확률": score,
                "확률(%)": score * 100,
            }
        )
    return pd.DataFrame(records).sort_values("확률", ascending=False).reset_index(drop=True)


def load_demo_images() -> list:
    if not DEMO_DIR.exists():
        return []
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(DEMO_DIR.glob(f"**/{ext}"))
    return sorted(image_paths)


st.title("🕵️ 모델 · 서비스 (AI 합성 이미지 판별)")
st.caption("얼굴 이미지가 실사 사진인지 생성/아바타 계열 이미지인지 판별합니다.")

st.info(
    "이 서비스는 영상 기반 딥페이크 탐지기가 아니라, 현재 데이터 특성에 맞춘 이미지 기반 "
    "실사/생성 이미지 구분 MVP입니다. 결과는 보조 판단용으로만 사용해야 합니다."
)

with st.expander("사용 방법", expanded=False):
    st.markdown(
        """
        1. 파일 업로드, URL 입력, 또는 로컬 데모 샘플 중 하나로 이미지를 선택합니다.
        2. 입력 이미지의 해상도, 밝기, 선명도 품질 지표를 확인합니다.
        3. `판별 시작` 버튼을 누르면 사전학습 이미지 분류 모델이 결과를 반환합니다.
        4. 최종 판정, 확률, 신뢰도, 후보별 확률을 함께 확인합니다.
        """
    )

st.markdown("---")

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
    st.stop()

st.markdown("---")
st.subheader("1. 입력 이미지 확인")
st.caption(input_source)

image_col, quality_col = st.columns([1.2, 1])
with image_col:
    st.image(image, caption="입력 이미지", use_container_width=True)

quality = inspect_image_quality(image)
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
    st.stop()

try:
    classifier = load_img_model()
except Exception as e:
    st.error("모델을 불러오지 못했습니다.")
    st.write("인터넷 연결, HuggingFace 접근 가능 여부, `torch`/`transformers` 설치 상태를 확인해 주세요.")
    with st.expander("오류 상세"):
        st.exception(e)
    st.stop()

try:
    results = classifier(image)
except Exception as e:
    st.error("이미지 예측 중 오류가 발생했습니다.")
    with st.expander("오류 상세"):
        st.exception(e)
    st.stop()

if not results:
    st.error("모델이 예측 결과를 반환하지 않았습니다.")
    st.stop()

results_df = format_results(results)
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

st.subheader("3. 후보별 확률")
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

st.markdown("---")
st.warning(
    "주의: 이 결과는 사전학습 모델의 추론 결과이며 실제 딥페이크 여부, 법적 판단, 악의적 조작 여부를 보장하지 않습니다. "
    "URL 출처 편향, 데이터 도메인 차이, 이미지 품질에 따라 일반화 성능이 제한될 수 있습니다."
)