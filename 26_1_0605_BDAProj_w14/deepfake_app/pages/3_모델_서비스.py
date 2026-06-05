# pages/3_모델_서비스.py — 3차 작업: 입력 → 예측 → 결과 (이 프로젝트의 핵심)
# 이 데이터는 '이미지'이므로 템플릿의 경로 B(사전학습 ViT pipeline)를 사용합니다.
#   (경로 A 머신러닝 / C 텍스트 / D LLM 은 이 프로젝트에 해당 없음)
# 목표(MVP): 사용자가 이미지 입력 → 모델이 FAKE/REAL 반환 → 화면에 표시. 동작부터 시킨 뒤 꾸미기.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import torch
from transformers import pipeline
from PIL import Image

from src.data_loader import fetch_image

st.title("🕵️ 모델 · 서비스 (딥페이크 판별)")

# =====================================================================
# 경로 B) 이미지 분류 — 사전학습 ViT pipeline  (배운 곳: 11·12·13주)
# =====================================================================
# 주의: 13주차 google/vit-base-patch16-224 는 ImageNet 분류기라 FAKE/REAL을 모름.
#       → 딥페이크 탐지 전용 사전학습 모델을 그대로(파인튜닝 없이) 사용한다.
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"


@st.cache_resource(show_spinner="모델 로드 중... (최초 1회)")
def load_img_model():
    device = 0 if torch.cuda.is_available() else -1   # GPU(8GB)면 0, 없으면 CPU
    return pipeline("image-classification", model=MODEL_NAME, device=device)


st.caption(f"모델: `{MODEL_NAME}` — 사전학습 ViT 기반")

# 입력: 파일 업로드 또는 URL
tab_file, tab_url = st.tabs(["📁 파일 업로드", "🔗 URL 입력"])
image = None
with tab_file:
    up = st.file_uploader("얼굴 이미지", type=["jpg", "jpeg", "png"])
    if up:
        image = Image.open(up).convert("RGB")
with tab_url:
    url = st.text_input("이미지 URL")
    if url:
        image = fetch_image(url)
        if image is None:
            st.error("이미지를 불러오지 못했습니다. URL을 확인해 주세요.")

# 예측 → 결과 표시
if image is not None:
    clf = load_img_model()
    results = clf(image)
    top = results[0]

    col_img, col_res = st.columns(2)
    col_img.image(image, caption="입력 이미지", use_container_width=True)
    col_res.metric("판정", top["label"].upper(), f"{top['score']*100:.1f}%")
    col_res.bar_chart({r["label"]: r["score"] for r in results})

    # TODO(여유 시): LLM/Ollama로 판정 근거 한 줄 해설 추가
else:
    st.info("이미지를 업로드하거나 URL을 입력하면 FAKE/REAL을 판별합니다.")
