# pages/3_모델_서비스.py — 3차 작업: 입력 → 예측 → 결과 (이 프로젝트의 핵심)
# 아래 4가지 경로(A~D) 중 "내 데이터에 맞는 것 하나만" 골라 쓰세요. 나머지는 지우거나 주석.
# 목표(MVP): 사용자가 입력 → 모델/LLM이 결과 반환 → 화면에 표시. 동작부터 시킨 뒤 꾸미기.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from src.data_loader import load_data
# from src.features import clean, add_features

st.title("🤖 모델 · 서비스")

# =====================================================================
# 경로 A) 표 데이터 분류/회귀 — scikit-learn  (배운 곳: 10주)
# =====================================================================
st.header("경로 A · 머신러닝 (표 데이터)")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


@st.cache_resource  # ★ 모델 학습/로드는 한 번만
def train_model():
    df = load_data()
    # df = add_features(clean(df))            # TODO: 정제·특성 적용
    target = "target"                          # TODO: 예측할 컬럼 이름
    X = df.drop(columns=[target]).select_dtypes("number")  # 숫자 특성만 (간단 버전)
    X = X.fillna(X.median())                   # 결측은 중앙값으로 (RandomForest는 NaN 불가)
    y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        "accuracy": accuracy_score(yte, pred),
        "f1": f1_score(yte, pred, average="weighted"),
        "report": classification_report(yte, pred),
    }
    return model, list(X.columns), metrics


model, feat_cols, metrics = train_model()

c1, c2 = st.columns(2)
c1.metric("정확도", f"{metrics['accuracy']*100:.1f}%")
c2.metric("F1 (weighted)", f"{metrics['f1']*100:.1f}%")
with st.expander("상세 리포트"):
    st.text(metrics["report"])

st.subheader("예측해 보기")
inputs = {}
for col in feat_cols:
    inputs[col] = st.number_input(col, value=0.0)
if st.button("예측"):
    x = pd.DataFrame([inputs])[feat_cols]
    st.success(f"예측 결과: **{model.predict(x)[0]}**")


# =====================================================================
# 경로 B) 이미지 분류 — 사전학습 ViT pipeline  (배운 곳: 11·12주)
# =====================================================================
# from transformers import pipeline
# import torch
# from PIL import Image
#
# @st.cache_resource
# def load_img_model():
#     device = 0 if torch.cuda.is_available() else -1   # GPU(8GB)면 0, 없으면 CPU
#     return pipeline("image-classification",
#                     model="google/vit-base-patch16-224", device=device)
#
# clf = load_img_model()
# up = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
# if up:
#     img = Image.open(up).convert("RGB")
#     st.image(img, use_container_width=True)
#     for r in clf(img, top_k=5):
#         st.write(f"- {r['label']}: {r['score']*100:.1f}%")


# =====================================================================
# 경로 C) 텍스트 분석 — HuggingFace pipeline  (배운 곳: 5주)
# =====================================================================
# from transformers import pipeline
# @st.cache_resource
# def load_text_model():
#     return pipeline("sentiment-analysis")   # 또는 summarization, text-classification
# nlp = load_text_model()
# txt = st.text_area("문장 입력")
# if st.button("분석") and txt:
#     st.write(nlp(txt))


# =====================================================================
# 경로 D) LLM 서비스 — Ollama 로컬 LLM  (배운 곳: 6주)
# =====================================================================
# import ollama
# prompt = st.text_area("LLM에게 물어보기")
# if st.button("생성") and prompt:
#     res = ollama.chat(model="gemma3:4b",
#                       messages=[{"role": "user", "content": prompt}])
#     st.write(res["message"]["content"])
