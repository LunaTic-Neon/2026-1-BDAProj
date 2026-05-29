# src/data_loader.py — 데이터 적재 (이미지 데이터용, CSV+URL 방식)
# 템플릿의 data_loader_image.py(로컬 폴더 스캔)를 이 프로젝트에 맞게 교체한 것.
#   - 이 데이터는 이미지가 로컬이 아니라 CSV의 image_url(원격 주소)로 제공됨
#   - 메타데이터(CSV)만 캐싱해 두고, 이미지 자체는 필요할 때 1장씩 URL에서 받는다(8GB 보호)
import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# data/FINAL_DATASET.csv 기본 경로 (이 파일 기준 상대경로)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "FINAL_DATASET.csv")

# 라벨과 거의 1:1로 대응해 데이터 누수(leakage)를 일으키는 컬럼
#   → 모델 학습 feature로 쓰면 안 되고, EDA 분석 용도로만 사용
LEAKAGE_COLS = ["category", "source", "fake_method", "detection_difficulty"]


@st.cache_data   # CSV(메타데이터)만 읽는다 — 이미지는 fetch_image()로 필요할 때만
def load_data(nrows=None):
    """FINAL_DATASET.csv 적재.

    nrows: 개발 중 일부만 빠르게 불러올 때 지정 (가이드 FAQ 권장).
    """
    return pd.read_csv(DATA_PATH, nrows=nrows)


@st.cache_data(show_spinner="이미지 다운로드 중...")
def fetch_image(url, timeout=10):
    """image_url에서 이미지를 받아 RGB PIL 이미지로 반환. 실패 시 None."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None
