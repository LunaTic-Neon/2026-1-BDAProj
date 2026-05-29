# data_loader_image.py — 이미지 데이터용 적재
# 사용법: 이 파일 내용을 src/data_loader.py 에 덮어쓰세요. (이미지_데이터_쓸_때/README.md 참고)
import os
import glob
from PIL import Image
import pandas as pd
import streamlit as st

IMG_DIR = "data/images"   # 하위에 클래스별 폴더 (data/images/cat, data/images/dog ...)

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@st.cache_data   # 메타데이터(경로·라벨·크기)만 스캔 — 이미지 자체는 필요할 때 1장씩 연다(8GB 보호)
def load_image_meta():
    rows = []
    for path in glob.glob(os.path.join(IMG_DIR, "*", "*")):
        if not path.lower().endswith(EXTS):
            continue
        label = os.path.basename(os.path.dirname(path))   # 폴더명 = 라벨
        try:
            with Image.open(path) as im:
                w, h, mode = im.width, im.height, im.mode
        except Exception:
            w = h = None
            mode = "손상"
        rows.append({
            "path": path, "label": label, "width": w, "height": h,
            "mode": mode, "size_kb": round(os.path.getsize(path) / 1024, 1),
        })
    return pd.DataFrame(rows)


# 2_시각화.py 등 다른 페이지 호환용: load_data()도 같은 메타데이터를 돌려준다.
# → 시각화 페이지가 width/height/size_kb/label 로 그대로 그래프를 그릴 수 있다.
def load_data():
    return load_image_meta()
