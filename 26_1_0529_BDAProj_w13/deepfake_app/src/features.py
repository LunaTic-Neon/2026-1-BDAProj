# src/features.py — 정제·특성 엔지니어링 (2차 작업에서 채움)
import pandas as pd

 
def url_domain(df: pd.DataFrame) -> pd.Series:
    """image_url에서 도메인만 추출 (예: images.unsplash.com).

    REAL(Unsplash 실사)과 FAKE(아바타 API) 출처를 구분하는 EDA에 유용.
    """
    return df["image_url"].str.extract(r"https?://([^/]+)")[0]


# 아래는 이미지 파일/객체로부터 추출하는 기본 특성 유틸
from PIL import Image, ImageStat
import numpy as np
from typing import Union, Dict, Any
from pathlib import Path

try:
    import cv2
    _has_cv2 = True
except Exception:
    _has_cv2 = False


def _open_image(input: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(input, Image.Image):
        return input.convert("RGB")
    p = Path(input)
    return Image.open(p).convert("RGB")


def avg_brightness(pil_img: Image.Image) -> float:
    img = pil_img.convert("L")
    stat = ImageStat.Stat(img)
    return float(stat.mean[0])


def avg_color(pil_img: Image.Image) -> tuple:
    stat = ImageStat.Stat(pil_img)
    return tuple(float(x) for x in stat.mean[:3])


def color_std(pil_img: Image.Image) -> tuple:
    stat = ImageStat.Stat(pil_img)
    return tuple(float(x) for x in stat.stddev[:3])


def sharpness_score(pil_img: Image.Image) -> float:
    """선명도 점수(Variance of Laplacian). cv2가 있으면 Laplacian, 없으면 numpy gradient로 근사.

    값이 높을수록 더 선명함(blur 낮음).
    """
    if _has_cv2:
        arr = np.array(pil_img.convert("L"))
        lap = cv2.Laplacian(arr, cv2.CV_64F)
        return float(lap.var())
    # fallback: numpy gradient variance
    arr = np.array(pil_img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    grad_mag_sq = gx ** 2 + gy ** 2
    return float(grad_mag_sq.var())


def extract_basic_features(input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
    """이미지(파일 경로 또는 PIL.Image)를 받아 기본 특성 딕셔너리 반환.

    반환 항목: width, height, aspect, brightness, avg_r/g/b, std_r/g/b, sharpness, mean_pixel, std_pixel
    """
    try:
        img = _open_image(input)
    except Exception:
        return {}
    w, h = img.size
    arr = np.array(img)
    mean_pixel = float(arr.mean())
    std_pixel = float(arr.std())

    r_mean, g_mean, b_mean = avg_color(img)
    r_std, g_std, b_std = color_std(img)

    features = {
        "width": int(w),
        "height": int(h),
        "aspect": float(w) / h if h else None,
        "brightness": avg_brightness(img),
        "avg_r": r_mean,
        "avg_g": g_mean,
        "avg_b": b_mean,
        "std_r": r_std,
        "std_g": g_std,
        "std_b": b_std,
        "sharpness": sharpness_score(img),
        "mean_pixel": mean_pixel,
        "std_pixel": std_pixel,
    }
    return features


def batch_extract_features(df: pd.DataFrame, image_col: str = "image_path", nrows: int | None = None) -> pd.DataFrame:
    """데이터프레임의 이미지 경로 컬럼에 대해 특성 추출을 수행하고 결과 데이터프레임을 반환합니다.

    - image_col: 파일 시스템의 경로(예: data/image_cache/xxxx.jpg)를 가리키는 컬럼명.
    - nrows: 일부만 처리할 때 사용.
    """
    if nrows is not None:
        df = df.head(nrows)
    records = []
    for idx, row in df.iterrows():
        path = row.get(image_col)
        feats = extract_basic_features(path)
        if feats:
            feats["_idx"] = idx
            records.append(feats)
    if not records:
        return pd.DataFrame()
    feats_df = pd.DataFrame.from_records(records).set_index("_idx")
    # 원래 df와 인덱스로 합치기
    return df.join(feats_df, how="left")


# 사용 예시(리포지토리 문서에 추가 권장):
# from src.features import extract_basic_features, batch_extract_features
# df_with_paths = df.copy(); df_with_paths['image_path'] = df_with_paths['image_url'].map(lambda u: cached_path_for_url(u))
# df_features = batch_extract_features(df_with_paths, image_col='image_path', nrows=100)
