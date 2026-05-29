# src/features.py — 정제·특성 엔지니어링 (2차 작업에서 채움)
import pandas as pd


def url_domain(df: pd.DataFrame) -> pd.Series:
    """image_url에서 도메인만 추출 (예: images.unsplash.com).

    REAL(Unsplash 실사)과 FAKE(아바타 API) 출처를 구분하는 EDA에 유용.
    """
    return df["image_url"].str.extract(r"https?://([^/]+)")[0]


# TODO(2차 작업):
#   - 결측 처리 (fake_method 등)
#   - 범주형 인코딩 / 파생 변수
#   - 모델 입력용 전처리 함수
