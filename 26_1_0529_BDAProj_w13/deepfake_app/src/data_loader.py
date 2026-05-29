# src/data_loader.py — 데이터 적재 (이미지 데이터용, CSV+URL 방식)
# 템플릿의 data_loader_image.py(로컬 폴더 스캔)를 이 프로젝트에 맞게 교체한 것.
#   - 이 데이터는 이미지가 로컬이 아니라 CSV의 image_url(원격 주소)로 제공됨
#   - 메타데이터(CSV)만 캐싱해 두고, 이미지 자체는 필요할 때 URL에서 받되 디스크에 캐시하여 재사용
 
import os
from io import BytesIO
import hashlib
from pathlib import Path
from typing import List, Optional
import time

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# data/FINAL_DATASET.csv 기본 경로 (이 파일 기준 상대경로)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "FINAL_DATASET.csv")

# 이미지 캐시 디렉토리
CACHE_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "data" / "image_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 라벨과 거의 1:1로 대응해 데이터 누수(leakage)를 일으키는 컬럼
#   → 모델 학습 feature로 쓰면 안 되고, EDA 분석 용도로만 사용
LEAKAGE_COLS = ["category", "source", "fake_method", "detection_difficulty"]


@st.cache_data   # CSV(메타데이터)만 읽는다 — 이미지는 fetch_image()로 필요할 때만
def load_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """FINAL_DATASET.csv 적재.

    nrows: 개발 중 일부만 빠르게 불러올 때 지정 (가이드 FAQ 권장).
    """
    return pd.read_csv(DATA_PATH, nrows=nrows)


def _url_to_name(url: str) -> str:
    """URL을 안전한 파일명으로 변환 (SHA1 + 확장자 추출).
    확장자가 없으면 .jpg 사용.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    # 쿼리 제거 후 확장자 추출
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1]
    if not ext or len(ext) > 5:
        ext = ".jpg"
    return f"{h}{ext}"


def _download_single(url: str, timeout: int = 10, retry: int = 2, session: Optional[requests.Session] = None) -> Optional[Path]:
    """단일 URL을 다운로드하여 캐시에 저장한 뒤 저장 경로를 반환. 실패 시 None."""
    if not url or not isinstance(url, str):
        return None
    fname = _url_to_name(url)
    dest = CACHE_DIR / fname
    if dest.exists():
        return dest

    sess = session or requests.Session()
    for attempt in range(retry + 1):
        try:
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            # 안전하게 저장 (임시파일 -> rename)
            tmp = CACHE_DIR / (fname + ".tmp")
            img.save(tmp)
            tmp.replace(dest)
            return dest
        except Exception:
            if attempt < retry:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None


def fetch_image(url: str, timeout: int = 10, retry: int = 2) -> Optional[Image.Image]:
    """image_url에서 이미지를 받아 RGB PIL 이미지로 반환. 디스크 캐시를 우선 사용.
    실패 시 None을 반환.
    """
    try:
        path = _url_to_name(url)
    except Exception:
        return None
    dest = CACHE_DIR / path
    if dest.exists():
        try:
            return Image.open(dest).convert("RGB")
        except Exception:
            # 캐시가 손상되었을 수 있으므로 삭제 후 재시도
            try:
                dest.unlink()
            except Exception:
                pass
    # 다운로드 시도
    p = _download_single(url, timeout=timeout, retry=retry)
    if p:
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None
    return None


def cache_size_info() -> dict:
    """캐시 디렉토리의 파일 수 및 총 용량(byte)을 반환."""
    total = 0
    count = 0
    for p in CACHE_DIR.glob("*"):
        if p.is_file():
            count += 1
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return {"count": count, "bytes": total}

# ---- 캐시 관리 기능 --------------------------------------------------
# 전역: 캐시 최대 바이트 수(없음이면 제한 없음)
CACHE_MAX_BYTES: Optional[int] = None


def set_cache_max_bytes(max_bytes: Optional[int]):
    """캐시 최대 크기(바이트)를 설정. None이면 제한 없음.

    예: set_cache_max_bytes(1024*1024*1024)  # 1GB
    """
    global CACHE_MAX_BYTES
    if max_bytes is None:
        CACHE_MAX_BYTES = None
    else:
        CACHE_MAX_BYTES = int(max_bytes)


def _ensure_cache_under_limit():
    """캐시 디렉토리가 설정된 최대 바이트를 넘으면 오래된 파일부터 삭제하여 제한을 맞춘다.
    삭제 기준: 파일의 최종 수정 시간(mtime).
    """
    if CACHE_MAX_BYTES is None:
        return
    info = cache_size_info()
    total = info.get("bytes", 0)
    if total <= CACHE_MAX_BYTES:
        return
    # 오래된 파일부터 삭제
    files = [p for p in CACHE_DIR.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)  # 오래된 순
    for p in files:
        try:
            size = p.stat().st_size
            p.unlink()
            total -= size
            if total <= CACHE_MAX_BYTES:
                break
        except Exception:
            continue


def clear_cache():
    """이미지 캐시 전체를 삭제. 복구 불가함.
    스트리밍 앱에서 캐시 용량이 커졌을 때 사용자 버튼으로 연결하기 좋음.
    """
    removed = 0
    for p in CACHE_DIR.glob("*"):
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except Exception:
            continue
    return {"removed_files": removed}


# 다운로드 후 캐시 정리 보강: bulk 함수 마지막 단계에서 호출
def download_images_bulk(urls: List[str], max_workers: int = 8, timeout: int = 10, retry: int = 2) -> List[Optional[Path]]:
    """병렬로 여러 이미지를 다운로드하여 캐시에 저장. 순서 보존된 Path 리스트 반환(실패는 None).
    사용 예: paths = download_images_bulk(df['image_url'].unique()[:100].tolist())
    """
    results = [None] * len(urls)
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_download_single, url, timeout, retry, session): idx for idx, url in enumerate(urls)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None
    # 다운로드 이후 캐시 용량 초과 시 오래된 파일을 삭제하여 제한을 맞춤
    try:
        _ensure_cache_under_limit()
    except Exception:
        pass
    return results
