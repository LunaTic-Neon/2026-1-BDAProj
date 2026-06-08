# src/data_loader.py — 데이터 적재 (이미지 데이터용, CSV+URL 방식)
# 템플릿의 data_loader_image.py(로컬 폴더 스캔)를 이 프로젝트에 맞게 교체한 것.
#   - 이 데이터는 이미지가 로컬이 아니라 CSV의 image_url(원격 주소)로 제공됨
#   - 메타데이터(CSV)만 캐싱해 두고, 이미지 자체는 필요할 때 1장씩 URL에서 받는다(8GB 보호)
import os
from io import BytesIO
import hashlib
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# data/FINAL_DATASET.csv 기본 경로 (이 파일 기준 상대경로)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "FINAL_DATASET.csv")

# 라벨과 거의 1:1로 대응해 데이터 누수(leakage)를 일으키는 컬럼
#   → 모델 학습 feature로 쓰면 안 되고, EDA 분석 용도로만 사용
LEAKAGE_COLS = ["category", "source", "fake_method", "detection_difficulty"]


def data_missing_message() -> str:
    app_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    rel_path = Path("data") / "FINAL_DATASET.csv"
    return (
        "데이터 파일을 찾을 수 없습니다.\n\n"
        f"- 필요한 위치: {app_dir / rel_path}\n"
        "- 해결 방법: Kaggle 원본 CSV 또는 로컬 CSV를 ai_image_detection_app/data/FINAL_DATASET.csv 경로에 넣어 주세요.\n"
        "- 데이터 파일은 용량 문제로 git에서 제외될 수 있으므로 제출/실행 PC에서 별도로 배치해야 합니다."
    )


@st.cache_data   # CSV(메타데이터)만 읽는다 — 이미지는 fetch_image()로 필요할 때만
def load_data(nrows=None):
    """FINAL_DATASET.csv 적재.

    nrows: 개발 중 일부만 빠르게 불러올 때 지정 (가이드 FAQ 권장).
    """
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(data_missing_message())
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


# 이미지 캐시 디렉토리 (data/image_cache)
CACHE_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "data" / "image_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _url_to_name(url: str) -> str:
    """URL을 안전한 파일명으로 변환 (SHA1 + 확장자 추출).
    확장자가 없으면 .jpg 사용.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
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
            tmp = CACHE_DIR / (fname + ".tmp")
            img.save(tmp, format="JPEG")
            tmp.replace(dest)
            return dest
        except Exception:
            if attempt < retry:
                time.sleep(0.5 * (attempt + 1))
                continue
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


def clear_cache() -> dict:
    """이미지 캐시 전체를 삭제. 반환: removed_files 수."""
    removed = 0
    for p in CACHE_DIR.glob("*"):
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except Exception:
            continue
    return {"removed_files": removed}


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


def download_images_bulk(urls: List[str], max_workers: int = 8, timeout: int = 10, retry: int = 2) -> List[Optional[Path]]:
    """병렬로 여러 이미지를 다운로드하여 캐시에 저장. 순서 보존된 Path 리스트 반환(실패는 None)."""
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


import json
import random
from typing import Tuple


def normalize_label(df: pd.DataFrame, label_col: str = "label", mapping: Optional[dict] = None) -> pd.DataFrame:
    """라벨 정규화: 매핑을 적용하거나 간단한 규칙으로 'FAKE'/'REAL'로 변환 시도.

    매핑이 주어지면 먼저 적용하고, 그렇지 않으면 문자열 패턴(fake/real 등)을 통해 자동 매핑을 시도합니다.
    """
    if label_col not in df.columns:
        return df
    if mapping:
        df[label_col] = df[label_col].map(mapping).fillna(df[label_col])
        return df

    # 자동 매핑 시도
    unique_vals = df[label_col].dropna().unique()
    auto_map = {}
    for v in unique_vals:
        s = str(v).lower()
        if "fake" in s or "deep" in s or "synth" in s or "generated" in s:
            auto_map[v] = "FAKE"
        elif "real" in s or "auth" in s or "original" in s or "orig" in s:
            auto_map[v] = "REAL"
    if auto_map:
        df[label_col] = df[label_col].map(auto_map).fillna(df[label_col])
    return df


def save_report_json(report: dict, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(p)


def validate_metadata(df: pd.DataFrame,
                      date_cols: Optional[List[str]] = None,
                      required_cols: Optional[List[str]] = None,
                      dup_subset: Optional[List[str]] = None,
                      save_report: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """간단한 메타데이터 검증 및 정규화

    수행 내용:
      - 컬럼별 결측치 집계
      - 필수 컬럼 누락/결측 체크
      - 중복 제거(옵션)
      - 날짜 파싱(date_cols 목록이 주어지면 _parsed 컬럼 추가)
      - 라벨 정규화 시도

    반환: (cleaned_df, report_dict)
    """
    report = {}
    report["shape_before"] = df.shape
    report["missing_by_col"] = df.isnull().sum().to_dict()

    if required_cols:
        missing_required = {c: int(df[c].isnull().sum()) if c in df.columns else None for c in required_cols}
        report["missing_required"] = missing_required

    if dup_subset is None:
        dup_subset = ["image_url"] if "image_url" in df.columns else None
    if dup_subset is not None:
        dup_count = int(df.duplicated(subset=dup_subset).sum())
        report["duplicate_count"] = dup_count
        if dup_count > 0:
            df = df.drop_duplicates(subset=dup_subset).reset_index(drop=True)
    else:
        report["duplicate_count"] = 0

    # 날짜 파싱
    if date_cols:
        for dc in date_cols:
            if dc in df.columns:
                parsed = pd.to_datetime(df[dc], errors="coerce")
                df[f"{dc}_parsed"] = parsed
                report[f"{dc}_parsed_na"] = int(parsed.isnull().sum())

    # 라벨 정규화 시도
    if "label" in df.columns:
        before = df["label"].unique().tolist()
        df = normalize_label(df, "label")
        after = df["label"].unique().tolist()
        report["label_unique_before"] = before
        report["label_unique_after"] = after

    report["shape_after"] = df.shape

    if save_report:
        try:
            save_report_json(report, save_report)
        except Exception:
            pass

    return df, report


def _url_health_check(url: str, timeout: int = 5) -> dict:
    """단일 URL에 대해 빠르게 상태를 확인. HEAD 요청을 우선 시도하고 실패 시 GET으로 폴백.

    반환 dict: {ok: bool, status: Optional[int], content_type: Optional[str], error: Optional[str]}
    """
    if not url or not isinstance(url, str):
        return {"ok": False, "status": None, "content_type": None, "error": "invalid_url"}
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        status = getattr(resp, "status_code", None)
        ctype = resp.headers.get("content-type") if resp is not None else None
        # 4xx/5xx이면 GET으로 확인
        if status is None or status >= 400:
            resp2 = requests.get(url, stream=True, timeout=timeout)
            status = getattr(resp2, "status_code", None)
            ctype = resp2.headers.get("content-type") if resp2 is not None else ctype
        ok = (status is not None and status < 400)
        return {"ok": ok, "status": status, "content_type": ctype, "error": None}
    except Exception as e:
        return {"ok": False, "status": None, "content_type": None, "error": str(e)}


def validate_urls(df: pd.DataFrame, url_col: str = "image_url", sample_n: int = 0, max_workers: int = 8, timeout: int = 5) -> pd.DataFrame:
    """DataFrame의 URL 컬럼에 대해 병렬로 상태 검사 수행.

    sample_n>0이면 랜덤 샘플만 검사(개발 시 빠르게 동작 확인용).
    반환: 원본 df에 url_ok, url_status, url_content_type, url_error 컬럼을 추가한 복사본.
    """
    if url_col not in df.columns:
        raise ValueError(f"{url_col} not in dataframe")
    df_out = df.copy()
    idxs = list(range(len(df_out)))
    if sample_n and sample_n > 0:
        idxs = random.sample(idxs, min(sample_n, len(idxs)))

    results = {i: {"ok": None, "status": None, "content_type": None, "error": None} for i in idxs}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_url_health_check, df_out.at[i, url_col], timeout): i for i in idxs}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                res = fut.result()
                results[i] = res
            except Exception as e:
                results[i] = {"ok": False, "status": None, "content_type": None, "error": str(e)}

    # 컬럼 채우기 (샘플 검사인 경우 해당 인덱스만 채움)
    ok_col = []
    status_col = []
    ctype_col = []
    err_col = []
    for i in range(len(df_out)):
        r = results.get(i)
        if r is None:
            ok_col.append(None)
            status_col.append(None)
            ctype_col.append(None)
            err_col.append(None)
        else:
            ok_col.append(r.get("ok"))
            status_col.append(r.get("status"))
            ctype_col.append(r.get("content_type"))
            err_col.append(r.get("error"))
    df_out["url_ok"] = ok_col
    df_out["url_status"] = status_col
    df_out["url_content_type"] = ctype_col
    df_out["url_error"] = err_col
    return df_out


def download_images_for_df(df, url_col: str = "image_url", image_col: str = "image_path", chunk_size: int = 500,
                           max_workers: int = 8, timeout: int = 10, retry: int = 2, show_progress: bool = False):
    """데이터프레임의 URL 컬럼을 청크 단위로 병렬 다운로드하고 image_path 컬럼을 붙여 반환.

    반환: df_copy (image_col 컬럼이 추가되며 다운로드 실패는 None)
    """
    if url_col not in df.columns:
        raise ValueError(f"{url_col} not in dataframe")
    df_out = df.copy().reset_index(drop=True)
    n = len(df_out)
    if n == 0:
        df_out[image_col] = []
        df_out["download_ok"] = []
        df_out["download_error"] = []
        return df_out

    df_out[image_col] = None
    df_out["download_ok"] = False
    df_out["download_error"] = None

    prog = None
    if show_progress:
        try:
            import streamlit as st
            prog = st.progress(0)
        except Exception:
            prog = None

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        urls = df_out.loc[start:end - 1, url_col].tolist()
        results = download_images_bulk(urls, max_workers=max_workers, timeout=timeout, retry=retry)
        for i, p in enumerate(results, start=start):
            df_out.at[i, image_col] = str(p) if p is not None else None
            df_out.at[i, "download_ok"] = p is not None
            df_out.at[i, "download_error"] = None if p is not None else "download_failed"
        if prog is not None:
            prog.progress(min(100, int((end / n) * 100)))

    try:
        _ensure_cache_under_limit()
    except Exception:
        pass

    return df_out


def download_and_validate(df, url_col: str = "image_url", image_col: str = "image_path",
                          chunk_size: int = 500, max_workers: int = 8, timeout: int = 10, retry: int = 2,
                          iq_min_width: int = 64, iq_min_height: int = 64, iq_min_sharpness: float = 50.0,
                          iq_brightness_range: tuple = (10, 245), require_face: bool = False,
                          validate_sample: int = 0):
    """편의 함수: 다운로드 -> URL 상태 검사(선택적 샘플) -> 이미지 품질 검사

    반환: (df_with_paths, df_quality_report)
    """
    df_paths = download_images_for_df(df, url_col=url_col, image_col=image_col, chunk_size=chunk_size,
                                      max_workers=max_workers, timeout=timeout, retry=retry)

    if validate_sample and validate_sample > 0:
        try:
            df_checked = validate_urls(df_paths, url_col=url_col, sample_n=validate_sample, max_workers=min(8, max_workers))
            df_paths = df_checked
        except Exception:
            pass

    try:
        from .image_quality import filter_valid_images
        df_quality = filter_valid_images(df_paths, image_col=image_col, min_width=iq_min_width, min_height=iq_min_height,
                                         min_sharpness=iq_min_sharpness, min_brightness=iq_brightness_range[0],
                                         max_brightness=iq_brightness_range[1], require_face=require_face,
                                         max_workers=max_workers)
    except Exception:
        df_quality = df_paths.copy()
        df_quality["iq_pass"] = None

    return df_paths, df_quality
