"""
src/feature_pipeline.py
오프라인(터미널) 또는 노트북에서 전체 데이터(또는 큰 샘플)에 대해
이미지 다운로드 -> 캐시 저장 -> features.batch_extract_features 실행 -> parquet/csv 저장

사용법 예:
python -m src.feature_pipeline --out data/features_all.parquet --chunk 500 --workers 8

의존성: pandas, tqdm, (선택) facenet_pytorch, opencv-python
"""
from pathlib import Path
from typing import List, Optional
import os
import argparse
import math
import time

import pandas as pd
from tqdm import tqdm

from src.data_loader import load_data, download_images_bulk, CACHE_DIR, cache_size_info
from src.features import batch_extract_features, save_features


def run_full_feature_extraction(out_path: str, chunk_size: int = 500, max_workers: int = 8, limit: Optional[int] = None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(None)
    if limit is not None:
        df = df.head(limit)

    n = len(df)
    if n == 0:
        raise RuntimeError("메타 데이터가 비어있습니다.")

    parts = []
    start_time = time.time()
    for start in tqdm(range(0, n, chunk_size), desc="chunks"):
        chunk = df.iloc[start : start + chunk_size].copy().reset_index(drop=True)
        urls = chunk.get("image_url", pd.Series([None] * len(chunk))).fillna("").tolist()
        # download images for this chunk to cache
        paths = download_images_bulk(urls, max_workers=max_workers)
        chunk["image_path"] = [str(p) if p is not None else None for p in paths]
        # extract features (parallel inside)
        feats_df = batch_extract_features(chunk, image_col="image_path", nrows=None, n_workers=max_workers)
        if feats_df.empty:
            continue
        # combine chunk metadata (reset index) and features
        parts.append(feats_df)
        # save intermediate
        tmp_path = out_path.with_suffix(f".part{len(parts)}.parquet")
        feats_df.to_parquet(tmp_path, index=False)

    if not parts:
        raise RuntimeError("어떤 chunk에서도 특성 추출 실패")

    # concat and save final
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(out_path, index=False)
    elapsed = time.time() - start_time
    print(f"완료: {len(full)} rows saved to {out_path} in {elapsed:.1f}s")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--chunk", type=int, default=500, help="chunk size per batch")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers for download/extract")
    parser.add_argument("--limit", type=int, default=None, help="optional limit on number of rows to process")
    args = parser.parse_args()
    run_full_feature_extraction(args.out, chunk_size=args.chunk, max_workers=args.workers, limit=args.limit)
