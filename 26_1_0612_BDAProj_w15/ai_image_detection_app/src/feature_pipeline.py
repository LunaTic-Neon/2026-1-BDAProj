"""
오프라인 전처리·특징추출 파이프라인.

사용 예:
python -m src.feature_pipeline --out data/features_all.parquet --chunk 500 --workers 8 --limit 1000 --quality-check --face-crop
"""
from pathlib import Path
from typing import Optional
import argparse
import json
import time

import pandas as pd
from tqdm import tqdm

from src.data_loader import load_data
from src.face_preprocess import detect_and_crop_for_df
from src.features import add_resolution_features, batch_extract_features, save_features, url_domain
from src.image_quality import filter_valid_images
from src.data_loader import download_images_for_df


def _save_output(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)


def run_full_feature_extraction(
    out_path: str,
    chunk_size: int = 500,
    max_workers: int = 8,
    limit: Optional[int] = None,
    quality_check: bool = True,
    face_crop: bool = False,
    require_face: bool = False,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(None)
    if limit is not None:
        df = df.head(limit)
    if len(df) == 0:
        raise RuntimeError("메타 데이터가 비어있습니다.")

    if "image_url" in df.columns:
        df = df.assign(domain=url_domain(df))
    df = add_resolution_features(df)

    parts = []
    failed_parts = []
    summary = {
        "input_rows": int(len(df)),
        "download_ok": 0,
        "download_failed": 0,
        "quality_check": bool(quality_check),
        "quality_pass": 0,
        "face_crop": bool(face_crop),
        "face_found": 0,
        "feature_rows": 0,
    }

    start_time = time.time()
    for start in tqdm(range(0, len(df), chunk_size), desc="chunks"):
        chunk = df.iloc[start : start + chunk_size].copy().reset_index(drop=True)
        chunk = download_images_for_df(chunk, max_workers=max_workers)
        summary["download_ok"] += int(chunk["download_ok"].fillna(False).sum())
        summary["download_failed"] += int((chunk["download_ok"].fillna(False) == False).sum())

        if quality_check:
            chunk = filter_valid_images(chunk, max_workers=max_workers)
            summary["quality_pass"] += int(chunk["iq_pass"].fillna(False).sum())
            feature_base = chunk[chunk["iq_pass"] == True].copy()
        else:
            feature_base = chunk[chunk["download_ok"] == True].copy()

        if face_crop and len(feature_base):
            feature_base = detect_and_crop_for_df(
                feature_base,
                image_col="image_path",
                id_col=("image_id" if "image_id" in feature_base.columns else None),
                require_face=require_face,
                n_workers=max_workers,
            )
            summary["face_found"] += int(feature_base["face_found"].fillna(False).sum())

        if require_face and "face_found" in feature_base.columns:
            feature_base = feature_base[feature_base["face_found"] == True].copy()

        image_col = "face_path" if face_crop and "face_path" in feature_base.columns and feature_base["face_path"].notna().any() else "image_path"
        feats_df = batch_extract_features(feature_base, image_col=image_col, n_workers=max_workers) if len(feature_base) else pd.DataFrame()
        if feats_df.empty:
            failed_parts.append(chunk)
            continue

        parts.append(feats_df)
        summary["feature_rows"] += int(len(feats_df))
        tmp_path = out_path.with_suffix(f".part{len(parts)}{out_path.suffix}")
        _save_output(feats_df, tmp_path)

    if not parts:
        full = pd.DataFrame()
        _save_output(full, out_path)
    else:
        full = pd.concat(parts, ignore_index=True)
        _save_output(full, out_path)

    if failed_parts:
        failed = pd.concat(failed_parts, ignore_index=True)
        failed.to_csv(reports_dir / "failed_feature_rows.csv", index=False)

    summary["elapsed_seconds"] = round(time.time() - start_time, 2)
    summary["output_path"] = str(out_path)
    summary["status"] = "ok" if len(full) else "no_features_extracted"
    with open(reports_dir / "feature_pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output parquet/csv path")
    parser.add_argument("--chunk", type=int, default=500, help="chunk size per batch")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers for download/extract")
    parser.add_argument("--limit", type=int, default=None, help="optional limit on number of rows to process")
    parser.add_argument("--quality-check", action="store_true", help="run image quality filtering")
    parser.add_argument("--face-crop", action="store_true", help="run face crop before feature extraction")
    parser.add_argument("--require-face", action="store_true", help="drop rows where face was not detected")
    args = parser.parse_args()
    run_full_feature_extraction(
        args.out,
        chunk_size=args.chunk,
        max_workers=args.workers,
        limit=args.limit,
        quality_check=args.quality_check,
        face_crop=args.face_crop,
        require_face=args.require_face,
    )