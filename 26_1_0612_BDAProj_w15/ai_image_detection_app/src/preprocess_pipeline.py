from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_loader import download_images_for_df
from src.face_preprocess import detect_and_crop_for_df
from src.features import add_resolution_features, batch_extract_features, save_features, url_domain
from src.image_quality import filter_valid_images


def run_preprocess_and_feature_extraction(
    df: pd.DataFrame,
    sample_n: Optional[int] = None,
    max_workers: int = 8,
    use_quality_check: bool = True,
    use_face_crop: bool = True,
    require_face: bool = False,
    min_width: int = 64,
    min_height: int = 64,
    min_sharpness: float = 20.0,
    min_brightness: float = 10.0,
    max_brightness: float = 245.0,
    out_dir: str | Path = "data",
    prefix: str = "features_sample",
) -> tuple[pd.DataFrame, dict, str]:
    if sample_n is not None:
        df_proc = df.head(int(sample_n)).copy()
    else:
        df_proc = df.copy()

    if "image_url" in df_proc.columns:
        df_proc = df_proc.assign(domain=url_domain(df_proc))
    df_proc = add_resolution_features(df_proc)

    df_proc = download_images_for_df(
        df_proc,
        url_col="image_url",
        image_col="image_path",
        max_workers=max_workers,
    )

    if use_quality_check:
        df_proc = filter_valid_images(
            df_proc,
            image_col="image_path",
            min_width=min_width,
            min_height=min_height,
            min_sharpness=min_sharpness,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            require_face=require_face,
            max_workers=max_workers,
        )
        feature_base = df_proc[df_proc["iq_pass"] == True].copy()
    else:
        feature_base = df_proc[df_proc["download_ok"] == True].copy()

    if use_face_crop and len(feature_base):
        feature_base = detect_and_crop_for_df(
            feature_base,
            image_col="image_path",
            id_col=("image_id" if "image_id" in feature_base.columns else None),
            require_face=require_face,
            n_workers=max_workers,
        )

    if require_face and "face_found" in feature_base.columns:
        feature_base = feature_base[feature_base["face_found"] == True].copy()

    feature_image_col = "image_path"
    if use_face_crop and "face_path" in feature_base.columns and feature_base["face_path"].notna().any():
        feature_image_col = "face_path"

    if len(feature_base):
        features_df = batch_extract_features(
            feature_base,
            image_col=feature_image_col,
            nrows=None,
            n_workers=max_workers,
        )
    else:
        features_df = pd.DataFrame()

    summary = {
        "input_rows": int(len(df_proc)),
        "download_ok": int(df_proc.get("download_ok", pd.Series(dtype=bool)).fillna(False).sum()),
        "download_failed": int((df_proc.get("download_ok", pd.Series(dtype=bool)).fillna(False) == False).sum()),
        "quality_checked": bool(use_quality_check),
        "quality_pass": int(df_proc.get("iq_pass", pd.Series(dtype=bool)).fillna(False).sum()) if use_quality_check else None,
        "feature_rows": int(len(features_df)),
        "face_crop_used": bool(use_face_crop),
        "face_found": int(feature_base.get("face_found", pd.Series(dtype=bool)).fillna(False).sum()) if "face_found" in feature_base.columns else None,
        "feature_image_col": feature_image_col,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}_{ts}.csv"
    if not features_df.empty:
        save_features(features_df, out_path)
    else:
        features_df.to_csv(out_path, index=False)
    return features_df, summary, str(out_path)