from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image

from src.data_loader import download_images_for_df


def normalize_prediction_label(label: str) -> str:
    text = str(label).lower()
    fake_keywords = ["fake", "synthetic", "generated", "ai", "gan", "deepfake"]
    real_keywords = ["real", "authentic", "original", "human", "photo"]
    if any(keyword in text for keyword in fake_keywords):
        return "FAKE"
    if any(keyword in text for keyword in real_keywords):
        return "REAL"
    return str(label).upper()


def format_prediction_results(results: list) -> pd.DataFrame:
    records = []
    for result in results:
        raw_label = result.get("label", "UNKNOWN")
        score = float(result.get("score", 0.0))
        records.append(
            {
                "원본 라벨": raw_label,
                "정규화 라벨": normalize_prediction_label(raw_label),
                "확률": score,
                "확률(%)": score * 100,
            }
        )
    return pd.DataFrame(records).sort_values("확률", ascending=False).reset_index(drop=True)


def sample_evaluation_df(
    df: pd.DataFrame,
    sample_size: int = 30,
    balance_by_label: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    required_cols = {"label", "image_url"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"평가에 필요한 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")

    eval_base = df.dropna(subset=["label", "image_url"]).copy()
    eval_base["true_label"] = eval_base["label"].map(normalize_prediction_label)
    eval_base = eval_base[eval_base["true_label"].isin(["FAKE", "REAL"])]
    if eval_base.empty:
        raise ValueError("FAKE/REAL로 정규화 가능한 평가 샘플이 없습니다.")

    sample_size = min(int(sample_size), len(eval_base))
    if not balance_by_label:
        return eval_base.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    labels = sorted(eval_base["true_label"].unique())
    per_label = max(1, sample_size // len(labels))
    sampled_parts = []
    for label in labels:
        part = eval_base[eval_base["true_label"] == label]
        take_n = min(per_label, len(part))
        sampled_parts.append(part.sample(n=take_n, random_state=random_state))

    sampled = pd.concat(sampled_parts)
    if len(sampled) < sample_size:
        remaining = eval_base.drop(index=sampled.index, errors="ignore")
        if len(remaining):
            extra_n = min(sample_size - len(sampled), len(remaining))
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra_n, random_state=random_state + 1)],
            )
    return sampled.sample(frac=1, random_state=random_state).head(sample_size).reset_index(drop=True)


def evaluate_image_sample(
    df_sample: pd.DataFrame,
    classifier,
    max_workers: int = 4,
    timeout: int = 10,
    retry: int = 1,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    df_paths = download_images_for_df(
        df_sample,
        url_col="image_url",
        image_col="image_path",
        chunk_size=max(1, len(df_sample)),
        max_workers=max_workers,
        timeout=timeout,
        retry=retry,
    )

    records = []
    total = len(df_paths)
    for idx, row in df_paths.iterrows():
        record = {
            "image_id": row.get("image_id", idx),
            "image_url": row.get("image_url"),
            "true_label": row.get("true_label", normalize_prediction_label(row.get("label", ""))),
            "pred_label": None,
            "raw_pred_label": None,
            "score": None,
            "is_correct": None,
            "image_path": row.get("image_path"),
            "error": None,
        }

        image_path = row.get("image_path")
        if not image_path or not Path(str(image_path)).exists():
            record["error"] = "download_failed"
            records.append(record)
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            results = classifier(image)
            if not results:
                record["error"] = "empty_prediction"
            else:
                top = format_prediction_results(results).iloc[0]
                record["raw_pred_label"] = top["원본 라벨"]
                record["pred_label"] = top["정규화 라벨"]
                record["score"] = float(top["확률"])
                record["is_correct"] = record["true_label"] == record["pred_label"]
        except Exception as e:
            record["error"] = str(e)

        records.append(record)
        if progress_callback:
            progress_callback((idx + 1) / total)

    return pd.DataFrame(records)


def compute_eval_metrics(eval_df: pd.DataFrame) -> dict:
    if eval_df.empty:
        return {
            "attempted": 0,
            "success": 0,
            "failed": 0,
            "accuracy": None,
            "avg_score": None,
            "confusion_matrix": pd.DataFrame(),
            "class_summary": pd.DataFrame(),
        }

    success_mask = eval_df["error"].isna() & eval_df["pred_label"].notna()
    success_df = eval_df[success_mask].copy()
    attempted = len(eval_df)
    success = len(success_df)
    failed = attempted - success
    accuracy = float(success_df["is_correct"].mean()) if success else None
    avg_score = float(success_df["score"].mean()) if success else None

    confusion = pd.crosstab(
        success_df["true_label"],
        success_df["pred_label"],
        rownames=["실제 라벨"],
        colnames=["예측 라벨"],
        dropna=False,
    )

    if success:
        class_summary = (
            success_df.groupby("true_label")
            .agg(
                sample_count=("is_correct", "size"),
                correct_count=("is_correct", "sum"),
                avg_score=("score", "mean"),
            )
            .reset_index()
        )
        class_summary["accuracy"] = class_summary["correct_count"] / class_summary["sample_count"]
    else:
        class_summary = pd.DataFrame()

    return {
        "attempted": attempted,
        "success": success,
        "failed": failed,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "confusion_matrix": confusion,
        "class_summary": class_summary,
    }