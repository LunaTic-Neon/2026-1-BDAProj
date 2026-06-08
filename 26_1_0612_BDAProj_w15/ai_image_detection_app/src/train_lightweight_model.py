from pathlib import Path
from typing import Optional
import argparse
import json

import pandas as pd


LEAKAGE_COLS = {
    "image_id",
    "image_url",
    "image_path",
    "face_path",
    "dataset_split",
    "resolution",
    "date_collected",
    "version",
    "year",
    "confidence_score",
    "category",
    "source",
    "fake_method",
    "detection_difficulty",
    "domain",
    "label_numeric",
    "meta_width",
    "meta_height",
    "meta_total_pixels",
    "meta_aspect_ratio",
    "width",
    "height",
    "aspect",
    "iq_width",
    "iq_height",
    "download_error",
    "iq_reason",
    "face_error",
}


def _load_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _safe_metric_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    labels = ["FAKE", "REAL"]
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy),
        "labels": labels,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
    }


def prepare_training_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "label" not in df.columns:
        raise ValueError("학습 파일에 label 컬럼이 필요합니다.")
    y = df["label"].astype(str).str.upper()
    valid = y.isin(["FAKE", "REAL"])
    df = df[valid].copy()
    y = y[valid].copy()

    numeric_cols = [
        col
        for col in df.columns
        if col not in LEAKAGE_COLS
        and col != "label"
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    embedding_cols = [col for col in df.columns if col.startswith("embedding_") and pd.api.types.is_numeric_dtype(df[col])]
    if embedding_cols:
        numeric_cols = embedding_cols
    if not numeric_cols:
        raise ValueError("학습에 사용할 숫자형 특징 컬럼이 없습니다.")
    x = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    return x, y, numeric_cols


def _safe_train_test_split(x: pd.DataFrame, y: pd.Series, random_state: int):
    from sklearn.model_selection import train_test_split

    if y.nunique() < 2:
        raise ValueError("학습에는 최소 2개 클래스(FAKE/REAL)가 필요합니다.")
    if len(y) < 8 or y.value_counts().min() < 3:
        return x, x, y, y
    return train_test_split(
        x,
        y,
        test_size=max(0.25, 2 / len(y)),
        random_state=random_state,
        stratify=y,
    )


def train_lightweight_model(
    feature_path: str | Path,
    out_path: str | Path = "models/lightweight_model.joblib",
    model_type: str = "logistic_regression",
    test_split_name: str = "test",
    random_state: int = 42,
) -> tuple[dict, Path]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = _load_features(feature_path)
    x, y, feature_cols = prepare_training_matrix(df)

    if "dataset_split" in df.columns and df["dataset_split"].notna().any():
        split_values = df.loc[x.index, "dataset_split"].astype(str)
        train_mask = split_values != test_split_name
        test_mask = split_values == test_split_name
        if train_mask.sum() >= 2 and test_mask.sum() >= 2:
            x_train, x_test = x[train_mask], x[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            if y_train.nunique() < 2:
                x_train, x_test, y_train, y_test = _safe_train_test_split(x, y, random_state)
        else:
            x_train, x_test, y_train, y_test = _safe_train_test_split(x, y, random_state)
    else:
        x_train, x_test, y_train, y_test = _safe_train_test_split(x, y, random_state)

    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        model = estimator
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = _safe_metric_dict(y_test, pred)
    metrics.update(
        {
            "model_type": model_type,
            "feature_path": str(feature_path),
            "feature_count": len(feature_cols),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "feature_columns": feature_cols,
        }
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({"model": model, "feature_columns": feature_cols, "metrics": metrics}, out_path)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    summary_path = reports_dir / "lightweight_eval_summary.json"
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"true_label": y_test.values, "pred_label": pred})
    pred_df.to_csv(reports_dir / "lightweight_eval_results.csv", index=False, encoding="utf-8-sig")
    return metrics, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="models/lightweight_model.joblib")
    parser.add_argument("--model-type", default="logistic_regression", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--test-split-name", default="test")
    args = parser.parse_args()
    result, model_path = train_lightweight_model(
        args.features,
        out_path=args.out,
        model_type=args.model_type,
        test_split_name=args.test_split_name,
    )
    print(json.dumps({"model_path": str(model_path), **result}, ensure_ascii=False, indent=2))