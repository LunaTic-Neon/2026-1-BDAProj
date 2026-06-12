from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_MODEL_PATHS = [
    Path(__file__).resolve().parents[1] / "models" / "lightweight_embedding_smoke.joblib",
    Path(__file__).resolve().parents[1] / "models" / "lightweight_smoke.joblib",
]


def find_available_model() -> Path | None:
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path
    return None


def load_lightweight_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else find_available_model()
    if path is None or not path.exists():
        raise FileNotFoundError("사용 가능한 경량 학습 모델(.joblib)을 찾지 못했습니다.")
    bundle = joblib.load(path)
    bundle["model_path"] = str(path)
    return bundle


def extract_single_embedding(image: Image.Image, model_name: str = "resnet18") -> np.ndarray:
    try:
        import torch
        from src.image_embeddings import _load_torchvision_model, get_device
    except Exception as exc:
        raise RuntimeError("추가학습 모델 추론에는 torch와 torchvision이 필요합니다.") from exc

    model, transform, _ = _load_torchvision_model(model_name)
    device = get_device()
    model = model.to(device)
    model.eval()

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor).detach().cpu().numpy().reshape(-1)
    return embedding


def predict_with_lightweight_model(image: Image.Image, bundle: dict[str, Any], model_name: str = "resnet18") -> dict[str, Any]:
    feature_columns = bundle.get("feature_columns", [])
    model = bundle["model"]
    metrics = bundle.get("metrics", {})

    embedding = extract_single_embedding(image, model_name=model_name)
    row = {f"embedding_{i}": float(v) for i, v in enumerate(embedding)}
    x = pd.DataFrame([row])

    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0.0
    x = x[feature_columns]

    pred = model.predict(x)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = list(model.classes_)
        score_map = {cls: float(score) for cls, score in zip(classes, proba)}
        score = score_map.get(pred, max(score_map.values()))
    else:
        score_map = {str(pred): 1.0}
        score = 1.0

    return {
        "pred_label": str(pred),
        "score": float(score),
        "score_map": score_map,
        "metrics": metrics,
        "model_path": bundle.get("model_path"),
        "feature_count": len(feature_columns),
        "backbone": model_name,
    }
