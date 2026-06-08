from pathlib import Path
from typing import Optional
import argparse
import json
import time

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_loader import download_images_for_df, load_data


MODEL_CONFIGS = {
    "resnet18": {
        "builder": "resnet18",
        "weights": "ResNet18_Weights",
        "dim": 512,
    },
    "mobilenet_v3_small": {
        "builder": "mobilenet_v3_small",
        "weights": "MobileNet_V3_Small_Weights",
        "dim": 576,
    },
    "efficientnet_b0": {
        "builder": "efficientnet_b0",
        "weights": "EfficientNet_B0_Weights",
        "dim": 1280,
    },
}


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), idx


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_summary() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda": False, "vram_gb": None, "name": "CPU"}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "cuda": True,
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "name": props.name,
    }


def _load_torchvision_model(model_name: str):
    try:
        import torchvision.models as models
    except Exception as exc:
        raise RuntimeError("torchvision이 필요합니다. `pip install -r requirements.txt`를 실행해 주세요.") from exc

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"지원하지 않는 임베딩 모델입니다: {model_name}")

    cfg = MODEL_CONFIGS[model_name]
    weights_cls = getattr(models, cfg["weights"])
    weights = weights_cls.DEFAULT
    model = getattr(models, cfg["builder"])(weights=weights)

    if model_name == "resnet18":
        model.fc = torch.nn.Identity()
    elif model_name == "mobilenet_v3_small":
        model.classifier = torch.nn.Identity()
    elif model_name == "efficientnet_b0":
        model.classifier = torch.nn.Identity()

    model.eval()
    return model, weights.transforms(), cfg["dim"]


def _balanced_sample(df: pd.DataFrame, limit: Optional[int], random_state: int = 42, oversample_factor: int = 1) -> pd.DataFrame:
    if limit is None or "label" not in df.columns:
        return df
    target_limit = min(int(limit), len(df))
    sample_limit = min(max(target_limit, target_limit * max(1, oversample_factor)), len(df))
    df = df.copy()
    df["label"] = df["label"].astype(str).str.upper()
    labels = [label for label in ["FAKE", "REAL"] if label in set(df["label"])]
    if len(labels) < 2:
        return df.head(sample_limit)

    per_label = max(1, sample_limit // len(labels))
    parts = []
    for label in labels:
        part = df[df["label"] == label]
        parts.append(part.sample(n=min(per_label, len(part)), random_state=random_state))
    sampled = pd.concat(parts)
    if len(sampled) < sample_limit:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if len(remaining):
            sampled = pd.concat(
                [sampled, remaining.sample(n=min(sample_limit - len(sampled), len(remaining)), random_state=random_state + 1)]
            )
    return sampled.sample(frac=1, random_state=random_state).head(sample_limit).reset_index(drop=True)


def prepare_image_dataframe(limit: Optional[int] = None, max_workers: int = 4, balanced: bool = True) -> pd.DataFrame:
    df = load_data(None)
    required = {"image_url", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"임베딩 추출에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
    if balanced:
        df = _balanced_sample(df, limit, oversample_factor=4)
    elif limit is not None:
        df = df.head(int(limit)).copy()
    df = download_images_for_df(df, max_workers=max_workers)
    df = df[df["download_ok"] == True].copy()
    if df.empty:
        raise RuntimeError("다운로드에 성공한 이미지가 없어 임베딩을 추출할 수 없습니다.")
    if balanced and limit is not None:
        df = _balanced_sample(df, min(int(limit), len(df)), oversample_factor=1)
    return df.reset_index(drop=True)


def extract_image_embeddings(
    df: pd.DataFrame,
    model_name: str = "resnet18",
    batch_size: int = 4,
    out_path: str | Path = "data/embeddings_sample.parquet",
    num_workers: int = 0,
) -> tuple[pd.DataFrame, dict]:
    model, transform, dim = _load_torchvision_model(model_name)
    device = get_device()
    model = model.to(device)
    dataset = ImagePathDataset(df, transform)
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=False, num_workers=num_workers)

    records = []
    start = time.time()
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="embeddings"):
            images = images.to(device)
            outputs = model(images).detach().cpu()
            for vector, idx in zip(outputs, indices):
                row = df.iloc[int(idx)]
                record = {
                    "image_id": row.get("image_id", int(idx)),
                    "label": str(row.get("label", "")).upper(),
                    "dataset_split": row.get("dataset_split", None),
                    "image_path": row.get("image_path"),
                }
                for i, value in enumerate(vector.flatten().tolist()):
                    record[f"embedding_{i}"] = value
                records.append(record)

    embeddings = pd.DataFrame(records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        embeddings.to_csv(out_path, index=False)
    else:
        embeddings.to_parquet(out_path, index=False)

    summary = {
        "model_name": model_name,
        "embedding_dim": dim,
        "rows": len(embeddings),
        "batch_size": batch_size,
        "device": str(device),
        "gpu": gpu_summary(),
        "elapsed_seconds": round(time.time() - start, 2),
        "output_path": str(out_path),
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "embedding_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return embeddings, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="resnet18", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="data/embeddings_smoke.parquet")
    parser.add_argument("--no-balance", action="store_true", help="disable FAKE/REAL balanced sampling")
    args = parser.parse_args()
    df_images = prepare_image_dataframe(limit=args.limit, max_workers=args.workers, balanced=not args.no_balance)
    _, summary = extract_image_embeddings(
        df_images,
        model_name=args.model,
        batch_size=args.batch_size,
        out_path=args.out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))