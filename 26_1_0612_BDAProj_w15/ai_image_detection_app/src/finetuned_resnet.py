from pathlib import Path
from typing import Any
import argparse
import json
import time

import pandas as pd
from PIL import Image


APP_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_DIR / "models" / "ai_image_detector.pt"
REPORT_PATH = APP_DIR / "reports" / "resnet18_finetuned_summary.json"
LABELS = ["FAKE", "REAL"]
DEFAULT_BACKBONE = "mobilenet_v3_small"


def _torch_imports():
    import torch
    import torchvision.models as models
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights

    return torch, models, DataLoader, Dataset, transforms, MobileNet_V3_Small_Weights, ResNet18_Weights


def build_model(load_pretrained: bool = False, backbone_name: str = DEFAULT_BACKBONE, use_dropout_head: bool = True):
    """사전학습 backbone에 2-class 분류층만 붙입니다."""
    torch, models, _, _, _, MobileNet_V3_Small_Weights, ResNet18_Weights = _torch_imports()
    if backbone_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if load_pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        if use_dropout_head:
            model.classifier[-1] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.45),
                torch.nn.Linear(in_features, len(LABELS)),
            )
        else:
            model.classifier[-1] = torch.nn.Linear(in_features, len(LABELS))
    elif backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if load_pretrained else None
        model = models.resnet18(weights=weights)
        if use_dropout_head:
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.25),
                torch.nn.Linear(model.fc.in_features, len(LABELS)),
            )
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    else:
        raise ValueError(f"지원하지 않는 backbone입니다: {backbone_name}")
    return model


def image_transform(backbone_name: str = DEFAULT_BACKBONE):
    _, _, _, _, _, MobileNet_V3_Small_Weights, ResNet18_Weights = _torch_imports()
    if backbone_name == "mobilenet_v3_small":
        return MobileNet_V3_Small_Weights.DEFAULT.transforms()
    return ResNet18_Weights.DEFAULT.transforms()


def train_transform():
    """임의 이미지에 덜 과적합되도록 약한 증강을 적용합니다."""
    _, _, _, _, transforms, _, _ = _torch_imports()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageFrameDataset:
    def __init__(self, df: pd.DataFrame, transform):
        _, _, _, Dataset, _, _, _ = _torch_imports()

        class _Dataset(Dataset):
            def __init__(self, frame: pd.DataFrame, image_transform):
                self.frame = frame.reset_index(drop=True)
                self.transform = image_transform

            def __len__(self):
                return len(self.frame)

            def __getitem__(self, index):
                row = self.frame.iloc[index]
                image = Image.open(row["image_path"]).convert("RGB")
                x = self.transform(image)
                y = LABELS.index(str(row["label"]).upper())
                return x, y

        self.dataset = _Dataset(df, transform)


def _split_frame(df: pd.DataFrame, test_split_name: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    if "dataset_split" in df.columns and df["dataset_split"].notna().any():
        split = df["dataset_split"].astype(str).str.lower()
        train_df = df[split != test_split_name].copy()
        test_df = df[split == test_split_name].copy()
        if len(train_df) >= 10 and len(test_df) >= 4 and train_df["label"].nunique() == 2 and test_df["label"].nunique() == 2:
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_cached_image_dataframe(limit: int) -> pd.DataFrame:
    """이미 다운로드된 캐시 이미지만 이용해 학교 PC 실행 부담을 줄입니다."""
    from src.data_loader import CACHE_DIR, _url_to_name, load_data
    from src.image_embeddings import _balanced_sample

    df = load_data(None).copy()
    df["label"] = df["label"].astype(str).str.upper()
    df = df[df["label"].isin(LABELS)].copy()
    df["image_path"] = df["image_url"].map(lambda url: str(CACHE_DIR / _url_to_name(str(url))))
    df = df[df["image_path"].map(lambda path: Path(path).exists())].copy()
    if df.empty:
        raise RuntimeError("캐시된 이미지가 없습니다. 먼저 이미지 다운로드가 필요합니다.")
    return _balanced_sample(df, limit, random_state=42, oversample_factor=1).reset_index(drop=True)


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy),
        "labels": LABELS,
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": cm.tolist(),
    }


def train_finetuned_resnet(limit: int = 800, epochs: int = 3, batch_size: int = 16, workers: int = 6, lr: float = 0.001, cache_only: bool = False, backbone_name: str = DEFAULT_BACKBONE) -> dict[str, Any]:
    """MobileNetV3의 마지막 분류층만 짧게 학습합니다."""
    torch, _, DataLoader, _, _, _, _ = _torch_imports()
    from src.image_embeddings import prepare_image_dataframe

    start = time.time()
    print(f"[1/4] 이미지 준비 시작: limit={limit}", flush=True)
    if cache_only:
        df = prepare_cached_image_dataframe(limit=limit)
    else:
        df = prepare_image_dataframe(limit=limit, max_workers=workers, balanced=True)
    df = df[df["label"].astype(str).str.upper().isin(LABELS)].copy()
    train_df, test_df = _split_frame(df)
    print(f"[2/4] 데이터 준비 완료: train={len(train_df)}, test={len(test_df)}", flush=True)
    eval_transform = image_transform(backbone_name)
    aug_transform = train_transform()

    train_data = ImageFrameDataset(train_df, aug_transform).dataset
    test_data = ImageFrameDataset(test_df, eval_transform).dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(load_pretrained=True, backbone_name=backbone_name, use_dropout_head=True).to(device)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.") or name.startswith("classifier.")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.08)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.18)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_seen += len(labels)
        epoch_loss = total_loss / max(total_seen, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"[3/4] epoch {epoch}/{epochs} loss={epoch_loss:.4f}", flush=True)

    model.eval()
    print("[4/4] 평가 및 모델 저장 중", flush=True)
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    metrics = _metrics(y_true, y_pred)
    summary = {
        **metrics,
        "model_type": "frozen_backbone_light_head",
        "backbone_name": backbone_name,
        "pretrained_backbone": f"{backbone_name}_DEFAULT",
        "model_path": str(MODEL_PATH),
        "input_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": str(device),
        "cache_only": bool(cache_only),
        "history": history,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "labels": LABELS, "summary": summary}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_finetuned_model(model_path: Path = MODEL_PATH):
    """저장된 모델 구조를 자동 인식해 로드합니다."""
    torch, _, _, _, _, _, _ = _torch_imports()
    if not model_path.exists():
        raise FileNotFoundError(f"파인튜닝 모델이 없습니다: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    backbone_name = checkpoint.get("summary", {}).get("backbone_name", "resnet18")
    if any(key.startswith("classifier.") for key in state_dict):
        model = build_model(load_pretrained=False, backbone_name="mobilenet_v3_small", use_dropout_head=("classifier.3.1.weight" in state_dict))
    elif "fc.1.weight" in state_dict:
        model = build_model(load_pretrained=False, backbone_name="resnet18", use_dropout_head=True)
    elif "fc.weight" in state_dict:
        model = build_model(load_pretrained=False, backbone_name="resnet18", use_dropout_head=False)
    else:
        model = build_model(load_pretrained=False, backbone_name=backbone_name, use_dropout_head=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def predict_with_finetuned_resnet(image: Image.Image, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """단일 PIL 이미지를 FAKE/REAL로 예측합니다."""
    torch, _, _, _, _, _, _ = _torch_imports()
    model, checkpoint = load_finetuned_model(model_path)
    backbone_name = checkpoint.get("summary", {}).get("backbone_name", "resnet18")
    transform = image_transform(backbone_name)
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    score_map = {label: float(probs[index]) for index, label in enumerate(LABELS)}
    pred_label = max(score_map, key=score_map.get)
    return {
        "pred_label": pred_label,
        "score": score_map[pred_label],
        "score_map": score_map,
        "metrics": checkpoint.get("summary", {}),
        "model_path": str(model_path),
        "backbone": "resnet18_finetuned",
    }


def evaluate_finetuned_sample(df_sample: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
    """샘플 평가 탭에서 여러 이미지를 한 번에 평가합니다."""
    from src.data_loader import download_images_for_df
    from src.model_eval import normalize_prediction_label

    torch, _, _, _, _, _, _ = _torch_imports()
    model, checkpoint = load_finetuned_model(MODEL_PATH)
    backbone_name = checkpoint.get("summary", {}).get("backbone_name", "resnet18")
    transform = image_transform(backbone_name)
    df_paths = download_images_for_df(
        df_sample,
        url_col="image_url",
        image_col="image_path",
        chunk_size=max(1, len(df_sample)),
        max_workers=max_workers,
        timeout=10,
        retry=1,
    )

    records = []
    with torch.inference_mode():
        for idx, row in df_paths.iterrows():
            true_label = row.get("true_label", normalize_prediction_label(row.get("label", "")))
            record = {
                "image_id": row.get("image_id", idx),
                "image_url": row.get("image_url"),
                "true_label": true_label,
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
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                logits = model(transform(image).unsqueeze(0))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                score_map = {label: float(probs[index]) for index, label in enumerate(LABELS)}
                pred_label = max(score_map, key=score_map.get)
                record["pred_label"] = pred_label
                record["raw_pred_label"] = pred_label
                record["score"] = score_map[pred_label]
                record["is_correct"] = true_label == pred_label
            except Exception as exc:
                record["error"] = str(exc)
            records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=["mobilenet_v3_small", "resnet18"])
    args = parser.parse_args()
    result = train_finetuned_resnet(
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        cache_only=args.cache_only,
        backbone_name=args.backbone,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))