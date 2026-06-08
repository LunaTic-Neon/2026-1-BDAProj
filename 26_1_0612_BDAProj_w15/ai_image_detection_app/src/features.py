# src/features.py — 정제·특성 엔지니어링
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
from pathlib import Path
from typing import Union, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cv2
    _has_cv2 = True
except Exception:
    _has_cv2 = False

try:
    from facenet_pytorch import MTCNN
    _has_mtcnn = True
except Exception:
    _has_mtcnn = False

try:
    from retinaface import RetinaFace
    _has_retina = True
except Exception:
    _has_retina = False


def url_domain(df: pd.DataFrame) -> pd.Series:
    """image_url에서 도메인만 추출 (예: images.unsplash.com).

    REAL(Unsplash 실사)과 FAKE(아바타 API) 출처를 구분하는 EDA에 유용.
    """
    return df["image_url"].str.extract(r"https?://([^/]+)")[0]


def add_resolution_features(df: pd.DataFrame, resolution_col: str = "resolution") -> pd.DataFrame:
    df_out = df.copy()
    if resolution_col not in df_out.columns:
        return df_out

    parsed = df_out[resolution_col].astype(str).str.lower().str.extract(r"(?P<meta_width>\d+)\s*x\s*(?P<meta_height>\d+)")
    df_out["meta_width"] = pd.to_numeric(parsed["meta_width"], errors="coerce")
    df_out["meta_height"] = pd.to_numeric(parsed["meta_height"], errors="coerce")
    df_out["meta_total_pixels"] = df_out["meta_width"] * df_out["meta_height"]
    df_out["meta_aspect_ratio"] = df_out["meta_width"] / df_out["meta_height"]
    return df_out


def _open_image(input: Union[str, Path, Image.Image]) -> Optional[Image.Image]:
    try:
        if isinstance(input, Image.Image):
            return input.convert("RGB")
        p = Path(input)
        if p.exists():
            return Image.open(p).convert("RGB")
        # not a file, try treating as URL (not ideal here)
    except Exception:
        return None
    return None


def avg_brightness(pil_img: Image.Image) -> float:
    img = pil_img.convert("L")
    stat = ImageStat.Stat(img)
    return float(stat.mean[0])


def avg_color(pil_img: Image.Image) -> tuple:
    stat = ImageStat.Stat(pil_img)
    return tuple(float(x) for x in stat.mean[:3])


def color_std(pil_img: Image.Image) -> tuple:
    stat = ImageStat.Stat(pil_img)
    return tuple(float(x) for x in stat.stddev[:3])


def sharpness_score(pil_img: Image.Image) -> float:
    if _has_cv2:
        arr = np.array(pil_img.convert("L"))
        lap = cv2.Laplacian(arr, cv2.CV_64F)
        return float(lap.var())
    arr = np.array(pil_img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    grad_mag_sq = gx ** 2 + gy ** 2
    return float(grad_mag_sq.var())


_mtcnn_detector = None
if _has_mtcnn:
    try:
        _mtcnn_detector = MTCNN(keep_all=True)
    except Exception:
        _mtcnn_detector = None


# face detection using OpenCV Haar cascade if available
_cascade = None
if _has_cv2:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(cascade_path)
        if _cascade.empty():
            _cascade = None
    except Exception:
        _cascade = None


def detect_faces(pil_img: Image.Image) -> Dict[str, Any]:
    """Return face_count and face_area_ratio.
    우선 RetinaFace를 사용하고, 없으면 MTCNN, 없으면 OpenCV cascade로 폴백.
    """
    # RetinaFace 사용
    if _has_retina:
        try:
            arr = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
            resp = RetinaFace.detect_faces(arr)
            if not resp:
                return {"face_count": 0, "face_area_ratio": 0.0}
            h, w = pil_img.size[1], pil_img.size[0]
            face_area = 0
            count = 0
            for k, v in resp.items():
                face = v.get("facial_area")
                if face and len(face) == 4:
                    x1, y1, x2, y2 = face
                    fw = max(0, x2 - x1)
                    fh = max(0, y2 - y1)
                    face_area += fw * fh
                    count += 1
            area = w * h
            ratio = float(face_area) / area if area else 0.0
            return {"face_count": int(count), "face_area_ratio": float(ratio)}
        except Exception:
            pass

    # MTCNN 사용
    if _has_mtcnn and _mtcnn_detector is not None:
        try:
            boxes, probs = _mtcnn_detector.detect(pil_img)
            h, w = pil_img.size[1], pil_img.size[0]
            if boxes is None:
                return {"face_count": 0, "face_area_ratio": 0.0}
            face_area = 0
            for b in boxes:
                x1, y1, x2, y2 = b
                fw = max(0, x2 - x1)
                fh = max(0, y2 - y1)
                face_area += fw * fh
            area = w * h
            ratio = float(face_area) / area if area else 0.0
            return {"face_count": int(len(boxes)), "face_area_ratio": float(ratio)}
        except Exception:
            pass

    # fallback: OpenCV Haar cascade if available
    if _has_cv2 and _cascade is not None:
        try:
            arr = np.array(pil_img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            h, w = gray.shape
            area = w * h
            face_area = 0
            for (x, y, fw, fh) in faces:
                face_area += fw * fh
            ratio = float(face_area) / area if area else 0.0
            return {"face_count": int(len(faces)), "face_area_ratio": float(ratio)}
        except Exception:
            return {"face_count": 0, "face_area_ratio": 0.0}

    # detector 없음
    return {"face_count": 0, "face_area_ratio": 0.0}


def extract_basic_features(input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
    img = _open_image(input)
    if img is None:
        return {}
    w, h = img.size
    arr = np.array(img)
    mean_pixel = float(arr.mean())
    std_pixel = float(arr.std())
    r_mean, g_mean, b_mean = avg_color(img)
    r_std, g_std, b_std = color_std(img)
    face = detect_faces(img)
    feats = {
        "width": int(w),
        "height": int(h),
        "aspect": float(w) / h if h else None,
        "brightness": avg_brightness(img),
        "avg_r": r_mean,
        "avg_g": g_mean,
        "avg_b": b_mean,
        "std_r": r_std,
        "std_g": g_std,
        "std_b": b_std,
        "sharpness": sharpness_score(img),
        "mean_pixel": mean_pixel,
        "std_pixel": std_pixel,
        "face_count": face.get("face_count", 0),
        "face_area_ratio": face.get("face_area_ratio", 0.0),
    }
    return feats


def _process_row(idx: int, path: Union[str, Path], image_col: str) -> Dict[str, Any]:
    feats = extract_basic_features(path)
    feats["_idx"] = idx
    return feats


def batch_extract_features(df: pd.DataFrame, image_col: str = "image_path", nrows: Optional[int] = None, n_workers: int = 8) -> pd.DataFrame:
    """Extract features for images referenced in df[image_col]. Returns df joined with features.

    image_col should point to local cached image paths for best performance.
    """
    if nrows is not None:
        df_proc = df.head(nrows).copy()
    else:
        df_proc = df.copy()
    records = []
    paths = df_proc[image_col].tolist()
    # parallel execution
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(extract_basic_features, p): i for i, p in enumerate(paths)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                feats = fut.result()
                if feats:
                    feats["_idx"] = idx
                    records.append(feats)
            except Exception:
                continue
    if not records:
        return pd.DataFrame()
    feats_df = pd.DataFrame.from_records(records).set_index("_idx")
    feats_df.index = feats_df.index.astype(int)
    # align indices with original df_proc
    df_out = df_proc.reset_index(drop=True).join(feats_df, how="left")
    return df_out


# small convenience: save features to CSV
def save_features(df_features: pd.DataFrame, out_path: Union[str, Path]) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out, index=False)
    return str(out)


# End of features.py
