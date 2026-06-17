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

MTCNN = None
RetinaFace = None
_has_mtcnn = False
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


def _analysis_gray_array(pil_img: Image.Image, size: int = 192) -> np.ndarray:
    resample = getattr(Image, "Resampling", Image).BILINEAR
    gray = pil_img.convert("L").resize((size, size), resample)
    return np.asarray(gray, dtype=np.float32) / 255.0


def texture_entropy(pil_img: Image.Image) -> float:
    gray = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def edge_density(pil_img: Image.Image) -> float:
    gray_u8 = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    if _has_cv2:
        try:
            edges = cv2.Canny(gray_u8, 80, 160)
            return float((edges > 0).mean())
        except Exception:
            pass
    gray = gray_u8.astype(np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    threshold = float(mag.mean() + mag.std())
    return float((mag > threshold).mean())


def high_frequency_ratio(pil_img: Image.Image) -> float:
    gray = _analysis_gray_array(pil_img)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spectrum) ** 2
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    high_mask = radius > min(h, w) * 0.25
    total_power = float(power.sum())
    if total_power == 0:
        return 0.0
    return float(power[high_mask].sum() / total_power)


def saturation_stats(pil_img: Image.Image) -> tuple[float, float]:
    hsv = np.asarray(pil_img.convert("HSV"), dtype=np.float32)
    saturation = hsv[:, :, 1] / 255.0
    return float(saturation.mean()), float(saturation.std())


def colorfulness_score(pil_img: Image.Image) -> float:
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    std_root = np.sqrt(float(rg.std() ** 2 + yb.std() ** 2))
    mean_root = np.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2))
    return float(std_root + 0.3 * mean_root)


def extract_ai_detection_features(input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
    img = _open_image(input)
    if img is None:
        return {}
    feats = extract_basic_features(img)
    width = feats.get("width", 0) or 0
    height = feats.get("height", 0) or 0
    saturation_mean, saturation_std = saturation_stats(img)
    feats.update(
        {
            "megapixels": float(width * height) / 1_000_000,
            "texture_entropy": texture_entropy(img),
            "edge_density": edge_density(img),
            "high_frequency_ratio": high_frequency_ratio(img),
            "saturation_mean": saturation_mean,
            "saturation_std": saturation_std,
            "colorfulness": colorfulness_score(img),
        }
    )
    feats["detail_score"] = float(feats["texture_entropy"] * (1.0 + feats["edge_density"]))
    return feats


def _feature_value(features: Dict[str, Any], key: str, digits: int = 3) -> str:
    value = features.get(key)
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def ai_feature_interpretations(features: Dict[str, Any]) -> list[Dict[str, str]]:
    megapixels = float(features.get("megapixels", 0) or 0)
    sharpness = float(features.get("sharpness", 0) or 0)
    entropy = float(features.get("texture_entropy", 0) or 0)
    edge = float(features.get("edge_density", 0) or 0)
    high_freq = float(features.get("high_frequency_ratio", 0) or 0)
    colorfulness = float(features.get("colorfulness", 0) or 0)
    saturation_std = float(features.get("saturation_std", 0) or 0)
    face_ratio = float(features.get("face_area_ratio", 0) or 0)

    rows = []
    rows.append(
        {
            "특성": "해상도",
            "값": f"{int(features.get('width', 0))}x{int(features.get('height', 0))} ({megapixels:.3f}MP)",
            "왜 보는가": "현재 데이터셋은 REAL이 1080x1080, FAKE가 작은 이미지인 경우가 많습니다.",
            "해석": "FAKE 쪽 신호" if megapixels < 0.1 else "REAL 쪽 신호" if megapixels >= 0.5 else "중간",
        }
    )
    rows.append(
        {
            "특성": "선명도",
            "값": _feature_value(features, "sharpness", 1),
            "왜 보는가": "작거나 흐린 이미지는 얼굴 디테일이 줄어 판별이 어려워집니다.",
            "해석": "낮음: FAKE/저품질 신호" if sharpness < 80 else "높음: 디테일 많음",
        }
    )
    rows.append(
        {
            "특성": "질감 복잡도",
            "값": _feature_value(features, "texture_entropy", 3),
            "왜 보는가": "실사 사진은 배경, 머리카락, 피부 질감 등으로 정보량이 커지는 편입니다.",
            "해석": "낮음: 단순한 이미지 신호" if entropy < 6.2 else "높음: 실사 디테일 신호",
        }
    )
    rows.append(
        {
            "특성": "엣지 밀도",
            "값": _feature_value(features, "edge_density", 3),
            "왜 보는가": "윤곽선과 세부 경계가 얼마나 많은지 봅니다.",
            "해석": "낮음: 매끈한 이미지 신호" if edge < 0.05 else "높음: 세부 경계 많음",
        }
    )
    rows.append(
        {
            "특성": "고주파 비율",
            "값": _feature_value(features, "high_frequency_ratio", 3),
            "왜 보는가": "작은 질감과 압축 흔적 같은 미세 패턴을 봅니다.",
            "해석": "낮음: 디테일 부족" if high_freq < 0.08 else "높음: 미세 패턴 많음",
        }
    )
    rows.append(
        {
            "특성": "색상 다양성",
            "값": _feature_value(features, "colorfulness", 1),
            "왜 보는가": "배경과 조명 변화가 많으면 색상 분포가 다양해집니다.",
            "해석": "낮음: 단조로운 색감" if colorfulness < 25 else "높음: 색 변화 많음",
        }
    )
    rows.append(
        {
            "특성": "채도 변화량",
            "값": _feature_value(features, "saturation_std", 3),
            "왜 보는가": "색이 한 톤으로 정리된 이미지인지 확인합니다.",
            "해석": "낮음: 균일한 색감" if saturation_std < 0.16 else "높음: 색감 변화 큼",
        }
    )
    rows.append(
        {
            "특성": "얼굴 영역 비율",
            "값": _feature_value(features, "face_area_ratio", 3),
            "왜 보는가": "생성/프로필 이미지는 얼굴이 중앙에 크게 잡히는 경우가 많습니다.",
            "해석": "큰 얼굴 중심 이미지" if face_ratio > 0.25 else "배경/상체 포함 가능성" if face_ratio > 0 else "얼굴 미검출",
        }
    )
    return rows


def ai_feature_signal_summary(features: Dict[str, Any]) -> Dict[str, Any]:
    fake_signals = []
    real_signals = []
    if float(features.get("megapixels", 0) or 0) < 0.1:
        fake_signals.append("해상도가 낮아 데이터셋의 FAKE 샘플 패턴과 가깝습니다.")
    else:
        real_signals.append("해상도가 충분해 데이터셋의 REAL 샘플 패턴과 가깝습니다.")
    if float(features.get("texture_entropy", 0) or 0) >= 6.2:
        real_signals.append("질감 정보량이 많아 실사 사진 쪽 신호가 있습니다.")
    else:
        fake_signals.append("질감 정보량이 낮아 단순하거나 압축된 이미지 신호가 있습니다.")
    if float(features.get("edge_density", 0) or 0) < 0.05:
        fake_signals.append("엣지 밀도가 낮아 매끈한 이미지 신호가 있습니다.")
    else:
        real_signals.append("엣지 밀도가 높아 머리카락/배경 같은 세부 경계가 많습니다.")
    if float(features.get("face_area_ratio", 0) or 0) > 0.25:
        fake_signals.append("얼굴이 화면에서 크게 잡혀 프로필/생성 이미지 패턴과 가깝습니다.")

    if len(fake_signals) > len(real_signals):
        verdict = "FAKE 쪽 특성 신호가 더 많음"
    elif len(real_signals) > len(fake_signals):
        verdict = "REAL 쪽 특성 신호가 더 많음"
    else:
        verdict = "특성 신호가 비슷함"
    return {"fake_signals": fake_signals, "real_signals": real_signals, "verdict": verdict}


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
