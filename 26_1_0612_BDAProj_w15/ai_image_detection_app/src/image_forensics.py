from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


def _open_image(input_image) -> Image.Image | None:
    try:
        if isinstance(input_image, Image.Image):
            return input_image.convert("RGB")
        path = Path(input_image)
        if path.exists():
            return Image.open(path).convert("RGB")
    except Exception:
        return None
    return None


def _sharpness_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if _HAS_CV2:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    return float(ImageStat.Stat(image.convert("L")).stddev[0])


def _detect_face_ratio(image: Image.Image) -> tuple[int, float]:
    if not _HAS_CV2:
        return 0, 0.0
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0, 0.0
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
        image_area = gray.shape[0] * gray.shape[1]
        face_area = sum(int(w) * int(h) for (_, _, w, h) in faces)
        return int(len(faces)), float(face_area / image_area) if image_area else 0.0
    except Exception:
        return 0, 0.0


def _texture_entropy(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def _edge_density(image: Image.Image) -> float:
    gray_u8 = np.asarray(image.convert("L"), dtype=np.uint8)
    if _HAS_CV2:
        edges = cv2.Canny(gray_u8, 80, 160)
        return float((edges > 0).mean())
    gray = gray_u8.astype(np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float((mag > (mag.mean() + mag.std())).mean())


def _high_frequency_ratio(image: Image.Image) -> float:
    resample = getattr(Image, "Resampling", Image).BILINEAR
    gray = image.convert("L").resize((192, 192), resample)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(spectrum) ** 2
    height, width = arr.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - height / 2) ** 2 + (xx - width / 2) ** 2)
    high_mask = radius > min(height, width) * 0.25
    total_power = float(power.sum())
    return float(power[high_mask].sum() / total_power) if total_power else 0.0


def _colorfulness(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    std_root = np.sqrt(float(rg.std() ** 2 + yb.std() ** 2))
    mean_root = np.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2))
    return float(std_root + 0.3 * mean_root)


def _saturation_stats(image: Image.Image) -> tuple[float, float]:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    saturation = hsv[:, :, 1] / 255.0
    return float(saturation.mean()), float(saturation.std())


def extract_ai_detection_features(input_image) -> dict[str, Any]:
    image = _open_image(input_image)
    if image is None:
        return {}
    width, height = image.size
    gray_stat = ImageStat.Stat(image.convert("L"))
    rgb_stat = ImageStat.Stat(image.convert("RGB"))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    face_count, face_area_ratio = _detect_face_ratio(image)
    saturation_mean, saturation_std = _saturation_stats(image)
    texture_entropy = _texture_entropy(image)
    edge_density = _edge_density(image)
    return {
        "width": int(width),
        "height": int(height),
        "megapixels": float(width * height) / 1_000_000,
        "aspect_ratio": float(width / height) if height else 0.0,
        "brightness": float(gray_stat.mean[0]),
        "sharpness": _sharpness_score(image),
        "mean_pixel": float(arr.mean()),
        "std_pixel": float(arr.std()),
        "avg_r": float(rgb_stat.mean[0]),
        "avg_g": float(rgb_stat.mean[1]),
        "avg_b": float(rgb_stat.mean[2]),
        "std_r": float(rgb_stat.stddev[0]),
        "std_g": float(rgb_stat.stddev[1]),
        "std_b": float(rgb_stat.stddev[2]),
        "texture_entropy": texture_entropy,
        "edge_density": edge_density,
        "high_frequency_ratio": _high_frequency_ratio(image),
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "colorfulness": _colorfulness(image),
        "face_count": face_count,
        "face_area_ratio": face_area_ratio,
        "detail_score": float(texture_entropy * (1.0 + edge_density)),
    }


def _value(features: dict[str, Any], key: str, digits: int = 3) -> str:
    try:
        return f"{float(features.get(key, 0)):.{digits}f}"
    except Exception:
        return "-"


def ai_feature_interpretations(features: dict[str, Any]) -> list[dict[str, str]]:
    megapixels = float(features.get("megapixels", 0) or 0)
    sharpness = float(features.get("sharpness", 0) or 0)
    entropy = float(features.get("texture_entropy", 0) or 0)
    edge = float(features.get("edge_density", 0) or 0)
    high_freq = float(features.get("high_frequency_ratio", 0) or 0)
    colorfulness = float(features.get("colorfulness", 0) or 0)
    saturation_std = float(features.get("saturation_std", 0) or 0)
    face_ratio = float(features.get("face_area_ratio", 0) or 0)
    return [
        {
            "특성": "해상도",
            "값": f"{int(features.get('width', 0))}x{int(features.get('height', 0))} ({megapixels:.3f}MP)",
            "왜 보는가": "현재 데이터셋은 REAL이 큰 사진, FAKE가 작은 프로필 이미지인 경우가 많습니다.",
            "해석": "FAKE 쪽 신호" if megapixels < 0.1 else "REAL 쪽 신호" if megapixels >= 0.5 else "중간",
        },
        {
            "특성": "선명도",
            "값": _value(features, "sharpness", 1),
            "왜 보는가": "작거나 흐린 이미지는 얼굴 디테일이 줄어듭니다.",
            "해석": "낮음: FAKE/저품질 신호" if sharpness < 80 else "높음: 디테일 많음",
        },
        {
            "특성": "질감 복잡도",
            "값": _value(features, "texture_entropy", 3),
            "왜 보는가": "실사는 배경, 머리카락, 피부 질감 등 정보량이 큰 편입니다.",
            "해석": "낮음: 단순한 이미지 신호" if entropy < 6.2 else "높음: 실사 디테일 신호",
        },
        {
            "특성": "엣지 밀도",
            "값": _value(features, "edge_density", 3),
            "왜 보는가": "윤곽선과 세부 경계가 얼마나 많은지 봅니다.",
            "해석": "낮음: 매끈한 이미지 신호" if edge < 0.05 else "높음: 세부 경계 많음",
        },
        {
            "특성": "고주파 비율",
            "값": _value(features, "high_frequency_ratio", 3),
            "왜 보는가": "작은 질감과 압축 흔적 같은 미세 패턴을 봅니다.",
            "해석": "낮음: 디테일 부족" if high_freq < 0.08 else "높음: 미세 패턴 많음",
        },
        {
            "특성": "색상 다양성",
            "값": _value(features, "colorfulness", 1),
            "왜 보는가": "배경과 조명 변화가 많으면 색상 분포가 다양해집니다.",
            "해석": "낮음: 단조로운 색감" if colorfulness < 25 else "높음: 색 변화 많음",
        },
        {
            "특성": "채도 변화량",
            "값": _value(features, "saturation_std", 3),
            "왜 보는가": "색이 한 톤으로 정리된 이미지인지 확인합니다.",
            "해석": "낮음: 균일한 색감" if saturation_std < 0.16 else "높음: 색감 변화 큼",
        },
        {
            "특성": "얼굴 영역 비율",
            "값": _value(features, "face_area_ratio", 3),
            "왜 보는가": "프로필/생성 이미지는 얼굴이 중앙에 크게 잡히는 경우가 많습니다.",
            "해석": "큰 얼굴 중심 이미지" if face_ratio > 0.25 else "배경/상체 포함 가능성" if face_ratio > 0 else "얼굴 미검출",
        },
    ]


def ai_feature_signal_summary(features: dict[str, Any]) -> dict[str, Any]:
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
