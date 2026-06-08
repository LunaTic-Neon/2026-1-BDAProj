from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


def _load_cascade():
    try:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None, None
        return cv2, detector
    except Exception:
        return None, None


def _crop_single(image_path: str, out_dir: Path, stem: str, margin: float = 0.2, require_face: bool = False) -> dict:
    result = {"face_path": None, "face_bbox": None, "face_found": False, "face_error": None}
    if not image_path:
        result["face_error"] = "no_path"
        return result

    src = Path(str(image_path))
    if not src.exists():
        result["face_error"] = "path_not_found"
        return result

    cv2, detector = _load_cascade()
    if cv2 is None or detector is None:
        if require_face:
            result["face_error"] = "detector_unavailable"
            return result
        result["face_path"] = str(src)
        result["face_error"] = "detector_unavailable_original_used"
        return result

    try:
        image = cv2.imread(str(src))
        if image is None:
            result["face_error"] = "image_read_failed"
            return result
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            if require_face:
                result["face_error"] = "face_not_found"
                return result
            result["face_path"] = str(src)
            result["face_error"] = "face_not_found_original_used"
            return result

        x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
        height, width = image.shape[:2]
        mx = int(w * margin)
        my = int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(width, x + w + mx)
        y2 = min(height, y + h + my)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return result

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_face.jpg"
        cv2.imwrite(str(out_path), crop)
        result["face_path"] = str(out_path)
        result["face_bbox"] = f"{x1},{y1},{x2},{y2}"
        result["face_found"] = True
        return result
    except Exception as e:
        result["face_error"] = str(e)
        return result


def detect_and_crop_for_df(
    df: pd.DataFrame,
    image_col: str = "image_path",
    id_col: Optional[str] = None,
    out_dir: str = "data/cache/crops",
    margin: float = 0.2,
    require_face: bool = False,
    n_workers: int = 8,
) -> pd.DataFrame:
    df_out = df.copy().reset_index(drop=True)
    if image_col not in df_out.columns:
        df_out["face_path"] = None
        df_out["face_bbox"] = None
        df_out["face_found"] = False
        df_out["face_error"] = "missing_image_col"
        return df_out

    app_dir = Path(__file__).resolve().parents[1]
    crop_dir = Path(out_dir)
    if not crop_dir.is_absolute():
        crop_dir = app_dir / crop_dir

    tasks = []
    for idx, row in df_out.iterrows():
        image_path = row.get(image_col)
        if id_col and id_col in df_out.columns and pd.notna(row.get(id_col)):
            stem = str(row.get(id_col))
        else:
            stem = f"row_{idx}"
        tasks.append((idx, image_path, stem))

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as executor:
        future_to_idx = {
            executor.submit(_crop_single, image_path, crop_dir, stem, margin, require_face): idx
            for idx, image_path, stem in tasks
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = {"face_path": None, "face_bbox": None, "face_found": False, "face_error": "error"}

    df_out["face_path"] = [results.get(i, {}).get("face_path") for i in range(len(df_out))]
    df_out["face_bbox"] = [results.get(i, {}).get("face_bbox") for i in range(len(df_out))]
    df_out["face_found"] = [bool(results.get(i, {}).get("face_found")) for i in range(len(df_out))]
    df_out["face_error"] = [results.get(i, {}).get("face_error") for i in range(len(df_out))]
    return df_out