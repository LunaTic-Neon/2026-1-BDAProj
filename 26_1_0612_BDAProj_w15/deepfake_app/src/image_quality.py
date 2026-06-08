from PIL import Image, ImageStat
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def _open_image_safe(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def sharpness_score(pil_img: Image.Image) -> float:
    # variance of Laplacian을 이용하려면 OpenCV가 필요하지만, Pillow로 간단히 대체
    try:
        import cv2
        arr = np.array(pil_img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception:
        # 대체: 이미지의 대비(standard deviation of luminance)
        stat = ImageStat.Stat(pil_img.convert("L"))
        return float(stat.stddev[0])


def avg_brightness(pil_img: Image.Image) -> float:
    stat = ImageStat.Stat(pil_img.convert("L"))
    return float(stat.mean[0])


def is_valid_size(pil_img: Image.Image, min_width: int, min_height: int) -> bool:
    w, h = pil_img.size
    return (w >= min_width and h >= min_height)


def filter_single(path: str, min_width: int, min_height: int, min_sharpness: float,
                  min_brightness: float, max_brightness: float, require_face: bool = False):
    res = {"path": path, "ok": False, "reason": None, "width": None, "height": None,
           "sharpness": None, "brightness": None}
    if not path:
        res["reason"] = "no_path"
        return res
    img = _open_image_safe(path)
    if img is None:
        res["reason"] = "open_failed"
        return res
    w, h = img.size
    res["width"] = w
    res["height"] = h
    if not is_valid_size(img, min_width, min_height):
        res["reason"] = "small_size"
        return res
    sharp = sharpness_score(img)
    res["sharpness"] = sharp
    if sharp < min_sharpness:
        res["reason"] = "low_sharpness"
        return res
    bright = avg_brightness(img)
    res["brightness"] = bright
    if bright < min_brightness or bright > max_brightness:
        res["reason"] = "bad_brightness"
        return res
    # face requirement는 추후 face_preprocess 에서 처리 가능
    res["ok"] = True
    return res


def filter_valid_images(df, image_col: str = "image_path", min_width: int = 64, min_height: int = 64,
                        min_sharpness: float = 50.0, min_brightness: float = 10.0, max_brightness: float = 245.0,
                        require_face: bool = False, max_workers: int = 8):
    """DataFrame에서 image_col 경로를 검사하여 품질 리포트 열을 붙여 반환합니다.

    반환: df_report (copy) — 각 행에 iq_pass(bool), iq_reason(str), iq_sharpness, iq_brightness, width, height
    """
    paths = df[image_col].fillna("").tolist()
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(filter_single, paths[i], min_width, min_height, min_sharpness, min_brightness, max_brightness, require_face): i for i in range(len(paths))}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = {"path": paths[idx], "ok": False, "reason": "error"}

    # build report
    df_out = df.copy()
    df_out["iq_pass"] = [r.get("ok") if r else None for r in results]
    df_out["iq_reason"] = [r.get("reason") if r else None for r in results]
    df_out["iq_sharpness"] = [r.get("sharpness") if r else None for r in results]
    df_out["iq_brightness"] = [r.get("brightness") if r else None for r in results]
    df_out["iq_width"] = [r.get("width") if r else None for r in results]
    df_out["iq_height"] = [r.get("height") if r else None for r in results]
    return df_out
