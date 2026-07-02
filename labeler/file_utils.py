"""파일 저장/복사 유틸."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_bmode_png(path: str | Path) -> np.ndarray:
    """B-mode PNG를 단일 채널 grayscale uint8로 로드.

    RGB(A) 이미지는 luma 가중치 대신 채널 단순 평균으로 grayscale화한다.
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        gray = arr
    else:
        # RGBA면 알파 제외, RGB 채널 단순 평균
        rgb = arr[:, :, :3].astype(np.float32)
        gray = np.round(rgb.mean(axis=2))

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def save_label_png(label: np.ndarray, path: str | Path) -> None:
    """클래스 인덱스 맵을 uint8 grayscale PNG로 저장.

    픽셀 값 = 클래스 index (0, 1, 2, ...). UI 오버레이 색상이 아님.
    LabelBox_TUS.py / LabelBox_BUS.py export 형식과 동일.
    """
    arr = np.asarray(label)
    if arr.ndim != 2:
        raise ValueError(
            f"Label must be 2D index map (H, W), got shape {arr.shape}. "
            "RGB/RGBA 오버레이 이미지는 저장할 수 없습니다."
        )
    arr = arr.astype(np.uint8)
    ensure_dir(Path(path).parent)
    Image.fromarray(arr, mode="L").save(str(path))


def load_label_png(path: str | Path) -> np.ndarray:
    """인덱스 맵 PNG 로드 — 픽셀 값을 그대로 uint8 배열로 반환."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def save_bmode_png(image: np.ndarray, path: str | Path) -> None:
    cv2.imwrite(str(path), image)


def save_display_png(arr: np.ndarray, path: str | Path) -> None:
    """화면 합성본을 numpy 배열에서 직접 PNG로 저장 (디바이스 독립).

    2D면 grayscale로, (H,W,3) RGB면 BGR로 변환해 저장한다.
    """
    a = np.ascontiguousarray(np.asarray(arr))
    ensure_dir(Path(path).parent)
    if a.ndim == 2:
        cv2.imwrite(str(path), a.astype(np.uint8))
    elif a.ndim == 3 and a.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        raise ValueError(f"Unsupported display array shape: {a.shape}")


def copy_file(src: str | Path, dst: str | Path) -> None:
    ensure_dir(Path(dst).parent)
    shutil.copy2(str(src), str(dst))


def build_output_paths(
    output_root: str | Path,
    base_filename: str,
) -> dict[str, Path]:
    root = ensure_dir(output_root)
    stem = Path(base_filename).stem
    return {
        "bmode": root / f"{stem}.png",
        "label": root / "labels" / f"{stem}.png",
        "dcm": root / "dcm" / f"{stem}.dcm",
        "h5": root / "h5" / f"{stem}.h5",
    }
