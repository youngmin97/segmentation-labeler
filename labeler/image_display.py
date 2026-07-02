"""numpy 배열 ↔ Qt QImage/QPixmap 변환 (stride/색공간 안전)."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

# 디바이스 포맷 변환 시 디더링 방지 (grayscale가 화면에서 컬러로 깨지는 것 방지)
_NO_DITHER = Qt.ImageConversionFlags(Qt.AvoidDither | Qt.ColorOnly)


def _to_pixmap(qimg: QImage) -> QPixmap:
    return QPixmap.fromImage(qimg.copy(), _NO_DITHER)


def _aligned_bpl(width: int, channels: int) -> int:
    """Qt QImage 행 바이트 정렬 (4바이트 배수)."""
    return ((width * channels + 3) // 4) * 4


def gray_to_qpixmap(gray: np.ndarray) -> QPixmap:
    """단일 채널 uint8 grayscale → QPixmap (RGB 변환 없음)."""
    arr = np.ascontiguousarray(gray, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale, got shape {arr.shape}")
    h, w = arr.shape
    bpl = _aligned_bpl(w, 1)
    if bpl == w:
        buf = arr
    else:
        buf = np.zeros((h, bpl), dtype=np.uint8)
        buf[:, :w] = arr
    qimg = QImage(buf.data, w, h, bpl, QImage.Format_Grayscale8)
    return _to_pixmap(qimg)


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """(H, W, 3) uint8 RGB → QPixmap."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, c = arr.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")
    bpl = _aligned_bpl(w, 3)
    if bpl == w * 3:
        buf = arr
    else:
        flat = np.zeros((h, bpl), dtype=np.uint8)
        flat[:, : w * 3] = arr.reshape(h, w * 3)
        buf = flat
    qimg = QImage(buf.data, w, h, bpl, QImage.Format_RGB888)
    return _to_pixmap(qimg)


def rgba_to_qpixmap(rgba: np.ndarray) -> QPixmap:
    """(H, W, 4) uint8 RGBA → QPixmap."""
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, c = arr.shape
    if c != 4:
        raise ValueError(f"Expected 4 channels, got {c}")
    bpl = _aligned_bpl(w, 4)
    if bpl == w * 4:
        buf = arr
    else:
        flat = np.zeros((h, bpl), dtype=np.uint8)
        flat[:, : w * 4] = arr.reshape(h, w * 4)
        buf = flat
    qimg = QImage(buf.data, w, h, bpl, QImage.Format_RGBA8888)
    return _to_pixmap(qimg)


def blend_labels_on_gray(
    gray: np.ndarray,
    label: np.ndarray,
    class_map: dict[str, int],
    colors: dict[str, tuple[int, int, int, int]],
    *,
    fill_alpha: float = 0.5,
    draw_outline: bool = True,
) -> np.ndarray:
    """Grayscale 위에 클래스 색상을 alpha blend → RGB uint8.

    밝은/어두운 영역 모두에서 라벨이 잘 보이도록 alpha를 균일하게 적용하고
    경계선(outline)을 그린다.
    """
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    for cls_name, cls_id in class_map.items():
        if cls_id == 0:
            continue
        mask = label == cls_id
        if not np.any(mask):
            continue
        r, g, b, _ = colors.get(cls_name, (128, 128, 128, 100))
        rgb[mask, 0] = rgb[mask, 0] * (1.0 - fill_alpha) + r * fill_alpha
        rgb[mask, 1] = rgb[mask, 1] * (1.0 - fill_alpha) + g * fill_alpha
        rgb[mask, 2] = rgb[mask, 2] * (1.0 - fill_alpha) + b * fill_alpha

        if draw_outline:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(rgb, contours, -1, (float(r), float(g), float(b)), 2)

    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
