"""자유곡선(스타일러스/터치) → 폐곡선 변환 유틸.

OpenCV approxPolyDP (Douglas-Peucker)로 스트로크를 단순화합니다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def subsample_by_distance(
    points: list[tuple[float, float]],
    min_dist: float,
) -> list[tuple[float, float]]:
    """연속 입력 중 최소 거리 이상일 때만 점을 샘플링."""
    if not points:
        return []
    out = [points[0]]
    for x, y in points[1:]:
        lx, ly = out[-1]
        if math.hypot(x - lx, y - ly) >= min_dist:
            out.append((x, y))
    return out


def stroke_to_closed_polygon(
    points: list[tuple[float, float]],
    *,
    min_dist: float = 2.0,
    simplify_epsilon: float = 2.0,
    min_points: int = 3,
) -> np.ndarray | None:
    """드래그 스트로크를 폐곡선 polygon (N, 2) int32로 변환.

    마지막 점과 첫 점은 fillPoly 시 자동 연결되므로 별도 중복 추가 불필요.
    """
    sampled = subsample_by_distance(points, min_dist)
    if len(sampled) < min_points:
        return None

    contour = np.array(sampled, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, simplify_epsilon, closed=False)

    if len(simplified) < min_points:
        simplified = contour

    return simplified.reshape(-1, 2).astype(np.int32)
