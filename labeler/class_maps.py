"""Organ별 segmentation 클래스 맵 및 시각화 색상."""

from __future__ import annotations

from enum import Enum


class Organ(str, Enum):
    BREAST = "Breast"
    THYROID = "Thyroid"
    MSK = "Musculoskeletal"


CLASS_MAP_BREAST: dict[str, int] = {
    "background": 0,
    "skin": 1,
    "premammary_fat": 2,
    "intramammary_fat": 3,
    "retromammary_fat": 4,
    "mammary": 5,
    "lesion": 6,
    "muscle": 7,
    "unknown": 8,
    "vessel": 9,
    "bone": 10,
    "fat": 11,
}

CLASS_MAP_THYROID: dict[str, int] = {
    "background": 0,
    "skin": 1,
    "thyroid": 2,
    "fat": 3,
    "muscle": 4,
    "vessel": 5,
    "trachea_including_shadow": 6,
    "esophagus": 7,
    "lesion": 8,
    "unknown": 9,
}

CLASS_MAP_MSK: dict[str, int] = {
    "background": 0,
    "skin": 1,
    "fat": 2,
    "muscle": 3,
    "tendon-ligament-complex": 4,
    "nerve": 5,
    "vessel": 6,
    "bone": 7,
    "joint-fluid-or-bursa": 8,
    "lesion": 9,
    "unknown": 10,
}

ORGAN_CLASS_MAPS: dict[Organ, dict[str, int]] = {
    Organ.BREAST: CLASS_MAP_BREAST,
    Organ.THYROID: CLASS_MAP_THYROID,
    Organ.MSK: CLASS_MAP_MSK,
}

# StudyDescription 약어 -> Organ
STUDY_DESC_TO_ORGAN: dict[str, Organ] = {
    "BUS": Organ.BREAST,
    "TUS": Organ.THYROID,
    "MSK": Organ.MSK,
    "MSK-ARM": Organ.MSK,
    "MSK-LEG": Organ.MSK,
}


def organ_from_study_description(study_desc: str) -> Organ | None:
    key = str(study_desc).strip().upper()
    if key in STUDY_DESC_TO_ORGAN:
        return STUDY_DESC_TO_ORGAN[key]
    if key.startswith("MSK"):
        return Organ.MSK
    if key.startswith("BUS"):
        return Organ.BREAST
    if key.startswith("TUS"):
        return Organ.THYROID
    return None


def class_map_for_organ(organ: Organ) -> dict[str, int]:
    return ORGAN_CLASS_MAPS[organ]


def class_colors(class_map: dict[str, int]) -> dict[str, tuple[int, int, int, int]]:
    """클래스별 RGBA 오버레이 색상 (background는 투명)."""
    palette = [
        (0, 0, 0, 0),
        (255, 80, 80, 100),
        (80, 200, 120, 100),
        (80, 160, 255, 100),
        (255, 200, 60, 100),
        (200, 80, 255, 100),
        (255, 120, 200, 100),
        (120, 255, 255, 100),
        (255, 160, 80, 100),
        (160, 80, 255, 100),
        (80, 255, 160, 100),
        (255, 255, 80, 100),
        (180, 180, 180, 100),
    ]
    colors: dict[str, tuple[int, int, int, int]] = {}
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        colors[name] = palette[idx % len(palette)]
    colors["background"] = (0, 0, 0, 0)
    return colors


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ").title()
