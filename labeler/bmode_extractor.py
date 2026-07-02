"""DICOM + H5(IQ)에서 B-mode 이미지 추출."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import h5py
import numpy as np
import pydicom

from labeler.class_maps import Organ, organ_from_study_description

C0_SOUND_SPEED = 1540.0
FC_MHZ_MAP = {6.5: 6.149, 7: 6.7013, 8: 6.904, 8.5: 7.2357}
DEFAULT_PROBE_TYPE = "L3-12"

CROP_MAP_BARR_H820_W1092 = {
    40: (131, 722, 116, 872),
    45: (131, 722, 158, 830),
    50: (131, 722, 191, 797),
    55: (131, 722, 219, 769),
    60: (131, 722, 242, 746),
}

CROP_MAP_BARR_H970_W1292 = {
    40: (143, 853, 139, 1049),
    45: (143, 853, 190, 998),
    50: (143, 853, 231, 958),
    55: (143, 853, 263, 925),
    60: (143, 853, 291, 897),
}


@dataclass
class BModeResult:
    image: np.ndarray  # uint8 grayscale (H, W)
    patient_id: str
    organ: str
    organ_enum: Organ | None
    content_date: str
    content_time: str
    depth: int
    probe_type: str
    dcm_path: str
    h5_path: str | None

    @property
    def base_filename(self) -> str:
        time_tag = self.content_time.split(".")[0]
        return f"{self.patient_id}_{self.organ}_{self.content_date}_{time_tag}"


def get_crop_map_by_bmode_shape(h: int, w: int) -> dict[int, tuple[int, int, int, int]]:
    if (h, w) == (820, 1092):
        return CROP_MAP_BARR_H820_W1092
    if (h, w) == (970, 1292):
        return CROP_MAP_BARR_H970_W1292
    raise RuntimeError(f"Unsupported B-mode shape: {(h, w)}")


def get_depth_from_h5(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        f_samp = float(f["dump-1/streams/0/setups/BeamFormerParam/inputSampleRate"][()].item())
        n = int(f["dump-1/streams/0/setups/TissueProcessingParam/inputSamples"][()].item())
    depth_mm = (n * C0_SOUND_SPEED / (2.0 * f_samp)) * 1000.0 / 2.0
    return int(depth_mm)


def normalize_bmode(bmode: np.ndarray) -> np.ndarray:
    bm = bmode.astype(np.float32)
    bm = bm - bm.min()
    if bm.max() > 0:
        bm = bm / bm.max()
    return (bm * 255).astype(np.uint8)


def extract_bmode(
    dcm_path: str,
    h5_path: str | None = None,
    *,
    probe_type_filter: str = DEFAULT_PROBE_TYPE,
) -> BModeResult:
    ds = pydicom.dcmread(dcm_path, force=True)
    probe_type = str(ds["TransducerData"][0]) if "TransducerData" in ds else "unknown"
    if probe_type_filter and probe_type != probe_type_filter:
        raise ValueError(f"Probe type mismatch: expected {probe_type_filter}, got {probe_type}")

    bmode = ds.pixel_array
    try:
        organ = str(ds.StudyDescription)
    except Exception:
        organ = "unknown"

    patient_id = str(ds.PatientID)
    content_date = str(getattr(ds, "ContentDate", "") or getattr(ds, "StudyDate", ""))
    content_time = str(getattr(ds, "ContentTime", "000000")).split(".")[0]

    if h5_path:
        depth = get_depth_from_h5(h5_path)
    else:
        # H5 없을 때 기본 depth (50mm) — 수동 열기용 fallback
        depth = 50

    crop_map = get_crop_map_by_bmode_shape(*bmode.shape[:2])
    if depth not in crop_map:
        raise RuntimeError(
            f"Unsupported depth {depth} for shape {bmode.shape[:2]}. "
            f"Provide matching H5 file."
        )

    y0, y1, x0, x1 = crop_map[depth]
    bmode_crop = normalize_bmode(bmode[y0:y1, x0:x1])

    return BModeResult(
        image=bmode_crop,
        patient_id=patient_id,
        organ=organ,
        organ_enum=organ_from_study_description(organ),
        content_date=content_date,
        content_time=content_time,
        depth=depth,
        probe_type=probe_type,
        dcm_path=dcm_path,
        h5_path=h5_path,
    )


def save_bmode_png(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)


def get_best_datetime(ds) -> datetime | None:
    if "StudyDate" in ds and "ContentTime" in ds:
        try:
            t = ds.ContentTime.split(".")[0].ljust(6, "0")
            return datetime.strptime(ds.StudyDate + t, "%Y%m%d%H%M%S")
        except Exception:
            pass
    if "StudyDate" in ds and "StudyTime" in ds:
        try:
            t = ds.StudyTime.split(".")[0].ljust(6, "0")
            return datetime.strptime(ds.StudyDate + t, "%Y%m%d%H%M%S")
        except Exception:
            pass
    return None


def find_matching_h5(dcm_path: str, h5_folder: str) -> str | None:
    """CurrentPatients 폴더에서 PatientID가 일치하는 H5 파일 검색."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    patient_id = str(ds.PatientID)
    dcm_time = get_best_datetime(ds)

    candidates: list[tuple[str, datetime | None]] = []

    for root, _, files in os.walk(h5_folder):
        for fname in files:
            if not fname.lower().endswith((".h5", ".hdf5")):
                continue
            if patient_id not in fname:
                continue
            full = os.path.join(root, fname)
            candidates.append((full, None))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    # 여러 후보: 파일명에 patient_id 포함, 가장 최근 수정 파일 우선
    candidates.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    return candidates[0][0]
