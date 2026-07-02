"""앱 설정 저장/로드."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class AppSettings:
    image_buffer_path: str = r"E:\ImageBuffer"
    iq_data_path: str = r"E:\IQ_Data\CurrentPatients"
    output_path: str = r"D:\Barreleye_Labeled"
    watcher_enabled: bool = False
    probe_type: str = "L3-12"
    window_geometry: str = ""
    last_organ: str = "Thyroid"

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> AppSettings:
        if not CONFIG_PATH.exists():
            s = cls()
            s.save()
            return s
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
