"""폴더 감시 (Watcher) — ImageBuffer / CurrentPatients."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from labeler.bmode_extractor import find_matching_h5


@dataclass
class WatchItem:
    dcm_path: str
    h5_path: str | None


class FolderWatcher(QObject):
    """네트워크 폴더를 주기적으로 스캔하여 새 DCM 파일을 감지."""

    new_file = pyqtSignal(object)  # WatchItem
    log_message = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._image_buffer = ""
        self._iq_data = ""
        self._enabled = False
        self._seen: set[str] = set()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._poll_interval_ms = 2000
        self._lock = threading.Lock()

    def configure(self, image_buffer: str, iq_data: str) -> None:
        self._image_buffer = image_buffer
        self._iq_data = iq_data

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if enabled:
            self._bootstrap_seen()
            self._timer.start(self._poll_interval_ms)
            self.log_message.emit("Watcher 시작")
        else:
            self._timer.stop()
            self.log_message.emit("Watcher 중지")

    def is_enabled(self) -> bool:
        return self._enabled

    def _bootstrap_seen(self) -> None:
        """기존 파일은 무시하고 이후 생성되는 파일만 처리."""
        with self._lock:
            self._seen.clear()
            for path in self._iter_dcm_files(self._image_buffer):
                self._seen.add(os.path.normcase(path))

    def _iter_dcm_files(self, folder: str) -> list[str]:
        if not folder or not os.path.isdir(folder):
            return []
        result = []
        for root, _, files in os.walk(folder):
            for fname in files:
                low = fname.lower()
                if low.endswith(".dcm") or (not low.endswith((".png", ".jpg", ".json", ".txt", ".h5"))):
                    full = os.path.join(root, fname)
                    if os.path.isfile(full):
                        result.append(full)
        return result

    def _poll(self) -> None:
        if not self._enabled:
            return
        if not os.path.isdir(self._image_buffer):
            self.log_message.emit(f"ImageBuffer 경로 없음: {self._image_buffer}")
            return

        for dcm_path in self._iter_dcm_files(self._image_buffer):
            key = os.path.normcase(dcm_path)
            with self._lock:
                if key in self._seen:
                    continue
                self._seen.add(key)

            # 파일 쓰기 완료 대기
            if not self._wait_stable(dcm_path):
                continue

            h5_path = None
            if os.path.isdir(self._iq_data):
                h5_path = find_matching_h5(dcm_path, self._iq_data)

            item = WatchItem(dcm_path=dcm_path, h5_path=h5_path)
            self.log_message.emit(f"새 파일 감지: {Path(dcm_path).name}")
            self.new_file.emit(item)

    @staticmethod
    def _wait_stable(path: str, checks: int = 3, delay: float = 0.5) -> bool:
        try:
            prev = -1
            for _ in range(checks):
                size = os.path.getsize(path)
                if size == prev and size > 0:
                    return True
                prev = size
                time.sleep(delay)
        except OSError:
            return False
        return False
