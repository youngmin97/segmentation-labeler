"""Segmentation Labeler 메인 윈도우."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from labeler.bmode_extractor import BModeResult, extract_bmode
from labeler.class_maps import Organ, ORGAN_CLASS_MAPS, class_map_for_organ
from labeler.file_utils import build_output_paths, copy_file, load_bmode_png, load_label_png, save_bmode_png, save_label_png
from labeler.settings import AppSettings
from labeler.ui.canvas import EditMode, SegmentationCanvas
from labeler.ui.class_panel import ClassPanel
from labeler.watcher import FolderWatcher, WatchItem


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Segmentation Labeler")
        self.resize(1400, 900)

        self._settings = AppSettings.load()
        self._current: BModeResult | None = None
        self._queue: list[WatchItem] = []

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._apply_organ(self._organ_from_name(self._settings.last_organ))
        self._restore_watcher_state()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        # Left: canvas
        canvas_wrap = QWidget()
        canvas_layout = QVBoxLayout(canvas_wrap)
        self.canvas = SegmentationCanvas()
        canvas_layout.addWidget(self.canvas)
        splitter.addWidget(canvas_wrap)

        # Right: controls
        right = QWidget()
        right.setMaximumWidth(340)
        right_layout = QVBoxLayout(right)

        # Organ
        organ_box = QGroupBox("Organ")
        organ_layout = QVBoxLayout(organ_box)
        self.organ_combo = QComboBox()
        for o in Organ:
            self.organ_combo.addItem(o.value, o)
        organ_layout.addWidget(self.organ_combo)
        right_layout.addWidget(organ_box)

        # Class panel
        self.class_panel = ClassPanel()
        right_layout.addWidget(self.class_panel)

        # Edit mode
        mode_box = QGroupBox("편집 모드")
        mode_layout = QVBoxLayout(mode_box)
        self.btn_add = QPushButton("영역 추가 (Add)")
        self.btn_sub = QPushButton("영역 제외 (Subtract)")
        self.btn_remove = QPushButton("클래스 전체 제거 (Remove)")
        mode_layout.addWidget(self.btn_add)
        mode_layout.addWidget(self.btn_sub)
        mode_layout.addWidget(self.btn_remove)
        right_layout.addWidget(mode_box)

        # File info
        info_box = QGroupBox("현재 파일")
        info_layout = QVBoxLayout(info_box)
        self.lbl_info = QLabel("파일 없음")
        self.lbl_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_info)
        right_layout.addWidget(info_box)

        # Queue
        queue_box = QGroupBox("대기열 (Watcher)")
        queue_layout = QVBoxLayout(queue_box)
        self.queue_list = QListWidget()
        queue_layout.addWidget(self.queue_list)
        self.btn_next = QPushButton("다음 항목 불러오기")
        queue_layout.addWidget(self.btn_next)
        right_layout.addWidget(queue_box)

        # Watcher settings
        watch_box = QGroupBox("Watcher 설정")
        watch_layout = QVBoxLayout(watch_box)

        self.edit_image_buffer = QLineEdit(self._settings.image_buffer_path)
        self.edit_iq_data = QLineEdit(self._settings.iq_data_path)
        self.edit_output = QLineEdit(self._settings.output_path)

        watch_layout.addWidget(QLabel("ImageBuffer:"))
        watch_layout.addWidget(self.edit_image_buffer)
        watch_layout.addWidget(QLabel("IQ Data (CurrentPatients):"))
        watch_layout.addWidget(self.edit_iq_data)
        watch_layout.addWidget(QLabel("저장 경로:"))
        watch_layout.addWidget(self.edit_output)

        btn_row = QHBoxLayout()
        self.btn_watcher = QPushButton("Watcher OFF")
        self.btn_watcher.setCheckable(True)
        self.btn_save_settings = QPushButton("설정 저장")
        btn_row.addWidget(self.btn_watcher)
        btn_row.addWidget(self.btn_save_settings)
        watch_layout.addLayout(btn_row)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        watch_layout.addWidget(self.log_view)
        right_layout.addWidget(watch_box)

        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _build_menu(self) -> None:
        tb = QToolBar("Main")
        self.addToolBar(tb)

        act_open_dcm = QAction("DCM 열기", self)
        act_open_dcm.setShortcut(QKeySequence("Ctrl+O"))
        act_open_dcm.triggered.connect(self._open_dcm)
        tb.addAction(act_open_dcm)

        act_open_png = QAction("PNG 열기", self)
        act_open_png.triggered.connect(self._open_png_pair)
        tb.addAction(act_open_png)

        act_save = QAction("저장", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save)
        tb.addAction(act_save)

        act_undo = QAction("실행 취소", self)
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.canvas.undo)
        tb.addAction(act_undo)

        act_fit = QAction("화면 맞춤", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(self.canvas.fit_in_view)
        tb.addAction(act_fit)

    def _connect_signals(self) -> None:
        self.organ_combo.currentIndexChanged.connect(self._on_organ_changed)
        self.class_panel.class_selected.connect(self._on_class_selected)
        self.canvas.status_message.connect(self.status.showMessage)

        self.btn_add.clicked.connect(lambda: self._set_mode(EditMode.ADD))
        self.btn_sub.clicked.connect(lambda: self._set_mode(EditMode.SUBTRACT))
        self.btn_remove.clicked.connect(self._on_remove_class)

        self.btn_next.clicked.connect(self._load_next_queue)
        self.btn_save_settings.clicked.connect(self._save_settings)
        self.btn_watcher.toggled.connect(self._toggle_watcher)

        self.watcher = FolderWatcher(self)
        self.watcher.new_file.connect(self._on_watch_item)
        self.watcher.log_message.connect(self._log)

        self._set_mode(EditMode.ADD)

    def _organ_from_name(self, name: str) -> Organ:
        for o in Organ:
            if o.value == name:
                return o
        return Organ.THYROID

    def _apply_organ(self, organ: Organ) -> None:
        idx = self.organ_combo.findData(organ)
        if idx >= 0:
            self.organ_combo.blockSignals(True)
            self.organ_combo.setCurrentIndex(idx)
            self.organ_combo.blockSignals(False)
        cmap = class_map_for_organ(organ)
        self.class_panel.set_class_map(cmap)
        self.canvas.set_class_map(cmap)
        self.canvas.set_current_class(self.class_panel.current_class())

    def _on_organ_changed(self) -> None:
        organ = self.organ_combo.currentData()
        if organ:
            self._apply_organ(organ)
            self._settings.last_organ = organ.value

    def _set_mode(self, mode: EditMode) -> None:
        self.canvas.set_edit_mode(mode)
        self.btn_add.setStyleSheet("" if mode != EditMode.ADD else "font-weight: bold; background: #2d5a2d;")
        self.btn_sub.setStyleSheet("" if mode != EditMode.SUBTRACT else "font-weight: bold; background: #5a4a2d;")
        self.btn_remove.setStyleSheet("")

    def _on_class_selected(self, class_name: str) -> None:
        self.canvas.set_current_class(class_name)
        # 클래스 변경 시 항상 '영역 추가(Add)' 모드로 전환
        self._set_mode(EditMode.ADD)

    def _on_remove_class(self) -> None:
        self.canvas.remove_current_class()

    def _log(self, msg: str) -> None:
        self.log_view.append(msg)
        self.status.showMessage(msg, 5000)

    def _save_settings(self) -> None:
        self._settings.image_buffer_path = self.edit_image_buffer.text().strip()
        self._settings.iq_data_path = self.edit_iq_data.text().strip()
        self._settings.output_path = self.edit_output.text().strip()
        self._settings.save()
        self.watcher.configure(self._settings.image_buffer_path, self._settings.iq_data_path)
        self._log("설정 저장됨")

    def _restore_watcher_state(self) -> None:
        self.watcher.configure(self._settings.image_buffer_path, self._settings.iq_data_path)
        if self._settings.watcher_enabled:
            self.btn_watcher.setChecked(True)

    def _toggle_watcher(self, on: bool) -> None:
        self._save_settings()
        self.watcher.set_enabled(on)
        self._settings.watcher_enabled = on
        self._settings.save()
        self.btn_watcher.setText("Watcher ON" if on else "Watcher OFF")
        self.btn_watcher.setStyleSheet("background: #2d5a2d;" if on else "")

    def _on_watch_item(self, item: WatchItem) -> None:
        self._queue.append(item)
        name = Path(item.dcm_path).name
        self.queue_list.addItem(name)

    def _load_next_queue(self) -> None:
        if not self._queue:
            QMessageBox.information(self, "대기열", "대기 중인 항목이 없습니다.")
            return
        item = self._queue.pop(0)
        self.queue_list.takeItem(0)
        self._load_dcm(item.dcm_path, item.h5_path)

    def _open_dcm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "DCM 열기", "", "DICOM (*.dcm);;All (*)")
        if not path:
            return
        h5_path, _ = QFileDialog.getOpenFileName(
            self, "IQ H5 파일 (선택)", "", "HDF5 (*.h5 *.hdf5);;All (*)"
        )
        self._load_dcm(path, h5_path or None)

    def _load_dcm(self, dcm_path: str, h5_path: str | None) -> None:
        try:
            result = extract_bmode(
                dcm_path,
                h5_path,
                probe_type_filter=self._settings.probe_type,
            )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"B-mode 추출 실패:\n{e}")
            return

        if result.organ_enum:
            self._apply_organ(result.organ_enum)

        self._current = result
        self.canvas.load_bmode(result.image)
        self._update_info()
        self._log(f"로드: {result.base_filename}")

    def _open_png_pair(self) -> None:
        img_path, _ = QFileDialog.getOpenFileName(self, "B-mode PNG", "", "PNG (*.png)")
        if not img_path:
            return
        lbl_path, _ = QFileDialog.getOpenFileName(self, "Label PNG (선택)", "", "PNG (*.png)")
        try:
            bmode = load_bmode_png(img_path)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지를 읽을 수 없습니다:\n{e}")
            return
        label = load_label_png(lbl_path) if lbl_path else None
        if label is not None and label.shape != bmode.shape:
            QMessageBox.warning(
                self, "경고",
                f"Label 크기({label.shape})와 B-mode({bmode.shape})가 다릅니다. Label은 무시합니다.",
            )
            label = None
        stem = Path(img_path).stem
        parts = stem.split("_")
        self._current = BModeResult(
            image=bmode,
            patient_id=parts[0] if parts else "unknown",
            organ=parts[1] if len(parts) > 1 else "unknown",
            organ_enum=None,
            content_date=parts[2] if len(parts) > 2 else "",
            content_time=parts[3] if len(parts) > 3 else "",
            depth=0,
            probe_type="",
            dcm_path="",
            h5_path=None,
        )
        self.canvas.load_bmode(bmode, label)
        self._update_info()

    def _update_info(self) -> None:
        if not self._current:
            self.lbl_info.setText("파일 없음")
            return
        c = self._current
        h5 = Path(c.h5_path).name if c.h5_path else "(없음)"
        self.lbl_info.setText(
            f"Patient: {c.patient_id}\n"
            f"Organ: {c.organ}\n"
            f"Date/Time: {c.content_date}_{c.content_time}\n"
            f"Depth: {c.depth}mm\n"
            f"H5: {h5}\n"
            f"파일명: {c.base_filename}.png"
        )

    def _save(self) -> None:
        if not self._current or not self.canvas.has_image():
            QMessageBox.warning(self, "저장", "저장할 이미지가 없습니다.")
            return

        label = self.canvas.get_label()
        bmode = self.canvas.get_bmode()
        if label is None or bmode is None:
            return

        out_root = self.edit_output.text().strip() or self._settings.output_path
        paths = build_output_paths(out_root, self._current.base_filename)

        try:
            save_bmode_png(bmode, paths["bmode"])
            save_label_png(label, paths["label"])

            if self._current.dcm_path and os.path.isfile(self._current.dcm_path):
                copy_file(self._current.dcm_path, paths["dcm"])
            if self._current.h5_path and os.path.isfile(self._current.h5_path):
                copy_file(self._current.h5_path, paths["h5"])

            self._log(f"저장 완료: {paths['bmode']}")
            QMessageBox.information(self, "저장", f"저장 완료:\n{paths['bmode']}\n{paths['label']}")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", str(e))

    def closeEvent(self, event) -> None:
        self._settings.window_geometry = self.saveGeometry().toHex().data().decode()
        self._settings.save()
        self.watcher.set_enabled(False)
        super().closeEvent(event)


def run_app() -> None:
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
