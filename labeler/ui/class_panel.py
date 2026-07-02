"""클래스 선택 패널."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from labeler.class_maps import class_colors, display_name


class ClassPanel(QGroupBox):
    class_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("클래스", parent)
        self._layout = QVBoxLayout(self)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, QRadioButton] = {}
        self._class_map: dict[str, int] = {}

    def set_class_map(self, class_map: dict[str, int]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._buttons.clear()
        self._class_map = dict(class_map)
        colors = class_colors(class_map)

        for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
            if name == "background":
                continue
            btn = QRadioButton(f"{idx}: {display_name(name)}")
            r, g, b, _ = colors[name]
            btn.setStyleSheet(f"QRadioButton {{ color: rgb({r},{g},{b}); }}")
            self._group.addButton(btn)
            self._layout.addWidget(btn)
            self._buttons[name] = btn
            btn.toggled.connect(lambda checked, n=name: self._on_toggle(checked, n))

        # 첫 non-background 선택
        for name in self._buttons:
            self._buttons[name].setChecked(True)
            break

    def set_current_class(self, class_name: str) -> None:
        btn = self._buttons.get(class_name)
        if btn:
            btn.setChecked(True)

    def current_class(self) -> str:
        for name, btn in self._buttons.items():
            if btn.isChecked():
                return name
        return "skin"

    def _on_toggle(self, checked: bool, name: str) -> None:
        if checked:
            self.class_selected.emit(name)
