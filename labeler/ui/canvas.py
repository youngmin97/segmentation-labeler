"""자유곡선(터치/스타일러스) 기반 segmentation 캔버스."""

from __future__ import annotations

from enum import Enum, auto

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTabletEvent,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from pathlib import Path

from labeler.class_maps import class_colors
from labeler.file_utils import save_bmode_png, save_display_png
from labeler.freehand import stroke_to_closed_polygon
from labeler.image_display import blend_labels_on_gray, gray_to_qpixmap, rgba_to_qpixmap, rgb_to_qpixmap

# 로드 시 저장되는 고정 grayscale B-mode 경로 (프로젝트 루트/image.png)
LOADED_BMODE_PATH = Path(__file__).resolve().parent.parent.parent / "image.png"
# 화면에 실제 표시되는 최종 합성본(grayscale + 라벨 오버레이) 경로
LOADED_DISPLAY_PATH = Path(__file__).resolve().parent.parent.parent / "display.png"


class EditMode(Enum):
    ADD = auto()
    SUBTRACT = auto()


class SegmentationCanvas(QGraphicsView):
    """B-mode 위에 스타일러스/터치 드래그로 segmentation을 그리는 캔버스."""

    label_changed = pyqtSignal()
    status_message = pyqtSignal(str)

    MIN_SAMPLE_DIST = 2.0
    SIMPLIFY_EPSILON = 2.0
    PREVIEW_FILL_ALPHA = 140

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing)
        # 의료 영상: 픽셀 보간 끔 (스케일 시 RGB 채널 어긋남/색 얼룩 방지)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))

        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.viewport().setMouseTracking(True)

        self._composite_item: QGraphicsPixmapItem | None = None
        self._preview_fill_item: QGraphicsPixmapItem | None = None
        self._preview_path_item: QGraphicsPathItem | None = None

        self._bmode: np.ndarray | None = None
        self._label: np.ndarray | None = None
        self._class_map: dict[str, int] = {}
        self._colors: dict[str, tuple[int, int, int, int]] = {}

        self._current_class = "skin"
        self._current_class_id = 1
        self._edit_mode = EditMode.ADD
        self._drawing = False
        self._stroke_raw: list[tuple[float, float]] = []
        self._using_tablet = False

        self._panning = False
        self._pan_start = QPointF()

        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_class_map(self, class_map: dict[str, int]) -> None:
        self._class_map = dict(class_map)
        self._colors = class_colors(class_map)

    def set_current_class(self, class_name: str) -> None:
        if class_name not in self._class_map:
            return
        self._current_class = class_name
        self._current_class_id = self._class_map[class_name]

    def set_edit_mode(self, mode: EditMode) -> None:
        self._cancel_drawing()
        self._edit_mode = mode
        if mode == EditMode.ADD:
            self.status_message.emit("편집 모드: 추가 — 드래그 후 떼면 자동 폐곡선")
        else:
            self.status_message.emit("편집 모드: 제외 — 드래그 영역을 현재 클래스에서 차집합")

    def remove_current_class(self) -> None:
        """선택 클래스 전체 마스크를 즉시 제거 (그리기 불필요)."""
        if self._label is None:
            return
        if not np.any(self._label == self._current_class_id):
            self.status_message.emit(f"'{self._current_class}' 마스크 없음")
            return
        self._push_undo()
        self._label[self._label == self._current_class_id] = 0
        self._refresh_composite()
        self.label_changed.emit()
        self.status_message.emit(f"'{self._current_class}' 클래스 전체 제거됨")

    def load_bmode(self, image: np.ndarray, label: np.ndarray | None = None) -> None:
        gray = np.asarray(image)
        if gray.ndim == 3:
            # RGB → 채널 단순 평균으로 grayscale화 (luma 가중치 아님)
            gray = np.round(gray[:, :, :3].astype(np.float32).mean(axis=2))
        gray = np.clip(gray, 0, 255)
        self._bmode = np.ascontiguousarray(gray.astype(np.uint8))

        # 로드된 grayscale B-mode를 고정 파일명으로 즉시 저장
        try:
            save_bmode_png(self._bmode, LOADED_BMODE_PATH)
        except Exception:
            pass

        h, w = self._bmode.shape
        if label is None:
            self._label = np.zeros((h, w), dtype=np.uint8)
        else:
            self._label = np.ascontiguousarray(label.astype(np.uint8))
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._rebuild_scene()
        self.fit_in_view()

    def get_label(self) -> np.ndarray | None:
        return None if self._label is None else self._label.copy()

    def get_bmode(self) -> np.ndarray | None:
        return None if self._bmode is None else self._bmode.copy()

    def has_image(self) -> bool:
        return self._bmode is not None

    def undo(self) -> None:
        if not self._undo_stack or self._label is None:
            return
        self._redo_stack.append(self._label.copy())
        self._label = self._undo_stack.pop()
        self._refresh_composite()
        self.label_changed.emit()

    def redo(self) -> None:
        if not self._redo_stack or self._label is None:
            return
        self._undo_stack.append(self._label.copy())
        self._label = self._redo_stack.pop()
        self._refresh_composite()
        self.label_changed.emit()

    def fit_in_view(self) -> None:
        if self._composite_item is None:
            return
        self.resetTransform()
        self.fitInView(self._composite_item, Qt.KeepAspectRatio)

    def clear_label(self) -> None:
        if self._label is None:
            return
        self._push_undo()
        self._label[:] = 0
        self._refresh_composite()
        self.label_changed.emit()

    # ------------------------------------------------------------------
    # Rendering — grayscale B-mode + alpha-blended labels (single composite)
    # ------------------------------------------------------------------
    def _rebuild_scene(self) -> None:
        self._clear_preview()
        self._scene.clear()
        self._composite_item = None

        if self._bmode is None:
            return

        pixmap = self._render_composite()
        self._composite_item = self._scene.addPixmap(pixmap)
        self._composite_item.setTransformationMode(Qt.FastTransformation)
        self._composite_item.setZValue(0)
        self._scene.setSceneRect(self._composite_item.boundingRect())

    def _render_composite(self) -> QPixmap:
        gray = self._bmode
        assert gray is not None

        has_labels = self._label is not None and np.any(self._label > 0)
        if not has_labels:
            # 라벨 없음 → Grayscale8 직접 표시 (RGB 변환 없음)
            pixmap = gray_to_qpixmap(gray)
            disp_arr = gray  # 2D grayscale
        else:
            rgb = blend_labels_on_gray(gray, self._label, self._class_map, self._colors)
            pixmap = rgb_to_qpixmap(rgb)
            disp_arr = rgb  # RGB uint8

        # 화면에 뜨는 최종 합성본을 numpy 배열에서 직접 저장.
        # QPixmap은 디바이스(화면) 종속 포맷이라 디더링으로 컬러가 낄 수 있어
        # 저장은 반드시 배열에서 수행한다.
        try:
            save_display_png(disp_arr, LOADED_DISPLAY_PATH)
        except Exception:
            pass
        return pixmap

    def _refresh_composite(self) -> None:
        if self._composite_item is None:
            self._rebuild_scene()
            return
        self._composite_item.setPixmap(self._render_composite())

    def _push_undo(self) -> None:
        if self._label is not None:
            self._undo_stack.append(self._label.copy())
            self._redo_stack.clear()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------
    def _scene_pos(self, event) -> QPointF:
        """뷰 이벤트 좌표 → 이미지(scene) 좌표."""
        return self.mapToScene(event.pos())

    def _clamp_scene(self, pt: QPointF) -> QPointF:
        if self._label is None:
            return pt
        h, w = self._label.shape
        x = max(0.0, min(float(w - 1), pt.x()))
        y = max(0.0, min(float(h - 1), pt.y()))
        return QPointF(x, y)

    # ------------------------------------------------------------------
    # Freehand stroke + live preview
    # ------------------------------------------------------------------
    def _clear_preview(self) -> None:
        if self._preview_fill_item is not None:
            self._scene.removeItem(self._preview_fill_item)
            self._preview_fill_item = None
        if self._preview_path_item is not None:
            self._scene.removeItem(self._preview_path_item)
            self._preview_path_item = None

    def _cancel_drawing(self) -> None:
        self._drawing = False
        self._stroke_raw.clear()
        self._clear_preview()

    def _can_draw(self) -> bool:
        return self._edit_mode in (EditMode.ADD, EditMode.SUBTRACT)

    def _start_stroke(self, pt: QPointF) -> None:
        if not self._can_draw():
            return
        pt = self._clamp_scene(pt)
        self._drawing = True
        self._stroke_raw = [(pt.x(), pt.y())]
        self._update_preview(pt)

    def _extend_stroke(self, pt: QPointF) -> None:
        if not self._drawing:
            return
        pt = self._clamp_scene(pt)
        lx, ly = self._stroke_raw[-1]
        if ((pt.x() - lx) ** 2 + (pt.y() - ly) ** 2) ** 0.5 < self.MIN_SAMPLE_DIST:
            self._update_preview(pt)
            return
        self._stroke_raw.append((pt.x(), pt.y()))
        self._update_preview(pt)

    def _preview_polygon(self, cursor: QPointF) -> np.ndarray | None:
        if len(self._stroke_raw) < 1:
            return None
        pts = list(self._stroke_raw)
        pts.append((cursor.x(), cursor.y()))
        if len(pts) < 3:
            return None
        return np.array(pts, dtype=np.int32)

    def _update_preview(self, cursor: QPointF) -> None:
        """드래그 중: 경로 + 끝점↔시작점 직선으로 이루는 폐곡선 미리보기."""
        cursor = self._clamp_scene(cursor)
        polygon = self._preview_polygon(cursor)
        if polygon is None or self._label is None:
            self._clear_preview()
            return

        h, w = self._label.shape

        # --- fill preview ---
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)

        r, g, b, _ = self._colors.get(self._current_class, (0, 255, 128, 100))
        if self._edit_mode == EditMode.SUBTRACT:
            r, g, b = 255, 80, 80

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        alpha = self.PREVIEW_FILL_ALPHA
        rgba[mask > 0] = (r, g, b, alpha)
        fill_pix = rgba_to_qpixmap(rgba)

        if self._preview_fill_item is None:
            self._preview_fill_item = self._scene.addPixmap(fill_pix)
            self._preview_fill_item.setZValue(2)
        else:
            self._preview_fill_item.setPixmap(fill_pix)

        # --- outline preview: stroke + closing line ---
        path = QPainterPath(QPointF(self._stroke_raw[0][0], self._stroke_raw[0][1]))
        for x, y in self._stroke_raw[1:]:
            path.lineTo(x, y)
        path.lineTo(cursor)
        path.lineTo(self._stroke_raw[0][0], self._stroke_raw[0][1])

        pen = QPen(QColor(0, 255, 128), 2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        if self._preview_path_item is None:
            self._preview_path_item = self._scene.addPath(path, pen)
            self._preview_path_item.setZValue(3)
        else:
            self._preview_path_item.setPath(path)
            self._preview_path_item.setPen(pen)

    def _finish_stroke(self) -> None:
        if not self._drawing or self._label is None:
            self._cancel_drawing()
            return

        polygon = stroke_to_closed_polygon(
            self._stroke_raw,
            min_dist=self.MIN_SAMPLE_DIST,
            simplify_epsilon=self.SIMPLIFY_EPSILON,
        )
        self._cancel_drawing()

        if polygon is None or len(polygon) < 3:
            return

        mask = np.zeros(self._label.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        stroke_bool = mask.astype(bool)

        self._push_undo()

        if self._edit_mode == EditMode.ADD:
            self._label[stroke_bool & (self._label == 0)] = self._current_class_id
        elif self._edit_mode == EditMode.SUBTRACT:
            # 현재 클래스 마스크에서 새 영역만큼 차집합
            self._label[stroke_bool & (self._label == self._current_class_id)] = 0

        self._refresh_composite()
        self.label_changed.emit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._composite_item is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self._cancel_drawing()
        elif key in (Qt.Key_Return, Qt.Key_Enter) and self._drawing:
            if self._stroke_raw:
                last = self._stroke_raw[-1]
                self._finish_stroke()
        elif key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif key == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.redo()
        elif key == Qt.Key_F:
            self.fit_in_view()
        else:
            super().keyPressEvent(event)

    def tabletEvent(self, event: QTabletEvent) -> None:
        if self._label is None:
            event.ignore()
            return

        pt = self._scene_pos(event)
        et = event.type()

        if et == QTabletEvent.TabletPress:
            if event.button() == Qt.LeftButton and self._can_draw():
                self._using_tablet = True
                self._start_stroke(pt)
            event.accept()
        elif et == QTabletEvent.TabletMove:
            if self._drawing and self._using_tablet:
                self._extend_stroke(pt)
            event.accept()
        elif et == QTabletEvent.TabletRelease:
            if self._using_tablet and event.button() == Qt.LeftButton and self._drawing:
                self._finish_stroke()
                self._using_tablet = False
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._label is None:
            super().mousePressEvent(event)
            return

        if self._using_tablet:
            event.ignore()
            return

        if event.button() == Qt.MidButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier
        ):
            self._panning = True
            self._pan_start = QPointF(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            self._cancel_drawing()
            return

        if event.button() == Qt.LeftButton and self._can_draw():
            self._start_stroke(self._scene_pos(event))
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._using_tablet:
            event.ignore()
            return

        if self._panning:
            delta = QPointF(event.pos()) - self._pan_start
            self._pan_start = QPointF(event.pos())
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            return

        if self._drawing and event.buttons() & Qt.LeftButton:
            self._extend_stroke(self._scene_pos(event))
        elif self._drawing:
            self._update_preview(self._scene_pos(event))

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._using_tablet:
            event.ignore()
            return

        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self._drawing:
            self._finish_stroke()
            return

        super().mouseReleaseEvent(event)
