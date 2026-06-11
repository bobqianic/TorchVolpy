from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .core import (
    TiffMovie,
    available_roi_ids,
    build_cellpose_rois,
    default_corrected_h5_path,
    ensure_shape_matches,
    extract_mean_trace,
    extract_mean_traces,
    freehand_to_mask,
    iter_spikepursuit_results,
    is_motion_corrected_h5,
    load_mask_file,
    motion_correct_h5,
    motion_correct_movie,
    normalize_to_uint8,
    open_movie,
    read_display_frame,
    release_torch_memory,
    rectangle_to_mask,
    run_ali,
    run_spikepursuit,
)

try:
    from PyQt6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, pyqtProperty, pyqtSignal, qInstallMessageHandler
    from PyQt6.QtGui import QBrush, QColor, QCursor, QFont, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QRegion, QShortcut
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractSpinBox,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGraphicsPathItem,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListView,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QProxyStyle,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QStyle,
        QVBoxLayout,
        QWidget,
    )

    PYQT_VERSION = 6
except ImportError as pyqt6_error:
    try:
        from PyQt5.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, pyqtProperty, pyqtSignal, qInstallMessageHandler
        from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QRegion
        from PyQt5.QtWidgets import (
            QApplication,
            QAbstractSpinBox,
            QBoxLayout,
            QCheckBox,
            QComboBox,
            QDialog,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QGraphicsPathItem,
            QGraphicsPixmapItem,
            QGraphicsRectItem,
            QGraphicsScene,
            QGraphicsSimpleTextItem,
            QGraphicsView,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListView,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QProxyStyle,
            QPushButton,
            QShortcut,
            QScrollArea,
            QSizePolicy,
            QSlider,
            QSpinBox,
            QStyle,
            QVBoxLayout,
            QWidget,
        )

        PYQT_VERSION = 5
    except ImportError as pyqt5_error:
        raise ImportError(
            "The torch-volpy GUI requires PyQt. Install it with "
            "`pip install 'torch-volpy[gui]'` or install PyQt6 manually."
        ) from pyqt5_error

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector


Signal = pyqtSignal


def _truthy_env(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "debug"}


def _qt_debug_enabled(argv: list[str]) -> bool:
    return _truthy_env(os.environ.get("TORCH_VOLPY_DEBUG")) or "--debug" in argv


def _quiet_qt_message_handler(message_type, context, message: str) -> None:
    name = getattr(message_type, "name", str(message_type))
    try:
        numeric_type = int(message_type)
    except (TypeError, ValueError):
        numeric_type = None
    is_critical = "Critical" in name or "Fatal" in name or numeric_type in {2, 3}
    if is_critical:
        sys.stderr.write(f"{message}\n")


def _configure_qt_message_logging(argv: list[str]) -> list[str]:
    filtered_argv = [arg for arg in argv if arg != "--debug"]
    if _qt_debug_enabled(argv):
        return filtered_argv
    qInstallMessageHandler(_quiet_qt_message_handler)
    return filtered_argv


def _qt_enum(group: str, name: str):
    namespace = getattr(Qt, group, None)
    if namespace is not None:
        return getattr(namespace, name)
    return getattr(Qt, name)


def _style_enum(group: str, name: str):
    namespace = getattr(QStyle, group, None)
    if namespace is not None:
        return getattr(namespace, name)
    return getattr(QStyle, name)


def _box_direction(name: str):
    namespace = getattr(QBoxLayout, "Direction", None)
    if namespace is not None:
        return getattr(namespace, name)
    return getattr(QBoxLayout, name)


def _event_enum(name: str):
    namespace = getattr(QEvent, "Type", None)
    if namespace is not None:
        return getattr(namespace, name)
    return getattr(QEvent, name)


LEFT_BUTTON = _qt_enum("MouseButton", "LeftButton")
RIGHT_BUTTON = _qt_enum("MouseButton", "RightButton")
HORIZONTAL = _qt_enum("Orientation", "Horizontal")
VERTICAL = _qt_enum("Orientation", "Vertical")
ALIGN_RIGHT = _qt_enum("AlignmentFlag", "AlignRight")
ALIGN_LEFT = _qt_enum("AlignmentFlag", "AlignLeft")
ALIGN_TOP = _qt_enum("AlignmentFlag", "AlignTop")
ALIGN_CENTER = _qt_enum("AlignmentFlag", "AlignCenter")
CLICK_FOCUS = _qt_enum("FocusPolicy", "ClickFocus")
KEEP_ASPECT = _qt_enum("AspectRatioMode", "KeepAspectRatio")
SMOOTH_TRANSFORM = _qt_enum("TransformationMode", "SmoothTransformation")
SCROLLBAR_ALWAYS_OFF = _qt_enum("ScrollBarPolicy", "ScrollBarAlwaysOff")
NO_BRUSH = _qt_enum("BrushStyle", "NoBrush")
SOLID_LINE = _qt_enum("PenStyle", "SolidLine")
ROUND_CAP = _qt_enum("PenCapStyle", "RoundCap")
ROUND_JOIN = _qt_enum("PenJoinStyle", "RoundJoin")
BOX_LEFT_TO_RIGHT = _box_direction("LeftToRight")
BOX_TOP_TO_BOTTOM = _box_direction("TopToBottom")
POPUP_WINDOW = _qt_enum("WindowType", "Popup")
TOOLTIP_WINDOW = _qt_enum("WindowType", "ToolTip")
FRAMELESS_WINDOW_HINT = _qt_enum("WindowType", "FramelessWindowHint")
NO_DROP_SHADOW_WINDOW_HINT = _qt_enum("WindowType", "NoDropShadowWindowHint")
WA_STYLED_BACKGROUND = _qt_enum("WidgetAttribute", "WA_StyledBackground")
WA_TRANSLUCENT_BACKGROUND = _qt_enum("WidgetAttribute", "WA_TranslucentBackground")
WA_TRANSPARENT_FOR_MOUSE_EVENTS = _qt_enum("WidgetAttribute", "WA_TransparentForMouseEvents")
EVENT_TOOLTIP = _event_enum("ToolTip")
EVENT_LEAVE = _event_enum("Leave")
EVENT_HIDE = _event_enum("Hide")
EVENT_MOUSE_BUTTON_PRESS = _event_enum("MouseButtonPress")
EVENT_MOUSE_MOVE = _event_enum("MouseMove")
EVENT_MOUSE_BUTTON_RELEASE = _event_enum("MouseButtonRelease")
EVENT_WINDOW_DEACTIVATE = _event_enum("WindowDeactivate")
SPIN_UP_INDICATOR = _style_enum("PrimitiveElement", "PE_IndicatorSpinUp")
SPIN_DOWN_INDICATOR = _style_enum("PrimitiveElement", "PE_IndicatorSpinDown")
STATE_ENABLED = _style_enum("StateFlag", "State_Enabled")
DEFAULT_WINDOW_WIDTH = 1560
DEFAULT_WINDOW_HEIGHT = 980
MIN_WINDOW_WIDTH = 820
MIN_WINDOW_HEIGHT = 520
INITIAL_SCREEN_WIDTH_FRACTION = 0.92
INITIAL_SCREEN_HEIGHT_FRACTION = 0.90
MOVIE_VIEW_MIN_HEIGHT = 120
ROI_MASK_HISTORY_LIMIT = 32
MOVIE_ZOOM_MIN_PERCENT = 10.0
MOVIE_ZOOM_MAX_PERCENT = 100000.0
APP_STYLESHEET = """
QWidget {
    color: #0f172a;
    font-family: "Segoe UI";
    font-size: 13px;
}

QMainWindow,
QDialog#advancedOptionsDialog,
QDialog#traceWindow,
QWidget#appRoot,
QWidget#workspaceBody,
QWidget#centerWorkspace,
QWidget#controlPanel,
QWidget#inspectorPanel,
QScrollArea#controlsScroll,
QScrollArea#controlsScroll > QWidget > QWidget {
    background-color: #f3f6fb;
}

QWidget#topBar {
    background-color: #ffffff;
    border-bottom: 1px solid #dfe7f2;
}

QLabel#topAppTitle {
    color: #0f172a;
    font-size: 18px;
    font-weight: 800;
}

QLabel#versionBadge,
QLabel#footerMetric,
QLabel[role="muted"] {
    color: #64748b;
}

QLabel#versionBadge {
    padding: 1px 10px;
    border-left: 1px solid #e2e8f0;
}

QWidget#footerBar {
    background-color: #ffffff;
    border-top: 1px solid #dfe7f2;
}

QLabel#readyDot {
    background-color: #10b981;
    border-radius: 5px;
}

QLabel#footerMetric {
    padding: 0 12px;
    border-left: 1px solid #e2e8f0;
}

QWidget#tracePanel,
QDialog#traceWindow,
QFrame#inspectorCard,
QFrame#metricTile {
    background-color: #ffffff;
    border: 1px solid #dfe7f2;
    border-radius: 8px;
}

QFrame#viewerCard {
    background-color: #ffffff;
    border: none;
    border-radius: 8px;
}

QWidget#viewerToolbar,
QWidget#traceTabs,
QWidget#inspectorTabs {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}

QWidget#tracePlotArea {
    background-color: #ffffff;
}

QLabel[role="eyebrow"] {
    color: #334155;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

QLabel[role="hero"] {
    color: #0f172a;
    font-size: 20px;
    font-weight: 700;
}

QLabel[role="metricLabel"] {
    color: #475569;
    font-size: 11px;
}

QLabel[role="metricValue"] {
    color: #0f172a;
    font-size: 18px;
    font-weight: 700;
}

QLabel[role="successValue"] {
    color: #059669;
    font-size: 18px;
    font-weight: 700;
}

QLabel[role="pill"],
QLabel[role="okPill"] {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 5px;
    padding: 4px 8px;
    font-weight: 600;
}

QLabel[role="okPill"] {
    background-color: #dcfce7;
    color: #047857;
    border-color: #a7f3d0;
}

QLabel[role="warningDot"] {
    background-color: #facc15;
    border-radius: 7px;
}

QFrame#verticalDivider {
    background-color: #e2e8f0;
    border: none;
}

QScrollArea#controlsScroll {
    border: none;
    background-color: transparent;
}

QScrollArea#advancedOptionsScroll,
QWidget#advancedOptionsViewport,
QWidget#advancedOptionsForm {
    background-color: #f3f6fb;
}

QFrame#advancedOptionsPanel {
    background-color: #f3f6fb;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
}

QScrollArea#advancedOptionsScroll {
    border: none;
}

QWidget#sidebarHeader {
    background-color: transparent;
    margin-bottom: 2px;
}

QLabel#appMark {
    background-color: #0f766e;
    border-radius: 6px;
}

QLabel#appTitle {
    color: #0f172a;
    font-size: 18px;
    font-weight: 700;
}

QLabel#appSubtitle {
    color: #64748b;
}

QLabel#frameCounter {
    color: #334155;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px 8px;
    font-weight: 600;
}

QLabel[role="status"] {
    color: #334155;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 7px 8px;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    margin-top: 0;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
    padding-right: 26px;
    min-height: 24px;
    selection-background-color: #2563eb;
}

QLabel#sectionTitle {
    color: #0f172a;
    font-size: 14px;
    font-weight: 700;
    padding: 0 0 6px 0;
}

QLineEdit:hover,
QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover {
    border-color: #94a3b8;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #2563eb;
}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background-color: #f1f5f9;
    border-color: #dbe3ef;
    color: #94a3b8;
}

QComboBox::drop-down {
    border: none;
    width: 26px;
}

QComboBox::down-arrow {
    image: url("__COMBOBOX_DOWN_ARROW__");
    width: 11px;
    height: 11px;
}

QComboBox::down-arrow:disabled {
    image: url("__COMBOBOX_DOWN_ARROW_DISABLED__");
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 0;
    color: #0f172a;
    outline: 0;
    padding: 3px;
    selection-background-color: #dbeafe;
    selection-color: #1d4ed8;
}

QFrame#comboPopup {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
}

QFrame#comboPopup QListView {
    background-color: #ffffff;
    border: none;
    border-radius: 5px;
    color: #0f172a;
    outline: 0;
    padding: 3px;
    selection-background-color: #dbeafe;
    selection-color: #1d4ed8;
}

QComboBox QAbstractItemView::item,
QListView::item {
    min-height: 28px;
    padding: 6px 10px;
    border-radius: 4px;
}

QComboBox QAbstractItemView::item:hover,
QListView::item:hover {
    background-color: #f1f5f9;
}

QComboBox QAbstractItemView::item:selected,
QListView::item:selected {
    background-color: #dbeafe;
    color: #1d4ed8;
}

QPushButton {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    color: #1e293b;
    font-weight: 600;
    min-height: 30px;
    padding: 6px 10px;
}

QPushButton:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #e2e8f0;
}

QPushButton:disabled {
    background-color: #f1f5f9;
    border-color: #dbe3ef;
    color: #94a3b8;
}

QPushButton[variant="primary"] {
    background-color: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}

QPushButton[variant="primary"]:hover {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
}

QPushButton[variant="primary"]:pressed {
    background-color: #1e40af;
    border-color: #1e40af;
}

QPushButton[variant="primary"]:disabled {
    background-color: #cbd5e1;
    border-color: #cbd5e1;
    color: #f8fafc;
}

QPushButton[variant="secondary"] {
    background-color: #f8fafc;
}

QPushButton[variant="toolbar"],
QPushButton[variant="toggle"],
QPushButton[variant="tab"],
QPushButton[variant="link"] {
    background-color: transparent;
    border-color: transparent;
    min-height: 26px;
    padding: 4px 8px;
    color: #1e293b;
}

QPushButton[variant="toolbar"]:hover,
QPushButton[variant="toggle"]:hover,
QPushButton[variant="tab"]:hover {
    background-color: #f1f5f9;
    border-color: #e2e8f0;
}

QPushButton[variant="toggle"]:checked {
    background-color: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}

QPushButton[variant="tab"] {
    border-radius: 0;
    border-bottom: 2px solid transparent;
    font-weight: 700;
    color: #334155;
}

QPushButton[variant="tab"]:checked {
    color: #2563eb;
    border-bottom-color: #2563eb;
}

QPushButton[variant="link"] {
    color: #2563eb;
    font-weight: 700;
}

QPushButton[variant="quiet"] {
    background-color: transparent;
    border-color: transparent;
}

QPushButton[variant="icon"] {
    background-color: rgba(255, 255, 255, 210);
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    min-height: 24px;
    min-width: 26px;
    padding: 0;
}

QPushButton[variant="iconQuiet"] {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    min-height: 24px;
    min-width: 26px;
    padding: 0;
}

QPushButton[variant="iconQuiet"]:hover {
    background-color: #f1f5f9;
}

QPushButton[variant="iconQuiet"]:pressed {
    background-color: #e2e8f0;
}

QPushButton[variant="iconQuiet"]:disabled {
    background-color: transparent;
    color: #94a3b8;
}

QProgressBar {
    background-color: #e2e8f0;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #0f766e;
    border-radius: 4px;
}

QSlider {
    min-height: 28px;
}

QSlider::groove:horizontal {
    height: 4px;
    background-color: #dbe3ef;
    border-radius: 2px;
    margin: 0 8px;
}

QSlider::sub-page:horizontal {
    background-color: #2563eb;
    border-radius: 2px;
    margin: 0 8px;
}

QSlider::add-page:horizontal {
    background-color: #dbe3ef;
    border-radius: 2px;
    margin: 0 8px;
}

QSlider::handle:horizontal {
    background-color: #2563eb;
    border: 2px solid #ffffff;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::groove:horizontal:disabled {
    background-color: #e2e8f0;
}

QSlider::sub-page:horizontal:disabled,
QSlider::add-page:horizontal:disabled {
    background-color: #e2e8f0;
}

QSlider::handle:horizontal:disabled {
    background-color: #cbd5e1;
    border-color: #f8fafc;
}

QCheckBox {
    spacing: 8px;
    color: #334155;
}

QGraphicsView#movieView {
    background-color: #f8fafc;
    border: none;
}

QFrame#roiToolIsland {
    background-color: rgba(255, 255, 255, 236);
    border: 1px solid #dfe7f2;
    border-radius: 8px;
}

QPushButton[variant="roiTool"] {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    min-height: 28px;
    min-width: 30px;
    padding: 0;
}

QPushButton[variant="roiTool"]:hover {
    background-color: #f1f5f9;
    border-color: #e2e8f0;
}

QPushButton[variant="roiTool"]:checked {
    background-color: #dbeafe;
    border-color: #60a5fa;
}

QWidget#tracePanel,
QWidget#traceContent {
    background-color: #ffffff;
}

QWidget#traceControls {
    background-color: #ffffff;
    border-top: 1px solid #e2e8f0;
}

QWidget#traceRestoreBar {
    background-color: #f8fafc;
    border-top: 1px solid #cbd5e1;
}

QScrollBar:vertical {
    background-color: transparent;
    width: 14px;
    margin: 2px 0;
}

QScrollBar::handle:vertical {
    background-color: #cbd5e1;
    border-radius: 3px;
    min-height: 28px;
    margin: 0 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #94a3b8;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background-color: transparent;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QWidget#roundedTooltip QLabel {
    background-color: transparent;
    color: #f8fafc;
}
"""
TRACE_LINE_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#65a30d",
    "#7c2d12",
    "#4f46e5",
    "#db2777",
    "#0f766e",
    "#ca8a04",
    "#7e22ce",
    "#16a34a",
    "#e11d48",
    "#0369a1",
    "#a16207",
    "#c026d3",
    "#15803d",
    "#f97316",
    "#0e7490",
    "#6d28d9",
    "#b91c1c",
    "#047857",
    "#a21caf",
    "#1d4ed8",
    "#854d0e",
    "#0d9488",
    "#9f1239",
    "#7c3aed",
    "#b45309",
)
STACKED_TRACE_ROW_PADDING = 1.15
STACKED_TRACE_RENDER_MAX_POINTS = 4000
STACKED_TRACE_OVERVIEW_MAX_POINTS = 1200
STACKED_DFF_ROW_HEIGHT = 0.34
STACKED_DFF_CLIP_HEIGHT = 0.42
ADVANCED_OPTION_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "Spikepursuit": (
        {"key": "template_size", "label": "Template size", "type": "float", "default": 0.02, "minimum": 0.001, "maximum": 1.0, "decimals": 4, "step": 0.005, "suffix": " s"},
        {"key": "context_size", "label": "Context size", "type": "int", "default": 35, "minimum": 0, "maximum": 1000, "step": 5, "suffix": " px"},
        {"key": "censor_size", "label": "Censor size", "type": "int", "default": 12, "minimum": 0, "maximum": 10000, "step": 1, "suffix": " frames"},
        {"key": "hp_freq_pb", "label": "Background high-pass", "type": "float", "default": 10.0, "minimum": 0.0, "maximum": 1000.0, "decimals": 4, "step": 0.1, "suffix": " Hz"},
        {"key": "nPC_bg", "label": "Background PCs", "type": "int", "default": 8, "minimum": 0, "maximum": 100, "step": 1},
        {"key": "ridge_bg", "label": "Background ridge", "type": "float", "default": 0.01, "minimum": 0.0, "maximum": 1000.0, "decimals": 5, "step": 0.01},
        {"key": "hp_freq", "label": "Trace high-pass", "type": "float", "default": 1.0, "minimum": 0.0, "maximum": 1000.0, "decimals": 4, "step": 0.1, "suffix": " Hz"},
        {"key": "clip", "label": "Max candidate spikes", "type": "int", "default": 100, "minimum": 0, "maximum": 100000, "step": 10},
        {"key": "threshold_method", "label": "Threshold method", "type": "choice", "default": "adaptive_threshold", "choices": ("adaptive_threshold", "simple")},
        {"key": "min_spikes", "label": "Minimum spikes", "type": "int", "default": 5, "minimum": 0, "maximum": 100000, "step": 1},
        {"key": "pnorm", "label": "Adaptive p-norm", "type": "float", "default": 0.5, "minimum": 0.0, "maximum": 10.0, "decimals": 4, "step": 0.05},
        {"key": "threshold", "label": "Simple threshold", "type": "float", "default": 2.0, "minimum": 0.0, "maximum": 1000.0, "decimals": 4, "step": 0.1},
        {"key": "sigmas", "label": "Spatial sigmas", "type": "float_list", "default": (1.0, 1.5, 2.0)},
        {"key": "n_iter", "label": "Iterations", "type": "int", "default": 2, "minimum": 1, "maximum": 100, "step": 1},
        {"key": "weight_update", "label": "Weight update", "type": "choice", "default": "ridge", "choices": ("ridge", "NMF")},
        {"key": "sub_freq", "label": "Sub-frequency", "type": "float", "default": 20.0, "minimum": 0.0, "maximum": 10000.0, "decimals": 3, "step": 1.0, "suffix": " Hz"},
        {"key": "roi_batch_patch_mb", "label": "ROI batch memory cap", "type": "float", "default": 0.0, "minimum": 0.0, "maximum": 1048576.0, "decimals": 1, "step": 64.0, "suffix": " MB"},
        {"key": "roi_batch_max_rois", "label": "Max ROIs per batch", "type": "int", "default": 0, "minimum": 0, "maximum": 100000, "step": 1},
        {"key": "prefetch_next_batch_patch", "label": "Prefetch next batch patch", "type": "bool", "default": True},
    ),
    "ALI": (
        {"key": "padding", "label": "ROI crop padding", "type": "int", "default": 0, "minimum": 0, "maximum": 1000, "step": 1, "suffix": " px"},
        {"key": "hp_window_ms", "label": "High-pass window", "type": "float", "default": 10.0, "minimum": 0.1, "maximum": 10000.0, "decimals": 3, "step": 1.0, "suffix": " ms"},
        {"key": "nsvd", "label": "SVD components", "type": "int", "default": 25, "minimum": 1, "maximum": 1000, "step": 1},
        {"key": "factor", "label": "Temporal factor", "type": "int", "default": 4, "minimum": 1, "maximum": 1000, "step": 1},
        {"key": "coarse_sigma", "label": "Coarse sigma", "type": "float", "default": 1.8, "minimum": 0.01, "maximum": 100.0, "decimals": 3, "step": 0.1},
        {"key": "coarse_gaussian_radius", "label": "Coarse radius", "type": "int", "default": 2, "minimum": 1, "maximum": 100, "step": 1},
        {"key": "coarse_threshold_std", "label": "Coarse threshold", "type": "float", "default": 3.0, "minimum": 0.0, "maximum": 100.0, "decimals": 3, "step": 0.1, "suffix": " std"},
        {"key": "min_component_size", "label": "Min component size", "type": "int", "default": 4, "minimum": 1, "maximum": 100000, "step": 1},
        {"key": "fine_npix", "label": "Fine pixels", "type": "int", "default": 15, "minimum": 1, "maximum": 100000, "step": 1},
        {"key": "fine_radius", "label": "Fine radius", "type": "float", "default": 4.0, "minimum": 0.0, "maximum": 1000.0, "decimals": 3, "step": 0.5},
        {"key": "cluster_threshold", "label": "Cluster threshold", "type": "float", "default": 2.0, "minimum": 0.0, "maximum": 1000.0, "decimals": 3, "step": 0.1},
        {"key": "peak_kernel_size", "label": "Peak kernel", "type": "int", "default": 3, "minimum": 1, "maximum": 999, "step": 2},
        {"key": "assign_radius", "label": "Assign radius", "type": "float", "default": 1.5, "minimum": 0.0, "maximum": 1000.0, "decimals": 3, "step": 0.1},
        {"key": "alimap_sigma", "label": "ALI map sigma", "type": "float", "default": 0.7, "minimum": 0.01, "maximum": 100.0, "decimals": 3, "step": 0.1},
        {"key": "alimap_gaussian_radius", "label": "ALI map radius", "type": "int", "default": 2, "minimum": 1, "maximum": 100, "step": 1},
        {"key": "footprint_radius", "label": "Footprint radius", "type": "float", "default": 10.0, "minimum": 0.0, "maximum": 1000.0, "decimals": 3, "step": 1.0},
        {"key": "solve_eps", "label": "Solve epsilon", "type": "float", "default": 1e-6, "minimum": 0.0, "maximum": 1.0, "decimals": 8, "step": 0.000001},
        {"key": "cc_max_iter", "label": "CC max iterations", "type": "int", "default": 2048, "minimum": 1, "maximum": 1000000, "step": 128},
    ),
    "Mean ROI": (
        {"key": "batch_size", "label": "Batch size", "type": "int", "default": 256, "minimum": 1, "maximum": 100000, "step": 64, "suffix": " frames"},
    ),
}

if PYQT_VERSION == 6:
    FORMAT_GRAY8 = QImage.Format.Format_Grayscale8
    FORMAT_RGB888 = QImage.Format.Format_RGB888
    FORMAT_RGBA = QImage.Format.Format_RGBA8888
    SIZE_EXPANDING = QSizePolicy.Policy.Expanding
    SIZE_IGNORED = QSizePolicy.Policy.Ignored
    SIZE_FIXED = QSizePolicy.Policy.Fixed
    ANTIALIASING = QPainter.RenderHint.Antialiasing
    SMOOTH_PIXMAP = QPainter.RenderHint.SmoothPixmapTransform
    SPINBOX_UP_DOWN_BUTTONS = QAbstractSpinBox.ButtonSymbols.UpDownArrows
    SPINBOX_NO_BUTTONS = QAbstractSpinBox.ButtonSymbols.NoButtons
else:
    FORMAT_GRAY8 = QImage.Format_Grayscale8
    FORMAT_RGB888 = QImage.Format_RGB888
    FORMAT_RGBA = QImage.Format_RGBA8888
    SIZE_EXPANDING = QSizePolicy.Expanding
    SIZE_IGNORED = QSizePolicy.Ignored
    SIZE_FIXED = QSizePolicy.Fixed
    ANTIALIASING = QPainter.Antialiasing
    SMOOTH_PIXMAP = QPainter.SmoothPixmapTransform
    SPINBOX_UP_DOWN_BUTTONS = QAbstractSpinBox.UpDownArrows
    SPINBOX_NO_BUTTONS = QAbstractSpinBox.NoButtons


TRACE_SIGNAL_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M3.5 11h2.4l1.4-5.5 2.7 10 2-7.5 1.2 3h3.3" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

APP_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
  <rect x="7" y="7" width="50" height="50" rx="12" fill="#0f172a"/>
  <path d="M15 38h7l4-14 7 25 5-18h11" fill="none" stroke="#38bdf8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="47" cy="25" r="4" fill="#14b8a6"/>
</svg>
"""

FOLDER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M3 6.5A2.5 2.5 0 0 1 5.5 4h3l1.8 2h4.2A2.5 2.5 0 0 1 17 8.5v5A2.5 2.5 0 0 1 14.5 16h-9A2.5 2.5 0 0 1 3 13.5v-7Z" fill="none" stroke="{color}" stroke-width="1.7" stroke-linejoin="round"/>
</svg>
"""

PLAY_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M7 5.5v9l7-4.5-7-4.5Z" fill="{color}"/>
</svg>
"""

PAUSE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M6 5h3v10H6V5Zm5 0h3v10h-3V5Z" fill="{color}"/>
</svg>
"""

SLIDERS_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M4 6h4m4 0h4M4 10h8m4 0h0M4 14h2m4 0h6" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round"/>
  <circle cx="10" cy="6" r="2" fill="none" stroke="{color}" stroke-width="1.7"/>
  <circle cx="14" cy="10" r="2" fill="none" stroke="{color}" stroke-width="1.7"/>
  <circle cx="8" cy="14" r="2" fill="none" stroke="{color}" stroke-width="1.7"/>
</svg>
"""

TARGET_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <circle cx="10" cy="10" r="6" fill="none" stroke="{color}" stroke-width="1.7"/>
  <circle cx="10" cy="10" r="2" fill="{color}"/>
  <path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round"/>
</svg>
"""

DOWNLOAD_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M10 4v8m0 0 3-3m-3 3-3-3M4.5 15.5h11" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SPARK_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M10 3.5 11.7 8l4.8 1.7-4.8 1.6L10 16l-1.7-4.7-4.8-1.6L8.3 8 10 3.5Z" fill="none" stroke="{color}" stroke-width="1.7" stroke-linejoin="round"/>
</svg>
"""

ZOOM_OUT_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <circle cx="8.5" cy="8.5" r="5" fill="none" stroke="{color}" stroke-width="1.7"/>
  <path d="M6.2 8.5h4.6m1.5 3.8 3.4 3.4" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round"/>
</svg>
"""

ZOOM_IN_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <circle cx="8.5" cy="8.5" r="5" fill="none" stroke="{color}" stroke-width="1.7"/>
  <path d="M6.2 8.5h4.6M8.5 6.2v4.6m3.8 1.5 3.4 3.4" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round"/>
</svg>
"""

RESET_VIEW_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M15.6 7.4A6 6 0 1 0 16 10" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M15.8 3.8v3.6h-3.6" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

BRUSH_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M12.8 4.2 16 7.4l-6.4 6.4H6.4v-3.2l6.4-6.4Z" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M5.8 14.4c-.9.9-1.8 1.3-3.1 1.2.4-1.2.7-2.2 1.6-3.1" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

RECTANGLE_ROI_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <rect x="4.2" y="5.2" width="11.6" height="9.6" rx="1.2" fill="none" stroke="{color}" stroke-width="1.7"/>
</svg>
"""

ERASER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M8.1 15.2h7.2" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M4.1 10.5 10.8 3.8a1.7 1.7 0 0 1 2.4 0l2 2a1.7 1.7 0 0 1 0 2.4L8.7 14.7a1.9 1.9 0 0 1-2.7 0l-1.9-1.9a1.9 1.9 0 0 1 0-2.7Z" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M7.1 7.5 11.5 12" fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

UNDO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M7.3 7H5.1V4.8" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M5.4 7.1A5.6 5.6 0 1 1 5.1 12" fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round"/>
</svg>
"""


def _icon_from_svg(svg: str) -> QIcon:
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode("utf-8"), "SVG")
    return QIcon(pixmap)


def _icon_from_template(svg: str, color: str = "#334155") -> QIcon:
    return _icon_from_svg(svg.format(color=color))


def _set_button_icon(button: QPushButton, svg: str, color: str = "#334155") -> None:
    button.setIcon(_icon_from_template(svg, color))
    button.setIconSize(QSize(16, 16))


def _set_button_role(button: QPushButton, role: str) -> None:
    button.setProperty("variant", role)


class _ModernSpinBoxMixin:
    _button_width = 22
    _arrow_spacing = 6.0

    def _init_modern_spinbox(self) -> None:
        self.setButtonSymbols(SPINBOX_NO_BUTTONS)
        self.setMouseTracking(True)
        self._pressed_spin_button: Optional[str] = None
        self._hover_spin_button: Optional[str] = None

    def _spin_button_rect(self) -> QRect:
        rect = self.rect().adjusted(1, 1, -1, -1)
        width = min(self._button_width, rect.width())
        return QRect(rect.right() - width + 1, rect.top(), width, rect.height())

    def _spin_button_at(self, point) -> Optional[str]:
        rect = self._spin_button_rect()
        if not self.isEnabled() or not rect.contains(point):
            return None
        return "up" if point.y() < rect.top() + rect.height() / 2 else "down"

    def _can_step_spin_button(self, button: str) -> bool:
        if button == "up":
            return self.value() < self.maximum()
        return self.value() > self.minimum()

    def _step_spin_button(self, button: str) -> None:
        if self._can_step_spin_button(button):
            self.stepBy(1 if button == "up" else -1)

    def _event_point(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _draw_modern_spin_arrow(self, painter: QPainter, center: QPointF, *, up: bool, enabled: bool) -> None:
        offset = 2.6
        if up:
            points = [
                QPointF(center.x() - offset, center.y() + 1.4),
                QPointF(center.x(), center.y() - 1.4),
                QPointF(center.x() + offset, center.y() + 1.4),
            ]
        else:
            points = [
                QPointF(center.x() - offset, center.y() - 1.4),
                QPointF(center.x(), center.y() + 1.4),
                QPointF(center.x() + offset, center.y() - 1.4),
            ]

        pen = QPen(QColor("#64748b" if enabled else "#cbd5e1"), 1.5)
        pen.setCapStyle(ROUND_CAP)
        pen.setJoinStyle(ROUND_JOIN)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF(points))

    def _draw_modern_spin_buttons(self) -> None:
        rect = self._spin_button_rect()
        rect_f = QRectF(rect)
        enabled = self.isEnabled()
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIASING)

        radius = 5.0
        path = QPainterPath()
        path.moveTo(rect_f.left(), rect_f.top())
        path.lineTo(rect_f.right() - radius, rect_f.top())
        path.quadTo(rect_f.right(), rect_f.top(), rect_f.right(), rect_f.top() + radius)
        path.lineTo(rect_f.right(), rect_f.bottom() - radius)
        path.quadTo(rect_f.right(), rect_f.bottom(), rect_f.right() - radius, rect_f.bottom())
        path.lineTo(rect_f.left(), rect_f.bottom())
        path.closeSubpath()

        painter.fillPath(path, QColor("#f8fafc" if enabled else "#f1f5f9"))
        active_button = self._pressed_spin_button or self._hover_spin_button
        if enabled and active_button is not None:
            painter.save()
            painter.setClipPath(path)
            half_height = rect.height() / 2
            if active_button == "up":
                painter.fillRect(QRectF(rect_f.left(), rect_f.top(), rect_f.width(), half_height), QColor("#e2e8f0"))
            else:
                painter.fillRect(
                    QRectF(rect_f.left(), rect_f.top() + half_height, rect_f.width(), rect_f.height() - half_height),
                    QColor("#e2e8f0"),
                )
            painter.restore()

        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawLine(QPointF(rect_f.left(), rect_f.top()), QPointF(rect_f.left(), rect_f.bottom()))

        center_x = rect_f.center().x()
        center_y = rect_f.center().y()
        self._draw_modern_spin_arrow(
            painter,
            QPointF(center_x, center_y - self._arrow_spacing),
            up=True,
            enabled=enabled and self._can_step_spin_button("up"),
        )
        self._draw_modern_spin_arrow(
            painter,
            QPointF(center_x, center_y + self._arrow_spacing),
            up=False,
            enabled=enabled and self._can_step_spin_button("down"),
        )

    def paintEvent(self, event) -> None:
        """Paint the custom spinbox frame and arrow controls."""
        super().paintEvent(event)
        self._draw_modern_spin_buttons()

    def mousePressEvent(self, event) -> None:
        """Track pointer presses on custom arrow controls."""
        point = self._event_point(event)
        button = self._spin_button_at(point)
        if event.button() == LEFT_BUTTON and button is not None:
            self._pressed_spin_button = button
            self._step_spin_button(button)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Complete a custom arrow press and trigger the step action."""
        if self._pressed_spin_button is not None:
            self._pressed_spin_button = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Update hover state for custom arrow controls."""
        point = self._event_point(event)
        hover_button = self._spin_button_at(point)
        if hover_button != self._hover_spin_button:
            self._hover_spin_button = hover_button
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        """Clear hover state when the pointer leaves the spinbox."""
        if self._hover_spin_button is not None:
            self._hover_spin_button = None
            self.update()
        super().leaveEvent(event)


class ModernSpinBox(_ModernSpinBoxMixin, QSpinBox):
    """Integer spinbox with custom modern arrow rendering."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_modern_spinbox()


class RoiBrushSizeSpinBox(ModernSpinBox):
    """Spinbox for controlling ROI brush size in the movie view."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._vertical_display = False

    def sizeHint(self) -> QSize:
        """Return the preferred size for the ROI brush spinbox."""
        if self._vertical_display:
            return QSize(32, 88)
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        """Return the minimum usable size for the ROI brush spinbox."""
        if self._vertical_display:
            return QSize(32, 88)
        return super().minimumSizeHint()

    def set_vertical_display(self, vertical: bool) -> None:
        """Switch the brush-size control between vertical and horizontal display."""
        vertical = bool(vertical)
        self._vertical_display = vertical
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        if vertical:
            self.setSuffix("")
            self.setFixedSize(32, 88)
        else:
            self.setSuffix(" px")
            self.setFixedWidth(70)
        self._sync_line_edit_visibility()
        self.updateGeometry()
        self.update()

    def _sync_line_edit_visibility(self) -> None:
        self.lineEdit().setVisible(not self._vertical_display)

    def _spin_button_at(self, point) -> Optional[str]:
        if not self._vertical_display:
            return super()._spin_button_at(point)
        if not self.isEnabled():
            return None
        if point.y() <= 18:
            return "up"
        if point.y() >= self.height() - 18:
            return "down"
        return None

    def paintEvent(self, event) -> None:
        """Paint the ROI brush-size value and custom arrow controls."""
        if not self._vertical_display:
            super().paintEvent(event)
            return

        self._sync_line_edit_visibility()
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIASING)
        enabled = self.isEnabled()
        outer = QRectF(0.5, 0.5, max(1.0, self.width() - 1.0), max(1.0, self.height() - 1.0))
        painter.setPen(QPen(QColor("#cbd5e1" if enabled else "#dbe3ef"), 1))
        painter.setBrush(QBrush(QColor("#ffffff" if enabled else "#f8fafc")))
        painter.drawRoundedRect(outer, 7.0, 7.0)

        top_button = QRectF(1.0, 1.0, max(1.0, self.width() - 2.0), 18.0)
        bottom_button = QRectF(1.0, max(1.0, self.height() - 19.0), max(1.0, self.width() - 2.0), 18.0)
        active_button = self._pressed_spin_button or self._hover_spin_button
        if enabled and active_button == "up":
            painter.fillRect(top_button.adjusted(1.0, 1.0, -1.0, 0.0), QColor("#e2e8f0"))
        elif enabled and active_button == "down":
            painter.fillRect(bottom_button.adjusted(1.0, 0.0, -1.0, -1.0), QColor("#e2e8f0"))

        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawLine(QPointF(outer.left(), top_button.bottom()), QPointF(outer.right(), top_button.bottom()))
        painter.drawLine(QPointF(outer.left(), bottom_button.top()), QPointF(outer.right(), bottom_button.top()))

        self._draw_modern_spin_arrow(
            painter,
            QPointF(top_button.center().x(), top_button.center().y() + 0.5),
            up=True,
            enabled=enabled and self._can_step_spin_button("up"),
        )
        self._draw_modern_spin_arrow(
            painter,
            QPointF(bottom_button.center().x(), bottom_button.center().y() - 0.5),
            up=False,
            enabled=enabled and self._can_step_spin_button("down"),
        )

        value_rect = QRectF(1.0, top_button.bottom(), max(1.0, self.width() - 2.0), max(1.0, bottom_button.top() - top_button.bottom()))
        painter.save()
        painter.translate(value_rect.center())
        painter.rotate(-90.0)
        rotated_rect = QRectF(
            -value_rect.height() / 2.0,
            -value_rect.width() / 2.0,
            value_rect.height(),
            value_rect.width(),
        )
        painter.setPen(QPen(QColor("#0f172a" if enabled else "#94a3b8")))
        painter.drawText(rotated_rect, ALIGN_CENTER, f"{self.value()} px")
        painter.restore()

    def resizeEvent(self, event) -> None:
        """Recalculate arrow hit regions after resizing."""
        super().resizeEvent(event)
        if self._vertical_display:
            self._sync_line_edit_visibility()

    def showEvent(self, event) -> None:
        """Initialize arrow hit regions when the widget is shown."""
        super().showEvent(event)
        self._sync_line_edit_visibility()
        if self._vertical_display:
            QTimer.singleShot(0, self._sync_line_edit_visibility)


class ModernDoubleSpinBox(_ModernSpinBoxMixin, QDoubleSpinBox):
    """Floating-point spinbox with custom modern arrow rendering."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_modern_spinbox()


def _show_spinbox_buttons(spinbox: QAbstractSpinBox) -> None:
    if isinstance(spinbox, _ModernSpinBoxMixin):
        spinbox.setButtonSymbols(SPINBOX_NO_BUTTONS)
    else:
        spinbox.setButtonSymbols(SPINBOX_UP_DOWN_BUTTONS)


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def _styled_frame(name: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    frame.setAttribute(WA_STYLED_BACKGROUND, True)
    return frame


def _role_label(text: str = "", role: str = "muted") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", role)
    return label


def _make_tab_button(text: str, checked: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setCheckable(True)
    button.setChecked(checked)
    _set_button_role(button, "tab")
    return button


def _make_metric_tile(title: str, value: str = "--", role: str = "metricValue") -> tuple[QFrame, QLabel]:
    tile = _styled_frame("metricTile")
    layout = QVBoxLayout(tile)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(3)
    label = _role_label(title, "metricLabel")
    value_label = _role_label(value, role)
    layout.addWidget(label)
    layout.addWidget(value_label)
    return tile, value_label


def _set_metric_value_color(label: QLabel, color: str) -> None:
    label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 700;")


def _normalize_rgb_to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        image = normalize_to_uint8(arr)
        return np.repeat(image[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"RGB display requires a 2D or 3D frame, got {arr.shape}")

    if arr.shape[2] < 3:
        image = normalize_to_uint8(arr[..., 0])
        return np.repeat(image[:, :, None], 3, axis=2)

    channels = []
    for channel_index in range(3):
        channels.append(normalize_to_uint8(arr[..., channel_index]))
    return np.stack(channels, axis=2)


def _make_toolbar_button(text: str, svg: Optional[str] = None, *, checkable: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setCheckable(checkable)
    _set_button_role(button, "toggle" if checkable else "toolbar")
    if svg is not None:
        _set_button_icon(button, svg)
    return button


def _make_roi_tool_button(svg: str, tooltip: str, *, checkable: bool = True) -> QPushButton:
    button = QPushButton()
    button.setCheckable(checkable)
    button.setIcon(_icon_from_template(svg))
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(32, 30)
    button.setToolTip(tooltip)
    _set_button_role(button, "roiTool")
    return button


def _make_divider() -> QFrame:
    divider = QFrame()
    divider.setObjectName("verticalDivider")
    divider.setFixedWidth(1)
    divider.setAttribute(WA_STYLED_BACKGROUND, True)
    return divider


class RoiToolIsland(QFrame):
    """Floating toolbar that hosts ROI drawing and selection controls."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("roiToolIsland")
        self.setAttribute(WA_STYLED_BACKGROUND, True)
        self.setSizePolicy(SIZE_FIXED, SIZE_FIXED)

        self.margin = 10
        self.dock_side = "left"
        self._drag_global_pos = None
        self._drag_widget_pos = None
        self._drag_source: Optional[QObject] = None
        self._dragging = False
        self._drag_threshold = 6
        self._drag_button_threshold = max(14, int(QApplication.startDragDistance()))

        self.tool_layout = QBoxLayout(BOX_TOP_TO_BOTTOM, self)
        self.tool_layout.setContentsMargins(5, 6, 5, 6)
        self.tool_layout.setSpacing(4)

        self.installEventFilter(self)
        self._apply_orientation()

    def eventFilter(self, obj, event) -> bool:
        """Handle drag interactions for the floating ROI tool island."""
        if obj is self or self.isAncestorOf(obj):
            event_type = event.type()
            if event_type == EVENT_MOUSE_BUTTON_PRESS and event.button() == LEFT_BUTTON:
                self._drag_global_pos = _event_global_pos(event)
                self._drag_widget_pos = self.pos()
                self._drag_source = obj if isinstance(obj, QObject) else None
                self._dragging = False
                self.raise_()
                return False
            if self._drag_global_pos is not None and event_type == EVENT_MOUSE_MOVE:
                delta = _event_global_pos(event) - self._drag_global_pos
                threshold = (
                    self._drag_button_threshold
                    if isinstance(self._drag_source, (QPushButton, QAbstractSpinBox, QLineEdit))
                    else self._drag_threshold
                )
                if not self._dragging and delta.manhattanLength() < threshold:
                    return False
                self._dragging = True
                if isinstance(self._drag_source, QPushButton):
                    self._drag_source.setDown(False)
                target = self._drag_widget_pos + delta
                self._move_clamped(target.x(), target.y())
                event.accept()
                return True
            if self._drag_global_pos is not None and event_type == EVENT_MOUSE_BUTTON_RELEASE:
                was_dragging = self._dragging
                if was_dragging and isinstance(self._drag_source, QPushButton):
                    self._drag_source.setDown(False)
                self._drag_global_pos = None
                self._drag_widget_pos = None
                self._drag_source = None
                self._dragging = False
                if was_dragging:
                    self.snap_to_nearest_side()
                    event.accept()
                    return True
                return False
        return super().eventFilter(obj, event)

    def add_tool_widget(self, widget: QWidget) -> None:
        """Add a tool control to the ROI tool island."""
        self._install_drag_filter(widget)
        self.tool_layout.addWidget(widget)

    def _install_drag_filter(self, widget: QWidget) -> None:
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    def set_dock_side(self, side: str) -> None:
        """Set the side of the movie stage where the island is docked."""
        side = side if side in {"left", "right", "top", "bottom"} else "left"
        if side != self.dock_side:
            self.dock_side = side
            self._apply_orientation()
        self.reposition_to_dock()

    def snap_to_nearest_side(self) -> None:
        """Dock the island to the nearest side of the target widget."""
        parent = self.parentWidget()
        if parent is None:
            return
        center_x = self.x() + self.width() / 2.0
        center_y = self.y() + self.height() / 2.0
        distances = {
            "left": center_x,
            "right": max(0.0, parent.width() - center_x),
            "top": center_y,
            "bottom": max(0.0, parent.height() - center_y),
        }
        self.set_dock_side(min(distances, key=distances.get))

    def reposition_to_dock(self) -> None:
        """Move the island to its current docked position."""
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        if self.dock_side in {"top", "bottom"}:
            x = self.x() if self.x() > 0 else self.margin
            y = self.margin if self.dock_side == "top" else parent.height() - self.height() - self.margin
        else:
            x = self.margin if self.dock_side == "left" else parent.width() - self.width() - self.margin
            y = self.y() if self.y() > 0 else self.margin
        self._move_clamped(x, y)
        self.raise_()

    def _apply_orientation(self) -> None:
        horizontal = self.dock_side in {"top", "bottom"}
        self.tool_layout.setDirection(BOX_LEFT_TO_RIGHT if horizontal else BOX_TOP_TO_BOTTOM)
        if horizontal:
            self.tool_layout.setContentsMargins(6, 5, 6, 5)
        else:
            self.tool_layout.setContentsMargins(5, 6, 5, 6)
        for index in range(self.tool_layout.count()):
            item = self.tool_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if isinstance(widget, RoiBrushSizeSpinBox):
                widget.set_vertical_display(not horizontal)
        self.adjustSize()

    def _move_clamped(self, x: float, y: float) -> None:
        parent = self.parentWidget()
        if parent is None:
            self.move(int(round(x)), int(round(y)))
            return
        max_x = max(self.margin, parent.width() - self.width() - self.margin)
        max_y = max(self.margin, parent.height() - self.height() - self.margin)
        clamped_x = max(self.margin, min(float(max_x), float(x)))
        clamped_y = max(self.margin, min(float(max_y), float(y)))
        self.move(int(round(clamped_x)), int(round(clamped_y)))


class MovieCardBorderOverlay(QWidget):
    """Overlay widget that draws the movie card border."""
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(WA_TRANSPARENT_FOR_MOUSE_EVENTS, True)
        self.setAttribute(WA_TRANSLUCENT_BACKGROUND, True)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:
        """Paint the movie card border overlay."""
        outer_rect = QRectF(self.rect())
        border_rect = outer_rect.adjusted(0.5, 0.5, -0.5, -0.5)
        if border_rect.width() <= 0 or border_rect.height() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(ANTIALIASING)

        outer_path = QPainterPath()
        outer_path.addRect(outer_rect)
        card_path = QPainterPath()
        card_path.addRoundedRect(border_rect, 8.0, 8.0)
        painter.fillPath(outer_path.subtracted(card_path), QBrush(QColor("#f3f6fb")))

        painter.setBrush(QBrush(NO_BRUSH))
        pen = QPen(QColor("#dfe7f2"), 1)
        pen.setJoinStyle(ROUND_JOIN)
        painter.setPen(pen)
        painter.drawRoundedRect(border_rect, 8.0, 8.0)


class MovieCardFrame(QFrame):
    """Frame widget that keeps a border overlay above movie controls."""
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewerCard")
        self.setAttribute(WA_STYLED_BACKGROUND, True)
        self.border_overlay = MovieCardBorderOverlay(self)
        self.raise_border()

    def raise_border(self) -> None:
        """Raise the border overlay above child widgets."""
        self.border_overlay.setGeometry(self.rect())
        self.border_overlay.raise_()
        self.border_overlay.update()

    def resizeEvent(self, event) -> None:
        """Resize the border overlay with the movie card frame."""
        super().resizeEvent(event)
        self.raise_border()


class MovieStage(QWidget):
    """Movie viewing stage that positions floating ROI controls."""
    resized = Signal()

    def resizeEvent(self, event) -> None:
        """Reposition floating controls when the movie stage resizes."""
        super().resizeEvent(event)
        self.resized.emit()


def _modernize_combo_box(combo_box: QComboBox) -> None:
    view = QListView(combo_box)
    view.setFrameShape(QFrame.Shape.NoFrame if PYQT_VERSION == 6 else QFrame.NoFrame)
    view.setUniformItemSizes(True)
    combo_box.setView(view)


class _ComboPopupFrame(QFrame):
    def __init__(self, combo_box: "ModernComboBox", flags) -> None:
        super().__init__(None, flags)
        self._combo_box = combo_box

    def hideEvent(self, event) -> None:
        """Notify the combo box when the custom popup frame closes."""
        self._combo_box._popup_frame_hidden()
        super().hideEvent(event)


class ModernComboBox(QComboBox):
    """Combo box with a custom popup frame and styling."""
    _POPUP_REOPEN_GUARD_SECONDS = 0.15

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        _modernize_combo_box(self)
        self._popup_frame: Optional[QFrame] = None
        self._popup_view: Optional[QListView] = None
        self._ignore_popup_show_until = 0.0

    def mousePressEvent(self, event) -> None:
        """Open the custom combo popup on mouse press."""
        if event.button() == LEFT_BUTTON and self._popup_frame is not None and self._popup_frame.isVisible():
            self.hidePopup()
            self._ignore_popup_show_until = 0.0
            event.accept()
            return
        super().mousePressEvent(event)

    def showPopup(self) -> None:
        """Display the custom-styled combo popup."""
        if time.monotonic() < self._ignore_popup_show_until:
            self._ignore_popup_show_until = 0.0
            return

        if self.count() <= 0:
            return

        if self._popup_frame is None:
            self._popup_frame = _ComboPopupFrame(self, POPUP_WINDOW | FRAMELESS_WINDOW_HINT | NO_DROP_SHADOW_WINDOW_HINT)
            self._popup_frame.setObjectName("comboPopup")
            self._popup_frame.setAttribute(WA_STYLED_BACKGROUND, True)
            self._popup_frame.setAutoFillBackground(False)
            popup_layout = QVBoxLayout(self._popup_frame)
            popup_layout.setContentsMargins(1, 1, 1, 1)
            popup_layout.setSpacing(0)

            self._popup_view = QListView(self._popup_frame)
            self._popup_view.setFrameShape(QFrame.Shape.NoFrame if PYQT_VERSION == 6 else QFrame.NoFrame)
            self._popup_view.setUniformItemSizes(True)
            self._popup_view.setModel(self.model())
            self._popup_view.clicked.connect(self._select_popup_index)
            self._popup_view.activated.connect(self._select_popup_index)
            popup_layout.addWidget(self._popup_view)

        if self._popup_view is None or self._popup_frame is None:
            return

        self._popup_view.setModel(self.model())
        current = self.model().index(max(0, self.currentIndex()), self.modelColumn(), self.rootModelIndex())
        if current.isValid():
            self._popup_view.setCurrentIndex(current)
            self._popup_view.scrollTo(current)

        visible_rows = max(1, min(self.count(), self.maxVisibleItems()))
        row_height = self._popup_view.sizeHintForRow(max(0, self.currentIndex()))
        if row_height <= 0:
            row_height = 32
        popup_width = max(self.width(), self.sizeHint().width())
        popup_height = visible_rows * row_height + 10
        self._popup_frame.resize(popup_width, popup_height)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, popup_width, popup_height), 6, 6)
        self._popup_frame.setMask(QRegion(path.toFillPolygon().toPolygon()))
        self._popup_frame.move(self.mapToGlobal(self.rect().bottomLeft()))
        self._popup_frame.show()
        self._popup_view.setFocus()

    def hidePopup(self) -> None:
        """Hide the custom-styled combo popup."""
        if self._popup_frame is not None:
            self._popup_frame.hide()

    def _popup_frame_hidden(self) -> None:
        if self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self._ignore_popup_show_until = time.monotonic() + self._POPUP_REOPEN_GUARD_SECONDS

    def _select_popup_index(self, index) -> None:
        if index.isValid():
            self.setCurrentIndex(index.row())
        self.hidePopup()


class RoundedToolTip(QWidget):
    """Custom rounded tooltip widget for the GUI."""
    def __init__(self) -> None:
        super().__init__(None, TOOLTIP_WINDOW | FRAMELESS_WINDOW_HINT | NO_DROP_SHADOW_WINDOW_HINT)
        self.setObjectName("roundedTooltip")
        self.setAttribute(WA_TRANSLUCENT_BACKGROUND, True)
        self.setAutoFillBackground(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(0)

        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(520)
        layout.addWidget(self.label)

    def paintEvent(self, event) -> None:
        """Paint the rounded tooltip background and text."""
        painter = QPainter(self)
        antialiasing = QPainter.RenderHint.Antialiasing if hasattr(QPainter, "RenderHint") else QPainter.Antialiasing
        painter.setRenderHint(antialiasing, True)
        rect = QRectF(0.5, 0.5, max(0, self.width() - 1), max(0, self.height() - 1))
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRoundedRect(rect, 6, 6)
        super().paintEvent(event)

    def show_text(self, text: str, pos) -> None:
        """Show tooltip text near a global screen position."""
        self.label.setText(text)
        self.adjustSize()
        target_x = pos.x() + 14
        target_y = pos.y() + 20

        screen = QApplication.screenAt(pos) if hasattr(QApplication, "screenAt") else QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_x = min(max(available.left(), target_x), available.right() - self.width())
            target_y = min(max(available.top(), target_y), available.bottom() - self.height())

        self.move(target_x, target_y)
        self.show()
        self.raise_()


class RoundedToolTipFilter(QObject):
    """Event filter that manages rounded tooltips for watched widgets."""
    def __init__(self, parent: QApplication) -> None:
        super().__init__(parent)
        self.tooltip = RoundedToolTip()

    def eventFilter(self, obj, event) -> bool:
        """Show or hide a rounded tooltip for watched widgets."""
        event_type = event.type()
        if event_type == EVENT_TOOLTIP:
            text = obj.toolTip() if isinstance(obj, QWidget) else ""
            if text:
                pos = event.globalPos() if hasattr(event, "globalPos") else QCursor.pos()
                self.tooltip.show_text(text, pos)
                return True
            self.tooltip.hide()
        elif event_type in (EVENT_LEAVE, EVENT_HIDE, EVENT_MOUSE_BUTTON_PRESS, EVENT_WINDOW_DEACTIVATE):
            self.tooltip.hide()
        return super().eventFilter(obj, event)


class SwitchButton(QCheckBox):
    """Animated toggle switch used by the GUI."""
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QCursor(_qt_enum("CursorShape", "PointingHandCursor")))
        self.setMinimumSize(self.sizeHint())

    def sizeHint(self) -> QSize:
        """Return the preferred size for the switch button."""
        text_width = self.fontMetrics().horizontalAdvance(self.text()) if hasattr(self.fontMetrics(), "horizontalAdvance") else self.fontMetrics().boundingRect(self.text()).width()
        return QSize(max(94, int(text_width) + 54), 28)

    def paintEvent(self, event) -> None:
        """Paint the switch track, thumb, and label."""
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIASING)

        enabled = self.isEnabled()
        checked = self.isChecked()
        track = QRectF(0.5, 5.0, 36.0, 18.0)
        track_color = "#2563eb" if checked else "#cbd5e1"
        if not enabled:
            track_color = "#e2e8f0"
        painter.setPen(QPen(QColor("#dbe3ef"), 1))
        painter.setBrush(QBrush(QColor(track_color)))
        painter.drawRoundedRect(track, 9.0, 9.0)

        knob_size = 14.0
        knob_x = track.right() - knob_size - 2.0 if checked else track.left() + 2.0
        knob = QRectF(knob_x, track.top() + 2.0, knob_size, knob_size)
        painter.setPen(QPen(QColor(0, 0, 0, 28), 1))
        painter.setBrush(QBrush(QColor("#ffffff" if enabled else "#f8fafc")))
        painter.drawEllipse(knob)

        painter.setPen(QPen(QColor("#1e293b" if enabled else "#94a3b8")))
        font = self.font()
        font.setWeight(QFont.Weight.DemiBold if PYQT_VERSION == 6 else QFont.DemiBold)
        painter.setFont(font)
        text_rect = QRectF(44.0, 0.0, max(0, self.width() - 44.0), self.height())
        painter.drawText(text_rect, _qt_enum("AlignmentFlag", "AlignVCenter"), self.text())


class SegmentedScopeSwitch(QWidget):
    """Two-option segmented switch for trace display scope."""
    valueChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items = (("all", "All ROIs"), ("selected", "Selected ROI"))
        self._value = "all"
        self._highlight_value: Optional[str] = "all"
        self._label_source_index = 0
        self._label_target_index = 0
        self._label_color_progress = 1.0
        self._thumb_position = 0.0
        self._animation_duration_ms = 150.0
        self._animation_start_time = 0.0
        self._animation_start_position = 0.0
        self._animation_target_position = 0.0
        self._pending_emit_value: Optional[str] = None
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._advance_thumb_animation)
        self.setCursor(QCursor(_qt_enum("CursorShape", "PointingHandCursor")))
        self.setMinimumSize(self.sizeHint())
        self.setSizePolicy(SIZE_FIXED, SIZE_FIXED)

    def sizeHint(self) -> QSize:
        """Return the preferred size for the segmented scope switch."""
        return QSize(254, 34)

    def value(self) -> str:
        """Return the currently selected segment value."""
        return self._value

    def setValue(self, value: str, *, emit_signal: bool = False, animated: bool = True) -> None:
        """Set the selected segment value."""
        if value not in {key for key, _ in self._items}:
            return
        target = float(self._index_for_value(value))
        changed = value != self._value
        self._value = value
        if animated and self.isVisible():
            self._start_thumb_animation(target)
        else:
            self._animation_timer.stop()
            self._pending_emit_value = None
            self.thumbPosition = target
            self._highlight_value = value
            self._label_source_index = int(target)
            self._label_target_index = int(target)
            self._label_color_progress = 1.0
        self.update()
        if changed and emit_signal:
            if animated and self.isVisible() and abs(target - self._animation_start_position) >= 1e-6:
                self._pending_emit_value = value
            else:
                self.valueChanged.emit(value)

    def _start_thumb_animation(self, target: float) -> None:
        self._animation_start_position = float(self._thumb_position)
        self._animation_target_position = float(target)
        self._animation_start_time = time.monotonic()
        self._label_source_index = max(0, min(len(self._items) - 1, int(round(self._animation_start_position))))
        self._label_target_index = max(0, min(len(self._items) - 1, int(round(self._animation_target_position))))
        self._label_color_progress = 0.0
        self._highlight_value = None
        if abs(self._animation_target_position - self._animation_start_position) < 1e-6:
            self.thumbPosition = self._animation_target_position
            self._highlight_value = self._value
            self._label_color_progress = 1.0
            self._emit_pending_value_changed()
            return
        self._animation_timer.start()

    def _advance_thumb_animation(self) -> None:
        elapsed_ms = (time.monotonic() - self._animation_start_time) * 1000.0
        progress = max(0.0, min(1.0, elapsed_ms / self._animation_duration_ms))
        eased = self._tanh_ease(progress)
        delta = self._animation_target_position - self._animation_start_position
        self._label_color_progress = eased
        self.thumbPosition = self._animation_start_position + delta * eased
        if progress >= 1.0:
            self._animation_timer.stop()
            self.thumbPosition = self._animation_target_position
            self._highlight_value = self._value
            self._label_color_progress = 1.0
            self.update()
            self._emit_pending_value_changed()

    def _emit_pending_value_changed(self) -> None:
        value = self._pending_emit_value
        self._pending_emit_value = None
        if value is not None and value == self._value:
            self.valueChanged.emit(value)

    @staticmethod
    def _tanh_ease(progress: float) -> float:
        progress = max(0.0, min(1.0, float(progress)))
        curve_strength = 3.0
        low = math.tanh(-curve_strength)
        high = math.tanh(curve_strength)
        value = math.tanh((progress * 2.0 - 1.0) * curve_strength)
        return (value - low) / (high - low)

    def _index_for_value(self, value: str) -> int:
        for index, (key, _) in enumerate(self._items):
            if key == value:
                return index
        return 0

    @staticmethod
    def _mix_color(start: str, end: str, progress: float) -> QColor:
        progress = max(0.0, min(1.0, float(progress)))
        start_color = QColor(start)
        end_color = QColor(end)
        red = round(start_color.red() + (end_color.red() - start_color.red()) * progress)
        green = round(start_color.green() + (end_color.green() - start_color.green()) * progress)
        blue = round(start_color.blue() + (end_color.blue() - start_color.blue()) * progress)
        return QColor(red, green, blue)

    def _label_color(self, index: int, *, enabled: bool) -> QColor:
        if not enabled:
            return QColor("#94a3b8")
        if self._highlight_value is not None:
            return QColor("#ffffff" if index == self._index_for_value(self._highlight_value) else "#1e293b")
        progress = float(self._label_color_progress)
        if index == self._label_source_index:
            return self._mix_color("#ffffff", "#1e293b", progress)
        if index == self._label_target_index:
            return self._mix_color("#1e293b", "#ffffff", progress)
        return QColor("#1e293b")

    def _thumb_rect(self) -> QRectF:
        outer = self._outer_rect()
        segment_width = (outer.width() - 4.0) / 2.0
        x = outer.left() + 2.0 + segment_width * float(self._thumb_position)
        return QRectF(x, outer.top() + 2.0, segment_width, outer.height() - 4.0)

    def _outer_rect(self) -> QRectF:
        return QRectF(0.5, 2.5, max(1.0, self.width() - 1.0), max(1.0, self.height() - 5.0))

    def mousePressEvent(self, event) -> None:
        """Select a segment from a mouse press."""
        outer = self._outer_rect()
        point = _event_pos(event)
        next_value = "selected" if point.x() >= outer.center().x() else "all"
        self.setValue(next_value, emit_signal=True)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        """Handle keyboard navigation between switch segments."""
        key = event.key()
        left_key = _qt_enum("Key", "Key_Left")
        right_key = _qt_enum("Key", "Key_Right")
        space_key = _qt_enum("Key", "Key_Space")
        return_key = _qt_enum("Key", "Key_Return")
        enter_key = _qt_enum("Key", "Key_Enter")
        if key == left_key:
            self.setValue("all", emit_signal=True)
            return
        if key == right_key:
            self.setValue("selected", emit_signal=True)
            return
        if key in (space_key, return_key, enter_key):
            self.setValue("selected" if self._value == "all" else "all", emit_signal=True)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        """Paint the segmented switch and animated thumb."""
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIASING)
        enabled = self.isEnabled()
        outer = self._outer_rect()

        painter.setPen(QPen(QColor("#cbd5e1" if enabled else "#dbe3ef"), 1))
        painter.setBrush(QBrush(QColor("#ffffff" if enabled else "#f8fafc")))
        painter.drawRoundedRect(outer, 8.0, 8.0)

        thumb = self._thumb_rect()
        painter.setPen(QPen(QColor("#2563eb" if enabled else "#cbd5e1"), 1))
        painter.setBrush(QBrush(QColor("#2563eb" if enabled else "#e2e8f0")))
        painter.drawRoundedRect(thumb, 6.5, 6.5)

        font = self.font()
        font.setWeight(QFont.Weight.DemiBold if PYQT_VERSION == 6 else QFont.DemiBold)
        painter.setFont(font)
        segment_width = (outer.width() - 4.0) / 2.0
        for index, (_, text) in enumerate(self._items):
            rect = QRectF(outer.left() + 2.0 + segment_width * index, outer.top() + 2.0, segment_width, outer.height() - 4.0)
            painter.setPen(QPen(self._label_color(index, enabled=enabled)))
            painter.drawText(rect, _qt_enum("AlignmentFlag", "AlignCenter"), text)

    def getThumbPosition(self) -> float:
        """Return the animated thumb position."""
        return float(self._thumb_position)

    def setThumbPosition(self, value: float) -> None:
        """Set the animated thumb position."""
        self._thumb_position = max(0.0, min(1.0, float(value)))
        self.update()

    thumbPosition = pyqtProperty(float, fget=getThumbPosition, fset=setThumbPosition)


def _make_icon_button(svg: str, tooltip: str) -> QPushButton:
    button = QPushButton()
    button.setIcon(_icon_from_svg(svg))
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(28, 26)
    button.setToolTip(tooltip)
    _set_button_role(button, "icon")
    return button


def _apply_app_theme(app: QApplication) -> None:
    app.setStyle(SpinBoxArrowStyle("Fusion"))
    font = QFont("Segoe UI")
    font.setPointSize(9)
    app.setFont(font)
    asset_dir = Path(__file__).resolve().parent / "assets"
    stylesheet = (
        APP_STYLESHEET
        .replace("__SPINBOX_UP_ARROW__", (asset_dir / "chevron-up.svg").as_posix())
        .replace("__SPINBOX_DOWN_ARROW__", (asset_dir / "chevron-down.svg").as_posix())
        .replace("__SPINBOX_UP_ARROW_DISABLED__", (asset_dir / "chevron-up-disabled.svg").as_posix())
        .replace("__SPINBOX_DOWN_ARROW_DISABLED__", (asset_dir / "chevron-down-disabled.svg").as_posix())
        .replace("__COMBOBOX_DOWN_ARROW__", (asset_dir / "chevron-down.svg").as_posix())
        .replace("__COMBOBOX_DOWN_ARROW_DISABLED__", (asset_dir / "chevron-down-disabled.svg").as_posix())
    )
    app.setStyleSheet(stylesheet)
    tooltip_filter = RoundedToolTipFilter(app)
    app.installEventFilter(tooltip_filter)
    app.rounded_tooltip_filter = tooltip_filter


def _event_pos(event):
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


def _event_global_pos(event):
    if hasattr(event, "globalPosition"):
        return event.globalPosition().toPoint()
    return event.globalPos()


def _advanced_defaults(method: str) -> dict[str, Any]:
    return {spec["key"]: spec["default"] for spec in ADVANCED_OPTION_SPECS.get(method, ())}


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _available_device_options() -> list[tuple[str, str]]:
    options = [("CPU", "cpu")]
    if torch.cuda.is_available():
        options.append(("CUDA", "cuda"))
    if _mps_available():
        options.append(("MPS", "mps"))
    return options


def _validate_selected_device(device: str, task: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was selected for {task}, but torch.cuda.is_available() is False.")
    if device == "mps" and not _mps_available():
        raise RuntimeError(f"MPS was selected for {task}, but torch.backends.mps.is_available() is False.")


def _freeze_options(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_options(value[key])) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_options(item) for item in value)
    return value


def _dialog_exec(dialog: QDialog) -> int:
    return dialog.exec() if PYQT_VERSION == 6 else dialog.exec_()


def _get_movie_frame_rate(parent: QWidget, current_value: float) -> tuple[float, bool]:
    dialog = QDialog(parent)
    dialog.setWindowTitle("Movie Frame Rate")
    dialog.setWindowIcon(_icon_from_svg(APP_ICON_SVG))
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 18)
    layout.setSpacing(12)

    label = QLabel("Frame rate (Hz):")
    layout.addWidget(label)
    spin = ModernDoubleSpinBox(dialog)
    spin.setRange(0.001, 1000000.0)
    spin.setDecimals(3)
    spin.setValue(float(current_value))
    spin.setMinimumWidth(260)
    layout.addWidget(spin)

    button_layout = QHBoxLayout()
    button_layout.addStretch(1)
    ok_button = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    _set_button_role(ok_button, "secondary")
    _set_button_role(cancel_button, "secondary")
    ok_button.clicked.connect(dialog.accept)
    cancel_button.clicked.connect(dialog.reject)
    button_layout.addWidget(ok_button)
    button_layout.addWidget(cancel_button)
    layout.addLayout(button_layout)

    spin.setFocus()
    spin.selectAll()
    accepted = QDialog.DialogCode.Accepted if PYQT_VERSION == 6 else QDialog.Accepted
    if _dialog_exec(dialog) == accepted:
        return float(spin.value()), True
    return float(current_value), False


def _format_float_list(values: Any) -> str:
    return ", ".join(f"{float(value):g}" for value in values)


def _parse_float_list(text: str, label: str) -> tuple[float, ...]:
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} must contain at least one number.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated list of numbers.") from exc


class TraceWindow(QDialog):
    """Window that hosts the trace plotting canvas."""
    visibilityChanged = Signal(bool)

    def showEvent(self, event) -> None:
        """Track when the trace window becomes visible."""
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event) -> None:
        """Track when the trace window is hidden."""
        super().hideEvent(event)
        self.visibilityChanged.emit(False)


class SpinBoxArrowStyle(QProxyStyle):
    """Proxy style for drawing custom spinbox arrows."""
    def _draw_spinbox_arrow(self, painter, center: QPointF, *, up: bool, enabled: bool) -> None:
        offset = 2.6
        if up:
            points = [
                QPointF(center.x() - offset, center.y() + 1.4),
                QPointF(center.x(), center.y() - 1.4),
                QPointF(center.x() + offset, center.y() + 1.4),
            ]
        else:
            points = [
                QPointF(center.x() - offset, center.y() - 1.4),
                QPointF(center.x(), center.y() + 1.4),
                QPointF(center.x() + offset, center.y() - 1.4),
            ]

        color = QColor("#64748b" if enabled else "#cbd5e1")
        pen = QPen(color, 1.5)
        pen.setCapStyle(ROUND_CAP)
        pen.setJoinStyle(ROUND_JOIN)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF(points))

    def drawPrimitive(self, element, option, painter, widget=None) -> None:
        """Draw custom spinbox arrow primitives."""
        if element in (SPIN_UP_INDICATOR, SPIN_DOWN_INDICATOR):
            painter.save()
            painter.setRenderHint(ANTIALIASING)
            self._draw_spinbox_arrow(
                painter,
                QPointF(option.rect.center()),
                up=element == SPIN_UP_INDICATOR,
                enabled=bool(option.state & STATE_ENABLED),
            )
            painter.restore()
            return
        super().drawPrimitive(element, option, painter, widget)


class TraceCanvas(FigureCanvas):
    """Matplotlib canvas for ROI trace and spike visualizations."""
    viewChanged = Signal(float, float)
    roiDoubleClicked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self.figure = Figure(figsize=(7, 3.6))
        self.figure.patch.set_facecolor("#ffffff")
        grid = self.figure.add_gridspec(2, 1, height_ratios=[5, 1])
        self.ax = self.figure.add_subplot(grid[0])
        self.overview_ax = self.figure.add_subplot(grid[1])
        self._apply_figure_layout(has_data=False)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setObjectName("traceCanvas")
        self.setMinimumHeight(0)
        self.setSizePolicy(SIZE_EXPANDING, SIZE_IGNORED)
        self.setFocusPolicy(CLICK_FOCUS)
        self._x = np.array([], dtype=float)
        self._display_trace = np.empty((0, 0), dtype=float)
        self._frame_rate = 1.0
        self._view_xlim: Optional[tuple[float, float]] = None
        self._overview_selection: Optional[Rectangle] = None
        self._overview_handles = []
        self._overview_background = None
        self._span_selector: Optional[SpanSelector] = None
        self._pan_start: Optional[tuple[float, tuple[float, float]]] = None
        self._stacked_multi_trace = False
        self._stacked_groups: list[int] = []
        self._stacked_roi_ids: list[int] = []
        self._stacked_row_spacing = 1.0
        self.mpl_connect("draw_event", self._on_draw)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_button_press)
        self.mpl_connect("button_release_event", self._on_button_release)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.plot_empty()

    def plot_empty(self) -> None:
        """Render an empty trace plot state."""
        self._apply_figure_layout(has_data=False)
        self._x = np.array([], dtype=float)
        self._display_trace = np.empty((0, 0), dtype=float)
        self._frame_rate = 1.0
        self._view_xlim = None
        self._pan_start = None
        self._span_selector = None
        self._overview_selection = None
        self._overview_handles = []
        self._overview_background = None
        self._stacked_multi_trace = False
        self._stacked_groups = []
        self._stacked_roi_ids = []
        self._stacked_row_spacing = 1.0
        self.ax.clear()
        self.overview_ax.clear()
        self.ax.set_title("Trace")
        self._draw_time_axis_label()
        self.ax.set_ylabel("Signal")
        self.ax.tick_params(axis="x", labelbottom=True, labelsize=8, pad=1)
        self._style_main_axis()
        self.overview_ax.set_xlabel("")
        self.overview_ax.set_yticks([])
        self._style_overview_axis()
        self.draw_idle()

    def plot_result(
        self,
        result: dict,
        *,
        mode: str = "trace",
        time_window: Optional[tuple[float, float]] = None,
    ) -> None:
        """Plot extraction results and ROI trace data."""
        self._apply_figure_layout(has_data=True)
        self._span_selector = None
        self._overview_selection = None
        self._overview_handles = []
        self._overview_background = None
        self._pan_start = None
        self.ax.clear()
        self.overview_ax.clear()
        self._style_main_axis()
        self._style_overview_axis()
        method = result.get("method", "Trace")
        trace, view_title, y_label, series_labels, stack_groups, roi_ids = self._result_series_for_mode(result, mode)
        if mode == "dff" and trace.size:
            trace = self._normalize_dff_for_display(trace)
            y_label = "Normalized dF/F"
        self._frame_rate = max(1e-12, float(result.get("frame_rate", 1.0) or 1.0))

        self._stacked_multi_trace = (
            bool(result.get("multi_roi"))
            and mode in {"trace", "dff", "reconstruction"}
            and trace.ndim == 2
            and trace.shape[1] > 1
        )
        self._stacked_groups = list(stack_groups or range(trace.shape[1])) if self._stacked_multi_trace else []
        self._stacked_roi_ids = list(roi_ids or []) if self._stacked_multi_trace else []
        display_trace = self._build_display_trace(
            trace,
            stacked=self._stacked_multi_trace,
            stack_groups=self._stacked_groups,
            min_row_spacing=1.0,
        )
        if display_trace.size:
            self._x = np.arange(display_trace.shape[0], dtype=float) / self._frame_rate
            self._display_trace = display_trace
            n_traces = display_trace.shape[1]
            line_width = 0.8 if self._stacked_multi_trace else 1.0
            overview_width = 0.45 if self._stacked_multi_trace else 0.55
            if self._stacked_multi_trace and mode == "dff":
                line_width = 0.45
                overview_width = 0.32
            series_colors = self._series_colors(result, mode, n_traces)

            if self._stacked_multi_trace:
                main_segments = []
                overview_segments = []
                colors = []
                for idx in range(n_traces):
                    y = display_trace[:, idx]
                    x_main, y_main = self._decimate_line_for_plot(
                        self._x,
                        y,
                        max_points=STACKED_TRACE_RENDER_MAX_POINTS,
                    )
                    x_overview, y_overview = self._decimate_line_for_plot(
                        self._x,
                        y,
                        max_points=STACKED_TRACE_OVERVIEW_MAX_POINTS,
                    )
                    main_segments.append(np.column_stack((x_main, y_main)))
                    overview_segments.append(np.column_stack((x_overview, y_overview)))
                    colors.append(series_colors[idx])
                self.ax.add_collection(LineCollection(main_segments, colors=colors, linewidths=line_width))
                self.overview_ax.add_collection(LineCollection(overview_segments, colors=colors, linewidths=overview_width))
            else:
                for idx in range(n_traces):
                    label = series_labels[idx] if idx < len(series_labels) else (f"trace {idx + 1}" if n_traces > 1 else None)
                    self.ax.plot(self._x, display_trace[:, idx], linewidth=line_width, label=label, color=series_colors[idx])
                    self.overview_ax.plot(self._x, display_trace[:, idx], linewidth=overview_width, color=series_colors[idx])

            if 1 < n_traces <= 8 and not self._stacked_multi_trace:
                self.ax.legend(loc="upper right", fontsize=8)

            self.overview_ax.set_xlabel("")
            self.overview_ax.set_yticks([])
            self.overview_ax.set_xlim(float(self._x[0]), float(self._x[-1]) if self._x.size > 1 else 1.0 / self._frame_rate)
            self._style_overview_axis()
            self._span_selector = SpanSelector(
                self.overview_ax,
                self._on_overview_select,
                "horizontal",
                useblit=True,
                props={"facecolor": "#2563eb", "alpha": 0.0},
                onmove_callback=self._on_overview_move,
                handle_props={"color": "#1d4ed8", "linewidth": 0.0, "alpha": 0.0},
                interactive=True,
                drag_from_anywhere=True,
                grab_range=12,
                minspan=1.0 / self._frame_rate,
            )
            if time_window is None:
                view_min = float(self._x[0])
                view_max = float(self._x[-1]) if self._x.size > 1 else 1.0 / self._frame_rate
            else:
                view_min, view_max = time_window
            self._set_view(view_min, view_max, redraw=False)
        else:
            self._x = np.array([], dtype=float)
            self._display_trace = np.empty((0, 0), dtype=float)
            self._view_xlim = None
            self.ax.text(0.5, 0.5, "No trace returned", ha="center", va="center", transform=self.ax.transAxes)
            self.overview_ax.set_xlabel("")
            self.overview_ax.set_yticks([])
            self._style_overview_axis()

        title = view_title or method
        metrics = result.get("metrics") or {}
        if metrics and not result.get("multi_roi"):
            parts = [f"{key}={value}" for key, value in metrics.items()]
            title = f"{title} ({', '.join(parts)})"
        self.ax.set_title(title)
        self._draw_time_axis_label()
        self.ax.tick_params(axis="x", labelbottom=True, labelsize=8, pad=1)
        self.ax.set_ylabel("ROI" if self._stacked_multi_trace else y_label)
        self._style_main_axis()
        if self._stacked_multi_trace:
            self._style_stacked_axes()
        self.draw_idle()

    def _draw_time_axis_label(self) -> None:
        self.ax.set_xlabel("Time (s)", labelpad=2)

    def _apply_figure_layout(self, *, has_data: bool) -> None:
        if has_data:
            self.figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.40)
        else:
            self.figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, hspace=0.68)

    def _build_display_trace(
        self,
        trace: np.ndarray,
        *,
        stacked: bool = False,
        stack_groups: Optional[list[int]] = None,
        min_row_spacing: float = 1.0,
    ) -> np.ndarray:
        if trace.ndim == 1 and trace.size:
            return trace[:, None].astype(float, copy=False)
        if trace.ndim != 2 or not trace.size:
            return np.empty((0, 0), dtype=float)

        if stacked:
            groups = list(stack_groups or range(trace.shape[1]))
            if len(groups) != trace.shape[1]:
                groups = list(range(trace.shape[1]))
            display = np.empty_like(trace, dtype=float)
            finite = trace[np.isfinite(trace)]
            signal_span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 0.0
            self._stacked_row_spacing = max(float(min_row_spacing), 1e-6, signal_span * STACKED_TRACE_ROW_PADDING)
            n_groups = max(groups) + 1 if groups else trace.shape[1]
            for idx in range(trace.shape[1]):
                y = trace[:, idx].astype(float, copy=False)
                display[:, idx] = self._stacked_row_position(groups[idx], n_groups) + y
            return display

        display = np.empty_like(trace, dtype=float)
        offset = 0.0
        for idx in range(trace.shape[1]):
            y = trace[:, idx].astype(float, copy=False)
            display[:, idx] = y + offset
            finite = y[np.isfinite(y)]
            span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 1.0
            offset += max(1.0, span * 1.15)
        return display

    def _normalize_dff_for_display(self, trace: np.ndarray) -> np.ndarray:
        if trace.ndim == 1:
            return self._normalize_dff_column(trace)
        if trace.ndim != 2:
            return trace
        display = np.empty_like(trace, dtype=float)
        for idx in range(trace.shape[1]):
            display[:, idx] = self._normalize_dff_column(trace[:, idx])
        return display

    @staticmethod
    def _normalize_dff_column(y: np.ndarray) -> np.ndarray:
        y = y.astype(float, copy=False)
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return np.zeros_like(y, dtype=float)
        center = float(np.nanmedian(finite))
        if finite.size >= 3:
            low, high = np.nanpercentile(finite, [1.0, 99.0])
            span = float(high - low)
        else:
            span = float(np.nanmax(finite) - np.nanmin(finite))
        if not np.isfinite(span) or span <= 0.0:
            span = float(np.nanmax(finite) - np.nanmin(finite))
        if not np.isfinite(span) or span <= 0.0:
            return np.zeros_like(y, dtype=float)
        scaled = (y - center) * (STACKED_DFF_ROW_HEIGHT / span)
        return np.clip(scaled, -STACKED_DFF_CLIP_HEIGHT, STACKED_DFF_CLIP_HEIGHT)

    @staticmethod
    def _decimate_line_for_plot(x: np.ndarray, y: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        if x.size <= max_points or y.size <= max_points or max_points <= 0:
            return x, y

        bin_count = max(2, int(max_points) // 2)
        bin_size = int(np.ceil(y.size / bin_count))
        padded_len = bin_count * bin_size
        padded = np.full(padded_len, np.nan, dtype=float)
        padded[: y.size] = y
        blocks = padded.reshape(bin_count, bin_size)
        finite = np.isfinite(blocks)
        valid = finite.any(axis=1)
        if not valid.any():
            step = max(1, int(np.ceil(y.size / max_points)))
            return x[::step], y[::step]

        starts = np.arange(bin_count) * bin_size
        min_blocks = np.where(finite, blocks, np.inf)
        max_blocks = np.where(finite, blocks, -np.inf)
        min_indices = starts + np.argmin(min_blocks, axis=1)
        max_indices = starts + np.argmax(max_blocks, axis=1)
        pairs = np.column_stack((min_indices[valid], max_indices[valid]))
        pairs.sort(axis=1)
        indices = np.unique(np.concatenate(([0, y.size - 1], pairs.ravel())))
        indices = indices[indices < y.size]
        return x[indices], y[indices]

    def has_data(self) -> bool:
        """Return whether the canvas currently has plotted trace data."""
        return bool(self._x.size)

    def time_bounds(self) -> tuple[float, float]:
        """Return the time range covered by the current trace data."""
        if self._x.size == 0:
            return 0.0, 0.0
        return float(self._x[0]), float(self._x[-1])

    def current_time_window(self) -> tuple[float, float]:
        """Return the currently visible time window."""
        if self._view_xlim is not None:
            return self._view_xlim
        return self.time_bounds()

    def set_time_window(self, xmin: float, xmax: float) -> None:
        """Set the visible trace time window."""
        self._set_view(float(xmin), float(xmax))

    def reset_view(self) -> None:
        """Reset the trace canvas to the full time range."""
        xmin, xmax = self.time_bounds()
        self._set_view(xmin, xmax)

    def zoom_view(self, factor: float) -> None:
        """Zoom the trace time window by a scale factor."""
        if self._view_xlim is None:
            return
        x0, x1 = self._view_xlim
        center = (x0 + x1) / 2.0
        width = max(1.0 / self._frame_rate, (x1 - x0) * float(factor))
        self._set_view(center - width / 2.0, center + width / 2.0)

    @staticmethod
    def _as_1d_float_array(value: Any) -> np.ndarray:
        arr = np.asarray(value if value is not None else [], dtype=float)
        if arr.ndim == 0:
            return np.empty((0,), dtype=float)
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[1] == 1:
                return arr[:, 0]
        return arr

    def _result_series_for_mode(self, result: dict, mode: str) -> tuple[np.ndarray, str, str, list[str], Optional[list[int]], list[int]]:
        method = str(result.get("method", "Trace"))
        if mode == "dff":
            dff = self._as_1d_float_array(result.get("dff"))
            if dff.size:
                return dff, str(result.get("dff_title") or "dF/F"), "dF/F", list(result.get("dff_labels") or []), None, list(result.get("dff_roi_ids") or [])
            return np.asarray(result.get("trace", []), dtype=float), f"{method} trace", "Signal", [], None, list(result.get("trace_roi_ids") or [])

        if mode == "reconstruction":
            rec = self._as_1d_float_array(result.get("reconstruction"))
            sub = self._as_1d_float_array(result.get("subthreshold"))
            if bool(result.get("multi_roi")):
                rec_labels = list(result.get("reconstruction_labels") or result.get("trace_labels") or [])
                sub_labels = list(result.get("subthreshold_labels") or rec_labels)
                rec_roi_ids = list(result.get("reconstruction_roi_ids") or result.get("trace_roi_ids") or [])
                sub_roi_ids = list(result.get("subthreshold_roi_ids") or rec_roi_ids)
                view_title = str(result.get("reconstruction_title") or "All ROI Reconstruction")
                if rec.ndim == 2 and rec.size and sub.ndim == 2 and sub.size:
                    length = min(rec.shape[0], sub.shape[0])
                    columns = min(rec.shape[1], sub.shape[1])
                    series: list[np.ndarray] = []
                    labels: list[str] = []
                    groups: list[int] = []
                    roi_ids: list[int] = []
                    for idx in range(columns):
                        roi_label = rec_labels[idx] if idx < len(rec_labels) else f"ROI {idx + 1}"
                        series.append(rec[:length, idx])
                        labels.append(f"{roi_label} reconstructed spikes")
                        groups.append(idx)
                        roi_ids.append(int(rec_roi_ids[idx]) if idx < len(rec_roi_ids) else idx + 1)
                        sub_label = sub_labels[idx] if idx < len(sub_labels) else roi_label
                        series.append(sub[:length, idx])
                        labels.append(f"{sub_label} subthreshold")
                        groups.append(idx)
                        roi_ids.append(int(sub_roi_ids[idx]) if idx < len(sub_roi_ids) else roi_ids[-1])
                    if rec.shape[1] > columns:
                        for idx in range(columns, rec.shape[1]):
                            roi_label = rec_labels[idx] if idx < len(rec_labels) else f"ROI {idx + 1}"
                            series.append(rec[:length, idx])
                            labels.append(f"{roi_label} reconstructed spikes")
                            groups.append(idx)
                            roi_ids.append(int(rec_roi_ids[idx]) if idx < len(rec_roi_ids) else idx + 1)
                    if series:
                        return np.column_stack(series), view_title, "Signal", labels, groups, roi_ids
                if rec.size:
                    labels = rec_labels if rec.ndim == 2 else ["reconstructed spikes"]
                    return rec, view_title, "Signal", labels, None, rec_roi_ids
                if sub.size:
                    labels = sub_labels if sub.ndim == 2 else ["subthreshold"]
                    return sub, view_title, "Signal", labels, None, sub_roi_ids
            if rec.size and sub.size:
                length = min(rec.shape[0], sub.shape[0])
                data = np.column_stack((rec[:length], sub[:length]))
                return data, "Reconstruction", "Signal", ["reconstructed spikes", "subthreshold"], None, []
            if rec.size:
                return rec, "Reconstruction", "Signal", ["reconstructed spikes"], None, []
            return np.asarray(result.get("trace", []), dtype=float), f"{method} trace", "Signal", [], None, list(result.get("trace_roi_ids") or [])

        return (
            np.asarray(result.get("trace", []), dtype=float),
            str(result.get("trace_title") or f"{method} trace"),
            "Signal",
            list(result.get("trace_labels") or []),
            None,
            list(result.get("trace_roi_ids") or []),
        )

    def _series_colors(self, result: dict, mode: str, n_traces: int) -> list[str]:
        default = [TRACE_LINE_COLORS[idx % len(TRACE_LINE_COLORS)] for idx in range(n_traces)]
        if n_traces <= 0:
            return []
        if not result.get("multi_roi"):
            color = result.get("trace_color")
            if isinstance(color, str) and color:
                return [color if idx == 0 else default[idx] for idx in range(n_traces)]
            return default

        ids_key = {
            "dff": "dff_roi_ids",
            "reconstruction": "reconstruction_roi_ids",
        }.get(mode, "trace_roi_ids")
        if self._stacked_multi_trace and len(self._stacked_roi_ids) >= n_traces:
            roi_ids = list(self._stacked_roi_ids)
        else:
            roi_ids = list(result.get(ids_key) or result.get("trace_roi_ids") or [])
        if not roi_ids:
            return default
        ordered_roi_ids = []
        for roi_id in roi_ids:
            roi_id = int(roi_id)
            if roi_id not in ordered_roi_ids:
                ordered_roi_ids.append(roi_id)
        color_by_roi = {
            roi_id: TRACE_LINE_COLORS[index % len(TRACE_LINE_COLORS)]
            for index, roi_id in enumerate(ordered_roi_ids)
        }
        return [
            color_by_roi.get(int(roi_ids[idx]), default[idx]) if idx < len(roi_ids) else default[idx]
            for idx in range(n_traces)
        ]

    def _on_overview_select(self, xmin: float, xmax: float) -> None:
        self._set_view(xmin, xmax)

    def _on_overview_move(self, xmin: float, xmax: float) -> None:
        self._move_overview_selection(xmin, xmax)

    def _on_draw(self, event) -> None:
        if event.canvas is not self:
            return
        if self._overview_selection is None:
            self._overview_background = None
            return
        self._overview_background = self.copy_from_bbox(self.overview_ax.bbox)
        self._draw_overview_selection_artists()
        self.blit(self.overview_ax.bbox)

    def _on_scroll(self, event) -> None:
        if event.inaxes not in (self.ax, self.overview_ax) or self._view_xlim is None:
            return

        x0, x1 = self._view_xlim
        center = float(event.xdata) if event.xdata is not None else (x0 + x1) / 2.0
        scale = 0.8 if event.button == "up" else 1.25
        width = max(1.0 / self._frame_rate, (x1 - x0) * scale)
        self._set_view(center - width / 2.0, center + width / 2.0)

    def _on_button_press(self, event) -> None:
        if event.inaxes in (self.ax, self.overview_ax):
            self.setFocus()
        if getattr(event, "dblclick", False):
            self._on_double_click(event)
            return
        if event.inaxes is self.ax and event.button == 1 and event.xdata is not None and self._view_xlim is not None:
            self._pan_start = (float(event.xdata), self._view_xlim)

    def _on_double_click(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1 or event.ydata is None:
            return
        roi_id = self._roi_id_at_stacked_y(float(event.ydata))
        if roi_id is not None:
            self.roiDoubleClicked.emit(int(roi_id))

    def _roi_id_at_stacked_y(self, y: float) -> Optional[int]:
        if not self._stacked_multi_trace or not self._stacked_groups:
            return None
        n_groups = self._stacked_group_count()
        if n_groups <= 0:
            return None
        rows = [(group, self._stacked_row_position(group, n_groups)) for group in range(n_groups)]
        group, row_y = min(rows, key=lambda item: abs(float(y) - item[1]))
        if abs(float(y) - row_y) > max(1e-6, self._stacked_row_spacing * 0.5):
            return None
        for index, stacked_group in enumerate(self._stacked_groups):
            if int(stacked_group) == int(group) and index < len(self._stacked_roi_ids):
                return int(self._stacked_roi_ids[index])
        return group + 1

    def _on_motion(self, event) -> None:
        if self._pan_start is None or event.inaxes is not self.ax or event.xdata is None:
            return
        start_x, (x0, x1) = self._pan_start
        dx = start_x - float(event.xdata)
        self._set_view(x0 + dx, x1 + dx)

    def _on_button_release(self, event) -> None:
        self._pan_start = None

    def _set_view(self, xmin: float, xmax: float, *, redraw: bool = True) -> None:
        if self._x.size == 0:
            return

        data_min = float(self._x[0])
        data_max = float(self._x[-1]) if self._x.size > 1 else data_min + (1.0 / self._frame_rate)
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        min_width = 1.0 / self._frame_rate
        width = max(min_width, float(xmax) - float(xmin))
        data_width = max(min_width, data_max - data_min)
        if width >= data_width:
            xmin, xmax = data_min, data_max
        else:
            xmin = max(data_min, min(float(xmin), data_max - width))
            xmax = xmin + width
        self._view_xlim = (xmin, xmax)
        self.ax.set_xlim(xmin, xmax)
        self._update_main_y_limits(xmin, xmax)
        self._update_overview_selection(xmin, xmax)
        if self._span_selector is not None:
            try:
                self._span_selector.extents = (xmin, xmax)
            except Exception:
                pass
        if redraw:
            self.draw_idle()
        self.viewChanged.emit(float(xmin), float(xmax))

    def _update_main_y_limits(self, xmin: float, xmax: float) -> None:
        if self._display_trace.size == 0:
            return
        if self._stacked_multi_trace:
            self.ax.set_ylim(*self._stacked_y_limits(xmin, xmax))
            return
        start = max(0, int(np.floor(xmin * self._frame_rate)))
        stop = min(self._display_trace.shape[0], int(np.ceil(xmax * self._frame_rate)) + 1)
        visible = self._display_trace[start:stop]
        finite = visible[np.isfinite(visible)]
        if not finite.size:
            return
        ymin = float(np.nanmin(finite))
        ymax = float(np.nanmax(finite))
        pad = max(1e-6, (ymax - ymin) * 0.08)
        self.ax.set_ylim(ymin - pad, ymax + pad)

    def _style_main_axis(self) -> None:
        self.figure.patch.set_facecolor("#ffffff")
        self.ax.set_facecolor("#ffffff")
        self.ax.set_prop_cycle(color=TRACE_LINE_COLORS)
        self.ax.grid(True, color="#e2e8f0", linewidth=0.7)
        self.ax.tick_params(colors="#64748b", labelsize=8)
        self.ax.title.set_color("#0f172a")
        self.ax.xaxis.label.set_color("#475569")
        self.ax.yaxis.label.set_color("#475569")
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            self.ax.spines[side].set_color("#cbd5e1")

    def _style_stacked_axes(self) -> None:
        n_groups = self._stacked_group_count()
        if n_groups <= 0:
            return
        step = max(1, int(np.ceil(n_groups / 8)))
        ticks = list(range(0, n_groups, step))
        if (n_groups - 1) not in ticks:
            ticks.append(n_groups - 1)
        tick_positions = [self._stacked_row_position(tick, n_groups) for tick in ticks]
        self.ax.set_yticks(tick_positions)
        self.ax.set_yticklabels([str(tick + 1) for tick in ticks])
        y_limits = self._stacked_y_limits()
        self.ax.set_ylim(*y_limits)
        self.overview_ax.set_ylim(*y_limits)
        self.ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.7)
        self.ax.grid(False, axis="y")

    def _stacked_group_count(self) -> int:
        if self._stacked_groups:
            return max(self._stacked_groups) + 1
        if self._display_trace.ndim == 2:
            return int(self._display_trace.shape[1])
        return 0

    def _stacked_row_position(self, group: int, n_groups: Optional[int] = None) -> float:
        if n_groups is None:
            n_groups = self._stacked_group_count()
        return max(0, n_groups - 1 - max(0, int(group))) * self._stacked_row_spacing

    def _stacked_y_limits(self, xmin: Optional[float] = None, xmax: Optional[float] = None) -> tuple[float, float]:
        data = self._display_trace
        if data.size and xmin is not None and xmax is not None and self._x.size:
            start = max(0, int(np.floor(float(xmin) * self._frame_rate)))
            stop = min(data.shape[0], int(np.ceil(float(xmax) * self._frame_rate)) + 1)
            data = data[start:stop]

        finite = data[np.isfinite(data)] if data.size else np.empty((0,), dtype=float)
        n_groups = self._stacked_group_count()
        row_positions = [self._stacked_row_position(group, n_groups) for group in range(n_groups)]
        if finite.size:
            ymin = float(np.nanmin(finite))
            ymax = float(np.nanmax(finite))
            if row_positions:
                ymin = min(ymin, min(row_positions))
                ymax = max(ymax, max(row_positions))
        elif row_positions:
            ymin = min(row_positions)
            ymax = max(row_positions)
        else:
            ymin, ymax = -0.5, 0.5

        pad = max(1e-6, self._stacked_row_spacing * 0.08)
        return ymin - pad, ymax + pad

    def save_png(self, path: str, *, dpi: int = 300) -> None:
        """Save the current trace figure as a PNG file."""
        old_size = tuple(self.figure.get_size_inches())
        animated_states = [
            (artist, bool(artist.get_animated()))
            for artist in self._overview_selection_artists()
            if hasattr(artist, "get_animated")
        ]
        try:
            for artist, _ in animated_states:
                artist.set_animated(False)
            if self._stacked_multi_trace and self._display_trace.ndim == 2:
                n_groups = self._stacked_group_count()
                height = max(6.0, min(14.0, 2.2 + n_groups * 0.24))
                self.figure.set_size_inches(11.0, height, forward=False)
            self.figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="#ffffff")
        finally:
            for artist, animated in animated_states:
                artist.set_animated(animated)
            self.figure.set_size_inches(old_size, forward=False)
            self._overview_background = None
            self.draw_idle()

    def _style_overview_axis(self) -> None:
        self.overview_ax.set_facecolor("#f8fafc")
        self.overview_ax.set_prop_cycle(color=TRACE_LINE_COLORS)
        self.overview_ax.tick_params(axis="x", bottom=False, labelbottom=False, colors="#64748b")
        self.overview_ax.tick_params(axis="y", left=False, labelleft=False, colors="#64748b")
        for side in ("top", "right", "left"):
            self.overview_ax.spines[side].set_visible(False)
        self.overview_ax.spines["bottom"].set_color("#cbd5e1")

    def _overview_handle_positions(self, xmin: float, xmax: float) -> list[tuple[float, float]]:
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        data_min, data_max = self.time_bounds()
        data_width = max(1.0 / self._frame_rate, data_max - data_min)
        selection_width = max(1.0 / self._frame_rate, abs(float(xmax) - float(xmin)))
        handle_width = min(
            data_width * 0.012,
            max(data_width * 0.004, 2.0 / self._frame_rate),
            selection_width * 0.45,
        )

        def handle_left(edge: float, *, is_start: bool) -> float:
            if is_start and edge <= data_min + handle_width / 2.0:
                return data_min
            if not is_start and edge >= data_max - handle_width / 2.0:
                return data_max - handle_width
            return edge - handle_width / 2.0

        return [
            (handle_left(float(xmin), is_start=True), handle_width),
            (handle_left(float(xmax), is_start=False), handle_width),
        ]

    def _overview_selection_artists(self) -> list[Any]:
        artists: list[Any] = []
        if self._overview_selection is not None:
            artists.append(self._overview_selection)
        artists.extend(self._overview_handles)
        return artists

    def _draw_overview_selection_artists(self) -> None:
        for artist in self._overview_selection_artists():
            self.overview_ax.draw_artist(artist)

    def _blit_overview_selection(self) -> bool:
        if self._overview_background is None:
            return False
        self.restore_region(self._overview_background)
        self._draw_overview_selection_artists()
        self.blit(self.overview_ax.bbox)
        return True

    def _move_overview_selection(self, xmin: float, xmax: float) -> None:
        if self._overview_selection is None or len(self._overview_handles) < 4:
            self._update_overview_selection(xmin, xmax)
            return
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        self._overview_selection.set_x(float(xmin))
        self._overview_selection.set_width(float(xmax) - float(xmin))
        for index, (left, width) in enumerate(self._overview_handle_positions(xmin, xmax)):
            handle = self._overview_handles[index * 2]
            grip = self._overview_handles[index * 2 + 1]
            handle.set_x(left)
            handle.set_width(width)
            center = left + width / 2.0
            grip.set_xdata([center, center])
        if not self._blit_overview_selection():
            self.draw_idle()

    def _update_overview_selection(self, xmin: float, xmax: float) -> None:
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        self._overview_background = None
        if self._overview_selection is not None:
            self._overview_selection.remove()
        for handle in self._overview_handles:
            handle.remove()
        self._overview_handles = []
        self._overview_selection = Rectangle(
            (xmin, 0.0),
            xmax - xmin,
            1.0,
            transform=self.overview_ax.get_xaxis_transform(),
            facecolor="#2563eb",
            edgecolor="#1d4ed8",
            alpha=0.14,
            linewidth=1.0,
            zorder=3,
        )
        self._overview_selection.set_animated(True)
        self.overview_ax.add_patch(self._overview_selection)
        transform = self.overview_ax.get_xaxis_transform()
        self._overview_handles = []
        for left, handle_width in self._overview_handle_positions(xmin, xmax):
            handle = Rectangle(
                (left, 0.22),
                handle_width,
                0.56,
                transform=transform,
                facecolor="#1d4ed8",
                edgecolor="#1e40af",
                linewidth=0.8,
                alpha=0.98,
                zorder=6,
                clip_on=False,
            )
            handle.set_animated(True)
            self.overview_ax.add_patch(handle)
            grip = self.overview_ax.plot(
                [left + handle_width / 2.0, left + handle_width / 2.0],
                [0.34, 0.66],
                transform=transform,
                color="#bfdbfe",
                linewidth=1.1,
                solid_capstyle="round",
                zorder=7,
                clip_on=False,
            )[0]
            grip.set_animated(True)
            self._overview_handles.extend([handle, grip])


class MovieGraphicsView(QGraphicsView):
    """Graphics view for movie frames, ROI drawing, and mask overlays."""
    roiChanged = Signal(object, object)
    roiPicked = Signal(int)
    roiSelectionCleared = Signal()
    statusChanged = Signal(str)
    zoomChanged = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("movieView")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setBackgroundBrush(QBrush(QColor("#f8fafc")))
        self.setBackgroundBrush(QBrush(QColor("#f8fafc")))
        self.setRenderHint(ANTIALIASING)
        self.setRenderHint(SMOOTH_PIXMAP)
        self.setAlignment(ALIGN_CENTER)
        self.setHorizontalScrollBarPolicy(SCROLLBAR_ALWAYS_OFF)
        self.setVerticalScrollBarPolicy(SCROLLBAR_ALWAYS_OFF)
        self.setFrameShape(QFrame.Shape.NoFrame if PYQT_VERSION == 6 else QFrame.NoFrame)
        self.viewport().setAttribute(WA_STYLED_BACKGROUND, True)

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.scene.addItem(self.image_item)

        self.empty_scene_rect = QRectF(0, 0, 800, 600)
        self.empty_text_item = QGraphicsSimpleTextItem("No movie loaded")
        empty_font = QFont("Segoe UI")
        empty_font.setPointSize(22)
        empty_font.setBold(True)
        self.empty_text_item.setFont(empty_font)
        self.empty_text_item.setBrush(QBrush(QColor("#475569")))
        self.empty_text_item.setZValue(10)
        self.scene.addItem(self.empty_text_item)

        self.mask_item: Optional[QGraphicsPixmapItem] = None
        self.label_items: list[QGraphicsSimpleTextItem] = []
        self.overlay_visible = True
        self.roi_item: Optional[object] = None
        self.roi_mode = "select"
        self.frame_shape: Optional[tuple[int, int]] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_roi_id: int = 1
        self._drag_start: Optional[QPointF] = None
        self._freehand_points: list[QPointF] = []
        self.brush_size = 5.0
        self.fit_mode = "fit"
        self.zoom_percent = 100.0

        self.setMouseTracking(True)
        self.show_empty_message()

    def _set_zoom_percent_value(self, percent: float, *, emit_signal: bool = True) -> None:
        percent = max(MOVIE_ZOOM_MIN_PERCENT, min(MOVIE_ZOOM_MAX_PERCENT, float(percent)))
        changed = not math.isclose(self.zoom_percent, percent, rel_tol=0.0, abs_tol=0.05)
        self.zoom_percent = percent
        if changed and emit_signal:
            self.zoomChanged.emit(self.zoom_percent)

    def set_fit_mode(self, mode: str) -> None:
        """Enable or disable fit-to-view scaling."""
        mode = mode.strip().lower()
        if mode not in {"fit", "fill", "actual"}:
            mode = "fit"
        self.fit_mode = mode
        self.apply_view_transform()

    def set_actual_size(self) -> None:
        """Display movie pixels at actual size."""
        self.fit_mode = "actual"
        self._set_zoom_percent_value(100.0)
        self.apply_view_transform()

    def set_zoom_percent(self, percent: float) -> None:
        """Set the movie view zoom as a percentage."""
        self.fit_mode = "actual"
        self._set_zoom_percent_value(percent)
        self.apply_view_transform()

    def zoom_by(self, factor: float) -> None:
        """Scale the current movie view zoom by a factor."""
        self.set_zoom_percent(self.zoom_percent * float(factor))

    def apply_view_transform(self) -> None:
        """Apply the current fit or zoom transform to the view."""
        rect = self.scene.sceneRect()
        if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return

        self.resetTransform()
        if self.fit_mode == "actual":
            scale = max(0.01, float(self.zoom_percent) / 100.0)
            self.scale(scale, scale)
            self.centerOn(rect.center())
            return

        viewport = self.viewport().rect()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        x_scale = viewport.width() / rect.width()
        y_scale = viewport.height() / rect.height()
        scale = min(x_scale, y_scale) if self.fit_mode == "fit" else max(x_scale, y_scale)
        scale = max(0.01, float(scale))
        self.scale(scale, scale)
        self.centerOn(rect.center())
        self._set_zoom_percent_value(scale * 100.0)

    def set_roi_mode(self, mode: str) -> None:
        """Set the active ROI drawing or selection mode."""
        mode = mode.strip().lower()
        if mode not in {"select", "freehand", "rectangle", "eraser"}:
            mode = "select"
        self.roi_mode = mode
        self._drag_start = None
        self._freehand_points.clear()
        self._remove_roi_item()
        cursor_name = "ArrowCursor" if mode == "select" else "CrossCursor"
        self.setCursor(QCursor(_qt_enum("CursorShape", cursor_name)))

    def set_brush_size(self, size: int) -> None:
        """Set the freehand ROI brush size."""
        self.brush_size = max(1.0, float(size))

    def set_frame(self, frame: np.ndarray, *, rgb: bool = False) -> None:
        """Display a movie frame and optional mask overlay."""
        frame = np.asarray(frame)
        if rgb:
            image = _normalize_rgb_to_uint8(frame)
            h, w = image.shape[:2]
            qimage = QImage(image.data, w, h, image.strides[0], FORMAT_RGB888).copy()
        else:
            if frame.ndim == 3:
                frame = frame.mean(axis=2)
            image = normalize_to_uint8(frame)
            h, w = image.shape
            qimage = QImage(image.data, w, h, image.strides[0], FORMAT_GRAY8).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.empty_text_item.setVisible(False)
        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(0, 0, w, h))
        self.frame_shape = (h, w)
        self.apply_view_transform()

    def show_empty_message(self, message: str = "No movie loaded") -> None:
        """Show a centered empty-state message in the movie view."""
        self._drag_start = None
        self._freehand_points.clear()
        self._remove_roi_item()
        self.clear_mask_overlay()
        self.image_item.setPixmap(QPixmap())
        self.frame_shape = None
        self.scene.setSceneRect(self.empty_scene_rect)
        self.empty_text_item.setText(message)
        self.empty_text_item.setVisible(True)
        self._center_empty_message()
        self.apply_view_transform()

    def clear_roi(self) -> None:
        """Clear the interactive ROI drawing overlay."""
        self._drag_start = None
        self._freehand_points.clear()
        self._remove_roi_item()
        self.clear_mask_overlay()

    def clear_mask_overlay(self) -> None:
        """Remove the displayed ROI mask overlay."""
        if self.mask_item is not None:
            self.scene.removeItem(self.mask_item)
            self.mask_item = None
        for item in self.label_items:
            self.scene.removeItem(item)
        self.label_items.clear()
        self.current_mask = None

    def set_mask_overlay(self, mask: np.ndarray, roi_id: int) -> None:
        """Display a labeled ROI mask overlay."""
        self.clear_mask_overlay()
        mask = np.asarray(mask)
        if mask.ndim != 2 or not np.any(mask > 0):
            return

        self.current_mask = mask
        self.current_roi_id = int(roi_id)
        rgba = self._mask_to_rgba(mask, int(roi_id))
        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], FORMAT_RGBA).copy()
        self.mask_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.mask_item.setZValue(1)
        self.mask_item.setVisible(self.overlay_visible)
        self.scene.addItem(self.mask_item)
        self._add_roi_labels(mask)

    def center_on_roi(self, roi_id: int) -> None:
        """Center the view on a specific ROI label."""
        if self.current_mask is None:
            return
        ys, xs = np.nonzero(self.current_mask == int(roi_id))
        if xs.size == 0:
            return
        self.centerOn(float(np.mean(xs)), float(np.mean(ys)))

    def set_overlay_visible(self, visible: bool) -> None:
        """Show or hide the ROI mask overlay."""
        self.overlay_visible = bool(visible)
        if self.mask_item is not None:
            self.mask_item.setVisible(self.overlay_visible)
        for item in self.label_items:
            item.setVisible(self.overlay_visible)

    @staticmethod
    def _mask_to_rgba(mask: np.ndarray, selected_roi_id: int) -> np.ndarray:
        labels = np.asarray(mask, dtype=np.int64)
        rgba = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
        positive = labels > 0
        if not positive.any():
            return rgba

        ids = labels[positive]
        rgba[..., 0][positive] = ((37 * ids + 67) % 180 + 45).astype(np.uint8)
        rgba[..., 1][positive] = ((73 * ids + 29) % 180 + 45).astype(np.uint8)
        rgba[..., 2][positive] = ((109 * ids + 101) % 180 + 45).astype(np.uint8)
        rgba[..., 3][positive] = 150

        selected = (labels == int(selected_roi_id)) if int(selected_roi_id) > 0 else np.zeros_like(labels, dtype=bool)
        if selected.any():
            rgba[..., 0][selected] = 255
            rgba[..., 1][selected] = 213
            rgba[..., 2][selected] = 79
            rgba[..., 3][selected] = 220
        return rgba

    def _add_roi_labels(self, mask: np.ndarray) -> None:
        font = QFont("Segoe UI")
        font.setPointSize(6)
        font.setBold(True)

        for roi_id in np.unique(mask):
            roi_id = int(roi_id)
            if roi_id <= 0:
                continue
            ys, xs = np.nonzero(mask == roi_id)
            if xs.size == 0:
                continue
            x = float(np.median(xs))
            y = float(np.median(ys))

            shadow = QGraphicsSimpleTextItem(str(roi_id))
            shadow.setFont(font)
            shadow.setBrush(QBrush(QColor(0, 0, 0, 210)))
            shadow.setPos(x + 0.8, y + 0.8)
            shadow.setZValue(3)
            shadow.setVisible(self.overlay_visible)
            self.scene.addItem(shadow)
            self.label_items.append(shadow)

            label = QGraphicsSimpleTextItem(str(roi_id))
            label.setFont(font)
            label.setBrush(QBrush(QColor(255, 255, 255, 245)))
            label.setPos(x, y)
            label.setZValue(4)
            label.setVisible(self.overlay_visible)
            self.scene.addItem(label)
            self.label_items.append(label)

    def _center_empty_message(self) -> None:
        rect = self.scene.sceneRect()
        text_rect = self.empty_text_item.boundingRect()
        x = rect.left() + max(0.0, (rect.width() - text_rect.width()) / 2.0)
        y = rect.top() + max(0.0, (rect.height() - text_rect.height()) / 2.0)
        self.empty_text_item.setPos(x, y)

    def _inside_image(self, point: QPointF) -> bool:
        if self.frame_shape is None:
            return False
        h, w = self.frame_shape
        return 0.0 <= point.x() < w and 0.0 <= point.y() < h

    def _map_event_to_image(self, event) -> Optional[QPointF]:
        point = self.mapToScene(_event_pos(event))
        if not self._inside_image(point):
            return None
        return point

    def _remove_roi_item(self) -> None:
        if self.roi_item is not None:
            self.scene.removeItem(self.roi_item)
            self.roi_item = None

    def _make_pen(self, width: float = 1.8, color: str = "#00bcd4") -> QPen:
        pen = QPen(QColor(color))
        pen.setWidthF(float(width))
        pen.setStyle(SOLID_LINE)
        pen.setCapStyle(ROUND_CAP)
        pen.setJoinStyle(ROUND_JOIN)
        return pen

    def mousePressEvent(self, event) -> None:
        """Handle movie-view mouse presses for ROI tools and panning."""
        point = self._map_event_to_image(event)
        if point is None:
            if event.button() == LEFT_BUTTON and self.roi_mode == "select":
                self.roiSelectionCleared.emit()
                event.accept()
                return
            super().mousePressEvent(event)
            return

        if event.button() == RIGHT_BUTTON:
            self._drag_start = None
            self._freehand_points.clear()
            self._remove_roi_item()
            return

        if event.button() != LEFT_BUTTON:
            super().mousePressEvent(event)
            return

        if self.roi_mode == "select":
            self._pick_roi_at(point)
            return

        if self.roi_mode in {"freehand", "eraser"}:
            self._freehand_points = [point]
            self._remove_roi_item()
            self._update_freehand_preview()
            return

        self._drag_start = point
        self._remove_roi_item()
        self.roi_item = QGraphicsRectItem(QRectF(point, point))
        self.roi_item.setPen(self._make_pen())
        self.roi_item.setBrush(QBrush(NO_BRUSH))
        self.roi_item.setZValue(2)
        self.scene.addItem(self.roi_item)

    def mouseMoveEvent(self, event) -> None:
        """Handle movie-view mouse movement for ROI tools and panning."""
        if self.roi_mode in {"freehand", "eraser"} and self._freehand_points:
            point = self._map_event_to_image(event)
            if point is None:
                return
            last = self._freehand_points[-1]
            if math.hypot(point.x() - last.x(), point.y() - last.y()) >= 0.35:
                self._freehand_points.append(point)
                self._update_freehand_preview()
            return

        if self._drag_start is None or self.roi_mode != "rectangle":
            super().mouseMoveEvent(event)
            return

        point = self._map_event_to_image(event)
        if point is None:
            return
        rect = QRectF(self._drag_start, point).normalized()
        if self.roi_item is not None:
            self.roi_item.setRect(rect)

    def mouseReleaseEvent(self, event) -> None:
        """Complete ROI drawing or panning interactions."""
        if self.roi_mode in {"freehand", "eraser"} and self._freehand_points:
            point = self._map_event_to_image(event)
            if point is not None:
                last = self._freehand_points[-1]
                if math.hypot(point.x() - last.x(), point.y() - last.y()) >= 0.01:
                    self._freehand_points.append(point)
            self._finalize_freehand()
            return

        if self._drag_start is None or self.roi_mode != "rectangle":
            super().mouseReleaseEvent(event)
            return

        point = self._map_event_to_image(event)
        start = self._drag_start
        self._drag_start = None
        if point is None or self.frame_shape is None:
            self._remove_roi_item()
            return

        rect = QRectF(start, point).normalized()
        if rect.width() < 1 or rect.height() < 1:
            self._remove_roi_item()
            return

        if self.roi_item is not None:
            self.roi_item.setRect(rect)

        mask = rectangle_to_mask(
            (rect.left(), rect.top(), rect.right(), rect.bottom()),
            self.frame_shape,
            label=1,
        )
        self._remove_roi_item()
        self.roiChanged.emit(mask, {"type": "rectangle", "rect": rect})

    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double-click actions in the movie view."""
        super().mouseDoubleClickEvent(event)

    def _update_freehand_preview(self) -> None:
        if not self._freehand_points:
            self._remove_roi_item()
            return

        path = QPainterPath(self._freehand_points[0])
        for point in self._freehand_points[1:]:
            path.lineTo(point)

        if self.roi_item is None:
            self.roi_item = QGraphicsPathItem(path)
            color = "#ef4444" if self.roi_mode == "eraser" else "#00bcd4"
            self.roi_item.setPen(self._make_pen(self.brush_size, color))
            self.roi_item.setBrush(QBrush(NO_BRUSH))
            self.roi_item.setZValue(2)
            self.scene.addItem(self.roi_item)
        else:
            self.roi_item.setPath(path)

    def _pick_roi_at(self, point: QPointF) -> None:
        if self.current_mask is None:
            self.roiSelectionCleared.emit()
            self.statusChanged.emit("No ROI mask is loaded")
            return
        x = int(point.x())
        y = int(point.y())
        if y < 0 or y >= self.current_mask.shape[0] or x < 0 or x >= self.current_mask.shape[1]:
            self.roiSelectionCleared.emit()
            return
        roi_id = int(self.current_mask[y, x])
        if roi_id <= 0:
            self.roiSelectionCleared.emit()
            self.statusChanged.emit("No ROI selected")
            return
        self.roiPicked.emit(roi_id)

    def _finalize_freehand(self) -> None:
        if self.frame_shape is None or not self._freehand_points:
            self._freehand_points.clear()
            self._remove_roi_item()
            return

        points = [(p.x(), p.y()) for p in self._freehand_points]
        mask = freehand_to_mask(
            points,
            self.frame_shape,
            radius=max(0.5, float(self.brush_size) / 2.0),
            label=1,
            fill_closed=self.roi_mode == "freehand",
        )
        edit_type = "eraser" if self.roi_mode == "eraser" else "freehand"
        self._freehand_points.clear()
        if not np.any(mask > 0):
            self._remove_roi_item()
            return

        self._remove_roi_item()
        self.roiChanged.emit(mask, {"type": edit_type, "points": points, "brush_size": self.brush_size})

    def resizeEvent(self, event) -> None:
        """Reapply the movie view transform after resizing."""
        super().resizeEvent(event)
        if self.empty_text_item.isVisible():
            self._center_empty_message()
        if not self.scene.sceneRect().isNull():
            self.apply_view_transform()


class PreprocessWorker(QObject):
    """Background worker for movie conversion and motion correction."""
    finished = Signal(str)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        *,
        source_path: str,
        dataset: str,
        motion_correction: bool = True,
        max_shifts: tuple[int, int] = (15, 15),
        frames_per_chunk: int = 256,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.source_path = source_path
        self.dataset = dataset
        self.motion_correction = bool(motion_correction)
        self.max_shifts = max_shifts
        self.frames_per_chunk = int(frames_per_chunk)
        self.device = device

    def _emit_phase_progress(self, phase_start: int, phase_span: int, value: int, message: str) -> None:
        value = max(0, min(100, int(value)))
        scaled = phase_start + int(round(phase_span * value / 100.0))
        self.progress.emit(max(0, min(100, scaled)), message)

    def run(self) -> None:
        """Run TIFF conversion and motion correction in a worker thread."""
        try:
            source = Path(self.source_path)
            suffix = source.suffix.lower()
            if not self.motion_correction:
                self.progress.emit(100, "Motion correction disabled")
                self.finished.emit(str(source))
                return

            _validate_selected_device(self.device, "motion correction")
            self.progress.emit(0, f"Preparing movie on {self.device.upper()}")

            if suffix in {".tif", ".tiff"}:
                corrected_path = default_corrected_h5_path(source)
                if is_motion_corrected_h5(corrected_path, dataset=self.dataset):
                    self.progress.emit(100, "Using existing corrected movie")
                    self.finished.emit(str(corrected_path))
                    return

                self.progress.emit(5, "Using TIFF input")
                movie = None
                try:
                    movie = TiffMovie(source)
                    corrected_path = motion_correct_movie(
                        movie,
                        out_h5_path=corrected_path,
                        dataset=self.dataset,
                        overwrite=False,
                        max_shifts=self.max_shifts,
                        frames_per_chunk=self.frames_per_chunk,
                        device=self.device,
                        progress_callback=lambda value, text: self._emit_phase_progress(5, 95, value, text),
                        source_path=source,
                        source_dataset="",
                        close_movie=True,
                    )
                finally:
                    if movie is not None:
                        movie.close()
                    release_torch_memory(self.device)
                self.progress.emit(100, "Corrected HDF5 ready")
                self.finished.emit(str(corrected_path))
                return
            elif suffix in {".h5", ".hdf5"}:
                h5_path = source
                self.progress.emit(40, "Using HDF5 input")
            else:
                raise ValueError(f"Unsupported movie file type: {suffix}")

            if is_motion_corrected_h5(h5_path, dataset=self.dataset):
                self.progress.emit(100, "Movie is already motion corrected")
                self.finished.emit(str(h5_path))
                return

            self.progress.emit(45, "Running motion correction")
            try:
                corrected_path = motion_correct_h5(
                    h5_path,
                    dataset=self.dataset,
                    overwrite=False,
                    max_shifts=self.max_shifts,
                    frames_per_chunk=self.frames_per_chunk,
                    device=self.device,
                    progress_callback=lambda value, text: self._emit_phase_progress(45, 55, value, text),
                )
            finally:
                release_torch_memory(self.device)
            self.progress.emit(100, "Corrected HDF5 ready")
            self.finished.emit(str(corrected_path))
        except Exception:
            self.failed.emit(traceback.format_exc())


class ExtractionWorker(QObject):
    """Background worker for ROI trace extraction."""
    finished = Signal(object)
    failed = Signal(str)
    status = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        *,
        movie_path: str,
        dataset: str,
        method: str,
        roi_mask: np.ndarray,
        roi_ids: list[int],
        channel: Optional[int],
        frame_rate: float,
        device: str,
        flip_signal: bool,
        batch_size: int = 256,
        advanced_options: Optional[dict[str, Any]] = None,
        all_rois: bool = False,
    ) -> None:
        super().__init__()
        self.movie_path = movie_path
        self.dataset = dataset
        self.method = method
        self.roi_mask = roi_mask.astype(np.int32, copy=False)
        self.roi_ids = [int(roi_id) for roi_id in roi_ids]
        self.channel = channel
        self.frame_rate = float(frame_rate)
        self.device = device
        self.flip_signal = bool(flip_signal)
        self.batch_size = int(batch_size)
        self.advanced_options = dict(advanced_options or {})
        self.all_rois = bool(all_rois)

    def run(self) -> None:
        """Run trace extraction in a worker thread."""
        movie = None
        try:
            _validate_selected_device(self.device, "extraction")

            movie = open_movie(self.movie_path, dataset=self.dataset)
            method = self.method.lower()
            total = max(1, len(self.roi_ids))
            results = []

            if method == "spikepursuit" and len(self.roi_ids) > 1:
                self.status.emit(f"Running batched Spikepursuit for {len(self.roi_ids)} ROI(s)")
                self.progress.emit(0, "Preparing batched Spikepursuit")
                for index, result in enumerate(
                    iter_spikepursuit_results(
                        movie,
                        self.roi_mask,
                        self.roi_ids,
                        frame_rate=self.frame_rate,
                        channel=self.channel,
                        device=self.device,
                        flip_signal=self.flip_signal,
                        **self.advanced_options,
                    )
                ):
                    roi_id = int(result.roi_id)
                    results.append(self._spikepursuit_payload(result))
                    del result
                    release_torch_memory(self.device)
                    self.progress.emit(
                        int(round((index + 1) * 100 / total)),
                        f"Finished ROI {roi_id}",
                    )
            elif method == "mean roi" and len(self.roi_ids) > 1:
                batch_size = int(self.advanced_options.get("batch_size", self.batch_size))
                self.status.emit(f"Extracting batched mean ROI traces for {len(self.roi_ids)} ROI(s)")
                self.progress.emit(0, "Preparing batched mean ROI traces")
                traces = extract_mean_traces(
                    movie,
                    self.roi_mask,
                    roi_ids=self.roi_ids,
                    channel=self.channel,
                    batch_size=batch_size,
                    device=self.device,
                )
                for index, roi_id in enumerate(self.roi_ids):
                    trace = traces[int(roi_id)]
                    results.append(
                        {
                            "method": "Mean ROI",
                            "trace": trace.detach().cpu().numpy(),
                            "spikes": np.array([], dtype=np.int64),
                            "metrics": {"roi": int(roi_id)},
                            "frame_rate": self.frame_rate,
                        }
                    )
                    self.progress.emit(
                        int(round((index + 1) * 100 / total)),
                        f"Finished ROI {roi_id}",
                    )
                del traces
                release_torch_memory(self.device)
            else:
                for index, roi_id in enumerate(self.roi_ids):
                    prefix = f"ROI {roi_id}"
                    self.progress.emit(int(round(index * 100 / total)), f"Extracting {prefix}")
                    results.append(self._extract_one(movie, method, roi_id, prefix))
                    release_torch_memory(self.device)
                    self.progress.emit(int(round((index + 1) * 100 / total)), f"Finished {prefix}")

            result_order = {int(roi_id): index for index, roi_id in enumerate(self.roi_ids)}
            results.sort(key=lambda payload: result_order.get(int((payload.get("metrics") or {}).get("roi", -1)), len(result_order)))
            self.finished.emit({"results": results, "all_rois": self.all_rois})
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if movie is not None and hasattr(movie, "close"):
                movie.close()
            release_torch_memory(self.device)

    def _extract_one(self, movie, method: str, roi_id: int, prefix: str) -> dict:
        if method == "mean roi":
            self.status.emit(f"Extracting mean ROI trace for {prefix}")
            batch_size = int(self.advanced_options.get("batch_size", self.batch_size))
            trace = extract_mean_trace(
                movie,
                self.roi_mask,
                roi_id=roi_id,
                channel=self.channel,
                batch_size=batch_size,
                device=self.device,
            )
            payload = {
                "method": "Mean ROI",
                "trace": trace.detach().cpu().numpy(),
                "spikes": np.array([], dtype=np.int64),
                "metrics": {"roi": roi_id},
            }
        elif method == "spikepursuit":
            self.status.emit(f"Running Spikepursuit for {prefix}")
            result = run_spikepursuit(
                movie,
                self.roi_mask,
                roi_id,
                frame_rate=self.frame_rate,
                channel=self.channel,
                device=self.device,
                flip_signal=self.flip_signal,
                **self.advanced_options,
            )
            payload = self._spikepursuit_payload(result)
        elif method == "ali":
            self.status.emit(f"Running ALI for {prefix}")
            result, bbox = run_ali(
                movie,
                self.roi_mask,
                roi_id,
                frame_rate=self.frame_rate,
                channel=self.channel,
                device=self.device,
                **self.advanced_options,
            )
            traces = result.traces.detach().cpu().numpy()
            payload = {
                "method": "ALI",
                "trace": traces,
                "spikes": result.spk[:, 2].detach().cpu().numpy() if result.spk.numel() else np.array([], dtype=np.int64),
                "bbox": bbox,
                "metrics": {
                    "roi": roi_id,
                    "clusters": traces.shape[1] if traces.ndim == 2 else 0,
                },
            }
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")

        payload["frame_rate"] = self.frame_rate
        return payload

    def _spikepursuit_payload(self, result) -> dict:
        payload = {
            "method": "Spikepursuit",
            "trace": result.t.detach().cpu().numpy(),
            "matched_trace": result.ts.detach().cpu().numpy(),
            "reconstruction": result.t_rec.detach().cpu().numpy(),
            "subthreshold": result.t_sub.detach().cpu().numpy(),
            "dff": result.dFF.detach().cpu().numpy(),
            "spikes": result.spikes.detach().cpu().numpy(),
            "templates": result.templates.detach().cpu().numpy(),
            "weights": result.weights.detach().cpu().numpy(),
            "mean_im": result.mean_im.detach().cpu().numpy(),
            "metrics": {
                "roi": int(result.roi_id),
                "snr": f"{result.snr:.2f}",
                "spikes": int(result.spikes.numel()),
                "threshold": f"{float(result.thresh):.3g}",
                "locality": bool(result.locality),
            },
            "frame_rate": self.frame_rate,
        }
        return payload


class SegmentationWorker(QObject):
    """Background worker for Cellpose ROI segmentation."""
    finished = Signal(object)
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        *,
        movie_path: str,
        dataset: str,
        model_path: str,
        channel: Optional[int],
        summary_window_size: int,
        baseline_percentile: float,
        device: str,
        gpu: bool,
        save_to_disk: bool,
    ) -> None:
        super().__init__()
        self.movie_path = movie_path
        self.dataset = dataset
        self.model_path = model_path
        self.channel = channel
        self.summary_window_size = int(summary_window_size)
        self.baseline_percentile = float(baseline_percentile)
        self.device = device
        self.gpu = bool(gpu)
        self.save_to_disk = bool(save_to_disk)

    def run(self) -> None:
        """Run Cellpose segmentation in a worker thread."""
        movie = None
        try:
            _validate_selected_device(self.device, "summary building")
            if not self.model_path:
                raise ValueError("Select a Cellpose model path first.")

            self.status.emit("Opening movie for summary")
            movie = open_movie(self.movie_path, dataset=self.dataset)
            save_dir = Path(self.movie_path).parent / "cellpose" if self.save_to_disk else None

            self.status.emit("Building summary image")
            mask, summary_img = build_cellpose_rois(
                movie,
                model_path=self.model_path,
                channel=self.channel,
                summary_window_size=self.summary_window_size,
                baseline_percentile=self.baseline_percentile,
                device=self.device,
                gpu=self.gpu,
                save_to_disk=self.save_to_disk,
                save_dir=save_dir,
            )
            ids = available_roi_ids(mask)
            if ids.size == 0:
                raise RuntimeError("Cellpose returned no positive ROI labels.")

            self.finished.emit(
                {
                    "mask": mask,
                    "ids": ids,
                    "summary_shape": tuple(summary_img.shape),
                    "save_dir": str(save_dir) if save_dir is not None else None,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if movie is not None and hasattr(movie, "close"):
                movie.close()
            release_torch_memory(self.device)


class MainWindow(QMainWindow):
    """Main torch-volpy GUI window."""
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TorchVolpy Studio")
        self.setWindowIcon(_icon_from_svg(APP_ICON_SVG))
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self.movie = None
        self.movie_path: Optional[str] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.selected_roi_id: Optional[int] = None
        self.last_result: Optional[dict] = None
        self.last_batch_results: list[dict] = []
        self.combined_trace_result: Optional[dict] = None
        self.combined_trace_result_key: tuple[int, ...] = ()
        self.trace_cache: dict[tuple, dict] = {}
        self.advanced_options_by_method = {
            method: _advanced_defaults(method) for method in ADVANCED_OPTION_SPECS
        }
        self.pending_extraction_keys: dict[int, tuple] = {}
        self.pending_extraction_roi_ids: list[int] = []
        self.roi_mask_version = 0
        self.roi_mask_history: list[tuple[Optional[np.ndarray], Optional[int]]] = []
        self.trace_view_mode = "trace"
        self.trace_display_scope = "all"
        self.inspector_view_mode = "trace"
        self.preprocess_thread: Optional[QThread] = None
        self.preprocess_worker: Optional[PreprocessWorker] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[ExtractionWorker] = None
        self.segmentation_thread: Optional[QThread] = None
        self.segmentation_worker: Optional[SegmentationWorker] = None

        self.timer = QTimer(self)
        self.playback_timer_ms = 16
        self._playback_last_time: Optional[float] = None
        self._playback_frame_credit = 0.0
        self.timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self.roi_undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.roi_clear_tools_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._resize_to_available_screen()
        self._connect_signals()
        self._set_controls_enabled(False)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())

        workspace = QWidget()
        workspace.setObjectName("workspaceBody")
        main_layout = QHBoxLayout(workspace)
        main_layout.setContentsMargins(14, 10, 14, 8)
        main_layout.setSpacing(10)

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setObjectName("controlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setFixedWidth(354)
        scroll.setMinimumHeight(0)
        main_layout.addWidget(scroll)

        main_layout.addWidget(self._build_center_workspace(), 1)
        main_layout.addWidget(self._build_inspector_panel())
        root_layout.addWidget(workspace, 1)
        root_layout.addWidget(self._build_footer_bar())

        self.setCentralWidget(root)
        self._build_trace_window()
        self._refresh_roi_inspector()

    def _build_top_bar(self) -> QWidget:
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(16, 8, 14, 8)
        layout.setSpacing(12)

        icon_label = QLabel()
        icon_pixmap = QPixmap()
        icon_pixmap.loadFromData(APP_ICON_SVG.encode("utf-8"), "SVG")
        icon_label.setPixmap(icon_pixmap.scaled(28, 28, KEEP_ASPECT, SMOOTH_TRANSFORM))
        icon_label.setFixedSize(30, 30)
        layout.addWidget(icon_label)

        title = QLabel("TorchVolpy Studio")
        title.setObjectName("topAppTitle")
        layout.addWidget(title)

        version = QLabel("v0.1.0")
        version.setObjectName("versionBadge")
        layout.addWidget(version)

        layout.addStretch(1)

        return top_bar

    def _build_center_workspace(self) -> QWidget:
        center = QWidget()
        center.setObjectName("centerWorkspace")
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_movie_view_card(), 1)
        return center

    def _build_movie_view_card(self) -> QFrame:
        card = MovieCardFrame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(1, 1, 1, 1)
        card_layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setObjectName("viewerToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        toolbar_layout.setSpacing(6)
        self.zoom_out_button = _make_icon_button(ZOOM_OUT_SVG.format(color="#334155"), "Zoom out")
        self.zoom_100_button = _make_toolbar_button("100%")
        self.zoom_100_button.setMinimumWidth(58)
        self.zoom_in_button = _make_icon_button(ZOOM_IN_SVG.format(color="#334155"), "Zoom in")
        toolbar_layout.addWidget(self.zoom_out_button)
        toolbar_layout.addWidget(self.zoom_100_button)
        toolbar_layout.addWidget(self.zoom_in_button)

        self.fit_mode_combo = ModernComboBox()
        self.fit_mode_combo.addItems(["Fit", "Fill", "Actual"])
        self.fit_mode_combo.setFixedWidth(108)
        toolbar_layout.addWidget(self.fit_mode_combo)

        self.display_mode_combo = ModernComboBox()
        self.display_mode_combo.addItems(["Greyscale", "RGB"])
        self.display_mode_combo.setFixedWidth(136)
        toolbar_layout.addWidget(self.display_mode_combo)

        self.labels_toggle_button = SwitchButton("Labels")
        self.labels_toggle_button.setChecked(True)
        toolbar_layout.addWidget(self.labels_toggle_button)
        toolbar_layout.addStretch(1)

        self.show_trace_button = _make_toolbar_button("Trace Window", TRACE_SIGNAL_SVG)
        toolbar_layout.addWidget(self.show_trace_button)
        self.reset_view_button = _make_toolbar_button("Reset View", RESET_VIEW_SVG)
        toolbar_layout.addWidget(self.reset_view_button)
        card_layout.addWidget(toolbar)

        self.movie_view = MovieGraphicsView()
        self.movie_view.setMinimumHeight(MOVIE_VIEW_MIN_HEIGHT)
        self.movie_view.setSizePolicy(SIZE_EXPANDING, SIZE_IGNORED)
        movie_stage = MovieStage()
        self.movie_stage = movie_stage
        movie_stage.setObjectName("movieStage")
        movie_stage_layout = QGridLayout(movie_stage)
        movie_stage_layout.setContentsMargins(0, 0, 0, 0)
        movie_stage_layout.setSpacing(0)
        movie_stage_layout.addWidget(self.movie_view, 0, 0)
        self.roi_tool_island = self._build_roi_tool_island(movie_stage)
        self.roi_tool_island.raise_()
        movie_stage.resized.connect(self.roi_tool_island.reposition_to_dock)
        QTimer.singleShot(0, self.roi_tool_island.reposition_to_dock)
        card_layout.addWidget(movie_stage, 1)
        card.raise_border()
        return card

    def _build_roi_tool_island(self, parent: QWidget) -> RoiToolIsland:
        island = RoiToolIsland(parent)
        self.roi_tool_buttons: dict[str, QPushButton] = {}
        for mode, icon, tooltip in (
            ("freehand", BRUSH_SVG, "Draw ROI with brush"),
            ("rectangle", RECTANGLE_ROI_SVG, "Draw rectangular ROI"),
            ("eraser", ERASER_SVG, "Erase touched ROI"),
        ):
            button = _make_roi_tool_button(icon, tooltip)
            self.roi_tool_buttons[mode] = button
            island.add_tool_widget(button)

        self.roi_brush_size_spin = RoiBrushSizeSpinBox()
        self.roi_brush_size_spin.setRange(1, 64)
        self.roi_brush_size_spin.setValue(5)
        self.roi_brush_size_spin.setSuffix(" px")
        self.roi_brush_size_spin.setFixedWidth(70)
        self.roi_brush_size_spin.setToolTip("Brush size")
        _show_spinbox_buttons(self.roi_brush_size_spin)
        self.roi_brush_size_spin.set_vertical_display(island.dock_side in {"left", "right"})
        island.add_tool_widget(self.roi_brush_size_spin)

        self.roi_undo_button = _make_roi_tool_button(UNDO_SVG, "Undo ROI edit (Ctrl+Z)", checkable=False)
        self.roi_undo_button.setEnabled(False)
        island.add_tool_widget(self.roi_undo_button)
        island.reposition_to_dock()
        return island

    def _build_trace_window(self) -> TraceWindow:
        self.trace_canvas = TraceCanvas()
        self.trace_window = TraceWindow(self)
        self.trace_window.setObjectName("traceWindow")
        self.trace_window.setWindowTitle("Trace Window")
        self.trace_window.setWindowIcon(_icon_from_svg(APP_ICON_SVG))
        self.trace_window.setModal(False)
        self.trace_window.setMinimumSize(760, 420)
        self.trace_window.resize(1120, 620)

        trace_layout = QVBoxLayout(self.trace_window)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.setSpacing(0)

        self.trace_content = QWidget()
        self.trace_content.setObjectName("traceContent")
        content_layout = QVBoxLayout(self.trace_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        tabs = QWidget()
        tabs.setObjectName("traceTabs")
        tabs_layout = QHBoxLayout(tabs)
        tabs_layout.setContentsMargins(12, 0, 10, 0)
        tabs_layout.setSpacing(10)
        self.trace_mode_buttons: dict[str, QPushButton] = {}
        for mode, text in (
            ("trace", "Voltage Trace"),
            ("dff", "dF/F"),
            ("reconstruction", "Reconstruction"),
        ):
            button = _make_tab_button(text, mode == self.trace_view_mode)
            self.trace_mode_buttons[mode] = button
            tabs_layout.addWidget(button)
        tabs_layout.addStretch(1)
        self.trace_scope_switch = SegmentedScopeSwitch()
        self.trace_scope_switch.setValue(self.trace_display_scope, animated=False)
        tabs_layout.addWidget(self.trace_scope_switch)
        content_layout.addWidget(tabs)

        plot_area = QWidget()
        plot_area.setObjectName("tracePlotArea")
        plot_layout = QGridLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.trace_canvas, 0, 0)
        content_layout.addWidget(plot_area, 1)

        trace_controls = QWidget()
        trace_controls.setObjectName("traceControls")
        trace_controls_layout = QHBoxLayout(trace_controls)
        trace_controls_layout.setContentsMargins(12, 8, 12, 8)
        trace_controls_layout.setSpacing(8)
        trace_controls_layout.addWidget(QLabel("Time range (s)"))
        self.trace_start_spin = ModernDoubleSpinBox()
        self.trace_start_spin.setDecimals(3)
        self.trace_start_spin.setRange(0.0, 0.0)
        self.trace_start_spin.setFixedWidth(92)
        _show_spinbox_buttons(self.trace_start_spin)
        self.trace_end_spin = ModernDoubleSpinBox()
        self.trace_end_spin.setDecimals(3)
        self.trace_end_spin.setRange(0.0, 0.0)
        self.trace_end_spin.setFixedWidth(92)
        _show_spinbox_buttons(self.trace_end_spin)
        trace_controls_layout.addWidget(self.trace_start_spin)
        trace_controls_layout.addWidget(QLabel("to"))
        trace_controls_layout.addWidget(self.trace_end_spin)
        self.trace_reset_button = _make_toolbar_button("Reset", RESET_VIEW_SVG)
        self.trace_save_png_button = _make_toolbar_button("Save PNG", DOWNLOAD_SVG)
        trace_controls_layout.addWidget(self.trace_reset_button)
        trace_controls_layout.addStretch(1)
        trace_controls_layout.addWidget(self.trace_save_png_button)
        content_layout.addWidget(trace_controls)

        trace_layout.addWidget(self.trace_content, 1)
        self.trace_content.show()
        for widget in (
            self.trace_start_spin,
            self.trace_end_spin,
            self.trace_reset_button,
            self.trace_save_png_button,
        ):
            widget.setEnabled(False)
        return self.trace_window

    def _build_inspector_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("inspectorPanel")
        panel.setFixedWidth(358)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        selected_card = _styled_frame("inspectorCard")
        selected_card.setSizePolicy(SIZE_EXPANDING, SIZE_EXPANDING)
        selected_layout = QVBoxLayout(selected_card)
        selected_layout.setContentsMargins(14, 12, 14, 14)
        selected_layout.setSpacing(11)

        selected_header = QHBoxLayout()
        selected_header.addWidget(_role_label("SELECTED ROI", "eyebrow"))
        selected_header.addStretch(1)
        self.roi_count_label = _role_label("0 / 0", "pill")
        selected_header.addWidget(self.roi_count_label)
        selected_layout.addLayout(selected_header)

        roi_title_row = QHBoxLayout()
        dot = _role_label("", "warningDot")
        dot.setFixedSize(14, 14)
        self.roi_title_label = _role_label("No ROI selected", "hero")
        roi_title_row.addWidget(dot)
        roi_title_row.addWidget(self.roi_title_label, 1)
        selected_layout.addLayout(roi_title_row)

        method_layout = QVBoxLayout()
        method_layout.setSpacing(2)
        method_layout.addWidget(_role_label("Method", "metricLabel"))
        self.roi_method_value_label = QLabel("--")
        self.roi_method_value_label.setStyleSheet("font-weight: 700;")
        method_layout.addWidget(self.roi_method_value_label)
        selected_layout.addLayout(method_layout)

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(8)
        snr_tile, self.roi_snr_value = _make_metric_tile("SNR")
        spikes_tile, self.roi_spikes_value = _make_metric_tile("Spikes")
        threshold_tile, self.roi_threshold_value = _make_metric_tile("Threshold")
        locality_tile, self.roi_locality_value = _make_metric_tile("Locality", role="successValue")
        metric_grid.addWidget(snr_tile, 0, 0)
        metric_grid.addWidget(spikes_tile, 0, 1)
        metric_grid.addWidget(threshold_tile, 1, 0)
        metric_grid.addWidget(locality_tile, 1, 1)
        selected_layout.addLayout(metric_grid)

        tabs = QWidget()
        tabs.setObjectName("inspectorTabs")
        tabs_layout = QHBoxLayout(tabs)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(4)
        self.inspector_mode_buttons: dict[str, QPushButton] = {}
        for mode, text in (
            ("trace", "Trace"),
            ("dff", "dF/F"),
            ("weights", "Weights"),
        ):
            button = _make_tab_button(text, mode == self.inspector_view_mode)
            self.inspector_mode_buttons[mode] = button
            tabs_layout.addWidget(button)
        tabs_layout.addStretch(1)
        selected_layout.addWidget(tabs)

        self.roi_stat_name_labels: list[QLabel] = []
        self.roi_stat_labels: list[QLabel] = []
        stats_grid = QGridLayout()
        stats_grid.setColumnStretch(0, 1)
        stats_grid.setColumnStretch(1, 0)
        stats_grid.setHorizontalSpacing(18)
        stats_grid.setVerticalSpacing(10)
        for row, key in enumerate(("Duration", "Sampling rate", "First spike", "Last spike", "Peak", "Status")):
            label = _role_label(key, "metricLabel")
            value = QLabel("--")
            value.setAlignment(ALIGN_RIGHT)
            value.setMinimumWidth(112)
            stats_grid.setRowMinimumHeight(row, 22)
            self.roi_stat_name_labels.append(label)
            self.roi_stat_labels.append(value)
            stats_grid.addWidget(label, row, 0)
            stats_grid.addWidget(value, row, 1)
        selected_layout.addLayout(stats_grid)
        layout.addWidget(selected_card, 1)
        layout.addStretch(1)
        return panel

    def _build_footer_bar(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("footerBar")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(0)

        dot = QLabel()
        dot.setObjectName("readyDot")
        dot.setFixedSize(10, 10)
        layout.addWidget(dot)
        ready = QLabel("  Ready")
        ready.setObjectName("footerMetric")
        ready.setStyleSheet("border-left: none; padding-left: 6px;")
        layout.addWidget(ready)

        self.footer_image_label = QLabel("Image: --")
        self.footer_image_label.setObjectName("footerMetric")
        self.footer_rois_label = QLabel("0 ROIs")
        self.footer_rois_label.setObjectName("footerMetric")
        self.footer_sampling_label = QLabel("dt: --")
        self.footer_sampling_label.setObjectName("footerMetric")
        layout.addWidget(self.footer_image_label)
        layout.addWidget(self.footer_rois_label)
        layout.addWidget(self.footer_sampling_label)
        layout.addStretch(1)

        self.footer_result_label = QLabel("Last extraction: --")
        self.footer_result_label.setObjectName("footerMetric")
        self.footer_method_label = QLabel("--")
        self.footer_method_label.setObjectName("footerMetric")
        self.footer_spikes_label = QLabel("-- spikes")
        self.footer_spikes_label.setObjectName("footerMetric")
        self.footer_snr_label = QLabel("SNR --")
        self.footer_snr_label.setObjectName("footerMetric")
        layout.addWidget(self.footer_result_label)
        layout.addWidget(self.footer_method_label)
        layout.addWidget(self.footer_spikes_label)
        layout.addWidget(self.footer_snr_label)
        return footer

    def _resize_to_available_screen(self) -> None:
        width = DEFAULT_WINDOW_WIDTH
        height = DEFAULT_WINDOW_HEIGHT
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            min_width = min(MIN_WINDOW_WIDTH, max(1, int(available.width())))
            min_height = min(MIN_WINDOW_HEIGHT, max(1, int(available.height())))
            self.setMinimumSize(min_width, min_height)
            width = min(width, max(min_width, int(available.width() * INITIAL_SCREEN_WIDTH_FRACTION)))
            height = min(height, max(min_height, int(available.height() * INITIAL_SCREEN_HEIGHT_FRACTION)))
        self.resize(width, height)

    def _reset_movie_view(self) -> None:
        fit_index = self.fit_mode_combo.findText("Fit")
        if fit_index >= 0 and self.fit_mode_combo.currentIndex() != fit_index:
            self.fit_mode_combo.setCurrentIndex(fit_index)
        self.movie_view.set_fit_mode("Fit")
        self._refresh_zoom_label()

    def _set_movie_fit_mode(self, mode: str) -> None:
        if mode.strip().lower() == "actual":
            self.movie_view.set_actual_size()
        else:
            self.movie_view.set_fit_mode(mode)
        self._refresh_zoom_label()

    def _set_movie_actual_size(self) -> None:
        index = self.fit_mode_combo.findText("Actual")
        if index >= 0 and self.fit_mode_combo.currentIndex() != index:
            self.fit_mode_combo.setCurrentIndex(index)
        self.movie_view.set_actual_size()
        self._refresh_zoom_label()

    def _zoom_movie_view(self, factor: float) -> None:
        index = self.fit_mode_combo.findText("Actual")
        if index >= 0 and self.fit_mode_combo.currentIndex() != index:
            self.fit_mode_combo.blockSignals(True)
            self.fit_mode_combo.setCurrentIndex(index)
            self.fit_mode_combo.blockSignals(False)
        self.movie_view.zoom_by(factor)
        self._refresh_zoom_label()

    def _refresh_zoom_label(self, zoom_percent: Optional[float] = None) -> None:
        if hasattr(self, "zoom_100_button"):
            self.zoom_100_button.setText(f"{int(round(self.movie_view.zoom_percent))}%")

    def _set_roi_tool_mode(self, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in {"select", "freehand", "rectangle", "eraser"}:
            mode = "select"
        self.movie_view.set_roi_mode(mode)
        for tool_mode, button in self.roi_tool_buttons.items():
            button.blockSignals(True)
            button.setChecked(tool_mode == mode)
            button.blockSignals(False)

    def _set_movie_display_mode(self, mode: str) -> None:
        if self.movie is not None:
            self.show_frame(self.frame_slider.value())

    def _show_trace_window(self) -> None:
        if not self.trace_window.isVisible():
            self.trace_window.show()
        self.trace_window.raise_()
        self.trace_window.activateWindow()
        self._update_trace_window_button(True)

    def _hide_trace_window(self) -> None:
        self.trace_window.hide()
        self._update_trace_window_button(False)
        self.status_label.setText("Trace window hidden")

    def _toggle_trace_window(self) -> None:
        if self.trace_window.isVisible():
            self._hide_trace_window()
        else:
            self._show_trace_window()

    def _update_trace_window_button(self, visible: Optional[bool] = None) -> None:
        if not hasattr(self, "show_trace_button"):
            return
        if visible is None:
            visible = self.trace_window.isVisible()
        self.show_trace_button.setText("Hide Trace" if visible else "Trace Window")

    def _set_trace_view_mode(self, mode: str) -> None:
        time_window = self.trace_canvas.current_time_window() if self.trace_canvas.has_data() else None
        self.trace_view_mode = mode
        for key, button in self.trace_mode_buttons.items():
            button.setChecked(key == mode)
        result = self._trace_result_for_scope()
        if result:
            self.last_result = result
            self.trace_canvas.plot_result(result, mode=mode, time_window=time_window)
            self._sync_trace_window_controls()

    def _set_trace_display_scope(self, scope: str) -> None:
        if scope not in {"all", "selected"}:
            return
        self.trace_display_scope = scope
        self._update_trace_scope_buttons()
        self._refresh_trace_display_for_scope()

    def _extraction_setting_changed(self, checked: bool) -> None:
        if self.method_combo.currentText() != "Spikepursuit":
            return
        self.last_result = None
        self._set_last_batch_results([])
        self.save_all_traces_button.setEnabled(False)
        if self.trace_window.isVisible() or self.trace_canvas.has_data():
            self._refresh_trace_display_for_scope()
        else:
            self._refresh_roi_inspector()
        self.status_label.setText("Flip signal changed; extract traces to apply")

    def _update_trace_scope_buttons(self) -> None:
        if not hasattr(self, "trace_scope_switch"):
            return
        if self.trace_scope_switch.value() == self.trace_display_scope:
            return
        self.trace_scope_switch.setValue(self.trace_display_scope, animated=False)

    def _trace_result_for_scope(self) -> Optional[dict]:
        if self.trace_display_scope == "selected":
            return self._selected_roi_trace_result()
        return self._all_roi_trace_result()

    def _set_last_batch_results(self, results: list[dict]) -> None:
        self.last_batch_results = self._with_result_colors(list(results))
        self.combined_trace_result = None
        self.combined_trace_result_key = ()

    @staticmethod
    def _with_result_colors(results: list[dict]) -> list[dict]:
        for index, result in enumerate(results):
            result.setdefault("trace_color", TRACE_LINE_COLORS[index % len(TRACE_LINE_COLORS)])
        return results

    def _cached_combined_trace_result(self, results: list[dict]) -> dict:
        key = tuple(id(result) for result in results)
        if self.combined_trace_result is None or self.combined_trace_result_key != key:
            self.combined_trace_result = self._combined_trace_result(results)
            self.combined_trace_result_key = key
        return self.combined_trace_result

    def _all_roi_trace_result(self) -> Optional[dict]:
        if self.last_batch_results:
            return self._cached_combined_trace_result(self.last_batch_results)
        if self.last_result and self.last_result.get("multi_roi"):
            return self.last_result
        cached_results = self._cached_results_for_all_rois()
        if cached_results:
            self._set_last_batch_results(cached_results)
            return self._cached_combined_trace_result(cached_results)
        return None

    def _selected_roi_trace_result(self) -> Optional[dict]:
        if self.selected_roi_id is None:
            return None
        result = self._result_for_selected_roi(self.last_batch_results)
        if result:
            return result
        cache_key = self._current_extraction_key()
        if cache_key is not None and cache_key in self.trace_cache:
            return self.trace_cache[cache_key]
        if self.last_result and not self.last_result.get("multi_roi"):
            metrics = self.last_result.get("metrics") or {}
            if int(metrics.get("roi", -1)) == int(self.selected_roi_id):
                return self.last_result
        return None

    def _refresh_trace_display_for_scope(self) -> bool:
        result = self._trace_result_for_scope()
        if result:
            if self.trace_display_scope == "selected" and self.selected_roi_id is not None:
                status = f"Showing trace for ROI {int(self.selected_roi_id)}"
            else:
                count = len(self.last_batch_results) if self.last_batch_results else int((result.get("metrics") or {}).get("rois", 0))
                status = f"Showing all ROI traces ({count})" if count else "Showing all ROI traces"
            self._show_trace_result(result, status=status)
            return True

        self.last_result = None
        self.trace_canvas.plot_empty()
        self._sync_trace_window_controls()
        if self.trace_display_scope == "selected":
            message = "Select an ROI with an extracted trace"
        else:
            message = "Extract all ROIs to show the all-ROI trace view"
        self.status_label.setText(message)
        self._refresh_roi_inspector()
        return False

    def _set_inspector_view_mode(self, mode: str) -> None:
        self.inspector_view_mode = mode
        for key, button in self.inspector_mode_buttons.items():
            button.setChecked(key == mode)
        self._refresh_roi_inspector()

    def _set_trace_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.trace_start_spin,
            self.trace_end_spin,
            self.trace_reset_button,
            self.trace_save_png_button,
        ):
            widget.setEnabled(enabled)

    def _sync_trace_window_controls(self, xmin: Optional[float] = None, xmax: Optional[float] = None) -> None:
        has_data = self.trace_canvas.has_data()
        self._set_trace_controls_enabled(has_data)
        if not has_data:
            for spin in (self.trace_start_spin, self.trace_end_spin):
                spin.blockSignals(True)
                spin.setRange(0.0, 0.0)
                spin.setValue(0.0)
                spin.blockSignals(False)
            return

        data_min, data_max = self.trace_canvas.time_bounds()
        if xmin is None or xmax is None:
            xmin, xmax = self.trace_canvas.current_time_window()
        for spin, value in ((self.trace_start_spin, xmin), (self.trace_end_spin, xmax)):
            spin.blockSignals(True)
            spin.setRange(data_min, data_max)
            spin.setValue(max(data_min, min(data_max, float(value))))
            spin.blockSignals(False)

    def _apply_trace_time_window(self) -> None:
        if not self.trace_canvas.has_data():
            return
        self.trace_canvas.set_time_window(self.trace_start_spin.value(), self.trace_end_spin.value())
        sender = self.sender()
        if isinstance(sender, QAbstractSpinBox) and sender.hasFocus():
            sender.clearFocus()
            self.trace_canvas.setFocus()

    def _reset_trace_view(self) -> None:
        self.trace_canvas.reset_view()

    def save_trace_png_dialog(self) -> None:
        """Prompt for a path and save the current trace plot as PNG."""
        if not self.trace_canvas.has_data():
            QMessageBox.warning(self, "Save Trace Failed", "No trace plot is available to save.")
            return

        default_name = "volpy_traces.png"
        if self.movie_path:
            default_name = f"{Path(self.movie_path).stem}_traces.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trace Plot",
            default_name,
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".png":
            path = f"{path}.png"
        self.trace_canvas.save_png(path, dpi=300)
        self.extraction_status_label.setText(f"Saved trace plot to {Path(path).name}")

    def _run_primary_action(self) -> None:
        if self.movie_path is None:
            self.open_movie_dialog()
            return
        if self._has_extractable_rois():
            self.extract_all_traces()
            return
        self.run_cellpose_segmentation()

    def _show_help_dialog(self) -> None:
        QMessageBox.information(
            self,
            "TorchVolpy Studio",
            "Open a movie, run Cellpose ROI detection or load an ROI mask, then extract traces.",
        )

    def _update_top_actions(self) -> None:
        if not hasattr(self, "run_top_button"):
            return
        has_movie = self.movie_path is not None
        self.run_top_button.setEnabled(has_movie)
        self.export_top_button.setEnabled(bool(self.last_batch_results))
        self.settings_top_button.setEnabled(has_movie)

    def _refresh_roi_inspector(self) -> None:
        if not hasattr(self, "roi_title_label"):
            return

        roi_ids = available_roi_ids(self.roi_mask) if self.roi_mask is not None else np.array([], dtype=np.int64)
        selected = self.selected_roi_id
        selected_index = 0
        if selected is not None and roi_ids.size:
            matches = np.where(roi_ids == int(selected))[0]
            if matches.size:
                selected_index = int(matches[0]) + 1

        self.roi_count_label.setText(f"{selected_index} / {int(roi_ids.size)}")
        self.roi_title_label.setText(f"ROI {int(selected)}" if selected is not None else "No ROI selected")

        result = self.last_result or {}
        result_metrics = result.get("metrics") or {}
        if selected is not None and result:
            if result.get("multi_roi"):
                result = self._result_for_selected_roi(self.last_batch_results) or {}
            elif int(result_metrics.get("roi", -1)) != int(selected):
                result = {}
        method = str(result.get("method") or self.method_combo.currentText())
        metrics = result.get("metrics") or {}
        self.roi_method_value_label.setText(method if selected is not None else "--")
        self.roi_snr_value.setText(str(metrics.get("snr", "--")))

        spikes = np.asarray(result.get("spikes", []), dtype=float)
        spikes_count = metrics.get("spikes")
        if spikes_count is None:
            spikes_count = int(spikes.size) if spikes.size else "--"
        self.roi_spikes_value.setText(str(spikes_count))

        advanced_options = self._advanced_options_for_method(method)
        threshold = metrics.get("threshold", advanced_options.get("threshold", "--"))
        if isinstance(threshold, float):
            threshold = f"{threshold:g}"
        self.roi_threshold_value.setText(str(threshold))
        locality = metrics.get("locality", "--")
        if isinstance(locality, bool):
            locality = str(locality)
        self.roi_locality_value.setText(str(locality) if result else "--")
        if not result or locality == "--":
            _set_metric_value_color(self.roi_locality_value, "#0f172a")
        elif str(locality) == "False":
            _set_metric_value_color(self.roi_locality_value, "#dc2626")
        else:
            _set_metric_value_color(self.roi_locality_value, "#059669")

        def array_for(key: str) -> np.ndarray:
            arr = np.asarray(result.get(key, []), dtype=float)
            if arr.ndim == 0:
                return np.empty((0,), dtype=float)
            return arr

        trace = array_for("trace")
        dff = array_for("dff")
        reconstruction = array_for("reconstruction")
        subthreshold = array_for("subthreshold")
        weights = array_for("weights")
        templates = array_for("templates")
        frame_rate = max(1e-12, float(result.get("frame_rate", self.movie_frame_rate_spin.value()) or 1.0))

        stat_rows: list[tuple[str, str]]
        if self.inspector_view_mode == "weights" and weights.size:
            finite_weights = weights[np.isfinite(weights)]
            if finite_weights.size:
                weight_min = f"{float(np.nanmin(finite_weights)):.3g}"
                weight_max = f"{float(np.nanmax(finite_weights)):.3g}"
                weight_mean = f"{float(np.nanmean(finite_weights)):.3g}"
            else:
                weight_min = weight_max = weight_mean = "--"
            stat_rows = [
                ("Weight min", weight_min),
                ("Weight max", weight_max),
                ("Weight mean", weight_mean),
                ("Template samples", str(int(templates.size)) if templates.size else "--"),
                ("Locality", str(locality) if result else "--"),
                ("Status", "OK" if result else "--"),
            ]
        elif self.inspector_view_mode == "dff" and dff.size:
            finite_dff = dff[np.isfinite(dff)]
            if finite_dff.size:
                dff_min = f"{float(np.nanmin(finite_dff)):.3g}"
                dff_max = f"{float(np.nanmax(finite_dff)):.3g}"
                dff_mean = f"{float(np.nanmean(finite_dff)):.3g}"
                peak = f"{float(np.nanmax(finite_dff) - np.nanmin(finite_dff)):.3g}"
            else:
                dff_min = dff_max = dff_mean = peak = "--"
            duration = dff.shape[0] / frame_rate
            stat_rows = [
                ("Duration", f"{duration:.2f} s"),
                ("Sampling rate", f"{frame_rate:.1f} Hz"),
                ("Min dF/F", dff_min),
                ("Max dF/F", dff_max),
                ("Mean dF/F", dff_mean),
                ("Peak dF/F", peak),
            ]
        elif trace.ndim >= 1 and trace.size:
            duration = trace.shape[0] / frame_rate
            peak = float(np.nanmax(trace) - np.nanmin(trace)) if np.isfinite(trace).any() else 0.0
            first_spike = f"{float(spikes.min()) / frame_rate:.2f} s" if spikes.size else "--"
            last_spike = f"{float(spikes.max()) / frame_rate:.2f} s" if spikes.size else "--"
            stat_rows = [
                ("Duration", f"{duration:.2f} s"),
                ("Sampling rate", f"{frame_rate:.1f} Hz"),
                ("First spike", first_spike),
                ("Last spike", last_spike),
                ("Peak", f"{peak:.3g}"),
                ("Status", "OK"),
            ]
        else:
            stat_rows = [
                ("Duration", "--"),
                ("Sampling rate", f"{frame_rate:.1f} Hz" if self.movie_path else "--"),
                ("First spike", "--"),
                ("Last spike", "--"),
                ("Peak", "--"),
                ("Status", "--"),
            ]
        for index, (name, value) in enumerate(stat_rows[: len(self.roi_stat_labels)]):
            self.roi_stat_name_labels[index].setText(name)
            self.roi_stat_labels[index].setText(value)

        for mode, button in self.inspector_mode_buttons.items():
            available = bool(result)
            if mode == "dff":
                available = available and dff.size > 0
            elif mode == "weights":
                available = available and weights.size > 0
            button.setEnabled(available or mode == "trace")
        for mode, button in self.trace_mode_buttons.items():
            button.setEnabled(True)

        if self.movie is not None:
            shape = tuple(self.movie.shape)
            image_shape = shape[1:3] if len(shape) >= 3 else shape
            self.footer_image_label.setText(f"Image: {image_shape[1]} x {image_shape[0]}" if len(image_shape) >= 2 else "Image: --")
        else:
            self.footer_image_label.setText("Image: --")
        self.footer_rois_label.setText(f"{int(roi_ids.size)} ROIs")
        if self.movie_frame_rate_spin.value() > 0:
            dt = 1.0 / float(self.movie_frame_rate_spin.value())
            self.footer_sampling_label.setText(f"dt: {dt:.4f} s ({self.movie_frame_rate_spin.value():.1f} Hz)")
        else:
            self.footer_sampling_label.setText("dt: --")
        self.footer_result_label.setText(f"Last extraction: ROI {metrics.get('roi', '--')}" if result else "Last extraction: --")
        self.footer_method_label.setText(method if result else "--")
        self.footer_spikes_label.setText(f"{spikes_count} spikes" if spikes_count != "--" else "-- spikes")
        self.footer_snr_label.setText(f"SNR {metrics.get('snr', '--')}")
        self._update_top_actions()

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("controlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        movie_group = QGroupBox()
        movie_form = QFormLayout(movie_group)
        movie_form.setContentsMargins(14, 14, 14, 14)
        movie_form.setSpacing(9)
        self.dataset_edit = QLineEdit("movie")
        self.movie_frame_rate_spin = ModernDoubleSpinBox()
        self.movie_frame_rate_spin.setRange(0.001, 1000000.0)
        self.movie_frame_rate_spin.setDecimals(3)
        self.movie_frame_rate_spin.setValue(400.0)
        self.movie_frame_rate_spin.setSuffix(" Hz")
        _show_spinbox_buttons(self.movie_frame_rate_spin)
        self.motion_correction_check = SwitchButton("Motion Correction")
        self.motion_correction_check.setChecked(True)
        self.open_button = QPushButton("Open Movie")
        _set_button_role(self.open_button, "primary")
        _set_button_icon(self.open_button, FOLDER_SVG, "#ffffff")
        self.movie_label = QLabel("No movie loaded")
        self.movie_label.setProperty("role", "status")
        self.movie_label.setWordWrap(True)
        self.preprocess_progress = QProgressBar()
        self.preprocess_progress.setRange(0, 100)
        self.preprocess_progress.setValue(0)
        self.preprocess_progress.hide()
        self.preprocess_label = QLabel("")
        self.preprocess_label.setProperty("role", "status")
        self.preprocess_label.setWordWrap(True)
        self.preprocess_label.hide()
        movie_form.addRow(_section_title("Movie"))
        movie_form.addRow("Dataset", self.dataset_edit)
        movie_form.addRow("Frame rate", self.movie_frame_rate_spin)
        movie_form.addRow(self.motion_correction_check)
        movie_form.addRow(self.open_button)
        movie_form.addRow(self.movie_label)
        movie_form.addRow(self.preprocess_progress)
        movie_form.addRow(self.preprocess_label)
        layout.addWidget(movie_group)

        frame_group = QGroupBox()
        frame_layout = QVBoxLayout(frame_group)
        frame_layout.setContentsMargins(14, 14, 14, 14)
        frame_layout.setSpacing(9)
        self.frame_slider = QSlider(HORIZONTAL)
        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_label.setObjectName("frameCounter")
        self.play_button = QPushButton("Play")
        _set_button_role(self.play_button, "secondary")
        _set_button_icon(self.play_button, PLAY_SVG)
        self.speed_ratio_spin = ModernDoubleSpinBox()
        self.speed_ratio_spin.setRange(0.01, 10.0)
        self.speed_ratio_spin.setDecimals(2)
        self.speed_ratio_spin.setSingleStep(0.1)
        self.speed_ratio_spin.setValue(1.0)
        self.speed_ratio_spin.setSuffix("x")
        _show_spinbox_buttons(self.speed_ratio_spin)
        self.channel_spin = ModernSpinBox()
        self.channel_spin.setRange(0, 0)
        _show_spinbox_buttons(self.channel_spin)
        frame_layout.addWidget(_section_title("Playback"))
        frame_layout.addWidget(self.frame_slider)
        frame_layout.addWidget(self.frame_label)
        frame_layout.addWidget(self.play_button)
        frame_form = QFormLayout()
        frame_form.setSpacing(9)
        frame_form.addRow("Speed ratio", self.speed_ratio_spin)
        frame_form.addRow("Channel", self.channel_spin)
        frame_layout.addLayout(frame_form)
        layout.addWidget(frame_group)

        device_group = QGroupBox()
        device_form = QFormLayout(device_group)
        device_form.setContentsMargins(14, 14, 14, 14)
        device_form.setSpacing(9)
        self.device_combo = ModernComboBox()
        for label, value in _available_device_options():
            self.device_combo.addItem(label, value)
        default_device = "cuda" if torch.cuda.is_available() else ("mps" if _mps_available() else "cpu")
        default_index = self.device_combo.findData(default_device)
        self.device_combo.setCurrentIndex(max(0, default_index))
        device_form.addRow(_section_title("Device"))
        device_form.addRow("Processing", self.device_combo)
        layout.addWidget(device_group)

        segmentation_group = QGroupBox()
        segmentation_layout = QVBoxLayout(segmentation_group)
        segmentation_layout.setContentsMargins(14, 14, 14, 14)
        segmentation_layout.setSpacing(9)
        default_model = Path.home() / "Downloads" / "volpy_finetuned"
        self.cellpose_model_edit = QLineEdit(str(default_model) if default_model.exists() else "")
        self.browse_cellpose_button = QPushButton("Browse Model")
        _set_button_role(self.browse_cellpose_button, "secondary")
        _set_button_icon(self.browse_cellpose_button, FOLDER_SVG)
        self.baseline_percentile_spin = ModernDoubleSpinBox()
        self.baseline_percentile_spin.setRange(0.0, 100.0)
        self.baseline_percentile_spin.setValue(8.0)
        self.baseline_percentile_spin.setSuffix(" %")
        _show_spinbox_buttons(self.baseline_percentile_spin)
        self.baseline_percentile_spin.setToolTip(
            "For each pixel, use this low-percentile frame value\n"
            "as the background before building the Cellpose image.\n"
            "Lower values subtract a dimmer baseline, so more bright\n"
            "changes remain and ROIs can look stronger/larger."
        )
        self.save_segmentation_check = SwitchButton("Save segmentation")
        self.save_segmentation_check.setChecked(False)
        self.load_mask_button = QPushButton("Load Mask")
        _set_button_role(self.load_mask_button, "secondary")
        _set_button_icon(self.load_mask_button, FOLDER_SVG)
        self.run_cellpose_button = QPushButton("Run segmentation")
        _set_button_role(self.run_cellpose_button, "primary")
        _set_button_icon(self.run_cellpose_button, SPARK_SVG, "#ffffff")
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setRange(0, 100)
        self.segmentation_progress.setValue(0)
        self.segmentation_progress.hide()
        self.segmentation_status_label = QLabel("Idle")
        self.segmentation_status_label.setProperty("role", "status")
        self.segmentation_status_label.setWordWrap(True)
        self.segmentation_status_label.hide()
        segmentation_form = QFormLayout()
        segmentation_form.setSpacing(9)
        segmentation_form.addRow("Model", self.cellpose_model_edit)
        baseline_label = QLabel("Baseline")
        baseline_label.setToolTip(self.baseline_percentile_spin.toolTip())
        segmentation_form.addRow(baseline_label, self.baseline_percentile_spin)
        segmentation_layout.addWidget(_section_title("ROI Masks"))
        segmentation_layout.addLayout(segmentation_form)
        segmentation_layout.addWidget(self.browse_cellpose_button)
        segmentation_layout.addWidget(self.save_segmentation_check)
        segmentation_layout.addWidget(self.load_mask_button)
        segmentation_layout.addWidget(self.run_cellpose_button)
        segmentation_layout.addWidget(self.segmentation_progress)
        segmentation_layout.addWidget(self.segmentation_status_label)
        layout.addWidget(segmentation_group)

        extraction_group = QGroupBox()
        extraction_layout = QVBoxLayout(extraction_group)
        extraction_layout.setContentsMargins(14, 14, 14, 14)
        extraction_layout.setSpacing(9)
        self.method_combo = ModernComboBox()
        self.method_combo.addItems(["Spikepursuit", "ALI", "Mean ROI"])
        self.advanced_button = QPushButton("Advanced...")
        _set_button_role(self.advanced_button, "secondary")
        _set_button_icon(self.advanced_button, SLIDERS_SVG)
        self.selected_roi_label = QLabel("Selected ROI: none")
        self.selected_roi_label.setProperty("role", "status")
        self.flip_signal_check = SwitchButton("Flip signal")
        self.flip_signal_check.setChecked(True)
        self.extract_all_button = QPushButton("Extract All ROIs")
        _set_button_role(self.extract_all_button, "primary")
        _set_button_icon(self.extract_all_button, TARGET_SVG, "#ffffff")
        self.save_all_traces_button = QPushButton("Save All Traces CSV")
        _set_button_role(self.save_all_traces_button, "secondary")
        _set_button_icon(self.save_all_traces_button, DOWNLOAD_SVG)
        self.save_all_traces_button.setEnabled(False)
        self.extraction_progress = QProgressBar()
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_progress.hide()
        self.extraction_status_label = QLabel("Idle")
        self.extraction_status_label.setProperty("role", "status")
        self.extraction_status_label.setWordWrap(True)
        self.status_label = self.extraction_status_label
        extraction_form = QFormLayout()
        extraction_form.setSpacing(9)
        extraction_form.addRow("Method", self.method_combo)
        extraction_layout.addWidget(_section_title("Extraction"))
        extraction_layout.addLayout(extraction_form)
        extraction_layout.addWidget(self.advanced_button)
        extraction_layout.addWidget(self.selected_roi_label)
        extraction_layout.addWidget(self.flip_signal_check)
        extraction_layout.addWidget(self.extract_all_button)
        extraction_layout.addWidget(self.save_all_traces_button)
        extraction_layout.addWidget(self.extraction_progress)
        extraction_layout.addWidget(self.extraction_status_label)
        layout.addWidget(extraction_group)

        layout.addStretch(1)
        return panel

    def _selected_device(self) -> str:
        data = self.device_combo.currentData()
        if data is not None:
            return str(data)
        return self.device_combo.currentText().strip().lower()

    def _connect_signals(self) -> None:
        self.reset_view_button.clicked.connect(self._reset_movie_view)
        self.zoom_out_button.clicked.connect(lambda: self._zoom_movie_view(0.8))
        self.zoom_100_button.clicked.connect(self._set_movie_actual_size)
        self.zoom_in_button.clicked.connect(lambda: self._zoom_movie_view(1.25))
        self.fit_mode_combo.currentTextChanged.connect(self._set_movie_fit_mode)
        self.display_mode_combo.currentTextChanged.connect(self._set_movie_display_mode)
        self.show_trace_button.clicked.connect(self._toggle_trace_window)
        for mode, button in self.trace_mode_buttons.items():
            button.clicked.connect(lambda checked=False, mode=mode: self._set_trace_view_mode(mode))
        self.trace_scope_switch.valueChanged.connect(self._set_trace_display_scope)
        for mode, button in self.inspector_mode_buttons.items():
            button.clicked.connect(lambda checked=False, mode=mode: self._set_inspector_view_mode(mode))
        self.open_button.clicked.connect(self.open_movie_dialog)
        self.frame_slider.valueChanged.connect(self.show_frame)
        self.play_button.clicked.connect(self.toggle_playback)
        self.movie_frame_rate_spin.valueChanged.connect(self._update_timer_interval)
        self.movie_frame_rate_spin.valueChanged.connect(lambda _: self._refresh_roi_inspector())
        self.speed_ratio_spin.valueChanged.connect(self._update_timer_interval)
        self.channel_spin.valueChanged.connect(lambda _: self.show_frame(self.frame_slider.value()))
        self.movie_view.roiPicked.connect(self._select_roi)
        self.movie_view.roiSelectionCleared.connect(self._clear_roi_selection)
        self.movie_view.roiChanged.connect(self._add_drawn_roi)
        self.movie_view.statusChanged.connect(self.extraction_status_label.setText)
        self.movie_view.zoomChanged.connect(self._refresh_zoom_label)
        for mode, button in self.roi_tool_buttons.items():
            button.clicked.connect(lambda checked=False, mode=mode: self._set_roi_tool_mode(mode if checked else "select"))
        self.roi_brush_size_spin.valueChanged.connect(self.movie_view.set_brush_size)
        self.roi_undo_button.clicked.connect(self._undo_roi_mask_edit)
        self.roi_undo_shortcut.activated.connect(self._undo_roi_mask_edit)
        self.roi_clear_tools_shortcut.activated.connect(lambda: self._set_roi_tool_mode("select"))
        self.browse_cellpose_button.clicked.connect(self.browse_cellpose_model)
        self.load_mask_button.clicked.connect(self.load_mask_dialog)
        self.run_cellpose_button.clicked.connect(self.run_cellpose_segmentation)
        self.labels_toggle_button.toggled.connect(self.movie_view.set_overlay_visible)
        self.extract_all_button.clicked.connect(self.extract_all_traces)
        self.advanced_button.clicked.connect(self.open_advanced_options_dialog)
        self.save_all_traces_button.clicked.connect(self.save_all_traces_dialog)
        self.trace_start_spin.editingFinished.connect(self._apply_trace_time_window)
        self.trace_end_spin.editingFinished.connect(self._apply_trace_time_window)
        self.trace_reset_button.clicked.connect(self._reset_trace_view)
        self.trace_save_png_button.clicked.connect(self.save_trace_png_dialog)
        self.trace_canvas.viewChanged.connect(self._sync_trace_window_controls)
        self.trace_canvas.roiDoubleClicked.connect(self._trace_roi_double_clicked)
        self.trace_window.visibilityChanged.connect(self._update_trace_window_button)
        self.flip_signal_check.toggled.connect(self._extraction_setting_changed)

    def _set_playback_button_active(self, active: bool) -> None:
        self.play_button.setText("Pause" if active else "Play")
        _set_button_icon(self.play_button, PAUSE_SVG if active else PLAY_SVG)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.frame_slider,
            self.play_button,
            self.speed_ratio_spin,
            self.channel_spin,
            self.cellpose_model_edit,
            self.browse_cellpose_button,
            self.baseline_percentile_spin,
            self.save_segmentation_check,
            self.load_mask_button,
            self.labels_toggle_button,
            self.fit_mode_combo,
            self.display_mode_combo,
            self.zoom_out_button,
            self.zoom_100_button,
            self.zoom_in_button,
            self.reset_view_button,
            self.roi_brush_size_spin,
            self.run_cellpose_button,
            self.method_combo,
            self.device_combo,
            self.advanced_button,
            self.flip_signal_check,
            self.extract_all_button,
        ):
            widget.setEnabled(enabled)
        for button in self.roi_tool_buttons.values():
            button.setEnabled(enabled)
        self._update_roi_undo_button()
        self._update_top_actions()

    def _has_extractable_rois(self) -> bool:
        return self.roi_mask is not None and available_roi_ids(self.roi_mask).size > 0

    def _update_roi_undo_button(self) -> None:
        if not hasattr(self, "roi_undo_button"):
            return
        self.roi_undo_button.setEnabled(self.movie_path is not None and bool(self.roi_mask_history))

    def _clear_roi_mask_history(self) -> None:
        self.roi_mask_history.clear()
        self._update_roi_undo_button()

    def _push_roi_mask_history(self) -> None:
        snapshot = None if self.roi_mask is None else np.asarray(self.roi_mask, dtype=np.int32).copy()
        self.roi_mask_history.append((snapshot, self.selected_roi_id))
        if len(self.roi_mask_history) > ROI_MASK_HISTORY_LIMIT:
            del self.roi_mask_history[0 : len(self.roi_mask_history) - ROI_MASK_HISTORY_LIMIT]
        self._update_roi_undo_button()

    def _set_extraction_controls_enabled(self, enabled: bool) -> None:
        enabled = enabled and self.movie_path is not None and self._has_extractable_rois()
        for widget in (
            self.method_combo,
            self.advanced_button,
            self.flip_signal_check,
            self.extract_all_button,
        ):
            widget.setEnabled(enabled)
        self.save_all_traces_button.setEnabled(enabled and bool(self.last_batch_results))
        self._update_top_actions()

    def _set_segmentation_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.cellpose_model_edit,
            self.browse_cellpose_button,
            self.baseline_percentile_spin,
            self.save_segmentation_check,
            self.load_mask_button,
            self.labels_toggle_button,
            self.run_cellpose_button,
        ):
            widget.setEnabled(enabled)
        self._update_top_actions()

    def _reset_loaded_movie_state(self) -> None:
        self.timer.stop()
        self._playback_last_time = None
        self._playback_frame_credit = 0.0
        self._set_playback_button_active(False)

        if self.movie is not None and hasattr(self.movie, "close"):
            self.movie.close()
        self.movie = None
        self.movie_path = None
        self.roi_mask = None
        self.selected_roi_id = None
        self.last_result = None
        self._set_last_batch_results([])
        self.trace_display_scope = "all"
        self._update_trace_scope_buttons()
        self.trace_cache.clear()
        self.pending_extraction_keys = {}
        self.pending_extraction_roi_ids = []
        self.roi_mask_version = 0
        self._clear_roi_mask_history()

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText("Frame 0 / 0")
        self.channel_spin.setRange(0, 0)

        self.movie_label.setText("No movie loaded")
        self.selected_roi_label.setText("Selected ROI: none")
        self.segmentation_progress.hide()
        self.segmentation_progress.setRange(0, 100)
        self.segmentation_progress.setValue(0)
        self.segmentation_status_label.setText("Idle")
        self.segmentation_status_label.hide()
        self.extraction_progress.hide()
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_status_label.setText("Idle")
        self.labels_toggle_button.setChecked(True)
        self._set_roi_tool_mode("select")
        self.save_all_traces_button.setEnabled(False)
        self.trace_canvas.plot_empty()
        self._sync_trace_window_controls()
        self.trace_window.hide()
        self._update_trace_window_button(False)
        self.movie_view.show_empty_message("No movie loaded")
        self._set_controls_enabled(False)
        self._refresh_roi_inspector()

    def open_movie_dialog(self) -> None:
        """Prompt the user to choose and open a movie file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Movie",
            "",
            "Movies (*.h5 *.hdf5 *.tif *.tiff);;All files (*)",
        )
        if path:
            self._reset_loaded_movie_state()
            frame_rate, ok = _get_movie_frame_rate(self, float(self.movie_frame_rate_spin.value()))
            if not ok:
                return
            self.movie_frame_rate_spin.setValue(frame_rate)
            self.preprocess_and_load_movie(path)

    def preprocess_and_load_movie(self, path: str) -> None:
        """Preprocess a source movie and load the corrected result."""
        self._reset_loaded_movie_state()
        device = self._selected_device()
        self.open_button.setEnabled(False)
        if hasattr(self, "open_top_button"):
            self.open_top_button.setEnabled(False)
        self.dataset_edit.setEnabled(False)
        self.motion_correction_check.setEnabled(False)
        self.preprocess_progress.setValue(0)
        self.preprocess_progress.show()
        motion_correction = self.motion_correction_check.isChecked()
        self.preprocess_label.setText(
            f"Preparing movie on {device.upper()}" if motion_correction else "Loading movie without motion correction"
        )
        self.preprocess_label.show()
        self.movie_label.setText(f"Preparing {Path(path).name}")

        self.preprocess_thread = QThread(self)
        self.preprocess_worker = PreprocessWorker(
            source_path=path,
            dataset=self.dataset_edit.text().strip() or "movie",
            motion_correction=motion_correction,
            device=device,
        )
        self.preprocess_worker.moveToThread(self.preprocess_thread)
        self.preprocess_thread.started.connect(self.preprocess_worker.run)
        self.preprocess_worker.progress.connect(self._preprocess_progress)
        self.preprocess_worker.finished.connect(self._preprocess_finished)
        self.preprocess_worker.failed.connect(self._preprocess_failed)
        self.preprocess_worker.finished.connect(self.preprocess_thread.quit)
        self.preprocess_worker.failed.connect(self.preprocess_thread.quit)
        self.preprocess_thread.finished.connect(self.preprocess_worker.deleteLater)
        self.preprocess_thread.finished.connect(self.preprocess_thread.deleteLater)
        self.preprocess_thread.finished.connect(lambda: self.open_button.setEnabled(True))
        if hasattr(self, "open_top_button"):
            self.preprocess_thread.finished.connect(lambda: self.open_top_button.setEnabled(True))
        self.preprocess_thread.finished.connect(lambda: self.dataset_edit.setEnabled(True))
        self.preprocess_thread.finished.connect(lambda: self.motion_correction_check.setEnabled(True))
        self.preprocess_thread.start()

    def _preprocess_progress(self, value: int, message: str) -> None:
        self.preprocess_progress.setValue(int(value))
        self.preprocess_label.setText(message)

    def _preprocess_finished(self, corrected_path: str) -> None:
        self.preprocess_progress.setValue(100)
        if self.motion_correction_check.isChecked():
            self.preprocess_label.setText(f"Loaded corrected movie: {Path(corrected_path).name}")
        else:
            self.preprocess_label.setText(f"Loaded movie: {Path(corrected_path).name}")
        self.load_movie(corrected_path)

    def _preprocess_failed(self, message: str) -> None:
        self.preprocess_label.setText("Movie preparation failed")
        self.movie_label.setText("No movie loaded")
        self.movie_view.show_empty_message("No movie loaded")
        QMessageBox.critical(self, "Movie Preparation Failed", message)

    def load_movie(self, path: str) -> None:
        """Load a movie into the GUI state and viewer."""
        self._reset_loaded_movie_state()

        dataset = self.dataset_edit.text().strip() or "movie"
        try:
            self.movie = open_movie(path, dataset=dataset)
            self.movie_path = path
            shape = tuple(self.movie.shape)
            if len(shape) not in (3, 4):
                raise ValueError(f"Expected movie shape (T, Y, X) or (T, Y, X, C), got {shape}")
            if len(shape) == 4:
                self.channel_spin.setRange(0, int(shape[3]) - 1)
                self.channel_spin.setEnabled(True)
            else:
                self.channel_spin.setRange(0, 0)
                self.channel_spin.setEnabled(False)

            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, int(shape[0]) - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.movie_label.setText(f"{Path(path).name}\nshape={shape}, dtype={self.movie.dtype}")
            self.roi_mask = None
            self.selected_roi_id = None
            self.selected_roi_label.setText("Selected ROI: none")
            self.last_result = None
            self._set_last_batch_results([])
            self.trace_display_scope = "all"
            self._update_trace_scope_buttons()
            self.trace_cache.clear()
            self.pending_extraction_keys = {}
            self.pending_extraction_roi_ids = []
            self.roi_mask_version = 0
            self.save_all_traces_button.setEnabled(False)
            self.extraction_progress.hide()
            self.extraction_progress.setRange(0, 100)
            self.extraction_progress.setValue(0)
            self.extraction_status_label.setText("Idle")
            self.segmentation_progress.hide()
            self.segmentation_progress.setRange(0, 100)
            self.segmentation_progress.setValue(0)
            self.segmentation_status_label.setText("Idle")
            self.segmentation_status_label.hide()
            self.labels_toggle_button.setChecked(True)
            self.movie_view.set_overlay_visible(True)
            self.trace_canvas.plot_empty()
            self._sync_trace_window_controls()
            self.trace_window.hide()
            self._update_trace_window_button(False)
            self.movie_view.clear_roi()
            self._set_controls_enabled(True)
            self.channel_spin.setEnabled(len(shape) == 4)
            self._set_extraction_controls_enabled(True)
            self._update_timer_interval()
            self.show_frame(0)
            self.extraction_status_label.setText("Movie loaded")
            self._refresh_roi_inspector()
        except Exception as exc:
            if self.movie is not None and hasattr(self.movie, "close"):
                self.movie.close()
            self.movie = None
            self.movie_path = None
            self.movie_label.setText("No movie loaded")
            self.movie_view.show_empty_message("No movie loaded")
            self._set_controls_enabled(False)
            self._refresh_roi_inspector()
            QMessageBox.critical(self, "Open Movie Failed", str(exc))

    def _read_rgb_display_frame(self, frame_index: int) -> np.ndarray:
        if self.movie is None:
            raise ValueError("No movie loaded")

        shape = tuple(self.movie.shape)
        if len(shape) == 4 and int(shape[3]) >= 3:
            frame = self.movie.read(
                (int(frame_index), slice(None), slice(None), slice(0, 3)),
                as_tensor=False,
                copy=True,
            )
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                return frame[..., :3]

        gray = read_display_frame(
            self.movie,
            int(frame_index),
            channel=self.channel_spin.value() if len(shape) == 4 else None,
        )
        return np.repeat(np.asarray(gray)[:, :, None], 3, axis=2)

    def show_frame(self, frame_index: int) -> None:
        """Display the current movie frame and synchronized overlays."""
        if self.movie is None:
            return
        try:
            display_mode = self.display_mode_combo.currentText().strip().lower()
            if display_mode == "rgb":
                frame = self._read_rgb_display_frame(int(frame_index))
                self.movie_view.set_frame(frame, rgb=True)
            else:
                frame = read_display_frame(
                    self.movie,
                    int(frame_index),
                    channel=self.channel_spin.value() if len(tuple(self.movie.shape)) == 4 else None,
                )
                self.movie_view.set_frame(frame, rgb=False)
            self._update_roi_overlay()
            total = int(self.movie.shape[0])
            self.frame_label.setText(f"Frame {int(frame_index) + 1} / {total}")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def toggle_playback(self) -> None:
        """Start or stop movie playback."""
        if self.timer.isActive():
            self.timer.stop()
            self._playback_last_time = None
            self._playback_frame_credit = 0.0
            self._set_playback_button_active(False)
        else:
            self._update_timer_interval()
            self._playback_last_time = time.perf_counter()
            self._playback_frame_credit = 0.0
            self.timer.start()
            self._set_playback_button_active(True)

    def _update_timer_interval(self) -> None:
        self.timer.setInterval(self.playback_timer_ms)
        if self.timer.isActive():
            self._playback_last_time = time.perf_counter()
            self._playback_frame_credit = 0.0

    def _effective_playback_fps(self) -> float:
        return max(
            0.001,
            float(self.movie_frame_rate_spin.value()) * float(self.speed_ratio_spin.value()),
        )

    def _advance_frame(self) -> None:
        if self.movie is None:
            return

        now = time.perf_counter()
        if self._playback_last_time is None:
            self._playback_last_time = now
            return

        elapsed = max(0.0, now - self._playback_last_time)
        self._playback_last_time = now
        self._playback_frame_credit += elapsed * self._effective_playback_fps()
        frames_to_advance = int(self._playback_frame_credit)
        if frames_to_advance < 1:
            return
        self._playback_frame_credit -= frames_to_advance

        minimum = self.frame_slider.minimum()
        maximum = self.frame_slider.maximum()
        frame_count = maximum - minimum + 1
        if frame_count <= 1:
            return

        value = minimum + ((self.frame_slider.value() - minimum + frames_to_advance) % frame_count)
        self.frame_slider.setValue(value)

    def browse_cellpose_model(self) -> None:
        """Prompt the user to select a Cellpose model file."""
        current_path = self.cellpose_model_edit.text().strip()
        initial_dir = str(Path(current_path).parent) if current_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cellpose Model",
            initial_dir,
            "Cellpose models (*.pt *.pth *.torch *.npy);;All files (*)",
        )
        if path:
            self.cellpose_model_edit.setText(path)

    def load_mask_dialog(self) -> None:
        """Prompt the user to load an ROI mask file."""
        if self.movie_path is None or self.movie is None:
            QMessageBox.warning(self, "Missing Movie", "Open a movie before loading an ROI mask.")
            return

        initial_dir = str(Path(self.movie_path).parent) if self.movie_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROI Mask",
            initial_dir,
            "ROI masks (*.h5 *.hdf5 *.npy *.npz *.tif *.tiff);;All files (*)",
        )
        if not path:
            return

        try:
            mask = load_mask_file(path)
            ids = self._apply_roi_mask(mask)
            self._clear_roi_mask_history()
        except Exception as exc:
            QMessageBox.critical(self, "Load Mask Failed", str(exc))
            return

        self.segmentation_progress.setRange(0, 100)
        self.segmentation_progress.setValue(100)
        self.segmentation_progress.hide()
        self.segmentation_status_label.setText(
            f"Loaded {ids.size} ROIs from {Path(path).name}; click an ROI to select it"
        )
        self.segmentation_status_label.show()
        self.extraction_status_label.setText("ROI mask loaded")

    def run_cellpose_segmentation(self) -> None:
        """Run Cellpose and load the generated ROI mask."""
        if self.movie_path is None:
            QMessageBox.warning(self, "Missing Movie", "Open a movie before running Cellpose.")
            return

        self.timer.stop()
        self._set_playback_button_active(False)
        self._set_segmentation_controls_enabled(False)
        self._set_extraction_controls_enabled(False)
        self.segmentation_progress.setRange(0, 0)
        self.segmentation_progress.show()
        self.segmentation_status_label.setText("Starting segmentation")
        self.segmentation_status_label.show()

        channel = self.channel_spin.value() if self.movie is not None and len(tuple(self.movie.shape)) == 4 else None
        summary_window_size = max(1, int(round(float(self.movie_frame_rate_spin.value()))))
        compute_device = self._selected_device()
        self.segmentation_thread = QThread(self)
        self.segmentation_worker = SegmentationWorker(
            movie_path=self.movie_path,
            dataset=self.dataset_edit.text().strip() or "movie",
            model_path=self.cellpose_model_edit.text().strip(),
            channel=channel,
            summary_window_size=summary_window_size,
            baseline_percentile=self.baseline_percentile_spin.value(),
            device=compute_device,
            gpu=compute_device in {"cuda", "mps"},
            save_to_disk=self.save_segmentation_check.isChecked(),
        )
        self.segmentation_worker.moveToThread(self.segmentation_thread)
        self.segmentation_thread.started.connect(self.segmentation_worker.run)
        self.segmentation_worker.status.connect(self.segmentation_status_label.setText)
        self.segmentation_worker.finished.connect(self._segmentation_finished)
        self.segmentation_worker.failed.connect(self._segmentation_failed)
        self.segmentation_worker.finished.connect(self.segmentation_thread.quit)
        self.segmentation_worker.failed.connect(self.segmentation_thread.quit)
        self.segmentation_thread.finished.connect(self.segmentation_worker.deleteLater)
        self.segmentation_thread.finished.connect(self.segmentation_thread.deleteLater)
        self.segmentation_thread.finished.connect(
            lambda: self._set_segmentation_controls_enabled(self.movie_path is not None)
        )
        self.segmentation_thread.finished.connect(
            lambda: self._set_extraction_controls_enabled(self.movie_path is not None)
        )
        self.segmentation_thread.start()

    def _clear_trace_state_for_new_roi_mask(self) -> None:
        self.selected_roi_id = None
        self.selected_roi_label.setText("Selected ROI: none")
        self.last_result = None
        self._set_last_batch_results([])
        self.trace_display_scope = "all"
        self._update_trace_scope_buttons()
        self.pending_extraction_keys = {}
        self.pending_extraction_roi_ids = []
        self.save_all_traces_button.setEnabled(False)
        self.extraction_progress.hide()
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_status_label.setText("Idle")
        self.trace_canvas.plot_empty()
        self._sync_trace_window_controls()
        self.trace_window.hide()
        self._update_trace_window_button(False)
        self.labels_toggle_button.setChecked(True)
        self.movie_view.set_overlay_visible(True)

    def _apply_roi_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.int32)
        if self.movie is not None:
            ensure_shape_matches(mask, tuple(self.movie.shape[1:]))
        ids = available_roi_ids(mask)
        if ids.size == 0:
            raise ValueError("ROI mask does not contain any positive labels.")

        return self._set_roi_mask_state(mask)

    def _set_roi_mask_state(
        self,
        mask: Optional[np.ndarray],
        *,
        selected_roi_id: Optional[int] = None,
    ) -> np.ndarray:
        ids = np.array([], dtype=np.int64)
        state_mask: Optional[np.ndarray] = None
        if mask is None:
            state_mask = None
        else:
            mask = np.asarray(mask, dtype=np.int32)
            if self.movie is not None:
                ensure_shape_matches(mask, tuple(self.movie.shape[1:]))
            ids = available_roi_ids(mask)
            state_mask = mask if ids.size else None

        self.roi_mask = state_mask
        if self.roi_mask is None:
            ids = np.array([], dtype=np.int64)

        selected_roi_id = int(selected_roi_id) if selected_roi_id is not None else None
        self.roi_mask_version += 1
        self._clear_trace_state_for_new_roi_mask()
        self._prune_trace_cache_for_current_roi_mask()
        if (
            self.roi_mask is not None
            and selected_roi_id is not None
            and np.any(self.roi_mask == int(selected_roi_id))
        ):
            self.selected_roi_id = int(selected_roi_id)
            self.selected_roi_label.setText(f"Selected ROI: {int(selected_roi_id)}")
        self._update_roi_overlay()
        self._set_extraction_controls_enabled(True)
        self._refresh_roi_inspector()
        self._update_roi_undo_button()
        return ids

    def _undo_roi_mask_edit(self) -> None:
        if not self.roi_mask_history:
            self.segmentation_status_label.setText("No ROI edit to undo")
            self.segmentation_status_label.show()
            self._update_roi_undo_button()
            return

        snapshot, selected_roi_id = self.roi_mask_history.pop()
        ids = self._set_roi_mask_state(snapshot, selected_roi_id=selected_roi_id)
        self._update_roi_undo_button()
        if ids.size:
            self.segmentation_status_label.setText("Undid ROI edit")
        else:
            self.segmentation_status_label.setText("Undid ROI edit; no ROI mask is active")
        self.segmentation_status_label.show()

    def _add_drawn_roi(self, mask: np.ndarray, metadata: dict) -> None:
        if self.movie is None:
            return

        drawn = np.asarray(mask, dtype=np.int32)
        ensure_shape_matches(drawn, tuple(self.movie.shape[1:]))
        pixels = drawn > 0
        if not np.any(pixels):
            return

        tool = str((metadata or {}).get("type", "drawn")).strip().lower()
        if tool == "eraser":
            if self.roi_mask is None:
                self.segmentation_status_label.setText("No ROI mask to erase")
                self.segmentation_status_label.show()
                return

            combined = np.asarray(self.roi_mask, dtype=np.int32).copy()
            touched_roi_ids = np.unique(combined[pixels & (combined > 0)])
            touched_roi_ids = touched_roi_ids[touched_roi_ids > 0]
            if touched_roi_ids.size == 0:
                self.segmentation_status_label.setText("Eraser did not touch any ROI")
                self.segmentation_status_label.show()
                return

            self._push_roi_mask_history()
            selected_roi_id = self.selected_roi_id
            combined[np.isin(combined, touched_roi_ids)] = 0
            ids = self._set_roi_mask_state(combined, selected_roi_id=selected_roi_id)
            if ids.size:
                erased_label = "ROI" if touched_roi_ids.size == 1 else "ROIs"
                self.segmentation_status_label.setText(f"Erased {int(touched_roi_ids.size)} {erased_label}")
            else:
                self.segmentation_status_label.setText("Erased all ROIs")
            self.segmentation_status_label.show()
            return

        if self.roi_mask is None:
            combined = np.zeros_like(drawn, dtype=np.int32)
            next_roi_id = 1
        else:
            combined = np.asarray(self.roi_mask, dtype=np.int32).copy()
            ids = available_roi_ids(combined)
            next_roi_id = int(ids.max()) + 1 if ids.size else 1

        self._push_roi_mask_history()
        combined[pixels] = int(next_roi_id)
        self._apply_roi_mask(combined)
        self._select_roi(int(next_roi_id))
        self.segmentation_status_label.setText(f"Drew ROI {int(next_roi_id)} with {tool.replace('_', ' ')}")
        self.segmentation_status_label.show()

    def _segmentation_finished(self, result: dict) -> None:
        mask = np.asarray(result["mask"], dtype=np.int32)
        ids = self._apply_roi_mask(mask)
        self._clear_roi_mask_history()
        where_saved = f", saved to {result['save_dir']}" if result.get("save_dir") else ""
        self.segmentation_progress.setRange(0, 100)
        self.segmentation_progress.setValue(100)
        self.segmentation_status_label.setText(f"Cellpose found {ids.size} ROIs{where_saved}; click an ROI to select it")

    def _segmentation_failed(self, message: str) -> None:
        self.segmentation_progress.setRange(0, 100)
        self.segmentation_progress.setValue(0)
        self.segmentation_status_label.setText("Cellpose failed")
        self.segmentation_status_label.show()
        QMessageBox.critical(self, "Cellpose Failed", message)

    def _update_roi_overlay(self) -> None:
        if self.roi_mask is None:
            self.movie_view.clear_mask_overlay()
            return
        self.movie_view.set_mask_overlay(self.roi_mask, self.selected_roi_id or 0)

    def _select_roi(self, roi_id: int) -> None:
        if self.roi_mask is None:
            return
        if not np.any(self.roi_mask == int(roi_id)):
            return
        self.selected_roi_id = int(roi_id)
        self.selected_roi_label.setText(f"Selected ROI: {int(roi_id)}")
        self._update_roi_overlay()
        self.movie_view.center_on_roi(int(roi_id))
        self._refresh_roi_inspector()
        if self.trace_display_scope == "selected":
            if self._refresh_trace_display_for_scope():
                return
            self.extraction_status_label.setText(f"No extracted trace available for ROI {int(roi_id)}")
            return
        self.extraction_status_label.setText(f"Selected ROI {int(roi_id)}")

    def _clear_roi_selection(self) -> None:
        if self.selected_roi_id is None:
            return
        self.selected_roi_id = None
        self.selected_roi_label.setText("Selected ROI: none")
        self._update_roi_overlay()
        self._refresh_roi_inspector()
        if self.trace_display_scope == "selected":
            self._refresh_trace_display_for_scope()
        self.extraction_status_label.setText("No ROI selected")

    def _trace_roi_double_clicked(self, roi_id: int) -> None:
        self._select_roi(int(roi_id))
        if self.selected_roi_id == int(roi_id):
            self._set_trace_display_scope("selected")

    def _advanced_options_for_method(self, method: str) -> dict[str, Any]:
        options = _advanced_defaults(method)
        options.update(self.advanced_options_by_method.get(method, {}))
        return options

    def open_advanced_options_dialog(self) -> None:
        """Open the advanced extraction options dialog."""
        method = self.method_combo.currentText()
        specs = ADVANCED_OPTION_SPECS.get(method, ())
        if not specs:
            QMessageBox.information(self, "Advanced Options", f"No advanced options are available for {method}.")
            return

        dialog = QDialog(self)
        dialog.setObjectName("advancedOptionsDialog")
        dialog.setWindowTitle(f"{method} Advanced Options")
        dialog_layout = QVBoxLayout(dialog)

        form_widget = QWidget()
        form_widget.setObjectName("advancedOptionsForm")
        form_layout = QFormLayout(form_widget)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow if PYQT_VERSION == 6 else QFormLayout.AllNonFixedFieldsGrow)
        current_options = self._advanced_options_for_method(method)
        editors: dict[str, QWidget] = {}

        def set_editor_value(editor: QWidget, spec: dict[str, Any], value: Any) -> None:
            option_type = spec["type"]
            if option_type in {"int", "float"}:
                editor.setValue(value)
            elif option_type == "choice":
                index = editor.findText(str(value))
                editor.setCurrentIndex(max(0, index))
            elif option_type == "float_list":
                editor.setText(_format_float_list(value))
            elif option_type == "bool":
                editor.setChecked(bool(value))

        for spec in specs:
            option_type = spec["type"]
            value = current_options.get(spec["key"], spec["default"])
            if option_type == "int":
                editor = ModernSpinBox()
                editor.setRange(int(spec.get("minimum", -2147483648)), int(spec.get("maximum", 2147483647)))
                editor.setSingleStep(int(spec.get("step", 1)))
                if spec.get("suffix"):
                    editor.setSuffix(str(spec["suffix"]))
                editor.setValue(int(value))
                _show_spinbox_buttons(editor)
            elif option_type == "float":
                editor = ModernDoubleSpinBox()
                editor.setRange(float(spec.get("minimum", -1e12)), float(spec.get("maximum", 1e12)))
                editor.setDecimals(int(spec.get("decimals", 3)))
                editor.setSingleStep(float(spec.get("step", 0.1)))
                if spec.get("suffix"):
                    editor.setSuffix(str(spec["suffix"]))
                editor.setValue(float(value))
                _show_spinbox_buttons(editor)
            elif option_type == "choice":
                editor = ModernComboBox()
                editor.addItems([str(choice) for choice in spec.get("choices", ())])
                index = editor.findText(str(value))
                editor.setCurrentIndex(max(0, index))
            elif option_type == "float_list":
                editor = QLineEdit(_format_float_list(value))
            elif option_type == "bool":
                editor = SwitchButton("")
                editor.setChecked(bool(value))
            else:
                continue

            editors[spec["key"]] = editor
            form_layout.addRow(str(spec["label"]), editor)

        scroll_panel = QFrame()
        scroll_panel.setObjectName("advancedOptionsPanel")
        scroll_panel.setAttribute(WA_STYLED_BACKGROUND, True)
        scroll_panel_layout = QVBoxLayout(scroll_panel)
        scroll_panel_layout.setContentsMargins(1, 1, 1, 1)
        scroll_panel_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("advancedOptionsScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame if PYQT_VERSION == 6 else QFrame.NoFrame)
        scroll.viewport().setObjectName("advancedOptionsViewport")
        scroll.viewport().setAttribute(WA_STYLED_BACKGROUND, True)
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll_panel_layout.addWidget(scroll)
        dialog_layout.addWidget(scroll_panel)

        button_layout = QHBoxLayout()
        defaults_button = QPushButton("Defaults")
        _set_button_role(defaults_button, "secondary")
        _set_button_icon(defaults_button, SLIDERS_SVG)
        cancel_button = QPushButton("Cancel")
        _set_button_role(cancel_button, "quiet")
        apply_button = QPushButton("Apply")
        _set_button_role(apply_button, "primary")
        apply_button.setDefault(True)
        button_layout.addWidget(defaults_button)
        button_layout.addStretch(1)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(apply_button)
        dialog_layout.addLayout(button_layout)

        collected: dict[str, dict[str, Any]] = {}

        def reset_defaults() -> None:
            defaults = _advanced_defaults(method)
            for spec in specs:
                set_editor_value(editors[spec["key"]], spec, defaults[spec["key"]])

        def collect_options() -> dict[str, Any]:
            options: dict[str, Any] = {}
            for spec in specs:
                editor = editors[spec["key"]]
                option_type = spec["type"]
                if option_type == "int":
                    options[spec["key"]] = int(editor.value())
                elif option_type == "float":
                    options[spec["key"]] = float(editor.value())
                elif option_type == "choice":
                    options[spec["key"]] = str(editor.currentText())
                elif option_type == "float_list":
                    options[spec["key"]] = _parse_float_list(editor.text(), str(spec["label"]))
                elif option_type == "bool":
                    options[spec["key"]] = bool(editor.isChecked())
            return options

        def apply_options() -> None:
            try:
                collected["options"] = collect_options()
            except ValueError as exc:
                QMessageBox.warning(dialog, "Invalid Advanced Option", str(exc))
                return
            dialog.accept()

        defaults_button.clicked.connect(reset_defaults)
        cancel_button.clicked.connect(dialog.reject)
        apply_button.clicked.connect(apply_options)
        dialog.resize(460, 560)

        accepted = QDialog.DialogCode.Accepted if PYQT_VERSION == 6 else QDialog.Accepted
        if _dialog_exec(dialog) == accepted and "options" in collected:
            self.advanced_options_by_method[method] = collected["options"]
            self.save_all_traces_button.setEnabled(False)
            self.status_label.setText(f"Updated {method} advanced options; extract trace to apply")
            self._refresh_roi_inspector()

    def _current_extraction_channel(self) -> Optional[int]:
        if self.movie is not None and len(tuple(self.movie.shape)) == 4:
            return int(self.channel_spin.value())
        return None

    def _roi_mask_fingerprint(self, roi_id: int) -> Optional[tuple]:
        if self.roi_mask is None:
            return None

        labels = np.asarray(self.roi_mask)
        roi_id = int(roi_id)
        if labels.ndim == 3:
            plane_index = roi_id - 1
            if plane_index < 0 or plane_index >= int(labels.shape[0]):
                return None
            pixels = np.asarray(labels[plane_index] != 0, dtype=np.bool_)
        else:
            pixels = np.asarray(labels == roi_id, dtype=np.bool_)

        if pixels.ndim != 2 or not np.any(pixels):
            return None

        pixels = np.ascontiguousarray(pixels)
        packed = np.packbits(pixels.reshape(-1))
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
        digest.update(packed.tobytes())
        return (
            tuple(int(value) for value in pixels.shape),
            int(np.count_nonzero(pixels)),
            digest.hexdigest(),
        )

    def _current_roi_cache_scopes(self) -> set[tuple]:
        if self.movie_path is None or self.roi_mask is None:
            return set()

        dataset = self.dataset_edit.text().strip() or "movie"
        scopes = set()
        for roi_id in available_roi_ids(self.roi_mask):
            fingerprint = self._roi_mask_fingerprint(int(roi_id))
            if fingerprint is not None:
                scopes.add((self.movie_path, dataset, int(roi_id), fingerprint))
        return scopes

    def _prune_trace_cache_for_current_roi_mask(self) -> None:
        if self.movie_path is None:
            return

        dataset = self.dataset_edit.text().strip() or "movie"
        valid_scopes = self._current_roi_cache_scopes()
        for key in list(self.trace_cache):
            if len(key) < 4:
                continue
            if key[0] == self.movie_path and key[1] == dataset and key[:4] not in valid_scopes:
                del self.trace_cache[key]

    def _current_extraction_key(self, roi_id: Optional[int] = None) -> Optional[tuple]:
        if roi_id is None:
            roi_id = self.selected_roi_id
        if self.movie_path is None or roi_id is None:
            return None

        roi_id = int(roi_id)
        roi_fingerprint = self._roi_mask_fingerprint(roi_id)
        if roi_fingerprint is None:
            return None

        method = self.method_combo.currentText()
        key = (
            self.movie_path,
            self.dataset_edit.text().strip() or "movie",
            roi_id,
            roi_fingerprint,
            method,
            self._current_extraction_channel(),
            self._selected_device(),
            round(float(self.movie_frame_rate_spin.value()), 6),
            _freeze_options(self._advanced_options_for_method(method)),
        )
        if method == "Spikepursuit":
            key += (bool(self.flip_signal_check.isChecked()),)
        return key

    def _show_trace_result(self, result: dict, *, status: str) -> None:
        self.last_result = result
        self._update_trace_scope_buttons()
        self.trace_canvas.plot_result(result, mode=self.trace_view_mode)
        self._sync_trace_window_controls()
        self._show_trace_window()
        self.status_label.setText(status)
        self._refresh_roi_inspector()

    def _remove_stale_trace_cache_entries(self, cache_key: Optional[tuple]) -> None:
        if cache_key is None:
            return
        roi_scope = cache_key[:4]
        for key in list(self.trace_cache):
            if key != cache_key and key[:4] == roi_scope:
                del self.trace_cache[key]

    def extract_all_traces(self) -> None:
        """Extract traces for all available ROI labels."""
        if self.movie_path is None or self.roi_mask is None:
            QMessageBox.warning(self, "Missing ROI", "Run Cellpose or load an ROI mask before extracting all ROI traces.")
            return
        roi_ids = [int(roi_id) for roi_id in available_roi_ids(self.roi_mask)]
        if not roi_ids:
            QMessageBox.warning(self, "Missing ROI", "No ROI labels are available to extract.")
            return

        self._start_extraction(roi_ids, all_rois=True)

    def _start_extraction(self, roi_ids: list[int], *, all_rois: bool) -> None:
        self.timer.stop()
        self._set_playback_button_active(False)
        requested_roi_ids = [int(roi_id) for roi_id in roi_ids]
        cache_keys = {
            roi_id: self._current_extraction_key(roi_id)
            for roi_id in requested_roi_ids
        }
        cached_results = []
        missing_roi_ids = []
        for roi_id in requested_roi_ids:
            cache_key = cache_keys.get(roi_id)
            if cache_key is not None and cache_key in self.trace_cache:
                cached_results.append(self.trace_cache[cache_key])
            else:
                missing_roi_ids.append(roi_id)

        if not missing_roi_ids:
            self._set_last_batch_results(cached_results if all_rois else [])
            self.save_all_traces_button.setEnabled(all_rois)
            if all_rois:
                self.trace_display_scope = "all"
            result = self._cached_combined_trace_result(cached_results) if all_rois else (self._result_for_selected_roi(cached_results) or cached_results[0])
            self._show_trace_result(
                result,
                status=f"Loaded cached traces for {len(cached_results)} ROI(s)",
            )
            self.extraction_progress.setRange(0, 100)
            self.extraction_progress.setValue(100)
            self.extraction_progress.show()
            return

        method = self.method_combo.currentText()
        advanced_options = self._advanced_options_for_method(method)

        self._set_extraction_controls_enabled(False)
        self._set_segmentation_controls_enabled(False)
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_progress.show()
        if cached_results:
            self.extraction_status_label.setText(
                f"Starting extraction for {len(missing_roi_ids)} ROI(s); reusing {len(cached_results)} cached"
            )
        else:
            self.extraction_status_label.setText("Starting extraction")

        channel = self._current_extraction_channel()
        self.pending_extraction_keys = {
            roi_id: cache_keys[roi_id]
            for roi_id in missing_roi_ids
            if cache_keys.get(roi_id) is not None
        }
        self.pending_extraction_roi_ids = requested_roi_ids
        self.worker_thread = QThread(self)
        self.worker = ExtractionWorker(
            movie_path=self.movie_path,
            dataset=self.dataset_edit.text().strip() or "movie",
            method=method,
            roi_mask=self.roi_mask.copy(),
            roi_ids=missing_roi_ids,
            channel=channel,
            frame_rate=self.movie_frame_rate_spin.value(),
            device=self._selected_device(),
            flip_signal=self.flip_signal_check.isChecked(),
            advanced_options=advanced_options,
            all_rois=all_rois,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status.connect(self.extraction_status_label.setText)
        self.worker.progress.connect(self._extraction_progress)
        self.worker.finished.connect(self._extraction_finished)
        self.worker.failed.connect(self._extraction_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(
            lambda: self._set_extraction_controls_enabled(self.movie_path is not None)
        )
        self.worker_thread.finished.connect(
            lambda: self._set_segmentation_controls_enabled(self.movie_path is not None)
        )
        self.worker_thread.start()

    def _extraction_progress(self, value: int, message: str) -> None:
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(max(0, min(100, int(value))))
        self.extraction_status_label.setText(message)

    def _result_for_selected_roi(self, results: list[dict]) -> Optional[dict]:
        if self.selected_roi_id is None:
            return None
        for result in results:
            metrics = result.get("metrics") or {}
            if int(metrics.get("roi", -1)) == int(self.selected_roi_id):
                return result
        return None

    @staticmethod
    def _stack_result_field(results: list[dict], field: str) -> tuple[np.ndarray, list[str], list[int]]:
        columns = []
        labels = []
        roi_ids = []
        max_len = 0
        for result in results:
            arr = np.asarray(result.get(field, []), dtype=float)
            if arr.ndim == 0 or arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim != 2 or arr.shape[0] <= 0:
                continue
            metrics = result.get("metrics") or {}
            roi_id = int(metrics.get("roi", len(labels) + 1))
            max_len = max(max_len, int(arr.shape[0]))
            for column_index in range(arr.shape[1]):
                label = f"ROI {roi_id}" if arr.shape[1] == 1 else f"ROI {roi_id}.{column_index + 1}"
                columns.append(arr[:, column_index])
                labels.append(label)
                roi_ids.append(roi_id)

        if not columns or max_len <= 0:
            return np.empty((0, 0), dtype=float), [], []

        data = np.full((max_len, len(columns)), np.nan, dtype=float)
        for index, column in enumerate(columns):
            data[: column.shape[0], index] = column
        return data, labels, roi_ids

    @staticmethod
    def _stack_spike_trains_for_trace_columns(results: list[dict]) -> list[np.ndarray]:
        spike_trains: list[np.ndarray] = []
        for result in results:
            trace = np.asarray(result.get("trace", []), dtype=float)
            if trace.ndim == 0 or trace.size == 0:
                continue
            if trace.ndim == 1:
                column_count = 1
            else:
                column_count = int(trace.reshape(trace.shape[0], -1).shape[1])
            spikes = np.asarray(result.get("spikes", []), dtype=int).reshape(-1)
            for _ in range(column_count):
                spike_trains.append(spikes)
        return spike_trains

    def _combined_trace_result(self, results: list[dict]) -> dict:
        results = self._with_result_colors(results)
        trace, trace_labels, trace_roi_ids = self._stack_result_field(results, "trace")
        dff, dff_labels, dff_roi_ids = self._stack_result_field(results, "dff")
        reconstruction, reconstruction_labels, reconstruction_roi_ids = self._stack_result_field(results, "reconstruction")
        subthreshold, subthreshold_labels, subthreshold_roi_ids = self._stack_result_field(results, "subthreshold")
        spike_trains = self._stack_spike_trains_for_trace_columns(results)
        first = results[0] if results else {}
        method = str(first.get("method", self.method_combo.currentText()))
        frame_rate = float(first.get("frame_rate", self.movie_frame_rate_spin.value()) or self.movie_frame_rate_spin.value())
        rois = len(results)
        combined = {
            "method": method,
            "multi_roi": True,
            "trace": trace,
            "trace_labels": trace_labels,
            "trace_roi_ids": trace_roi_ids,
            "trace_title": f"All ROI Traces ({rois})",
            "spikes": np.array([], dtype=np.int64),
            "spike_trains": spike_trains,
            "metrics": {"rois": rois},
            "frame_rate": frame_rate,
        }
        if dff.size:
            combined["dff"] = dff
            combined["dff_labels"] = dff_labels
            combined["dff_roi_ids"] = dff_roi_ids
            combined["dff_title"] = f"All ROI dF/F ({rois})"
        if reconstruction.size:
            combined["reconstruction"] = reconstruction
            combined["reconstruction_labels"] = reconstruction_labels
            combined["reconstruction_roi_ids"] = reconstruction_roi_ids
            combined["reconstruction_title"] = f"All ROI Reconstruction ({rois})"
        if subthreshold.size:
            combined["subthreshold"] = subthreshold
            combined["subthreshold_labels"] = subthreshold_labels
            combined["subthreshold_roi_ids"] = subthreshold_roi_ids
        return combined

    def _extraction_finished(self, payload: dict) -> None:
        results = list(payload.get("results", [])) if "results" in payload else [payload]
        extracted_count = len(results)
        requested_roi_ids = list(self.pending_extraction_roi_ids)
        for result in results:
            metrics = result.get("metrics") or {}
            roi_id = int(metrics.get("roi", -1))
            cache_key = self.pending_extraction_keys.get(roi_id)
            if cache_key is not None:
                self._remove_stale_trace_cache_entries(cache_key)
                self.trace_cache[cache_key] = result

        all_rois = bool(payload.get("all_rois"))
        display_results = results
        if all_rois and requested_roi_ids:
            cached_results = self._cached_results_for_roi_ids(requested_roi_ids)
            if cached_results:
                display_results = cached_results

        self.pending_extraction_keys = {}
        self.pending_extraction_roi_ids = []
        self._set_last_batch_results(display_results if all_rois else [])
        self.save_all_traces_button.setEnabled(bool(self.last_batch_results))
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(100)
        if not display_results:
            self.extraction_status_label.setText("No traces returned")
            return
        if all_rois:
            self.trace_display_scope = "all"
        result = self._cached_combined_trace_result(display_results) if all_rois else (self._result_for_selected_roi(display_results) or display_results[0])
        reused_count = max(0, len(display_results) - extracted_count) if all_rois else 0
        if reused_count:
            status = f"Extracted {extracted_count} ROI trace(s), reused {reused_count} cached"
        else:
            status = f"Extracted {len(display_results)} ROI trace(s)"
        self._show_trace_result(result, status=status)

    def _extraction_failed(self, message: str) -> None:
        self.pending_extraction_keys = {}
        self.pending_extraction_roi_ids = []
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_status_label.setText("Extraction failed")
        self._refresh_roi_inspector()
        QMessageBox.critical(self, "Extraction Failed", message)

    def _cached_results_for_roi_ids(self, roi_ids: list[int]) -> list[dict]:
        results = []
        for roi_id in roi_ids:
            cache_key = self._current_extraction_key(int(roi_id))
            if cache_key is None or cache_key not in self.trace_cache:
                return []
            results.append(self.trace_cache[cache_key])
        return results

    def _cached_results_for_all_rois(self) -> list[dict]:
        if self.roi_mask is None:
            return []
        return self._cached_results_for_roi_ids([int(roi_id) for roi_id in available_roi_ids(self.roi_mask)])

    @staticmethod
    def _csv_columns_for_results(results: list[dict]) -> tuple[np.ndarray, list[str]]:
        columns = []
        headers = []
        max_len = 0
        frame_rate = 1.0
        for result in results:
            frame_rate = float(result.get("frame_rate", frame_rate) or frame_rate)
            metrics = result.get("metrics") or {}
            roi_id = int(metrics.get("roi", len(headers) + 1))
            trace = np.asarray(result.get("trace", []), dtype=float)
            if trace.ndim == 1:
                trace = trace[:, None]
            elif trace.ndim > 2:
                trace = trace.reshape(trace.shape[0], -1)
            if trace.ndim != 2 or trace.shape[0] == 0:
                continue
            max_len = max(max_len, int(trace.shape[0]))
            for column_index in range(trace.shape[1]):
                label = f"roi_{roi_id}" if trace.shape[1] == 1 else f"roi_{roi_id}_trace_{column_index + 1}"
                columns.append(trace[:, column_index])
                headers.append(label)

        if not columns or max_len <= 0:
            return np.empty((0, 0), dtype=float), []

        data = np.full((max_len, len(columns) + 1), np.nan, dtype=float)
        data[:, 0] = np.arange(max_len, dtype=float) / max(1e-12, frame_rate)
        for index, column in enumerate(columns, start=1):
            data[: column.shape[0], index] = column
        return data, ["time_s", *headers]

    def save_all_traces_dialog(self) -> None:
        """Prompt for a path and save all extracted traces."""
        results = self._cached_results_for_all_rois()
        if not results:
            QMessageBox.warning(self, "Save All Failed", "Extract all ROIs with the current method/options before saving all traces.")
            self.save_all_traces_button.setEnabled(False)
            return

        data, headers = self._csv_columns_for_results(results)
        if data.size == 0:
            QMessageBox.warning(self, "Save All Failed", "No trace data is available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save All Traces",
            "all_traces.csv",
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return

        np.savetxt(path, data, delimiter=",", header=",".join(headers), comments="")
        self.extraction_status_label.setText(f"Saved all traces to {Path(path).name}")

    def closeEvent(self, event) -> None:
        """Clean up movie handles and worker state before closing."""
        self.timer.stop()
        if self.movie is not None and hasattr(self.movie, "close"):
            self.movie.close()
        super().closeEvent(event)


def main(argv: Optional[list[str]] = None) -> int:
    """Start the torch-volpy Qt application."""
    qt_argv = _configure_qt_message_logging(list(sys.argv if argv is None else argv))
    app = QApplication(qt_argv)
    _apply_app_theme(app)
    window = MainWindow()
    window.show()
    return app.exec() if PYQT_VERSION == 6 else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
