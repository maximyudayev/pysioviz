"""
Annotation Timeline Dock Widget.
Multi-track timeline with draggable activity masks across Act 0/1/2 tracks,
double-click activation to prevent accidental movement during scrubbing,
mask selection & deletion, horizontal-only zooming, and relative time axis.
"""

from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Optional
import numpy as np
import h5py

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.qt.time_axis_item import RelativeTimeAxisItem
from pysioviz.utils.time_utils import format_relative_time


# ==============================================================================
# SCRUBBER SNAPPING CONFIGURATION
# Tune this threshold (in seconds) to adjust how close a dragged edge must be
# to snap magnetically to the playback scrubber line.
# ==============================================================================
SCRUBBER_SNAP_THRESHOLD_S: float = 0.1  # In seconds (e.g. 0.25, 0.5, 1.0)


# Default activity classes from pysioviz
# TODO: replace with a file that specifies all the transitions
DEFAULT_ACTIVITIES = [
    'Transition',
    'Standing',
    'Walking',
    'Sitting',
    'Sitting Down',
    'Standing Up',
    'Stair Ascent',
    'Stair Descent',
    'Slope Ascent',
    'Slope Descent',
    'Step Over',
    'Cross Country',
    'Box Pickup',
    'Box Putdown',
    'Slalom',
    'Slalom Left Turn',
    'Slalom Right Turn',
    'Standing Turn Left',
    'Standing Turn Right',
    'Radius Turn Left',
    'Radius Turn Right',
]

COLOR_PALETTE = [
    '#38bdf8',  # Sky Blue
    '#4ade80',  # Green
    '#fbbf24',  # Amber
    '#f87171',  # Red
    '#c084fc',  # Purple
    '#fb923c',  # Orange
    '#2dd4bf',  # Teal
    '#f472b6',  # Pink
    '#a3e635',  # Lime
    '#818cf8',  # Indigo
]


class ActivityMaskItem(pg.GraphicsObject):
    """Interactive activity segmentation mask confined to GT 0/1/2 tracks.

    Single-click selects and highlights for inspection/deletion.
    Double-click activates dragging (horizontally in time and vertically between tracks GT 0/1/2).
    """

    sigSelected = QtCore.pyqtSignal(object)
    sigChanged = QtCore.pyqtSignal(object)
    sigDeleted = QtCore.pyqtSignal(object)

    def __init__(
        self,
        label: str,
        start_rel_s: float,
        end_rel_s: float,
        row: int = 0,
        color: str = '#38bdf8',
        scrubber_getter: Optional[Callable[[], float]] = None,
        snap_threshold_s: float = SCRUBBER_SNAP_THRESHOLD_S,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__()
        self.label = label
        self.start_rel_s = max(0.0, float(start_rel_s))
        self.end_rel_s = max(self.start_rel_s + 0.1, float(end_rel_s))
        self.row = max(0, min(2, int(row)))
        self.color = QtGui.QColor(color)
        self.scrubber_getter = scrubber_getter
        self.snap_threshold_s = float(snap_threshold_s)
        self.is_active = False  # True when double-clicked for dragging
        self.is_selected = False  # True when selected for inspection

        self._drag_mode: Optional[str] = None
        self._drag_start_pos: Optional[QtCore.QPointF] = None
        self._orig_start = self.start_rel_s
        self._orig_end = self.end_rel_s
        self._orig_row = self.row
        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QtCore.QRectF:
        """Data coordinates bounding rect."""
        w = max(0.05, self.end_rel_s - self.start_rel_s)
        return QtCore.QRectF(self.start_rel_s, self.row - 0.35, w, 0.70)

    def set_active(self, active: bool):
        if self.is_active != active:
            self.is_active = active
            if active:
                self.is_selected = True
            self.prepareGeometryChange()
            self.update()
            if active:
                self.sigSelected.emit(self)

    def set_selected(self, selected: bool):
        if self.is_selected != selected or (not selected and self.is_active):
            self.is_selected = selected
            if not selected:
                self.is_active = False
            self.prepareGeometryChange()
            self.update()

    def mousePressEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.set_selected(True)
            self.sigSelected.emit(self)
            if self.is_active:
                rect = self.boundingRect()
                pos = ev.pos()
                margin = max(0.1, rect.width() * 0.10)
                if abs(pos.x() - rect.left()) < margin:
                    self._drag_mode = 'left'
                elif abs(pos.x() - rect.right()) < margin:
                    self._drag_mode = 'right'
                else:
                    self._drag_mode = 'move'
                self._drag_start_pos = pos
                self._orig_start = self.start_rel_s
                self._orig_end = self.end_rel_s
                self._orig_row = self.row
        else:
            super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.set_active(True)
            rect = self.boundingRect()
            pos = ev.pos()
            margin = max(0.1, rect.width() * 0.10)
            if abs(pos.x() - rect.left()) < margin:
                self._drag_mode = 'left'
            elif abs(pos.x() - rect.right()) < margin:
                self._drag_mode = 'right'
            else:
                self._drag_mode = 'move'
            self._drag_start_pos = pos
            self._orig_start = self.start_rel_s
            self._orig_end = self.end_rel_s
            self._orig_row = self.row
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)

    def mouseMoveEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        if self.is_active and self._drag_mode is not None and self._drag_start_pos is not None:
            delta = ev.pos() - self._drag_start_pos
            self.prepareGeometryChange()
            if self._drag_mode == 'move':
                dx = delta.x()
                dy = delta.y()
                self.start_rel_s = max(0.0, self._orig_start + dx)
                self.end_rel_s = max(self.start_rel_s + 0.1, self._orig_end + dx)
                new_row = round(self._orig_row + dy)
                self.row = max(0, min(2, new_row))
            elif self._drag_mode == 'left':
                new_start = self._orig_start + delta.x()
                # Magnetic snapping to playback scrubber line
                if self.scrubber_getter is not None and self.snap_threshold_s > 0:
                    scrubber_pos = self.scrubber_getter()
                    if abs(new_start - scrubber_pos) <= self.snap_threshold_s and scrubber_pos < self.end_rel_s:
                        new_start = scrubber_pos
                self.start_rel_s = min(max(0.0, new_start), self.end_rel_s - 0.1)
            elif self._drag_mode == 'right':
                new_end = self._orig_end + delta.x()
                # Magnetic snapping to playback scrubber line
                if self.scrubber_getter is not None and self.snap_threshold_s > 0:
                    scrubber_pos = self.scrubber_getter()
                    if abs(new_end - scrubber_pos) <= self.snap_threshold_s and scrubber_pos > self.start_rel_s:
                        new_end = scrubber_pos
                self.end_rel_s = max(new_end, self.start_rel_s + 0.1)
            self.update()
            self.sigChanged.emit(self)
            ev.accept()
        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
        if self._drag_mode is not None:
            self._drag_mode = None
            self._drag_start_pos = None
            self.sigChanged.emit(self)
        ev.accept()

    def paint(self, painter: QtGui.QPainter, option, widget):
        rect = self.boundingRect()
        fill_color = QtGui.QColor(self.color)
        fill_color.setAlpha(180 if self.is_active else (135 if self.is_selected else 90))
        painter.fillRect(rect, fill_color)

        pen = QtGui.QPen()
        pen.setCosmetic(True)
        if self.is_active:
            pen.setColor(QtGui.QColor('#ffffff'))
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        elif self.is_selected:
            pen.setColor(QtGui.QColor('#ffffff'))
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        else:
            pen.setColor(QtGui.QColor(self.color).lighter(125))
            pen.setWidth(1)
            pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(rect)

        # Draw Label Text in device pixel coordinates to avoid anisotropic scaling artifacts
        painter.save()
        rect_px = painter.transform().mapRect(rect)
        painter.resetTransform()
        if rect_px.width() > 16 and rect_px.height() > 8:
            painter.setPen(QtGui.QColor('#ffffff'))
            font = QtGui.QFont('Segoe UI', 8)
            font.setBold(True)
            painter.setFont(font)
            text_rect = rect_px.adjusted(6, 2, -6, -2)
            if self.is_active:
                time_str = f' [{format_relative_time(self.start_rel_s)} - {format_relative_time(self.end_rel_s)}]'
                full_text = f'{self.label}{time_str}'
            else:
                full_text = self.label
            fm = QtGui.QFontMetrics(font)
            elided = fm.elidedText(full_text, QtCore.Qt.TextElideMode.ElideRight, int(max(10, text_rect.width())))
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided,
            )
        painter.restore()


@dataclass
class ActivityInterval:
    label: str
    start_toa_s: float
    end_toa_s: float
    row: int
    mask_item: Optional[ActivityMaskItem] = None


class AnnotationTimelineDock(QtWidgets.QDockWidget):
    """Spanning bottom timeline dock widget with multi-track masks and controls."""

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('Annotation Timeline Playback', parent)
        self.setObjectName('AnnotationTimelineDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)

        self._intervals: List[ActivityInterval] = []
        self._selected_mask: Optional[ActivityMaskItem] = None
        self._label_colors: Dict[str, str] = {}
        for i, act in enumerate(DEFAULT_ACTIVITIES):
            self._label_colors[act] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        self._is_dragging_scrubber = False

        self._build_ui()

        # Connect playback engine signals
        self.playback_engine.time_changed.connect(self._on_time_changed)
        self.playback_engine.range_changed.connect(self._on_range_changed)
        self.playback_engine.playback_state_changed.connect(self._on_playback_state_changed)
        self.playback_engine.speed_changed.connect(self._on_speed_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(1200, 205)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(350, 180)

    def _build_ui(self):
        container = QtWidgets.QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #0f1016;
                color: #f1f5f9;
            }
            QPushButton {
                background-color: #1e2230;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #38bdf8;
            }
            QComboBox {
                background-color: #1e2230;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        # 1. Timeline Canvas (PyQtGraph) with relative time formatting
        self.time_axis = RelativeTimeAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self.time_axis})
        self.plot_widget.setBackground('#12131c')
        self.plot_widget.setMinimumHeight(125)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        # self.plot_widget.setLabel('bottom', 'Elapsed Experiment Time (HH:MM:SS.mmm)')

        # Lock vertical zooming / scrubbing by default; mouse wheel zooms horizontally only
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.getViewBox().setMouseEnabled(x=True, y=False)
        self.plot_widget.getAxis('left').setTicks([[(0, 'GT 0'), (1, 'GT 1'), (2, 'GT 2')]])
        self.plot_widget.setYRange(-0.5, 2.5, padding=0.0)  # pyrefly: ignore [bad-keyword-argument]
        self.plot_widget.setLimits(yMin=-0.5, yMax=2.5)

        # Track divider horizontal dashed lines
        line_track1 = pg.InfiniteLine(
            pos=0.5, angle=0, pen=pg.mkPen(color='#334155', width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        line_track2 = pg.InfiniteLine(
            pos=1.5, angle=0, pen=pg.mkPen(color='#334155', width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(line_track1)
        self.plot_widget.addItem(line_track2)

        # Interactive Scrubber Line
        self.scrubber_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen(color='#38bdf8', width=2.5),
            hoverPen=pg.mkPen(color='#7dd3fc', width=3.5),
        )
        self.scrubber_line.sigDragged.connect(self._on_scrubber_dragged)
        self.scrubber_line.sigPositionChangeFinished.connect(self._on_scrubber_released)
        self.plot_widget.addItem(self.scrubber_line)

        # Connect canvas clicks to clear active selection when clicking outside
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

        layout.addWidget(self.plot_widget, stretch=1)

        # 2. Status banner for selected mask
        self.selection_banner = QtWidgets.QLabel('Double-click any mask to select, drag across tracks, or edit.')
        self.selection_banner.setStyleSheet('color: #94a3b8; font-size: 10px; padding-left: 4px;')
        layout.addWidget(self.selection_banner)

        # 3. Control Toolbar
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(4, 2, 4, 2)
        controls_layout.setSpacing(6)

        # Play / Pause
        self.btn_play = QtWidgets.QPushButton('▶ Play')
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: white;
                font-weight: 700;
                padding: 4px 14px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.btn_play.clicked.connect(self.playback_engine.toggle_play)
        controls_layout.addWidget(self.btn_play)

        # Step Back / Step Forward buttons
        self.btn_step_back_10 = QtWidgets.QPushButton('⏮ -10f')
        self.btn_step_back_10.clicked.connect(lambda: self.playback_engine.step_backward(10))
        controls_layout.addWidget(self.btn_step_back_10)

        self.btn_step_back = QtWidgets.QPushButton('◀ -1f')
        self.btn_step_back.clicked.connect(lambda: self.playback_engine.step_backward(1))
        controls_layout.addWidget(self.btn_step_back)

        self.btn_step_fwd = QtWidgets.QPushButton('+1f ▶')
        self.btn_step_fwd.clicked.connect(lambda: self.playback_engine.step_forward(1))
        controls_layout.addWidget(self.btn_step_fwd)

        self.btn_step_fwd_10 = QtWidgets.QPushButton('+10f ⏭')
        self.btn_step_fwd_10.clicked.connect(lambda: self.playback_engine.step_forward(10))
        controls_layout.addWidget(self.btn_step_fwd_10)

        # Speed Selector
        speed_label = QtWidgets.QLabel('Speed:')
        speed_label.setStyleSheet('color: #94a3b8; font-size: 11px;')
        controls_layout.addWidget(speed_label)

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(['0.25x', '0.5x', '1.0x', '1.5x', '2.0x', '4.0x'])
        self.speed_combo.setCurrentText('1.0x')
        self.speed_combo.currentTextChanged.connect(self._on_speed_combo_changed)
        controls_layout.addWidget(self.speed_combo)

        # Current Timestamp & Time display
        self.time_display = QtWidgets.QLabel('00:00:00.000')
        self.time_display.setStyleSheet("""
            QLabel {
                font-family: Consolas, monospace;
                font-weight: 700;
                font-size: 11px;
                color: #38bdf8;
                background-color: #161822;
                border: 1px solid #282a36;
                border-radius: 4px;
                padding: 3px 8px;
            }
        """)
        controls_layout.addWidget(self.time_display)

        controls_layout.addStretch()

        # Activity Annotation Creator
        act_label = QtWidgets.QLabel('Activity:')
        act_label.setStyleSheet('color: #94a3b8; font-size: 11px;')
        controls_layout.addWidget(act_label)

        self.activity_combo = QtWidgets.QComboBox()
        self.activity_combo.addItems(DEFAULT_ACTIVITIES)
        controls_layout.addWidget(self.activity_combo)

        track_label = QtWidgets.QLabel('Track:')
        track_label.setStyleSheet('color: #94a3b8; font-size: 11px;')
        controls_layout.addWidget(track_label)

        self.track_combo = QtWidgets.QComboBox()
        self.track_combo.addItems(['GT 0', 'GT 1', 'GT 2'])
        controls_layout.addWidget(self.track_combo)

        self.btn_add_interval = QtWidgets.QPushButton('➕ Add Mask')
        self.btn_add_interval.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                font-weight: bold;
                padding: 3px 10px;
            }
            QPushButton:hover { background-color: #059669; }
        """)
        self.btn_add_interval.clicked.connect(self._on_add_interval)
        controls_layout.addWidget(self.btn_add_interval)

        # Delete Selected Mask Button
        self.btn_delete_mask = QtWidgets.QPushButton('🗑️ Delete Mask')
        self.btn_delete_mask.setEnabled(False)
        self.btn_delete_mask.setStyleSheet("""
            QPushButton {
                background-color: #7f1d1d;
                color: #fca5a5;
                font-weight: bold;
                padding: 3px 10px;
            }
            QPushButton:hover { background-color: #991b1b; color: white; }
            QPushButton:disabled { background-color: #1e2230; color: #475569; border-color: #334155; }
        """)
        self.btn_delete_mask.clicked.connect(self._on_delete_selected_mask)
        controls_layout.addWidget(self.btn_delete_mask)

        # Save / Load Annotations
        self.btn_save = QtWidgets.QPushButton('💾 Save')
        self.btn_save.clicked.connect(self._on_save_hdf5)
        controls_layout.addWidget(self.btn_save)

        self.btn_load = QtWidgets.QPushButton('📂 Load')
        self.btn_load.clicked.connect(self._on_load_hdf5)
        controls_layout.addWidget(self.btn_load)

        layout.addLayout(controls_layout)
        self.setWidget(container)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Allow deleting selected mask via Delete or Backspace key."""
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self._selected_mask is not None:
                self._on_delete_selected_mask()
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_scene_clicked(self, event):
        """Deselect active mask on empty space click, or jump scrubber on double-click."""
        pos = event.scenePos()
        vb = self.plot_widget.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return

        items = self.plot_widget.scene().items(pos)
        has_mask = any(isinstance(it, ActivityMaskItem) for it in items)

        if not has_mask:
            # Deselect active mask
            if self._selected_mask is not None:
                self._selected_mask.set_selected(False)
                self._selected_mask.set_active(False)
                self._selected_mask = None
                self.btn_delete_mask.setEnabled(False)
                self.selection_banner.setText('Double-click any mask to select, drag across tracks, or edit.')

            # Jump scrubber on double-clicking empty timeline space
            if event.double() and event.button() == QtCore.Qt.MouseButton.LeftButton:
                view_pos = vb.mapSceneToView(pos)
                target_rel_s = view_pos.x()
                duration_s = max(0.0, self.playback_engine.max_toa_s - self.playback_engine.min_toa_s)
                clamped_rel_s = max(0.0, min(duration_s, float(target_rel_s)))
                target_toa_s = self.playback_engine.min_toa_s + clamped_rel_s

                self.scrubber_line.setValue(clamped_rel_s)
                self.playback_engine.set_time(target_toa_s)
                event.accept()

    def _on_mask_selected(self, mask: ActivityMaskItem):
        """Handle mask activation and display its boundary parameters."""
        if self._selected_mask is not None and self._selected_mask != mask:
            self._selected_mask.set_selected(False)
            self._selected_mask.set_active(False)

        self._selected_mask = mask
        mask.set_selected(True)
        self.btn_delete_mask.setEnabled(True)
        dur = mask.end_rel_s - mask.start_rel_s
        edit_tag = ' [ACTIVE DRAG]' if mask.is_active else ''
        self.selection_banner.setText(
            f'Selected: {mask.label} [Track GT {mask.row}]{edit_tag} | '
            f'Start: {format_relative_time(mask.start_rel_s)} | '
            f'End: {format_relative_time(mask.end_rel_s)} | '
            f'Duration: {dur:.3f}s'
        )

    def _on_mask_changed(self, mask: ActivityMaskItem):
        """Update synchronized ActivityInterval records when mask moves or changes tracks."""
        for inv in self._intervals:
            if inv.mask_item == mask:
                inv.start_toa_s = self.playback_engine.min_toa_s + mask.start_rel_s
                inv.end_toa_s = self.playback_engine.min_toa_s + mask.end_rel_s
                inv.row = mask.row
                break
        self._on_mask_selected(mask)

    def _on_delete_selected_mask(self):
        """Delete currently active annotation mask."""
        if self._selected_mask is not None:
            mask = self._selected_mask
            self.plot_widget.removeItem(mask)
            self._intervals = [inv for inv in self._intervals if inv.mask_item != mask]
            self._selected_mask = None
            self.btn_delete_mask.setEnabled(False)
            self.selection_banner.setText('Annotation mask deleted.')

    def _on_scrubber_dragged(self):
        """Throttled sweeping callback during active mouse scrubbing."""
        if not self._is_dragging_scrubber:
            self._is_dragging_scrubber = True
            self.playback_engine.start_scrubbing()

        rel_pos = self.scrubber_line.value()
        target_toa_s = self.playback_engine.min_toa_s + rel_pos
        self.playback_engine.scrub_to(target_toa_s)

    def _on_scrubber_released(self):
        """Called when user releases scrubber line."""
        self._is_dragging_scrubber = False
        rel_pos = self.scrubber_line.value()
        target_toa_s = self.playback_engine.min_toa_s + rel_pos
        self.playback_engine.finish_scrubbing(target_toa_s)

    def _on_time_changed(self, current_toa_s: float):
        """Update scrubber position and readout with relative time."""
        t0 = self.playback_engine.min_toa_s
        rel_s = max(0.0, current_toa_s - t0) if t0 > 0 else 0.0

        if not self._is_dragging_scrubber:
            self.scrubber_line.setValue(rel_s)

        time_str = format_relative_time(rel_s)
        self.time_display.setText(f'{time_str} (ToA: {current_toa_s:.3f}s)')

    def _on_range_changed(self, min_toa_s: float, max_toa_s: float):
        """Update timeline X limits relative to experiment start time (t0 = 0.0)."""
        duration_s = max(1.0, max_toa_s - min_toa_s)
        self.plot_widget.setXRange(0.0, duration_s, padding=0.01)  # pyrefly: ignore [bad-keyword-argument]
        self.plot_widget.setLimits(xMin=0.0, xMax=duration_s)

    def _on_playback_state_changed(self, is_playing: bool):
        self.btn_play.setText('⏸ Pause' if is_playing else '▶ Play')
        if is_playing:
            self.btn_play.setStyleSheet("""
                QPushButton { background-color: #f59e0b; color: white; font-weight: 700; padding: 4px 14px; }
                QPushButton:hover { background-color: #d97706; }
            """)
        else:
            self.btn_play.setStyleSheet("""
                QPushButton { background-color: #0284c7; color: white; font-weight: 700; padding: 4px 14px; }
                QPushButton:hover { background-color: #0369a1; }
            """)

    def _on_speed_changed(self, speed: float):
        text = f'{speed:.2f}x'
        self.speed_combo.setCurrentText(text)

    def _on_speed_combo_changed(self, text: str):
        try:
            val = float(text.replace('x', '').strip())
            self.playback_engine.set_speed(val)
        except ValueError:
            pass

    def _on_add_interval(self):
        """Add a draggable, colored activity interval mask at current playback position."""
        label = self.activity_combo.currentText()
        row_str = self.track_combo.currentText()
        row = int(row_str.replace('GT ', '').strip())

        cur_t = self.playback_engine.current_toa_s
        start_toa_s = cur_t
        if self.playback_engine.max_toa_s > cur_t:
            end_toa_s = min(self.playback_engine.max_toa_s, cur_t + 2.0)
        else:
            end_toa_s = cur_t + 2.0

        inv = self.add_activity_interval(label, start_toa_s, end_toa_s, row)
        if inv.mask_item is not None:
            self._on_mask_selected(inv.mask_item)

    def add_activity_interval(
        self,
        label: str,
        start_toa_s: float,
        end_toa_s: float,
        row: int = 0,
    ) -> ActivityInterval:
        """Create and place a track-confined ActivityMaskItem."""
        color_hex = self._label_colors.get(label, '#38bdf8')
        t0 = self.playback_engine.min_toa_s
        start_rel_s = max(0.0, start_toa_s - t0) if t0 > 0 else 0.0
        end_rel_s = max(start_rel_s + 0.1, end_toa_s - t0) if t0 > 0 else (start_rel_s + 2.0)

        mask_item = ActivityMaskItem(
            label=label,
            start_rel_s=start_rel_s,
            end_rel_s=end_rel_s,
            row=row,
            color=color_hex,
            scrubber_getter=lambda: float(self.scrubber_line.value()),
            snap_threshold_s=SCRUBBER_SNAP_THRESHOLD_S,
        )
        mask_item.sigSelected.connect(self._on_mask_selected)
        mask_item.sigChanged.connect(self._on_mask_changed)

        interval = ActivityInterval(
            label=label,
            start_toa_s=start_toa_s,
            end_toa_s=end_toa_s,
            row=row,
            mask_item=mask_item,
        )

        self.plot_widget.addItem(mask_item)
        self._intervals.append(interval)
        return interval

    def _on_save_hdf5(self):
        """Save annotations in structured HDF5 format matching HERMES specification."""
        if not self._intervals:
            QtWidgets.QMessageBox.warning(self, 'No Annotations', 'There are no annotation masks to save.')
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Annotations HDF5', 'annotations.hdf5', 'HDF5 Files (*.hdf5)'
        )
        if not file_path:
            return

        try:
            sorted_intervals = sorted(self._intervals, key=lambda x: x.start_toa_s)
            ann_dtype = np.dtype(
                [
                    ('label', 'S50'),
                    ('task_start', np.float64),
                    ('task_end', np.float64),
                    ('duration', np.float64),
                    ('track', np.int32),
                ]
            )

            ann_array = np.zeros(len(sorted_intervals), dtype=ann_dtype)
            for idx, ann in enumerate(sorted_intervals):
                ann_array[idx]['label'] = ann.label.encode('utf-8')
                ann_array[idx]['task_start'] = float(ann.start_toa_s)
                ann_array[idx]['task_end'] = float(ann.end_toa_s)
                ann_array[idx]['duration'] = float(ann.end_toa_s - ann.start_toa_s)
                ann_array[idx]['track'] = int(ann.row)

            with h5py.File(file_path, 'w') as hdf5:
                ds = hdf5.create_dataset('annotations', data=ann_array)
                ds.attrs['total_annotations'] = len(sorted_intervals)
                ds.attrs['saved_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

            QtWidgets.QMessageBox.information(
                self, 'Saved', f'Successfully saved {len(sorted_intervals)} annotations to {file_path}.'
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Save Error', f'Failed to save annotations: {e}')

    def _on_load_hdf5(self):
        """Load annotations from existing HDF5 file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Annotations HDF5', '', 'HDF5 Files (*.hdf5)')
        if not file_path:
            return

        try:
            with h5py.File(file_path, 'r') as hdf5:
                if 'annotations' not in hdf5:
                    QtWidgets.QMessageBox.warning(self, 'Invalid File', "No 'annotations' dataset found.")
                    return

                data = hdf5['annotations'][:]

                # Clear existing
                for inv in self._intervals:
                    if inv.mask_item is not None:
                        self.plot_widget.removeItem(inv.mask_item)
                self._intervals.clear()
                self._selected_mask = None
                self.btn_delete_mask.setEnabled(False)

                for i, row_data in enumerate(data):
                    label = (
                        row_data['label'].decode('utf-8')
                        if isinstance(row_data['label'], bytes)
                        else str(row_data['label'])
                    )
                    s = float(row_data['task_start_start'])
                    e = float(row_data['task_end_end'])
                    r = int(row_data['row']) if 'row' in row_data.dtype.names else (i % 3)
                    self.add_activity_interval(label, s, e, row=r)

            QtWidgets.QMessageBox.information(self, 'Loaded', f'Successfully loaded {len(data)} annotations.')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Load Error', f'Failed to load annotations: {e}')
