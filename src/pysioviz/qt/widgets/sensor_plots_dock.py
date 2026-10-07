"""
Sensor Modalities Dock Widget displaying a vertical stack of PyQtGraph plot widgets
with lazy-loaded HDF5 paging, relative time axis (HH:MM:SS.mmm), and sensor-specific ToA readouts.
"""

from typing import List, Optional
import numpy as np

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.qt.paged_sensor_reader import PagedSensorReader
from pysioviz.qt.time_axis_item import RelativeTimeAxisItem
from pysioviz.utils.time_utils import format_relative_time


class SensorPlotItem(QtCore.QObject):
    """Wrapper around a single pg.PlotWidget managing its curves, cursor, and paged reader."""

    CHANNEL_COLORS = ['#ff595e', '#8ac926', '#1982c4', '#ffd166', '#a855f7', '#06d6a0']

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        title: str,
        y_label: str,
        reader: PagedSensorReader,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.plot_widget = plot_widget
        self.title = title
        self.y_label = y_label
        self.reader = reader

        num_channels = reader.num_channels
        self.channel_names = channel_names or [f'Ch {i}' for i in range(num_channels)]

        # Configure plot appearance matching AidWear styling
        self.plot_widget.setBackground('#12131c')
        self._update_title_display(self.title)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.22)
        self.plot_widget.setLabel('bottom', 'Time (HH:MM:SS.mmm)')
        self.plot_widget.setLabel('left', self.y_label)

        leg = self.plot_widget.addLegend(offset=(5, 5))
        leg.setBrush(pg.mkBrush('#181922c0'))
        leg.setPen(pg.mkPen('#334155'))

        # Create plot curves
        self.curves: List[pg.PlotCurveItem] = []
        for i in range(num_channels):
            color = self.CHANNEL_COLORS[i % len(self.CHANNEL_COLORS)]
            pen = pg.mkPen(color=color, width=1.8)
            name = self.channel_names[i] if i < len(self.channel_names) else f'Ch {i}'
            curve = self.plot_widget.plot(name=name, pen=pen)
            self.curves.append(curve)

        # Scrubber cursor line
        self.cursor_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color='#38bdf8', width=1.8, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.cursor_line)

    def _update_title_display(self, title_text: str, subtitle: str = ''):
        sub_html = (
            f" <span style='font-size: 8pt; color: #94a3b8; font-family: Consolas;'>{subtitle}</span>"
            if subtitle
            else ''
        )
        self.plot_widget.setTitle(
            f"<span style='font-size: 10pt; font-weight: 700; color: #f8fafc;'>{title_text}</span>{sub_html}"
        )

    def update_view(self, center_toa_s: float, window_duration_s: float, experiment_start_toa_s: float = 0.0):
        """Fetch temporal slice from paged reader and update curves with relative time and sensor-specific ToA."""
        toas, vals = self.reader.get_data_window(center_toa_s, window_duration_s)

        t0 = experiment_start_toa_s
        center_rel_s = max(0.0, center_toa_s - t0) if t0 > 0 else center_toa_s

        if toas.size > 0:
            toas_rel = toas - t0 if t0 > 0 else toas
            for ch in range(min(len(self.curves), vals.shape[1])):
                self.curves[ch].setData(toas_rel, vals[:, ch])

            # Find nearest sample in this specific sensor modality
            idx = int(np.searchsorted(toas, center_toa_s))
            if idx >= len(toas):
                idx = len(toas) - 1
            if idx > 0 and abs(toas[idx - 1] - center_toa_s) < abs(toas[idx] - center_toa_s):
                idx = idx - 1

            modality_toa = float(toas[idx])
            modality_rel = modality_toa - t0 if t0 > 0 else 0.0
            time_str = format_relative_time(modality_rel)
            self._update_title_display(self.title, f'[{time_str} | ToA: {modality_toa:.4f}s]')

        half = window_duration_s / 2.0
        self.plot_widget.setXRange(center_rel_s - half, center_rel_s + half, padding=0.0)
        self.cursor_line.setValue(center_rel_s)

    def autoscale_y(self):
        self.plot_widget.enableAutoRange(axis='y', enable=True)


class SensorPlotsDock(QtWidgets.QDockWidget):
    """Dockable vertical stack of sensor modal plot widgets."""

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('Sensor Modalities', parent)
        self.setObjectName('SensorPlotsDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._plot_items: List[SensorPlotItem] = []
        self._window_duration_s = 10.0

        self._build_ui()
        self.playback_engine.time_changed.connect(self._on_time_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(1600, 260)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(400, 180)

    def _build_ui(self):
        container = QtWidgets.QWidget()
        container.setStyleSheet('background-color: #0f1016;')
        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Header toolbar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(2, 0, 2, 2)
        title = QtWidgets.QLabel('HDF5 SENSOR MEASUREMENTS')
        title.setStyleSheet('font-size: 11px; font-weight: 700; color: #38bdf8;')
        toolbar.addWidget(title)
        toolbar.addStretch()

        win_label = QtWidgets.QLabel('Window:')
        win_label.setStyleSheet('color: #94a3b8; font-size: 11px;')
        toolbar.addWidget(win_label)

        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(['2.0 s', '5.0 s', '10.0 s', '30.0 s', '60.0 s'])
        self.window_combo.setCurrentText('10.0 s')
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        self.window_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1b26;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }
        """)
        toolbar.addWidget(self.window_combo)

        self.btn_autoscale = QtWidgets.QPushButton('Autoscale Y')
        self.btn_autoscale.setStyleSheet("""
            QPushButton {
                background-color: #272738;
                color: #e2e8f0;
                border: 1px solid #3b3f54;
                border-radius: 4px;
                padding: 3px 10px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #35394d; border-color: #38bdf8; }
        """)
        self.btn_autoscale.clicked.connect(self._autoscale_all)
        toolbar.addWidget(self.btn_autoscale)

        main_layout.addLayout(toolbar)

        # Scrollable plots area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        self.plots_container = QtWidgets.QWidget()
        self.plots_container.setStyleSheet('background: transparent;')
        self.plots_layout = QtWidgets.QVBoxLayout(self.plots_container)
        self.plots_layout.setContentsMargins(0, 0, 0, 0)
        self.plots_layout.setSpacing(6)

        self.scroll_area.setWidget(self.plots_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        self.setWidget(container)

    def add_sensor_plot(
        self,
        title: str,
        y_label: str,
        reader: PagedSensorReader,
        channel_names: Optional[List[str]] = None,
        height: int = 160,
    ) -> SensorPlotItem:
        """Add a new paged sensor plot to the vertical stack with relative time axis."""
        time_axis = RelativeTimeAxisItem(orientation='bottom')
        plot_w = pg.PlotWidget(axisItems={'bottom': time_axis})
        plot_w.setMinimumHeight(height)
        plot_item = SensorPlotItem(plot_w, title, y_label, reader, channel_names)
        self._plot_items.append(plot_item)
        self.plots_layout.addWidget(plot_w)

        # Initial view update
        plot_item.update_view(
            self.playback_engine.current_toa_s, self._window_duration_s, self.playback_engine.min_toa_s
        )
        return plot_item

    def _on_window_changed(self, text: str):
        try:
            val = float(text.replace(' s', '').strip())
            self._window_duration_s = val
            self._on_time_changed(self.playback_engine.current_toa_s)
        except ValueError:
            pass

    def _autoscale_all(self):
        for item in self._plot_items:
            item.autoscale_y()

    def _on_time_changed(self, current_toa_s: float):
        t0 = self.playback_engine.min_toa_s
        for item in self._plot_items:
            item.update_view(current_toa_s, self._window_duration_s, t0)
