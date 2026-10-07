"""
Egocentric Video Dock Widget displaying first-person wearable glasses feed.
Independently dockable and tileable with exact aspect ratio container.
"""

from typing import Optional
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from pysioviz.qt.video_seeker import VideoSeekerWorker
from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.qt.widgets.video_grid_dock import VideoCanvasWidget, AspectRatioContainer
from pysioviz.utils.time_utils import format_relative_time


class EgocentricVideoDock(QtWidgets.QDockWidget):
    """Dockable widget displaying first-person egocentric video feed (smart glasses)."""

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('First-Person Feed', parent)
        self.setObjectName('EgocentricVideoDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self._ego_worker: Optional[VideoSeekerWorker] = None
        self._current_pixmap: Optional[QtGui.QPixmap] = None

        self._build_ui()
        self.playback_engine.time_changed.connect(self._on_time_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(360, 480)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(180, 160)

    def _build_ui(self):
        container = QtWidgets.QWidget()
        container.setStyleSheet('background-color: #0f1016;')
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Main viewport frame
        self.frame_card = QtWidgets.QFrame()
        self.frame_card.setStyleSheet("""
            QFrame {
                background-color: #0b0c10;
                border: 1px solid #282a36;
                border-radius: 6px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(self.frame_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Video canvas
        self.ego_canvas = VideoCanvasWidget(self)
        card_layout.addWidget(self.ego_canvas)

        # Translucent Header Overlay
        self.header_badge = QtWidgets.QWidget(self.frame_card)
        h_layout = QtWidgets.QHBoxLayout(self.header_badge)
        h_layout.setContentsMargins(6, 2, 6, 2)
        h_layout.setSpacing(6)

        title = QtWidgets.QLabel('FIRST-PERSON FEED')
        title.setStyleSheet('font-size: 10px; font-weight: 700; color: #a855f7; background: transparent;')
        h_layout.addWidget(title)

        status = QtWidgets.QLabel('GLASSES')
        status.setStyleSheet("""
            font-size: 8px; font-weight: bold; color: #a855f7;
            background-color: rgba(46, 16, 101, 0.85); border: 1px solid #7e22ce;
            border-radius: 3px; padding: 1px 4px;
        """)
        h_layout.addWidget(status)
        self.header_badge.setStyleSheet("""
            background-color: rgba(18, 19, 28, 0.82);
            border: 1px solid rgba(168, 85, 247, 0.35);
            border-radius: 4px;
        """)

        # Translucent Footer Overlay for Time and Frame
        self.ego_footer = QtWidgets.QLabel('ToA: -- | Frame: --', self.frame_card)
        self.ego_footer.setStyleSheet("""
            background-color: rgba(18, 19, 28, 0.82);
            color: #f1f5f9;
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 10px;
            font-family: Consolas, monospace;
        """)

        # Wrap in AspectRatioContainer
        self.aspect_container = AspectRatioContainer(self.frame_card, aspect_ratio=16.0 / 9.0, parent=container)
        layout.addWidget(self.aspect_container)

        self.setWidget(container)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.header_badge.adjustSize()
        self.header_badge.move(6, 6)
        self.ego_footer.adjustSize()
        self.ego_footer.move(6, max(6, self.frame_card.height() - self.ego_footer.height() - 6))

    def set_egocentric_video(self, video_path: str, toas: np.ndarray, fps: float = 30.0):
        """Link first-person video stream and timestamp dataset."""
        if self._ego_worker is not None:
            self._ego_worker.stop()

        self._ego_worker = VideoSeekerWorker(
            unique_id='glasses_ego',
            video_path=video_path,
            toas=toas,
            fps=fps,
            parent=self,
        )
        if self._ego_worker.aspect_ratio > 0.1:
            self.aspect_container.set_aspect_ratio(self._ego_worker.aspect_ratio)

        self._ego_worker.frame_ready.connect(self._on_ego_frame_ready)
        self._ego_worker.start()

        # Initial frame request
        f_id = self._ego_worker.get_frame_for_toa(self.playback_engine.current_toa_s)
        self._ego_worker.request_frame(f_id, self.playback_engine.current_toa_s)

    def _on_time_changed(self, current_toa_s: float):
        if self._ego_worker is not None:
            f_id = self._ego_worker.get_frame_for_toa(current_toa_s)
            self._ego_worker.request_frame(f_id, current_toa_s)

    def _on_ego_frame_ready(self, _, frame_id: int, toa_s: float, image: QtGui.QImage):
        """Update viewport with specific glasses frame and exact modality timestamp."""
        self._current_pixmap = QtGui.QPixmap.fromImage(image)
        self.ego_canvas.set_pixmap(self._current_pixmap)

        rel_time_s = toa_s - self.playback_engine.min_toa_s if self.playback_engine.min_toa_s > 0 else 0.0
        time_str = format_relative_time(rel_time_s)
        self.ego_footer.setText(f'{time_str} | #{frame_id} | ToA: {toa_s:.4f}s')
        self.ego_footer.adjustSize()
        self.ego_footer.move(6, max(6, self.frame_card.height() - self.ego_footer.height() - 6))

    def stop_all(self):
        """Stop background video worker."""
        if self._ego_worker is not None:
            self._ego_worker.stop()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.stop_all()
        super().closeEvent(event)
