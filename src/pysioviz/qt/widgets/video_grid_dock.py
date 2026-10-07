"""
Video Grid Dock Widget displaying a 2x2 synchronized grid of external camera feeds.
Enforces exact aspect ratio matching video frames with zero margin gaps.
"""

from typing import Dict, Optional
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from pysioviz.qt.video_seeker import VideoSeekerWorker
from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.utils.time_utils import format_relative_time


class VideoCanvasWidget(QtWidgets.QWidget):
    """Flicker-free responsive video canvas maintaining aspect ratio without forcing large minimum size hints."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QtCore.QSize(100, 56))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(320, 180)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(100, 56)

    def set_pixmap(self, pixmap: QtGui.QPixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor('#0b0c10'))

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)


class AspectRatioContainer(QtWidgets.QWidget):
    """Container widget that constrains its child to the exact aspect ratio of the underlying video."""

    def __init__(
        self,
        child: QtWidgets.QWidget,
        aspect_ratio: float = 16.0 / 9.0,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.child = child
        self.child.setParent(self)
        self.aspect_ratio = aspect_ratio
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QtCore.QSize(100, 56))

    def set_aspect_ratio(self, ratio: float):
        if ratio > 0.1:
            self.aspect_ratio = ratio
            self._update_child_geometry()

    def _update_child_geometry(self):
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        target_w = w
        target_h = int(w / self.aspect_ratio)
        if target_h > h:
            target_h = h
            target_w = int(h * self.aspect_ratio)
        x = (w - target_w) // 2
        y = (h - target_h) // 2
        self.child.setGeometry(x, y, max(1, target_w), max(1, target_h))

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._update_child_geometry()


class CameraViewportWidget(QtWidgets.QFrame):
    """Camera viewport card with border directly hugging the video frame and translucent badge overlays."""

    def __init__(self, camera_id: str, label: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.label_name = label
        self._current_pixmap: Optional[QtGui.QPixmap] = None

        self.setStyleSheet("""
            CameraViewportWidget {
                background-color: #0b0c10;
                border: 1px solid #282a36;
                border-radius: 6px;
            }
        """)

        # Edge-to-edge layout with zero margin/padding to prevent white space
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video Frame Canvas
        self.canvas = VideoCanvasWidget(self)
        layout.addWidget(self.canvas)

        # Translucent Header Overlay
        self.header_badge = QtWidgets.QWidget(self)
        h_layout = QtWidgets.QHBoxLayout(self.header_badge)
        h_layout.setContentsMargins(6, 2, 6, 2)
        h_layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel(f'CAM: {self.label_name}')
        self.title_label.setStyleSheet('font-size: 10px; font-weight: 700; color: #38bdf8; background: transparent;')
        h_layout.addWidget(self.title_label)

        self.status_badge = QtWidgets.QLabel('SYNC')
        self.status_badge.setStyleSheet("""
            font-size: 8px; font-weight: bold; color: #22c55e;
            background-color: rgba(5, 46, 22, 0.85); border: 1px solid #16a34a;
            border-radius: 3px; padding: 1px 4px;
        """)
        h_layout.addWidget(self.status_badge)
        self.header_badge.setStyleSheet("""
            background-color: rgba(18, 19, 28, 0.82);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 4px;
        """)

        # Translucent Footer Overlay for Time and Frame
        self.footer_label = QtWidgets.QLabel('ToA: -- | Frame: --', self)
        self.footer_label.setStyleSheet("""
            background-color: rgba(18, 19, 28, 0.82);
            color: #f1f5f9;
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 10px;
            font-family: Consolas, monospace;
        """)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.header_badge.adjustSize()
        self.header_badge.move(6, 6)
        self.footer_label.adjustSize()
        self.footer_label.move(6, max(6, self.height() - self.footer_label.height() - 6))

    def set_frame(self, frame_id: int, toa_s: float, image: QtGui.QImage, experiment_start_toa_s: float = 0.0):
        """Update viewport with new decoded video frame and precise modality timestamp."""
        self._current_pixmap = QtGui.QPixmap.fromImage(image)
        self.canvas.set_pixmap(self._current_pixmap)

        rel_time_s = toa_s - experiment_start_toa_s if experiment_start_toa_s > 0 else 0.0
        time_str = format_relative_time(rel_time_s)
        self.footer_label.setText(f'{time_str} | #{frame_id} | ToA: {toa_s:.4f}s')
        self.footer_label.adjustSize()
        self.footer_label.move(6, max(6, self.height() - self.footer_label.height() - 6))


class VideoGridDock(QtWidgets.QDockWidget):
    """Dockable 2x2 grid displaying synchronized feeds from four external cameras."""

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('External Video Grid', parent)
        self.setObjectName('VideoGridDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self._workers: Dict[str, VideoSeekerWorker] = {}
        self._viewports: Dict[str, CameraViewportWidget] = {}
        self._containers: Dict[str, AspectRatioContainer] = {}

        container = QtWidgets.QWidget()
        container.setStyleSheet('background-color: #0f1016;')
        self.grid_layout = QtWidgets.QGridLayout(container)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(4)

        self.setWidget(container)

        self.playback_engine.time_changed.connect(self._on_time_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(1150, 480)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(300, 200)

    def add_camera(
        self,
        camera_id: str,
        label: str,
        video_path: str,
        toas: np.ndarray,
        row: int,
        col: int,
        fps: float = 30.0,
    ):
        """Register and place a camera into the 2x2 layout with exact aspect ratio matching."""
        viewport = CameraViewportWidget(camera_id, label, self)
        self._viewports[camera_id] = viewport

        aspect_container = AspectRatioContainer(viewport, aspect_ratio=16.0 / 9.0, parent=self)
        self._containers[camera_id] = aspect_container
        self.grid_layout.addWidget(aspect_container, row, col)

        worker = VideoSeekerWorker(unique_id=camera_id, video_path=video_path, toas=toas, fps=fps, parent=self)
        if worker.aspect_ratio > 0.1:
            aspect_container.set_aspect_ratio(worker.aspect_ratio)

        worker.frame_ready.connect(self._on_frame_ready)
        worker.start()
        self._workers[camera_id] = worker

        # Request initial frame
        initial_frame = worker.get_frame_for_toa(self.playback_engine.current_toa_s)
        worker.request_frame(initial_frame, self.playback_engine.current_toa_s)

    def _on_time_changed(self, current_toa_s: float):
        """Request nearest frame for each camera when timeline advances."""
        for cam_id, worker in self._workers.items():
            frame_id = worker.get_frame_for_toa(current_toa_s)
            worker.request_frame(frame_id, current_toa_s)

    def _on_frame_ready(self, cam_id: str, frame_id: int, toa_s: float, image: QtGui.QImage):
        """Slot receiving asynchronously decoded frame with sensor-specific toa_s."""
        viewport = self._viewports.get(cam_id)
        if viewport:
            viewport.set_frame(frame_id, toa_s, image, self.playback_engine.min_toa_s)

    def stop_all(self):
        """Stop all background seeker workers."""
        for worker in self._workers.values():
            worker.stop()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Stop worker threads cleanly upon dock destruction."""
        self.stop_all()
        super().closeEvent(event)
