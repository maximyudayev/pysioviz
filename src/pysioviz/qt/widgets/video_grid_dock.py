"""
Video Grid Dock Widget displaying a 2x2 synchronized grid of external camera feeds
with real-time overlays for bounding boxes, foreground segmentation masks, and human poses.
Enforces exact aspect ratio matching video frames with zero margin gaps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from pysioviz.qt.pipeline_overlays import (
    FrameOverlayData,
    PipelineOverlayManager,
    XSENS_BONES,
)
from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.qt.video_seeker import VideoSeekerWorker
from pysioviz.utils.time_utils import format_relative_time


COCO_BONES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Head
    (5, 6), (5, 7), (7, 9),              # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16)                   # Right leg
]

HALPE26_BONES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (17, 18), (18, 19),                  # Head - Neck - Hip spine
    (18, 5), (18, 6),                    # Clavicles
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12), (11, 19), (12, 19),# Torso / Pelvis
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16),                  # Right leg
    (15, 24), (15, 20), (20, 22),        # Left foot (heel, big toe, small toe)
    (16, 25), (16, 21), (21, 23),        # Right foot (heel, big toe, small toe)
]


class VideoCanvasWidget(QtWidgets.QWidget):
    """Flicker-free responsive video canvas with hardware-accelerated computer vision overlays."""

    # Pre-allocated drawing primitives for sub-millisecond paintEvent execution
    _BOX_WASH_COLOR = QtGui.QColor(56, 189, 248, 22)
    _BOX_PEN = QtGui.QPen(QtGui.QColor('#38bdf8'), 1.8)
    _CORNER_PEN = QtGui.QPen(QtGui.QColor('#00f5d4'), 2.6)
    _BADGE_FONT = QtGui.QFont('Segoe UI', 8, QtGui.QFont.Weight.Bold)
    _BADGE_FM = QtGui.QFontMetrics(_BADGE_FONT)
    _BADGE_BORDER_PEN = QtGui.QPen(QtGui.QColor('#00f5d4'), 1.0)
    _BADGE_BG_BRUSH = QtGui.QBrush(QtGui.QColor(15, 23, 42, 220))
    _BADGE_TEXT_PEN = QtGui.QPen(QtGui.QColor('#f8fafc'))
    _REPROJ_PEN = QtGui.QPen(QtGui.QColor('#38bdf8'), 2.2)
    _BONE_PEN = QtGui.QPen(QtGui.QColor('#a855f7'), 2.0)
    _JOINT_BORDER_PEN = QtGui.QPen(QtGui.QColor('#0b0c10'), 1.0)
    _HEAD_BRUSH = QtGui.QBrush(QtGui.QColor('#00f5d4'))
    _BODY_BRUSH = QtGui.QBrush(QtGui.QColor('#38bdf8'))

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._pixmap: Optional[QtGui.QPixmap] = None
        self._current_overlay: Optional[FrameOverlayData] = None
        self._mask_info: Optional[Tuple[QtGui.QImage, float, float, float]] = None

        # Display toggle states
        self.show_bbox: bool = True
        self.show_mask: bool = True
        self.show_pose: bool = True
        self.show_reid_label: bool = True

        # Reference original video resolution for coordinate normalization
        self.orig_video_width: float = 1920.0
        self.orig_video_height: float = 1080.0

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QtCore.QSize(100, 56))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(320, 180)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(100, 56)

    def set_pixmap(self, pixmap: QtGui.QPixmap):
        self._pixmap = pixmap
        self.update()

    def set_overlay(
        self,
        overlay_data: Optional[FrameOverlayData] = None,
        mask_info: Optional[Tuple[QtGui.QImage, float, float, float]] = None,
    ):
        self._current_overlay = overlay_data
        self._mask_info = mask_info
        self.update()

    def set_overlay_visibility(
        self,
        show_bbox: Optional[bool] = None,
        show_mask: Optional[bool] = None,
        show_pose: Optional[bool] = None,
        show_reid_label: Optional[bool] = None,
    ):
        if show_bbox is not None:
            self.show_bbox = show_bbox
        if show_mask is not None:
            self.show_mask = show_mask
        if show_pose is not None:
            self.show_pose = show_pose
        if show_reid_label is not None:
            self.show_reid_label = show_reid_label
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#0b0c10'))

        if self._pixmap and not self._pixmap.isNull():
            w_canvas = self.width()
            h_canvas = self.height()
            pix_w = self._pixmap.width()
            pix_h = self._pixmap.height()

            if pix_w > 0 and pix_h > 0 and w_canvas > 0 and h_canvas > 0:
                aspect = pix_w / pix_h
                canvas_aspect = w_canvas / h_canvas

                if canvas_aspect > aspect:
                    target_h = h_canvas
                    target_w = int(h_canvas * aspect)
                else:
                    target_w = w_canvas
                    target_h = int(w_canvas / aspect)

                x = (w_canvas - target_w) // 2
                y = (h_canvas - target_h) // 2
                target_rect = QtCore.QRect(x, y, target_w, target_h)

                # Direct drawPixmap avoiding CPU-side intermediate QPixmap allocation
                painter.drawPixmap(target_rect, self._pixmap)

                # Map from video space to canvas pixels
                norm_w = self.orig_video_width if self.orig_video_width > 0 else float(pix_w)
                norm_h = self.orig_video_height if self.orig_video_height > 0 else float(pix_h)
                sx = target_w / norm_w
                sy = target_h / norm_h

                # 1. Overlay Foreground Segmentation Mask (Zarr Format_Indexed8)
                if self.show_mask and self._mask_info is not None:
                    mask_qimg, xmin, ymin, side = self._mask_info
                    mask_rect = QtCore.QRectF(
                        x + xmin * sx,
                        y + ymin * sy,
                        side * sx,
                        side * sy,
                    )
                    painter.drawImage(mask_rect, mask_qimg)

                # 2. Overlay Bounding Box (Parquet)
                if (
                    self.show_bbox
                    and self._current_overlay is not None
                    and self._current_overlay.subj_present
                    and self._current_overlay.bbox_xyxy is not None
                ):
                    bx1, by1, bx2, by2 = self._current_overlay.bbox_xyxy
                    if bx2 > bx1 and by2 > by1:
                        rx = x + bx1 * sx
                        ry = y + by1 * sy
                        rw = (bx2 - bx1) * sx
                        rh = (by2 - by1) * sy
                        rect = QtCore.QRectF(rx, ry, rw, rh)

                        # Semi-transparent wash & neon cyan boundary
                        painter.fillRect(rect, self._BOX_WASH_COLOR)
                        painter.setPen(self._BOX_PEN)
                        painter.drawRoundedRect(rect, 3.0, 3.0)

                        # High-tech corner bracket accents (batched drawLines)
                        c_len = min(12.0, rw * 0.25, rh * 0.25)
                        painter.setPen(self._CORNER_PEN)
                        corner_lines = [
                            QtCore.QLineF(rx, ry, rx + c_len, ry),
                            QtCore.QLineF(rx, ry, rx, ry + c_len),
                            QtCore.QLineF(rx + rw, ry, rx + rw - c_len, ry),
                            QtCore.QLineF(rx + rw, ry, rx + rw, ry + c_len),
                            QtCore.QLineF(rx, ry + rh, rx + c_len, ry + rh),
                            QtCore.QLineF(rx, ry + rh, rx, ry + rh - c_len),
                            QtCore.QLineF(rx + rw, ry + rh, rx + rw - c_len, ry + rh),
                            QtCore.QLineF(rx + rw, ry + rh, rx + rw, ry + rh - c_len),
                        ]
                        painter.drawLines(corner_lines)

                        # Re-ID Confidence badge pill
                        if self.show_reid_label:
                            sim = self._current_overlay.reid_cosine_sim
                            label_text = f'RE-ID {sim:.2f}' if sim > 0 else 'TARGET'
                            painter.setFont(self._BADGE_FONT)
                            fm = self._BADGE_FM
                            text_w = fm.horizontalAdvance(label_text)
                            text_h = fm.height()

                            badge_w = text_w + 12
                            badge_h = text_h + 4
                            badge_y = ry - badge_h - 2 if (ry - badge_h - 2) >= y else ry + 4
                            badge_rect = QtCore.QRectF(rx + 2, badge_y, badge_w, badge_h)

                            painter.setPen(self._BADGE_BORDER_PEN)
                            painter.setBrush(self._BADGE_BG_BRUSH)
                            painter.drawRoundedRect(badge_rect, 3.0, 3.0)

                            painter.setPen(self._BADGE_TEXT_PEN)
                            painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, label_text)

                # 3. Overlay Monocular Human Pose / Skeleton
                if self.show_pose and self._current_overlay is not None:
                    # Wireframe polylines (Stage 7 reprojection)
                    if self._current_overlay.bone_polylines:
                        painter.setPen(self._REPROJ_PEN)
                        reproj_lines = []
                        for poly in self._current_overlay.bone_polylines:
                            for i in range(len(poly) - 1):
                                reproj_lines.append(QtCore.QLineF(
                                    x + poly[i][0] * sx,
                                    y + poly[i][1] * sy,
                                    x + poly[i + 1][0] * sx,
                                    y + poly[i + 1][1] * sy,
                                ))
                        if reproj_lines:
                            painter.drawLines(reproj_lines)

                    # 2D Keypoints & Skeleton (Stage 3 dense / monocular)
                    elif self._current_overlay.keypoints_2d is not None:
                        kps = self._current_overlay.keypoints_2d
                        confs = self._current_overlay.keypoint_confs
                        k_count = len(kps)

                        if k_count == 23:
                            bones = XSENS_BONES
                        elif k_count == 26:
                            bones = HALPE26_BONES
                        elif k_count == 17:
                            bones = COCO_BONES
                        else:
                            bones = []

                        painter.setPen(self._BONE_PEN)
                        bone_lines = []
                        for j1, j2 in bones:
                            if j1 < k_count and j2 < k_count:
                                if confs is None or (confs[j1] > 0.2 and confs[j2] > 0.2):
                                    bone_lines.append(QtCore.QLineF(
                                        x + kps[j1, 0] * sx,
                                        y + kps[j1, 1] * sy,
                                        x + kps[j2, 0] * sx,
                                        y + kps[j2, 1] * sy,
                                    ))
                        if bone_lines:
                            painter.drawLines(bone_lines)

                        # Joint node markers
                        painter.setPen(self._JOINT_BORDER_PEN)
                        for idx, pt in enumerate(kps):
                            if confs is None or confs[idx] > 0.2:
                                kx = x + pt[0] * sx
                                ky = y + pt[1] * sy
                                painter.setBrush(self._HEAD_BRUSH if idx < 7 else self._BODY_BRUSH)
                                painter.drawEllipse(QtCore.QPointF(kx, ky), 2.5, 2.5)


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

    _TARGET_PRESENT_STYLE = """
        background-color: rgba(4, 47, 46, 0.90);
        color: #00f5d4;
        border: 1px solid #0d9488;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 9px;
        font-weight: bold;
        font-family: Consolas, monospace;
    """
    _TARGET_ABSENT_STYLE = """
        background-color: rgba(30, 41, 59, 0.90);
        color: #94a3b8;
        border: 1px solid #475569;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 9px;
        font-weight: bold;
        font-family: Consolas, monospace;
    """

    def __init__(self, camera_id: str, label: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.label_name = label
        self._current_pixmap: Optional[QtGui.QPixmap] = None
        self._target_present_state: Optional[bool] = None
        self._last_badge_text: str = ''
        self._last_footer_text: str = ''

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

        # Video Frame Canvas with Live Overlay Rendering
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

        # Translucent Target Badge Overlay (Top-Right Corner)
        self.target_badge = QtWidgets.QLabel('TARGET', self)
        self.target_badge.setStyleSheet("""
            background-color: rgba(18, 19, 28, 0.85);
            color: #00f5d4;
            border: 1px solid rgba(0, 245, 212, 0.4);
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 9px;
            font-weight: bold;
            font-family: Consolas, monospace;
        """)
        self.target_badge.hide()

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

        self.target_badge.adjustSize()
        tb_x = max(6, self.width() - self.target_badge.width() - 6)
        self.target_badge.move(tb_x, 6)

        self.footer_label.adjustSize()
        self.footer_label.move(6, max(6, self.height() - self.footer_label.height() - 6))

    def set_frame(
        self,
        frame_id: int,
        toa_s: float,
        image: QtGui.QImage,
        experiment_start_toa_s: float = 0.0,
        overlay_data: Optional[FrameOverlayData] = None,
        mask_info: Optional[Tuple[QtGui.QImage, float, float, float]] = None,
    ):
        """Update viewport with new decoded video frame, timestamp, and computer vision overlays."""
        self._current_pixmap = QtGui.QPixmap.fromImage(image)
        self.canvas.set_pixmap(self._current_pixmap)
        self.canvas.set_overlay(overlay_data=overlay_data, mask_info=mask_info)

        # Update target badge status in top-right corner without reparsing stylesheets every frame
        if overlay_data is not None:
            is_present = overlay_data.subj_present
            sim = overlay_data.reid_cosine_sim
            badge_text = f'TARGET: RE-ID {sim:.2f}' if is_present else 'TARGET: ABSENT'
            if badge_text != self._last_badge_text:
                self._last_badge_text = badge_text
                self.target_badge.setText(badge_text)
                self.target_badge.adjustSize()
                tb_x = max(6, self.width() - self.target_badge.width() - 6)
                self.target_badge.move(tb_x, 6)

            if is_present != self._target_present_state:
                self._target_present_state = is_present
                self.target_badge.setStyleSheet(
                    self._TARGET_PRESENT_STYLE if is_present else self._TARGET_ABSENT_STYLE
                )

            if not self.target_badge.isVisible():
                self.target_badge.show()
        else:
            if self.target_badge.isVisible():
                self.target_badge.hide()

        rel_time_s = toa_s - experiment_start_toa_s if experiment_start_toa_s > 0 else 0.0
        time_str = format_relative_time(rel_time_s)
        footer_text = f'{time_str} | #{frame_id} | ToA: {toa_s:.4f}s'
        if footer_text != self._last_footer_text:
            self._last_footer_text = footer_text
            self.footer_label.setText(footer_text)
            self.footer_label.adjustSize()


class VideoGridDock(QtWidgets.QDockWidget):
    """Dockable 2x2 grid displaying synchronized feeds from four external cameras with CV overlays."""

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('External Video Grid', parent)
        self.setObjectName('VideoGridDock')
        self.playback_engine = playback_engine
        self.overlay_manager: Optional[PipelineOverlayManager] = None

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

    def set_overlay_manager(self, manager: Optional[PipelineOverlayManager]):
        """Attach or detach the pipeline overlay manager for real-time overlay queries."""
        self.overlay_manager = manager
        # Refresh current frames
        for cam_id, worker in self._workers.items():
            frame_id = worker.get_frame_for_toa(self.playback_engine.current_toa_s)
            worker.request_frame(frame_id, self.playback_engine.current_toa_s)

    def set_overlay_visibility(
        self,
        show_bbox: Optional[bool] = None,
        show_mask: Optional[bool] = None,
        show_pose: Optional[bool] = None,
        show_reid_label: Optional[bool] = None,
    ):
        """Update overlay visibility across all registered camera viewports."""
        if self.overlay_manager:
            if show_bbox is not None:
                self.overlay_manager.show_bboxes = show_bbox
            if show_mask is not None:
                self.overlay_manager.show_masks = show_mask
            if show_pose is not None:
                self.overlay_manager.show_poses = show_pose
            if show_reid_label is not None:
                self.overlay_manager.show_reid_labels = show_reid_label

        for viewport in self._viewports.values():
            viewport.canvas.set_overlay_visibility(
                show_bbox=show_bbox,
                show_mask=show_mask,
                show_pose=show_pose,
                show_reid_label=show_reid_label,
            )

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

        # Pass true video resolution to canvas for pixel-perfect overlay alignment
        if worker._orig_width > 0 and worker._orig_height > 0:
            viewport.canvas.orig_video_width = float(worker._orig_width)
            viewport.canvas.orig_video_height = float(worker._orig_height)

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
        if not viewport:
            return

        overlay_data: Optional[FrameOverlayData] = None
        mask_info: Optional[Tuple[QtGui.QImage, float, float, float]] = None

        if self.overlay_manager is not None:
            overlay_data = self.overlay_manager.get_overlay_data(cam_id, frame_id)
            if overlay_data is not None and overlay_data.v_idx is not None and self.overlay_manager.show_masks:
                mask_info = self.overlay_manager.get_mask_and_bounds(cam_id, overlay_data.v_idx)

        viewport.set_frame(
            frame_id=frame_id,
            toa_s=toa_s,
            image=image,
            experiment_start_toa_s=self.playback_engine.min_toa_s,
            overlay_data=overlay_data,
            mask_info=mask_info,
        )

    def stop_all(self):
        """Stop all background seeker workers and overlay manager prefetcher."""
        for worker in self._workers.values():
            worker.stop()
        if self.overlay_manager is not None:
            self.overlay_manager.stop()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Stop worker threads cleanly upon dock destruction."""
        self.stop_all()
        super().closeEvent(event)
