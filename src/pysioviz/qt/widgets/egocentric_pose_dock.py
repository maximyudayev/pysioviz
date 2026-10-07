"""
Egocentric Video and 3D Pose Skeleton Dock Widget.
Displays the first-person video feed (smart glasses ego camera) alongside the
reconstructed 3D kinematic skeleton (Xsens MVN).
"""

from typing import List, Optional, Tuple
import numpy as np
import h5py

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.qt.video_seeker import VideoSeekerWorker


class EgocentricPoseDock(QtWidgets.QDockWidget):
    """Dockable widget displaying first-person video feed and 3D reconstructed pose skeleton."""

    # 23-segment Xsens MVN Kinematic Chains (from AidWear)
    SPINE_BONES: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    RIGHT_ARM_BONES: List[Tuple[int, int]] = [(4, 7), (7, 8), (8, 9), (9, 10)]
    LEFT_ARM_BONES: List[Tuple[int, int]] = [(4, 11), (11, 12), (12, 13), (13, 14)]
    RIGHT_LEG_BONES: List[Tuple[int, int]] = [(0, 15), (15, 16), (16, 17), (17, 18)]
    LEFT_LEG_BONES: List[Tuple[int, int]] = [(0, 19), (19, 20), (20, 21), (21, 22)]

    DEFAULT_REST_POSE: np.ndarray = np.array(
        [
            [0.0, 0.0, 1.1],  # 0: Pelvis
            [0.0, 0.0, 1.18],  # 1: L5
            [0.0, 0.0, 1.26],  # 2: L3
            [0.0, 0.0, 1.35],  # 3: T12
            [0.0, 0.0, 1.45],  # 4: T8
            [0.0, 0.0, 1.57],  # 5: Neck
            [0.0, 0.0, 1.73],  # 6: Head
            [0.0, -0.18, 1.45],  # 7: Right Shoulder
            [0.0, -0.32, 1.30],  # 8: Right Upper Arm
            [0.0, -0.35, 1.05],  # 9: Right Forearm
            [0.0, -0.35, 0.9],  # 10: Right Hand
            [0.0, 0.18, 1.45],  # 11: Left Shoulder
            [0.0, 0.32, 1.30],  # 12: Left Upper Arm
            [0.0, 0.35, 1.05],  # 13: Left Forearm
            [0.0, 0.35, 0.9],  # 14: Left Hand
            [0.0, -0.10, 0.70],  # 15: Right Upper Leg
            [0.0, -0.10, 0.30],  # 16: Right Lower Leg
            [0.0, -0.10, 0.15],  # 17: Right Foot
            [0.15, -0.10, 0.15],  # 18: Right Toe
            [0.0, 0.10, 0.70],  # 19: Left Upper Leg
            [0.0, 0.10, 0.30],  # 20: Left Lower Leg
            [0.0, 0.10, 0.15],  # 21: Left Foot
            [0.15, 0.10, 0.15],  # 22: Left Toe
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('Egocentric & 3D Pose', parent)
        self.setObjectName('EgocentricPoseDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
        )

        self._ego_worker: Optional[VideoSeekerWorker] = None
        self._current_pixmap: Optional[QtGui.QPixmap] = None

        # 3D pose data references
        self._pose_hdf5_path: Optional[str] = None
        self._pose_toas: np.ndarray = np.empty(0, dtype=np.float64)
        self._pose_positions: Optional[h5py.Dataset] = None
        self._pose_h5_file: Optional[h5py.File] = None

        self._build_ui()
        self.playback_engine.time_changed.connect(self._on_time_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(320, 480)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(200, 200)

    def _build_ui(self):
        main_widget = QtWidgets.QWidget()
        main_widget.setStyleSheet('background-color: #0f1016;')
        layout = QtWidgets.QVBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Top Section: First-Person Video Card
        self.ego_card = QtWidgets.QFrame()
        self.ego_card.setStyleSheet("""
            QFrame {
                background-color: #12131c;
                border: 1px solid #282a36;
                border-radius: 6px;
            }
        """)
        ego_layout = QtWidgets.QVBoxLayout(self.ego_card)
        ego_layout.setContentsMargins(6, 6, 6, 6)
        ego_layout.setSpacing(4)

        ego_header = QtWidgets.QHBoxLayout()
        ego_title = QtWidgets.QLabel('FIRST-PERSON FEED')
        ego_title.setStyleSheet('font-size: 11px; font-weight: 700; color: #a855f7;')
        ego_header.addWidget(ego_title)
        ego_header.addStretch()

        self.ego_status = QtWidgets.QLabel('● GLASSES')
        self.ego_status.setStyleSheet('color: #a855f7; font-size: 10px; font-weight: bold;')
        ego_header.addWidget(self.ego_status)
        ego_layout.addLayout(ego_header)

        from pysioviz.qt.widgets.video_grid_dock import VideoCanvasWidget

        self.ego_canvas = VideoCanvasWidget(self)
        ego_layout.addWidget(self.ego_canvas, stretch=1)

        self.ego_footer = QtWidgets.QLabel('ToA: -- | Frame: --')
        self.ego_footer.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ego_footer.setStyleSheet('font-size: 10px; color: #94a3b8;')
        ego_layout.addWidget(self.ego_footer)

        layout.addWidget(self.ego_card, stretch=1)

        # 2. Bottom Section: 3D Reconstructed Pose Skeleton Card
        self.pose_card = QtWidgets.QFrame()
        self.pose_card.setStyleSheet("""
            QFrame {
                background-color: #12131c;
                border: 1px solid #282a36;
                border-radius: 6px;
            }
        """)
        pose_layout = QtWidgets.QVBoxLayout(self.pose_card)
        pose_layout.setContentsMargins(6, 6, 6, 6)
        pose_layout.setSpacing(4)

        pose_header = QtWidgets.QHBoxLayout()
        pose_title = QtWidgets.QLabel('3D POSE SKELETON')
        pose_title.setStyleSheet('font-size: 11px; font-weight: 700; color: #38bdf8;')
        pose_header.addWidget(pose_title)
        pose_header.addStretch()

        self.pose_status_badge = QtWidgets.QLabel('● STANDBY')
        self.pose_status_badge.setStyleSheet('color: #64748b; font-size: 10px; font-weight: bold;')
        pose_header.addWidget(self.pose_status_badge)
        pose_layout.addLayout(pose_header)

        pose_ctrls = QtWidgets.QHBoxLayout()
        self.pose_center_checkbox = QtWidgets.QCheckBox('Center Root')
        self.pose_center_checkbox.setChecked(True)
        self.pose_center_checkbox.setStyleSheet('color: #94a3b8; font-size: 10px;')
        pose_ctrls.addWidget(self.pose_center_checkbox)
        pose_ctrls.addStretch()

        self.pose_reset_btn = QtWidgets.QPushButton('↺ Reset')
        self.pose_reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #272738;
                color: #cbd5e1;
                border: 1px solid #3b3f54;
                border-radius: 3px;
                padding: 1px 6px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #35394d; color: #38bdf8; }
        """)
        self.pose_reset_btn.clicked.connect(self._reset_camera)
        pose_ctrls.addWidget(self.pose_reset_btn)
        pose_layout.addLayout(pose_ctrls)

        # 3D GL Viewport
        self.pose_view = gl.GLViewWidget()
        self.pose_view.setBackgroundColor('#10121a')
        self.pose_view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._reset_camera()

        # Ground Grid (3.0m x 3.0m, spacing 0.5m)
        self._grid = gl.GLGridItem()
        self._grid.setSize(3.0, 3.0)
        self._grid.setSpacing(0.5, 0.5)
        self._grid.setColor((0.3, 0.35, 0.45, 0.35))
        self.pose_view.addItem(self._grid)

        # Skeleton Line Items
        self._pose_spine = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode='lines', width=3.0, color=(0.22, 0.74, 0.97, 1.0)
        )
        self._pose_r_arm = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode='lines', width=2.5, color=(0.2, 0.83, 0.6, 1.0)
        )
        self._pose_l_arm = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode='lines', width=2.5, color=(0.65, 0.55, 0.98, 1.0)
        )
        self._pose_r_leg = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode='lines', width=3.0, color=(0.98, 0.75, 0.14, 1.0)
        )
        self._pose_l_leg = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode='lines', width=3.0, color=(0.96, 0.45, 0.71, 1.0)
        )
        for item in [self._pose_spine, self._pose_r_arm, self._pose_l_arm, self._pose_r_leg, self._pose_l_leg]:
            self.pose_view.addItem(item)

        # Joints Scatter Item
        self._pose_joints = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), size=7, color=(0.95, 0.96, 0.98, 1.0)
        )
        self.pose_view.addItem(self._pose_joints)

        # Render rest pose initially
        self._render_pose(self.DEFAULT_REST_POSE)

        pose_layout.addWidget(self.pose_view, stretch=1)
        layout.addWidget(self.pose_card, stretch=1)

        self.setWidget(main_widget)

    def _reset_camera(self):
        self.pose_view.setCameraPosition(
            pos=QtGui.QVector3D(0.0, 0.0, 1.1), distance=2.4, elevation=15.0, azimuth=-60.0
        )

    def set_egocentric_video(self, video_path: str, toas: np.ndarray, fps: float = 30.0):
        """Configure the first-person egocentric video stream."""
        if self._ego_worker is not None:
            self._ego_worker.stop()

        self._ego_worker = VideoSeekerWorker(
            unique_id='glasses_ego', video_path=video_path, toas=toas, fps=fps, parent=self
        )
        self._ego_worker.frame_ready.connect(self._on_ego_frame_ready)
        self._ego_worker.start()

        init_f = self._ego_worker.get_frame_for_toa(self.playback_engine.current_toa_s)
        self._ego_worker.request_frame(init_f, self.playback_engine.current_toa_s)

    def set_pose_dataset(self, hdf5_path: str, pos_ds_path: str, toa_ds_path: str):
        """Link the 3D reconstructed skeleton HDF5 dataset."""
        try:
            self._pose_hdf5_path = hdf5_path
            self._pose_h5_file = h5py.File(hdf5_path, 'r')
            self._pose_positions = self._pose_h5_file[pos_ds_path]
            # Read toas lazily
            toa_ds = self._pose_h5_file[toa_ds_path]
            self._pose_toas = np.asarray(toa_ds).ravel().astype(np.float64)

            self.pose_status_badge.setText('● TRACKING (23 Seg)')
            self.pose_status_badge.setStyleSheet('color: #22c55e; font-size: 10px; font-weight: bold;')
        except Exception as e:
            print(f'Error loading 3D pose dataset from {hdf5_path}: {e}', flush=True)

    @staticmethod
    def _build_line_segments(bones: List[Tuple[int, int]], pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        verts = []
        for i, j in bones:
            if i < n and j < n:
                verts.append(pos[i])
                verts.append(pos[j])
        if len(verts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(verts, dtype=np.float32)

    def _render_pose(self, pos: np.ndarray):
        if pos.ndim != 2 or pos.shape[0] == 0:
            return

        # Auto-detect units (cm -> m)
        if np.max(np.abs(pos)) > 10.0:
            pos = pos / 100.0

        if self.pose_center_checkbox.isChecked():
            root_xy = pos[0, :2].copy()
            pos_render = pos.copy()
            pos_render[:, 0] -= root_xy[0]
            pos_render[:, 1] -= root_xy[1]
        else:
            pos_render = pos

        self._pose_spine.setData(pos=self._build_line_segments(self.SPINE_BONES, pos_render))
        self._pose_r_arm.setData(pos=self._build_line_segments(self.RIGHT_ARM_BONES, pos_render))
        self._pose_l_arm.setData(pos=self._build_line_segments(self.LEFT_ARM_BONES, pos_render))
        self._pose_r_leg.setData(pos=self._build_line_segments(self.RIGHT_LEG_BONES, pos_render))
        self._pose_l_leg.setData(pos=self._build_line_segments(self.LEFT_LEG_BONES, pos_render))
        self._pose_joints.setData(pos=pos_render)

    def _on_time_changed(self, current_toa_s: float):
        # 1. Update egocentric video
        if self._ego_worker is not None:
            f_id = self._ego_worker.get_frame_for_toa(current_toa_s)
            self._ego_worker.request_frame(f_id, current_toa_s)

        # 2. Update 3D pose
        if self._pose_positions is not None and self._pose_toas.size > 0:
            idx = int(np.searchsorted(self._pose_toas, current_toa_s))
            if idx >= len(self._pose_toas):
                idx = len(self._pose_toas) - 1
            if idx > 0 and abs(self._pose_toas[idx - 1] - current_toa_s) < abs(self._pose_toas[idx] - current_toa_s):
                idx = idx - 1

            pos = np.asarray(self._pose_positions[idx]).astype(np.float32)
            self._render_pose(pos)

    def _on_ego_frame_ready(self, _, frame_id: int, toa_s: float, image: QtGui.QImage):
        self._current_pixmap = QtGui.QPixmap.fromImage(image)
        self.ego_canvas.set_pixmap(self._current_pixmap)
        self.ego_footer.setText(f'ToA: {toa_s:.4f}s | Frame #{frame_id}')

    def stop_all(self):
        """Stop background video worker and close HDF5 file."""
        if self._ego_worker is not None:
            self._ego_worker.stop()
        if self._pose_h5_file is not None:
            try:
                self._pose_h5_file.close()
            except Exception:
                pass
            self._pose_h5_file = None

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.stop_all()
        super().closeEvent(event)
