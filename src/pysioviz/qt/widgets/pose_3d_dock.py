"""
3D Reconstructed Kinematic Pose Skeleton Dock Widget (Xsens MVN).
Independently dockable and tileable 3D OpenGL viewport.
"""

from typing import Optional
import numpy as np
import h5py

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.utils.time_utils import format_relative_time


# Standard Xsens 23-segment kinematic kinematic connectivity definition
XSENS_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),  # Spine / Pelvis / Torso / Neck / Head
    (4, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # Right Shoulder -> Hand
    (4, 11),
    (11, 12),
    (12, 13),
    (13, 14),  # Left Shoulder -> Hand
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),  # Right Hip -> Foot
    (0, 19),
    (19, 20),
    (20, 21),
    (21, 22),  # Left Hip -> Foot
]


class Pose3DDock(QtWidgets.QDockWidget):
    """Dockable 3D OpenGL skeleton viewport displaying reconstructed human movement."""

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('3D Pose Skeleton', parent)
        self.setObjectName('Pose3DDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self._pose_hdf5_path: Optional[str] = None
        self._pose_toas: np.ndarray = np.empty(0, dtype=np.float64)
        self._pose_positions: Optional[h5py.Dataset] = None
        self._pose_h5_file: Optional[h5py.File] = None

        # Camera framing settings: aim at skeleton mid-torso, minimal floor, full height
        self._target_pos = QtGui.QVector3D(0.0, 0.0, 0.85)
        self._target_distance = 1.70
        self._target_elevation = 6.0
        self._target_azimuth = -60.0

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

        card = QtWidgets.QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #0b0c10;
                border: 1px solid #282a36;
                border-radius: 6px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        # Header Title & Status
        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('3D POSE SKELETON')
        title.setStyleSheet('font-size: 11px; font-weight: 700; color: #38bdf8;')
        header_row.addWidget(title)
        header_row.addStretch()

        self.pose_status_badge = QtWidgets.QLabel('STANDBY')
        self.pose_status_badge.setStyleSheet('color: #64748b; font-size: 9px; font-weight: bold;')
        header_row.addWidget(self.pose_status_badge)
        card_layout.addLayout(header_row)

        # Controls row
        ctrls_row = QtWidgets.QHBoxLayout()
        self.pose_center_checkbox = QtWidgets.QCheckBox('Center Root')
        self.pose_center_checkbox.setChecked(True)
        self.pose_center_checkbox.setStyleSheet('color: #94a3b8; font-size: 10px;')
        ctrls_row.addWidget(self.pose_center_checkbox)
        ctrls_row.addStretch()

        self.pose_reset_btn = QtWidgets.QPushButton('↺ Reset View')
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
        ctrls_row.addWidget(self.pose_reset_btn)
        card_layout.addLayout(ctrls_row)

        # 3D GL Viewport
        self.pose_view = gl.GLViewWidget()
        self.pose_view.setBackgroundColor('#10121a')
        self.pose_view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._reset_camera()

        # Minimal subtle floor reference directly under the feet
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=1.8, y=1.8, z=0)
        self.grid.setSpacing(x=0.3, y=0.3, z=0.3)
        self.grid.setColor((45, 50, 70, 192))
        self.pose_view.addItem(self.grid)

        # Skeleton joint mesh & bone lines
        self.skeleton_lines = gl.GLLinePlotItem(
            pos=np.zeros((len(XSENS_BONES) * 2, 3), dtype=np.float32),
            color=QtGui.QColor('#38bdf8'),
            width=2.5,
            mode='lines',
        )
        self.pose_view.addItem(self.skeleton_lines)

        self.joints_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((23, 3), dtype=np.float32),
            color=QtGui.QColor('#a855f7'),
            size=6.0,
        )
        self.pose_view.addItem(self.joints_scatter)

        card_layout.addWidget(self.pose_view, stretch=1)

        # Timestamp footer
        self.footer_label = QtWidgets.QLabel('ToA: -- | Sample: --')
        self.footer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.footer_label.setStyleSheet('font-size: 10px; color: #94a3b8; font-family: Consolas, monospace;')
        card_layout.addWidget(self.footer_label)

        layout.addWidget(card)
        self.setWidget(container)

    def _reset_camera(self):
        """Set default camera distance and viewpoint angles framing the skeleton in full height."""
        self.pose_view.setCameraPosition(
            pos=self._target_pos,
            distance=self._target_distance,
            elevation=self._target_elevation,
            azimuth=self._target_azimuth,
        )

    def set_pose_dataset(
        self,
        hdf5_path: str,
        pos_ds_path: str = 'mvn-analyze/xsens_pose/position',
        toa_ds_path: str = 'mvn-analyze/xsens_pose/toa_s',
    ):
        """Link 3D pose HDF5 dataset and auto-frame the skeleton in full height."""
        self._pose_hdf5_path = hdf5_path
        try:
            self._pose_h5_file = h5py.File(hdf5_path, 'r')
            if pos_ds_path in self._pose_h5_file and toa_ds_path in self._pose_h5_file:
                self._pose_positions = self._pose_h5_file[pos_ds_path]
                self._pose_toas = np.asarray(self._pose_h5_file[toa_ds_path]).ravel().astype(np.float64)

                # Compute skeleton height and center to frame the skeleton in full height with minimal ground
                if len(self._pose_positions) > 0:
                    first_sample = np.asarray(self._pose_positions[0], dtype=np.float32)
                    if first_sample.ndim == 1:
                        first_sample = first_sample.reshape(-1, 3)
                    z_min = float(np.nanmin(first_sample[:, 2]))
                    z_max = float(np.nanmax(first_sample[:, 2]))
                    skeleton_height = max(0.8, z_max - z_min)
                    center_z = float((z_min + z_max) / 2.0)
                    self._target_pos = QtGui.QVector3D(0.0, 0.0, center_z)
                    # Distance of ~1.15 * height frames the skeleton in full vertical height with minimal margins
                    self._target_distance = float(skeleton_height * 1.15)
                    self._reset_camera()

                self.pose_status_badge.setText('TRACKING')
                self.pose_status_badge.setStyleSheet('color: #22c55e; font-size: 9px; font-weight: bold;')
                self._on_time_changed(self.playback_engine.current_toa_s)
        except Exception as e:
            print(f'Error initializing 3D pose viewer: {e}', flush=True)

    def _on_time_changed(self, current_toa_s: float):
        """Render 3D skeleton for nearest timestamp from Xsens pose dataset."""
        if self._pose_positions is not None and self._pose_toas.size > 0:
            idx = int(np.searchsorted(self._pose_toas, current_toa_s))
            if idx >= len(self._pose_toas):
                idx = len(self._pose_toas) - 1
            if idx > 0 and abs(self._pose_toas[idx - 1] - current_toa_s) < abs(self._pose_toas[idx] - current_toa_s):
                idx = idx - 1

            pos = np.asarray(self._pose_positions[idx]).astype(np.float32)
            actual_toa_s = float(self._pose_toas[idx])
            self._render_pose(pos)

            rel_time_s = actual_toa_s - self.playback_engine.min_toa_s if self.playback_engine.min_toa_s > 0 else 0.0
            time_str = format_relative_time(rel_time_s)
            self.footer_label.setText(f'{time_str} | Sample #{idx} | ToA: {actual_toa_s:.4f}s')

    def _render_pose(self, positions_3d: np.ndarray):
        """Update 3D OpenGL joints and bone segments."""
        if positions_3d.ndim == 1:
            positions_3d = positions_3d.reshape(-1, 3)

        num_joints = positions_3d.shape[0]
        if num_joints == 0:
            return

        pts = positions_3d.copy()
        if self.pose_center_checkbox.isChecked():
            root_xy = pts[0, :2].copy()
            pts[:, 0] -= root_xy[0]
            pts[:, 1] -= root_xy[1]

        self.joints_scatter.setData(pos=pts)

        line_pts = []
        for i_start, i_end in XSENS_BONES:
            if i_start < num_joints and i_end < num_joints:
                line_pts.append(pts[i_start])
                line_pts.append(pts[i_end])

        if line_pts:
            self.skeleton_lines.setData(pos=np.asarray(line_pts, dtype=np.float32))

    def stop_all(self):
        if self._pose_h5_file is not None:
            try:
                self._pose_h5_file.close()
            except Exception:
                pass
            self._pose_h5_file = None

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.stop_all()
        super().closeEvent(event)
