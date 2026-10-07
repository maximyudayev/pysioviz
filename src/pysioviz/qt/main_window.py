"""
PysioViz Main Window: Qt-based desktop environment for offline multimodal replay,
alignment, and annotation of continuous recordings collected via HERMES.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import h5py

from PyQt6 import QtCore, QtGui, QtWidgets

from pysioviz.qt.playback_engine import PlaybackEngine
from pysioviz.qt.paged_sensor_reader import PagedSensorReader
from pysioviz.qt.widgets.video_grid_dock import VideoGridDock
from pysioviz.qt.widgets.egocentric_video_dock import EgocentricVideoDock
from pysioviz.qt.widgets.pose_3d_dock import Pose3DDock
from pysioviz.qt.widgets.notes_editor_dock import NotesEditorDock
from pysioviz.qt.widgets.sensor_plots_dock import SensorPlotsDock
from pysioviz.qt.widgets.annotation_timeline_dock import AnnotationTimelineDock
from pysioviz.qt.widgets.modality_dialog import AddModalityDialog
from pysioviz.utils.time_utils import format_relative_time


class PysiovizMainWindow(QtWidgets.QMainWindow):
    """Main window hosting the modular, dockable PysioViz workspace."""

    CAMERA_IDS = ['40478064', '40549960', '40549975', '40549976']

    def __init__(self, session_path: Optional[str] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle('PysioViz - Multimodal Annotation Dashboard')
        self.resize(1720, 980)

        # Enable dock nesting and customize corners for spanning top and bottom docks
        self.setDockNestingEnabled(True)
        self.setCorner(QtCore.Qt.Corner.TopLeftCorner, QtCore.Qt.DockWidgetArea.TopDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.TopRightCorner, QtCore.Qt.DockWidgetArea.TopDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.BottomLeftCorner, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.BottomRightCorner, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)

        # Central widget placeholder (all views are in dock widgets)
        central = QtWidgets.QWidget()
        central.setMaximumSize(0, 0)
        self.setCentralWidget(central)

        # Apply dark aesthetic theme inspired by AidWear
        self._apply_dark_theme()

        # Core playback engine
        self.playback_engine = PlaybackEngine(parent=self)

        # Build dock widgets
        self._create_docks()
        self._create_menus_and_toolbars()
        self._create_status_bar()

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Load initial session if provided or in environment
        target_path = session_path or os.environ.get('PYSIOVIZ_BASE_PATH')
        if target_path and os.path.isdir(target_path):
            self.load_session(target_path)

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0c10;
            }
            QDockWidget {
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(float.png);
                font-weight: 700;
                font-size: 11px;
                color: #e2e8f0;
            }
            QDockWidget::title {
                text-align: left;
                background-color: #1a1b26;
                padding: 5px 8px;
                border-bottom: 1px solid #282a36;
            }
            QMenuBar {
                background-color: #12131c;
                color: #e2e8f0;
                border-bottom: 1px solid #282a36;
                font-size: 12px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 10px;
            }
            QMenuBar::item:selected {
                background-color: #272738;
                color: #38bdf8;
            }
            QMenu {
                background-color: #181926;
                color: #f1f5f9;
                border: 1px solid #2d3142;
                font-size: 12px;
            }
            QMenu::item:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
            QStatusBar {
                background-color: #12131c;
                color: #94a3b8;
                border-top: 1px solid #282a36;
                font-size: 11px;
            }
        """)

    def _create_docks(self):
        """Construct the modular dockable views matching the default layout:
        - Top Left: Notes Editor (~19% width)
        - Top Center: External Video Grid (~56% width)
        - Top Right: First-Person Feed (top) and 3D Pose Skeleton (bottom) (~25% width)
        - Bottom: Annotation Timeline & Playback (full width, ~22% height)
        - Sensor Modalities: tabified with Timeline, hidden by default
        """
        # 1. Notes Editor Dock (Top Left)
        self.notes_dock = NotesEditorDock(self.playback_engine, self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, self.notes_dock)

        # 2. External Video Grid Dock (Top Center)
        self.video_grid_dock = VideoGridDock(self.playback_engine, self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, self.video_grid_dock)

        # 3. Egocentric Glasses Feed Dock (Top Right - Upper)
        self.ego_video_dock = EgocentricVideoDock(self.playback_engine, self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, self.ego_video_dock)

        # 4. 3D Kinematic Pose Skeleton Dock (Top Right - Lower)
        self.pose_3d_dock = Pose3DDock(self.playback_engine, self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, self.pose_3d_dock)

        # Split top area horizontally: Notes | Video Grid | Ego Video
        self.splitDockWidget(self.notes_dock, self.video_grid_dock, QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.video_grid_dock, self.ego_video_dock, QtCore.Qt.Orientation.Horizontal)

        # Split right column vertically: Ego Video above 3D Pose
        self.splitDockWidget(self.ego_video_dock, self.pose_3d_dock, QtCore.Qt.Orientation.Vertical)

        # 5. Annotation Timeline Dock (Bottom Area - Full Width)
        self.timeline_dock = AnnotationTimelineDock(self.playback_engine, self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.timeline_dock)

        # 6. Sensor Modalities Dock (Tabified with Timeline, hidden by default)
        self.sensor_plots_dock = SensorPlotsDock(self.playback_engine, self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.sensor_plots_dock)
        self.tabifyDockWidget(self.timeline_dock, self.sensor_plots_dock)
        self.timeline_dock.raise_()
        self.sensor_plots_dock.hide()

    def _create_menus_and_toolbars(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')

        open_session_action = QtGui.QAction('Open Session Directory...', self)
        open_session_action.setShortcut('Ctrl+O')
        open_session_action.triggered.connect(self._on_open_session_dialog)
        file_menu.addAction(open_session_action)

        add_modality_action = QtGui.QAction('Add Modality Stream...', self)
        add_modality_action.setShortcut('Ctrl+M')
        add_modality_action.triggered.connect(self._on_add_modality_dialog)
        file_menu.addAction(add_modality_action)

        file_menu.addSeparator()

        save_ann_action = QtGui.QAction('Save Annotations HDF5...', self)
        save_ann_action.setShortcut('Ctrl+S')
        save_ann_action.triggered.connect(self.timeline_dock._on_save_hdf5)
        file_menu.addAction(save_ann_action)

        load_ann_action = QtGui.QAction('Load Annotations HDF5...', self)
        load_ann_action.setShortcut('Ctrl+L')
        load_ann_action.triggered.connect(self.timeline_dock._on_load_hdf5)
        file_menu.addAction(load_ann_action)

        file_menu.addSeparator()

        exit_action = QtGui.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menubar.addMenu('&View')
        view_menu.addAction(self.video_grid_dock.toggleViewAction())
        view_menu.addAction(self.ego_video_dock.toggleViewAction())
        view_menu.addAction(self.pose_3d_dock.toggleViewAction())
        view_menu.addAction(self.sensor_plots_dock.toggleViewAction())
        view_menu.addAction(self.timeline_dock.toggleViewAction())
        view_menu.addAction(self.notes_dock.toggleViewAction())

        view_menu.addSeparator()
        reset_layout_action = QtGui.QAction('Reset Default Dock Layout', self)
        reset_layout_action.triggered.connect(self._reset_default_layout)
        view_menu.addAction(reset_layout_action)

        # Playback Menu
        playback_menu = menubar.addMenu('&Playback')

        toggle_play_action = QtGui.QAction('Play / Pause', self)
        toggle_play_action.setShortcut('Space')
        toggle_play_action.triggered.connect(self.playback_engine.toggle_play)
        playback_menu.addAction(toggle_play_action)

        step_fwd_action = QtGui.QAction('Step +1 Frame', self)
        step_fwd_action.setShortcut('Right')
        step_fwd_action.triggered.connect(lambda: self.playback_engine.step_forward(1))
        playback_menu.addAction(step_fwd_action)

        step_back_action = QtGui.QAction('Step -1 Frame', self)
        step_back_action.setShortcut('Left')
        step_back_action.triggered.connect(lambda: self.playback_engine.step_backward(1))
        playback_menu.addAction(step_back_action)

        step_fwd_10_action = QtGui.QAction('Step +10 Frames', self)
        step_fwd_10_action.setShortcut('PageDown')
        step_fwd_10_action.triggered.connect(lambda: self.playback_engine.step_forward(10))
        playback_menu.addAction(step_fwd_10_action)

        step_back_10_action = QtGui.QAction('Step -10 Frames', self)
        step_back_10_action.setShortcut('PageUp')
        step_back_10_action.triggered.connect(lambda: self.playback_engine.step_backward(10))
        playback_menu.addAction(step_back_10_action)

        # Help Menu
        help_menu = menubar.addMenu('&Help')
        about_action = QtGui.QAction('About PysioViz', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        status = self.statusBar()
        self.status_session_label = QtWidgets.QLabel('Session: None')
        self.status_session_label.setStyleSheet('color: #cbd5e1; margin-left: 6px;')
        status.addWidget(self.status_session_label, stretch=1)

        self.status_range_label = QtWidgets.QLabel('Range: --')
        self.status_range_label.setStyleSheet('color: #38bdf8; margin-right: 12px;')
        status.addPermanentWidget(self.status_range_label)

        self.status_fps_label = QtWidgets.QLabel('Status: READY')
        self.status_fps_label.setStyleSheet('color: #22c55e; font-weight: bold; margin-right: 12px;')
        status.addPermanentWidget(self.status_fps_label)

    def _setup_shortcuts(self):
        # Additional hotkeys
        pass

    def apply_proportional_layout(self):
        """Enforce standard layout proportions matching the target UI:
        - Top Row:
          - Left column: Notes Editor (~19% width)
          - Middle column: External Video Grid (~56% width)
          - Right column: First-Person Feed & 3D Pose Skeleton (~25% width)
          - Right column vertical split: Ego (~48% height) & 3D Pose (~52% height)
        - Bottom Row:
          - Annotation Timeline & Playback spanning full window width (~22% height)
        """
        total_w = self.width()
        total_h = self.height()
        if total_w <= 100 or total_h <= 100:
            return

        # 1. Vertical proportions: Top row (~78% height) vs Bottom Timeline (~22% height)
        h_timeline = max(180, int(total_h * 0.22))
        h_top = total_h - h_timeline
        self.resizeDocks(
            [self.video_grid_dock, self.timeline_dock],
            [h_top, h_timeline],
            QtCore.Qt.Orientation.Vertical,
        )

        # 2. Horizontal proportions across top row: Notes (19%) | Video Grid (56%) | Right Column (25%)
        w_notes = int(total_w * 0.19)
        w_grid = int(total_w * 0.56)
        w_right = total_w - w_notes - w_grid
        self.resizeDocks(
            [self.notes_dock, self.video_grid_dock, self.ego_video_dock],
            [w_notes, w_grid, w_right],
            QtCore.Qt.Orientation.Horizontal,
        )

        # 3. Vertical proportions in right column: Ego Video (top) & 3D Pose (bottom)
        h_ego = int(h_top * 0.48)
        h_pose = h_top - h_ego
        self.resizeDocks(
            [self.ego_video_dock, self.pose_3d_dock],
            [h_ego, h_pose],
            QtCore.Qt.Orientation.Vertical,
        )

    def showEvent(self, event: QtGui.QShowEvent):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.apply_proportional_layout)

    def _reset_default_layout(self):
        """Restore all docks to visible default layout matching target UI."""
        for dock in [
            self.notes_dock,
            self.video_grid_dock,
            self.ego_video_dock,
            self.pose_3d_dock,
            self.timeline_dock,
        ]:
            dock.show()
        self.sensor_plots_dock.hide()
        self.apply_proportional_layout()

    def _on_open_session_dialog(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Multimodal Session Directory', '')
        if dir_path:
            self.load_session(dir_path)

    def _on_add_modality_dialog(self):
        dialog = AddModalityDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.modality_config:
            cfg = dialog.modality_config
            if cfg['type'] == 'sensor':
                reader = PagedSensorReader(
                    hdf5_path=cfg['hdf5_path'],
                    channel_name=cfg['title'],
                    value_dataset_path=cfg['value_path'],
                    toa_dataset_path=cfg['toa_path'],
                )
                self.sensor_plots_dock.add_sensor_plot(
                    title=cfg['title'],
                    y_label=cfg['unit'],
                    reader=reader,
                )
                self.statusBar().showMessage(f'Added sensor stream: {cfg["title"]}', 4000)

    def load_session(self, session_path: str):
        """Automatically scan and populate multimodal streams from session directory."""
        path = Path(session_path)
        if not path.exists():
            return

        self.status_session_label.setText(f'Session: {path.name} ({path})')
        trial_toas_min: list[float] = []
        trial_toas_max: list[float] = []

        # 1. External Cameras (cameras.hdf5 and cameras_*.mkv)
        cam_h5_file = path / 'cameras.hdf5'
        if cam_h5_file.exists():
            try:
                with h5py.File(cam_h5_file, 'r') as f:
                    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    for i, cam_id in enumerate(self.CAMERA_IDS):
                        video_file = path / f'cameras_{cam_id}.mkv'
                        if not video_file.exists():
                            video_file = path / f'cameras_{cam_id}.mp4'

                        toa_ds_path = f'cameras/{cam_id}/toa_s'
                        if video_file.exists() and toa_ds_path in f:
                            cam_toas = np.asarray(f[toa_ds_path]).ravel().astype(np.float64)
                            if len(cam_toas) > 0:
                                trial_toas_min.append(float(cam_toas[0]))
                                trial_toas_max.append(float(cam_toas[-1]))

                            r, c = coords[i % 4]
                            self.video_grid_dock.add_camera(
                                camera_id=cam_id,
                                label=f'Cam {cam_id}',
                                video_path=str(video_file),
                                toas=cam_toas,
                                row=r,
                                col=c,
                            )
            except Exception as e:
                print(f'Error loading cameras: {e}', flush=True)

        # 2. First-Person Egocentric Video (glasses_ego.mkv & glasses.hdf5)
        glasses_video = path / 'glasses_ego.mkv'
        glasses_h5 = path / 'glasses.hdf5'
        if glasses_video.exists() and glasses_h5.exists():
            try:
                with h5py.File(glasses_h5, 'r') as f:
                    if 'glasses/ego/toa_s' in f:
                        ego_toas = np.asarray(f['glasses/ego/toa_s']).ravel().astype(np.float64)
                        if len(ego_toas) > 0:
                            trial_toas_min.append(float(ego_toas[0]))
                            trial_toas_max.append(float(ego_toas[-1]))
                        self.ego_video_dock.set_egocentric_video(video_path=str(glasses_video), toas=ego_toas)
            except Exception as e:
                print(f'Error loading glasses video: {e}', flush=True)

        # 3. 3D Reconstructed Pose Skeleton (mvn_analyze.hdf5)
        mvn_h5 = path / 'mvn_analyze.hdf5'
        if not mvn_h5.exists():
            mvn_h5 = path / 'mvn-analyze.hdf5'

        if mvn_h5.exists():
            self.pose_3d_dock.set_pose_dataset(
                hdf5_path=str(mvn_h5),
                pos_ds_path='mvn-analyze/xsens_pose/position',
                toa_ds_path='mvn-analyze/xsens_pose/toa_s',
            )

        # 4. Sensor Modalities (revalexo.hdf5)
        revalexo_h5 = path / 'revalexo.hdf5'
        if revalexo_h5.exists():
            try:
                # Add Torso Euler IMU
                reader_torso = PagedSensorReader(
                    hdf5_path=str(revalexo_h5),
                    channel_name='Torso IMU (Roll, Pitch, Yaw)',
                    value_dataset_path='revalexo/nicla_torso/euler',
                    toa_dataset_path='revalexo/nicla_torso/toa_s',
                )
                self.sensor_plots_dock.add_sensor_plot(
                    title='IMU: Torso Euler Angles',
                    y_label='deg',
                    reader=reader_torso,
                    channel_names=['Roll / X', 'Pitch / Y', 'Yaw / Z'],
                )

                # Add Shank Left Euler IMU
                reader_shank_l = PagedSensorReader(
                    hdf5_path=str(revalexo_h5),
                    channel_name='Left Shank IMU',
                    value_dataset_path='revalexo/nicla_shank_left/euler',
                    toa_dataset_path='revalexo/nicla_shank_left/toa_s',
                )
                self.sensor_plots_dock.add_sensor_plot(
                    title='IMU: Left Shank Euler Angles',
                    y_label='deg',
                    reader=reader_shank_l,
                    channel_names=['Roll / X', 'Pitch / Y', 'Yaw / Z'],
                )

                # Add Knee Motor Position & Velocity
                reader_knee_l = PagedSensorReader(
                    hdf5_path=str(revalexo_h5),
                    channel_name='Left Knee Motor Position',
                    value_dataset_path='revalexo/motor_knee_left/position',
                    toa_dataset_path='revalexo/motor_knee_left/timestamp',
                )
                self.sensor_plots_dock.add_sensor_plot(
                    title='Motor: Left Knee Joint Position',
                    y_label='rad / turns',
                    reader=reader_knee_l,
                    channel_names=['Knee Angle'],
                )

                # Add Knee Motor Current
                reader_knee_cur = PagedSensorReader(
                    hdf5_path=str(revalexo_h5),
                    channel_name='Left Knee Motor Current',
                    value_dataset_path='revalexo/motor_knee_left/current',
                    toa_dataset_path='revalexo/motor_knee_left/timestamp',
                )
                self.sensor_plots_dock.add_sensor_plot(
                    title='Motor: Left Knee Actuator Current',
                    y_label='Amps',
                    reader=reader_knee_cur,
                    channel_names=['Current'],
                )
            except Exception as e:
                print(f'Error loading revalexo sensors: {e}', flush=True)

        # 5. Calculate synchronized universal timeline bounds
        if trial_toas_min and trial_toas_max:
            global_min = max(trial_toas_min)  # latest start across all synchronized sensors
            global_max = min(trial_toas_max)  # earliest end
            if global_max > global_min:
                self.playback_engine.set_range(global_min, global_max, global_min)
                dur = global_max - global_min
                self.status_range_label.setText(
                    f'Duration: {format_relative_time(dur)} ({dur:.2f}s) | Range: {global_min:.2f}s – {global_max:.2f}s'
                )
            else:
                self.playback_engine.set_range(min(trial_toas_min), max(trial_toas_max))

        # Re-apply proportional layout after session widgets are populated
        QtCore.QTimer.singleShot(50, self.apply_proportional_layout)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            'About PysioViz',
            'PysioViz Desktop Visualization & Annotation Dashboard\n\n'
            'High-performance native Qt environment for offline, post-hoc replay,\n'
            'alignment, and annotation of continuous multimodal sensor data.',
        )

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Cleanly stop playback and all asynchronous worker threads upon window close."""
        self.playback_engine.pause()
        if hasattr(self, 'video_grid_dock'):
            self.video_grid_dock.stop_all()
        if hasattr(self, 'ego_video_dock'):
            self.ego_video_dock.stop_all()
        if hasattr(self, 'pose_3d_dock'):
            self.pose_3d_dock.stop_all()
        event.accept()
