"""
Add Modality Dialog implementing Decoupled Disk Linkage.
Allows users to instantiate a new QDockWidget, select arbitrary HDF5 or MKV/MP4 source files,
and specify modality paths to plot on demand.
"""

from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtWidgets
from pysioviz.qt.paged_sensor_reader import PagedSensorReader


class AddModalityDialog(QtWidgets.QDialog):
    """Dialog for creating and docking a new arbitrary modality stream."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle('Add Modality Stream')
        self.resize(520, 380)

        self.modality_config: Optional[Dict[str, Any]] = None

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #12131c;
                color: #f1f5f9;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QLineEdit, QComboBox {
                background-color: #1a1b26;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton {
                background-color: #272738;
                color: #e2e8f0;
                border: 1px solid #3b3f54;
                border-radius: 4px;
                padding: 5px 14px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #35394d;
                border-color: #38bdf8;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Modality Type Selection
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel('Modality Type:'))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(['HDF5 Sensor Plot', 'Video Feed (MKV/MP4)'])
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Stacked pages for parameters
        self.stack = QtWidgets.QStackedWidget()

        # Page 0: Sensor Plot
        sensor_widget = QtWidgets.QWidget()
        s_layout = QtWidgets.QFormLayout(sensor_widget)

        file_row = QtWidgets.QHBoxLayout()
        self.sensor_file_edit = QtWidgets.QLineEdit()
        self.sensor_file_edit.setPlaceholderText('Select .hdf5 file')
        btn_sensor_browse = QtWidgets.QPushButton('Browse...')
        btn_sensor_browse.clicked.connect(self._browse_sensor_file)
        file_row.addWidget(self.sensor_file_edit)
        file_row.addWidget(btn_sensor_browse)
        s_layout.addRow('HDF5 File:', file_row)

        self.sensor_stream_combo = QtWidgets.QComboBox()
        self.sensor_stream_combo.setEditable(True)
        self.sensor_stream_combo.setPlaceholderText('Select or enter dataset path (e.g. revalexo/nicla_torso/euler)')
        s_layout.addRow('Value Dataset:', self.sensor_stream_combo)

        self.sensor_toa_edit = QtWidgets.QLineEdit()
        self.sensor_toa_edit.setPlaceholderText('ToA dataset path (e.g. revalexo/nicla_torso/toa_s)')
        s_layout.addRow('ToA Dataset:', self.sensor_toa_edit)

        self.sensor_title_edit = QtWidgets.QLineEdit()
        self.sensor_title_edit.setPlaceholderText('Plot Title (e.g. Torso IMU Euler)')
        s_layout.addRow('Title:', self.sensor_title_edit)

        self.sensor_unit_edit = QtWidgets.QLineEdit()
        self.sensor_unit_edit.setPlaceholderText('Y-Axis Unit (e.g. deg or m/s²)')
        s_layout.addRow('Y-Unit:', self.sensor_unit_edit)

        self.stack.addWidget(sensor_widget)

        # Page 1: Video Feed
        video_widget = QtWidgets.QWidget()
        v_layout = QtWidgets.QFormLayout(video_widget)

        vfile_row = QtWidgets.QHBoxLayout()
        self.video_file_edit = QtWidgets.QLineEdit()
        self.video_file_edit.setPlaceholderText('Select video file (.mkv, .mp4)')
        btn_video_browse = QtWidgets.QPushButton('Browse...')
        btn_video_browse.clicked.connect(self._browse_video_file)
        vfile_row.addWidget(self.video_file_edit)
        vfile_row.addWidget(btn_video_browse)
        v_layout.addRow('Video File:', vfile_row)

        vh5_row = QtWidgets.QHBoxLayout()
        self.video_h5_edit = QtWidgets.QLineEdit()
        self.video_h5_edit.setPlaceholderText('Associated timestamps .hdf5 file')
        btn_vh5_browse = QtWidgets.QPushButton('Browse...')
        btn_vh5_browse.clicked.connect(self._browse_video_h5_file)
        vh5_row.addWidget(self.video_h5_edit)
        vh5_row.addWidget(btn_vh5_browse)
        v_layout.addRow('Timestamps HDF5:', vh5_row)

        self.video_toa_edit = QtWidgets.QLineEdit()
        self.video_toa_edit.setPlaceholderText('ToA dataset path (e.g. cameras/40478064/toa_s)')
        v_layout.addRow('ToA Dataset:', self.video_toa_edit)

        self.video_name_edit = QtWidgets.QLineEdit()
        self.video_name_edit.setPlaceholderText('Camera Name / Unique ID')
        v_layout.addRow('Camera Name:', self.video_name_edit)

        self.stack.addWidget(video_widget)

        layout.addWidget(self.stack)

        # Action Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QtWidgets.QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        self.btn_confirm = QtWidgets.QPushButton('Add Modality')
        self.btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.btn_confirm.clicked.connect(self._on_confirm)
        btn_row.addWidget(self.btn_confirm)

        layout.addLayout(btn_row)

    def _on_type_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def _browse_sensor_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select HDF5 File', '', 'HDF5 Files (*.hdf5 *.h5)')
        if fn:
            self.sensor_file_edit.setText(fn)
            # Auto-discover streams
            streams = PagedSensorReader.auto_discover_streams(fn)
            self.sensor_stream_combo.clear()
            for s in streams:
                self.sensor_stream_combo.addItem(f'{s["value_dataset"]} ({s["shape"]})', s)

            if streams:
                self._on_sensor_stream_selected(0)
                self.sensor_stream_combo.currentIndexChanged.connect(self._on_sensor_stream_selected)

    def _on_sensor_stream_selected(self, idx: int):
        data = self.sensor_stream_combo.itemData(idx)
        if data:
            self.sensor_toa_edit.setText(data['toa_dataset'])
            nice_title = data['value_dataset'].replace('/', ' - ').title()
            self.sensor_title_edit.setText(nice_title)
            self.sensor_unit_edit.setText('raw')

    def _browse_video_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.mkv *.mp4 *.avi)')
        if fn:
            self.video_file_edit.setText(fn)
            if not self.video_name_edit.text():
                self.video_name_edit.setText(QtCore.QFileInfo(fn).baseName())

    def _browse_video_h5_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Timestamp HDF5 File', '', 'HDF5 Files (*.hdf5 *.h5)'
        )
        if fn:
            self.video_h5_edit.setText(fn)

    def _on_confirm(self):
        if self.stack.currentIndex() == 0:
            # Sensor
            h5_path = self.sensor_file_edit.text().strip()
            data = self.sensor_stream_combo.currentData()
            val_path = data['value_dataset'] if data else self.sensor_stream_combo.currentText().strip()
            toa_path = self.sensor_toa_edit.text().strip()
            title = self.sensor_title_edit.text().strip() or 'Sensor Plot'
            unit = self.sensor_unit_edit.text().strip() or 'units'

            if not h5_path or not val_path or not toa_path:
                QtWidgets.QMessageBox.warning(self, 'Incomplete', 'Please specify the HDF5 file and datasets.')
                return

            self.modality_config = {
                'type': 'sensor',
                'hdf5_path': h5_path,
                'value_path': val_path,
                'toa_path': toa_path,
                'title': title,
                'unit': unit,
            }
        else:
            # Video
            v_path = self.video_file_edit.text().strip()
            vh5_path = self.video_h5_edit.text().strip()
            toa_path = self.video_toa_edit.text().strip()
            name = self.video_name_edit.text().strip() or 'External Camera'

            if not v_path:
                QtWidgets.QMessageBox.warning(self, 'Incomplete', 'Please select a video file.')
                return

            self.modality_config = {
                'type': 'video',
                'video_path': v_path,
                'hdf5_path': vh5_path,
                'toa_path': toa_path,
                'name': name,
            }

        self.accept()
