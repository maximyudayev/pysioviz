"""
Notes Editor Dock Widget for entering natural language annotations tied either to
the currently visualized absolute frame or a user-selected temporal window.
"""

from dataclasses import asdict, dataclass
import json
import time
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from pysioviz.qt.playback_engine import PlaybackEngine


@dataclass
class NaturalLanguageNote:
    note_id: str
    created_at: float
    start_toa_s: float
    end_toa_s: Optional[float]
    tag: str
    text: str


class NotesEditorDock(QtWidgets.QDockWidget):
    """Dockable natural language notes editor for multimodal recordings."""

    note_added = QtCore.pyqtSignal(object)
    note_deleted = QtCore.pyqtSignal(str)

    def __init__(
        self,
        playback_engine: PlaybackEngine,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__('Notes Editor', parent)
        self.setObjectName('NotesEditorDock')
        self.playback_engine = playback_engine
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)

        self._notes: List[NaturalLanguageNote] = []

        self._build_ui()
        self.playback_engine.time_changed.connect(self._on_time_changed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(260, 480)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(180, 200)

    def _build_ui(self):
        container = QtWidgets.QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #12131c;
                color: #f1f5f9;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #1a1b26;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #38bdf8;
            }
            QPushButton {
                background-color: #272738;
                color: #e2e8f0;
                border: 1px solid #3b3f54;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #35394d;
                border-color: #38bdf8;
                color: #ffffff;
            }
            QRadioButton {
                color: #cbd5e1;
                font-size: 11px;
                spacing: 5px;
            }
            QRadioButton:hover {
                color: #ffffff;
            }
            QRadioButton:checked {
                color: #38bdf8;
                font-weight: 600;
            }
            QRadioButton::indicator {
                width: 12px;
                height: 12px;
                border-radius: 7px;
                border: 2px solid #475569;
                background-color: #1a1b26;
            }
            QRadioButton::indicator:hover {
                border-color: #38bdf8;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #38bdf8;
                background-color: #38bdf8;
            }
        """)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header card
        header_card = QtWidgets.QFrame()
        header_card.setStyleSheet('background-color: #181926; border-radius: 6px; padding: 4px;')
        h_layout = QtWidgets.QVBoxLayout(header_card)
        h_layout.setContentsMargins(4, 4, 4, 4)
        h_layout.setSpacing(6)

        title = QtWidgets.QLabel('NOTES & ANNOTATIONS')
        title.setStyleSheet('font-size: 11px; font-weight: 700; color: #38bdf8;')
        h_layout.addWidget(title)

        # Mode selection: Frame vs Window
        mode_layout = QtWidgets.QHBoxLayout()
        self.radio_frame = QtWidgets.QRadioButton('Frame')
        self.radio_frame.setChecked(True)
        self.radio_window = QtWidgets.QRadioButton('Window')
        mode_layout.addWidget(self.radio_frame)
        mode_layout.addWidget(self.radio_window)
        h_layout.addLayout(mode_layout)

        # Timestamp Inputs
        time_grid = QtWidgets.QGridLayout()
        time_grid.setContentsMargins(0, 0, 0, 0)
        time_grid.setSpacing(4)

        time_grid.addWidget(QtWidgets.QLabel('Start:'), 0, 0)
        self.start_input = QtWidgets.QLineEdit()
        self.start_input.setPlaceholderText('Start ToA (s)')
        time_grid.addWidget(self.start_input, 0, 1)

        self.btn_set_start = QtWidgets.QPushButton('Set')
        self.btn_set_start.setToolTip('Set to current playback time')
        self.btn_set_start.clicked.connect(self._set_start_to_current)
        time_grid.addWidget(self.btn_set_start, 0, 2)

        time_grid.addWidget(QtWidgets.QLabel('End:'), 1, 0)
        self.end_input = QtWidgets.QLineEdit()
        self.end_input.setPlaceholderText('End ToA (s)')
        self.end_input.setEnabled(False)
        time_grid.addWidget(self.end_input, 1, 1)

        self.btn_set_end = QtWidgets.QPushButton('Set')
        self.btn_set_end.setToolTip('Set to current playback time')
        self.btn_set_end.setEnabled(False)
        self.btn_set_end.clicked.connect(self._set_end_to_current)
        time_grid.addWidget(self.btn_set_end, 1, 2)

        h_layout.addLayout(time_grid)

        self.radio_frame.toggled.connect(self._on_mode_toggled)

        # Tag category
        tag_row = QtWidgets.QHBoxLayout()
        tag_row.addWidget(QtWidgets.QLabel('Tag:'))
        self.tag_combo = QtWidgets.QComboBox()
        self.tag_combo.setEditable(True)
        self.tag_combo.addItems(
            ['Observation', 'Gait Event', 'Artifact / Noise', 'Clinical Note', 'Subject Feedback', 'Anomaly']
        )
        tag_row.addWidget(self.tag_combo)
        h_layout.addLayout(tag_row)

        # Note Text
        self.note_edit = QtWidgets.QTextEdit()
        self.note_edit.setPlaceholderText('Enter natural language notes and multimodal observations here...')
        self.note_edit.setMinimumHeight(60)
        h_layout.addWidget(self.note_edit)

        # Save Note Button
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add_note = QtWidgets.QPushButton('➕ Add Note')
        self.btn_add_note.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: white;
                border: 1px solid #38bdf8;
                font-weight: bold;
                padding: 5px 14px;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_add_note.clicked.connect(self._on_add_note)
        btn_layout.addWidget(self.btn_add_note)

        self.btn_export = QtWidgets.QPushButton('Export Notes')
        self.btn_export.clicked.connect(self._on_export_notes)
        btn_layout.addWidget(self.btn_export)
        h_layout.addLayout(btn_layout)

        layout.addWidget(header_card)

        # Notes Table List
        notes_label = QtWidgets.QLabel('SAVED NOTES (Click to Seek)')
        notes_label.setStyleSheet('font-size: 11px; font-weight: 700; color: #94a3b8;')
        layout.addWidget(notes_label)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Timestamp', 'Tag', 'Note', 'Action'])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #181926;
                gridline-color: #2d3142;
                border: 1px solid #2d3142;
                border-radius: 4px;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #212330;
                color: #94a3b8;
                border: 1px solid #2d3142;
                padding: 3px;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #0369a1;
                color: #ffffff;
            }
        """)
        layout.addWidget(self.table, stretch=1)

        self.setWidget(container)

    def _on_time_changed(self, current_toa_s: float):
        if self.radio_frame.isChecked() and not self.start_input.hasFocus():
            self.start_input.setText(f'{current_toa_s:.4f}')

    def _on_mode_toggled(self, checked: bool):
        is_window = self.radio_window.isChecked()
        self.end_input.setEnabled(is_window)
        self.btn_set_end.setEnabled(is_window)

    def _set_start_to_current(self):
        self.start_input.setText(f'{self.playback_engine.current_toa_s:.4f}')

    def _set_end_to_current(self):
        self.end_input.setText(f'{self.playback_engine.current_toa_s:.4f}')

    def _on_add_note(self):
        text = self.note_edit.toPlainText().strip()
        if not text:
            return

        try:
            start_toa = float(self.start_input.text().strip())
        except ValueError:
            start_toa = self.playback_engine.current_toa_s

        end_toa = None
        if self.radio_window.isChecked():
            try:
                end_toa = float(self.end_input.text().strip())
            except ValueError:
                end_toa = start_toa + 1.0

        tag = self.tag_combo.currentText().strip() or 'General'
        note = NaturalLanguageNote(
            note_id=f'note_{int(time.time() * 1000)}',
            created_at=time.time(),
            start_toa_s=start_toa,
            end_toa_s=end_toa,
            tag=tag,
            text=text,
        )

        self._notes.append(note)
        self._refresh_table()
        self.note_edit.clear()
        self.note_added.emit(note)

    def _refresh_table(self):
        self.table.setRowCount(len(self._notes))
        for row, n in enumerate(self._notes):
            if n.end_toa_s is not None:
                t_str = f'{n.start_toa_s:.2f}s - {n.end_toa_s:.2f}s'
            else:
                t_str = f'{n.start_toa_s:.2f}s'

            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(t_str))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(n.tag))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(n.text))

            del_btn = QtWidgets.QPushButton('✕')
            del_btn.setMaximumWidth(28)
            del_btn.setStyleSheet('color: #ef4444; font-weight: bold;')
            del_btn.clicked.connect(lambda _, idx=row: self._delete_note_at(idx))
            self.table.setCellWidget(row, 3, del_btn)

    def _delete_note_at(self, idx: int):
        if 0 <= idx < len(self._notes):
            n = self._notes.pop(idx)
            self._refresh_table()
            self.note_deleted.emit(n.note_id)

    def _on_table_double_clicked(self, row: int, _):
        if 0 <= row < len(self._notes):
            target_toa = self._notes[row].start_toa_s
            self.playback_engine.set_time(target_toa)

    def _on_export_notes(self):
        if not self._notes:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Notes', 'notes.json', 'JSON Files (*.json)')
        if fn:
            data = [asdict(n) for n in self._notes]
            with open(fn, 'w') as f:
                json.dump(data, f, indent=2)
            QtWidgets.QMessageBox.information(self, 'Exported', f'Successfully exported {len(data)} notes.')
