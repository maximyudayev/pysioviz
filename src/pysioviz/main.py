"""Interactive graphical user interface for HERMES data annotation and visualization.

Usage:
    uv run -m pysioviz -p <path_to_hermes_data_folder>
"""

import ctypes
import argparse
import os
import sys
from pathlib import Path

try:
    # For Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python < 3.8
    # pyrefly: ignore [missing-import]
    from importlib_metadata import PackageNotFoundError, version
try:
    __version__ = version('pysioviz')
except PackageNotFoundError:
    __version__ = 'NA'

from PyQt6 import QtCore, QtGui, QtWidgets
from pysioviz.qt.main_window import PysiovizMainWindow

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
IMAGES_DIR = PACKAGE_DIR / 'assets' if (PACKAGE_DIR / 'assets').exists() else REPO_ROOT / 'assets'


def _get_icon(
    app: QtWidgets.QApplication,
) -> QtGui.QIcon:
    is_dark = app.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
    icon_path = IMAGES_DIR / f'icon_{"dark" if is_dark else "light"}.png'
    return QtGui.QIcon(str(icon_path))


def _update_icons(
    app: QtWidgets.QApplication,
    window: QtWidgets.QMainWindow,
) -> None:
    icon = _get_icon(app)
    app.setWindowIcon(icon)
    window.setWindowIcon(icon)


def main(argv: list[str] | None = None):
    # Decouple from python.exe so Windows taskbar shows custom icon
    if sys.platform == 'win32':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f'kuleuven.emedia.pysioviz.v{__version__.replace(".", "_")}'
            )
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description='PysioViz - Native Qt Desktop Multimodal Replay & Annotation Environment'
    )
    parser.add_argument(
        '--base-path',
        '-p',
        type=str,
        required=True,
        help='Path to multimodal recording session directory (containing HDF5 and MKV files)',
    )
    args = parser.parse_args(argv)

    # Native Qt PysiovizMainWindow Desktop App
    # Enable High DPI scaling
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('PysioViz - HERMES')
    app.setOrganizationName('KU Leuven eMedia Lab')
    app.setApplicationVersion(__version__)

    # Set application font
    font = QtGui.QFont('Segoe UI', 10)
    app.setFont(font)

    session_path = args.base_path if os.path.isdir(args.base_path) else None
    window = PysiovizMainWindow(session_path=session_path)

    # Force creation of native Win32 HWND and apply window icon
    window.winId()
    _update_icons(app, window)

    # React to OS theme changes
    app.styleHints().colorSchemeChanged.connect(lambda _: _update_icons(app, window))

    window.show()

    # Re-send WM_SETICON after startup once the event loop has settled
    #   to force Windows Taskbar to redraw and replace any fallback python.exe icon
    QtCore.QTimer.singleShot(150, lambda: _update_icons(app, window))

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
