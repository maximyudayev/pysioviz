############
#
# Copyright (c) 2026 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2026 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
############

import argparse
import os
import sys

from PyQt6 import QtCore, QtGui, QtWidgets
from pysioviz.qt.main_window import PysiovizMainWindow


def main():
    parser = argparse.ArgumentParser(
        description='PysioViz - Native Qt Desktop Multimodal Replay & Annotation Environment'
    )
    parser.add_argument(
        '--base-path',
        '-p',
        type=str,
        default=os.environ.get('PYSIOVIZ_BASE_PATH', 'C:/Users/maxim/Desktop/prebeta/manual/raw/trial_0'),
        help='Path to multimodal recording session directory (containing HDF5 and MKV files)',
    )
    args = parser.parse_args()

    # Native Qt PysiovizMainWindow Desktop App
    # Enable High DPI scaling
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('PysioViz - HERMES')
    app.setOrganizationName('KU Leuven eMedia Lab')

    # Set application font
    font = QtGui.QFont('Segoe UI', 10)
    app.setFont(font)

    session_path = args.base_path if os.path.isdir(args.base_path) else None
    window = PysiovizMainWindow(session_path=session_path)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
