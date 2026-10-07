"""
PysioViz Multimodal Visualization & Annotation Framework.
"""

from .qt import PysiovizMainWindow, PlaybackEngine

try:
    # For Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python < 3.8
    from importlib_metadata import PackageNotFoundError, version  # pyrefly: ignore [missing-import]
try:
    __version__ = version('pysioviz')
except PackageNotFoundError:
    __version__ = 'NA'

__all__ = ['__version__', 'PysiovizMainWindow', 'PlaybackEngine']
