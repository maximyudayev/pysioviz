"""
Qt Desktop Architecture for PysioViz multimodal replay, alignment, and annotation.
"""

from .main_window import PysiovizMainWindow
from .playback_engine import PlaybackEngine

__all__ = ['PysiovizMainWindow', 'PlaybackEngine']
