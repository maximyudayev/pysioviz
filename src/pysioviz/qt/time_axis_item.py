"""
PyQtGraph AxisItem displaying timestamps formatted as HH:MM:SS.mmm relative to experiment start.
"""

from typing import List
import pyqtgraph as pg
from pysioviz.utils.time_utils import format_relative_time


class RelativeTimeAxisItem(pg.AxisItem):
    """PyQtGraph bottom axis visualizing elapsed time in HH:MM:SS.mmm format."""

    def __init__(self, orientation: str = 'bottom', *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)

    def tickStrings(self, values: List[float], scale: float, spacing: float) -> List[str]:
        return [format_relative_time(float(v)) for v in values]
