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
# ############

from abc import abstractmethod
import numpy as np
from dash import Input, html

from pysioviz.components import BaseComponent
from pysioviz.utils.types import AlignmentInfo


class DataComponent(BaseComponent):
    def __init__(self, unique_id: str):
        self._toa_s: np.ndarray | None = None
        self.read_data()
        self._align_info = AlignmentInfo(0, len(self._toa_s))
        self._offset_s = 0.0
        super().__init__(unique_id=unique_id)

    @property
    def layout(self) -> html.Div:
        return self._layout

    @property
    def legend_name(self) -> html.Div:
        return self._legend_name

    @abstractmethod
    def read_data(self) -> None:
        """Read the component specific data from the files used in the constructor."""
        pass

    @abstractmethod
    def get_sync_info(self) -> dict:
        """Return synchronization info for this component."""
        pass

    @abstractmethod
    def make_click_input(self) -> Input:
        pass

    @abstractmethod
    def handle_click(self, ref_frame_timestamp, sync_timestamp) -> str:
        pass

    def get_frame_for_toa(self, sync_timestamp: float) -> int:
        """Find the index closest to a given timestamp with offset."""
        time_diffs = (self._toa_s - (sync_timestamp + self._offset_s)) <= 0
        return max(np.sum(time_diffs).item() - 1, 0)
    
    def set_offset(self, offset_ms: float) -> None:
        self._offset_s = offset_ms/1000

    def get_offset(self) -> float:
        return self._offset_s

    def set_align_info(self, alignment_info: AlignmentInfo) -> None:
        """Set alignment info for the component."""
        self._align_info = alignment_info

    def get_align_info(self) -> AlignmentInfo:
        """Get current alignment info for the component."""
        return self._align_info
