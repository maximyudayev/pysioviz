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

from pysioviz.components import BaseComponent


class DataComponent(BaseComponent):
  def __init__(self, unique_id: str, col_width: int):
    self._sync_offset = 0
    self._timestamps: np.ndarray | None = None
    self.read_data()
    super().__init__(unique_id=unique_id, col_width=col_width)

  @abstractmethod
  def read_data(self) -> None:
    """Read the component specific data from the files used in the constructor."""
    pass

  @abstractmethod
  def get_sync_info(self) -> dict:
    """Return synchronization info for this component."""
    pass

  @abstractmethod
  def set_truncation_points(self, start_idx: int, end_idx: int) -> None:
    """Set truncation points for the modality."""
    pass

  def get_timestamp_for_sync(self, sync_timestamp: float) -> int:
    """Find the index closest to a given timestamp with offset."""
    if self._timestamps is not None:
      time_diffs = np.abs(self._timestamps - sync_timestamp)
      closest_idx = np.argmin(time_diffs)
      # Apply offset
      offset_idx = closest_idx + self._sync_offset
      # Ensure within bounds
      offset_idx = max(0, min(len(self._timestamps) - 1, offset_idx))
      return int(offset_idx)
    else:
      return 0

  def set_sync_offset(self, offset: int) -> None:
    """Set synchronization offset for the component."""
    self._sync_offset = int(offset)

  def get_sync_offset(self) -> int:
    """Get current synchronization offset for the component."""
    return self._sync_offset
