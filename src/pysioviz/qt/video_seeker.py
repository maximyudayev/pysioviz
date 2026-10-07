"""
On-Demand Video Frame Seeker using Subprocess FFmpeg and Shared Memory.
Interfaces on-disk video files through VideoDecoderSubprocessManager and
SharedVideoCircularBuffer, eliminating GUI blocking and GIL contention.
"""

from typing import Optional, Tuple
import threading
import numpy as np
import ffmpeg

from PyQt6 import QtCore, QtGui

from pysioviz.utils.cache import VideoDecoderSubprocessManager
from pysioviz.utils.types import HwAccelEnum


class VideoSeekerWorker(QtCore.QThread):
    """Background worker fetching on-demand frames via subprocess and shared memory."""

    frame_ready = QtCore.pyqtSignal(str, int, float, QtGui.QImage)

    def __init__(
        self,
        unique_id: str,
        video_path: str,
        toas: np.ndarray,
        fps: float = 30.0,
        hwaccel: str = HwAccelEnum.D3D12VA.value,
        prefetch_window_s: float = 2.0,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.unique_id = unique_id
        self.video_path = video_path
        self.toas = np.asarray(toas).squeeze()
        self.hwaccel = hwaccel

        # Get video properties via ffprobe
        self._width, self._height, self._fps, self._total_frames = self._get_video_properties(fallback_fps=fps)
        if len(self.toas) > 0:
            self._total_frames = max(self._total_frames, len(self.toas))

        self._prefetch_window_s = prefetch_window_s
        self._buf_len = max(30, round(self._fps * self._prefetch_window_s))

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending_request: Optional[Tuple[int, float]] = None
        self._running = True

        # Initialize background subprocess decoder with shared memory ring buffer
        self._decoder_mgr = VideoDecoderSubprocessManager(
            unique_id=self.unique_id,
            video_path=self.video_path,
            toas=self.toas,
            fps=self._fps,
            width=self._width,
            height=self._height,
            buf_len=self._buf_len,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def aspect_ratio(self) -> float:
        if self._height > 0:
            return self._width / self._height
        return 16.0 / 9.0

    def _get_video_properties(self, fallback_fps: float = 30.0) -> Tuple[int, int, float, int]:
        """Get video width, height, fps, and total frames using ffprobe."""
        try:
            probe = ffmpeg.probe(self.video_path)
            video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            fps_num, fps_denum = map(float, video_stream['r_frame_rate'].split('/'))
            fps = fps_num / fps_denum if fps_denum != 0 else fallback_fps
            duration = float(probe['format'].get('duration', len(self.toas) / fps))
            num_frames = round(duration * fps)
            return width, height, fps, num_frames
        except Exception as e:
            print(f'[{self.unique_id}] ffprobe error: {e}, using fallbacks', flush=True)
            return 1280, 720, fallback_fps, max(1, len(self.toas))

    def get_frame_for_toa(self, toa_s: float) -> int:
        """Find the closest frame index for a given `toa_s` timestamp."""
        if len(self.toas) == 0:
            return 0
        idx = int(np.searchsorted(self.toas, toa_s))
        if idx >= len(self.toas):
            return len(self.toas) - 1
        if idx > 0 and abs(self.toas[idx - 1] - toa_s) < abs(self.toas[idx] - toa_s):
            return idx - 1
        return idx

    def request_frame(self, frame_id: int, toa_s: float):
        """Request frame retrieval. Fast O(1) cache hit path returns instantly;

        cache misses are queued to the worker thread without blocking caller.
        """
        # Fast path: check if frame is already present in shared memory circular buffer
        cached = self._decoder_mgr.get_frame(frame_id)
        if cached is not None:
            frame_arr, actual_toa = cached
            image = QtGui.QImage(
                frame_arr.data,
                self._width,
                self._height,
                self._width * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            self.frame_ready.emit(self.unique_id, frame_id, actual_toa, image)
            return

        # Slow path: queue seek request to worker thread
        with self._lock:
            self._pending_request = (frame_id, toa_s)
            self._cond.notify()

    def run(self):
        """Worker thread processing cache-miss seek requests."""
        while self._running:
            with self._cond:
                while self._running and self._pending_request is None:
                    self._cond.wait(timeout=0.05)

                if not self._running:
                    break

                target_frame_id, target_toa_s = self._pending_request
                self._pending_request = None

            # Request seek from decoder subprocess
            self._decoder_mgr.request_seek(target_frame_id)
            res = self._decoder_mgr.wait_for_frame(target_frame_id, timeout_s=2.5)
            if res is not None:
                frame_arr, actual_toa = res
                image = QtGui.QImage(
                    frame_arr.data,
                    self._width,
                    self._height,
                    self._width * 3,
                    QtGui.QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(self.unique_id, target_frame_id, actual_toa, image)

    def stop(self):
        """Signal thread to terminate and stop decoder subprocess."""
        with self._lock:
            self._running = False
            self._cond.notify_all()
        self.wait(500)
        self._decoder_mgr.stop()
