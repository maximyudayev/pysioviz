"""
On-Demand Video Frame Seeker using FFmpeg and Cache module.
Interfaces on-disk video files through the Cache module using FFmpeg image2pipe extraction,
matching the caching approach of VideoComponent.py.
"""

from typing import Dict, Optional, Tuple
import threading
import numpy as np
import ffmpeg

from PyQt6 import QtCore, QtGui

from pysioviz.utils.cache import Cache
from pysioviz.utils.types import HwAccelEnum

# JPEG End of Image marker
EOI = b'\xff\xd9'


class VideoSeekerWorker(QtCore.QThread):
    """Background worker fetching on-demand frames via Cache and FFmpeg image2pipe."""

    frame_ready = QtCore.pyqtSignal(str, int, float, QtGui.QImage)

    def __init__(
        self,
        unique_id: str,
        video_path: str,
        toas: np.ndarray,
        fps: float = 30.0,
        hwaccel: str = HwAccelEnum.D3D12VA.value,
        prefetch_window_s: float = 10.0,
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

        self._empty_frame = np.zeros([self._height, self._width, 3], np.uint8)

        self._prefetch_window_s = prefetch_window_s
        self._num_prefetch_frames = round(self._fps * self._prefetch_window_s)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending_request: Optional[Tuple[int, float]] = None
        self._running = True

        # Initialize FFmpeg cache
        self._create_ffmpeg_cacher(hwaccel=self.hwaccel)

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

    def _create_ffmpeg_cacher(self, hwaccel: str):
        """Create FFmpeg decode cache prefetching a window centered on cache miss."""

        def _decode(frame_id: int) -> Dict[int, bytes]:
            timestamp_start = frame_id / self._fps
            try:
                buf, _ = (
                    ffmpeg.input(
                        filename=self.video_path,
                        hwaccel=hwaccel,
                        ss=timestamp_start,
                    )
                    .output(
                        'pipe:',
                        format='image2pipe',
                        vframes=self._num_prefetch_frames,
                    )
                    .run(capture_stdout=True, quiet=True)
                )
            except Exception:
                # Fallback to software decoding if hardware acceleration fails
                buf, _ = (
                    ffmpeg.input(
                        filename=self.video_path,
                        ss=timestamp_start,
                    )
                    .output(
                        'pipe:',
                        format='image2pipe',
                        vframes=self._num_prefetch_frames,
                    )
                    .run(capture_stdout=True, quiet=True)
                )

            new_cache = dict(
                zip(
                    range(frame_id, frame_id + self._num_prefetch_frames),
                    map(lambda frame: frame + EOI, buf.split(EOI)[:-1]),
                )
            )
            return new_cache

        self._cache = Cache(
            fetch_fn=_decode,
            fetch_offset=round(self._num_prefetch_frames / 3),
        )
        self._cache.start()

    def _get_frame(self, frame_id: int) -> bytes:
        """Get raw jpeg bytes of frame at a specific index."""
        if frame_id < 0:
            frame_id = 0
        elif frame_id >= self._total_frames:
            frame_id = self._total_frames - 1

        try:
            return self._cache.get_data(frame_id)
        except Exception as e:
            print(f'[{self.unique_id}] Error getting frame {frame_id}: {e}', flush=True)
            return self._empty_frame.tobytes()

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
        """Queue a request for frame retrieval. Collapses pending requests during rapid sweeping."""
        with self._lock:
            self._pending_request = (frame_id, toa_s)
            self._cond.notify()

    def run(self):
        """Worker thread processing requested frames and emitting QImage signals."""
        while self._running:
            with self._cond:
                while self._running and self._pending_request is None:
                    self._cond.wait(timeout=0.1)

                if not self._running:
                    break

                target_frame_id, target_toa_s = self._pending_request
                self._pending_request = None

            jpeg_bytes = self._get_frame(target_frame_id)
            if jpeg_bytes:
                image = QtGui.QImage()
                if image.loadFromData(jpeg_bytes, 'JPEG'):
                    actual_toa_s = target_toa_s
                    if len(self.toas) > 0 and 0 <= target_frame_id < len(self.toas):
                        actual_toa_s = float(self.toas[target_frame_id])
                    self.frame_ready.emit(self.unique_id, target_frame_id, actual_toa_s, image)

    def stop(self):
        """Signal thread to terminate and wait."""
        with self._lock:
            self._running = False
            self._cond.notify_all()
        self.wait(500)
