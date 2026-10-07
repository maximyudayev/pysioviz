"""
Playback Engine for PysioViz desktop environment.
Manages deterministic, frame-accurate synchronization across all distributed multimodal
streams aligned by `toa_s` (UNIX time-of-arrival in seconds).
"""

import time
from typing import Optional
from PyQt6 import QtCore


class PlaybackEngine(QtCore.QObject):
    """Deterministic playback engine controlling multimodal time synchronization.

    Signals:
        time_changed(float): Emitted when current `toa_s` changes.
        range_changed(float, float): Emitted when timeline `(min_toa_s, max_toa_s)` changes.
        playback_state_changed(bool): Emitted when play/pause state changes.
        speed_changed(float): Emitted when playback speed changes.
        scrubbing_state_changed(bool): Emitted when interactive scrubbing starts or ends.
    """

    time_changed = QtCore.pyqtSignal(float)
    range_changed = QtCore.pyqtSignal(float, float)
    playback_state_changed = QtCore.pyqtSignal(bool)
    speed_changed = QtCore.pyqtSignal(float)
    scrubbing_state_changed = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        min_toa_s: float = 0.0,
        max_toa_s: float = 100.0,
        fps: float = 30.0,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._min_toa_s = float(min_toa_s)
        self._max_toa_s = float(max_toa_s)
        self._current_toa_s = self._min_toa_s
        self._nominal_fps = float(fps)
        self._frame_duration_s = 1.0 / max(1.0, self._nominal_fps)

        self._is_playing = False
        self._speed = 1.0
        self._is_scrubbing = False

        # Throttled sweeping variables
        self._last_scrub_emit_time = 0.0
        self._scrub_throttle_interval_s = 0.05  # 20 FPS during rapid scrubbing
        self._pending_scrub_toa_s: Optional[float] = None

        # Playback timer
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000.0 / self._nominal_fps))
        self._timer.timeout.connect(self._on_tick)
        self._last_tick_time: float = 0.0

    @property
    def min_toa_s(self) -> float:
        return self._min_toa_s

    @property
    def max_toa_s(self) -> float:
        return self._max_toa_s

    @property
    def current_toa_s(self) -> float:
        return self._current_toa_s

    @property
    def duration_s(self) -> float:
        return max(0.0, self._max_toa_s - self._min_toa_s)

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self._current_toa_s - self._min_toa_s)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def is_scrubbing(self) -> bool:
        return self._is_scrubbing

    def set_range(self, min_toa_s: float, max_toa_s: float, current_toa_s: Optional[float] = None):
        """Set the global timeline range based on synchronized trial start and end."""
        self._min_toa_s = float(min_toa_s)
        self._max_toa_s = max(self._min_toa_s, float(max_toa_s))
        if current_toa_s is not None:
            self._current_toa_s = min(max(float(current_toa_s), self._min_toa_s), self._max_toa_s)
        else:
            self._current_toa_s = min(max(self._current_toa_s, self._min_toa_s), self._max_toa_s)

        self.range_changed.emit(self._min_toa_s, self._max_toa_s)
        self.time_changed.emit(self._current_toa_s)

    def set_time(self, toa_s: float, emit: bool = True):
        """Set current playback timestamp directly (clamped to range)."""
        clamped = min(max(float(toa_s), self._min_toa_s), self._max_toa_s)
        if abs(clamped - self._current_toa_s) > 1e-7:
            self._current_toa_s = clamped
            if emit:
                self.time_changed.emit(self._current_toa_s)

    def set_speed(self, speed: float):
        """Change playback speed factor (e.g. 0.25, 0.5, 1.0, 2.0)."""
        self._speed = max(0.1, float(speed))
        self.speed_changed.emit(self._speed)

    def play(self):
        """Start continuous playback."""
        if not self._is_playing:
            if self._current_toa_s >= self._max_toa_s:
                self._current_toa_s = self._min_toa_s
            self._is_playing = True
            self._last_tick_time = time.perf_counter()
            self._timer.start()
            self.playback_state_changed.emit(True)

    def pause(self):
        """Pause playback."""
        if self._is_playing:
            self._is_playing = False
            self._timer.stop()
            self.playback_state_changed.emit(False)

    def toggle_play(self):
        """Toggle play/pause state."""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def step_forward(self, frames: int = 1):
        """Step playback forward by N nominal frames."""
        self.pause()
        dt = frames * self._frame_duration_s
        self.set_time(self._current_toa_s + dt)

    def step_backward(self, frames: int = 1):
        """Step playback backward by N nominal frames."""
        self.pause()
        dt = frames * self._frame_duration_s
        self.set_time(self._current_toa_s - dt)

    def start_scrubbing(self):
        """Call when user begins dragging the interactive scrubber."""
        if not self._is_scrubbing:
            self.pause()
            self._is_scrubbing = True
            self._last_scrub_emit_time = 0.0
            self.scrubbing_state_changed.emit(True)

    def scrub_to(self, toa_s: float):
        """Scrub to a specific timestamp with throttled UI sweeping performance.

        To prevent UI freezing during rapid timeline dragging, emissions are rate-limited.
        """
        clamped = min(max(float(toa_s), self._min_toa_s), self._max_toa_s)
        self._current_toa_s = clamped
        self._pending_scrub_toa_s = clamped

        now = time.perf_counter()
        if (now - self._last_scrub_emit_time) >= self._scrub_throttle_interval_s:
            self._last_scrub_emit_time = now
            self.time_changed.emit(self._current_toa_s)

    def finish_scrubbing(self, final_toa_s: Optional[float] = None):
        """Call when user releases the scrubber. Resumes full-accuracy high-FPS rendering."""
        if final_toa_s is not None:
            self.set_time(final_toa_s, emit=False)
        self._is_scrubbing = False
        self.scrubbing_state_changed.emit(False)
        # Immediate full-fidelity update on release
        self.time_changed.emit(self._current_toa_s)

    def _on_tick(self):
        """Timer callback advancing timeline deterministically based on real elapsed time."""
        now = time.perf_counter()
        dt = (now - self._last_tick_time) * self._speed
        self._last_tick_time = now

        new_t = self._current_toa_s + dt
        if new_t >= self._max_toa_s:
            self._current_toa_s = self._max_toa_s
            self.time_changed.emit(self._current_toa_s)
            self.pause()
        else:
            self._current_toa_s = new_t
            self.time_changed.emit(self._current_toa_s)

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format timestamp into HH:MM:SS.mmm."""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f'{hours:02d}:{minutes:02d}:{secs:06.3f}'
