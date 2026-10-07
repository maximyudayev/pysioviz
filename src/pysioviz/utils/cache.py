############
#
# Copyright (c) 2024-2026 Maxim Yudayev and KU Leuven eMedia Lab
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

"""Shared memory circular buffer and subprocess-based video decoder.

Provides zero-copy frame sharing between a background FFmpeg decoder subprocess
and the Qt UI process, preventing UI blocking and ensuring smooth 1x playback.
"""

from collections import defaultdict
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from queue import Empty, Queue
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from pysioviz.utils.types import DataRequest


class SharedVideoCircularBuffer:
    """Zero-copy circular frame buffer allocated in shared memory.

    Structured similarly to SharedTensorCircularBuffer in HERMES aidwear:
    holds raw RGB frame slots, timestamps, and sequence IDs with
    multiprocessing synchronization for non-blocking UI access.
    """

    def __init__(
        self,
        name: str,
        buf_len: int,
        width: int,
        height: int,
        channels: int = 3,
        create: bool = True,
    ):
        self.name = name
        self.buf_len = int(buf_len)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.frame_bytes = self.width * self.height * self.channels
        self.total_bytes = self.buf_len * self.frame_bytes
        self.is_owner = create

        if create:
            # Clean up potential leftover segment with the same name
            try:
                old_shm = shared_memory.SharedMemory(name=self.name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_bytes)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)

        self.frames = np.ndarray(
            (self.buf_len, self.height, self.width, self.channels),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )

        self._closed = False

    def get_frame(
        self,
        frame_id: int,
        slot_fids: Any,
        slot_toas: Any,
        slot_valid: Any,
        lock: Any,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """O(1) lookup of a frame in the circular buffer.

        Returns (frame_view, toa_s) if available and valid, else None.
        """
        slot_idx = frame_id % self.buf_len
        with lock:
            if slot_fids[slot_idx] == frame_id and slot_valid[slot_idx] == 1:
                toa_s = float(slot_toas[slot_idx])
                return self.frames[slot_idx], toa_s
        return None

    def close(self):
        """Release buffer and close shared memory handle."""
        if not self._closed:
            self._closed = True
            try:
                # Release numpy buffer reference before closing mmap
                del self.frames
            except Exception:
                pass
            try:
                self.shm.close()
            except Exception:
                pass
            if self.is_owner:
                try:
                    self.shm.unlink()
                except Exception:
                    pass


def _video_decoder_worker_entry(
    video_path: str,
    toas_arr: Optional[np.ndarray],
    fps: float,
    shm_name: str,
    width: int,
    height: int,
    buf_len: int,
    cmd_queue: mp.Queue,
    slot_fids: Any,
    slot_toas: Any,
    slot_valid: Any,
    write_head: Any,
    read_head: Any,
    new_frame_event: Any,
    metadata_lock: Any,
):
    """Subprocess main loop: streams FFmpeg raw RGB frames into shared memory."""
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        frame_bytes = width * height * 3
        frames = np.ndarray((buf_len, height, width, 3), dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        print(f'[VideoDecoderSubprocess] Init error: {e}', flush=True)
        return

    toas_len = len(toas_arr) if toas_arr is not None else 0
    ffmpeg_proc: Optional[subprocess.Popen] = None
    cur_fid = 0
    running = True

    try:
        while running:
            # 1. Check commands, collapsing rapid seek requests
            seek_target = None
            while not cmd_queue.empty():
                try:
                    cmd, arg = cmd_queue.get_nowait()
                    if cmd == 'stop':
                        running = False
                        break
                    elif cmd == 'seek':
                        seek_target = int(arg)
                except Empty:
                    break
                except Exception:
                    break

            if not running:
                break

            # 2. Handle seek request
            if seek_target is not None:
                if ffmpeg_proc is not None:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait()
                    except Exception:
                        pass
                    ffmpeg_proc = None

                cur_fid = max(0, seek_target)

                # Invalidate buffer slots
                with metadata_lock:
                    for i in range(buf_len):
                        slot_valid[i] = 0
                        slot_fids[i] = -1

                # Calculate relative start timestamp in video file (0.0 at file start)
                start_s = max(0.0, cur_fid / fps)

                cmd = [
                    'ffmpeg',
                    '-y',
                    '-ss',
                    f'{start_s:.4f}',
                    '-i',
                    video_path,
                    '-vf',
                    f'scale={width}:{height}',
                    '-f',
                    'rawvideo',
                    '-pix_fmt',
                    'rgb24',
                    '-',
                ]
                try:
                    ffmpeg_proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        bufsize=frame_bytes * 2,
                    )
                except Exception as e:
                    print(f'[VideoDecoderSubprocess] FFmpeg spawn error: {e}', flush=True)
                    ffmpeg_proc = None

            # 3. Stream frames if FFmpeg is active
            if ffmpeg_proc is not None:
                # Backpressure: don't get more than buf_len ahead of read_head
                with metadata_lock:
                    rh = read_head.value
                if (cur_fid - rh) >= buf_len:
                    time.sleep(0.005)
                    continue

                slot_idx = cur_fid % buf_len
                view = memoryview(frames[slot_idx])
                try:
                    nread = ffmpeg_proc.stdout.readinto(view)
                except Exception:
                    nread = 0
                finally:
                    del view

                if nread < frame_bytes:
                    # Video stream ended or pipe broken
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait()
                    except Exception:
                        pass
                    ffmpeg_proc = None
                else:
                    # Calculate timestamp for this frame
                    if toas_len > 0 and 0 <= cur_fid < toas_len:
                        cur_toa = float(toas_arr[cur_fid])
                    else:
                        cur_toa = cur_fid / fps

                    with metadata_lock:
                        slot_fids[slot_idx] = cur_fid
                        slot_toas[slot_idx] = cur_toa
                        slot_valid[slot_idx] = 1
                        write_head.value = slot_idx

                    new_frame_event.set()
                    cur_fid += 1
            else:
                time.sleep(0.005)

    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.kill()
                ffmpeg_proc.wait()
            except Exception:
                pass
        try:
            del frames
            shm.close()
        except Exception:
            pass


class VideoDecoderSubprocessManager:
    """Manages the lifecycle of a dedicated FFmpeg video decoding subprocess

    and its shared memory circular buffer.
    """

    @staticmethod
    def _compute_scaled_dimensions(
        orig_width: int,
        orig_height: int,
        target_width: Optional[int] = None,
        target_height: Optional[int] = 320,
    ) -> Tuple[int, int]:
        """Calculates scaled width and height preserving aspect ratio, ensuring even dimensions."""
        if orig_width <= 0 or orig_height <= 0:
            fallback_w = (orig_width // 2) * 2 if orig_width > 0 else 568
            fallback_h = (orig_height // 2) * 2 if orig_height > 0 else 320
            return max(2, fallback_w), max(2, fallback_h)

        if target_width is None and target_height is None:
            return max(2, (orig_width // 2) * 2), max(2, (orig_height // 2) * 2)

        scale_w = target_width / orig_width if target_width is not None else 1.0
        scale_h = target_height / orig_height if target_height is not None else 1.0

        if target_width is not None and target_height is not None:
            scale = min(scale_w, scale_h)
        elif target_height is not None:
            scale = scale_h
        else:
            scale = scale_w

        # If already smaller or equal, do not upscale
        if scale >= 1.0:
            return max(2, (orig_width // 2) * 2), max(2, (orig_height // 2) * 2)

        scaled_w = int(round(orig_width * scale))
        scaled_h = int(round(orig_height * scale))

        # Ensure dimensions are positive and even (multiples of 2) for FFmpeg compatibility
        scaled_w = max(2, (scaled_w // 2) * 2)
        scaled_h = max(2, (scaled_h // 2) * 2)
        return scaled_w, scaled_h

    def __init__(
        self,
        unique_id: str,
        video_path: str,
        toas: np.ndarray,
        fps: float,
        width: int,
        height: int,
        buf_len: int = 60,
        target_height: Optional[int] = 320,
        target_width: Optional[int] = None,
    ):
        self.unique_id = unique_id
        self.video_path = video_path
        self.toas = np.asarray(toas).squeeze()
        self.fps = float(fps)
        self.orig_width = int(width)
        self.orig_height = int(height)
        self.target_height = target_height
        self.target_width = target_width
        self.buf_len = int(buf_len)

        # Compute scaled resolution to conserve shared memory and decode overhead
        self.width, self.height = self._compute_scaled_dimensions(
            self.orig_width,
            self.orig_height,
            target_width=self.target_width,
            target_height=self.target_height,
        )

        # Unique shared memory name
        pid = os.getpid()
        safe_id = self.unique_id.replace('/', '_').replace('\\', '_')
        self.shm_name = f'psio_{safe_id}_{pid}_{int(time.time() * 1000) % 1000000}'

        # Allocate shared memory circular buffer
        self.buffer = SharedVideoCircularBuffer(
            name=self.shm_name,
            buf_len=self.buf_len,
            width=self.width,
            height=self.height,
            channels=3,
            create=True,
        )

        # Shared synchronization objects
        self.cmd_queue: mp.Queue = mp.Queue()
        self.slot_fids = mp.Array('q', [-1] * self.buf_len)
        self.slot_toas = mp.Array('d', [0.0] * self.buf_len)
        self.slot_valid = mp.Array('b', [0] * self.buf_len)
        self.write_head = mp.Value('i', 0)
        self.read_head = mp.Value('q', 0)
        self.new_frame_event = mp.Event()
        self.metadata_lock = mp.Lock()

        # Start subprocess
        self.process = mp.Process(
            target=_video_decoder_worker_entry,
            args=(
                self.video_path,
                self.toas if len(self.toas) > 0 else None,
                self.fps,
                self.shm_name,
                self.width,
                self.height,
                self.buf_len,
                self.cmd_queue,
                self.slot_fids,
                self.slot_toas,
                self.slot_valid,
                self.write_head,
                self.read_head,
                self.new_frame_event,
                self.metadata_lock,
            ),
            daemon=True,
        )
        self.process.start()
        # Pre-seed buffer from frame 0
        self.request_seek(0)

    def request_seek(self, frame_id: int):
        """Request the decoding subprocess to seek to target frame."""
        with self.metadata_lock:
            self.read_head.value = frame_id
        self.cmd_queue.put(('seek', frame_id))

    def get_frame(self, frame_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """Instant O(1) lookup of a frame from shared memory."""
        with self.metadata_lock:
            self.read_head.value = frame_id
        return self.buffer.get_frame(
            frame_id=frame_id,
            slot_fids=self.slot_fids,
            slot_toas=self.slot_toas,
            slot_valid=self.slot_valid,
            lock=self.metadata_lock,
        )

    def wait_for_frame(self, frame_id: int, timeout_s: float = 0.5) -> Optional[Tuple[np.ndarray, float]]:
        """Wait for frame to be decoded, or return None on timeout."""
        start_t = time.perf_counter()
        while (time.perf_counter() - start_t) < timeout_s:
            res = self.get_frame(frame_id)
            if res is not None:
                return res
            self.new_frame_event.wait(timeout=0.02)
            self.new_frame_event.clear()
        return self.get_frame(frame_id)

    def stop(self):
        """Terminate decoding subprocess and release shared memory."""
        try:
            self.cmd_queue.put(('stop', None))
        except Exception:
            pass

        if self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.5)

        self.buffer.close()


class Cache:
    """Legacy Cache interface retained for backwards compatibility."""

    def __init__(self, fetch_fn: Callable[[Any], Dict[Any, Any]], fetch_offset: int):
        self._cache: Dict[Any, Any] = {}
        self._fetch_fn = fetch_fn
        self._fetch_offset = fetch_offset
        self._request_queue: Queue[DataRequest] = Queue()
        self._data_events: Dict[Any, Any] = defaultdict(mp.Event)

    def start(self):
        pass

    def join(self):
        pass

    def get_data(self, key: Any) -> Any:
        if key in self._cache:
            return self._cache[key]
        self._cache = self._fetch_fn(max(0, key - self._fetch_offset))
        return self._cache.get(key, b'')
