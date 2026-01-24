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

from typing import Dict, Tuple
import numpy as np
import ffmpeg
import base64
import h5py

from dash import Output, Input, dcc, html, Patch
import dash_bootstrap_components as dbc
import plotly.express as px

from pysioviz.components.data import DataComponent
from pysioviz.utils.cache import Cache
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import GlobalVariableId, HwAccelEnum, VideoComponentInfo

# JPEG End of Image marker
EOI = b'\xff\xd9'


class VideoComponent(DataComponent):
    def __init__(
        self,
        video_filepath: str,
        hdf5_filepath: str,
        unique_id: str,
        toa_hdf5_path: str,
        timestamp_hdf5_path: str,
        sequence_hdf5_path: str,
        legend_name: str,
        hwaccel: str = HwAccelEnum.D3D12VA.value,
        prefetch_window_s: float = 10.0,
        div_height: str = '30vh',
        is_highlight: bool = False,
    ):
        self._legend_name = legend_name
        self._video_path = video_filepath
        self._hdf5_path = hdf5_filepath
        self._toa_hdf5_path = toa_hdf5_path
        self._timestamp_hdf5_path = timestamp_hdf5_path
        self._sequence_hdf5_path = sequence_hdf5_path

        # Get video properties
        self._width, self._height, self._fps, self._total_frames = self._get_video_properties()
        self._empty_frame = np.zeros([self._height, self._width, 3], np.uint8)

        self._current_frame_id = 0

        self._prefetch_window_s = prefetch_window_s
        self._num_prefetch_frames = round(self._fps * self._prefetch_window_s)

        self._create_ffmpeg_cacher(hwaccel=hwaccel)
        self._create_layout(unique_id=unique_id, is_highlight=is_highlight, div_height=div_height)
        super().__init__(unique_id=unique_id)

    @property
    def current_frame_id(self) -> int:
        return self._current_frame_id

    def _create_ffmpeg_cacher(self, hwaccel: str):
        # Create FFmpeg decode cache, will prefetch a window, centered 1/3 of requested cache miss frame
        def _decode(frame_id: int) -> Dict[int, bytes]:
            # Seek to the timestamp because it is much faster than using frame index
            timestamp_start = frame_id / self._fps
            # Get multiple frames for caching, to mask decoding latency
            buf, _ = (
                ffmpeg.input(
                    filename=self._video_path,
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
            # Split continuous images buffer of jpeg-encoded frames by known end-of-image delimeter
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

    def _create_layout(self, unique_id, is_highlight, div_height):
        # Create image placeholder
        self._fig = px.imshow(
            img=self._empty_frame,
            binary_string=True,
        )
        self._fig.update_layout(
            title_text=self._legend_name,
            title_font_size=11,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=20, b=0),
            autosize=True,
        )
        self._fig.update_xaxes(showticklabels=False, showgrid=False)
        self._fig.update_yaxes(showticklabels=False, showgrid=False)

        # Create layout with timestamp display
        self._image = dcc.Graph(
            figure=self._fig,
            id='%s-video' % (unique_id),
            config={
                'displayModeBar': False,
            },
            responsive=True,
            clear_on_unhover=True,
            style={
                "width": "100%",
                "height": div_height,
                'cursor': 'pointer' if not is_highlight else 'default',
            },
        )
        self._timestamp_display = html.Div(
            id='%s-timestamp' % (unique_id),
            className='text-center small text-muted',
            style={
                'fontSize': '11px',
                'height': '20px',
                'lineHeight': '20px'
            },
        )

        self._layout = html.Div(
            [
                dcc.Loading(
                    [self._image],
                    id='%s-loader' % (unique_id),
                    type='default',
                    target_components={'%s-video' % (unique_id): 'figure'},
                ),
                self._timestamp_display,
            ],
            style={
                'width': '100%',
            }
        )

    def read_data(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            try:
                self._toa_s = hdf5[self._toa_hdf5_path][:]
                self._timestamp = hdf5[self._timestamp_hdf5_path][:]
                self._sequence = hdf5[self._sequence_hdf5_path][:]
            except Exception as e:
                print(f'Error reading timestamps for cameras: {e}', flush=True)

    def _get_frame(self, frame_id: int) -> bytes:
        """Get video frame at a specific index."""
        # Ensure we don't go beyond bounds
        if frame_id < 0:
            frame_id = 0
        elif frame_id >= self._total_frames:
            frame_id = self._total_frames - 1

        # Get the frame from the cache manager
        try:
            return self._cache.get_data(frame_id)
        except Exception as e:
            print(f'Error getting frame {frame_id}: {e}')
            return self._empty_frame.tobytes()

    def _get_video_properties(self) -> Tuple[int, int, float, int]:
        """Get video width, height, fps, and total frames using `ffprobe`."""
        probe = ffmpeg.probe(self._video_path)
        video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps_num, fps_denum = map(lambda x: float(x), video_stream['r_frame_rate'].split('/'))
        fps = fps_num / fps_denum
        num_frames = round(float(probe['format']['duration']) * fps)

        return width, height, fps, num_frames

    def get_frame_for_timestamp(self, timestamp: float) -> int:
        """Find the frame index closest to, but not later than the given timestamp."""
        timestamp_diffs = np.abs(self._timestamp - timestamp)
        return np.argmin(timestamp_diffs).item()

    def get_frame_for_toa(self, toa_s: float) -> int:
        """Find the frame index closest to, but no later than the given time-of-arrival."""
        toas_diffs = np.abs(self._toa_s - toa_s)
        return np.argmin(toas_diffs).item()

    def get_timestamp_at_frame(self, frame_id: int) -> float:
        """Get the timestamp for a given frame."""
        return self._timestamp[frame_id].item()

    def get_toa_at_frame(self, frame_id: int) -> float:
        """Get the time-of-arrival for a given frame."""
        return self._toa_s[frame_id].item()

    def get_sequence_at_frame(self, frame_id: int) -> int:
        """Get the aligned sequence id for a given frame."""
        return (self._sequence[frame_id] - self._sequence[self._align_info.start_id]).item()

    def get_sync_info(self):
        return VideoComponentInfo(
            type='camera',
            unique_id=self._unique_id,
            toa_s=self._toa_s,
            frame_timestamp=self._timestamp,
            sequence=self._sequence,
        )

    def _generate_patch_from_frame(self, frame_id: int):
        """Extracts specified frame of the video.
        
        Args:
            frame_id (int): Exact frame of the video to extract.
        """
        try:
            img = self._get_frame(frame_id)
            fig = Patch()
            fig['data'][0]['source'] = 'data:image/jpeg;base64,%s' % base64.b64encode(img).decode('utf-8')

            # Get timestamp for display.
            toa = self.get_toa_at_frame(frame_id)
            toa_text = f'toa_s: {toa:.5f} (frame: {frame_id})'

            return fig, toa_text
        except Exception as e:
            print(f'Error loading frame for {self._unique_id}: {e}')
            return Patch(), 'Error'

    def activate_callbacks(self):
        @app.callback(
            Output('%s-video' % (self._unique_id), 'figure'),
            Output('%s-timestamp' % (self._unique_id), 'children'),
            Input(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
        )
        def update_camera(sync_timestamp):
            # Regular cameras match frame as a slave by globally synced `sync_timestamp`.
            self._current_frame_id = self.get_frame_for_toa(sync_timestamp)
            fig, toa_text = self._generate_patch_from_frame(self._current_frame_id)
            return fig, f'{toa_text} [offset: {self._align_info.start_id:+d}]'
