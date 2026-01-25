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

import numpy as np
import h5py

from dash import Output, Input, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pysioviz.components.data import DataComponent
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import GlobalVariableId


class LinePlotComponent(DataComponent):
    def __init__(
        self,
        hdf5_path: str,
        data_path: list[str],
        timestamp_path: list[str],
        unique_id: str,
        legend_name: str,
        sampling_rate: float,
        y_units: str,
        plot_window_seconds: float = 5.0,
        channel_names: list[str] | None = None,
        is_burst_data: bool = False,
        burst_samples_count_path: list[str] | None = None,
    ):
        self._y_units = y_units
        self._channel_names = channel_names
        self._hdf5_path = hdf5_path
        self._data_path = data_path
        self._timestamp_path = timestamp_path
        self._burst_samples_count_path = burst_samples_count_path
        self._legend_name = legend_name
        self._plot_window_seconds = plot_window_seconds
        self._sampling_rate = sampling_rate

        if len(timestamp_path) > 1 and len(timestamp_path) != len(data_path):
            raise ValueError(f'Timeline for multiple modalities must be shared, or must be provide per modality.')

        if is_burst_data and len(burst_samples_count_path) != len(timestamp_path):
            raise ValueError(f'Number of burst samples counters and timestamps must be the same - per modality.')

        if channel_names is not None and len(channel_names) != len(data_path):
            raise ValueError(f'Number of channel names for the legend must match the number of traces.')

        self._burst_timestamps: list[np.ndarray] = []
        self._sample_timestamps: list[np.ndarray] = []
        self._first_timestamp: list[int] = []
        self._last_timestamp: list[int] = []

        self._is_burst_data = is_burst_data
        self._num_samples: list[int] = []
        self._cumsum_samples: list[int] = []
        self._data: list[np.ndarray] = []

        self._y_min: list[float] = []
        self._y_max: list[float] = []
        self._y_range: list[tuple[float, float]] = []
        self._start_idx: list[int] = []
        self._end_idx: list[int] = []

        # Create layout
        self._graph = dcc.Graph(
            id=f'{unique_id}-lineplot',
            config={'displayModeBar': False},
            clear_on_unhover=True,
        )

        self._timestamp_display = html.Div(
            id=f'{unique_id}-timestamp',
            className='text-center small text-muted',
            style={'fontSize': '12px'},
        )

        self._layout = html.Div(
            [
                html.H6(self._legend_name, className='text-center mb-2'),
                self._graph,
                self._timestamp_display,
            ],
        )

        super().__init__(unique_id=unique_id)

    def read_data(self):
        self._read_timestamps()
        self._read_data()
        self._match_data_to_time()
        self._adjust_plot_ranges()

    def _read_timestamps(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            if self._is_burst_data and self._burst_samples_count_path is not None:
                self._read_burst_timestamps(hdf5)
            else:
                self._read_individual_timestamps(hdf5)

    def _read_burst_timestamps(self, hdf5: h5py.File):
        # Create cumulative sum for quick index lookup - ensure integer type
        for p in self._burst_samples_count_path:
            if p in hdf5:
                self._num_samples.append(hdf5[p][:])
                self._cumsum_samples.append(np.concatenate([[0], np.cumsum(self._num_samples)]).astype(np.int64))
            else:
                raise ValueError(f'Burst samples counter path {p} not found in HDF5.')

        # Read burst timestamps for data that arrived in bursts.
        for p in self._timestamp_path:
            if p in hdf5:
                timestamps = hdf5[p][:]
                self._burst_timestamps.append(timestamps)
                self._first_timestamp.append(float(timestamps[0]))
                self._last_timestamp.append(float(timestamps[-1]))
            else:
                raise ValueError(f'Timestamp path {p} not found in HDF5.')

        self._interpolate_bursted_timestamps()

    def _read_individual_timestamps(self, hdf5: h5py.File):
        for p in self._timestamp_path:
            if p in hdf5:
                timestamps = hdf5[p][:, 0]
                self._sample_timestamps.append(timestamps)
                self._first_timestamp.append(float(timestamps[0]))
                self._last_timestamp.append(float(timestamps[-1]))
            else:
                raise ValueError(f'Timestamp path {p} not found in HDF5')

    def _read_data(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            if self._is_burst_data:
                self._read_burst_samples(hdf5)
            else:
                self._read_individual_samples(hdf5)

    def _read_burst_samples(self, hdf5: h5py.File):
        # TODO: check logic.
        if self._data_path in hdf5:
            self._data = hdf5[self._data_path][:]

            # Handle both 1D and 2D data
            if len(self._data.shape) == 1:
                # Reshape 1D data to 2D with single channel
                self._data = self._data.reshape(-1, 1)
                self._num_channels = 1
            else:
                self._num_channels = self._data.shape[1]
        else:
            raise ValueError(f'Data path {self._data_path} not found in HDF5')

    def _read_individual_samples(self, hdf5: h5py.File):
        for p in self._data_path:
            if p in hdf5:
                data = hdf5[p][:]
                # Ensure 1D data is reshaped to column vector
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                self._data.append(data)
            else:
                raise ValueError(f'Data path {p} not found in HDF5.')
        self._num_channels = len(self._data)

    def _match_data_to_time(self):
        if self._is_burst_data:
            self._match_burst_samples_to_time()
        else:
            self._match_individual_samples_to_time()

    def _match_burst_samples_to_time(self):
        # TODO: check logic.
        expected_samples = int(self._cumsum_samples[-1])
        if len(self._data) != expected_samples:
            raise ValueError(
                f'Data length ({len(self._data)}) != total samples from bursts ({expected_samples})', flush=True
            )

    def _match_individual_samples_to_time(self):
        if len(self._sample_timestamps) == 1:
            min_length = min(*[len(d) for d in self._data], len(self._sample_timestamps))
            for d in self._data:
                d = d[:min_length]
            self._sample_timestamps[0] = self._sample_timestamps[0][:min_length]
        else:
            for d, t in zip(self._data, self._sample_timestamps):
                min_length = min(len(d), len(t))
                d = d[:min_length]
                t = t[:min_length]

    def _adjust_plot_ranges(self):
        # Calculate symmetric y-axis scaling using percentiles
        # Use percentiles to handle outliers, then create symmetric scale
        for d in self._data:
            percentile_low = np.percentile(d, 1)
            percentile_high = np.percentile(d, 99)

            # Find the maximum absolute value with 2x factor for more headroom
            max_abs_value = max(abs(2 * percentile_low), abs(2 * percentile_high))
            min_abs_value = min(abs(2 * percentile_low), abs(2 * percentile_high))

            # Create non-symmetric range around zero (no additional padding needed with 2x factor)
            self._y_min.append(-max_abs_value)
            self._y_max.append(max_abs_value)
            self._y_range.append((-max_abs_value, max_abs_value))

            # Initialize truncation points
            self._start_idx.append(0)
            self._end_idx.append(
                len(self._burst_timestamps) - 1 if self._is_burst_data else len(self._sample_timestamps) - 1
            )
            # Initialize sample indices for burst data
            # TODO: adjust for burst data.
            if self._is_burst_data:
                self._start_sample_idx = 0
                self._end_sample_idx = int(self._cumsum_samples[-1]) - 1

        # Channel names
        if self._channel_names is None:
            self._channels = [f'Channel {i + 1}' for i in range(self._num_channels)]
        else:
            self._channels = self._channel_names

    def _interpolate_bursted_timestamps(self):
        """Create interpolated timestamps for samples within bursts

        TODO: check logic to accomodate multiple paths of burst modalities.
        """
        total_samples = int(self._cumsum_samples[-1])
        self._sample_timestamps = np.zeros(total_samples)

        for i in range(len(self._burst_timestamps) - 1):
            start_idx = int(self._cumsum_samples[i])
            end_idx = int(self._cumsum_samples[i + 1])
            num_samples_in_burst = end_idx - start_idx

            # Linear interpolation between current and next burst timestamp
            start_time = float(self._burst_timestamps[i])
            end_time = float(self._burst_timestamps[i + 1])

            # Create evenly spaced timestamps within this burst
            # Ensure we get a 1D array by flattening
            timestamps = np.linspace(
                start_time,
                end_time - (end_time - start_time) / num_samples_in_burst,  # Don't include the exact end time
                num_samples_in_burst,
            ).flatten()
            self._sample_timestamps[start_idx:end_idx] = timestamps

        # Handle the last burst - use the same interval as the previous burst
        if len(self._burst_timestamps) > 1:
            last_burst_idx = len(self._burst_timestamps) - 1
            start_idx = int(self._cumsum_samples[last_burst_idx])
            end_idx = int(self._cumsum_samples[last_burst_idx + 1])
            num_samples_in_burst = end_idx - start_idx

            # Calculate average interval from previous bursts
            avg_interval = (float(self._burst_timestamps[-1]) - float(self._burst_timestamps[0])) / (
                len(self._burst_timestamps) - 1
            )

            # Create evenly spaced timestamps for last burst
            start_time = float(self._burst_timestamps[last_burst_idx])
            # Ensure we get a 1D array by flattening
            timestamps = np.linspace(
                start_time,
                start_time + avg_interval * (num_samples_in_burst - 1) / num_samples_in_burst,
                num_samples_in_burst,
            ).flatten()
            self._sample_timestamps[start_idx:end_idx] = timestamps

    def get_sync_info(self):
        # TODO: check logic to accomodate multiple modalities.
        if self._is_burst_data:
            return {
                'type': 'burst_data',
                'unique_id': self._unique_id,
                'first_timestamp': self._first_timestamp,
                'last_timestamp': self._last_timestamp,
                'timestamps': self._burst_timestamps,
                'num_samples': self._num_samples,
                'cumsum_samples': self._cumsum_samples,
            }
        else:
            return {
                'type': 'continuous_data',
                'unique_id': self._unique_id,
                'first_timestamp': self._first_timestamp,
                'last_timestamp': self._last_timestamp,
                'timestamps': self._sample_timestamps,
            }

    def set_truncation_points(self, start_idx: int, end_idx: int):
        if self._is_burst_data:
            # EMG
            # For burst data, these are burst indices
            self._start_idx = int(max(0, start_idx))
            self._end_idx = int(min(len(self._burst_timestamps) - 1, end_idx))
            # Convert to sample indices
            self._start_sample_idx = int(self._cumsum_samples[self._start_idx])
            self._end_sample_idx = int(self._cumsum_samples[self._end_idx + 1]) - 1
            print(
                f'{self._legend_name}: Start burst = {self._start_idx}, Start sample = {self._start_sample_idx}',
                flush=True,
            )
        else:
            # Insole
            self._start_idx = int(max(0, start_idx))
            self._end_idx = int(min(len(self._sample_timestamps) - 1, end_idx))
            print(f'{self._legend_name}: Start index = {self._start_idx}', flush=True)

    def get_frame_for_toa(self, sync_timestamp: float) -> int:
        # Overrides the abstract method.
        if self._is_burst_data:
            # EMG
            # Find closest burst
            time_diffs = np.abs(self._burst_timestamps - sync_timestamp)
            closest_burst_idx = np.argmin(time_diffs)
            # Convert to sample index (use middle of burst for centering)
            start_sample = int(self._cumsum_samples[closest_burst_idx])
            end_sample = int(self._cumsum_samples[closest_burst_idx + 1])
            center_idx = int((start_sample + end_sample) // 2)
            # Apply offset
            offset_idx = center_idx + self._align_info.start_id
            # Ensure within bounds
            offset_idx = max(0, min(len(self._data) - 1, offset_idx))
            return offset_idx
        else:
            # Insole
            if self._sample_timestamps is not None:
                time_diffs = np.abs(self._sample_timestamps - sync_timestamp)
                closest_idx = np.argmin(time_diffs)
                # Apply offset
                offset_idx = closest_idx + self._align_info.start_id
                # Ensure within bounds
                offset_idx = max(0, min(len(self._sample_timestamps) - 1, offset_idx))
                return int(offset_idx)
            else:
                return 0

    def _create_figure(self, center_idx: int):
        """Create the line plot figure for the given center index"""
        # Ensure center_idx is within bounds
        center_idx = max(0, min(center_idx, len(self._data) - 1))

        # Calculate window bounds
        window_samples = int(self._plot_window_seconds * self._sampling_rate)
        half_window = window_samples // 2

        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(self._data) - 1, center_idx + half_window)

        # Get data slice
        data_slice = self._data[start_idx : end_idx + 1]
        time_slice = np.arange(len(data_slice)) / self._sampling_rate

        # Calculate where the red line should be (current position in window)
        red_line_position = (center_idx - start_idx) / self._sampling_rate

        # Create subplots for each channel
        fig = make_subplots(
            rows=self._num_channels,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=self._channels,
        )

        # Add traces for each channel
        for i in range(self._num_channels):
            if self._num_channels > 1:
                channel_data = data_slice[:, i]
            else:
                channel_data = data_slice.flatten()

            fig.add_trace(
                go.Scatter(
                    x=time_slice,
                    y=channel_data,
                    mode='lines',
                    name=self._channels[i],
                    line=dict(width=1),
                ),
                row=i + 1,
                col=1,
            )

            # Add vertical line at current position
            fig.add_vline(
                x=red_line_position,
                line_dash='dash',
                line_color='red',
                row=i + 1,
                col=1,
            )

        # Update layout
        title = self._legend_name
        if self._align_info.start_id != 0:
            title += f' [offset: {self._align_info.start_id:+d}]'

        fig.update_layout(
            title_text=title,
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=20),
            height=400,
            width=400,
        )

        fig.update_xaxes(title_text='Time (s)', row=self._num_channels, col=1)

        for i in range(self._num_channels):
            fig.update_yaxes(
                title_text=self._y_units,
                range=self._y_range,  # Apply same range to all channels
                row=i + 1,
                col=1,
            )

        return fig

    def activate_callbacks(self):
        @app.callback(
            Output(f'{self._unique_id}-lineplot', 'figure'),
            Output(f'{self._unique_id}-timestamp', 'children'),
            Input(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
            Input('offset-update-trigger', 'data'),
            prevent_initial_call=False,
        )
        def update_plot(sync_timestamp, offset_trigger):
            try:
                if sync_timestamp is not None:
                    # Find the index matching the sync timestamp
                    current_idx = self.get_frame_for_toa(sync_timestamp)

                    # Create the figure
                    fig = self._create_figure(current_idx)

                    # Get timestamp for display
                    timestamp = (
                        self._sample_timestamps[current_idx] if current_idx < len(self._sample_timestamps) else 0
                    )
                    # Convert numpy scalar to Python float for formatting
                    timestamp_float = float(timestamp)
                    timestamp_text = f'toa_s: {timestamp_float:.7f} (index: {current_idx})'

                    return fig, timestamp_text
                else:
                    # Show initial data at start_idx if no sync
                    if self._is_burst_data:
                        initial_idx = int(self._start_sample_idx)
                    else:
                        initial_idx = int(self._start_idx)

                    fig = self._create_figure(initial_idx)
                    timestamp = (
                        self._sample_timestamps[initial_idx] if initial_idx < len(self._sample_timestamps) else 0
                    )
                    timestamp_float = float(timestamp)
                    timestamp_text = f'toa_s: {timestamp_float:.7f} (index: {initial_idx}) {self._offset_s*1000:.0f}'
                    return fig, timestamp_text

            except Exception as e:
                print(f'Error updating plot: {e}')
                import traceback

                traceback.print_exc()
                return go.Figure(), 'Error'
