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


class ImuComponent(DataComponent):
    def __init__(
        self,
        hdf5_path: str,
        data_path: str,
        data_counter_path: str,
        timestamp_path: str,
        ref_counter_path: str,
        unique_id: str,
        legend_name: str,
        sensor_type: str,  # 'accelerometer', 'gyroscope', or 'magnetometer'
        plot_window_seconds: float = 1.0,
    ):
        self._hdf5_path = hdf5_path
        self._data_path = data_path
        self._timestamp_path = timestamp_path
        self._ref_counter_path = ref_counter_path
        self._data_counter_path = data_counter_path
        self._legend_name = legend_name
        self._sensor_type = sensor_type
        self._plot_window_seconds = plot_window_seconds

        # Joint names corresponding to the 17 sensors (taken from the description from the HDF5 file)
        self._joint_names = [
            'Pelvis',
            'T8',
            'Head',
            'Right Shoulder',
            'Right Upper Arm',
            'Right Forearm',
            'Right Hand',
            'Left Shoulder',
            'Left Upper Arm',
            'Left Forearm',
            'Left Hand',
            'Right Upper Leg',
            'Right Lower Leg',
            'Right Foot',
            'Left Upper Leg',
            'Left Lower Leg',
            'Left Foot',
        ]

        # Set units based on sensor type
        if sensor_type == 'accelerometer':
            self._y_units = 'm/s²'
        elif sensor_type == 'gyroscope':
            self._y_units = 'm/s'
        elif sensor_type == 'magnetometer':
            self._y_units = 'μT'
        else:
            self._y_units = 'units'

        # Create layout
        self._dropdown = dcc.Dropdown(
            id=f'{unique_id}-joint-dropdown',
            options=[{'label': joint, 'value': i} for i, joint in enumerate(self._joint_names)],
            value=0,
            clearable=False,
            style={'marginBottom': '10px'},
        )

        self._graph = dcc.Graph(
            id=f'{unique_id}-imu-plot',
            config={'displayModeBar': True},
            responsive=True,
            clear_on_unhover=True,
            style={
                'width': '100%',
                'height': '40vh',
            },
        )

        self._timestamp_display = html.Div(
            id=f'{unique_id}-timestamp',
            className='text-center small text-muted',
            style={'fontSize': '12px'},
        )

        self._layout = html.Div(
            [
                html.H6(self._legend_name, className='text-center mb-2'),
                self._dropdown,
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
            if self._timestamp_path in hdf5:
                self._toa_s = hdf5[self._timestamp_path][:, 0]
                self._first_timestamp = float(self._toa_s[0])
                self._last_timestamp = float(self._toa_s[-1])
            else:
                raise ValueError(f'Timestamp path {self._timestamp_path} not found in HDF5')

    def _read_data(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            if self._data_path in hdf5:
                self._data = hdf5[self._data_path][:]
                # Expected shape: (num_timestamps, 17 joints, 3 axes) - TODO: NUM_JOINTS FLEXIBLE FOR REVALEXO
                if len(self._data.shape) != 3 or self._data.shape[1] != 17 or self._data.shape[2] != 3:
                    raise ValueError(f'Expected IMU data shape (timestamps, 17, 3), got {self._data.shape}')
            else:
                raise ValueError(f'Data path {self._data_path} not found in HDF5')

    def _match_data_to_time(self):
        """Match data and timestamp by `counter` sequence id."""
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            ref_counters = hdf5[self._ref_counter_path][:, 0]
            data_counters = hdf5[self._data_counter_path][:, 0]

            # Create a mapping from values to their first occurrence index in data counters
            _, first_indices = np.unique(data_counters, return_index=True)
            value_to_first_idx = dict(zip(data_counters[first_indices], first_indices))

            # Look up each element of reference counters
            matches = np.array([value_to_first_idx.get(val, -1) for val in ref_counters])
            self._toa_s = self._toa_s[matches >= 0]
            self._first_timestamp = float(self._toa_s[0])
            self._last_timestamp = float(self._toa_s[-1])
            self._data = self._data[matches[matches >= 0]]
            print(
                f'{self._sensor_type} data length ({len(self._data)}) ?= timestamp length ({len(self._toa_s)})',
                flush=True,
            )

    def _adjust_plot_ranges(self):
        # Calculate symmetric y-axis scaling using percentiles
        # Use percentiles to handle outliers, then create symmetric scale
        # This could truncate extreme values, but double clicking on the graph will bring them back to view
        percentile_low = np.percentile(self._data, 1)
        percentile_high = np.percentile(self._data, 99)

        # Find the maximum absolute value with 2x factor for more headroom
        max_abs_value = max(abs(2 * percentile_low), abs(2 * percentile_high))

        # Create symmetric range around zero
        self._y_min = -max_abs_value
        self._y_max = max_abs_value
        self._y_range = [-max_abs_value, max_abs_value]

        # Initialize truncation points
        self._start_idx = 0
        self._end_idx = len(self._toa_s) - 1

        # Current selected joint (default to first - pelvis)
        self._selected_joint_idx = 0

    def get_sync_info(self):
        return {
            'type': 'imu',
            'unique_id': self._unique_id,
            'first_timestamp': self._first_timestamp,
            'last_timestamp': self._last_timestamp,
            'timestamps': self._toa_s,
        }

    def make_click_input(self):
        return Input(f'{self._unique_id}-imu-plot', 'clickData')

    def handle_click(self, ref_frame_timestamp, sync_timestamp):
        sample_id = self.get_frame_for_toa(sync_timestamp)
        toa_s = self._toa_s[sample_id].item() if sample_id < len(self._toa_s) else 0
        return f'IMU {self._sensor_type} - toa_s: {toa_s:.5f} (index: {sample_id})'

    def _create_figure(self, sync_timestamp: float, joint_idx: int) -> go.Figure:
        center_idx = self.get_frame_for_toa(sync_timestamp)
        start_idx = self.get_frame_for_toa(sync_timestamp - (self._plot_window_seconds / 2))
        end_idx = self.get_frame_for_toa(sync_timestamp + (self._plot_window_seconds / 2))

        # Get data slice
        data_slice = self._data[start_idx:end_idx+1, joint_idx, :]
        time_slice = self._toa_s[start_idx:end_idx+1] - self._toa_s[start_idx]

        # Calculate where the red line should be (current position in window)
        red_line_position = self._toa_s[center_idx] - self._toa_s[start_idx]

        # Create subplots for X, Y, Z
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=['X', 'Y', 'Z'],
        )

        # Add traces for each axis
        colors = ['blue', 'green', 'red']
        axes = ['X', 'Y', 'Z']

        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=time_slice,
                    y=data_slice[:, i],
                    mode='lines',
                    name=axes[i],
                    line=dict(width=1, color=colors[i]),
                    # yaxis=dict(title_text=self._y_units)
                ),
                row=i+1,
                col=1,
            )

            # Add vertical line at current position
            fig.add_vline(
                x=red_line_position,
                line_dash='dash',
                line_color='red',
                row=i+1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
        )

        fig.update_xaxes(
            title_text='Time (s)',
            range=[0, self._plot_window_seconds],
            row=3,
            col=1,
        )

        return fig, center_idx

    def activate_callbacks(self):
        @app.callback(
            Output(f'{self._unique_id}-imu-plot', 'figure'),
            Output(f'{self._unique_id}-timestamp', 'children'),
            Input(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
            Input(f'{self._unique_id}-joint-dropdown', 'value'),
            Input('offset-update-trigger', 'data'),
            prevent_initial_call=False,
        )
        def update_plot(sync_timestamp, selected_joint, offset_trigger):
            try:
                # Update selected joint
                joint_idx = selected_joint if selected_joint is not None else 0

                if sync_timestamp is not None:
                    fig, center_idx = self._create_figure(sync_timestamp, joint_idx)
                else:
                    # Show initial data at start_idx if no sync
                    fig, center_idx = self._create_figure(self._first_timestamp, joint_idx)

                # Get timestamp for display
                toa_s = self._toa_s[center_idx]
                toa_text = f'toa_s: {toa_s:.5f} (index: {center_idx})'

                return fig, f'{toa_text} [offset: {self._offset_s*1000:+.0f}ms]'

            except Exception as e:
                print(f'Error updating IMU plot: {e}', flush=True)
                import traceback

                traceback.print_exc()
                return go.Figure(), 'Error'
