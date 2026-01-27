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


class ExoImuComponent(DataComponent):
    def __init__(
        self,
        hdf5_path: str,
        data_path: str,
        unique_id: str,
        legend_name: str,
        plot_window_seconds: float = 5.0,
    ):
        self._hdf5_path = hdf5_path
        self._data_path = data_path
        self._legend_name = legend_name
        self._plot_window_seconds = plot_window_seconds
        self._features = ['euler', 'gyroscope']
        self._data: dict[str, np.ndarray] = {feat: None for feat in self._features}

        # Create layout
        self._graph = dcc.Graph(
            id=f'{unique_id}-exo_imu-plot',
            config={'displayModeBar': False},
            responsive=True,
            clear_on_unhover=True,
            style={
                'width': '100%',
                'height': '30vh',
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
                self._graph,
                self._timestamp_display,
            ],
        )

        super().__init__(unique_id=unique_id)

    def read_data(self):
        self._read_timestamps()
        self._read_data()

    def _read_timestamps(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            self._toa_s = hdf5[self._data_path]['toa_s'][:, 0]
            self._first_timestamp = float(self._toa_s[0])
            self._last_timestamp = float(self._toa_s[-1])

    def _read_data(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            for feature in self._features:
                self._data[feature] = hdf5[self._data_path][feature][:]

    def get_sync_info(self):
        return {
            'type': 'exo_imu',
            'unique_id': self._unique_id,
            'first_timestamp': self._first_timestamp,
            'last_timestamp': self._last_timestamp,
            'timestamps': self._toa_s,
        }

    def make_click_input(self):
        return Input(f'{self._unique_id}-exo_imu-plot', 'clickData')

    def handle_click(self, ref_frame_timestamp, sync_timestamp):
        sample_id = self.get_frame_for_toa(sync_timestamp)
        toa_s = self._toa_s[sample_id].item() if sample_id < len(self._toa_s) else 0
        return f'{self._legend_name} - toa_s: {toa_s:.5f} (index: {sample_id})'

    def _create_figure(self, sync_timestamp: float):
        """Create the line plot figure for the given center index."""
        center_idx = self.get_frame_for_toa(sync_timestamp)
        start_idx = self.get_frame_for_toa(sync_timestamp - (self._plot_window_seconds / 2))
        end_idx = self.get_frame_for_toa(sync_timestamp + (self._plot_window_seconds / 2))

        # Get data slice
        time_slice = self._toa_s[start_idx:end_idx+1] - self._toa_s[start_idx]

        # Calculate where the red line should be (current position in window)
        red_line_position = self._toa_s[center_idx] - self._toa_s[start_idx]

        # Create subplots for `euler` and `gyroscope`
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            # subplot_titles=['Euler', 'Gyro'],
        )

        # Create plot
        for i, feature_name in enumerate(self._features):
            data_slice = self._data[feature_name][start_idx:end_idx+1]
            for j in range(3):
                fig.add_trace(
                    go.Scatter(
                        x=time_slice,
                        y=data_slice[:, j],
                        mode='lines',
                        name=f'{['Euler', 'Gyro'][i]} ({['X', 'Y', 'Z'][j]})',
                        line=dict(width=1, color=['blue', 'green', 'red'][j]),
                    ),
                    row=i+1,
                    col=1,
                )

                # fig.update_yaxes(
                #     title_text=['°', 'm/s'][i],
                #     row=i+1,
                #     col=1,
                # )

        # Add vertical line at current position
        fig.add_vline(
            x=red_line_position,
            line_dash='dash',
            line_color='red',
        )

        fig.update_layout(
            # showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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
            Output(f'{self._unique_id}-exo_imu-plot', 'figure'),
            Output(f'{self._unique_id}-timestamp', 'children'),
            Input(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
            Input('offset-update-trigger', 'data'),
            prevent_initial_call=False,
        )
        def update_plot(sync_timestamp, offset_trigger):
            try:
                if sync_timestamp is not None:
                    fig, center_idx = self._create_figure(sync_timestamp)
                else:
                    # Show initial data at start_idx if no sync
                    fig, center_idx = self._create_figure(self._first_timestamp)

                # Get timestamp for display
                toa_s = self._toa_s[center_idx]
                toa_text = f'toa_s: {toa_s:.5f} (index: {center_idx})'

                return fig, f'{toa_text} [offset: {self._offset_s*1000:+.0f}ms]'

            except Exception as e:
                print(f'Error updating plot: {e}')
                import traceback

                traceback.print_exc()
                return go.Figure(), 'Error'
