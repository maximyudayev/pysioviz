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

from dash import Output, Input, State, dcc, html
import plotly.graph_objects as go

from pysioviz.components.data import DataComponent
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import GlobalVariableId


class MotorComponent(DataComponent):
    def __init__(
        self,
        hdf5_path: str,
        data_path: str,
        features: list[str],
        units: list[str],
        unique_id: str,
        legend_name: str,
        plot_window_seconds: float = 5.0,
    ):
        self._hdf5_path = hdf5_path
        self._data_path = data_path
        self._features = features
        self._units = units
        self._legend_name = legend_name
        self._plot_window_seconds = plot_window_seconds
        self._num_channels = len(features)

        self._data: dict[str, np.ndarray] = {feat: None for feat in features}

        # Create layout
        self._graph = dcc.Graph(
            id=f'{unique_id}-motor-plot',
            config={'displayModeBar': False},
            responsive=True,
            clear_on_unhover=True,
            style={
                'width': '100%',
                'height': '30vh',
            },
        )

        self._dropdown = dcc.Dropdown(
            id=f'{unique_id}-features-dropdown',
            options=[{'label': feature, 'value': i} for i, feature in enumerate(features)],
            value=0,
            clearable=False,
            style={'marginBottom': '10px'},
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

    def _read_timestamps(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            self._toa_s = hdf5[self._data_path]['timestamp'][:, 0]
            self._first_timestamp = float(self._toa_s[0])
            self._last_timestamp = float(self._toa_s[-1])

    def _read_data(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            for feature in self._features:
                self._data[feature] = hdf5[self._data_path][feature][:, 0]

    def get_sync_info(self):
        return {
            'type': 'motor',
            'unique_id': self._unique_id,
            'first_timestamp': self._first_timestamp,
            'last_timestamp': self._last_timestamp,
            'timestamps': self._toa_s,
        }

    def make_click_input(self):
        return Input(f'{self._unique_id}-motor-plot', 'clickData')

    def handle_click(self, ref_frame_timestamp, sync_timestamp):
        sample_id = self.get_frame_for_toa(sync_timestamp)
        toa_s = self._toa_s[sample_id].item() if sample_id < len(self._toa_s) else 0
        return f'{self._legend_name} - toa_s: {toa_s:.5f} (index: {sample_id})'

    def _create_figure(self, sync_timestamp: float, selected_feature: int):
        """Create the line plot figure for the given center index."""
        center_idx = self.get_frame_for_toa(sync_timestamp)
        start_idx = self.get_frame_for_toa(sync_timestamp - (self._plot_window_seconds / 2))
        end_idx = self.get_frame_for_toa(sync_timestamp + (self._plot_window_seconds / 2))

        # Get data slice
        feature_name = self._features[selected_feature]
        data_slice = self._data[feature_name][start_idx:end_idx+1]
        time_slice = self._toa_s[start_idx:end_idx+1] - self._toa_s[start_idx]

        # Calculate where the red line should be (current position in window)
        red_line_position = self._toa_s[center_idx] - self._toa_s[start_idx]

        # Create plot
        fig = go.Figure(
            data=go.Scatter(
                x=time_slice,
                y=data_slice,
                mode='lines',
                name='Motor',
                line=dict(width=1),
            ),
        )

        # Add vertical line at current position
        fig.add_vline(
            x=red_line_position,
            line_dash='dash',
            line_color='red',
        )

        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
        )
        fig.update_xaxes(
            title_text='Time (s)',
            range=[0, self._plot_window_seconds],
        )
        # fig.update_yaxes(
        #     title_text=self._units[selected_feature],
        # )

        return fig, center_idx

    def activate_callbacks(self):
        @app.callback(
            Output(f'{self._unique_id}-motor-plot', 'figure'),
            Output(f'{self._unique_id}-timestamp', 'children'),
            Input(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
            Input(f'{self._unique_id}-features-dropdown', 'value'),
            Input('offset-update-trigger', 'data'),
            State(f'{self._unique_id}-features-dropdown', 'value'),
            prevent_initial_call=False,
        )
        def update_plot(sync_timestamp, _, offset_trigger, selected_feature):
            try:
                if sync_timestamp is not None:
                    fig, center_idx = self._create_figure(sync_timestamp, selected_feature)
                else:
                    # Show initial data at start_idx if no sync
                    fig, center_idx = self._create_figure(self._first_timestamp, selected_feature)

                # Get timestamp for display
                toa_s = self._toa_s[center_idx]
                toa_text = f'toa_s: {toa_s:.5f} (index: {center_idx})'

                return fig, f'{toa_text} [offset: {self._offset_s*1000:+.0f}ms]'

            except Exception as e:
                print(f'Error updating plot: {e}')
                import traceback

                traceback.print_exc()
                return go.Figure(), 'Error'
