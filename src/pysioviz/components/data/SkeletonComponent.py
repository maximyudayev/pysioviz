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

from pysioviz.components.data import DataComponent
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import GlobalVariableId


class SkeletonComponent(DataComponent):
    def __init__(
        self,
        hdf5_path: str,
        position_path: str,
        pos_counter_path: str,
        timestamp_path: str,
        ref_counter_path: str,
        unique_id: str,
        legend_name: str,
        div_height: str = '40vh',
    ):
        self._hdf5_path = hdf5_path
        self._position_path = position_path
        self._timestamp_path = timestamp_path
        self._ref_counter_path = ref_counter_path
        self._pos_counter_path = pos_counter_path
        self._legend_name = legend_name

        # Segment names and indices (BASED ON THE HDF5 FILE DESCRIPTION)
        self._segment_names = [
            'Pelvis',
            'L5',
            'L3',
            'T12',
            'T8',
            'Neck',
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
            'Right Toe',
            'Left Upper Leg',
            'Left Lower Leg',
            'Left Foot',
            'Left Toe',
        ]

        # Define skeletal connections
        self._connections = [
            # Spine
            ('Pelvis', 'L5'),
            ('L5', 'L3'),
            ('L3', 'T12'),
            ('T12', 'T8'),
            ('T8', 'Neck'),
            ('Neck', 'Head'),
            # Right arm
            ('Neck', 'Right Shoulder'),
            ('Right Shoulder', 'Right Upper Arm'),
            ('Right Upper Arm', 'Right Forearm'),
            ('Right Forearm', 'Right Hand'),
            # Left arm
            ('Neck', 'Left Shoulder'),
            ('Left Shoulder', 'Left Upper Arm'),
            ('Left Upper Arm', 'Left Forearm'),
            ('Left Forearm', 'Left Hand'),
            # Right leg
            ('Pelvis', 'Right Upper Leg'),
            ('Right Upper Leg', 'Right Lower Leg'),
            ('Right Lower Leg', 'Right Foot'),
            ('Right Foot', 'Right Toe'),
            # Left leg
            ('Pelvis', 'Left Upper Leg'),
            ('Left Upper Leg', 'Left Lower Leg'),
            ('Left Lower Leg', 'Left Foot'),
            ('Left Foot', 'Left Toe'),
        ]

        # Create layout
        self._graph = dcc.Graph(
            id=f'{unique_id}-skeleton',
            responsive=True,
            clear_on_unhover=True,
            style={
                'width': '100%',
                'height': div_height,
            },
        )

        self._timestamp_display = html.Div(
            id=f'{unique_id}-timestamp',
            className='text-center small text-muted',
            style={'fontSize': '11px', 'height': '20px', 'lineHeight': '20px'},
        )

        self._layout = html.Div(
            [
                html.H6(self._legend_name, className='text-center mb-2'),
                self._graph,
                self._timestamp_display,
            ],
            style={
                'width': '100%',
            },
        )

        super().__init__(unique_id=unique_id)

    def read_data(self):
        self._read_timestamps()
        self._read_data()
        self._match_data_to_time()

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
            if self._position_path in hdf5:
                self._positions = hdf5[self._position_path][:]
                if len(self._positions.shape) != 3 or self._positions.shape[2] != 3:
                    raise ValueError(f'Expected position data shape (frames, segments, 3), got {self._positions.shape}')
            else:
                raise ValueError(f'Position path {self._position_path} not found in HDF5')

    def _match_data_to_time(self):
        with h5py.File(self._hdf5_path, 'r') as hdf5:
            ref_counters = hdf5[self._ref_counter_path][:, 0]
            pos_counters = hdf5[self._pos_counter_path][:, 0]

            # Create a mapping from values to their first occurrence index in position counters
            _, first_indices = np.unique(pos_counters, return_index=True)
            value_to_first_idx = dict(zip(pos_counters[first_indices], first_indices))

            # Look up each element of reference counters
            matches = np.array([value_to_first_idx.get(val, -1) for val in ref_counters])
            self._toa_s = self._toa_s[matches >= 0]
            self._positions = self._positions[matches[matches >= 0]]

            self._start_idx = 0
            self._end_idx = len(self._toa_s) - 1
            print(f'Position data length ({len(self._positions)}) ?= timestamp length ({len(self._toa_s)})', flush=True)

    def get_sync_info(self):
        return {
            'type': 'skeleton',
            'unique_id': self._unique_id,
            'first_timestamp': self._first_timestamp,
            'last_timestamp': self._last_timestamp,
            'timestamps': self._toa_s,
        }

    def set_truncation_points(self, start_idx: int, end_idx: int):
        self._start_idx = int(max(0, start_idx))
        self._end_idx = int(min(len(self._toa_s) - 1, end_idx))
        print(f'{self._legend_name}: Start index = {self._start_idx}', flush=True)

    def make_click_input(self):
        return Input('skeleton_mvn-skeleton', 'clickData')

    def handle_click(self, ref_frame_timestamp, sync_timestamp):
        sample_id = self.get_frame_for_toa(sync_timestamp)
        toa_s = self._toa_s[sample_id].item() if sample_id < len(self._toa_s) else 0
        return f'Skeleton MVN - toa_s: {toa_s:.5f} (index: {sample_id})'

    def _create_figure(self, frame_idx: int) -> go.Figure:
        # Ensure frame_idx is within bounds
        frame_idx = max(0, min(frame_idx, len(self._positions) - 1))

        # Get positions for this frame
        positions = self._positions[frame_idx]  # Shape: (num_segments, 3)

        # Create 3D scatter plot
        fig = go.Figure()

        # Add joints
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(size=6, color='blue'),
                text=self._segment_names,
                hovertemplate='%{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>',
                name='Joints',
            )
        )

        # Add bones
        for conn in self._connections:
            start_idx = self._segment_names.index(conn[0])
            end_idx = self._segment_names.index(conn[1])

            fig.add_trace(
                go.Scatter3d(
                    x=[positions[start_idx, 0], positions[end_idx, 0]],
                    y=[positions[start_idx, 1], positions[end_idx, 1]],
                    z=[positions[start_idx, 2], positions[end_idx, 2]],
                    mode='lines',
                    line=dict(color='gray', width=4),
                    showlegend=False,
                    hoverinfo='skip',
                )
            )

        return fig

    def activate_callbacks(self):
        @app.callback(
            Output(f'{self._unique_id}-skeleton', 'figure'),
            Output(f'{self._unique_id}-timestamp', 'children'),
            Input(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
            Input('offset-update-trigger', 'data'),
        )
        def update_skeleton(sync_timestamp, offset_trigger):
            try:
                if sync_timestamp is not None:
                    # Find the index matching the sync timestamp
                    current_idx = self.get_frame_for_toa(sync_timestamp)

                    # Create the figure
                    fig = self._create_figure(current_idx)

                    # Get timestamp for display
                    timestamp = self._toa_s[current_idx] if current_idx < len(self._toa_s) else 0
                    timestamp_float = float(timestamp)
                    timestamp_text = (
                        f'toa_s: {timestamp_float:.5f} (index: {current_idx}) [offset: {self._offset_s*1000:+.0f}ms]'
                    )

                else:
                    # Show initial data at start_idx if no sync
                    fig = self._create_figure(self._start_idx)
                    timestamp = self._toa_s[self._start_idx] if self._start_idx < len(self._toa_s) else 0
                    timestamp_float = float(timestamp)
                    timestamp_text = f'toa_s: {timestamp_float:.5f} (index: {self._start_idx}) [offset: {self._offset_s*1000:+.0f}ms]'

                # Update layout
                fig.update_layout(
                    scene=dict(
                        xaxis_title='X (m)',
                        yaxis_title='Y (m)',
                        zaxis_title='Z (m)',
                        aspectmode='data',
                        camera=dict(eye=dict(x=2.0, y=2.0, z=2.0)),
                    ),
                    showlegend=False,
                    autosize=True,
                    margin=dict(l=0, r=0, t=0, b=0),
                )

                return fig, timestamp_text

            except Exception as e:
                print(f'Error updating skeleton: {e}', flush=True)
                import traceback

                traceback.print_exc()
                return go.Figure(), 'Error'
