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

import json

from dash import html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc

from pysioviz.components.control import ControlComponent
from pysioviz.components.data import DataComponent
from pysioviz.utils.gui_utils import app


class OffsetComponent(ControlComponent):
    """Offset alignment control component.

    - Individual offset controls for each component.
    - Shared offset for skeleton/IMU components.
    - Increment/decrement buttons (+/-1, +/-10).
    - Reset buttons (individual and all).
    - Apply offsets to components.
    """

    def __init__(self, offset_components: list[DataComponent]):
        """
        Synchronization offset control component.

        Args:
            offset_components: Components that can have offsets
            all_components: All components (for applying offsets)
        """
        self._offset_components = offset_components
        super().__init__(unique_id='offset_panel')

    def _create_layout(self):
        """Create offset control UI."""
        self._layout = dbc.Tab(
            label='🎚️ Offsets',
            tab_id='offsets-tab',
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5('Synchronization Offsets', className='text-center mb-2 mt-2'),
                                html.P(
                                    'Adjust offsets to synchronize components:',
                                    className='text-muted text-center small mb-3',
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Row(  # Scrollable offset controls area
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    id='offsets-container',
                                    style={
                                        'maxHeight': '50vh',
                                        'overflowY': 'auto',
                                        'paddingRight': '5px',
                                    },
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Row(  # Reset All button
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        dbc.Button(
                                            'Reset All Offsets',
                                            id='reset-all-offsets-btn',
                                            color='danger',
                                            className='w-100 mt-3',
                                            size='sm',
                                        )
                                    ]
                                ),
                            ]
                        )
                    ]
                ),
            ],
        )

    def _create_offset_controls(self, components: list[DataComponent], current_offsets: dict[str, int]):
        """Create offset controls for each component"""
        offset_controls = []

        for comp in components:
            if comp:
                offset_key = comp._unique_id
                current_offset = current_offsets.get(offset_key, 0)

                control = dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H6(comp.legend_name, className='mb-2'),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    '-10',
                                                    id={
                                                        'type': 'offset-dec-10',
                                                        'index': offset_key,
                                                    },
                                                    color='secondary',
                                                    size='sm',
                                                    style={'width': '45px'},
                                                )
                                            ],
                                            width='auto',
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    '-',
                                                    id={
                                                        'type': 'offset-dec',
                                                        'index': offset_key,
                                                    },
                                                    color='primary',
                                                    size='sm',
                                                    style={'width': '35px'},
                                                )
                                            ],
                                            width='auto',
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Input(
                                                    id={
                                                        'type': 'offset-value',
                                                        'index': offset_key,
                                                    },
                                                    value=str(current_offset),
                                                    type='text',
                                                    # readonly=True,
                                                    className='text-center',
                                                    style={'fontWeight': 'bold'},
                                                )
                                            ],
                                            width=True,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    '+',
                                                    id={
                                                        'type': 'offset-inc',
                                                        'index': offset_key,
                                                    },
                                                    color='primary',
                                                    size='sm',
                                                    style={'width': '35px'},
                                                )
                                            ],
                                            width='auto',
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    '+10',
                                                    id={
                                                        'type': 'offset-inc-10',
                                                        'index': offset_key,
                                                    },
                                                    color='secondary',
                                                    size='sm',
                                                    style={'width': '45px'},
                                                )
                                            ],
                                            width='auto',
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    'Reset',
                                                    id={
                                                        'type': 'offset-reset',
                                                        'index': offset_key,
                                                    },
                                                    color='warning',
                                                    size='sm',
                                                    style={'width': '60px'},
                                                )
                                            ],
                                            width='auto',
                                        ),
                                    ],
                                    align='center',
                                    className='g-1',
                                ),
                            ],
                            className='p-2',
                        )
                    ],
                    className='mb-2',
                )

                offset_controls.append(control)

        return offset_controls

    def _apply_offsets_to_components(self, offsets: dict[str, int]):
        """Apply offsets to components."""
        for comp_id, offset_value in offsets.items():
            for comp in self._offset_components:
                if comp._unique_id == comp_id:
                    comp.set_offset(offset_value)
                    break

    def activate_callbacks(self):
        @app.callback(
            Output('offsets-store', 'data'),
            Output('offset-update-trigger', 'data'),
            Output('offsets-container', 'children'),
            Input({'type': 'offset-value', 'index': ALL}, 'value'),
            Input({'type': 'offset-dec-10', 'index': ALL}, 'n_clicks'),
            Input({'type': 'offset-dec', 'index': ALL}, 'n_clicks'),
            Input({'type': 'offset-inc', 'index': ALL}, 'n_clicks'),
            Input({'type': 'offset-inc-10', 'index': ALL}, 'n_clicks'),
            Input({'type': 'offset-reset', 'index': ALL}, 'n_clicks'),
            Input('reset-all-offsets-btn', 'n_clicks'),
            Input('offsets-store', 'data'),
            State('offsets-store', 'data'),
            State('offset-update-trigger', 'data'),
            prevent_initial_call=False,
        )
        def manage_offsets(
            offset_input,
            dec_10_clicks,
            dec_clicks,
            inc_clicks,
            inc_10_clicks,
            reset_clicks,
            reset_all_clicks,
            offsets_from_load,
            current_offsets: dict[str, int],
            trigger_counter,
        ):
            ctx = callback_context

            if not current_offsets:
                current_offsets = {}

            # Initialize trigger counter
            if trigger_counter is None:
                trigger_counter = 0

            # Check if this was triggered by loading offsets
            if ctx.triggered and ctx.triggered[0]['prop_id'] == 'offsets-store.data':
                # Offsets were loaded, apply them to components
                self._apply_offsets_to_components(current_offsets)

                # Update UI
                offset_controls = self._create_offset_controls(self._offset_components, current_offsets)
                return current_offsets, trigger_counter + 1, offset_controls

            # Handle button clicks
            elif ctx.triggered:
                trigger = ctx.triggered[0]

                prop_id: str = trigger['prop_id']
                if prop_id == 'reset-all-offsets-btn.n_clicks':
                    # Reset all offsets
                    current_offsets = {}
                    # Reset all components
                    for comp in self._offset_components:
                        comp.set_offset(0)

                elif '.n_clicks' in prop_id or '.value' in prop_id:
                    # Parse the trigger to find which button was clicked
                    parsed_id = json.loads(prop_id.split('.')[0].replace("'", '"'))
                    button_type = parsed_id['type']
                    component_id = parsed_id['index']

                    # Get current offset
                    current_offset = current_offsets.get(component_id, 0)

                    if '.n_clicks' in prop_id:
                        # Update offset based on button type
                        if button_type == 'offset-dec-10':
                            new_offset = current_offset - 10
                        elif button_type == 'offset-dec':
                            new_offset = current_offset - 1
                        elif button_type == 'offset-inc':
                            new_offset = current_offset + 1
                        elif button_type == 'offset-inc-10':
                            new_offset = current_offset + 10
                        elif button_type == 'offset-reset':
                            new_offset = 0
                        else:
                            new_offset = current_offset

                    else:
                        new_offset = int(trigger['value'])

                    # Update offset in store
                    if new_offset == 0:
                        # Remove zero offsets
                        current_offsets.pop(component_id, None)
                    else:
                        current_offsets[component_id] = new_offset

                    # Apply to specific component
                    self._apply_offsets_to_components({component_id: new_offset})

            # Create offset controls UI
            offset_controls = self._create_offset_controls(self._offset_components, current_offsets)

            # Increment trigger counter to force component updates
            return current_offsets, trigger_counter + 1, offset_controls
