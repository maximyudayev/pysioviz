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

# Hotfix run using python multimodal_annotation.py (adding parent directory to path)
import os
from pathlib import Path

from dash import ClientsideFunction, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from pysioviz.components.control import AnnotationComponent, FrameSliderComponent, OffsetComponent, SaveLoadComponent
from pysioviz.components.data import (
    DataComponent,
    ReferenceVideoComponent,
    VideoComponent,
    ImuComponent,
    SkeletonComponent,
    LinePlotComponent,
)
from pysioviz.utils.sync_utils import add_alignment_info, extract_refticks_from_cameras
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import GlobalVariableId, GroundTruthLabel, CameraConfig, HwAccelEnum

# ============================================================================
# CONFIGURATION
# ============================================================================
# Annotation options (label dropdown and values to save in HDF5)
ANNOTATION_OPTIONS = [
    GroundTruthLabel(label='1. Standing', value='Standing'),
    GroundTruthLabel(label='2. Walking', value='Walking'),
    GroundTruthLabel(label='3. Sitting', value='Sitting'),
    GroundTruthLabel(label='4. Sitting Down', value='Sitting Down'),
    GroundTruthLabel(label='5. Standing Up', value='Standing Up'),
    GroundTruthLabel(label='6. Stair Ascent', value='Stair Ascent'),
    GroundTruthLabel(label='7. Stair Descent', value='Stair Descent'),
    GroundTruthLabel(label='8. Slope Ascent', value='Slope Ascent'),
    GroundTruthLabel(label='9. Slope Descent', value='Slope Descent'),
    GroundTruthLabel(label='10. Step Over', value='Step Over'),
    GroundTruthLabel(label='11. Cross Country', value='Cross Country'),
    GroundTruthLabel(label='12. Box Pickup', value='Box Pickup'),
    GroundTruthLabel(label='13. Box Putdown', value='Box Putdown'),
    GroundTruthLabel(label='14. Slalom', value='Slalom'),
    GroundTruthLabel(label='15. Slalom Left Turn', value='Slalom Left Turn'),
    GroundTruthLabel(label='16. Slalom Right Turn', value='Slalom Right Turn'),
    GroundTruthLabel(label='17. Standing Turn Left', value='Standing Turn Left'),
    GroundTruthLabel(label='18. Standing Turn Right', value='Standing Turn Right'),
    GroundTruthLabel(label='19. Radius Turn Left', value='Radius Turn Left'),
    GroundTruthLabel(label='20. Radius Turn Right', value='Radius Turn Right'),
]

# Camera configurations
CAMERA_CONFIGS = [
    CameraConfig(
        video_file='cameras_40478064.mkv',
        unique_id='40478064',
    ),
    CameraConfig(
        video_file='cameras_40549960.mkv',
        unique_id='40549960',
    ),
    CameraConfig(
        video_file='cameras_40549975.mkv',
        unique_id='40549975',
    ),
    CameraConfig(
        video_file='cameras_40549976.mkv',
        unique_id='40549976',
    ),
]

# File paths, expected that other hdf5 files are in this directory
BASE_PATH = os.environ['PYSIOVIZ_BASE_PATH']
eye_video_path = Path(f'{BASE_PATH}/glasses_ego.mkv')
eye_hdf5_path = Path(f'{BASE_PATH}/glasses.hdf5')
cameras_hdf5_path = Path(f'{BASE_PATH}/cameras.hdf5')
mvn_hdf5_path = Path(f'{BASE_PATH}/mvn-analyze.hdf5')
exo_hdf5_path = Path(f'{BASE_PATH}/revalexo.hdf5')

HWACCEL = HwAccelEnum.D3D12VA.value

# ============================================================================
# APP INITIALIZATION & SHARED STATE
# ============================================================================
# Define all shared stores BEFORE components that will use them
# This allows different components to access the same stores (global variables)
shared_stores = [
    # Frame navigation stores
    dcc.Store(
        id=GlobalVariableId.FRAME_ID.value, data=0
    ),  # ID of the frame from the combined timestamps of the external reference cameras
    dcc.Store(
        id=GlobalVariableId.REF_FRAME_TIMESTAMP.value, data=None
    ),  # synced `frame_timestamp` of external reference cameras
    dcc.Store(
        id=GlobalVariableId.SYNC_TIMESTAMP.value, data=None
    ),  # Inter-modality synchronized `toa_s` timeline for reliable retrieval
    dcc.Store(id='fine-slider-window', data=250),
    dcc.Store(id='fine-slider-center', data=0),
    dcc.Store(id='controls-visible', data=True),
    # Annotation stores
    dcc.Store(id='annotations-store', data=[]),
    dcc.Store(id='annotation-expanded', data={}),
    dcc.Store(id='active-input', data=None),
    dcc.Store(id='delete-target', data=None),
    # Offset stores
    dcc.Store(id='offsets-store', data={}),
    dcc.Store(id='offset-update-trigger', data=0),
    # Event stores
    dcc.Store(id='keyboard-event', data=None),
    dcc.Store(id='feedback-message', data=None),
    # Hidden div to trigger keyboard setup
    html.Div(id='keyboard-setup-trigger', style={'display': 'none'}),
    # Dummy output for callbacks that don't need real output
    html.Div(id='dummy-output', style={'display': 'none'}),
]

# =========================================
# MAIN PROGRAM - Initialize Data Components
# =========================================
if __name__ == '__main__':
    print('Loading data components...', flush=True)

    # =======
    # CAMERAS
    # =======
    # Initialize camera components.
    camera_components: list[ReferenceVideoComponent] = []
    for config in CAMERA_CONFIGS:
        video_path = Path(f'{BASE_PATH}/{config.video_file}')
        camera = ReferenceVideoComponent(
            video_filepath=str(video_path),
            hdf5_filepath=str(cameras_hdf5_path),
            unique_id=config.unique_id,
            toa_hdf5_path=f'/cameras/{config.unique_id}/toa_s',
            timestamp_hdf5_path=f'/cameras/{config.unique_id}/frame_timestamp',
            sequence_hdf5_path=f'/cameras/{config.unique_id}/frame_index',
            legend_name=f'Camera {config.unique_id}',
            hwaccel=HWACCEL,
            prefetch_window_s=2.0,
        )
        camera_components.append(camera)

    # ============
    # SMARTGLASSES
    # ============
    eye_component = VideoComponent(
        video_filepath=str(eye_video_path),
        hdf5_filepath=str(eye_hdf5_path),
        unique_id='eye_world',
        toa_hdf5_path=f'/glasses/ego/toa_s',
        timestamp_hdf5_path=f'/glasses/ego/frame_timestamp',
        sequence_hdf5_path=f'/glasses/ego/frame_index',
        legend_name='Eye World Camera',
        hwaccel=HWACCEL,
        prefetch_window_s=10.0,
        div_height='20vh',
    )

    # ======
    # MOTORS
    # ======
    # motors_component = LinePlotComponent(
    #     hdf5_path=str(exo_hdf5_path),
    #     data_path=[
    #     '/revalexo/motor_hip_right/position',
    #     '/revalexo/motor_knee_right/position',
    #     '/revalexo/motor_hip_left/position',
    #     '/revalexo/motor_knee_left/position',
    #     ],
    #     timestamp_path=[
    #     '/revalexo/motor_hip_right/timestamp',
    #     '/revalexo/motor_knee_right/timestamp',
    #     '/revalexo/motor_hip_left/timestamp',
    #     '/revalexo/motor_knee_left/timestamp',
    #     ],
    #     unique_id='motor_angles',
    #     legend_name='Motor Angles',
    #     channel_names=['Hip Right', 'Knee Right', 'Hip Left', 'Knee Left'],
    #     plot_window_seconds=10,
    #     sampling_rate=50.0,
    #     y_units="degrees"
    # )

    # ========
    # SKELETON
    # ========
    skeleton_component = SkeletonComponent(
        hdf5_path=str(mvn_hdf5_path),
        position_path='/mvn-analyze/xsens-pose/position',
        pos_counter_path='/mvn-analyze/xsens-pose/counter',
        timestamp_path='/mvn-analyze/xsens-time/timestamp_s',
        ref_counter_path='/mvn-analyze/xsens-time/counter',
        unique_id='skeleton_mvn',
        legend_name='Skeleton',
    )

    # ============
    # Wearable IMU
    # ============
    imu_acc_component = ImuComponent(
        hdf5_path=str(mvn_hdf5_path),
        data_path='/mvn-analyze/xsens-motion-trackers/acceleration',
        data_counter_path='/mvn-analyze/xsens-motion-trackers/counter',
        timestamp_path='/mvn-analyze/xsens-time/timestamp_s',
        ref_counter_path='/mvn-analyze/xsens-time/counter',
        unique_id='imu_accelerometer',
        legend_name='Accelerometer',
        sensor_type='accelerometer',
        plot_window_seconds=10,
        sampling_rate=60.0,
    )

    imu_gyr_component = ImuComponent(
        hdf5_path=str(mvn_hdf5_path),
        data_path='/mvn-analyze/xsens-motion-trackers/gyroscope',
        data_counter_path='/mvn-analyze/xsens-motion-trackers/counter',
        timestamp_path='/mvn-analyze/xsens-time/timestamp_s',
        ref_counter_path='/mvn-analyze/xsens-time/counter',
        unique_id='imu_gyroscope',
        legend_name='Gyroscope',
        sensor_type='gyroscope',
        plot_window_seconds=10,
        sampling_rate=60.0,
    )

    imu_mag_component = ImuComponent(
        hdf5_path=str(mvn_hdf5_path),
        data_path='/mvn-analyze/xsens-motion-trackers/magnetometer',
        data_counter_path='/mvn-analyze/xsens-motion-trackers/counter',
        timestamp_path='/mvn-analyze/xsens-time/timestamp_s',
        ref_counter_path='/mvn-analyze/xsens-time/counter',
        unique_id='imu_magnetometer',
        legend_name='Magnetometer',
        sensor_type='magnetometer',
        plot_window_seconds=10,
        sampling_rate=60.0,
    )

    # =============================================
    # Extract reference time ticks for frame slider
    # =============================================
    combined_timestamps, combined_toas, camera_align_info, start_trial_toa, end_trial_toa = (
        extract_refticks_from_cameras(camera_components)
    )
    add_alignment_info(camera_components, camera_align_info)

    offset_components: list[DataComponent] = [
        eye_component,
        skeleton_component,
        imu_acc_component,
        imu_gyr_component,
        imu_mag_component,
        # TODO: motors
        # TODO: integrated IMUs
    ]

    data_components: list[DataComponent] = [
        *camera_components,
        *offset_components,
    ]

    print(f'\nUsing Cameras as reference for visualization duration', flush=True)
    print(f'Total {len(combined_timestamps)} video frames in trial, for {end_trial_toa - start_trial_toa}s', flush=True)

    # ================================
    # CONTROL COMPONENT INITIALIZATION
    # ================================
    print('Initializing control components...', flush=True)

    # These components handle frame navigation, annotations, offsets, and saving/loading
    # Initialize control components
    frame_slider = FrameSliderComponent(camera_components, combined_timestamps)
    annotations = AnnotationComponent(ANNOTATION_OPTIONS, combined_timestamps, combined_toas)
    offsets = OffsetComponent(offset_components)
    save_load = SaveLoadComponent()

    # ================
    # COMPONENT LAYOUT
    # ================
    # Cameras 2x2 mini grid
    camera_grid = dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [camera_components[0].layout],
                        width=6,
                    ),
                    dbc.Col(
                        [camera_components[1].layout],
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [camera_components[2].layout],
                        width=6,
                    ),
                    dbc.Col(
                        [camera_components[3].layout],
                        width=6,
                    ),
                ]
            ),
        ],
        width=9,
    )

    # Skeleton from Xsens
    skeleton = dbc.Col(
        [
            skeleton_component.layout,
        ],
        width=3,
        align='center',
    )

    # Egocentric view
    ego_view = dbc.Col([eye_component.layout], width=3)

    # Wearable IMUs
    imu_row = [
        dbc.Col([imu_acc_component.layout], width=4),
        dbc.Col([imu_gyr_component.layout], width=4),
        dbc.Col([imu_mag_component.layout], width=4),
    ]

    # Frame slider
    slider = dbc.Col([frame_slider.layout], width=9)

    # Right panel with tabs
    right_panel = dbc.Card(
        [
            dbc.CardBody(
                [
                    save_load.layout,
                    dbc.Row(
                        [
                            dbc.Tabs(
                                [
                                    annotations.layout,
                                    offsets.layout,
                                ],
                                id='right-panel-tabs',
                                active_tab='annotations-tab',
                            )
                        ],
                        className='g-0',
                    ),
                ],
            )
        ],
    )

    # ===================
    # APP LAYOUT ASSEMBLY
    # ===================
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(  # Main scrollable content
                        [
                            dbc.Row(  # Skeleton and reference cameras
                                [
                                    skeleton,
                                    camera_grid,
                                ]
                            ),
                            dbc.Row(  # Motors and ego view
                                [
                                    ego_view,
                                    # TODO: motors
                                ]
                            ),
                            dbc.Row(  # Wearable IMUs
                                imu_row,
                                id='imu-row',
                                style={
                                    'margin-bottom': '250px',
                                },
                            ),
                            dbc.Row(
                                [slider],
                            ),
                        ],
                        width=9,
                    ),
                    dbc.Col(  # Annotation / offsets right sidebar
                        [right_panel],
                        width=3,
                        style={
                            'position': 'fixed',
                            'right': '0',
                            'top': '0',
                            'zIndex': '2000',
                        },
                    ),
                ]
            ),
            # Hidden stores
            *shared_stores,
        ],
        fluid=True,
    )

    # =========================
    # CROSS-COMPONENT CALLBACKS
    # =========================
    # Click handler for all visualization components
    # This allows clicking on any data visualization (video, plot, skeleton)
    # to display its timestamp in the annotation panel (can be used to copy for more precise annotations)
    click_inputs: list[Input] = [comp.make_click_input() for comp in data_components]
    clicked_components = dict(zip(map(lambda x: x.component_id_str(), click_inputs), data_components))

    @app.callback(
        Output(GlobalVariableId.SELECTED_TIMESTAMP.value, 'value'),
        click_inputs,
        State(GlobalVariableId.REF_FRAME_TIMESTAMP.value, 'data'),
        State(GlobalVariableId.SYNC_TIMESTAMP.value, 'data'),
        prevent_initial_call=True,
    )
    def handle_all_clicks(*args):
        """Centralized handler for click events from all components."""
        ctx = callback_context
        if not ctx.triggered:
            return ''

        # Get which component was clicked
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Get the current frame and sync timestamp from states
        ref_frame_timestamp = args[-2]
        sync_timestamp = args[-1]

        # Handle click
        comp = clicked_components[trigger_id]
        return comp.handle_click(ref_frame_timestamp, sync_timestamp)

    # Fix for the annotation display update callback
    # This is necessary to ensure the annotation UI refreshes when annotations are loaded from file
    # Without this, loaded annotations would be in the store but not visible
    @app.callback(
        Output('annotations-container', 'children', allow_duplicate=True),
        Output('annotation-counter', 'children', allow_duplicate=True),
        Input('annotations-store', 'data'),
        State('annotation-expanded', 'data'),
        prevent_initial_call=True,
    )
    def update_annotations_display_fix(annotations_data: list[dict], expanded_state):
        """Update the annotations display when store changes (e.g., after loading)"""
        if annotations_data is None:
            annotations_data = []
        if expanded_state is None:
            expanded_state = {}

        # Use the annotation component's method to create cards
        annotation_cards = annotations._create_annotation_cards(annotations_data, expanded_state)
        counter_text = f'Total: {len(annotations_data)} annotations'
        return annotation_cards, counter_text

    # ============================================================================
    # APPLICATION STARTUP
    # ============================================================================
    print(f'Total frames: {len(combined_timestamps)}', flush=True)
    app.run(debug=True)
