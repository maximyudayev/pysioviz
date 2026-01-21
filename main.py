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
from pathlib import Path
# import sys
# import os

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash import ClientsideFunction, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from pysioviz.components.control import AnnotationComponent, FrameSliderComponent, OffsetComponent, SaveLoadComponent
from pysioviz.components.data import DataComponent, VideoComponent, ImuComponent, SkeletonComponent, LinePlotComponent
from pysioviz.utils.sync_utils import calculate_truncation_points, apply_truncation
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import Annotation

# ============================================================================
# CONFIGURATION
# ============================================================================
# Annotation options (label dropdown and values to save in HDF5)
ANNOTATION_OPTIONS = [
  Annotation(label='1. Standing', value='Standing'),
  Annotation(label='2. Walking', value='Walking'),
  Annotation(label='3. Sitting', value='Sitting'),
  Annotation(label='4. Sitting Down', value='Sitting Down'),
  Annotation(label='5. Standing Up', value='Standing Up'),
  Annotation(label='6. Stair Ascent', value='Stair Ascent'),
  Annotation(label='7. Stair Descent', value='Stair Descent'),
  Annotation(label='8. Slope Ascent', value='Slope Ascent'),
  Annotation(label='9. Slope Descent', value='Slope Descent'),
  Annotation(label='10. Step Over', value='Step Over'),
  Annotation(label='11. Cross Country', value='Cross Country'),
  Annotation(label='12. Box Pickup', value='Box Pickup'),
  Annotation(label='13. Box Putdown', value='Box Putdown'),
  Annotation(label='14. Slalom', value='Slalom'),
  Annotation(label='15. Slalom Left Turn', value='Slalom Left Turn'),
  Annotation(label='16. Slalom Right Turn', value='Slalom Right Turn'),
  Annotation(label='17. Standing Turn Left', value='Standing Turn Left'),
  Annotation(label='18. Standing Turn Right', value='Standing Turn Right'),
  Annotation(label='19. Radius Turn Left', value='Radius Turn Left'),
  Annotation(label='20. Radius Turn Right', value='Radius Turn Right'),
]

# Camera configurations
CAMERA_CONFIGS = [
  {
    'video_file': 'cameras_40478064.mkv',
    'unique_id': '40478064',
    'is_reference': True,
  },
  {
    'video_file': 'cameras_40549960.mkv',
    'unique_id': '40549960',
    'is_reference': False,
  },
  {
    'video_file': 'cameras_40549975.mkv',
    'unique_id': '40549975',
    'is_reference': False,
  },
  {
    'video_file': 'cameras_40549976.mkv',
    'unique_id': '40549976',
    'is_reference': False,
  },
]

# File paths, expected that other hdf5 files are in this directory, change if not the case
BASE_PATH = 'C:/Users/maxim/Documents/Code/Projects/kdd2026/data/project_Revalexo/type_Transparent/trial_1'
eye_video_path = Path(f'{BASE_PATH}/glasses_ego.mkv')
eye_hdf5_path = Path(f'{BASE_PATH}/glasses.hdf5')
cameras_hdf5_path = Path(f'{BASE_PATH}/cameras.hdf5')
mvn_hdf5_path = Path(f'{BASE_PATH}/mvn-analyze.hdf5')
exo_hdf5_path = Path(f'{BASE_PATH}/revalexo.hdf5')

HWACCEL = 'd3d12va'

# ============================================================================
# APP INITIALIZATION & SHARED STATE
# ============================================================================
# Define all shared stores BEFORE components that will use them
# This allows different components to access the same stores (global variables)
shared_stores = html.Div(
  [
    # Frame navigation stores
    dcc.Store(id='frame-id', data=0),
    dcc.Store(id='sync-timestamp', data=None),
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
    # Hidden frame input with different ID, NO LONGER USED, REPLACED BY frame-id
    dbc.Input(id='frame-input', type='number', value=0, style={'display': 'none'}),
    # Dummy output for callbacks that don't need real output
    html.Div(id='dummy-output', style={'display': 'none'}),
  ],
  style={'display': 'none'},
)

# ============================================================================
# DATA LOADING - Initialize Data Components
# ============================================================================
if __name__ == '__main__':
  print('Loading data components...', flush=True)

  # Check camera hdf5 exists.
  if not cameras_hdf5_path.exists():
    raise FileNotFoundError(f'Required file {cameras_hdf5_path} not found.')

  # Initialize camera components.
  camera_components: list[VideoComponent] = []
  reference_camera: VideoComponent | None = None
  reference_camera_count = 0

  for config in CAMERA_CONFIGS:
    video_path = Path(f'{BASE_PATH}/{config["video_file"]}')
    if video_path.exists():
      camera = VideoComponent(
        video_filepath=str(video_path),
        hdf5_filepath=str(cameras_hdf5_path),
        unique_id=config['unique_id'],
        time_hdf5_path=f'/cameras/{config["unique_id"]}/toa_s',
        sequence_hdf5_path=f'/cameras/{config["unique_id"]}/frame_sequence_id',
        legend_name=f'Camera {config["unique_id"]}' + (' (Reference)' if config['is_reference'] else ''),
        hwaccel=HWACCEL,
        col_width=6,
        is_reference=config['is_reference'],
        is_highlight=False,
      )
      camera_components.append(camera)
      if config['is_reference']:
        reference_camera = camera
        reference_camera_count += 1
    else:
      print(f'Warning: Camera video {config["video_file"]} not found, skipping...')

  # Validate reference camera
  if reference_camera_count == 0:
    raise ValueError('No reference camera found. At least one camera must be marked as reference.')
  elif reference_camera_count > 1:
    raise ValueError(
      f'Multiple reference cameras found ({reference_camera_count}). Only one camera should be marked as reference.'
    )

  # Eye camera.
  eye_component: VideoComponent | None = None
  if eye_video_path.exists():
    try:
      eye_component = VideoComponent(
        video_filepath=str(eye_video_path),
        hdf5_filepath=str(eye_hdf5_path),
        unique_id='eye_world',
        time_hdf5_path=f'/glasses/ego/toa_s',
        sequence_hdf5_path=f'/glasses/ego/frame_sequence_id',
        legend_name='Eye World Camera',
        hwaccel=HWACCEL,
        col_width=3,
        is_highlight=False,
      )
    except Exception as e:
      print(f'Warning: Failed to load Eye Video: {e}')
  else:
    print('Warning: Eye camera video or HDF5 not found, skipping eye camera...')

  # # TODO: Motors component
  # insole_component = None
  # if insole_hdf5_path.exists():
  #   try:
  #     insole_component = LinePlotComponent(
  #       hdf5_path=str(insole_hdf5_path),
  #       data_path=['/insoles/insoles-data/total_force_left',
  #                 '/insoles/insoles-data/total_force_right'],
  #       timestamp_path='/insoles/insoles-data/toa_s',
  #       unique_id='insole_forces',
  #       legend_name='Insole Pressure',
  #       channel_names=['Left', 'Right'],
  #       plot_window_seconds=0.5,
  #       sampling_rate=100.0,
  #       col_width=3,
  #       y_units="N")
  #   except Exception as e:
  #     print(f"Warning: Failed to load insole data: {e}")
  #     insole_component = None
  # else:
  #   print("Warning: Insole HDF5 not found, skipping insole data...")

  # MVN Analyze components (skeleton and IMU)
  skeleton_component: SkeletonComponent | None = None
  imu_accel_component: ImuComponent | None = None
  imu_gyro_component: ImuComponent | None = None
  imu_mag_component: ImuComponent | None = None

  if mvn_hdf5_path.exists():
    try:
      # Create skeleton component.
      skeleton_component = SkeletonComponent(
        hdf5_path=str(mvn_hdf5_path),
        position_path='/mvn-analyze/xsens-pose/position',
        pos_counter_path='/mvn-analyze/xsens-pose/counter',
        timestamp_path='/mvn-analyze/xsens-time/timestamp_s',
        ref_counter_path='/mvn-analyze/xsens-time/counter',
        unique_id='skeleton_mvn',
        legend_name='Skeleton',
        col_width=3,
      )
    except Exception as e:
      print(f'Warning: Failed to load skeleton data: {e}')
      skeleton_component = None

    try:
      # Create IMU components.
      imu_accel_component = ImuComponent(
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
        col_width=4,
      )

      imu_gyro_component = ImuComponent(
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
        col_width=4,
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
        col_width=4,
      )
    except Exception as e:
      print(f'Warning: Failed to load IMU data: {e}')
      imu_accel_component = None
      imu_gyro_component = None
      imu_mag_component = None
  else:
    print('Warning: MVN Analyze HDF5 not found, skipping skeleton and IMU data...')

  # Calculate synchronization truncation points.
  skeleton_components = [skeleton_component] if skeleton_component else []
  imu_components = [c for c in [imu_accel_component, imu_gyro_component, imu_mag_component] if c is not None]

  # This finds a common starting point across all data streams using the
  # reference camera's 100th frame as the synchronization anchor
  truncation_points = calculate_truncation_points(
    camera_components,
    eye_component,
    emg_components=None,
    skeleton_components=skeleton_components,
    insole_components=None,
    imu_components=imu_components,
    baseline_frame=100,
  )  # Using 100th frame of reference camera

  # Apply truncation to all components
  all_components: list[DataComponent] = camera_components + ([eye_component] if eye_component else []) + skeleton_components + imu_components
  apply_truncation(all_components, truncation_points)

  # Use reference camera's frame count as the reference for slider
  total_frames = reference_camera.get_truncated_frame_count()
  fps = float(reference_camera._fps)

  # Store reference camera globally
  app.reference_camera = reference_camera
  app.all_components = all_components

  print(f'\nUsing Camera {reference_camera._unique_id} as reference for synchronization')
  print(f'Total frames in reference camera: {total_frames}')

  # ============================================================================
  # CONTROL COMPONENT INITIALIZATION
  # These components handle frame navigation, annotations, offsets, and saving/loading
  # ============================================================================
  print('Initializing control components...')

  # Initialize control components
  frame_slider = FrameSliderComponent(total_frames, fps, reference_camera)
  annotations = AnnotationComponent(ANNOTATION_OPTIONS)

  # Get all non-camera components that can have offsets
  offset_components = []
  if eye_component:
    offset_components.append(eye_component)
  offset_components.extend(skeleton_components)
  offset_components.extend(imu_components)

  offsets = OffsetComponent(offset_components, all_components)
  save_load = SaveLoadComponent()

  # ============================================================================
  # LAYOUT ASSEMBLY
  # ============================================================================
  # Hidden frame input
  frame_input = dbc.Input(id='frame-id', type='number', value=0, style={'display': 'none'})

  # Main camera grid - 2x2 layout with full height cameras
  main_camera_grid = html.Div(
    [
      dbc.Row(
        [
          dbc.Col(
            [
              html.Div(
                [
                  html.Div(
                    camera_components[0]._image if len(camera_components) > 0 else html.Div('Camera 1 Missing'),
                    style={
                      'height': 'calc(100% - 25px)',
                      'width': '100%',
                      'overflow': 'hidden',
                    },
                  ),
                  camera_components[0]._timestamp_display if len(camera_components) > 0 else html.Div(),
                ],
                style={
                  'height': '100%',
                  'backgroundColor': '#ffffff',
                  'border': '1px solid #dee2e6',
                  'borderRadius': '4px',
                  'padding': '5px',
                  'display': 'flex',
                  'flexDirection': 'column',
                },
              )
            ],
            width=6,
            className='pe-1',
            style={'height': '50vh'},
          ),
          dbc.Col(
            [
              html.Div(
                [
                  html.Div(
                    camera_components[1]._image if len(camera_components) > 1 else html.Div('Camera 2 Missing'),
                    style={
                      'height': 'calc(100% - 25px)',
                      'width': '100%',
                      'overflow': 'hidden',
                    },
                  ),
                  camera_components[1]._timestamp_display if len(camera_components) > 1 else html.Div(),
                ],
                style={
                  'height': '100%',
                  'backgroundColor': '#ffffff',
                  'border': '1px solid #dee2e6',
                  'borderRadius': '4px',
                  'padding': '5px',
                  'display': 'flex',
                  'flexDirection': 'column',
                },
              )
            ],
            width=6,
            className='ps-1',
            style={'height': '50vh'},
          ),
        ],
        className='g-0 mb-2',
      ),
      dbc.Row(
        [
          dbc.Col(
            [
              html.Div(
                [
                  html.Div(
                    camera_components[2]._image if len(camera_components) > 2 else html.Div('Camera 3 Missing'),
                    style={
                      'height': 'calc(100% - 25px)',
                      'width': '100%',
                      'overflow': 'hidden',
                    },
                  ),
                  camera_components[2]._timestamp_display if len(camera_components) > 2 else html.Div(),
                ],
                style={
                  'height': '100%',
                  'backgroundColor': '#ffffff',
                  'border': '1px solid #dee2e6',
                  'borderRadius': '4px',
                  'padding': '5px',
                  'display': 'flex',
                  'flexDirection': 'column',
                },
              )
            ],
            width=6,
            className='pe-1',
            style={'height': '50vh'},
          ),
          dbc.Col(
            [
              html.Div(
                [
                  html.Div(
                    camera_components[3]._image if len(camera_components) > 3 else html.Div('Camera 4 Missing'),
                    style={
                      'height': 'calc(100% - 25px)',
                      'width': '100%',
                      'overflow': 'hidden',
                    },
                  ),
                  camera_components[3]._timestamp_display if len(camera_components) > 3 else html.Div(),
                ],
                style={
                  'height': '100%',
                  'backgroundColor': '#ffffff',
                  'border': '1px solid #dee2e6',
                  'borderRadius': '4px',
                  'padding': '5px',
                  'display': 'flex',
                  'flexDirection': 'column',
                },
              )
            ],
            width=6,
            className='ps-1',
            style={'height': '50vh'},
          ),
        ],
        className='g-0',
      ),
    ],
    style={'marginBottom': '20px'},
  )

  # Data visualization rows
  data_viz_rows = []

  # Row with Eye camera, EMG, Skeleton, and Pressure - only show if components exist
  if any([eye_component, skeleton_component]):
    data_viz_rows.append(
      dbc.Row(
        [
          eye_component.layout if eye_component else html.Div(),
          skeleton_component.layout if skeleton_component else html.Div(),
        ],
        className='mb-3',
      )
    )

  # Row with IMU sensors - only show if components exist
  if any([imu_accel_component, imu_gyro_component, imu_mag_component]):
    data_viz_rows.append(
      dbc.Row(
        [
          imu_accel_component.layout if imu_accel_component else html.Div(),
          imu_gyro_component.layout if imu_gyro_component else html.Div(),
          imu_mag_component.layout if imu_mag_component else html.Div(),
        ],
        id='imu-row',
        className='mb-3',
      )
    )

  # Right panel with tabs
  right_panel = html.Div(
    [
      dbc.Card(
        [
          dbc.CardBody(
            [
              dbc.Tabs(
                [
                  dbc.Tab(
                    annotations.layout,
                    label='📝 Annotations',
                    tab_id='annotations-tab',
                  ),
                  dbc.Tab(
                    offsets.layout,
                    label='🎚️ Offsets',
                    tab_id='offsets-tab',
                  ),
                ],
                id='right-panel-tabs',
                active_tab='annotations-tab',
              )
            ],
            style={
              'height': 'calc(100% - 50px)',
              'overflowY': 'hidden',
              'padding': '10px',
            },
          )
        ],
        style={'height': 'calc(100% - 50px)'},
      ),
      save_load.layout,
    ],
    style={
      'position': 'fixed',
      'right': '0',
      'top': '10px',
      'bottom': '80px',
      'width': '20%',
      'zIndex': '2000',
      'backgroundColor': 'white',
      'border': '1px solid #dee2e6',
      'borderRadius': '4px',
    },
  )

  # Main content area with margin for fixed elements
  main_content = html.Div(
    [
      main_camera_grid,
      *data_viz_rows,  # Unpack all data visualization rows
    ],
    id='main-content',
    style={
      'marginRight': '20%',  # Leave space for right block (matches its width)
      'marginBottom': '180px',  # Leave space for frame controls
      'padding': '10px',
      'transition': 'margin-bottom 0.3s ease-in-out',
    },
  )

  # Assemble complete layout
  app.layout = html.Div(
    [
      # Hidden stores
      shared_stores,
      # Main content
      main_content,
      # Right panel
      right_panel,
      # Frame controls at bottom
      frame_slider.layout,
    ]
  )

  # ============================================================================
  # CROSS-COMPONENT CALLBACKS
  # ============================================================================
  # Click handler for all visualization components
  # This allows clicking on any data visualization (video, plot, skeleton)
  # to display its timestamp in the annotation panel (can be used to copy for more precise annotations)
  click_inputs = []
  for cam in camera_components:
    click_inputs.append(Input(f'{cam._unique_id}-video', 'clickData'))
  if eye_component:
    click_inputs.append(Input('eye_world-video', 'clickData'))
  if skeleton_component:
    click_inputs.append(Input('skeleton_mvn-skeleton', 'clickData'))
  if imu_accel_component:
    click_inputs.append(Input('imu_accelerometer-imu-plot', 'clickData'))
  if imu_gyro_component:
    click_inputs.append(Input('imu_gyroscope-imu-plot', 'clickData'))
  if imu_mag_component:
    click_inputs.append(Input('imu_magnetometer-imu-plot', 'clickData'))

  @app.callback(
    Output('selected-timestamp', 'value'),
    click_inputs,
    State('frame-id', 'data'),
    State('sync-timestamp', 'data'),
    prevent_initial_call=True,
  )
  def handle_all_clicks(*args):
    """Centralized handler for click events from all components."""
    from dash import callback_context

    ctx = callback_context
    if not ctx.triggered:
      return ''

    # Get which component was clicked
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Get the current frame and sync timestamp from states
    current_frame = args[-2]
    sync_timestamp = args[-1]

    # Handle camera clicks
    for cam in camera_components:
      if trigger_id == f'{cam._unique_id}-video':
        if cam._is_reference:
          actual_frame = cam._start_frame + current_frame
        else:
          actual_frame = cam._get_frame_for_timestamp(sync_timestamp)
        timestamp = cam._get_timestamp_at_frame(actual_frame)
        return f'Camera {cam._unique_id} - toa_s: {timestamp:.7f}'

    # Handle eye video click
    if eye_component and trigger_id == 'eye_world-video':
      if sync_timestamp:
        actual_frame = eye_component._get_frame_for_timestamp(sync_timestamp)
        timestamp = eye_component._get_timestamp_at_frame(actual_frame)
        return f'Eye video - frame_timestamp: {timestamp:.7f}'

    # Handle skeleton click
    if skeleton_component and trigger_id == 'skeleton_mvn-skeleton':
      if sync_timestamp:
        current_idx = skeleton_component.get_timestamp_for_sync(sync_timestamp)
        timestamp = (
          skeleton_component._timestamps[current_idx].item() if current_idx < len(skeleton_component._timestamps) else 0
        )
        return f'Skeleton MVN - timestamp_s: {timestamp:.7f} (index: {current_idx})'

    # Handle IMU clicks
    if trigger_id in [
      'imu_accelerometer-imu-plot',
      'imu_gyroscope-imu-plot',
      'imu_magnetometer-imu-plot',
    ]:
      imu_type = trigger_id.split('-')[0].split('_')[1]  # Extract accelerometer/gyroscope/magnetometer

      # Get the appropriate component
      if imu_type == 'accelerometer' and imu_accel_component:
        imu_component = imu_accel_component
      elif imu_type == 'gyroscope' and imu_gyro_component:
        imu_component = imu_gyro_component
      elif imu_type == 'magnetometer' and imu_mag_component:
        imu_component = imu_mag_component
      else:
        return ''

      if sync_timestamp:
        current_idx = imu_component.get_timestamp_for_sync(sync_timestamp)
        timestamp = imu_component._timestamps[current_idx].item() if current_idx < len(imu_component._timestamps) else 0
        return f'IMU {imu_type} - timestamp_s: {timestamp:.7f} (index: {current_idx})'

    return ''

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
  def update_annotations_display_fix(annotations_data, expanded_state):
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
  print(f'\nUsing Camera {reference_camera._unique_id} as reference')
  print(f'Total frames: {total_frames}')
  app.run(debug=True)
