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

from dataclasses import dataclass
from enum import Enum
import numpy as np


class HwAccelEnum(Enum):
    CUDA = 'cuda'
    QSV = 'qsv'
    AMF = 'amf'
    OPENCL = 'opencl'
    VULKAN = 'vulkan'
    D3D12VA = 'd3d12va'
    D3D11VA = 'd3d11va'
    VAAPI = 'vaapi'
    DXVA2 = 'dxva2'


class TaskType(Enum):
    TASK_START_START = 'task-start-start'
    TASK_START_END = 'task-start-end'
    TASK_END_START = 'task-end-start'
    TASK_END_END = 'task-end-end'


class KeyType(Enum):
    ARROW_LEFT = 'ArrowLeft'
    ARROW_RIGHT = 'ArrowRight'
    PAGE_UP = 'PageUp'
    PAGE_DOWN = 'PageDown'


class TriggerId(Enum):
    DECREMENT = 'decrement-btn'
    INCREMENT = 'increment-btn'
    DECREMENT_10 = 'decrement-10-btn'
    INCREMENT_10 = 'increment-10-btn'
    MAIN_SLIDER = 'frame-slider'
    FINE_SLIDER = 'fine-frame-slider'
    KEYBOARD = 'keyboard-event'
    TASK_START_START = 'task-start-start-btn'
    TASK_START_END = 'task-start-end-btn'
    TASK_END_START = 'task-end-start-btn'
    TASK_END_END = 'task-end-end-btn'
    CARD_START_START = 'ts-start-btn'
    CARD_START_END = 'ts-end-btn'
    CARD_END_START = 'te-start-btn'
    CARD_END_END = 'te-end-btn'


class InputId(Enum):
    TASK_START_START = 'task-start-start-input'
    TASK_START_END = 'task-start-end-input'
    TASK_END_START = 'task-end-start-input'
    TASK_END_END = 'task-end-end-input'
    CARD_START_START = 'annotation-ts-start-edit'
    CARD_START_END = 'annotation-ts-end-edit'
    CARD_END_START = 'annotation-te-start-edit'
    CARD_END_END = 'annotation-te-end-edit'


class GlobalVariableId(Enum):
    SELECTED_TIMESTAMP = 'selected-timestamp'
    SYNC_TIMESTAMP = 'sync-timestamp'
    REF_FRAME_TIMESTAMP = 'ref-frame-timestamp'
    FRAME_ID = 'frame-id'


@dataclass
class GroundTruthLabel:
    label: str
    value: str


@dataclass
class CameraConfig:
    video_file: str
    unique_id: str


@dataclass
class VideoComponentInfo:
    type: str
    unique_id: str
    toa_s: np.ndarray
    frame_timestamp: np.ndarray
    sequence: np.ndarray


@dataclass
class AlignmentInfo:
    start_id: int
    end_id: int


@dataclass
class DataRequest:
    """Object wrapping user's element of interest into a fetch request."""

    key: str
    timestamp: float
