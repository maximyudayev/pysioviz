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
