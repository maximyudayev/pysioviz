"""
Pipeline Overlay Manager for PysioViz.

Loads and synchronizes computer vision gait analysis artifacts (Stage 1 ingestion
bounding boxes and foreground segmentation masks, Stage 3 2D/3D dense landmarks,
and Stage 7 wireframe reprojection overlays) on top of external camera video feeds.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtGui

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


# Hardware camera serial IDs matching hermes-gait-analysis conventions
DEFAULT_CAMERA_IDS: Tuple[int, ...] = (40478064, 40549960, 40549975, 40549976)
CAMERA_ID_TO_INT8: Dict[int, int] = {cam_id: idx for idx, cam_id in enumerate(DEFAULT_CAMERA_IDS)}
INT8_TO_CAMERA_ID: Dict[int, int] = {idx: cam_id for idx, cam_id in enumerate(DEFAULT_CAMERA_IDS)}

# Standard anatomical bone linkages for monocular poses (e.g. 17 COCO / 23 Xsens joints)
XSENS_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (4, 7), (7, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18),
    (0, 19), (19, 20), (20, 21), (21, 22),
]


@dataclass
class FrameOverlayData:
    """Synchronized overlay telemetry for a single camera frame."""

    cam_id: str
    frame_idx: int
    t_mid_sec: float
    subj_present: bool
    reid_cosine_sim: float
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    v_idx: Optional[int] = None
    keypoints_2d: Optional[np.ndarray] = None  # (K, 2) float32 in full-frame pixels
    keypoint_confs: Optional[np.ndarray] = None  # (K,) float32
    bone_polylines: Optional[List[List[Tuple[float, float]]]] = None


class PipelineOverlayManager:
    """Manages lazy loading, memory indexing, and fast retrieval of pipeline overlays."""

    def __init__(
        self,
        parquet_path: Optional[str | Path] = None,
        zarr_path: Optional[str | Path] = None,
        dense_zarr_path: Optional[str | Path] = None,
        telemetry_dir: Optional[str | Path] = None,
        max_mask_cache_size: int = 1024,
    ):
        self.parquet_path: Optional[Path] = Path(parquet_path) if parquet_path else None
        self.zarr_path: Optional[Path] = Path(zarr_path) if zarr_path else None
        self.dense_zarr_path: Optional[Path] = Path(dense_zarr_path) if dense_zarr_path else None
        self.telemetry_dir: Optional[Path] = Path(telemetry_dir) if telemetry_dir else None

        # Lookups: (cam_str, frame_idx) -> FrameOverlayData
        self._frame_data: Dict[Tuple[str, int], FrameOverlayData] = {}
        # Sorted frame indices per camera for fast nearest-neighbor fallback
        self._cam_frames: Dict[str, np.ndarray] = {}

        # Zarr store references
        self._zarr_root: Optional[Any] = None
        self._dense_zarr_root: Optional[Any] = None

        # In-memory numerical caches: cam_str -> np.ndarray
        self._affines: Dict[str, np.ndarray] = {}
        self._mono_kps: Dict[str, np.ndarray] = {}
        self._mono_confs: Dict[str, np.ndarray] = {}
        self._dense_kps: Dict[str, np.ndarray] = {}
        self._dense_confs: Dict[str, np.ndarray] = {}

        # High-performance LRU mask cache: (cam_str, v_idx) -> (QImage, xmin, ymin, side)
        self._max_mask_cache_size = max(128, max_mask_cache_size)
        self._mask_qimage_cache: OrderedDict[Tuple[str, int], Tuple[QtGui.QImage, float, float, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Pre-built color table for Format_Indexed8 (0: transparent, 1: neon emerald #00f5a0 with ~38% alpha)
        self._mask_color_table: List[int] = [
            QtGui.qRgba(0, 0, 0, 0),
            QtGui.qRgba(0, 245, 160, 95),
        ]

        # Background asynchronous mask prefetch queue and worker thread
        self._prefetch_queue: queue.Queue[Optional[Tuple[str, int]]] = queue.Queue(maxsize=64)
        self._prefetch_running = True
        self._prefetch_thread = threading.Thread(
            target=self._run_prefetch_worker,
            daemon=True,
            name='OverlayMaskPrefetchWorker',
        )

        # Reprojection overlays from Stage 7 JSON: (cam_str, frame_idx) -> polylines
        self._reproj_polylines: Dict[Tuple[str, int], List[List[Tuple[float, float]]]] = {}

        # Display flags
        self.show_bboxes: bool = True
        self.show_masks: bool = True
        self.show_poses: bool = True
        self.show_reid_labels: bool = True

        self._load_all()
        self._prefetch_thread.start()

    @classmethod
    def discover(cls, search_root: str | Path) -> Optional[PipelineOverlayManager]:
        """Auto-discover Stage 1/3/7 artifacts in a session or workspace root."""
        root = Path(search_root).resolve()
        candidates_parquet = [
            root / '01_ingestion' / 'frame_index.parquet',
            root / 'frame_index.parquet',
            root.parent / '01_ingestion' / 'frame_index.parquet',
            root / '..' / '01_ingestion' / 'frame_index.parquet',
        ]
        candidates_zarr = [
            root / '01_ingestion' / 'masks_and_crops.zarr',
            root / 'masks_and_crops.zarr',
            root.parent / '01_ingestion' / 'masks_and_crops.zarr',
            root / '..' / '01_ingestion' / 'masks_and_crops.zarr',
        ]
        candidates_dense = [
            root / '03_landmarks_2d_3d' / 'dense_observations.zarr',
            root / 'dense_observations.zarr',
            root.parent / '03_landmarks_2d_3d' / 'dense_observations.zarr',
        ]
        candidates_telemetry = [
            root / '07_gui_telemetry',
            root.parent / '07_gui_telemetry',
        ]

        found_parquet = next((p for p in candidates_parquet if p.exists() and p.is_file()), None)
        found_zarr = next((p for p in candidates_zarr if p.exists()), None)
        found_dense = next((p for p in candidates_dense if p.exists()), None)
        found_telem = next((p for p in candidates_telemetry if p.exists() and p.is_dir()), None)

        if found_parquet or found_zarr:
            return cls(
                parquet_path=found_parquet,
                zarr_path=found_zarr,
                dense_zarr_path=found_dense,
                telemetry_dir=found_telem,
            )
        return None

    def _load_all(self):
        """Load parquet index, initialize zarr handles, and load telemetry polylines."""
        self._load_parquet()
        self._open_zarr()
        self._load_telemetry_json()

    def _normalize_cam_id(self, cam_id: Any) -> str:
        """Convert camera ID representation into canonical serial string."""
        s = str(cam_id).strip()
        if s.startswith('cam_'):
            s = s[4:]
        elif s.startswith('cameras_'):
            s = s[8:]
        # Check if int8 index (0..3)
        if s.isdigit():
            val = int(s)
            if val in INT8_TO_CAMERA_ID:
                return str(INT8_TO_CAMERA_ID[val])
        return s

    def _load_parquet(self):
        """Load and index frame_index.parquet into fast memory tables."""
        if not self.parquet_path or not self.parquet_path.exists() or not HAS_PYARROW:
            return

        try:
            tbl = pq.read_table(str(self.parquet_path))
            pydict = tbl.to_pydict()
            cam_ids = pydict.get('cam_id', [])
            frame_indices = pydict.get('frame_idx', [])
            t_mids = pydict.get('t_mid_sec', [])
            subj_presents = pydict.get('subj_present', [])
            reid_sims = pydict.get('reid_cosine_sim', [])
            bboxes = pydict.get('bbox_xyxy', [])

            cam_v_counts: Dict[str, int] = {}
            frames_by_cam: Dict[str, List[int]] = {}

            for i in range(len(frame_indices)):
                raw_cid = cam_ids[i]
                cam_str = self._normalize_cam_id(raw_cid)
                f_idx = int(frame_indices[i])
                v_idx = cam_v_counts.get(cam_str, 0)
                cam_v_counts[cam_str] = v_idx + 1

                bbox_raw = bboxes[i] if i < len(bboxes) else [0.0, 0.0, 0.0, 0.0]
                bbox = tuple(float(x) for x in bbox_raw) if len(bbox_raw) == 4 else None

                entry = FrameOverlayData(
                    cam_id=cam_str,
                    frame_idx=f_idx,
                    t_mid_sec=float(t_mids[i]) if i < len(t_mids) else 0.0,
                    subj_present=bool(subj_presents[i]) if i < len(subj_presents) else False,
                    reid_cosine_sim=float(reid_sims[i]) if i < len(reid_sims) else 0.0,
                    bbox_xyxy=bbox,
                    v_idx=v_idx,
                )
                self._frame_data[(cam_str, f_idx)] = entry
                if cam_str not in frames_by_cam:
                    frames_by_cam[cam_str] = []
                frames_by_cam[cam_str].append(f_idx)

            for c, f_list in frames_by_cam.items():
                self._cam_frames[c] = np.array(f_list, dtype=np.int64)

        except Exception as e:
            print(f'[PipelineOverlayManager] Error loading parquet {self.parquet_path}: {e}', flush=True)

    def _open_zarr(self):
        """Open masks_and_crops.zarr and optional dense_observations.zarr in read-only mode and preload arrays."""
        if not HAS_ZARR:
            return

        cam_keys = set(self._cam_frames.keys()) | {str(c) for c in DEFAULT_CAMERA_IDS}

        if self.zarr_path and self.zarr_path.exists():
            try:
                self._zarr_root = zarr.open_group(store=str(self.zarr_path), mode='r')
                for cam_str in cam_keys:
                    aff_path = f'affine_c2i/cam_{cam_str}'
                    if aff_path in self._zarr_root:
                        self._affines[cam_str] = np.asarray(self._zarr_root[aff_path], dtype=np.float32)
                    kp_mono_path = f'kp2d_mono/cam_{cam_str}'
                    if kp_mono_path in self._zarr_root:
                        self._mono_kps[cam_str] = np.asarray(self._zarr_root[kp_mono_path], dtype=np.float32)
                    conf_mono_path = f'kp2d_mono_conf/cam_{cam_str}'
                    if conf_mono_path in self._zarr_root:
                        self._mono_confs[cam_str] = np.asarray(self._zarr_root[conf_mono_path], dtype=np.float32)
            except Exception as e:
                print(f'[PipelineOverlayManager] Error opening Zarr {self.zarr_path}: {e}', flush=True)

        if self.dense_zarr_path and self.dense_zarr_path.exists():
            try:
                self._dense_zarr_root = zarr.open_group(store=str(self.dense_zarr_path), mode='r')
                for cam_str in cam_keys:
                    kp_path = f'kp2d_full/cam_{cam_str}'
                    if kp_path in self._dense_zarr_root:
                        self._dense_kps[cam_str] = np.asarray(self._dense_zarr_root[kp_path], dtype=np.float32)
                    conf_path = f'kp2d_conf/cam_{cam_str}'
                    if conf_path in self._dense_zarr_root:
                        self._dense_confs[cam_str] = np.asarray(self._dense_zarr_root[conf_path], dtype=np.float32)
            except Exception as e:
                print(f'[PipelineOverlayManager] Error opening dense landmarks Zarr: {e}', flush=True)

    def _load_telemetry_json(self):
        """Load optional Stage 7 2D reprojection polylines."""
        if not self.telemetry_dir or not self.telemetry_dir.exists():
            return

        try:
            for p in self.telemetry_dir.glob('reprojection_overlays_*.json'):
                cam_stem = p.stem[len('reprojection_overlays_'):]
                cam_str = self._normalize_cam_id(cam_stem)
                data = json.loads(p.read_text(encoding='utf-8'))
                for f_item in data.get('frames', []):
                    f_idx = int(f_item['frame_idx'])
                    polys = [
                        [tuple(float(c) for c in pt) for pt in poly]
                        for poly in f_item.get('bone_polylines_uv', [])
                    ]
                    self._reproj_polylines[(cam_str, f_idx)] = polys
        except Exception as e:
            print(f'[PipelineOverlayManager] Error loading telemetry overlays: {e}', flush=True)

    def get_overlay_data(self, cam_id: str, frame_idx: int) -> Optional[FrameOverlayData]:
        """Retrieve synchronized overlay metadata for a camera and frame index."""
        cam_str = self._normalize_cam_id(cam_id)
        key = (cam_str, frame_idx)
        if key in self._frame_data:
            entry = self._frame_data[key]
        else:
            # Check if nearest frame is close (within 2 frames)
            if cam_str in self._cam_frames:
                arr = self._cam_frames[cam_str]
                pos = np.searchsorted(arr, frame_idx)
                cand_idx = None
                if pos < len(arr) and abs(arr[pos] - frame_idx) <= 2:
                    cand_idx = int(arr[pos])
                elif pos > 0 and abs(arr[pos - 1] - frame_idx) <= 2:
                    cand_idx = int(arr[pos - 1])
                if cand_idx is not None:
                    entry = self._frame_data.get((cam_str, cand_idx))
                else:
                    return None
            else:
                return None

        if entry is None:
            return None

        # Return cached pose data immediately if already populated
        if entry.bone_polylines is not None or entry.keypoints_2d is not None:
            return entry

        # Attach 2D pose / wireframe polylines if available
        if key in self._reproj_polylines:
            entry.bone_polylines = self._reproj_polylines[key]
        elif cam_str in self._dense_kps and entry.v_idx is not None:
            dense_arr = self._dense_kps[cam_str]
            if entry.v_idx < len(dense_arr):
                entry.keypoints_2d = dense_arr[entry.v_idx]
                if cam_str in self._dense_confs:
                    entry.keypoint_confs = self._dense_confs[cam_str][entry.v_idx]
        elif cam_str in self._mono_kps and entry.v_idx is not None:
            mono_arr = self._mono_kps[cam_str]
            if entry.v_idx < len(mono_arr):
                entry.keypoints_2d = mono_arr[entry.v_idx]
                if cam_str in self._mono_confs:
                    entry.keypoint_confs = self._mono_confs[cam_str][entry.v_idx]
        elif self._dense_zarr_root is not None and entry.v_idx is not None:
            kp_path = f'kp2d_full/cam_{cam_str}'
            conf_path = f'kp2d_conf/cam_{cam_str}'
            if kp_path in self._dense_zarr_root:
                try:
                    self._dense_kps[cam_str] = np.asarray(self._dense_zarr_root[kp_path], dtype=np.float32)
                    entry.keypoints_2d = self._dense_kps[cam_str][entry.v_idx]
                    if conf_path in self._dense_zarr_root:
                        self._dense_confs[cam_str] = np.asarray(self._dense_zarr_root[conf_path], dtype=np.float32)
                        entry.keypoint_confs = self._dense_confs[cam_str][entry.v_idx]
                except Exception:
                    pass
        elif self._zarr_root is not None and entry.v_idx is not None:
            kp_mono_path = f'kp2d_mono/cam_{cam_str}'
            conf_mono_path = f'kp2d_mono_conf/cam_{cam_str}'
            if kp_mono_path in self._zarr_root:
                try:
                    self._mono_kps[cam_str] = np.asarray(self._zarr_root[kp_mono_path], dtype=np.float32)
                    entry.keypoints_2d = self._mono_kps[cam_str][entry.v_idx]
                    if conf_mono_path in self._zarr_root:
                        self._mono_confs[cam_str] = np.asarray(self._zarr_root[conf_mono_path], dtype=np.float32)
                        entry.keypoint_confs = self._mono_confs[cam_str][entry.v_idx]
                except Exception:
                    pass

        return entry

    def _load_mask_chunk(
        self,
        cam_str: str,
        v_idx: int,
    ) -> Optional[Tuple[QtGui.QImage, float, float, float]]:
        """Synchronously load a 16-frame chunk from Zarr and populate the LRU mask cache."""
        if self._zarr_root is None or v_idx is None:
            return None

        mask_ds_name = f'fg_masks/cam_{cam_str}'
        if mask_ds_name not in self._zarr_root:
            return None

        try:
            mask_ds = self._zarr_root[mask_ds_name]
            total_v = mask_ds.shape[0]
            if v_idx < 0 or v_idx >= total_v:
                return None

            # Get or load affine transforms
            if cam_str in self._affines:
                aff_arr = self._affines[cam_str]
            else:
                affine_ds_name = f'affine_c2i/cam_{cam_str}'
                if affine_ds_name not in self._zarr_root:
                    return None
                aff_arr = np.asarray(self._zarr_root[affine_ds_name], dtype=np.float32)
                self._affines[cam_str] = aff_arr

            # Slice 16 frames aligned to chunk boundary
            chunk_start = (v_idx // 16) * 16
            chunk_end = min(total_v, chunk_start + 16)
            chunk_data = np.asarray(mask_ds[chunk_start:chunk_end], dtype=np.uint8)

            result: Optional[Tuple[QtGui.QImage, float, float, float]] = None

            with self._cache_lock:
                for idx in range(chunk_start, chunk_end):
                    target_key = (cam_str, idx)
                    if target_key in self._mask_qimage_cache:
                        continue

                    m = np.ascontiguousarray(chunk_data[idx - chunk_start])
                    h, w = m.shape

                    # Zero-copy buffer wrap into indexed 8-bit QImage, then copy
                    qimg = QtGui.QImage(m.data, w, h, w, QtGui.QImage.Format.Format_Indexed8)
                    qimg.setColorTable(self._mask_color_table)
                    qimg_copy = qimg.copy()

                    affine = aff_arr[idx] if idx < len(aff_arr) else aff_arr[0]
                    scale = float(affine[0, 0])
                    xmin = float(affine[0, 2])
                    ymin = float(affine[1, 2])
                    side = scale * float(w)

                    entry = (qimg_copy, xmin, ymin, side)
                    self._mask_qimage_cache[target_key] = entry
                    if idx == v_idx:
                        result = entry

                # Maintain LRU capacity limit
                while len(self._mask_qimage_cache) > self._max_mask_cache_size:
                    self._mask_qimage_cache.popitem(last=False)

            return result or self._mask_qimage_cache.get((cam_str, v_idx))
        except Exception:
            return None

    def _trigger_prefetch(self, cam_str: str, target_v_idx: int):
        """Enqueue next chunk for background prefetching without blocking caller."""
        if not self._prefetch_running:
            return
        try:
            self._prefetch_queue.put_nowait((cam_str, target_v_idx))
        except queue.Full:
            pass

    def _run_prefetch_worker(self):
        """Background thread pre-loading future mask chunks into LRU cache."""
        while self._prefetch_running:
            try:
                item = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None or not self._prefetch_running:
                break

            cam_str, target_v_idx = item
            with self._cache_lock:
                already_cached = (cam_str, target_v_idx) in self._mask_qimage_cache
            if not already_cached:
                try:
                    self._load_mask_chunk(cam_str, target_v_idx)
                except Exception:
                    pass

    def get_mask_and_bounds(
        self,
        cam_id: str,
        v_idx: int,
    ) -> Optional[Tuple[QtGui.QImage, float, float, float]]:
        """Retrieve pre-built transparent QImage mask and bounding geometry for canvas rendering.

        Returns:
            Tuple of (mask_qimage, xmin, ymin, side) in full-frame image pixel space,
            or None if mask cannot be retrieved.
        """
        if self._zarr_root is None or v_idx is None:
            return None

        cam_str = self._normalize_cam_id(cam_id)
        cache_key = (cam_str, v_idx)

        with self._cache_lock:
            if cache_key in self._mask_qimage_cache:
                self._mask_qimage_cache.move_to_end(cache_key)
                res = self._mask_qimage_cache[cache_key]
                # Prefetch ahead
                self._trigger_prefetch(cam_str, v_idx + 16)
                return res

        # Cache miss: load chunk synchronously
        res = self._load_mask_chunk(cam_str, v_idx)
        self._trigger_prefetch(cam_str, v_idx + 16)
        return res

    def stop(self):
        """Stop background prefetch worker cleanly."""
        self._prefetch_running = False
        try:
            self._prefetch_queue.put_nowait(None)
        except Exception:
            pass

