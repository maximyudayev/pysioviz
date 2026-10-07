"""
Paged HDF5 Sensor Reader implementing lazy-loading memory management.
Reads sensor measurements in temporal pages (e.g., 10-minute blocks) instead of loading
multi-gigabyte datasets entirely into RAM.
"""

from typing import Dict, List, Tuple
import numpy as np
import h5py


class PagedSensorReader:
    """Lazy-loading sensor stream reader fetching temporal blocks on demand."""

    def __init__(
        self,
        hdf5_path: str,
        channel_name: str,
        value_dataset_path: str,
        toa_dataset_path: str,
        page_duration_s: float = 600.0,  # 10 minutes default page
        margin_s: float = 60.0,
    ):
        self.hdf5_path = hdf5_path
        self.channel_name = channel_name
        self.value_dataset_path = value_dataset_path
        self.toa_dataset_path = toa_dataset_path
        self.page_duration_s = float(page_duration_s)
        self.margin_s = float(margin_s)

        # Dataset metadata (retrieved lazily without loading full arrays)
        self.total_samples: int = 0
        self.min_toa_s: float = 0.0
        self.max_toa_s: float = 0.0
        self.shape: Tuple[int, ...] = (0,)
        self.num_channels: int = 1

        # Current page in memory
        self._page_start_toa: float = -1.0
        self._page_end_toa: float = -1.0
        self._page_toas: np.ndarray = np.empty(0, dtype=np.float64)
        self._page_values: np.ndarray = np.empty((0, 1), dtype=np.float32)

        self._inspect_dataset()

    def _inspect_dataset(self):
        """Inspect dataset metadata without loading data into memory."""
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if self.toa_dataset_path not in f or self.value_dataset_path not in f:
                    raise KeyError(f'Dataset path not found in {self.hdf5_path}')

                toa_ds = f[self.toa_dataset_path]
                val_ds = f[self.value_dataset_path]

                self.total_samples = toa_ds.shape[0]
                self.shape = val_ds.shape
                if len(val_ds.shape) > 1:
                    self.num_channels = val_ds.shape[1]
                else:
                    self.num_channels = 1

                if self.total_samples > 0:
                    self.min_toa_s = float(toa_ds[0].item())
                    self.max_toa_s = float(toa_ds[-1].item())
                else:
                    self.min_toa_s = 0.0
                    self.max_toa_s = 0.0
        except Exception as e:
            print(f'Error inspecting HDF5 {self.hdf5_path}: {e}', flush=True)

    def _ensure_page(self, target_toa_s: float):
        """Check if target timestamp is within current page + margin; if not, reload page."""
        if self._page_toas.size > 0 and (self._page_start_toa + self.margin_s) <= target_toa_s <= (
            self._page_end_toa - self.margin_s
        ):
            return  # Target is safely within cached page

        # Calculate new page range centered around target
        page_start = max(self.min_toa_s, target_toa_s - (self.page_duration_s / 2.0))
        page_end = min(self.max_toa_s, page_start + self.page_duration_s)

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                toa_ds = f[self.toa_dataset_path]
                val_ds = f[self.value_dataset_path]

                # Binary search for index range in HDF5 without reading whole dataset
                start_idx = self._find_index_for_toa(toa_ds, page_start)
                end_idx = self._find_index_for_toa(toa_ds, page_end)
                end_idx = min(self.total_samples, end_idx + 1)

                if start_idx < end_idx:
                    self._page_toas = np.asarray(toa_ds[start_idx:end_idx]).ravel().astype(np.float64)
                    vals = np.asarray(val_ds[start_idx:end_idx])
                    if vals.ndim == 1:
                        self._page_values = vals[:, np.newaxis].astype(np.float32)
                    else:
                        self._page_values = vals.astype(np.float32)

                    self._page_start_toa = page_start
                    self._page_end_toa = page_end
        except Exception as e:
            print(f'Error loading sensor page: {e}', flush=True)

    def _find_index_for_toa(self, dataset: h5py.Dataset, target_toa: float) -> int:
        """Perform binary search directly on disk-backed HDF5 dataset to find slice bounds."""
        low = 0
        high = dataset.shape[0] - 1
        while low <= high:
            mid = (low + high) // 2
            mid_val = float(dataset[mid].item())
            if mid_val < target_toa:
                low = mid + 1
            elif mid_val > target_toa:
                high = mid - 1
            else:
                return mid
        return min(max(0, low), dataset.shape[0] - 1)

    def get_data_window(self, center_toa_s: float, window_duration_s: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch relative time window `[center - half, center + half]` from current page.

        Returns:
            Tuple[toas, values]: 1D timestamps and (N, channels) values.
        """
        self._ensure_page(center_toa_s)

        if self._page_toas.size == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, self.num_channels), dtype=np.float32)

        half = window_duration_s / 2.0
        t_min = center_toa_s - half
        t_max = center_toa_s + half

        mask = (self._page_toas >= t_min) & (self._page_toas <= t_max)
        return self._page_toas[mask], self._page_values[mask]

    @staticmethod
    def auto_discover_streams(hdf5_path: str) -> List[Dict[str, str]]:
        """Introspect an HDF5 file and discover potential time-series sensor streams."""
        streams = []
        try:
            with h5py.File(hdf5_path, 'r') as f:

                def visitor(name, obj):
                    if isinstance(obj, h5py.Group):
                        keys = list(obj.keys())
                        # Look for timestamp indicator
                        toa_key = None
                        for candidate in ['toa_s', 'process_time_s', 'timestamp', 'timestamp_s']:
                            if candidate in keys and isinstance(obj[candidate], h5py.Dataset):
                                toa_key = candidate
                                break

                        if toa_key is not None:
                            # Look for value candidates
                            for k in keys:
                                if k != toa_key and isinstance(obj[k], h5py.Dataset):
                                    ds = obj[k]
                                    if ds.ndim in (1, 2) and ds.dtype.kind in ('f', 'i', 'u'):
                                        streams.append(
                                            {
                                                'channel_name': f'{name}/{k}',
                                                'group': name,
                                                'value_dataset': f'{name}/{k}',
                                                'toa_dataset': f'{name}/{toa_key}',
                                                'shape': str(ds.shape),
                                            }
                                        )

                f.visititems(visitor)
        except Exception as e:
            print(f'Error auto-discovering streams in {hdf5_path}: {e}', flush=True)

        return streams
