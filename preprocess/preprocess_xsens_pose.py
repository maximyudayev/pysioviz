#!/usr/bin/env python3
"""Deduplicate MVN Xsens MoCap pose datasets by counter index.

This script opens an MVN Xsens HDF5 recording, extracts the `position` and `counter`
datasets from `xsens-pose`, filters the position and counter arrays by the last
occurrence of each unique counter value, and writes the updated arrays together
with all untouched datasets from `xsens-pose` into a new group named `xsens_pose`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def find_xsens_pose_group(h5_file: h5py.File, candidate_path: Optional[str] = None) -> str:
    """Locate the xsens-pose group within the HDF5 file.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file object.
    candidate_path : Optional[str]
        Explicit group path provided by the caller, if any.

    Returns
    -------
    str
        Path of the discovered xsens-pose group.
    """
    if candidate_path and candidate_path in h5_file:
        return candidate_path

    # Standard candidate locations
    common_candidates = [
        'mvn-analyze/xsens-pose',
        'xsens-pose',
        '/mvn-analyze/xsens-pose',
        '/xsens-pose',
    ]
    for cand in common_candidates:
        if cand in h5_file and isinstance(h5_file[cand], h5py.Group):
            return cand.lstrip('/')

    # Fallback: scan all groups for names ending with 'xsens-pose'
    matching_groups = []

    def _visitor(name, obj):
        if isinstance(obj, h5py.Group) and (name == 'xsens-pose' or name.endswith('/xsens-pose')):
            matching_groups.append(name)

    h5_file.visititems(_visitor)
    if matching_groups:
        return matching_groups[0]

    raise KeyError(
        "Could not find 'xsens-pose' group in the HDF5 file. Available root groups: " + str(list(h5_file.keys()))
    )


def preprocess_xsens_pose(
    hdf5_path: str | Path,
    input_group_path: Optional[str] = None,
    output_group_path: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """Deduplicate MVN Xsens MoCap pose data based on unique counter indices.

    Parameters
    ----------
    hdf5_path : str | Path
        Path to the HDF5 file to process.
    input_group_path : Optional[str]
        Group path containing original xsens-pose data. Defaults to auto-detection.
    output_group_path : Optional[str]
        Group path to write the processed data. Defaults to input path with
        'xsens-pose' replaced by 'xsens_pose'.
    overwrite : bool
        Whether to overwrite the output group if it already exists.
    dry_run : bool
        If True, calculate and print statistics without writing to disk.

    Returns
    -------
    dict[str, int | str]
        Execution summary with original and deduplicated sample counts.
    """
    path = Path(hdf5_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'HDF5 file not found: {path}')

    mode = 'r' if dry_run else 'r+'
    with h5py.File(path, mode) as h5_file:
        src_path = find_xsens_pose_group(h5_file, input_group_path)
        src_group = h5_file[src_path]

        # Determine target group path
        if output_group_path:
            dst_path = output_group_path.lstrip('/')
        else:
            dst_path = src_path.replace('xsens-pose', 'xsens_pose')

        if dst_path == src_path:
            raise ValueError(f'Output group path ({dst_path}) cannot be identical to input group path ({src_path}).')

        print(f'Processing: {path.name}')
        print(f'  Input group:  /{src_path}')
        print(f'  Output group: /{dst_path}')

        # Validate required datasets
        if 'position' not in src_group:
            raise KeyError(f"'position' dataset missing in /{src_path}")
        if 'counter' not in src_group:
            raise KeyError(f"'counter' dataset missing in /{src_path}")
        if 'toa_s' not in src_group and 'process_time_s' not in src_group:
            raise KeyError(f"'counter' dataset missing in /{src_path}")

        pos_ds = src_group['position']
        cnt_ds = src_group['counter']
        toa_ds = src_group['toa_s'] if 'toa_s' in src_group else src_group['process_time_s']

        n_pos = pos_ds.shape[0]
        n_cnt = cnt_ds.shape[0]

        if n_pos != n_cnt:
            raise ValueError(f"Mismatch in sample lengths: 'position' has {n_pos} samples but 'counter' has {n_cnt}.")

        print(f'  Original sample count: {n_pos}')

        # Read counter and compute last occurrence of each unique counter
        counter_raw = cnt_ds[:]
        counter_flat = np.asarray(counter_raw).ravel()
        toa_raw = toa_ds[:]

        # Find first occurrence from the reverse side to obtain the last occurrence indices
        _, rev_first_idx = np.unique(counter_flat[::-1], return_index=True)
        # Convert reversed indices back to original indexing and preserve chronological order
        last_occ_indices = np.sort(len(counter_flat) - 1 - rev_first_idx)

        n_unique = len(last_occ_indices)
        n_dropped = n_pos - n_unique
        print(f'  Unique counter samples: {n_unique} (dropped {n_dropped} duplicate samples)')

        if dry_run:
            print('  [Dry-run] No changes written to file.')
            return {
                'file': str(path),
                'input_group': src_path,
                'output_group': dst_path,
                'original_count': n_pos,
                'deduplicated_count': n_unique,
                'dropped_count': n_dropped,
            }

        # Handle existing destination group
        if dst_path in h5_file:
            if not overwrite:
                raise FileExistsError(
                    f'Destination group /{dst_path} already exists. '
                    'Specify overwrite=True (or pass --overwrite flag) to replace it.'
                )
            print(f'  Overwriting existing group /{dst_path}...')
            del h5_file[dst_path]

        # Ensure parent path exists
        parent_group_path = str(Path(dst_path).parent).replace('\\', '/')
        if parent_group_path not in ('.', '', '/'):
            parent_group = h5_file.require_group(parent_group_path)
            dst_group_name = Path(dst_path).name
            dst_group = parent_group.create_group(dst_group_name)
        else:
            dst_group = h5_file.create_group(dst_path)

        # Copy group-level attributes
        for attr_k, attr_v in src_group.attrs.items():
            dst_group.attrs[attr_k] = attr_v

        # 1. Slice and write position array
        print('  Slicing and writing position array...')
        sliced_pos = pos_ds[:][last_occ_indices]
        _write_dataset_like(
            dst_group=dst_group,
            name='position',
            data=sliced_pos,
            reference_ds=pos_ds,
        )

        # 2. Slice and write counter array
        print('  Slicing and writing counter array...')
        if counter_raw.ndim == 2:
            sliced_cnt = counter_raw[last_occ_indices, :]
        else:
            sliced_cnt = counter_raw[last_occ_indices]
        _write_dataset_like(
            dst_group=dst_group,
            name='counter',
            data=sliced_cnt,
            reference_ds=cnt_ds,
        )

        # 3. Slice and write process_time_s
        print('  Slicing and writing toa array...')
        if toa_raw.ndim == 2:
            sliced_toa = toa_raw[last_occ_indices, :]
        else:
            sliced_toa = toa_raw[last_occ_indices]
        _write_dataset_like(
            dst_group=dst_group,
            name='toa_s',
            data=sliced_toa,
            reference_ds=toa_ds,
        )

        # 4. Copy all untouched datasets / items
        untouched_keys = [k for k in src_group.keys() if k not in ('position', 'counter', 'process_time_s', 'toa_s')]
        for k in untouched_keys:
            print(f'  Copying untouched dataset: {k}...')
            src_group.copy(k, dst_group, name=k)

        print(f'Done! Successfully created /{dst_path} with {n_unique} deduplicated samples.')
        return {
            'file': str(path),
            'input_group': src_path,
            'output_group': dst_path,
            'original_count': n_pos,
            'deduplicated_count': n_unique,
            'dropped_count': n_dropped,
        }


def _write_dataset_like(
    dst_group: h5py.Group,
    name: str,
    data: np.ndarray,
    reference_ds: h5py.Dataset,
) -> h5py.Dataset:
    """Create a dataset in dst_group using metadata and chunking from reference_ds."""
    chunks = reference_ds.chunks
    if chunks is not None:
        # Prevent chunk dimension from exceeding total length
        first_dim = min(chunks[0], len(data))
        chunks = (first_dim, *chunks[1:])

    ds = dst_group.create_dataset(
        name=name,
        data=data,
        dtype=data.dtype,
        chunks=chunks,
        compression=reference_ds.compression,
        compression_opts=reference_ds.compression_opts,
    )

    # Copy dataset-level attributes
    for attr_k, attr_v in reference_ds.attrs.items():
        ds.attrs[attr_k] = attr_v

    return ds


def main() -> None:
    """CLI entrypoint for Xsens MoCap pose preprocessing."""
    parser = argparse.ArgumentParser(
        description='Deduplicate MVN Xsens MoCap position data using unique counter indices.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'hdf5_path',
        type=str,
        help='Path to the MVN Xsens HDF5 recording file.',
    )
    parser.add_argument(
        '--input-group',
        '-i',
        type=str,
        default=None,
        help="Input group path (defaults to auto-detecting 'xsens-pose').",
    )
    parser.add_argument(
        '--output-group',
        '-o',
        type=str,
        default=None,
        help="Output group path (defaults to input group with 'xsens-pose' replaced by 'xsens_pose').",
    )
    parser.add_argument(
        '--overwrite',
        '-f',
        action='store_true',
        help='Overwrite destination group if it already exists.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Inspect and print sample counts without modifying the HDF5 file.',
    )

    args = parser.parse_args()

    try:
        preprocess_xsens_pose(
            hdf5_path=args.hdf5_path,
            input_group_path=args.input_group,
            output_group_path=args.output_group,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
