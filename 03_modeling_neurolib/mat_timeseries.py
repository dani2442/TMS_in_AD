"""Helpers to load per-subject fMRI timeseries from MATLAB .mat files.

This repo mostly works with in-repo BIDS/XCP-D timeseries via `petTOAD_setup.py`.
These helpers exist so you can reuse the same modeling/evaluation code with external
MATLAB exports like `ts_young_TR0.72.mat`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import scipy.io


def load_mat(path: str) -> Dict[str, Any]:
    """Load a MATLAB .mat file into a python dict."""

    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)


def infer_n_nodes(mat: Dict[str, Any], fc_mean_var: str = "FC_mean", fc_all_var: str = "FC_all") -> Optional[int]:
    """Infer number of ROIs from FC variables if present."""

    fc_mean = mat.get(fc_mean_var)
    if fc_mean is not None:
        fc_mean_arr = np.asarray(fc_mean)
        if fc_mean_arr.ndim == 2 and fc_mean_arr.shape[0] == fc_mean_arr.shape[1]:
            return int(fc_mean_arr.shape[0])

    fc_all = mat.get(fc_all_var)
    if fc_all is not None:
        fc_all_arr = np.asarray(fc_all)
        if fc_all_arr.ndim == 3 and fc_all_arr.shape[0] == fc_all_arr.shape[1]:
            return int(fc_all_arr.shape[0])
        if fc_all_arr.ndim == 3 and fc_all_arr.shape[1] == fc_all_arr.shape[2]:
            return int(fc_all_arr.shape[1])

    return None


def _as_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {arr.shape}")
    return arr.astype(float, copy=False)


def extract_subject_timeseries(
    mat: Dict[str, Any],
    timeseries_var: str,
    subject_index: int,
    *,
    n_nodes: Optional[int] = None,
) -> np.ndarray:
    """Extract a single subject timeseries as a 2D array shaped (N, T).

    Supported layouts:
    - MATLAB cell array -> list-like / object array (one cell per subject)
    - 3D numeric array (S,N,T) or (N,T,S) or (T,N,S)
    - 2D numeric array (single subject)
    """

    obj = mat.get(timeseries_var)
    if obj is None:
        raise KeyError(f"Variable '{timeseries_var}' not found in .mat")

    # List-like (already converted from cell)
    if isinstance(obj, (list, tuple)):
        ts = _as_2d(obj[subject_index])
        return _ensure_n_t(ts, n_nodes)

    arr = np.asarray(obj)

    # Object array cell-like
    if arr.dtype == object:
        flat = arr.ravel().tolist()
        ts = _as_2d(flat[subject_index])
        return _ensure_n_t(ts, n_nodes)

    # Numeric array
    if arr.ndim == 2:
        if subject_index != 0:
            raise ValueError(
                f"'{timeseries_var}' is 2D (single subject). Use --subject-index 0; got {subject_index}."
            )
        ts = _as_2d(arr)
        return _ensure_n_t(ts, n_nodes)

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported '{timeseries_var}' shape {arr.shape} (expected 2D, 3D, or cell array)."
        )

    ts = _resolve_3d(arr, subject_index, n_nodes)
    return _ensure_n_t(ts, n_nodes)


def _resolve_3d(arr: np.ndarray, subject_index: int, n_nodes: Optional[int]) -> np.ndarray:
    # Prefer node-count based selection when available.
    if n_nodes is not None:
        # (S, N, T)
        if arr.shape[1] == n_nodes:
            return _as_2d(arr[subject_index])
        # (N, T, S)
        if arr.shape[0] == n_nodes:
            return _as_2d(arr[:, :, subject_index])
        # (T, N, S)
        if arr.shape[1] == n_nodes:
            return _as_2d(arr[:, :, subject_index]).T

    # Fallback: assume subject-first.
    return _as_2d(arr[subject_index])


def _ensure_n_t(ts: np.ndarray, n_nodes: Optional[int]) -> np.ndarray:
    # Make sure shape is (N, T)
    if n_nodes is not None:
        if ts.shape[0] == n_nodes:
            return ts
        if ts.shape[1] == n_nodes:
            return ts.T

    # If N is unknown, assume timepoints > nodes.
    if ts.shape[0] > ts.shape[1]:
        return ts.T
    return ts
