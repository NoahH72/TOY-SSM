"""
Read the hw2d HDF5 output and extract scalar time series for SSM training.

`load_energy_enstrophy` returns the two spatially-averaged scalars that
hw2d already computes per frame:

  energy   — total fluctuation energy E = ½⟨n² + |∇φ|²⟩
  enstrophy — Ω = ½⟨ω²⟩  (vorticity intensity)

We also expose a sliding-window helper so the long single trajectory can
be sliced into many short windows for SSM training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class HWTimeSeries:
    """All the bits the training pipeline needs in one place."""
    times: np.ndarray            # (T,)  physical time
    series: np.ndarray           # (T, n_channels)  e.g. [energy, enstrophy]
    channel_names: list[str]
    attrs: dict                  # simulation parameters

    @property
    def frame_dt(self) -> float:
        return float(self.attrs.get("frame_dt", self.attrs.get("dt", 1.0)))


def load_energy_enstrophy(path: str | Path,
                          drop_initial: int = 200) -> HWTimeSeries:
    """Open the hw2d HDF5 file and pull the two scalar series we'll predict.

    drop_initial: number of leading frames to discard (hw2d takes a while
    to saturate after initial conditions — the first few hundred frames
    are transient build-up, not the turbulent regime).
    """
    path = Path(path)
    with h5py.File(path, "r") as hf:
        energy = np.asarray(hf["energy"][:], dtype=np.float32).ravel()
        enstrophy = np.asarray(hf["enstrophy"][:], dtype=np.float32).ravel()
        attrs = {k: hf.attrs[k] for k in hf.attrs.keys()}

    T = len(energy)
    drop = min(drop_initial, T // 4)
    energy = energy[drop:]
    enstrophy = enstrophy[drop:]
    frame_dt = float(attrs.get("frame_dt", attrs.get("dt", 1.0)))
    times = np.arange(len(energy)) * frame_dt

    series = np.stack([energy, enstrophy], axis=1)
    return HWTimeSeries(times=times,
                        series=series,
                        channel_names=["energy", "enstrophy"],
                        attrs=attrs)


def make_windows(series: np.ndarray,
                 window: int = 256,
                 stride: int = 64) -> np.ndarray:
    """Slide a window of length `window` along axis 0 with given stride.

    Input  : (T, C)
    Output : (N, window, C)
    """
    T, C = series.shape
    if window > T:
        raise ValueError(f"window {window} > series length {T}")
    n = 1 + (T - window) // stride
    out = np.empty((n, window, C), dtype=series.dtype)
    for i in range(n):
        start = i * stride
        out[i] = series[start:start + window]
    return out


def train_val_window_split(windows: np.ndarray, train_frac: float = 0.8
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Split *along the window axis* with no shuffle (preserves time order)."""
    n_train = max(1, int(windows.shape[0] * train_frac))
    return windows[:n_train], windows[n_train:]
