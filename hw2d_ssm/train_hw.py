"""
Train the *same* LinearSSM (from `lorenz_ssm.models.linear_ssm`) on the
HW2D energy/enstrophy time series.

Why reuse it: the whole point of Stage 2 is to test whether a linear,
time-invariant SSM that worked for Lorenz also handles plasma turbulence.
Don't change the model — only the data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lorenz_ssm.models.linear_ssm import LinearSSM
from lorenz_ssm.utils import pick_device
from hw2d_ssm.data.extract import HWTimeSeries, make_windows, train_val_window_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class HWDatasetBundle:
    train: torch.Tensor          # (N_train, window, C)
    val: torch.Tensor            # (N_val, window, C)
    mean: np.ndarray             # (C,)
    std: np.ndarray              # (C,)
    times: np.ndarray            # (T,) global timeline
    frame_dt: float
    channel_names: list[str]


def build_hw_dataset(ts: HWTimeSeries,
                     window: int = 256,
                     stride: int = 64,
                     train_frac: float = 0.8,
                     device: torch.device | None = None) -> HWDatasetBundle:
    """Normalise per-channel using *train-window* statistics, then move
    to device.

    Important: stats are computed *before* the train/val split on the
    timeline so both halves are normalised consistently, but we still
    keep train and val temporally separated.
    """
    device = device or pick_device(verbose=False)
    windows = make_windows(ts.series, window=window, stride=stride)
    train_w, val_w = train_val_window_split(windows, train_frac=train_frac)

    flat = train_w.reshape(-1, train_w.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-6
    train_n = (train_w - mean) / std
    val_n = (val_w - mean) / std

    return HWDatasetBundle(
        train=torch.from_numpy(train_n.astype(np.float32)).to(device),
        val=torch.from_numpy(val_n.astype(np.float32)).to(device),
        mean=mean.squeeze(),
        std=std.squeeze(),
        times=ts.times,
        frame_dt=ts.frame_dt,
        channel_names=ts.channel_names,
    )


# ---------- training loop (mirrors Stage 1 so the *only* difference is data) -----

def train_step(model, batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    preds, _ = model.forward_sequence(batch)
    loss = criterion(preds, batch[:, 1:])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model, batch, criterion):
    model.eval()
    preds, _ = model.forward_sequence(batch)
    return float(criterion(preds, batch[:, 1:]).item())


def train_hw(model: LinearSSM, ds: HWDatasetBundle,
             epochs: int = 600, lr: float = 1e-3,
             batch_size: int | None = None, log_every: int = 50):
    """Mini-batched training over the window set."""
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = ds.train.shape[0]
    bs = batch_size or n
    train_hist, val_hist = [], []

    for ep in range(1, epochs + 1):
        # shuffle window indices each epoch
        perm = torch.randperm(n, device=ds.train.device)
        losses = []
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            losses.append(train_step(model, ds.train[idx], optim, criterion))
        tr = float(np.mean(losses))
        train_hist.append(tr)
        if ep % log_every == 0 or ep == 1:
            vl = eval_loss(model, ds.val, criterion)
            val_hist.append(vl)
            print(f"  epoch {ep:4d}   train MSE {tr:.6f}   val MSE {vl:.6f}")
    return train_hist, val_hist


# ---------- plots ------------------------------------------------------

def plot_hw_prediction(times, true, pred, channel_names, path, title):
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(10, 5), sharex=True)
    if len(channel_names) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(times, true[:, i], label="true", lw=1.2)
        ax.plot(times, pred[:, i], label="pred", lw=1.0, ls="--")
        ax.set_ylabel(channel_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_eigenvalues(A_lorenz: np.ndarray, A_hw: np.ndarray,
                        dt_lorenz: float, dt_hw: float,
                        path: Path | None = None):
    """Eigenvalues of two A matrices side-by-side for direct comparison."""
    eig_l = np.linalg.eigvals(A_lorenz)
    eig_h = np.linalg.eigvals(A_hw)
    theta = np.linspace(0, 2 * np.pi, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, eig, name, dt in [(axes[0], eig_l, "Lorenz", dt_lorenz),
                              (axes[1], eig_h, "HW2D", dt_hw)]:
        ax.scatter(eig.real, eig.imag, c=np.abs(eig), cmap="viridis",
                   s=50, edgecolors="k", zorder=3)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, label="|λ|=1")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name}  (dt={dt})\nmax|λ|={abs(eig).max():.3f}, "
                     f"# complex pairs = {(abs(eig.imag) > 1e-6).sum() // 2}")
        ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
        ax.legend(loc="upper right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
