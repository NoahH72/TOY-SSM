#!/usr/bin/env python3
"""
Train a linear SSM on a Lorenz-63 dataset (multiple trajectories) for
one-step-ahead prediction.

This module is *also* imported by the Jupyter notebook — the public
functions (`build_dataset`, `train_loop`, `rollout`, `plot_*`) are reused
there.  Running it as a script reproduces a full end-to-end pass:

    python -m lorenz_ssm.train
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Allow running as script or module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lorenz_ssm.data.lorenz import (
    LorenzConfig,
    normalize,
    simulate_many_lorenz,
    train_val_split,
)
from lorenz_ssm.models.linear_ssm import LinearSSM
from lorenz_ssm.utils import pick_device

OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------- data --------------------------------------------------------

@dataclass
class DatasetBundle:
    """Everything `train_loop` needs as input."""
    train: torch.Tensor          # (N_train, T, 3)  normalised
    val: torch.Tensor            # (N_val,   T, 3)  normalised
    mean: np.ndarray             # (3,)  for denormalising plots
    std: np.ndarray              # (3,)
    times: np.ndarray            # (T,) shared timeline


def build_dataset(
    n_trajectories: int = 8,
    train_frac: float = 0.75,
    cfg: LorenzConfig | None = None,
    device: torch.device | None = None,
    seed: int = 0,
) -> DatasetBundle:
    """Simulate, split (across trajectories), normalise, move to device."""
    cfg = cfg or LorenzConfig(t1=40.0, dt=0.02, burn_in=5.0)
    times, data = simulate_many_lorenz(n_trajectories=n_trajectories, cfg=cfg, seed=seed)
    n_train = max(1, int(n_trajectories * train_frac))
    train_raw, val_raw = data[:n_train], data[n_train:]
    train_n, val_n, mean, std = normalize(train_raw, val_raw)
    device = device or pick_device(verbose=False)
    return DatasetBundle(
        train=torch.from_numpy(train_n).to(device),
        val=torch.from_numpy(val_n).to(device),
        mean=mean,
        std=std,
        times=times,
    )


# ---------- training ---------------------------------------------------

def train_step(
    model: LinearSSM,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """One optimiser step on a batch (N, T, 3)."""
    model.train()
    optimizer.zero_grad()
    preds, _ = model.forward_sequence(batch)
    targets = batch[:, 1:]          # u_{t+1}
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model: LinearSSM, batch: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    preds, _ = model.forward_sequence(batch)
    return float(criterion(preds, batch[:, 1:]).item())


def train_loop(
    model: LinearSSM,
    ds: DatasetBundle,
    epochs: int = 400,
    lr: float = 1e-3,
    log_every: int = 50,
) -> tuple[list[float], list[float]]:
    """Full-batch training over the trajectory dataset."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_hist, val_hist = [], []
    for ep in range(1, epochs + 1):
        tr = train_step(model, ds.train, optimizer, criterion)
        train_hist.append(tr)
        if ep % log_every == 0 or ep == 1:
            vl = eval_loss(model, ds.val, criterion)
            val_hist.append(vl)
            print(f"  epoch {ep:4d}   train MSE {tr:.6f}   val MSE {vl:.6f}")
    return train_hist, val_hist


# ---------- inference helpers -----------------------------------------

@torch.no_grad()
def rollout(model: LinearSSM, u0: torch.Tensor, steps: int) -> np.ndarray:
    """Autoregressive rollout: feed each prediction back in as the next input.

    u0: (batch, input_dim).  Returns (steps+1, input_dim) for batch=1.
    """
    model.eval()
    h = torch.zeros(u0.shape[0], model.state_dim, device=u0.device, dtype=u0.dtype)
    u = u0.clone()
    out = [u0.squeeze(0).cpu().numpy()]
    for _ in range(steps):
        y, h = model.one_step(u, h)
        out.append(y.squeeze(0).cpu().numpy())
        u = y
    return np.stack(out, axis=0)


# ---------- plots ------------------------------------------------------

def plot_prediction(times, true, pred, path, title):
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(times, true[:, i], label="true", lw=1.2)
        ax.plot(times, pred[:, i], label="pred", lw=1.0, ls="--")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eigenvalues(A: np.ndarray, path: Path | None = None, dt: float | None = None):
    """Plot eigenvalues of the discrete-time A in the complex plane,
    plus a |λ|-bar chart for stability."""
    eig = np.linalg.eigvals(A)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.scatter(eig.real, eig.imag, c=np.abs(eig), cmap="viridis", s=50,
                edgecolors="k", zorder=3)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, label="unit circle |λ|=1")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_xlabel("Re(λ)")
    ax1.set_ylabel("Im(λ)")
    ax1.set_title("Eigenvalues of learned A (discrete time)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.bar(np.arange(len(eig)), np.abs(eig))
    ax2.axhline(1.0, color="red", ls="--", label="|λ|=1 (neutral)")
    ax2.set_xlabel("mode index")
    ax2.set_ylabel("|λ|")
    title2 = "Magnitudes (|λ|<1 ⇒ decaying mode)"
    if dt is not None:
        # Optional: show approximate continuous-time decay rate
        with np.errstate(divide="ignore"):
            tau = -dt / np.log(np.clip(np.abs(eig), 1e-12, None))
        title2 += f"\n(slowest decay τ ≈ {np.nanmax(tau):.2f} time units)"
    ax2.set_title(title2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# ---------- script entry ----------------------------------------------

def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()

    print("Building Lorenz dataset (8 trajectories, slight IC jitter)...")
    ds = build_dataset(n_trajectories=8, train_frac=0.75, device=device)
    print(f"  train batch: {tuple(ds.train.shape)}  val batch: {tuple(ds.val.shape)}")

    model = LinearSSM(input_dim=3, output_dim=3, state_dim=16).to(device)
    print(f"Training linear SSM (state_dim=16)…")
    train_loop(model, ds, epochs=400, lr=1e-3, log_every=50)

    # Teacher-forced 1-step prediction on val[0]
    val0 = ds.val[0:1]                                # (1, T, 3)
    with torch.no_grad():
        preds, _ = model.forward_sequence(val0)
    preds_np = preds.squeeze(0).cpu().numpy()
    true_np = val0.squeeze(0).cpu().numpy()[1:]
    plot_prediction(ds.times[1:], true_np, preds_np,
                    OUT_DIR / "prediction_teacher_forced.png",
                    "Validation (teacher-forced): predict u_{t+1} given u_t")

    # Autoregressive rollout
    n_roll = min(500, val0.shape[1] - 1)
    u0 = val0[:, 0]
    roll = rollout(model, u0, n_roll)
    plot_prediction(ds.times[: n_roll + 1], true_np[: n_roll + 1], roll,
                    OUT_DIR / "prediction_autoregressive.png",
                    f"Autoregressive rollout ({n_roll} steps)")

    A_np = model.A.detach().cpu().numpy()
    plot_eigenvalues(A_np, OUT_DIR / "eigenvalues.png", dt=0.02)
    print(f"\nSaved plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
