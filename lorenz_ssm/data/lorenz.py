"""Synthetic Lorenz-63 trajectories via scipy ODE integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class LorenzConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    t0: float = 0.0
    t1: float = 50.0
    dt: float = 0.01
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0)
    burn_in: float = 5.0  # discard transient before saving


def lorenz_rhs(t: float, state: np.ndarray, cfg: LorenzConfig) -> np.ndarray:
    x, y, z = state
    return np.array(
        [
            cfg.sigma * (y - x),
            x * (cfg.rho - z) - y,
            x * y - cfg.beta * z,
        ],
        dtype=np.float64,
    )


def simulate_lorenz(cfg: LorenzConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate Lorenz-63 and return (times, states).

    states shape: (T, 3) — columns are [x, y, z].
    """
    cfg = cfg or LorenzConfig()
    t_eval = np.arange(cfg.t0 + cfg.burn_in, cfg.t1 + cfg.dt / 2, cfg.dt)
    sol = solve_ivp(
        lorenz_rhs,
        (cfg.t0, cfg.t1),
        np.asarray(cfg.x0, dtype=np.float64),
        args=(cfg,),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Lorenz integration failed: {sol.message}")
    return sol.t, sol.y.T.astype(np.float32)


def train_val_split(
    trajectory: np.ndarray, train_frac: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Split (T, 3) along time."""
    n = trajectory.shape[0]
    split = int(n * train_frac)
    return trajectory[:split], trajectory[split:]


def normalize(
    train: np.ndarray, val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score using train statistics only.

    Works for both single (T, 3) arrays and batched (N, T, 3) arrays.
    """
    flat_train = train.reshape(-1, train.shape[-1])
    mean = flat_train.mean(axis=0, keepdims=True)
    std = flat_train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (val - mean) / std, mean.squeeze(), std.squeeze()


def simulate_many_lorenz(
    n_trajectories: int = 8,
    cfg: LorenzConfig | None = None,
    ic_jitter: float = 1e-2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a small dataset of Lorenz trajectories from perturbed initial conditions.

    Each trajectory uses the same dt/t1/burn_in, but its starting point is
    `x0 + N(0, ic_jitter)` so the trajectories diverge in a chaotic but
    physically consistent way (same attractor, different paths along it).

    Returns:
        times: (T,)
        data:  (n_trajectories, T, 3)
    """
    base = cfg or LorenzConfig()
    rng = np.random.default_rng(seed)
    trajectories = []
    times = None
    for i in range(n_trajectories):
        x0 = tuple(np.asarray(base.x0, dtype=np.float64) + rng.normal(0, ic_jitter, size=3))
        cfg_i = LorenzConfig(
            sigma=base.sigma,
            rho=base.rho,
            beta=base.beta,
            t0=base.t0,
            t1=base.t1,
            dt=base.dt,
            x0=x0,
            burn_in=base.burn_in,
        )
        t, traj = simulate_lorenz(cfg_i)
        times = t  # identical across trajectories by construction
        trajectories.append(traj)
    data = np.stack(trajectories, axis=0).astype(np.float32)
    return times, data
