"""
Thin wrapper around `hw2d.run.run` that runs a Hasegawa-Wakatani 2D
simulation and writes results to an HDF5 file.

Defaults are tuned for a 'medium' run on an M4 Pro laptop:
  128 x 128 grid, end_time = 800, step_size = 0.025, snaps = 10
  → ~3200 frames written, ~10-15 min wall clock.

The HDF5 file produced by hw2d contains:
  * 3-D fields  : 'density', 'omega' (vorticity), 'phi' (potential)
                  shape (T, y, x)
  * scalar props: 'energy', 'enstrophy', 'gamma_n', ... shape (T, 1)
  * attrs       : simulation parameters (dt, frame_dt, c1, grid_pts, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class HWConfig:
    """A small subset of `hw2d.run.run`'s knobs that we actually care about."""
    grid_pts: int = 128
    end_time: float = 800.0
    step_size: float = 0.025
    snaps: int = 10               # save every `snaps` steps → frame_dt = dt * snaps
    c1: float = 1.0               # adiabatic coefficient
    k0: float = 0.15              # initial energy injection scale
    nu: float = 5.0e-8            # dissipation
    N: int = 3                    # hyperviscosity order
    seed: int = 0
    output_path: str = "hw2d_ssm/data/hw2d_medium.h5"

    def to_kwargs(self) -> dict:
        return dict(asdict(self))


def run_hw2d(
    cfg: HWConfig | None = None,
    properties: Iterable[str] = ("energy", "enstrophy", "kinetic_energy",
                                 "thermal_energy", "gamma_n", "gamma_c"),
    plot_properties: Iterable[str] = (),
    movie: bool = False,
    force_recompute: bool = False,
) -> Path:
    """Run the simulation and return the output HDF5 path.

    If the file already exists and force_recompute is False, returns
    immediately — re-running takes minutes, no reason to overwrite.
    """
    cfg = cfg or HWConfig()
    out = Path(cfg.output_path)
    if out.exists() and not force_recompute:
        print(f"[hw2d] {out} already exists — skipping (set force_recompute=True to overwrite)")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)

    # Import here so the heavy package is only loaded on demand
    from hw2d.run import run as _run

    print(f"[hw2d] simulating: grid {cfg.grid_pts}x{cfg.grid_pts}, "
          f"end_time={cfg.end_time}, c1={cfg.c1}, dt={cfg.step_size}, snaps={cfg.snaps}")
    _run(
        step_size=cfg.step_size,
        end_time=cfg.end_time,
        grid_pts=cfg.grid_pts,
        k0=cfg.k0,
        N=cfg.N,
        nu=cfg.nu,
        c1=cfg.c1,
        seed=cfg.seed,
        output_path=str(out),
        snaps=cfg.snaps,
        properties=list(properties),
        plot_properties=list(plot_properties),
        movie=movie,
        force_recompute=force_recompute,
    )
    print(f"[hw2d] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out
