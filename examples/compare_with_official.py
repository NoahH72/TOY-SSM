#!/usr/bin/env python3
"""
Compare toy selective_scan against the official reference implementation.

Uses the vendored repo at mamba-main/ (no pip install required).
Requires: torch, einops (from mamba's deps if you install mamba; or pip install einops).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAMBA_ROOT = ROOT / "mamba-main"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAMBA_ROOT))

import torch

from toy_ssm.selective_scan import selective_scan

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
except ImportError as e:
    print("Could not import mamba_ssm:", e)
    print("Install deps: pip install einops  (and build mamba if you want CUDA kernels)")
    sys.exit(1)


def main() -> None:
    torch.manual_seed(42)
    device = "cpu"
    B, D, L, N = 2, 8, 16, 4

    u = torch.randn(B, D, L, device=device)
    delta = torch.randn(B, D, L, device=device)
    A = -torch.exp(torch.randn(D, N, device=device))
    B_mat = torch.randn(B, N, L, device=device)
    C_mat = torch.randn(B, N, L, device=device)
    D_skip = torch.randn(D, device=device)
    z = torch.randn(B, D, L, device=device)
    delta_bias = torch.randn(D, device=device)

    out_toy = selective_scan(
        u,
        delta,
        A,
        B_mat,
        C_mat,
        D_skip,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
    )
    out_ref = selective_scan_ref(
        u,
        delta,
        A,
        B_mat,
        C_mat,
        D_skip,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
    )

    max_diff = (out_toy - out_ref).abs().max().item()
    print(f"max |toy - official ref|: {max_diff:.6e}")
    if max_diff < 1e-4:
        print("Match — toy scan matches selective_scan_ref in mamba-main/")
    else:
        print("Mismatch — check discretization / shapes")
        sys.exit(1)


if __name__ == "__main__":
    main()
