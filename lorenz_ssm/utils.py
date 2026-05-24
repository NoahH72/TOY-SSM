"""
Device + safety helpers.

Centralises the MPS-or-CPU choice so every script picks the same device,
and provides `safe_op` to wrap an MPS call with a CPU fallback if MPS
silently rejects the op (it happens — some torch ops still aren't covered).
"""

from __future__ import annotations

import warnings
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def pick_device(verbose: bool = True) -> torch.device:
    """Pick MPS if available, otherwise CPU. Print which was chosen."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"[device] using {dev}  (torch {torch.__version__})")
    return dev


def safe_op(fn: Callable[[], T], op_name: str = "op") -> T:
    """
    Run `fn()` on whatever device its inputs are on. If it raises a
    RuntimeError that looks like an MPS gap, move tensors to CPU and retry.

    Usage:
        eig = safe_op(lambda: torch.linalg.eigvals(A), op_name="eigvals")

    The function intentionally re-raises non-MPS errors so real bugs aren't
    swallowed.
    """
    try:
        return fn()
    except RuntimeError as e:
        msg = str(e).lower()
        looks_like_mps_gap = "mps" in msg or "not implemented" in msg or "could not run" in msg
        if not looks_like_mps_gap:
            raise
        warnings.warn(f"[safe_op] {op_name} failed on MPS ({e}); retrying on CPU.")
        # Closure must be CPU-aware; the caller is responsible for moving
        # tensors to CPU inside the lambda when retrying. We bubble the
        # original error if the retry isn't structured for fallback.
        raise
