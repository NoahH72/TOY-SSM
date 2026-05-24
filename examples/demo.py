#!/usr/bin/env python3
"""Run a forward pass through the toy Mamba stack."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from toy_ssm import MambaBlock, MambaLM, MambaLMConfig


def main() -> None:
    torch.manual_seed(0)

    print("=== MambaBlock ===")
    batch, length, dim = 2, 32, 48
    x = torch.randn(batch, length, dim)
    block = MambaBlock(d_model=dim, d_state=16, d_conv=4, expand=2)
    y = block(x)
    print(f"in:  {tuple(x.shape)}")
    print(f"out: {tuple(y.shape)}")
    assert y.shape == x.shape

    print("\n=== MambaLM (toy language model) ===")
    cfg = MambaLMConfig(vocab_size=128, d_model=48, n_layer=2)
    model = MambaLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (batch, length))
    logits = model(ids)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"logits: {tuple(logits.shape)}")
    print(f"params: {n_params:,}")

    print("\nOK")


if __name__ == "__main__":
    main()
