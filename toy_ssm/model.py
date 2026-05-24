"""Small stacked Mamba LM for experiments — not meant to compete with official training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from toy_ssm.mamba import MambaBlock
from toy_ssm.norm import RMSNorm


@dataclass
class MambaLMConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layer: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


class MambaResidualBlock(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = MambaBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaLM(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            MambaResidualBlock(config) for _ in range(config.n_layer)
        )
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, L, vocab)."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
