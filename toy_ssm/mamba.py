"""
Mamba block — readable PyTorch, aligned with official ``mamba_ssm/modules/mamba_simple.py``.

Official path: ``mamba-main/mamba_ssm/modules/mamba_simple.py``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_ssm.selective_scan import selective_scan


class MambaBlock(nn.Module):
    """
    One Mamba layer (selective SSM + local conv + gating).

    Data flow (same as the paper / official code):
        x -> in_proj -> (x_branch, z_gate)
        x_branch -> causal depthwise conv -> SiLU
        -> x_proj splits into Δ, B, C (selective / input-dependent)
        -> selective_scan with fixed A, D
        -> multiply by silu(z_gate) -> out_proj
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"unknown dt_init: {dt_init}")

        # softplus(dt_proj.bias) ~ Uniform(dt_min, dt_max) at init
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D-style diagonal A: A[d,n] = -(n+1), stored as log
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, L, D) -> same shape."""
        batch, seqlen, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # (B, L, D_inner) -> (B, D_inner, L)
        x = x.transpose(1, 2)
        z = z.transpose(1, 2)

        x = self.act(self.conv1d(x)[..., :seqlen])

        # Selective parameters from conv output
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(batch * seqlen, self.d_inner))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt).reshape(batch, seqlen, self.d_inner).transpose(1, 2)
        B = B.reshape(batch, seqlen, self.d_state).transpose(1, 2)
        C = C.reshape(batch, seqlen, self.d_state).transpose(1, 2)

        A = -torch.exp(self.A_log.float())

        y = selective_scan(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y = y.transpose(1, 2)
        return self.out_proj(y)
