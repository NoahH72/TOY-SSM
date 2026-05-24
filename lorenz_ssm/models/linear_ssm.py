"""
Discrete-time linear state-space model.

    h_{t+1} = A h_t + B u_t
    y_t     = C h_{t+1} + D u_t

Here u_t is the observation at time t (e.g. Lorenz state x_t),
and y_t is trained to match u_{t+1} (one-step-ahead prediction).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearSSM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim

        # Small init → stable early training; A starts near a contraction
        self.A = nn.Parameter(torch.eye(state_dim) * 0.9 + torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.05)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.05)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.05)

    def forward_sequence(
        self,
        u: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u: (batch, T, input_dim) — observations x_t
        Returns:
            preds: (batch, T-1, output_dim) — y_t predicts u_{t+1}
            states: (batch, T, state_dim) — hidden state after each step
        """
        batch, T, _ = u.shape
        h = torch.zeros(batch, self.state_dim, device=u.device, dtype=u.dtype) if h0 is None else h0
        preds = []
        states = [h]

        for t in range(T - 1):
            ut = u[:, t]
            h = h @ self.A.T + ut @ self.B.T  # h_{t+1} = A h_t + B u_t
            yt = h @ self.C.T + ut @ self.D.T   # y_t = C h_{t+1} + D u_t
            preds.append(yt)
            states.append(h)

        states.append(h)
        return torch.stack(preds, dim=1), torch.stack(states, dim=1)

    def one_step(self, u: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single step: u (batch, input_dim), h (batch, state_dim)."""
        h_next = h @ self.A.T + u @ self.B.T
        y = h_next @ self.C.T + u @ self.D.T
        return y, h_next
