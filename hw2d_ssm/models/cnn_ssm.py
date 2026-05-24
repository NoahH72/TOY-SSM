"""
CNN-SSM: Convolutional encoder + Linear SSM + Convolutional decoder
for full 2-D field prediction on Hasegawa-Wakatani plasma turbulence.

Architecture
------------
  frame_t  (C, H, W)
      │
  CNNEncoder    ← learns spatial compression
      │
  latent_t  (latent_dim,)
      │
  LinearSSM     ← learns temporal dynamics in latent space
      │
  latent_t+1_pred
      │
  CNNDecoder    ← reconstructs spatial structure
      │
  frame_t+1_pred  (C, H, W)

Training loss: pixel MSE on the decoded next frame.
Also returns latent predictions so you can supervise/inspect
the latent space separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lorenz_ssm.models.linear_ssm import LinearSSM


class CNNEncoder(nn.Module):
    """Compress (C, H, W) → latent vector of dim `latent_dim`.

    4 stride-2 convolutions shrink 128×128 → 64 → 32 → 16 → 8,
    then flatten + linear projection.  GroupNorm keeps training
    stable across batch sizes as small as 1.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),   # 64×64
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),            # 32×32
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # 16×16
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),          # 8×8
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, latent_dim)"""
        return self.fc(self.conv(x).flatten(1))


class CNNDecoder(nn.Module):
    """Expand latent vector → (C, H, W) field.

    Mirror of CNNEncoder: linear + reshape, then 4 transposed convolutions.
    """

    def __init__(self, latent_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),      # 16×16
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),       # 32×32
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),        # 64×64
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 128×128
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, out_channels, 128, 128)"""
        return self.deconv(self.fc(z).view(-1, 128, 8, 8))


class CNNSSM(nn.Module):
    """End-to-end CNN encoder → LinearSSM → CNN decoder.

    Given a window of T consecutive frames, encodes each to a latent,
    runs the SSM recurrence to predict the next latent at every step,
    then decodes each predicted latent back to a full field.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        state_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.encoder = CNNEncoder(in_channels, latent_dim)
        self.ssm = LinearSSM(latent_dim, latent_dim, state_dim)
        self.decoder = CNNDecoder(latent_dim, in_channels)

    def forward_sequence(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        frames: (B, T, C, H, W)  — window of T consecutive frames

        Returns
        -------
        pred_frames  : (B, T-1, C, H, W)  predicted frame at t+1
        pred_latents : (B, T-1, latent_dim)
        true_latents : (B, T,   latent_dim)
        """
        B, T, C, H, W = frames.shape
        # Encode all frames in one batched forward pass
        true_latents = self.encoder(frames.view(B * T, C, H, W)).view(B, T, -1)
        # SSM predicts next latent from current latent sequence
        pred_latents, _ = self.ssm.forward_sequence(true_latents)   # (B, T-1, L)
        # Decode all predicted latents in one batched forward pass
        pred_frames = self.decoder(
            pred_latents.reshape(B * (T - 1), -1)
        ).view(B, T - 1, C, H, W)
        return pred_frames, pred_latents, true_latents

    @torch.no_grad()
    def rollout(
        self,
        seed_frames: torch.Tensor,
        n_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout seeded from one or more context frames.

        seed_frames : (T_seed, C, H, W)  — warm-up context (not returned)
        n_steps     : number of future frames to generate

        Returns
        -------
        pred_frames  : (n_steps, C, H, W)
        pred_latents : (n_steps, latent_dim)
        """
        self.eval()
        T_seed, C, H, W = seed_frames.shape
        device = seed_frames.device

        # Warm up the hidden state through the seed context
        h = torch.zeros(1, self.state_dim, device=device, dtype=seed_frames.dtype)
        for t in range(T_seed):
            z = self.encoder(seed_frames[t : t + 1])
            _, h = self.ssm.one_step(z, h)

        # Autoregressively generate future frames
        frame = seed_frames[-1:]          # (1, C, H, W) — last seed frame
        pred_frames, pred_latents = [], []
        for _ in range(n_steps):
            z = self.encoder(frame)
            y, h = self.ssm.one_step(z, h)
            frame = self.decoder(y)
            pred_frames.append(frame.squeeze(0).cpu())
            pred_latents.append(y.squeeze(0).cpu())

        return torch.stack(pred_frames), torch.stack(pred_latents)
