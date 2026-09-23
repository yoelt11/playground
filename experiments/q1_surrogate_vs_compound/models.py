"""Shared MLP and autograd Laplacian for Arms 1–3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SolutionNet(nn.Module):
    """Fourier-feature MLP with hard Dirichlet BC (u=0 on ∂Ω).

    Shared by Arm 1 and Arm 2 so the only difference is target/loss.
    Random Fourier features mitigate tanh spectral bias when regressing onto
    the smooth surrogate target v*. Hard BC via ω=x(1-x)y(1-y) is exact for
    the manufactured solution (and well-conditioned: u/ω stays bounded).
    """

    def __init__(self, width: int = 128, n_freq: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        b = torch.randn(2, n_freq, generator=g) * (2.0 * math.pi)
        self.register_buffer("B", b)
        in_dim = 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        raw = self.net(feat)
        omega = x[:, 0:1] * (1.0 - x[:, 0:1]) * x[:, 1:2] * (1.0 - x[:, 1:2])
        return omega * raw


def laplacian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Δu via nested autograd. x must have requires_grad=True."""
    u = model(x)
    grad_u = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    du_dx = grad_u[:, 0:1]
    du_dy = grad_u[:, 1:2]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(
        du_dy, x, grad_outputs=torch.ones_like(du_dy), create_graph=True
    )[0][:, 1:2]
    return d2u_dx2 + d2u_dy2


def flat_grads(params) -> torch.Tensor:
    """Concatenate parameter gradients into one vector (zeros if missing)."""
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            chunks.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(chunks)


def grad_cosine(g1: torch.Tensor, g2: torch.Tensor) -> float:
    n1 = torch.linalg.norm(g1)
    n2 = torch.linalg.norm(g2)
    if n1.item() < 1e-12 or n2.item() < 1e-12:
        return 0.0
    return float((g1 @ g2 / (n1 * n2)).item())
