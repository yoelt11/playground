"""Shared MLP, RBF field, and autograd Laplacian for Q1 arms."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RBFField(nn.Module):
    """Gaussian RBF expansion with trainable centers + O(1) weights (ε fixed).

    u(x) = Σ_j (σ · w̃_j) exp(−ε² ‖x − c_j‖²), where σ = max(|w_base|) is a
    frozen scale and w̃ = w_base / σ are O(1) trainable normalized weights.
    At init, σ · w̃ = w_base so the field matches the one-shot Kansa solution
    exactly; Adam sees O(1) parameters (centers + w̃) and can use a sane lr.

    Matches the Kansa / splat rbf-grad setup: centers and (normalized) weights
    are Adam-refined; epsilon and σ are frozen. Uses float64 so inheritance
    from the O(10^6) Kansa solve is not destroyed by float32 rounding.
    """

    def __init__(self, centers, weights, epsilon: float):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).clone()
        sigma = float(torch.max(torch.abs(w)).item())
        if sigma < 1e-30:
            sigma = 1.0
        self.register_buffer(
            "weight_scale", torch.tensor(sigma, dtype=torch.float64)
        )
        self.centers = nn.Parameter(
            torch.as_tensor(centers, dtype=torch.float64).clone()
        )
        # Trainable O(1) normalized weights; effective w = σ · w̃
        self.weights = nn.Parameter(w / sigma)
        self.register_buffer(
            "epsilon", torch.tensor(float(epsilon), dtype=torch.float64)
        )

    @property
    def effective_weights(self) -> torch.Tensor:
        return self.weight_scale * self.weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2) → (N, 1)
        if x.dtype != torch.float64:
            x = x.double()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, 2)
        r2 = (diff * diff).sum(dim=-1)  # (N, K)
        phi = torch.exp(-(self.epsilon ** 2) * r2)
        return (phi @ self.effective_weights).unsqueeze(-1)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Analytical Δu for Gaussian RBFs (matches common.laplacian_gaussian_rbf).

        Δφ_j = (4 a² r² − 4 a) exp(−a r²), a=ε²; Δu = Σ_j (σ w̃_j) Δφ_j.
        Prefer this over nested autograd for residual training — same formula as
        the Kansa operator, cleaner grads w.r.t. centers/weights.
        """
        if x.dtype != torch.float64:
            x = x.double()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, 2)
        r2 = (diff * diff).sum(dim=-1)  # (N, K)
        a = self.epsilon ** 2
        lap_phi = (4.0 * (a ** 2) * r2 - 4.0 * a) * torch.exp(-a * r2)
        return (lap_phi @ self.effective_weights).unsqueeze(-1)


class AnisotropicRBFField(nn.Module):
    """ePIL/VSD anisotropic Gaussian RBF with learnable shape (K, 6) latents.

    Per kernel j the VSD vector is [μ_x, μ_y, log σ_x, log σ_y, angle, (log)w].
    Field uses the same O(1) weight reparam as RBFField:
      u(x) = Σ_j (σ_w · w̃_j) φ_j(x),  σ_w = max(|w_base|) frozen,
      w̃ = w_base / σ_w trainable.

    Kernel (rotated anisotropic Gaussian, ePIL inv_cov form):
      R(a) = [[cos a, -sin a], [sin a, cos a]],
      Σ^{-1} = R diag(σ_x^{-2}, σ_y^{-2}) R^T,
      φ(x) = exp(−½ (x−μ)^T Σ^{-1} (x−μ)).

    Init matching isotropic Kansa φ = exp(−ε² r²):
      angle=0, σ_x=σ_y=1/(ε√2)  so  ½/σ² = ε²  (NOT σ=ε — that would be
      exp(−r²/(2ε²)), which does not reproduce the Kansa base). Centers and
      weights from the shared Kansa solve. At step 0 the field == u_base.
    """

    def __init__(self, centers, weights, epsilon: float):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1).clone()
        sigma_w = float(torch.max(torch.abs(w)).item())
        if sigma_w < 1e-30:
            sigma_w = 1.0
        self.register_buffer(
            "weight_scale", torch.tensor(sigma_w, dtype=torch.float64)
        )
        self.register_buffer(
            "epsilon", torch.tensor(float(epsilon), dtype=torch.float64)
        )
        # Match isotropic exp(−ε² r²) under Mahalanobis form: σ = 1/(ε√2).
        sigma_iso = 1.0 / (float(epsilon) * math.sqrt(2.0))
        k = w.numel()
        mu = torch.as_tensor(centers, dtype=torch.float64).clone()
        if mu.ndim != 2 or mu.shape[1] != 2:
            raise ValueError(f"centers must be (K,2), got {tuple(mu.shape)}")
        if mu.shape[0] != k:
            raise ValueError("centers/weights length mismatch")
        self.centers = nn.Parameter(mu)
        self.log_sigma_x = nn.Parameter(
            torch.full((k,), math.log(sigma_iso), dtype=torch.float64)
        )
        self.log_sigma_y = nn.Parameter(
            torch.full((k,), math.log(sigma_iso), dtype=torch.float64)
        )
        self.angles = nn.Parameter(torch.zeros(k, dtype=torch.float64))
        self.weights = nn.Parameter(w / sigma_w)

    @property
    def effective_weights(self) -> torch.Tensor:
        return self.weight_scale * self.weights

    @property
    def sigma_x(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_x)

    @property
    def sigma_y(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_y)

    def inv_covs(self) -> torch.Tensor:
        """Σ^{-1} for each kernel, shape (K, 2, 2) — ePIL precompute_params form."""
        sx = self.sigma_x
        sy = self.sigma_y
        c = torch.cos(self.angles)
        s = torch.sin(self.angles)
        # R = [[c, -s], [s, c]]; inv = R @ diag(1/sx², 1/sy²) @ R^T
        inv_xx = c * c / (sx * sx) + s * s / (sy * sy)
        inv_yy = s * s / (sx * sx) + c * c / (sy * sy)
        inv_xy = c * s * (1.0 / (sx * sx) - 1.0 / (sy * sy))
        row0 = torch.stack([inv_xx, inv_xy], dim=-1)
        row1 = torch.stack([inv_xy, inv_yy], dim=-1)
        return torch.stack([row0, row1], dim=-2)  # (K, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float64:
            x = x.double()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, 2)
        inv = self.inv_covs()  # (K, 2, 2)
        # mahalanobis_j = diff_j^T inv_j diff_j
        # (N,K,2) @ (K,2,2) → (N,K,2) then · diff
        tmp = torch.einsum("nkd,kde->nke", diff, inv)
        mahal = (tmp * diff).sum(dim=-1)  # (N, K)
        phi = torch.exp(-0.5 * mahal)
        return (phi @ self.effective_weights).unsqueeze(-1)


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
