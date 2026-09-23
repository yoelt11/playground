"""Shared experiment setup: PDE grid, GT subset, RBF base solve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common import Poisson2D, RBFKansaSolver
from surrogate import eval_rbf, project_surrogate_target, rbf_basis_matrix


@dataclass
class ExperimentData:
    pde: Poisson2D
    grid: np.ndarray
    u_exact: np.ndarray
    f_exact: np.ndarray
    interior: np.ndarray
    interior_idx: np.ndarray
    f_interior: np.ndarray
    gt_idx: np.ndarray
    gt_pts: np.ndarray
    u_gt: np.ndarray
    boundary_pts: np.ndarray
    u_bnd: np.ndarray
    rbf_centers: np.ndarray
    epsilon: float
    u_base: np.ndarray
    u_base_weights: np.ndarray
    residual_base: np.ndarray
    v_star: np.ndarray
    theta_star: np.ndarray
    surrogate_info: dict
    basis_phi: np.ndarray
    seed: int
    gt_fraction: float = 0.12


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interior_mask(grid: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    return (
        (grid[:, 0] > tol)
        & (grid[:, 0] < 1.0 - tol)
        & (grid[:, 1] > tol)
        & (grid[:, 1] < 1.0 - tol)
    )


def build_experiment(
    seed: int = 0,
    resolution: int = 40,
    gt_fraction: float = 0.12,
    n_rbf_centers: int = 48,
    epsilon: float = 1.5,
    alpha: float = 1.0,
    beta: float = 80.0,
) -> ExperimentData:
    """Build shared setup for all three arms (fixed GT subset per seed)."""
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    pde = Poisson2D(resolution=resolution)
    grid = pde.grid_points
    u_exact = pde.u_exact(grid)
    f_exact = pde.f_exact(grid)

    mask = interior_mask(grid)
    interior_idx = np.where(mask)[0]
    interior = grid[interior_idx]
    f_interior = f_exact[interior_idx]

    n_gt = max(8, int(round(gt_fraction * len(interior))))
    gt_local = rng.choice(len(interior), size=n_gt, replace=False)
    gt_idx = interior_idx[gt_local]
    gt_pts = grid[gt_idx]
    # Modest observation noise so physics and data can disagree (compound-loss pathology).
    # Noise std is relative to ||u*|| scale; same draw is shared across all arms.
    u_gt_clean = u_exact[gt_idx]
    noise_std = 0.15 * float(np.std(u_gt_clean) + 1e-12)
    u_gt = u_gt_clean + rng.normal(0.0, noise_std, size=u_gt_clean.shape)

    boundary_pts = pde.get_boundary_points(num_pts=80)
    u_bnd = pde.u_exact(boundary_pts)

    # RBF centers: subsample of interior (shared base / surrogate basis)
    n_cent = min(n_rbf_centers, len(interior))
    cent_local = rng.choice(len(interior), size=n_cent, replace=False)
    rbf_centers = interior[cent_local]

    # Base solve (Arm 3 freeze + span reference; also seeds rbf-grad)
    base = RBFKansaSolver(epsilon=epsilon)
    base.fit(rbf_centers, pde.f_exact(rbf_centers), boundary_pts, u_bnd)
    u_base = base.predict(grid)
    u_base_weights = np.asarray(base.weights, dtype=float)
    residual_base = base.compute_residual(interior, f_interior)

    # Surrogate target v* in RBF span (Arm 2)
    theta_star, _, sinfo = project_surrogate_target(
        collocation=interior,
        f_colloc=f_interior,
        gt_pts=gt_pts,
        u_gt=u_gt,
        centers=rbf_centers,
        epsilon=epsilon,
        alpha=alpha,
        beta=beta,
        boundary_pts=boundary_pts,
        u_bnd=u_bnd,
    )
    v_star = eval_rbf(theta_star, rbf_centers, grid, epsilon)
    basis_phi = rbf_basis_matrix(grid, rbf_centers, epsilon)

    return ExperimentData(
        pde=pde,
        grid=grid,
        u_exact=u_exact,
        f_exact=f_exact,
        interior=interior,
        interior_idx=interior_idx,
        f_interior=f_interior,
        gt_idx=gt_idx,
        gt_pts=gt_pts,
        u_gt=u_gt,
        boundary_pts=boundary_pts,
        u_bnd=u_bnd,
        rbf_centers=rbf_centers,
        epsilon=epsilon,
        u_base=u_base,
        u_base_weights=u_base_weights,
        residual_base=residual_base,
        v_star=v_star,
        theta_star=theta_star,
        surrogate_info=sinfo,
        basis_phi=basis_phi,
        seed=seed,
        gt_fraction=gt_fraction,
    )


def rel_l2(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    denom = np.linalg.norm(u_exact)
    if denom < 1e-15:
        return float(np.linalg.norm(u_pred))
    return float(np.linalg.norm(u_pred - u_exact) / denom)


def to_torch(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t
