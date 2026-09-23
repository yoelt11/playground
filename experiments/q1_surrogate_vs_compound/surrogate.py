"""Closed-form RBF function-space projection (Arm 2 surrogate target)."""

from __future__ import annotations

import numpy as np

from common import gaussian_rbf, laplacian_gaussian_rbf


def build_phys_matrix(points: np.ndarray, centers: np.ndarray, epsilon: float) -> np.ndarray:
    """A_phys[i,j] = L[phi_j](x_i) with L = -Delta."""
    n, k = len(points), len(centers)
    a = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            a[i, j] = -laplacian_gaussian_rbf(points[i] - centers[j], epsilon)
    return a


def build_eval_matrix(points: np.ndarray, centers: np.ndarray, epsilon: float) -> np.ndarray:
    """A_data[i,j] = phi_j(x_i)."""
    n, k = len(points), len(centers)
    a = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            a[i, j] = gaussian_rbf(np.linalg.norm(points[i] - centers[j]), epsilon)
    return a


def project_surrogate_target(
    collocation: np.ndarray,
    f_colloc: np.ndarray,
    gt_pts: np.ndarray,
    u_gt: np.ndarray,
    centers: np.ndarray,
    epsilon: float = 1.5,
    alpha: float = 1.0,
    beta: float = 50.0,
    boundary_pts: np.ndarray | None = None,
    u_bnd: np.ndarray | None = None,
    lambda_b: float = 1e4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Analytic minimizer of the penalized least-squares in the RBF basis.

    min_θ  α‖A_phys θ − f‖² + β‖A_data θ − u*‖²  (+ strong BC soft penalty)

    Solved via normal equations with np.linalg.pinv (rank-aware).
    Returns (theta, v_star_on_eval_pts_fn, info). Caller evaluates on any grid
    via eval_rbf(theta, centers, points, epsilon).
    """
    a_phys = build_phys_matrix(collocation, centers, epsilon)
    a_data = build_eval_matrix(gt_pts, centers, epsilon)

    # Normal equations: (α A_pᵀ A_p + β A_dᵀ A_d (+ BC)) θ = α A_pᵀ f + β A_dᵀ u*
    lhs = alpha * (a_phys.T @ a_phys) + beta * (a_data.T @ a_data)
    rhs = alpha * (a_phys.T @ f_colloc) + beta * (a_data.T @ u_gt)

    if boundary_pts is not None and u_bnd is not None:
        a_bnd = build_eval_matrix(boundary_pts, centers, epsilon)
        lhs = lhs + lambda_b * (a_bnd.T @ a_bnd)
        rhs = rhs + lambda_b * (a_bnd.T @ u_bnd)

    # Rank-aware solve
    singular_vals = np.linalg.svd(lhs, compute_uv=False)
    tol = max(singular_vals.shape) * np.finfo(float).eps * singular_vals.max()
    rank = int(np.sum(singular_vals > tol))
    theta = np.linalg.pinv(lhs, rcond=1e-10) @ rhs

    info = {
        "rank": rank,
        "cond_est": float(singular_vals.max() / max(singular_vals.min(), 1e-30)),
        "n_centers": len(centers),
        "alpha": alpha,
        "beta": beta,
    }
    return theta, theta, info


def eval_rbf(
    theta: np.ndarray,
    centers: np.ndarray,
    points: np.ndarray,
    epsilon: float = 1.5,
) -> np.ndarray:
    phi = build_eval_matrix(points, centers, epsilon)
    return phi @ theta


def rbf_basis_matrix(
    points: np.ndarray, centers: np.ndarray, epsilon: float = 1.5
) -> np.ndarray:
    return build_eval_matrix(points, centers, epsilon)
