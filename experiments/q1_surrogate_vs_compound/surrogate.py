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


RIDGE_CANDIDATES = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)


def _matrix_rank_cond(mat: np.ndarray) -> tuple[int, float, np.ndarray]:
    singular_vals = np.linalg.svd(mat, compute_uv=False)
    tol = max(singular_vals.shape) * np.finfo(float).eps * singular_vals.max()
    rank = int(np.sum(singular_vals > tol))
    cond = float(singular_vals.max() / max(singular_vals.min(), 1e-30))
    return rank, cond, singular_vals


def _solve_ridge(lhs: np.ndarray, rhs: np.ndarray, ridge: float) -> tuple[np.ndarray, dict]:
    """Closed-form (lhs + ridge I) θ = rhs via pinv; report rank/cond of ridged matrix."""
    k = lhs.shape[0]
    lhs_r = lhs + ridge * np.eye(k)
    rank, cond, _ = _matrix_rank_cond(lhs_r)
    theta = np.linalg.pinv(lhs_r, rcond=1e-10) @ rhs
    return theta, {"rank": rank, "cond_est": cond, "ridge": float(ridge)}


def select_ridge(
    lhs: np.ndarray,
    rhs: np.ndarray,
    candidates: tuple[float, ...] = RIDGE_CANDIDATES,
) -> tuple[np.ndarray, dict]:
    """Sweep ridge values; choose stable ε with the smallest condition number.

    Stability / no-explosion: θ finite and ‖θ‖ within [0.5, 2]× the median
    ‖θ‖ across finite candidates in the sweep (rejects over-regularized
    collapse and exploding unridged blow-ups). Among that stable set, pick
    the smallest cond(lhs+εI); tie-break toward smaller ridge. Still closed-form.
    """
    trials = []
    for ridge in candidates:
        theta, meta = _solve_ridge(lhs, rhs, ridge)
        finite = bool(np.all(np.isfinite(theta)))
        tnorm = float(np.linalg.norm(theta)) if finite else float("inf")
        trials.append(
            {
                "ridge": float(ridge),
                "theta": theta,
                "rank": meta["rank"],
                "cond_est": meta["cond_est"],
                "theta_norm": tnorm,
                "finite": finite,
            }
        )

    finite_trials = [t for t in trials if t["finite"]]
    if not finite_trials:
        best = trials[-1]
        return best["theta"], {
            "rank": best["rank"],
            "cond_est": best["cond_est"],
            "ridge": best["ridge"],
            "ridge_sweep": [
                {k: v for k, v in t.items() if k != "theta"} for t in trials
            ],
            "selection": "fallback-no-finite",
        }

    med = float(np.median([t["theta_norm"] for t in finite_trials]))
    lo, hi = 0.5 * med, 2.0 * med
    stable = [t for t in finite_trials if lo <= t["theta_norm"] <= hi]
    if not stable:
        # Fall back: closest ‖θ‖ to the median
        stable = [min(finite_trials, key=lambda t: abs(t["theta_norm"] - med))]

    # Smallest condition number among stable; tie-break toward smaller ridge.
    best = min(stable, key=lambda t: (t["cond_est"], t["ridge"]))
    return best["theta"], {
        "rank": best["rank"],
        "cond_est": best["cond_est"],
        "ridge": best["ridge"],
        "theta_norm": best["theta_norm"],
        "ridge_sweep": [
            {
                "ridge": t["ridge"],
                "rank": t["rank"],
                "cond_est": t["cond_est"],
                "theta_norm": t["theta_norm"],
                "finite": t["finite"],
            }
            for t in trials
        ],
        "selection": "min-cond-among-stable-norm-band",
        "theta_norm_median": med,
        "stable_norm_band": [lo, hi],
    }


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
    ridge_candidates: tuple[float, ...] = RIDGE_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Analytic minimizer of the penalized least-squares in the RBF basis.

    min_θ  α‖A_phys θ − f‖² + β‖A_data θ − u*‖²  (+ strong BC soft penalty)
         + ε‖θ‖²  (ridge chosen by cond/stability sweep; still closed-form)

    Solved via ridged normal equations with np.linalg.pinv (rank-aware).
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

    # Unridged diagnostics (what the old path saw)
    rank0, cond0, _ = _matrix_rank_cond(lhs)

    theta, ridge_info = select_ridge(lhs, rhs, candidates=ridge_candidates)

    info = {
        "rank": ridge_info["rank"],
        "cond_est": ridge_info["cond_est"],
        "ridge": ridge_info["ridge"],
        "rank_unridged": rank0,
        "cond_unridged": cond0,
        "ridge_sweep": ridge_info.get("ridge_sweep", []),
        "ridge_selection": ridge_info.get("selection", ""),
        "theta_norm": ridge_info.get("theta_norm", float(np.linalg.norm(theta))),
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
