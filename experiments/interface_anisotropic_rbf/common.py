"""Interface-Γ Poisson problem, manufactured solution, shared Kansa base (JAX/numpy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rbf_kernel import build_jax_kernel_fns, vsd_from_isotropic_base


# ---------------------------------------------------------------------------
# Manufactured interface-Γ solution
# ---------------------------------------------------------------------------

X_GAMMA = 0.5


def kappa_of_x(x: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """Piecewise-constant κ: Ω⁻ (x<0.5) → κ⁻, Ω⁺ (x>0.5) → κ⁺."""
    return np.where(np.asarray(x) < X_GAMMA, kappa_m, kappa_p)


def _slope_pair(kappa_m: float, kappa_p: float, a_m: float = 1.0):
    """a⁺ = (κ⁻/κ⁺) a⁻ so [κ ∂_x u]=0; b from [u]=0 at x=0.5."""
    a_p = (kappa_m / kappa_p) * a_m
    b = 0.5 * (a_m - a_p)
    return a_m, a_p, b


def u_exact(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """Piecewise u* with kink at Γ: u=a x s(y) on −, (a⁺x+b)s(y) on +.

    s(y)=sin(πy). Continuity and [κ ∂_ν u]=0 hold exactly at x=0.5.
    """
    x, y = points[:, 0], points[:, 1]
    a_m, a_p, b = _slope_pair(kappa_m, kappa_p)
    s = np.sin(np.pi * y)
    left = a_m * x * s
    right = (a_p * x + b) * s
    return np.where(x < X_GAMMA, left, right)


def f_exact(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """f = −div(κ ∇u*) = −κ Δu* on each side (κ piecewise-constant)."""
    x, y = points[:, 0], points[:, 1]
    a_m, a_p, b = _slope_pair(kappa_m, kappa_p)
    s = np.sin(np.pi * y)
    # Δu⁻ = a⁻ x (−π² s), Δu⁺ = (a⁺x+b)(−π² s)
    # f = −κ Δu = κ π² · (piecewise factor) · s
    left = kappa_m * (np.pi**2) * a_m * x * s
    right = kappa_p * (np.pi**2) * (a_p * x + b) * s
    return np.where(x < X_GAMMA, left, right)


def du_dx_exact(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    a_m, a_p, _b = _slope_pair(kappa_m, kappa_p)
    s = np.sin(np.pi * y)
    return np.where(x < X_GAMMA, a_m * s, a_p * s)


def verify_interface_kink(
    kappa_m: float,
    kappa_p: float,
    n_gamma: int = 64,
) -> dict:
    """Print/return [u] and [κ ∂ν u] at Γ collocation points (must be ~0)."""
    ys = np.linspace(0.02, 0.98, n_gamma)
    # Evaluate one-sided limits: x→0.5⁻ and x→0.5⁺
    pts_m = np.stack([np.full(n_gamma, X_GAMMA - 1e-12), ys], axis=1)
    pts_p = np.stack([np.full(n_gamma, X_GAMMA + 1e-12), ys], axis=1)
    u_m = u_exact(pts_m, kappa_m, kappa_p)
    u_p = u_exact(pts_p, kappa_m, kappa_p)
    jump_u = u_p - u_m
    dux_m = du_dx_exact(pts_m, kappa_m, kappa_p)
    dux_p = du_dx_exact(pts_p, kappa_m, kappa_p)
    jump_flux = kappa_p * dux_p - kappa_m * dux_m
    out = {
        "max_abs_jump_u": float(np.max(np.abs(jump_u))),
        "mean_abs_jump_u": float(np.mean(np.abs(jump_u))),
        "max_abs_jump_flux": float(np.max(np.abs(jump_flux))),
        "mean_abs_jump_flux": float(np.mean(np.abs(jump_flux))),
        "kappa_m": float(kappa_m),
        "kappa_p": float(kappa_p),
    }
    print(
        f"  [u* interface kink] [u]={out['mean_abs_jump_u']:.3e} "
        f"(max {out['max_abs_jump_u']:.3e}); "
        f"[κ∂νu]={out['mean_abs_jump_flux']:.3e} "
        f"(max {out['max_abs_jump_flux']:.3e}) — must be ~0",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Isotropic Gaussian RBF closed forms
# ---------------------------------------------------------------------------

def gaussian_rbf(r: np.ndarray, epsilon: float) -> np.ndarray:
    return np.exp(-(epsilon * r) ** 2)


def laplacian_gaussian_rbf(r_vec: np.ndarray, epsilon: float) -> np.ndarray:
    """Δ exp(−ε² r²) in 2D: (4 ε⁴ r² − 4 ε²) exp(−ε² r²)."""
    r2 = np.sum(r_vec**2, axis=-1)
    a = epsilon**2
    return (4 * (a**2) * r2 - 4 * a) * np.exp(-a * r2)


def grad_x_gaussian_rbf(r_vec: np.ndarray, epsilon: float) -> np.ndarray:
    """∂/∂x of φ = exp(−ε² r²): −2 ε² (x−μ_x) φ."""
    r2 = np.sum(r_vec**2, axis=-1)
    phi = np.exp(-(epsilon**2) * r2)
    return -2.0 * (epsilon**2) * r_vec[..., 0] * phi


def verify_laplacian_closed_form(epsilon: float = 1.5) -> bool:
    """Spot-check closed-form Laplacian vs central FD (relative tol)."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(0.15, 0.85, size=(4, 2))
    pts = rng.uniform(0.15, 0.85, size=(5, 2))
    h = 1e-5
    ok = True
    worst = 0.0
    for p in pts:
        for c in centers:
            r_vec = p - c
            lap_an = float(laplacian_gaussian_rbf(r_vec, epsilon))

            def phi(q):
                return float(gaussian_rbf(np.linalg.norm(q - c), epsilon))

            lap_fd = (
                phi(p + np.array([h, 0.0]))
                + phi(p - np.array([h, 0.0]))
                + phi(p + np.array([0.0, h]))
                + phi(p - np.array([0.0, h]))
                - 4.0 * phi(p)
            ) / (h**2)
            scale = max(abs(lap_an), abs(lap_fd), 1e-8)
            rel = abs(lap_an - lap_fd) / scale
            worst = max(worst, rel)
            if rel > 2e-3:
                ok = False
    print(
        f"  [laplacian check] closed-form vs FD: {'PASS' if ok else 'FAIL'} "
        f"(worst rel={worst:.2e})",
        flush=True,
    )
    return ok


# ---------------------------------------------------------------------------
# Geometry / sampling
# ---------------------------------------------------------------------------

def make_grid(resolution: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, resolution)
    y = np.linspace(0.0, 1.0, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
    return grid, X, Y


def interior_away_from_gamma(
    grid: np.ndarray,
    gamma_tol: float = 1e-3,
    bdry_tol: float = 1e-10,
) -> np.ndarray:
    """Interior indices with |x−0.5| > gamma_tol (Ω∖Γ)."""
    return np.where(
        (grid[:, 0] > bdry_tol)
        & (grid[:, 0] < 1.0 - bdry_tol)
        & (grid[:, 1] > bdry_tol)
        & (grid[:, 1] < 1.0 - bdry_tol)
        & (np.abs(grid[:, 0] - X_GAMMA) > gamma_tol)
    )[0]


def boundary_points(n_per_side: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_per_side, endpoint=False)
    bottom = np.stack([t, np.zeros_like(t)], axis=1)
    top = np.stack([t, np.ones_like(t)], axis=1)
    left = np.stack([np.zeros_like(t), t], axis=1)
    right = np.stack([np.ones_like(t), t], axis=1)
    return np.concatenate([bottom, top, left, right], axis=0)


def gamma_points(n: int = 40) -> np.ndarray:
    ys = np.linspace(0.02, 0.98, n)
    return np.stack([np.full(n, X_GAMMA), ys], axis=1)


def interface_band_mask(points: np.ndarray, band: float = 0.05) -> np.ndarray:
    return np.abs(points[:, 0] - X_GAMMA) <= band


def rel_l2(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    denom = float(np.linalg.norm(u_exact))
    if denom < 1e-15:
        return float(np.linalg.norm(u_pred))
    return float(np.linalg.norm(u_pred - u_exact) / denom)


def rel_l2_masked(
    u_pred: np.ndarray, u_exact: np.ndarray, mask: np.ndarray
) -> float:
    return rel_l2(u_pred[mask], u_exact[mask])


def matrix_cond(mat: np.ndarray) -> tuple[float, int]:
    s = np.linalg.svd(mat, compute_uv=False)
    smax = float(s.max()) if s.size else 0.0
    smin = float(s.min()) if s.size else 0.0
    tol = max(s.shape) * np.finfo(float).eps * smax if s.size else 0.0
    rank = int(np.sum(s > tol)) if s.size else 0
    cond = smax / max(smin, 1e-30)
    return float(cond), rank


# ---------------------------------------------------------------------------
# Kansa collocation on κ-operator + interface + BC
# ---------------------------------------------------------------------------

def kansa_fit(
    centers: np.ndarray,
    interior: np.ndarray,
    f_int: np.ndarray,
    kappa_int: np.ndarray,
    gamma_pts: np.ndarray,
    boundary_pts: np.ndarray,
    u_bnd: np.ndarray,
    epsilon: float,
    kappa_m: float,
    kappa_p: float,
    lambda_bc: float = 1e4,
    lambda_cont: float = 1e3,
    lambda_flux: float = 1e3,
) -> tuple[np.ndarray, dict]:
    """Weighted LS Kansa: −κ Δu = f on Ω∖Γ, [u]=0 & [κ∂νu]=0 on Γ, u=g on ∂Ω.

    Global smooth RBF ⇒ [u]=0 auto; flux row enforces (κ⁺−κ⁻)∂_x u ≈ 0 on Γ
    (equivalent for C¹ fields). Continuity row still included for completeness.
    """
    K = len(centers)
    n_int = len(interior)
    n_g = len(gamma_pts)
    n_b = len(boundary_pts)

    # Rows: interior + continuity + flux + BC
    n_rows = n_int + 2 * n_g + n_b
    M = np.zeros((n_rows, K), dtype=np.float64)
    rhs = np.zeros(n_rows, dtype=np.float64)

    # Interior: −κ Δφ_j
    for i in range(n_int):
        for j in range(K):
            r_vec = interior[i] - centers[j]
            M[i, j] = -kappa_int[i] * laplacian_gaussian_rbf(r_vec, epsilon)
        rhs[i] = f_int[i]

    # Continuity [u]=0: φ_j(x⁺) − φ_j(x⁻) — for global C⁰ φ this is ~0;
    # evaluate at γ±ε for numerical row (still ~0, keeps structure explicit).
    row0 = n_int
    eps_x = 1e-8
    for i in range(n_g):
        gp = gamma_pts[i]
        p_m = gp + np.array([-eps_x, 0.0])
        p_p = gp + np.array([+eps_x, 0.0])
        for j in range(K):
            d_m = np.linalg.norm(p_m - centers[j])
            d_p = np.linalg.norm(p_p - centers[j])
            M[row0 + i, j] = lambda_cont * (
                gaussian_rbf(d_p, epsilon) - gaussian_rbf(d_m, epsilon)
            )
        rhs[row0 + i] = 0.0

    # Flux [κ ∂_x u]=0: κ⁺ ∂x φ(x⁺) − κ⁻ ∂x φ(x⁻)
    row1 = n_int + n_g
    for i in range(n_g):
        gp = gamma_pts[i]
        p_m = gp + np.array([-eps_x, 0.0])
        p_p = gp + np.array([+eps_x, 0.0])
        for j in range(K):
            gx_m = grad_x_gaussian_rbf(p_m - centers[j], epsilon)
            gx_p = grad_x_gaussian_rbf(p_p - centers[j], epsilon)
            M[row1 + i, j] = lambda_flux * (kappa_p * gx_p - kappa_m * gx_m)
        rhs[row1 + i] = 0.0

    # BC
    row2 = n_int + 2 * n_g
    for i in range(n_b):
        for j in range(K):
            d = np.linalg.norm(boundary_pts[i] - centers[j])
            M[row2 + i, j] = lambda_bc * gaussian_rbf(d, epsilon)
        rhs[row2 + i] = lambda_bc * u_bnd[i]

    weights, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    cond, rank = matrix_cond(M)
    info = {
        "cond": cond,
        "rank": rank,
        "n_rows": n_rows,
        "n_centers": K,
        "shape": list(M.shape),
    }
    return weights.astype(np.float64), info


def rbf_eval(
    points: np.ndarray, centers: np.ndarray, weights: np.ndarray, epsilon: float
) -> np.ndarray:
    # Vectorized: (N,K)
    diff = points[:, None, :] - centers[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    return gaussian_rbf(r, epsilon) @ weights


def rbf_laplacian(
    points: np.ndarray, centers: np.ndarray, weights: np.ndarray, epsilon: float
) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]  # (N,K,2)
    laps = laplacian_gaussian_rbf(diff, epsilon)  # (N,K)
    return laps @ weights


def rbf_residual_kappa(
    points: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
    kappa: np.ndarray,
    f_vals: np.ndarray,
) -> np.ndarray:
    """Signed residual L[u]−f with L=−κΔ."""
    lap = rbf_laplacian(points, centers, weights, epsilon)
    return -kappa * lap - f_vals


# ---------------------------------------------------------------------------
# Surrogate projection (isotropic RBF span, κ-operator)
# ---------------------------------------------------------------------------

RIDGE_CANDIDATES = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)


def project_surrogate_target(
    collocation: np.ndarray,
    f_colloc: np.ndarray,
    kappa_colloc: np.ndarray,
    gt_pts: np.ndarray,
    u_gt: np.ndarray,
    centers: np.ndarray,
    epsilon: float,
    gamma_pts: np.ndarray,
    boundary_pts: np.ndarray,
    u_bnd: np.ndarray,
    kappa_m: float,
    kappa_p: float,
    alpha: float = 1.0,
    beta: float = 80.0,
    lambda_bc: float = 1e4,
    lambda_flux: float = 1e3,
) -> tuple[np.ndarray, dict]:
    """Analytic min_θ α‖A_phys θ−f‖² + β‖A_data θ−u*‖² + BC + flux (+ ridge)."""
    n, k = len(collocation), len(centers)
    a_phys = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            a_phys[i, j] = -kappa_colloc[i] * laplacian_gaussian_rbf(
                collocation[i] - centers[j], epsilon
            )
    a_data = np.zeros((len(gt_pts), k))
    for i in range(len(gt_pts)):
        for j in range(k):
            a_data[i, j] = gaussian_rbf(
                np.linalg.norm(gt_pts[i] - centers[j]), epsilon
            )
    lhs = alpha * (a_phys.T @ a_phys) + beta * (a_data.T @ a_data)
    rhs = alpha * (a_phys.T @ f_colloc) + beta * (a_data.T @ u_gt)

    a_bnd = np.zeros((len(boundary_pts), k))
    for i in range(len(boundary_pts)):
        for j in range(k):
            a_bnd[i, j] = gaussian_rbf(
                np.linalg.norm(boundary_pts[i] - centers[j]), epsilon
            )
    lhs = lhs + lambda_bc * (a_bnd.T @ a_bnd)
    rhs = rhs + lambda_bc * (a_bnd.T @ u_bnd)

    # Flux rows on Γ: (κ⁺−κ⁻) ∂x φ ≈ 0  (smooth-basis form)
    eps_x = 1e-8
    a_flux = np.zeros((len(gamma_pts), k))
    for i, gp in enumerate(gamma_pts):
        p_m = gp + np.array([-eps_x, 0.0])
        p_p = gp + np.array([+eps_x, 0.0])
        for j in range(k):
            gx_m = grad_x_gaussian_rbf(p_m - centers[j], epsilon)
            gx_p = grad_x_gaussian_rbf(p_p - centers[j], epsilon)
            a_flux[i, j] = kappa_p * gx_p - kappa_m * gx_m
    lhs = lhs + lambda_flux * (a_flux.T @ a_flux)

    cond0, rank0 = matrix_cond(lhs)
    best_theta, best_meta = None, None
    trials = []
    for ridge in RIDGE_CANDIDATES:
        lhs_r = lhs + ridge * np.eye(k)
        cond, rank = matrix_cond(lhs_r)
        theta = np.linalg.pinv(lhs_r, rcond=1e-10) @ rhs
        finite = bool(np.all(np.isfinite(theta)))
        tnorm = float(np.linalg.norm(theta)) if finite else float("inf")
        trials.append({"ridge": ridge, "cond": cond, "rank": rank, "tnorm": tnorm, "finite": finite})
        if not finite:
            continue
        if best_theta is None or cond < best_meta["cond"]:
            best_theta = theta
            best_meta = {"ridge": float(ridge), "cond": cond, "rank": rank, "tnorm": tnorm}
    if best_theta is None:
        best_theta = np.linalg.pinv(lhs + 1e-2 * np.eye(k)) @ rhs
        best_meta = {"ridge": 1e-2, "cond": float("nan"), "rank": 0, "tnorm": float("nan")}
    info = {
        **best_meta,
        "cond_unridged": cond0,
        "rank_unridged": rank0,
        "n_centers": k,
        "ridge_sweep": trials,
    }
    return best_theta.astype(np.float64), info


# ---------------------------------------------------------------------------
# Shared experiment pack
# ---------------------------------------------------------------------------

@dataclass
class ExperimentData:
    grid: np.ndarray
    u_exact: np.ndarray
    f_exact: np.ndarray
    interior: np.ndarray
    interior_idx: np.ndarray
    f_interior: np.ndarray
    kappa_interior: np.ndarray
    gt_idx: np.ndarray
    gt_pts: np.ndarray
    u_gt: np.ndarray
    boundary_pts: np.ndarray
    u_bnd: np.ndarray
    gamma_pts: np.ndarray
    rbf_centers: np.ndarray
    epsilon: float
    kappa_m: float
    kappa_p: float
    kappa_jump: float
    u_base: np.ndarray
    u_base_weights: np.ndarray
    residual_base: np.ndarray
    kansa_info: dict
    v_star: np.ndarray
    theta_star: np.ndarray
    surrogate_info: dict
    seed: int
    band_mask: np.ndarray
    interface_verify: dict
    resolution: int = 40
    gt_fraction: float = 0.12
    extras: dict = field(default_factory=dict)


def build_experiment(
    seed: int = 0,
    resolution: int = 36,
    gt_fraction: float = 0.12,
    n_rbf_centers: int = 40,
    epsilon: float = 1.5,
    kappa_m: float = 1.0,
    kappa_jump: float = 10.0,
    n_gamma: int = 32,
    n_boundary: int = 32,
    band: float = 0.05,
    alpha: float = 1.0,
    beta: float = 80.0,
) -> ExperimentData:
    """Shared setup for all arms (controlled centers + Kansa base + GT subset)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    kappa_p = kappa_m * kappa_jump

    verify_laplacian_closed_form(epsilon)
    iface = verify_interface_kink(kappa_m, kappa_p, n_gamma=n_gamma)

    grid, _, _ = make_grid(resolution)
    ue = u_exact(grid, kappa_m, kappa_p)
    fe = f_exact(grid, kappa_m, kappa_p)

    interior_idx = interior_away_from_gamma(grid)
    interior = grid[interior_idx]
    f_interior = fe[interior_idx]
    kappa_interior = kappa_of_x(interior[:, 0], kappa_m, kappa_p)

    n_gt = max(8, int(round(gt_fraction * len(interior))))
    gt_local = rng.choice(len(interior), size=n_gt, replace=False)
    gt_idx = interior_idx[gt_local]
    gt_pts = grid[gt_idx]
    u_gt_clean = ue[gt_idx]
    noise_std = 0.10 * float(np.std(u_gt_clean) + 1e-12)
    u_gt = u_gt_clean + rng.normal(0.0, noise_std, size=u_gt_clean.shape)

    bnd = boundary_points(n_boundary)
    u_bnd = u_exact(bnd, kappa_m, kappa_p)
    gamma = gamma_points(n_gamma)

    n_cent = min(n_rbf_centers, len(interior))
    cent_local = rng.choice(len(interior), size=n_cent, replace=False)
    centers = interior[cent_local].copy()

    weights, kansa_info = kansa_fit(
        centers=centers,
        interior=interior,
        f_int=f_interior,
        kappa_int=kappa_interior,
        gamma_pts=gamma,
        boundary_pts=bnd,
        u_bnd=u_bnd,
        epsilon=epsilon,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
    )
    print(
        f"  [Kansa] cond={kansa_info['cond']:.3e} rank={kansa_info['rank']}/"
        f"{kansa_info['n_centers']} rows={kansa_info['n_rows']}",
        flush=True,
    )

    u_base = rbf_eval(grid, centers, weights, epsilon)
    residual_base = rbf_residual_kappa(
        interior, centers, weights, epsilon, kappa_interior, f_interior
    )

    theta_star, sinfo = project_surrogate_target(
        collocation=interior,
        f_colloc=f_interior,
        kappa_colloc=kappa_interior,
        gt_pts=gt_pts,
        u_gt=u_gt,
        centers=centers,
        epsilon=epsilon,
        gamma_pts=gamma,
        boundary_pts=bnd,
        u_bnd=u_bnd,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        alpha=alpha,
        beta=beta,
    )
    v_star = rbf_eval(grid, centers, theta_star, epsilon)
    band_mask = interface_band_mask(grid, band=band)

    print(
        f"  [base] relL2={rel_l2(u_base, ue):.4e} "
        f"band={rel_l2_masked(u_base, ue, band_mask):.4e}; "
        f"v* relL2={rel_l2(v_star, ue):.4e}",
        flush=True,
    )

    return ExperimentData(
        grid=grid,
        u_exact=ue,
        f_exact=fe,
        interior=interior,
        interior_idx=interior_idx,
        f_interior=f_interior,
        kappa_interior=kappa_interior,
        gt_idx=gt_idx,
        gt_pts=gt_pts,
        u_gt=u_gt,
        boundary_pts=bnd,
        u_bnd=u_bnd,
        gamma_pts=gamma,
        rbf_centers=centers,
        epsilon=epsilon,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        kappa_jump=kappa_jump,
        u_base=u_base,
        u_base_weights=weights,
        residual_base=residual_base,
        kansa_info=kansa_info,
        v_star=v_star,
        theta_star=theta_star,
        surrogate_info=sinfo,
        seed=seed,
        band_mask=band_mask,
        interface_verify=iface,
        resolution=resolution,
        gt_fraction=gt_fraction,
    )


def anisotropic_field_from_params(
    X: Any,
    mus: Any,
    log_sigmas: Any,
    angles: Any,
    w_norm: Any,
    sigma_w: float,
    precompute,
    evaluate,
):
    """Evaluate anisotropic VSD field (jax arrays)."""
    import jax.numpy as jnp

    weights = sigma_w * w_norm
    mus_p, w_p, inv_covs = precompute(mus, log_sigmas, angles, weights)
    return evaluate(X, mus_p, w_p, inv_covs)


def build_aniso_init(data: ExperimentData) -> dict:
    """VSD params seeded so init field == isotropic base."""
    sigma_w = float(np.max(np.abs(data.u_base_weights)))
    sigma_w = max(sigma_w, 1e-30)
    return vsd_from_isotropic_base(
        data.rbf_centers, sigma_w, data.epsilon, data.u_base_weights
    ) | {"sigma_w": sigma_w}


def verify_aniso_init_matches_base(data: ExperimentData) -> float:
    """rel-L2 between anisotropic-at-init and rbf-base (must be ~0)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    precompute, evaluate = build_jax_kernel_fns()
    init = build_aniso_init(data)
    X = jnp.asarray(data.grid, dtype=jnp.float64)
    mus = jnp.asarray(init["mus"], dtype=jnp.float64)
    log_sigmas = jnp.asarray(init["log_sigmas"], dtype=jnp.float64)
    angles = jnp.asarray(init["angles"], dtype=jnp.float64)
    w_norm = jnp.asarray(init["w_norm"], dtype=jnp.float64)
    u_init = np.asarray(
        anisotropic_field_from_params(
            X, mus, log_sigmas, angles, w_norm, init["sigma_w"], precompute, evaluate
        )
    )
    err = float(
        np.linalg.norm(u_init - data.u_base)
        / max(np.linalg.norm(data.u_base), 1e-30)
    )
    print(
        f"  [aniso init vs base] ‖u_init−u_base‖/‖u_base‖={err:.3e} (must be ~0)",
        flush=True,
    )
    return err
