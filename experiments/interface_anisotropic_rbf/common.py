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
CIRCLE_CENTER = np.array([0.5, 0.5], dtype=np.float64)
CIRCLE_R = 0.3
# Inner radial exponent for circle MS (α⁺ = (κ⁻/κ⁺) α⁻ enforces [κ ∂_r u]=0).
CIRCLE_ALPHA_INNER = 2.0
CIRCLE_U_SCALE = 1.0
GAMMA_CHOICES = ("vertical", "circle")


def _validate_gamma(gamma: str) -> str:
    g = str(gamma).lower().strip()
    if g not in GAMMA_CHOICES:
        raise ValueError(f"gamma must be one of {GAMMA_CHOICES}, got {gamma!r}")
    return g


def radius_from_center(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(points, dtype=np.float64) - CIRCLE_CENTER, axis=-1)


def kappa_of_x(x: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """Piecewise-constant κ (vertical Γ): Ω⁻ (x<0.5) → κ⁻, Ω⁺ (x>0.5) → κ⁺."""
    return np.where(np.asarray(x) < X_GAMMA, kappa_m, kappa_p)


def kappa_of_points(
    points: np.ndarray,
    kappa_m: float,
    kappa_p: float,
    gamma: str = "vertical",
) -> np.ndarray:
    """Piecewise κ for vertical (x≷0.5) or circle (r≷R) geometry."""
    gamma = _validate_gamma(gamma)
    pts = np.asarray(points, dtype=np.float64)
    if gamma == "vertical":
        return kappa_of_x(pts[:, 0], kappa_m, kappa_p)
    return np.where(radius_from_center(pts) < CIRCLE_R, kappa_m, kappa_p)


def _slope_pair(kappa_m: float, kappa_p: float, a_m: float = 1.0):
    """a⁺ = (κ⁻/κ⁺) a⁻ so [κ ∂_x u]=0; b from [u]=0 at x=0.5."""
    a_p = (kappa_m / kappa_p) * a_m
    b = 0.5 * (a_m - a_p)
    return a_m, a_p, b


def _alpha_pair_circle(kappa_m: float, kappa_p: float):
    """α⁺ = (κ⁻/κ⁺) α⁻ so [κ ∂_r u]=0 at r=R with continuous u=C (r/R)^α."""
    a_m = float(CIRCLE_ALPHA_INNER)
    a_p = (kappa_m / kappa_p) * a_m
    return a_m, a_p


def _u_exact_vertical(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """Piecewise u* with kink at Γ: u=a x s(y) on −, (a⁺x+b)s(y) on +.

    s(y)=sin(πy). Continuity and [κ ∂_ν u]=0 hold exactly at x=0.5.
    """
    x, y = points[:, 0], points[:, 1]
    a_m, a_p, b = _slope_pair(kappa_m, kappa_p)
    s = np.sin(np.pi * y)
    left = a_m * x * s
    right = (a_p * x + b) * s
    return np.where(x < X_GAMMA, left, right)


def _f_exact_vertical(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """f = −div(κ ∇u*) = −κ Δu* on each side (κ piecewise-constant)."""
    x, y = points[:, 0], points[:, 1]
    a_m, a_p, b = _slope_pair(kappa_m, kappa_p)
    s = np.sin(np.pi * y)
    left = kappa_m * (np.pi**2) * a_m * x * s
    right = kappa_p * (np.pi**2) * (a_p * x + b) * s
    return np.where(x < X_GAMMA, left, right)


def du_dx_exact(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    a_m, a_p, _b = _slope_pair(kappa_m, kappa_p)
    s = np.sin(np.pi * y)
    return np.where(x < X_GAMMA, a_m * s, a_p * s)


def _u_exact_circle(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """Radial power-law MS: u = C (r/R)^α per side, α⁺=(κ⁻/κ⁺)α⁻, C fixed.

    Continuity and [κ ∂_r u]=0 hold exactly on the circle r=R. α⁻=2 keeps Δu
    bounded at the origin (Δ(r²)=4 in 2D).
    """
    r = radius_from_center(points)
    r_safe = np.maximum(r, 1e-30)
    a_m, a_p = _alpha_pair_circle(kappa_m, kappa_p)
    c = CIRCLE_U_SCALE
    u_m = c * (r_safe / CIRCLE_R) ** a_m
    u_p = c * (r_safe / CIRCLE_R) ** a_p
    return np.where(r < CIRCLE_R, u_m, u_p)


def _f_exact_circle(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """f = −κ Δu with Δ(r^α)=α² r^{α-2} in 2D (κ piecewise-constant)."""
    r = radius_from_center(points)
    r_safe = np.maximum(r, 1e-30)
    a_m, a_p = _alpha_pair_circle(kappa_m, kappa_p)
    c = CIRCLE_U_SCALE
    # u = C R^{-α} r^α  ⇒  Δu = C R^{-α} α² r^{α-2}  ⇒  f = −κ Δu
    f_m = -kappa_m * c * (a_m**2) * (CIRCLE_R ** (-a_m)) * (r_safe ** (a_m - 2.0))
    f_p = -kappa_p * c * (a_p**2) * (CIRCLE_R ** (-a_p)) * (r_safe ** (a_p - 2.0))
    return np.where(r < CIRCLE_R, f_m, f_p)


def du_dr_exact_circle(points: np.ndarray, kappa_m: float, kappa_p: float) -> np.ndarray:
    """Radial derivative ∂_r u* for the circle manufactured solution."""
    r = radius_from_center(points)
    r_safe = np.maximum(r, 1e-30)
    a_m, a_p = _alpha_pair_circle(kappa_m, kappa_p)
    c = CIRCLE_U_SCALE
    # ∂r [C (r/R)^α] = C α / R · (r/R)^{α-1}
    d_m = c * a_m / CIRCLE_R * (r_safe / CIRCLE_R) ** (a_m - 1.0)
    d_p = c * a_p / CIRCLE_R * (r_safe / CIRCLE_R) ** (a_p - 1.0)
    return np.where(r < CIRCLE_R, d_m, d_p)


def u_exact(
    points: np.ndarray,
    kappa_m: float,
    kappa_p: float,
    gamma: str = "vertical",
) -> np.ndarray:
    gamma = _validate_gamma(gamma)
    pts = np.asarray(points, dtype=np.float64)
    if gamma == "circle":
        return _u_exact_circle(pts, kappa_m, kappa_p)
    return _u_exact_vertical(pts, kappa_m, kappa_p)


def f_exact(
    points: np.ndarray,
    kappa_m: float,
    kappa_p: float,
    gamma: str = "vertical",
) -> np.ndarray:
    gamma = _validate_gamma(gamma)
    pts = np.asarray(points, dtype=np.float64)
    if gamma == "circle":
        return _f_exact_circle(pts, kappa_m, kappa_p)
    return _f_exact_vertical(pts, kappa_m, kappa_p)


def verify_interface_kink(
    kappa_m: float,
    kappa_p: float,
    n_gamma: int = 64,
    gamma: str = "vertical",
) -> dict:
    """Print/return [u] and [κ ∂ν u] at Γ collocation points (must be ~0)."""
    gamma = _validate_gamma(gamma)
    eps = 1e-12
    if gamma == "vertical":
        ys = np.linspace(0.02, 0.98, n_gamma)
        pts_m = np.stack([np.full(n_gamma, X_GAMMA - eps), ys], axis=1)
        pts_p = np.stack([np.full(n_gamma, X_GAMMA + eps), ys], axis=1)
        u_m = u_exact(pts_m, kappa_m, kappa_p, gamma=gamma)
        u_p = u_exact(pts_p, kappa_m, kappa_p, gamma=gamma)
        jump_u = u_p - u_m
        dux_m = du_dx_exact(pts_m, kappa_m, kappa_p)
        dux_p = du_dx_exact(pts_p, kappa_m, kappa_p)
        jump_flux = kappa_p * dux_p - kappa_m * dux_m
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, n_gamma, endpoint=False)
        normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        gamma_pts = CIRCLE_CENTER + CIRCLE_R * normals
        pts_m = gamma_pts - eps * normals
        pts_p = gamma_pts + eps * normals
        u_m = u_exact(pts_m, kappa_m, kappa_p, gamma=gamma)
        u_p = u_exact(pts_p, kappa_m, kappa_p, gamma=gamma)
        jump_u = u_p - u_m
        # One-sided ∂_r from the closed-form radial derivative
        dur_m = du_dr_exact_circle(pts_m, kappa_m, kappa_p)
        dur_p = du_dr_exact_circle(pts_p, kappa_m, kappa_p)
        jump_flux = kappa_p * dur_p - kappa_m * dur_m
    out = {
        "gamma": gamma,
        "max_abs_jump_u": float(np.max(np.abs(jump_u))),
        "mean_abs_jump_u": float(np.mean(np.abs(jump_u))),
        "max_abs_jump_flux": float(np.max(np.abs(jump_flux))),
        "mean_abs_jump_flux": float(np.mean(np.abs(jump_flux))),
        "kappa_m": float(kappa_m),
        "kappa_p": float(kappa_p),
    }
    print(
        f"  [u* interface kink γ={gamma}] max|[u]|={out['max_abs_jump_u']:.3e} "
        f"max|[κ∂νu]|={out['max_abs_jump_flux']:.3e} — must be ~1e-10 or better",
        flush=True,
    )
    return out


def verify_manufactured_pde(
    kappa_m: float,
    kappa_p: float,
    gamma: str = "vertical",
    n_check: int = 24,
    h: float = 1e-5,
) -> dict:
    """Check f == −div(κ ∇u) away from Γ via central-FD Laplacian (κ const/side)."""
    gamma = _validate_gamma(gamma)
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(n_check * 8):
        p = rng.uniform(0.08, 0.92, size=2)
        if gamma == "vertical":
            if abs(p[0] - X_GAMMA) < 0.04:
                continue
        else:
            if abs(radius_from_center(p[None, :])[0] - CIRCLE_R) < 0.04:
                continue
        pts.append(p)
        if len(pts) >= n_check:
            break
    pts = np.asarray(pts, dtype=np.float64)
    ue = u_exact(pts, kappa_m, kappa_p, gamma=gamma)
    fe = f_exact(pts, kappa_m, kappa_p, gamma=gamma)
    kap = kappa_of_points(pts, kappa_m, kappa_p, gamma=gamma)

    def u_at(q):
        return float(u_exact(np.asarray(q)[None, :], kappa_m, kappa_p, gamma=gamma)[0])

    lap = np.zeros(len(pts))
    for i, p in enumerate(pts):
        lap[i] = (
            u_at(p + np.array([h, 0.0]))
            + u_at(p - np.array([h, 0.0]))
            + u_at(p + np.array([0.0, h]))
            + u_at(p - np.array([0.0, h]))
            - 4.0 * ue[i]
        ) / (h**2)
    f_fd = -kap * lap
    abs_err = np.abs(f_fd - fe)
    scale = np.maximum(np.abs(fe), 1e-8)
    rel_err = abs_err / scale
    out = {
        "gamma": gamma,
        "max_abs_err": float(np.max(abs_err)),
        "max_rel_err": float(np.max(rel_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "n_check": int(len(pts)),
    }
    print(
        f"  [manufactured PDE γ={gamma}] f vs −κΔu_FD: "
        f"max|err|={out['max_abs_err']:.3e} max_rel={out['max_rel_err']:.3e}",
        flush=True,
    )
    return out


def default_n_centers(resolution: int) -> int:
    """N-scaled center count: ~48 at res=40, doubles at res=80."""
    return max(32, int(round(48 * int(resolution) / 40)))


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


def grad_gaussian_rbf(r_vec: np.ndarray, epsilon: float) -> np.ndarray:
    """∇φ for φ=exp(−ε² r²): −2 ε² (x−μ) φ. Returns shape (..., 2)."""
    r_vec = np.asarray(r_vec, dtype=np.float64)
    r2 = np.sum(r_vec**2, axis=-1)
    phi = np.exp(-(epsilon**2) * r2)
    return -2.0 * (epsilon**2) * r_vec * phi[..., None]

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
    gamma: str = "vertical",
) -> np.ndarray:
    """Interior indices with dist(·,Γ) > gamma_tol (Ω∖Γ)."""
    gamma = _validate_gamma(gamma)
    grid = np.asarray(grid, dtype=np.float64)
    interior = (
        (grid[:, 0] > bdry_tol)
        & (grid[:, 0] < 1.0 - bdry_tol)
        & (grid[:, 1] > bdry_tol)
        & (grid[:, 1] < 1.0 - bdry_tol)
    )
    if gamma == "vertical":
        away = np.abs(grid[:, 0] - X_GAMMA) > gamma_tol
    else:
        away = np.abs(radius_from_center(grid) - CIRCLE_R) > gamma_tol
    return np.where(interior & away)[0]


def boundary_points(n_per_side: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_per_side, endpoint=False)
    bottom = np.stack([t, np.zeros_like(t)], axis=1)
    top = np.stack([t, np.ones_like(t)], axis=1)
    left = np.stack([np.zeros_like(t), t], axis=1)
    right = np.stack([np.ones_like(t), t], axis=1)
    return np.concatenate([bottom, top, left, right], axis=0)


def gamma_points(n: int = 40, gamma: str = "vertical") -> np.ndarray:
    gamma = _validate_gamma(gamma)
    if gamma == "vertical":
        ys = np.linspace(0.02, 0.98, n)
        return np.stack([np.full(n, X_GAMMA), ys], axis=1)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return CIRCLE_CENTER + CIRCLE_R * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def gamma_normals(gamma_pts: np.ndarray, gamma: str = "vertical") -> np.ndarray:
    """Unit normals on Γ (outward from Ω⁻ / pointing into Ω⁺)."""
    gamma = _validate_gamma(gamma)
    n = len(gamma_pts)
    if gamma == "vertical":
        return np.tile(np.array([1.0, 0.0]), (n, 1))
    d = np.asarray(gamma_pts, dtype=np.float64) - CIRCLE_CENTER
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(norms, 1e-30)


def interface_band_mask(
    points: np.ndarray, band: float = 0.05, gamma: str = "vertical"
) -> np.ndarray:
    gamma = _validate_gamma(gamma)
    pts = np.asarray(points, dtype=np.float64)
    if gamma == "vertical":
        return np.abs(pts[:, 0] - X_GAMMA) <= band
    return np.abs(radius_from_center(pts) - CIRCLE_R) <= band

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
    gamma: str = "vertical",
) -> tuple[np.ndarray, dict]:
    """Weighted LS Kansa: −κ Δu = f on Ω∖Γ, [u]=0 & [κ∂νu]=0 on Γ, u=g on ∂Ω.

    Global smooth RBF ⇒ [u]=0 auto; flux row enforces [κ ∂_ν u]≈0 on Γ along the
    geometry normal (x-hat for vertical, radial for circle). Continuity row kept
    for completeness.
    """
    gamma = _validate_gamma(gamma)
    K = len(centers)
    n_int = len(interior)
    n_g = len(gamma_pts)
    n_b = len(boundary_pts)
    normals = gamma_normals(gamma_pts, gamma=gamma)

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

    # Continuity [u]=0: φ_j(x⁺) − φ_j(x⁻) along ±ε ν
    row0 = n_int
    eps_n = 1e-8
    for i in range(n_g):
        gp = gamma_pts[i]
        nrm = normals[i]
        p_m = gp - eps_n * nrm
        p_p = gp + eps_n * nrm
        for j in range(K):
            d_m = np.linalg.norm(p_m - centers[j])
            d_p = np.linalg.norm(p_p - centers[j])
            M[row0 + i, j] = lambda_cont * (
                gaussian_rbf(d_p, epsilon) - gaussian_rbf(d_m, epsilon)
            )
        rhs[row0 + i] = 0.0

    # Flux [κ ∂_ν u]=0: κ⁺ (∇φ·ν)(x⁺) − κ⁻ (∇φ·ν)(x⁻)
    row1 = n_int + n_g
    for i in range(n_g):
        gp = gamma_pts[i]
        nrm = normals[i]
        p_m = gp - eps_n * nrm
        p_p = gp + eps_n * nrm
        for j in range(K):
            g_m = grad_gaussian_rbf(p_m - centers[j], epsilon)
            g_p = grad_gaussian_rbf(p_p - centers[j], epsilon)
            M[row1 + i, j] = lambda_flux * (
                kappa_p * float(np.dot(g_p, nrm)) - kappa_m * float(np.dot(g_m, nrm))
            )
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
        "gamma": gamma,
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
    gamma: str = "vertical",
) -> tuple[np.ndarray, dict]:
    """Analytic min_θ α‖A_phys θ−f‖² + β‖A_data θ−u*‖² + BC + flux (+ ridge)."""
    gamma = _validate_gamma(gamma)
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

    # Flux rows on Γ: [κ ∂_ν φ] ≈ 0 along geometry normal
    eps_n = 1e-8
    normals = gamma_normals(gamma_pts, gamma=gamma)
    a_flux = np.zeros((len(gamma_pts), k))
    for i, gp in enumerate(gamma_pts):
        nrm = normals[i]
        p_m = gp - eps_n * nrm
        p_p = gp + eps_n * nrm
        for j in range(k):
            g_m = grad_gaussian_rbf(p_m - centers[j], epsilon)
            g_p = grad_gaussian_rbf(p_p - centers[j], epsilon)
            a_flux[i, j] = kappa_p * float(np.dot(g_p, nrm)) - kappa_m * float(
                np.dot(g_m, nrm)
            )
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
        "gamma": gamma,
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
    gamma: str = "vertical"
    pde_verify: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


def build_experiment(
    seed: int = 0,
    resolution: int = 40,
    gt_fraction: float = 0.12,
    n_rbf_centers: int | None = None,
    epsilon: float = 1.5,
    kappa_m: float = 1.0,
    kappa_jump: float = 10.0,
    n_gamma: int = 32,
    n_boundary: int = 32,
    band: float = 0.05,
    alpha: float = 1.0,
    beta: float = 80.0,
    gamma: str = "vertical",
) -> ExperimentData:
    """Shared setup for all arms (controlled centers + Kansa base + GT subset)."""
    gamma = _validate_gamma(gamma)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    kappa_p = kappa_m * kappa_jump
    if n_rbf_centers is None:
        n_rbf_centers = default_n_centers(resolution)

    verify_laplacian_closed_form(epsilon)
    iface = verify_interface_kink(kappa_m, kappa_p, n_gamma=n_gamma, gamma=gamma)
    pde = verify_manufactured_pde(kappa_m, kappa_p, gamma=gamma)

    grid, _, _ = make_grid(resolution)
    ue = u_exact(grid, kappa_m, kappa_p, gamma=gamma)
    fe = f_exact(grid, kappa_m, kappa_p, gamma=gamma)

    interior_idx = interior_away_from_gamma(grid, gamma=gamma)
    interior = grid[interior_idx]
    f_interior = fe[interior_idx]
    kappa_interior = kappa_of_points(interior, kappa_m, kappa_p, gamma=gamma)

    n_gt = max(8, int(round(gt_fraction * len(interior))))
    gt_local = rng.choice(len(interior), size=n_gt, replace=False)
    gt_idx = interior_idx[gt_local]
    gt_pts = grid[gt_idx]
    u_gt_clean = ue[gt_idx]
    noise_std = 0.10 * float(np.std(u_gt_clean) + 1e-12)
    u_gt = u_gt_clean + rng.normal(0.0, noise_std, size=u_gt_clean.shape)

    bnd = boundary_points(n_boundary)
    u_bnd = u_exact(bnd, kappa_m, kappa_p, gamma=gamma)
    gamma_pts = gamma_points(n_gamma, gamma=gamma)

    n_cent = min(n_rbf_centers, len(interior))
    cent_local = rng.choice(len(interior), size=n_cent, replace=False)
    centers = interior[cent_local].copy()

    weights, kansa_info = kansa_fit(
        centers=centers,
        interior=interior,
        f_int=f_interior,
        kappa_int=kappa_interior,
        gamma_pts=gamma_pts,
        boundary_pts=bnd,
        u_bnd=u_bnd,
        epsilon=epsilon,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        gamma=gamma,
    )
    print(
        f"  [Kansa] γ={gamma} cond={kansa_info['cond']:.3e} rank={kansa_info['rank']}/"
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
        gamma_pts=gamma_pts,
        boundary_pts=bnd,
        u_bnd=u_bnd,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    v_star = rbf_eval(grid, centers, theta_star, epsilon)
    band_mask = interface_band_mask(grid, band=band, gamma=gamma)

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
        gamma_pts=gamma_pts,
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
        gamma=gamma,
        pde_verify=pde,
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
