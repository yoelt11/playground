"""Local helpers + sibling benchmark imports + RBF center allocation policies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Sibling benchmark (interface_anisotropic_rbf/common.py)
# Loaded via importlib under a unique name so this module can also be "common".
# ---------------------------------------------------------------------------

_SIBLING_DIR = Path(__file__).resolve().parent.parent / "interface_anisotropic_rbf"
if str(_SIBLING_DIR) not in sys.path:
    sys.path.insert(0, str(_SIBLING_DIR))

_spec = importlib.util.spec_from_file_location(
    "iar_common",
    _SIBLING_DIR / "common.py",
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load sibling common from {_SIBLING_DIR}")
_iar = importlib.util.module_from_spec(_spec)
sys.modules["iar_common"] = _iar
_spec.loader.exec_module(_iar)

u_exact = _iar.u_exact
f_exact = _iar.f_exact
kappa_of_points = _iar.kappa_of_points
make_grid = _iar.make_grid
interior_away_from_gamma = _iar.interior_away_from_gamma
boundary_points = _iar.boundary_points
gamma_points = _iar.gamma_points
gamma_normals = _iar.gamma_normals
interface_band_mask = _iar.interface_band_mask
rel_l2 = _iar.rel_l2
rel_l2_masked = _iar.rel_l2_masked
kansa_fit = _iar.kansa_fit
rbf_eval = _iar.rbf_eval
rbf_laplacian = _iar.rbf_laplacian
matrix_cond = _iar.matrix_cond
gaussian_rbf = _iar.gaussian_rbf
laplacian_gaussian_rbf = _iar.laplacian_gaussian_rbf
grad_gaussian_rbf = _iar.grad_gaussian_rbf

# Default experiment geometry / κ
DEFAULT_RESOLUTION = 40
DEFAULT_K = 64
DEFAULT_EPSILON = 3.5
DEFAULT_KAPPA_M = 1.0
DEFAULT_KAPPA_JUMP = 10.0
DEFAULT_GAMMA = "vertical"
DEFAULT_SEEDS = (0, 1, 2)
UNIFORM_FLOOR = 0.03
KANSA_RIDGE = 0.1


def default_epsilon(k: int) -> float:
    """Narrower kernels as K grows so the colloc matrix stays usable.

    Empirically ε≈3.5 at K=64 keeps the error-PDE corrector stable and
    discriminative across allocation arms; larger K needs larger ε.
    """
    return float(np.clip(3.5 * np.sqrt(64.0 / max(float(k), 1.0)), 2.5, 8.0))


def reshape_field(values: np.ndarray, resolution: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(resolution, resolution)


def pde_residual_fd(
    u_grid: np.ndarray,
    f_grid: np.ndarray,
    kappa_grid: np.ndarray,
    h: float,
) -> np.ndarray:
    """Interior residual R = −κ Δu − f via central FD (zeros on boundary rows/cols).

    Valid away from Γ where κ is constant on each side; callers should mask near Γ
    for collocation the same way the sibling Kansa setup does.
    """
    u = np.asarray(u_grid, dtype=np.float64)
    f = np.asarray(f_grid, dtype=np.float64)
    kap = np.asarray(kappa_grid, dtype=np.float64)
    R = np.zeros_like(u)
    lap = (
        u[1:-1, 2:]
        + u[1:-1, :-2]
        + u[2:, 1:-1]
        + u[:-2, 1:-1]
        - 4.0 * u[1:-1, 1:-1]
    ) / (h * h)
    R[1:-1, 1:-1] = -kap[1:-1, 1:-1] * lap - f[1:-1, 1:-1]
    return R


def residual_on_points(
    residual_grid: np.ndarray,
    grid: np.ndarray,
    points: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Nearest-grid residual samples at arbitrary points (for collocation RHS)."""
    R = np.asarray(residual_grid, dtype=np.float64).ravel()
    p = np.asarray(points, dtype=np.float64)
    ix = np.clip(np.rint(p[:, 0] * (resolution - 1)).astype(int), 0, resolution - 1)
    iy = np.clip(np.rint(p[:, 1] * (resolution - 1)).astype(int), 0, resolution - 1)
    return R[iy * resolution + ix]


def interior_for_allocation(
    grid: np.ndarray,
    bdry_tol: float = 1e-10,
) -> np.ndarray:
    """All strictly-interior indices (includes near-Γ — used only for |R| sampling)."""
    g = np.asarray(grid, dtype=np.float64)
    mask = (
        (g[:, 0] > bdry_tol)
        & (g[:, 0] < 1.0 - bdry_tol)
        & (g[:, 1] > bdry_tol)
        & (g[:, 1] < 1.0 - bdry_tol)
    )
    return np.where(mask)[0]


def flux_jump_of_field(
    field_fn,
    gamma_pts: np.ndarray,
    normals: np.ndarray,
    kappa_m: float,
    kappa_p: float,
    eps: float = 1e-4,
) -> np.ndarray:
    """[κ ∂ν u] for a smooth field via central difference along ν.

    For a C¹ field, [κ ∂ν u] = (κ⁺ − κ⁻) ∂ν u.
    """
    gp = np.asarray(gamma_pts, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    u_p = np.asarray(field_fn(gp + eps * nrm), dtype=np.float64).reshape(-1)
    u_m = np.asarray(field_fn(gp - eps * nrm), dtype=np.float64).reshape(-1)
    dnu = (u_p - u_m) / (2.0 * eps)
    return (kappa_p - kappa_m) * dnu


def fit_error_pde_corrector(
    centers: np.ndarray,
    interior: np.ndarray,
    residual_int: np.ndarray,
    kappa_int: np.ndarray,
    gamma_pts: np.ndarray,
    boundary_pts: np.ndarray,
    v_bnd: np.ndarray,
    epsilon: float,
    kappa_m: float,
    kappa_p: float,
    gamma: str = DEFAULT_GAMMA,
    ridge: float = KANSA_RIDGE,
    max_colloc: int | None = None,
    seed: int = 0,
    flux_jump_rhs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit v_RBF with L[v] = −R(u_NO) using sibling kernel math + ridge.

    Sibling kansa_fit collocates −κ Δv = f_int, so pass f_int = −residual.
    Continuity rows are skipped (global RBF is already C^∞). Flux rows use
    [κ ∂ν v] = −[κ ∂ν u_NO] when ``flux_jump_rhs`` is provided (the correct
    interface condition for the error field).
    """
    centers = np.asarray(centers, dtype=np.float64)
    interior = np.asarray(interior, dtype=np.float64)
    residual_int = np.asarray(residual_int, dtype=np.float64)
    kappa_int = np.asarray(kappa_int, dtype=np.float64)
    gamma_pts = np.asarray(gamma_pts, dtype=np.float64)
    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)
    v_bnd = np.asarray(v_bnd, dtype=np.float64)

    if max_colloc is None:
        max_colloc = max(3 * len(centers), 200)
    if len(interior) > max_colloc:
        rng = np.random.default_rng(seed + 404)
        abs_r = np.abs(residual_int)
        w = abs_r / max(float(abs_r.sum()), 1e-30)
        w = 0.5 * w + 0.5 / len(w)
        w = w / w.sum()
        pick = rng.choice(len(interior), size=max_colloc, replace=False, p=w)
        pick.sort()
        interior = interior[pick]
        residual_int = residual_int[pick]
        kappa_int = kappa_int[pick]

    f_int = -residual_int
    if flux_jump_rhs is None:
        flux_jump_rhs = np.zeros(len(gamma_pts), dtype=np.float64)
    else:
        flux_jump_rhs = np.asarray(flux_jump_rhs, dtype=np.float64).reshape(-1)

    weights, info = _kansa_fit_corrector(
        centers=centers,
        interior=interior,
        f_int=f_int,
        kappa_int=kappa_int,
        gamma_pts=gamma_pts,
        boundary_pts=boundary_pts,
        u_bnd=v_bnd,
        flux_jump_rhs=flux_jump_rhs,
        epsilon=float(epsilon),
        kappa_m=float(kappa_m),
        kappa_p=float(kappa_p),
        gamma=gamma,
        ridge=float(ridge) if ridge > 0 else 1e-4,
    )
    return weights, info


def _kansa_fit_corrector(
    centers: np.ndarray,
    interior: np.ndarray,
    f_int: np.ndarray,
    kappa_int: np.ndarray,
    gamma_pts: np.ndarray,
    boundary_pts: np.ndarray,
    u_bnd: np.ndarray,
    flux_jump_rhs: np.ndarray,
    epsilon: float,
    kappa_m: float,
    kappa_p: float,
    gamma: str,
    ridge: float,
    lambda_bc: float = 1e3,
    lambda_flux: float = 1e2,
) -> tuple[np.ndarray, dict]:
    """Kansa rows for the error-PDE corrector (vectorized; ridged LS).

    Rows: interior (−κ Δφ = f), flux ([κ ∂ν φ] = flux_jump_rhs), BC (φ = g).
    Continuity omitted — identically ~0 for a global RBF and only hurts cond.
    """
    K = len(centers)
    n_int = len(interior)
    n_g = len(gamma_pts)
    n_b = len(boundary_pts)
    normals = gamma_normals(gamma_pts, gamma=gamma)

    # Interior block (vectorized)
    diff_i = interior[:, None, :] - centers[None, :, :]
    M_int = -kappa_int[:, None] * laplacian_gaussian_rbf(diff_i, epsilon)

    # Flux block
    eps_n = 1e-4
    p_m = gamma_pts - eps_n * normals
    p_p = gamma_pts + eps_n * normals
    # grad φ at p±: shape (n_g, K, 2)
    g_m = grad_gaussian_rbf(p_m[:, None, :] - centers[None, :, :], epsilon)
    g_p = grad_gaussian_rbf(p_p[:, None, :] - centers[None, :, :], epsilon)
    # ∂ν φ = grad·ν
    dnu_m = np.sum(g_m * normals[:, None, :], axis=-1)
    dnu_p = np.sum(g_p * normals[:, None, :], axis=-1)
    M_flux = lambda_flux * (kappa_p * dnu_p - kappa_m * dnu_m)
    rhs_flux = lambda_flux * flux_jump_rhs

    # BC block
    diff_b = boundary_pts[:, None, :] - centers[None, :, :]
    r_b = np.linalg.norm(diff_b, axis=-1)
    M_bc = lambda_bc * gaussian_rbf(r_b, epsilon)
    rhs_bc = lambda_bc * u_bnd

    M = np.vstack([M_int, M_flux, M_bc])
    rhs = np.concatenate([f_int, rhs_flux, rhs_bc])

    # Prefer lstsq on M (better than forming MᵀM when ill-conditioned); ridge
    # via vertically stacking √λ I.
    if ridge > 0:
        M_aug = np.vstack([M, np.sqrt(ridge) * np.eye(K)])
        rhs_aug = np.concatenate([rhs, np.zeros(K)])
    else:
        M_aug, rhs_aug = M, rhs
    weights, *_ = np.linalg.lstsq(M_aug, rhs_aug, rcond=1e-8)
    cond, rank = matrix_cond(M)
    info = {
        "cond": cond,
        "rank": rank,
        "n_rows": int(M.shape[0]),
        "n_centers": K,
        "shape": list(M.shape),
        "gamma": gamma,
        "ridge": float(ridge),
    }
    return weights.astype(np.float64), info

# ---------------------------------------------------------------------------
# Allocation policies (identical kernel count K)
# ---------------------------------------------------------------------------

def _clip_interior(centers: np.ndarray, margin: float = 1e-3) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64).copy()
    c = np.clip(c, margin, 1.0 - margin)
    return c


def allocate_uniform(k: int, seed: int, margin: float = 0.02) -> np.ndarray:
    """Centers on a cartesian grid in (margin, 1−margin)², truncated/padded to K."""
    rng = np.random.default_rng(seed + 101)
    n_side = int(np.ceil(np.sqrt(k)))
    xs = np.linspace(margin, 1.0 - margin, n_side)
    ys = np.linspace(margin, 1.0 - margin, n_side)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    centers = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    if len(centers) > k:
        # Deterministic subsample of the cartesian lattice
        idx = rng.choice(len(centers), size=k, replace=False)
        idx.sort()
        centers = centers[idx]
    elif len(centers) < k:
        extra = rng.uniform(margin, 1.0 - margin, size=(k - len(centers), 2))
        centers = np.concatenate([centers, extra], axis=0)
    return _clip_interior(centers)


def _importance_weights(
    abs_r: np.ndarray,
    uniform_floor: float = UNIFORM_FLOOR,
    power: float = 2.0,
) -> np.ndarray:
    w = np.asarray(abs_r, dtype=np.float64).ravel()
    w = np.maximum(w, 0.0) ** float(power)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w = w / w.sum()
    floor = float(np.clip(uniform_floor, 0.0, 0.5))
    w = (1.0 - floor) * w + floor / len(w)
    return w / w.sum()


def _greedy_weighted_centers(
    pts: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    min_sep: float,
) -> np.ndarray:
    """Greedy residual-weighted placement with a hard exclusion radius."""
    n = len(pts)
    w = weights.copy()
    chosen_idx: list[int] = []
    for _ in range(k):
        w = w / max(float(w.sum()), 1e-30)
        i = int(rng.choice(n, p=w))
        chosen_idx.append(i)
        # Zero out neighbors within min_sep
        d = np.linalg.norm(pts - pts[i], axis=1)
        w[d < min_sep] = 0.0
        if not np.any(w > 0):
            # Relax separation if the domain is exhausted
            w = weights.copy()
            w[chosen_idx] = 0.0
            if not np.any(w > 0):
                break
    centers = pts[np.asarray(chosen_idx, dtype=int)]
    if len(centers) < k:
        # Fill remainder uniformly among unused points
        unused = np.setdiff1d(np.arange(n), np.asarray(chosen_idx, dtype=int))
        if len(unused) > 0:
            extra = rng.choice(unused, size=min(k - len(centers), len(unused)), replace=False)
            centers = np.concatenate([centers, pts[extra]], axis=0)
    return centers[:k]


def allocate_residual(
    k: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    interior_idx: np.ndarray,
    seed: int,
    uniform_floor: float = UNIFORM_FLOOR,
) -> np.ndarray:
    """Importance-resample K centers from interior points with density ∝ |R|² + floor."""
    rng = np.random.default_rng(seed + 202)
    idx = np.asarray(interior_idx, dtype=int)
    pts = np.asarray(grid, dtype=np.float64)[idx]
    abs_r = np.abs(np.asarray(residual_flat, dtype=np.float64).ravel()[idx])
    w = _importance_weights(abs_r, uniform_floor=uniform_floor, power=2.0)
    # Prefer without-replacement; fall back to greedy if pool is tight
    if len(pts) >= k:
        chosen = rng.choice(len(pts), size=k, replace=False, p=w)
        centers = pts[chosen].copy()
    else:
        n_side = int(np.sqrt(len(grid)))
        h = 1.0 / max(n_side - 1, 1)
        centers = _greedy_weighted_centers(pts, w, k, rng, min_sep=0.5 * h)
    return _clip_interior(centers)


def allocate_shuffled(
    k: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    interior_idx: np.ndarray,
    seed: int,
    uniform_floor: float = UNIFORM_FLOOR,
) -> np.ndarray:
    """Same |R| weight histogram, spatially scrambled (control).

    Shuffle |R| values over interior nodes, then importance-sample — preserves the
    marginal residual-weight distribution while destroying the spatial map.
    """
    rng = np.random.default_rng(seed + 303)
    idx = np.asarray(interior_idx, dtype=int)
    pts = np.asarray(grid, dtype=np.float64)[idx]
    abs_r = np.abs(np.asarray(residual_flat, dtype=np.float64).ravel()[idx])
    abs_r_shuf = rng.permutation(abs_r)
    w = _importance_weights(abs_r_shuf, uniform_floor=uniform_floor, power=2.0)
    if len(pts) >= k:
        chosen = rng.choice(len(pts), size=k, replace=False, p=w)
        centers = pts[chosen].copy()
    else:
        n_side = int(np.sqrt(len(grid)))
        h = 1.0 / max(n_side - 1, 1)
        centers = _greedy_weighted_centers(pts, w, k, rng, min_sep=0.5 * h)
    return _clip_interior(centers)


ALLOCATORS: dict[str, Callable[..., np.ndarray]] = {
    "uniform": lambda k, seed, **kw: allocate_uniform(k, seed),
    "residual_alloc": allocate_residual,
    "shuffled_alloc": allocate_shuffled,
}

ARM_NAMES = ("uniform", "residual_alloc", "shuffled_alloc")

# ---------------------------------------------------------------------------
# Phase 2 allocation: residual ± uncertainty blend
# ---------------------------------------------------------------------------

DEFAULT_LAMBDA_UNC = 1.0
STOCH_ARM_NAMES = ("residual_only", "residual_unc", "shuffled_unc")

# Phase 2b: complementary-σ controls (σ-only + split subsets; not a global blend)
DEFAULT_SPLIT_ALPHA = 0.25
STOCH_ARM_NAMES2 = ("residual_only", "sigma_only", "residual_split_sigma")


def _normalize_unit_mean(v: np.ndarray) -> np.ndarray:
    """Scale nonnegative field so mean==1 (or all-ones if near-zero)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    v = np.maximum(v, 0.0)
    m = float(v.mean())
    if m < 1e-30:
        return np.ones_like(v)
    return v / m


def _blend_residual_sigma(
    abs_r: np.ndarray,
    sigma: np.ndarray,
    lambda_unc: float = DEFAULT_LAMBDA_UNC,
) -> np.ndarray:
    """Genuine blend: unit-mean |R| + λ · unit-mean σ (neither term dominates by scale)."""
    r_n = _normalize_unit_mean(abs_r)
    s_n = _normalize_unit_mean(sigma)
    return r_n + float(lambda_unc) * s_n


def allocate_residual_unc(
    k: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    seed: int,
    lambda_unc: float = DEFAULT_LAMBDA_UNC,
    uniform_floor: float = UNIFORM_FLOOR,
) -> np.ndarray:
    """Importance-sample K centers with density ∝ (|R|_norm + λ σ_norm)² + floor."""
    rng = np.random.default_rng(seed + 502)
    idx = np.asarray(interior_idx, dtype=int)
    pts = np.asarray(grid, dtype=np.float64)[idx]
    abs_r = np.abs(np.asarray(residual_flat, dtype=np.float64).ravel()[idx])
    sig = np.asarray(sigma_flat, dtype=np.float64).ravel()[idx]
    blend = _blend_residual_sigma(abs_r, sig, lambda_unc=lambda_unc)
    w = _importance_weights(blend, uniform_floor=uniform_floor, power=2.0)
    if len(pts) >= k:
        chosen = rng.choice(len(pts), size=k, replace=False, p=w)
        centers = pts[chosen].copy()
    else:
        n_side = int(np.sqrt(len(grid)))
        h = 1.0 / max(n_side - 1, 1)
        centers = _greedy_weighted_centers(pts, w, k, rng, min_sep=0.5 * h)
    return _clip_interior(centers)


def allocate_residual_shuffledunc(
    k: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    seed: int,
    lambda_unc: float = DEFAULT_LAMBDA_UNC,
    uniform_floor: float = UNIFORM_FLOOR,
) -> np.ndarray:
    """Same residual+σ blend, but σ permuted over interior nodes (spatial control).

    Preserves the marginal σ distribution while destroying the uncertainty map.
    |R| stays spatially intact so the control isolates whether the σ *map* matters.
    """
    rng = np.random.default_rng(seed + 603)
    idx = np.asarray(interior_idx, dtype=int)
    pts = np.asarray(grid, dtype=np.float64)[idx]
    abs_r = np.abs(np.asarray(residual_flat, dtype=np.float64).ravel()[idx])
    sig = np.asarray(sigma_flat, dtype=np.float64).ravel()[idx]
    sig_shuf = rng.permutation(sig)
    blend = _blend_residual_sigma(abs_r, sig_shuf, lambda_unc=lambda_unc)
    w = _importance_weights(blend, uniform_floor=uniform_floor, power=2.0)
    if len(pts) >= k:
        chosen = rng.choice(len(pts), size=k, replace=False, p=w)
        centers = pts[chosen].copy()
    else:
        n_side = int(np.sqrt(len(grid)))
        h = 1.0 / max(n_side - 1, 1)
        centers = _greedy_weighted_centers(pts, w, k, rng, min_sep=0.5 * h)
    return _clip_interior(centers)


def sigma_residual_overlap_diagnostics(
    residual_flat: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    top_frac: float = 0.1,
) -> dict[str, float]:
    """Spatial correlation / top-set overlap between σ and |R| (diagnostic)."""
    idx = np.asarray(interior_idx, dtype=int)
    abs_r = np.abs(np.asarray(residual_flat, dtype=np.float64).ravel()[idx])
    sig = np.asarray(sigma_flat, dtype=np.float64).ravel()[idx]
    n = len(idx)
    if n < 3:
        return {
            "corr_sigma_abs_r": float("nan"),
            "frac_top_unc_with_low_r": float("nan"),
            "frac_top_r_with_low_unc": float("nan"),
            "top_frac": float(top_frac),
        }
    # Pearson correlation
    r_c = abs_r - abs_r.mean()
    s_c = sig - sig.mean()
    denom = float(np.linalg.norm(r_c) * np.linalg.norm(s_c))
    corr = float(np.dot(r_c, s_c) / denom) if denom > 1e-30 else 0.0

    k_top = max(1, int(np.ceil(top_frac * n)))
    top_unc = np.argpartition(sig, -k_top)[-k_top:]
    top_r = np.argpartition(abs_r, -k_top)[-k_top:]
    # "low" = below median of the complementary field
    r_med = float(np.median(abs_r))
    s_med = float(np.median(sig))
    frac_top_unc_low_r = float(np.mean(abs_r[top_unc] < r_med))
    frac_top_r_low_unc = float(np.mean(sig[top_r] < s_med))
    return {
        "corr_sigma_abs_r": corr,
        "frac_top_unc_with_low_r": frac_top_unc_low_r,
        "frac_top_r_with_low_unc": frac_top_r_low_unc,
        "top_frac": float(top_frac),
        "sigma_mean": float(sig.mean()),
        "sigma_std": float(sig.std()),
        "sigma_max": float(sig.max()),
        "abs_r_mean": float(abs_r.mean()),
        "abs_r_max": float(abs_r.max()),
    }


# ---------------------------------------------------------------------------
# Phase 2b allocation: σ-only + residual/σ split subsets (NOT a global blend)
# ---------------------------------------------------------------------------

def allocate_sigma(
    k: int,
    grid: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    seed: int,
    uniform_floor: float = UNIFORM_FLOOR,
) -> np.ndarray:
    """Importance-sample K centers with density ∝ σ² + floor (no residual term).

    Mirrors ``allocate_residual`` (same weight power / floor / RNG style) but
    uses the ensemble disagreement field as the weight.
    """
    rng = np.random.default_rng(seed + 702)
    idx = np.asarray(interior_idx, dtype=int)
    pts = np.asarray(grid, dtype=np.float64)[idx]
    sig = np.asarray(sigma_flat, dtype=np.float64).ravel()[idx]
    w = _importance_weights(sig, uniform_floor=uniform_floor, power=2.0)
    if len(pts) >= k:
        chosen = rng.choice(len(pts), size=k, replace=False, p=w)
        centers = pts[chosen].copy()
    else:
        n_side = int(np.sqrt(len(grid)))
        h = 1.0 / max(n_side - 1, 1)
        centers = _greedy_weighted_centers(pts, w, k, rng, min_sep=0.5 * h)
    return _clip_interior(centers)


def allocate_residual_split_sigma(
    k: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    seed: int,
    alpha: float = DEFAULT_SPLIT_ALPHA,
    uniform_floor: float = UNIFORM_FLOOR,
) -> tuple[np.ndarray, dict]:
    """Split budget: (1−α)·K residual-placed + α·K σ-placed, concatenated.

    Two independently importance-sampled subsets (NOT a blended weight field).
    Centers are ordered ``[residual_subset; sigma_subset]``. Returns
    ``(centers, meta)`` with subset sizes and the split index.

    RNG offsets: residual subset ``seed+802``, σ subset ``seed+903`` (distinct
    from residual_only ``+202`` / sigma_only ``+702``).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    k_sigma = int(round(alpha * k))
    k_sigma = max(0, min(int(k), k_sigma))
    k_residual = int(k) - k_sigma

    idx = np.asarray(interior_idx, dtype=int)
    pts = np.asarray(grid, dtype=np.float64)[idx]
    abs_r = np.abs(np.asarray(residual_flat, dtype=np.float64).ravel()[idx])
    sig = np.asarray(sigma_flat, dtype=np.float64).ravel()[idx]
    w_r = _importance_weights(abs_r, uniform_floor=uniform_floor, power=2.0)
    w_s = _importance_weights(sig, uniform_floor=uniform_floor, power=2.0)

    n_side = int(np.sqrt(len(grid)))
    h = 1.0 / max(n_side - 1, 1)
    min_sep = 0.5 * h

    def _draw(n_draw: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if n_draw <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        if len(pts) >= n_draw:
            chosen = rng.choice(len(pts), size=n_draw, replace=False, p=weights)
            return pts[chosen].copy()
        return _greedy_weighted_centers(pts, weights, n_draw, rng, min_sep=min_sep)

    rng_r = np.random.default_rng(seed + 802)
    rng_s = np.random.default_rng(seed + 903)
    centers_r = _draw(k_residual, w_r, rng_r)
    centers_s = _draw(k_sigma, w_s, rng_s)
    centers = np.concatenate([centers_r, centers_s], axis=0)
    meta = {
        "alpha": alpha,
        "n_residual": int(k_residual),
        "n_sigma": int(k_sigma),
        "split_index": int(k_residual),
    }
    return _clip_interior(centers), meta


def frac_centers_in_top_error(
    centers: np.ndarray,
    grid: np.ndarray,
    err_flat: np.ndarray,
    top_frac: float = 0.1,
) -> float:
    """Fraction of centers whose nearest grid node is in the top-frac |err| sites."""
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) == 0:
        return float("nan")
    grid = np.asarray(grid, dtype=np.float64)
    err = np.abs(np.asarray(err_flat, dtype=np.float64).ravel())
    n = len(grid)
    if n == 0 or len(err) != n:
        return float("nan")
    k_top = max(1, int(np.ceil(float(top_frac) * n)))
    # Nearest grid index per center (euclidean)
    d2 = ((centers[:, None, :] - grid[None, :, :]) ** 2).sum(axis=-1)
    nearest = np.argmin(d2, axis=1)
    top_set = set(np.argpartition(err, -k_top)[-k_top:].tolist())
    hits = sum(1 for i in nearest if int(i) in top_set)
    return float(hits) / float(len(centers))
