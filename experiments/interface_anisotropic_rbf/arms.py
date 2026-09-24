"""JAX arms for interface_anisotropic_rbf (6-row kill table)."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

# Prefer float64 for RBF inheritance fidelity (Kansa ↔ anisotropic init).
jax.config.update("jax_enable_x64", True)

from common import (
    ExperimentData,
    anisotropic_field_from_params,
    build_aniso_init,
    gamma_normals,
    rel_l2,
    rel_l2_masked,
    verify_aniso_init_matches_base,
)
from rbf_kernel import _jax_fn_evaluate, _jax_precompute_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stability_verdict(loss_hist: list[float], rel_hist: list[float]) -> dict:
    loss = np.asarray(loss_hist, dtype=float)
    rel = np.asarray(rel_hist, dtype=float)
    if len(loss) < 4:
        return {"verdict": "unknown", "up_frac": 0.0, "loss_cv_diff": 0.0}
    dloss = np.diff(loss)
    up_frac = float(np.mean(dloss > 0))
    abs_d = np.abs(dloss)
    loss_cv = float(abs_d.std() / (abs_d.mean() + 1e-12))
    late = rel[len(rel) // 2 :]
    rel_osc = float(np.std(np.diff(late))) if len(late) > 2 else 0.0
    if up_frac > 0.35 and loss_cv > 1.0:
        verdict = "oscillating"
    elif up_frac > 0.28 or rel_osc > 5e-3:
        verdict = "mildly-oscillating"
    else:
        verdict = "smooth"
    return {
        "verdict": verdict,
        "up_frac": up_frac,
        "loss_cv_diff": loss_cv,
        "rel_l2_late_osc": rel_osc,
    }


def _pack_metrics(
    arm: str,
    u_pred: np.ndarray,
    data: ExperimentData,
    hist: dict,
    extra: dict | None = None,
) -> dict:
    stab = stability_verdict(hist.get("loss", []), hist.get("rel_l2", []))
    out = {
        "arm": arm,
        "u_pred": u_pred,
        "history": hist,
        "final_rel_l2": rel_l2(u_pred, data.u_exact),
        "final_rel_l2_band": rel_l2_masked(u_pred, data.u_exact, data.band_mask),
        "base_rel_l2": rel_l2(data.u_base, data.u_exact),
        "base_rel_l2_band": rel_l2_masked(data.u_base, data.u_exact, data.band_mask),
        "mean_step_time": float(np.nanmean(hist["step_time"]))
        if hist.get("step_time")
        else float("nan"),
        "stability": stab["verdict"],
        "stability_detail": stab,
        "mean_grad_cos": (
            float(np.nanmean(hist["grad_cos"]))
            if hist.get("grad_cos") and np.any(np.isfinite(hist["grad_cos"]))
            else float("nan")
        ),
        "late_grad_cos": (
            float(np.nanmean(hist["grad_cos"][len(hist["grad_cos"]) // 2 :]))
            if hist.get("grad_cos") and np.any(np.isfinite(hist["grad_cos"]))
            else float("nan")
        ),
        "kansa_cond": data.kansa_info.get("cond"),
        "diverged": False,
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Arm 1: rbf-base (analytic / one-shot)
# ---------------------------------------------------------------------------

def run_rbf_base(data: ExperimentData) -> dict:
    hist = {"loss": [], "rel_l2": [], "step_time": [], "grad_cos": []}
    return _pack_metrics(
        "rbf-base",
        data.u_base,
        data,
        hist,
        extra={
            "stability": "analytic",
            "mean_step_time": float("nan"),
            "kansa_info": data.kansa_info,
        },
    )


# ---------------------------------------------------------------------------
# Isotropic RBF field (JAX) for rbf-grad
# ---------------------------------------------------------------------------

def _iso_phi_lap(points, centers, epsilon):
    diff = points[:, None, :] - centers[None, :, :]
    r2 = jnp.sum(diff**2, axis=-1)
    a = epsilon**2
    phi = jnp.exp(-a * r2)
    lap = (4 * (a**2) * r2 - 4 * a) * phi
    return phi, lap


def train_rbf_grad(
    data: ExperimentData,
    steps: int = 1500,
    lr: float = 2e-6,
    lr_centers: float = 1e-8,
    lambda_bc: float = 1e4,
    eval_every: int = 50,
) -> dict:
    """Isotropic Adam refine of weights (σ_w-normalized) + centers."""
    sigma_w = float(np.max(np.abs(data.u_base_weights)))
    sigma_w = max(sigma_w, 1e-30)
    w_norm0 = jnp.asarray(data.u_base_weights / sigma_w)
    centers0 = jnp.asarray(data.rbf_centers)
    epsilon = float(data.epsilon)

    params = {"w_norm": w_norm0, "centers": centers0}
    opt = optax.multi_transform(
        {
            "w": optax.adam(lr / sigma_w),
            "c": optax.adam(lr_centers),
        },
        {"w_norm": "w", "centers": "c"},
    )
    opt_state = opt.init(params)

    x_int = jnp.asarray(data.interior)
    f_int = jnp.asarray(data.f_interior)
    kappa_int = jnp.asarray(data.kappa_interior)
    x_bnd = jnp.asarray(data.boundary_pts)
    u_bnd = jnp.asarray(data.u_bnd)
    grid = jnp.asarray(data.grid)

    def loss_fn(p):
        w = sigma_w * p["w_norm"]
        phi, lap = _iso_phi_lap(x_int, p["centers"], epsilon)
        u_lap = lap @ w
        phys = jnp.mean((-kappa_int * u_lap - f_int) ** 2)
        phi_b, _ = _iso_phi_lap(x_bnd, p["centers"], epsilon)
        bc = jnp.mean((phi_b @ w - u_bnd) ** 2)
        return phys + lambda_bc * bc, (phys, bc)

    @jax.jit
    def step(p, st):
        (loss, (phys, bc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        p = {**p, "centers": jnp.clip(p["centers"], 0.0, 1.0)}
        return p, st, loss, phys, bc

    def predict(p):
        w = sigma_w * p["w_norm"]
        phi, _ = _iso_phi_lap(grid, p["centers"], epsilon)
        return np.asarray(phi @ w)

    u_init = predict(params)
    rel_init = rel_l2(u_init, data.u_exact)
    base_rel = rel_l2(data.u_base, data.u_exact)

    hist = {
        "loss": [],
        "loss_phys": [],
        "loss_bc": [],
        "rel_l2": [],
        "rel_l2_band": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }
    diverged = False
    loss0_phys = None
    for s in range(steps):
        t0 = time.perf_counter()
        params, opt_state, loss, phys, bc = step(params, opt_state)
        dt = time.perf_counter() - t0
        loss_f, phys_f, bc_f = float(loss), float(phys), float(bc)
        if not np.isfinite(loss_f):
            diverged = True
            print(f"    [rbf-grad] non-finite loss at step {s}; aborting", flush=True)
            hist["loss"].append(float("nan"))
            hist["loss_phys"].append(float("nan"))
            hist["loss_bc"].append(float("nan"))
            hist["grad_cos"].append(float("nan"))
            hist["step"].append(s)
            hist["step_time"].append(dt)
            hist["rel_l2"].append(float("nan"))
            hist["rel_l2_band"].append(float("nan"))
            break
        if s == 0:
            loss0_phys = phys_f
        hist["loss"].append(loss_f)
        hist["loss_phys"].append(phys_f)
        hist["loss_bc"].append(bc_f)
        hist["grad_cos"].append(float("nan"))
        hist["step"].append(s)
        hist["step_time"].append(dt)
        if s % eval_every == 0 or s == steps - 1:
            u_now = predict(params)
            hist["rel_l2"].append(rel_l2(u_now, data.u_exact))
            hist["rel_l2_band"].append(
                rel_l2_masked(u_now, data.u_exact, data.band_mask)
            )
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))
            hist["rel_l2_band"].append(
                hist["rel_l2_band"][-1] if hist["rel_l2_band"] else float("nan")
            )

    w_eff = np.asarray(sigma_w * params["w_norm"])
    centers_f = np.asarray(params["centers"])
    mean_w = float(np.mean(np.abs(w_eff - data.u_base_weights)))
    mean_c = float(
        np.mean(np.linalg.norm(centers_f - data.rbf_centers, axis=-1))
    )
    u_final = predict(params)
    final_rel = rel_l2(u_final, data.u_exact)
    delta = base_rel - final_rel
    resid_dec = (
        loss0_phys is not None
        and np.isfinite(hist["loss_phys"][-1])
        and hist["loss_phys"][-1] < loss0_phys
    )
    trained = (mean_w > 1e-10 or mean_c > 1e-10) and resid_dec
    print(
        f"    [rbf-grad train-moved] mean|Δw|={mean_w:.4e} mean|Δμ|={mean_c:.4e} "
        f"(non_vacuous={trained}); σ_w={sigma_w:.4e}",
        flush=True,
    )
    print(
        f"    [rbf-grad sanity] phys {loss0_phys:.4e}→{hist['loss_phys'][-1]:.4e}; "
        f"relL2 {rel_init:.4e}→{final_rel:.4e} (Δ={delta:+.4e}); diverged={diverged}",
        flush=True,
    )
    return _pack_metrics(
        "rbf-grad",
        u_final,
        data,
        hist,
        extra={
            "rel_l2_init": float(rel_init),
            "rel_l2_delta_vs_base": float(delta),
            "beats_base": bool((not diverged) and final_rel < base_rel),
            "mean_weight_displacement": mean_w,
            "mean_center_displacement": mean_c,
            "trained_non_vacuous": bool(trained),
            "diverged": bool(diverged),
            "weight_scale": sigma_w,
            "lr": float(lr),
            "lr_centers": float(lr_centers),
            "phys_loss_initial": float(loss0_phys) if loss0_phys is not None else float("nan"),
            "phys_loss_final": float(hist["loss_phys"][-1]),
        },
    )


# ---------------------------------------------------------------------------
# Arm 3: rbf-shape (anisotropic ePIL VSD)
# ---------------------------------------------------------------------------

def _aniso_laplacian(X, mus, log_sigmas, angles, weights, epsilon_stab=1e-6):
    """Δ of anisotropic field via nested jacfwd (sum of diagonal Hessians)."""

    def u_scalar(x):
        x2 = x[None, :]
        mus_p, w_p, inv_covs = _jax_precompute_params(
            jax, jnp, mus, log_sigmas, angles, weights, epsilon=epsilon_stab
        )
        return _jax_fn_evaluate(jnp, x2, mus_p, w_p, inv_covs)[0]

    def lap_one(x):
        # Hessian via jacfwd of grad
        hess = jax.jacfwd(jax.jacfwd(u_scalar))(x)
        return hess[0, 0] + hess[1, 1]

    return jax.vmap(lap_one)(X)


def train_rbf_shape(
    data: ExperimentData,
    steps: int = 1500,
    lr: float = 2e-6,
    lr_centers: float = 1e-8,
    lr_shape: float = 5e-8,
    lambda_bc: float = 1e4,
    lambda_flux: float = 1e3,
    eval_every: int = 50,
) -> dict:
    """Anisotropic VSD refine seeded from isotropic base (init == base)."""
    init_err = verify_aniso_init_matches_base(data)
    init = build_aniso_init(data)
    sigma_w = float(init["sigma_w"])

    params = {
        "w_norm": jnp.asarray(init["w_norm"]),
        "mus": jnp.asarray(init["mus"]),
        "log_sigmas": jnp.asarray(init["log_sigmas"]),
        "angles": jnp.asarray(init["angles"]),
    }
    snap = {k: np.asarray(v).copy() for k, v in params.items()}

    opt = optax.multi_transform(
        {
            "w": optax.adam(lr / sigma_w),
            "c": optax.adam(lr_centers),
            "s": optax.adam(lr_shape),
        },
        {
            "w_norm": "w",
            "mus": "c",
            "log_sigmas": "s",
            "angles": "s",
        },
    )
    opt_state = opt.init(params)

    x_int = jnp.asarray(data.interior)
    f_int = jnp.asarray(data.f_interior)
    kappa_int = jnp.asarray(data.kappa_interior)
    x_bnd = jnp.asarray(data.boundary_pts)
    u_bnd = jnp.asarray(data.u_bnd)
    gamma = jnp.asarray(data.gamma_pts)
    # Geometry normals: vertical → (1,0); circle → radial outward from Ω⁻
    nrms = jnp.asarray(
        gamma_normals(np.asarray(data.gamma_pts), gamma=data.gamma)
    )
    kappa_m, kappa_p = float(data.kappa_m), float(data.kappa_p)
    grid = jnp.asarray(data.grid)
    # Subsample interior for anisotropic lap (expensive nested AD) — full every eval
    rng = np.random.default_rng(data.seed + 21)
    n_col = min(128, len(data.interior))
    col_idx = rng.choice(len(data.interior), size=n_col, replace=False)
    x_col = x_int[col_idx]
    f_col = f_int[col_idx]
    k_col = kappa_int[col_idx]

    def field(p, X):
        w = sigma_w * p["w_norm"]
        return anisotropic_field_from_params(
            X,
            p["mus"],
            p["log_sigmas"],
            p["angles"],
            p["w_norm"],
            sigma_w,
            # non-jitted path uses raw helpers for grad
            lambda m, ls, a, ww: _jax_precompute_params(jax, jnp, m, ls, a, ww),
            lambda X_, m, w_, ic: _jax_fn_evaluate(jnp, X_, m, w_, ic),
        )

    def loss_fn(p):
        w = sigma_w * p["w_norm"]
        lap = _aniso_laplacian(
            x_col, p["mus"], p["log_sigmas"], p["angles"], w
        )
        phys = jnp.mean((-k_col * lap - f_col) ** 2)
        u_b = field(p, x_bnd)
        bc = jnp.mean((u_b - u_bnd) ** 2)
        # Flux soft: [κ ∂_ν u] = κ⁺(∇u·n)|_{+} − κ⁻(∇u·n)|_{-} on Γ.
        # n = gamma_normals (vertical: x-hat; circle: radial). One-sided ±ε n
        # matches Kansa; for C¹ fields this ≈ (κ⁺−κ⁻)(∇u·n) (vertical ≡ old ∂x).
        eps_n = 1e-4

        def u_scalar(x):
            return field(p, x[None, :])[0]

        def saltus_at(gp, n):
            g_p = jax.grad(u_scalar)(gp + eps_n * n)
            g_m = jax.grad(u_scalar)(gp - eps_n * n)
            return kappa_p * jnp.dot(g_p, n) - kappa_m * jnp.dot(g_m, n)

        saltus = jax.vmap(saltus_at)(gamma, nrms)
        flux = jnp.mean(saltus**2)
        return phys + lambda_bc * bc + lambda_flux * flux, (phys, bc, flux)

    # Don't JIT the full step — nested AD + optax is heavy; use jax.grad
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    def predict(p):
        return np.asarray(field(p, grid))

    u_init = predict(params)
    rel_init = rel_l2(u_init, data.u_exact)
    base_rel = rel_l2(data.u_base, data.u_exact)

    hist = {
        "loss": [],
        "loss_phys": [],
        "loss_bc": [],
        "rel_l2": [],
        "rel_l2_band": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
        "mean_angle": [],
        "mean_aspect": [],
    }
    diverged = False
    loss0_phys = None
    aniso_cond = float("nan")

    # Conditioning proxy: Gram of anisotropic basis at centers (init)
    try:
        mus_p, w_p, inv_covs = _jax_precompute_params(
            jax,
            jnp,
            params["mus"],
            params["log_sigmas"],
            params["angles"],
            sigma_w * params["w_norm"],
        )
        # Evaluate each basis at all centers → Φ; cond(ΦᵀΦ)
        K = params["mus"].shape[0]
        ones = jnp.ones((K,))
        Phi = []
        for j in range(K):
            col = _jax_fn_evaluate(
                jnp,
                params["mus"],
                params["mus"][j : j + 1],
                ones[j : j + 1],
                inv_covs[j : j + 1],
            )
            Phi.append(col)
        Phi = jnp.stack(Phi, axis=1)
        gram = Phi.T @ Phi
        s = jnp.linalg.svd(gram, compute_uv=False)
        aniso_cond = float(s.max() / jnp.maximum(s.min(), 1e-30))
    except Exception as exc:  # noqa: BLE001
        print(f"    [rbf-shape] cond estimate failed: {exc}", flush=True)

    for s in range(steps):
        t0 = time.perf_counter()
        (loss, (phys, bc, flux)), grads = loss_and_grad(params)
        loss_f = float(loss)
        if not np.isfinite(loss_f):
            diverged = True
            print(f"    [rbf-shape] non-finite loss at step {s}; aborting", flush=True)
            hist["loss"].append(float("nan"))
            hist["loss_phys"].append(float("nan"))
            hist["loss_bc"].append(float("nan"))
            hist["grad_cos"].append(float("nan"))
            hist["step"].append(s)
            hist["step_time"].append(time.perf_counter() - t0)
            hist["rel_l2"].append(float("nan"))
            hist["rel_l2_band"].append(float("nan"))
            hist["mean_angle"].append(float("nan"))
            hist["mean_aspect"].append(float("nan"))
            break
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = {
            **params,
            "mus": jnp.clip(params["mus"], 0.0, 1.0),
            "log_sigmas": jnp.clip(params["log_sigmas"], -8.0, 4.0),
        }
        dt = time.perf_counter() - t0
        if s == 0:
            loss0_phys = float(phys)
        hist["loss"].append(loss_f)
        hist["loss_phys"].append(float(phys))
        hist["loss_bc"].append(float(bc))
        hist["grad_cos"].append(float("nan"))
        hist["step"].append(s)
        hist["step_time"].append(dt)
        sx = np.exp(np.asarray(params["log_sigmas"])[:, 0])
        sy = np.exp(np.asarray(params["log_sigmas"])[:, 1])
        ang = np.asarray(params["angles"])
        aspect = np.maximum(sx, sy) / np.maximum(np.minimum(sx, sy), 1e-30)
        hist["mean_angle"].append(float(np.mean(np.abs(ang))))
        hist["mean_aspect"].append(float(np.mean(aspect)))
        if s % eval_every == 0 or s == steps - 1:
            u_now = predict(params)
            hist["rel_l2"].append(rel_l2(u_now, data.u_exact))
            hist["rel_l2_band"].append(
                rel_l2_masked(u_now, data.u_exact, data.band_mask)
            )
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))
            hist["rel_l2_band"].append(
                hist["rel_l2_band"][-1] if hist["rel_l2_band"] else float("nan")
            )

    # Displacements
    w_eff = sigma_w * np.asarray(params["w_norm"])
    mean_w = float(np.mean(np.abs(w_eff - sigma_w * snap["w_norm"])))
    mean_c = float(
        np.mean(np.linalg.norm(np.asarray(params["mus"]) - snap["mus"], axis=-1))
    )
    dlog = np.abs(np.asarray(params["log_sigmas"]) - snap["log_sigmas"])
    mean_dlog = float(np.mean(dlog))
    mean_dangle = float(np.mean(np.abs(np.asarray(params["angles"]) - snap["angles"])))
    sx = np.exp(np.asarray(params["log_sigmas"])[:, 0])
    sy = np.exp(np.asarray(params["log_sigmas"])[:, 1])
    ang = np.asarray(params["angles"])
    aspect = np.maximum(sx, sy) / np.maximum(np.minimum(sx, sy), 1e-30)
    shape_range = {
        "sigma_x_min": float(sx.min()),
        "sigma_x_max": float(sx.max()),
        "sigma_y_min": float(sy.min()),
        "sigma_y_max": float(sy.max()),
        "angle_min": float(ang.min()),
        "angle_max": float(ang.max()),
        "aspect_max": float(aspect.max()),
        "aspect_mean": float(aspect.mean()),
        "sigma_iso_init": float(1.0 / (data.epsilon * np.sqrt(2.0))),
    }
    collapsed = bool(sx.min() < 1e-3 or sy.min() < 1e-3 or aspect.max() > 1e3)
    shape_moved = mean_dlog > 1e-8 or mean_dangle > 1e-8
    adapted = bool(
        shape_moved
        and (abs(shape_range["aspect_mean"] - 1.0) > 1e-3 or abs(ang).mean() > 1e-3)
    )

    u_final = predict(params)
    final_rel = rel_l2(u_final, data.u_exact)
    delta = base_rel - final_rel
    resid_dec = (
        loss0_phys is not None
        and np.isfinite(hist["loss_phys"][-1])
        and hist["loss_phys"][-1] < loss0_phys
    )
    trained = (
        mean_w > 1e-10 or mean_c > 1e-10 or shape_moved
    ) and resid_dec

    print(
        f"    [rbf-shape init] relL2={rel_init:.4e} (==base {base_rel:.4e}); "
        f"‖u_init−u_base‖/‖u_base‖={init_err:.3e}",
        flush=True,
    )
    print(
        f"    [rbf-shape SHAPES ACTUALLY MOVE] mean|Δw|={mean_w:.4e} "
        f"mean|Δμ|={mean_c:.4e} mean|Δlogσ|={mean_dlog:.4e} "
        f"mean|Δangle|={mean_dangle:.4e}; "
        f"σx∈[{shape_range['sigma_x_min']:.3e},{shape_range['sigma_x_max']:.3e}] "
        f"σy∈[{shape_range['sigma_y_min']:.3e},{shape_range['sigma_y_max']:.3e}] "
        f"angle∈[{shape_range['angle_min']:.3e},{shape_range['angle_max']:.3e}] "
        f"aspect_mean={shape_range['aspect_mean']:.4f} "
        f"adapted={adapted} collapsed={collapsed} shape_moved={shape_moved}",
        flush=True,
    )
    print(
        f"    [rbf-shape sanity] phys {loss0_phys:.4e}→{hist['loss_phys'][-1]:.4e}; "
        f"relL2 {rel_init:.4e}→{final_rel:.4e} (Δ={delta:+.4e}); "
        f"diverged={diverged}; aniso_gram_cond={aniso_cond:.3e}",
        flush=True,
    )
    return _pack_metrics(
        "rbf-shape",
        u_final,
        data,
        hist,
        extra={
            "rel_l2_init": float(rel_init),
            "init_match_rel_err": float(init_err),
            "rel_l2_delta_vs_base": float(delta),
            "beats_base": bool((not diverged) and final_rel < base_rel),
            "mean_weight_displacement": mean_w,
            "mean_center_displacement": mean_c,
            "mean_dlog_sigma": mean_dlog,
            "mean_dangle": mean_dangle,
            "shape_moved": bool(shape_moved),
            "shape_adapted": bool(adapted),
            "shape_range": shape_range,
            "shape_collapsed": bool(collapsed),
            "trained_non_vacuous": bool(trained),
            "diverged": bool(diverged),
            "weight_scale": sigma_w,
            "lr": float(lr),
            "lr_shape": float(lr_shape),
            "lr_centers": float(lr_centers),
            "aniso_cond": float(aniso_cond),
            "phys_loss_initial": float(loss0_phys) if loss0_phys is not None else float("nan"),
            "phys_loss_final": float(hist["loss_phys"][-1]),
        },
    )


# ---------------------------------------------------------------------------
# Neural MLP (Flax)
# ---------------------------------------------------------------------------

class SolutionMLP(nn.Module):
    width: int = 64
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.depth):
            h = nn.Dense(self.width)(h)
            h = nn.tanh(h)
        return nn.Dense(1)(h)[..., 0]


def _mlp_laplacian(apply_fn, params, x):
    def u_scalar(xi):
        return apply_fn(params, xi[None, :])[0]

    def lap_one(xi):
        hess = jax.jacfwd(jax.jacfwd(u_scalar))(xi)
        return hess[0, 0] + hess[1, 1]

    return jax.vmap(lap_one)(x)


def _grad_cosine(g1, g2):
    def flat(tree):
        return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])

    a, b = flat(g1), flat(g2)
    return float(
        np.asarray(
            jnp.dot(a, b)
            / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30)
        )
    )


def train_correction_field(
    data: ExperimentData,
    steps: int = 1500,
    lr: float = 2e-3,
    width: int = 64,
    lambda_anchor: float = 5.0,
    eval_every: int = 50,
    colloc_batch: int = 256,
) -> dict:
    """u = u_base + ê with L[ê] = −(κ-residual); GT soft anchor only."""
    key = jax.random.PRNGKey(data.seed)
    model = SolutionMLP(width=width)
    params = model.init(key, jnp.zeros((1, 2)))["params"]
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    apply = model.apply

    x_gt = jnp.asarray(data.gt_pts)
    u_gt = jnp.asarray(data.u_gt)
    u_base_gt = jnp.asarray(data.u_base[data.gt_idx])
    residual = jnp.asarray(data.residual_base)
    kappa_int = jnp.asarray(data.kappa_interior)
    interior = jnp.asarray(data.interior)
    grid = jnp.asarray(data.grid)
    u_base_grid = data.u_base
    rng = np.random.default_rng(data.seed + 11)
    n_int = len(data.interior)

    hist = {
        "loss": [],
        "loss_pde": [],
        "loss_anchor": [],
        "rel_l2": [],
        "rel_l2_band": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }

    # residual = L[u_base]−f = −κ Δu_base − f
    # Want L[ê] = −residual ⇒ −κ Δê + residual → 0
    def loss_parts(p, idx):
        x = interior[idx]
        res = residual[idx]
        kap = kappa_int[idx]
        lap_e = _mlp_laplacian(lambda pp, xx: apply({"params": pp}, xx), p, x)
        pde = jnp.mean((-kap * lap_e + res) ** 2)
        e_gt = apply({"params": p}, x_gt)
        anchor = jnp.mean((u_base_gt + e_gt - u_gt) ** 2)
        return pde, anchor

    @jax.jit
    def train_step(p, st, idx):
        def total(pp):
            pde, anc = loss_parts(pp, idx)
            return pde + lambda_anchor * anc, (pde, anc)

        (loss, (pde, anc)), grads = jax.value_and_grad(total, has_aux=True)(p)
        updates, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        return p, st, loss, pde, anc, grads

    # Separate grads for cosine (occasional)
    def cos_step(p, idx):
        g_pde = jax.grad(lambda pp: loss_parts(pp, idx)[0])(p)
        g_anc = jax.grad(lambda pp: loss_parts(pp, idx)[1])(p)
        return _grad_cosine(g_pde, g_anc)

    for s in range(steps):
        t0 = time.perf_counter()
        idx = rng.choice(n_int, size=min(colloc_batch, n_int), replace=False)
        idx_j = jnp.asarray(idx)
        params, opt_state, loss, pde, anc, _ = train_step(params, opt_state, idx_j)
        dt = time.perf_counter() - t0
        cos = cos_step(params, idx_j) if (s % eval_every == 0) else float("nan")
        hist["loss"].append(float(loss))
        hist["loss_pde"].append(float(pde))
        hist["loss_anchor"].append(float(anc))
        hist["grad_cos"].append(float(cos) if cos == cos else float("nan"))
        hist["step"].append(s)
        hist["step_time"].append(dt)
        if s % eval_every == 0 or s == steps - 1:
            e_hat = np.asarray(apply({"params": params}, grid))
            u_now = u_base_grid + e_hat
            hist["rel_l2"].append(rel_l2(u_now, data.u_exact))
            hist["rel_l2_band"].append(
                rel_l2_masked(u_now, data.u_exact, data.band_mask)
            )
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))
            hist["rel_l2_band"].append(
                hist["rel_l2_band"][-1] if hist["rel_l2_band"] else float("nan")
            )

    e_final = np.asarray(apply({"params": params}, grid))
    u_final = u_base_grid + e_final
    oracle = (
        "GT used only as soft constraint anchor ‖(u_base+e_hat)−u*‖_GT; "
        "primary target is L[e_hat]=−(κ-residual) with residual=L[u_base]−f "
        "(L=−κΔ). Never trained e_hat←(u*−u_base) as sole regression target."
    )
    print(f"    [correction-field oracle-leak] {oracle}", flush=True)
    return _pack_metrics(
        "correction-field",
        u_final,
        data,
        hist,
        extra={"oracle_leak_audit": oracle, "e_hat": e_final},
    )


def train_surrogate_target(
    data: ExperimentData,
    steps: int = 1500,
    lr: float = 5e-3,
    width: int = 64,
    eval_every: int = 50,
) -> dict:
    """Distill MLP onto analytic projection v*."""
    key = jax.random.PRNGKey(data.seed + 3)
    model = SolutionMLP(width=width)
    params = model.init(key, jnp.zeros((1, 2)))["params"]
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    grid = jnp.asarray(data.grid)
    v_star = jnp.asarray(data.v_star)

    @jax.jit
    def step(p, st):
        def loss_fn(pp):
            return jnp.mean((model.apply({"params": pp}, grid) - v_star) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        return p, st, loss

    hist = {
        "loss": [],
        "rel_l2": [],
        "rel_l2_band": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }
    for s in range(steps):
        t0 = time.perf_counter()
        params, opt_state, loss = step(params, opt_state)
        dt = time.perf_counter() - t0
        hist["loss"].append(float(loss))
        hist["grad_cos"].append(0.0)
        hist["step"].append(s)
        hist["step_time"].append(dt)
        if s % eval_every == 0 or s == steps - 1:
            u_now = np.asarray(model.apply({"params": params}, grid))
            hist["rel_l2"].append(rel_l2(u_now, data.u_exact))
            hist["rel_l2_band"].append(
                rel_l2_masked(u_now, data.u_exact, data.band_mask)
            )
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))
            hist["rel_l2_band"].append(
                hist["rel_l2_band"][-1] if hist["rel_l2_band"] else float("nan")
            )

    u_final = np.asarray(model.apply({"params": params}, grid))
    return _pack_metrics(
        "surrogate-target",
        u_final,
        data,
        hist,
        extra={
            "v_star_rel_l2": rel_l2(data.v_star, data.u_exact),
            "surrogate_info": data.surrogate_info,
        },
    )


def train_compound_loss(
    data: ExperimentData,
    steps: int = 1500,
    lr: float = 2e-3,
    width: int = 64,
    lambda_data: float = 500.0,
    eval_every: int = 50,
    colloc_batch: int = 256,
) -> dict:
    """Compound ‖L[u]−f‖² + λ‖u−u*‖² with grad-conflict cosine."""
    key = jax.random.PRNGKey(data.seed + 5)
    model = SolutionMLP(width=width)
    params = model.init(key, jnp.zeros((1, 2)))["params"]
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    apply = model.apply

    x_gt = jnp.asarray(data.gt_pts)
    u_gt = jnp.asarray(data.u_gt)
    interior = jnp.asarray(data.interior)
    f_int = jnp.asarray(data.f_interior)
    kappa_int = jnp.asarray(data.kappa_interior)
    grid = jnp.asarray(data.grid)
    rng = np.random.default_rng(data.seed + 7)
    n_int = len(data.interior)

    def parts(p, idx):
        x = interior[idx]
        f = f_int[idx]
        kap = kappa_int[idx]
        lap = _mlp_laplacian(lambda pp, xx: apply({"params": pp}, xx), p, x)
        phys = jnp.mean((-kap * lap - f) ** 2)
        data_term = jnp.mean((apply({"params": p}, x_gt) - u_gt) ** 2)
        return phys, data_term

    @jax.jit
    def train_step(p, st, idx):
        def total(pp):
            phys, dat = parts(pp, idx)
            return phys + lambda_data * dat, (phys, dat)

        (loss, (phys, dat)), grads = jax.value_and_grad(total, has_aux=True)(p)
        updates, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        return p, st, loss, phys, dat

    hist = {
        "loss": [],
        "loss_phys": [],
        "loss_data": [],
        "rel_l2": [],
        "rel_l2_band": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }
    for s in range(steps):
        t0 = time.perf_counter()
        idx = rng.choice(n_int, size=min(colloc_batch, n_int), replace=False)
        idx_j = jnp.asarray(idx)
        params, opt_state, loss, phys, dat = train_step(params, opt_state, idx_j)
        dt = time.perf_counter() - t0
        if s % eval_every == 0:
            g_p = jax.grad(lambda pp: parts(pp, idx_j)[0])(params)
            g_d = jax.grad(lambda pp: parts(pp, idx_j)[1])(params)
            cos = _grad_cosine(g_p, g_d)
        else:
            cos = float("nan")
        hist["loss"].append(float(loss))
        hist["loss_phys"].append(float(phys))
        hist["loss_data"].append(float(dat))
        hist["grad_cos"].append(float(cos) if cos == cos else float("nan"))
        hist["step"].append(s)
        hist["step_time"].append(dt)
        if s % eval_every == 0 or s == steps - 1:
            u_now = np.asarray(apply({"params": params}, grid))
            hist["rel_l2"].append(rel_l2(u_now, data.u_exact))
            hist["rel_l2_band"].append(
                rel_l2_masked(u_now, data.u_exact, data.band_mask)
            )
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))
            hist["rel_l2_band"].append(
                hist["rel_l2_band"][-1] if hist["rel_l2_band"] else float("nan")
            )

    u_final = np.asarray(apply({"params": params}, grid))
    return _pack_metrics("compound-loss", u_final, data, hist)
