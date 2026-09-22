"""Shared JAX model + target builders for splat vs additive-RBF correctability.

Ported from Edgar's gs-rbf.py toy into pure JAX (no optax):
two Gaussian-basis representations (mode in {"rbf","splat"}) share
centers/scales/rotation/basis; the ONLY difference is the compositing
(additive vs alpha/transmittance). Signs allowed (signed scalar fields).

Design notes:
  * `mode` is passed explicitly (not stored in the param pytree), so the pytree
    handed to jax.value_and_grad/grad is all arrays and differentiates cleanly.
  * Everything is a pure function of the param pytree; jax.jit/grad/vmap compose.
  * Training loop uses an in-python Adam (no optax dependency).
"""
import functools
import math
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

MODES = ("rbf", "splat")


# ----------------------------------------------------------------------------
# Target fields (synthetic, signed, smooth)
# ----------------------------------------------------------------------------
def make_grid(h, w):
    """Return stacked HxWx2 coordinate grid (x, y) in -1..1, aspect-preserving."""
    xs = jnp.linspace(-1.0, 1.0, w)
    ys = jnp.linspace(-1.0, 1.0, h)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([xx, yy], axis=-1)              # H,W,2


def build_target(name="blob", h=128, w=128, seed=0):
    grid = make_grid(h, w)
    xx, yy = grid[..., 0], grid[..., 1]
    if name == "blob":
        f = jnp.exp(-(((xx - 0.3) / 0.2) ** 2 + (yy / 0.35) ** 2))
        f += 0.7 * jnp.exp(-(((xx + 0.4) / 0.25) ** 2 + ((yy + 0.2) / 0.15) ** 2))
        f -= 0.4 * jnp.exp(-((xx / 0.15) ** 2 + ((yy - 0.4) / 0.2) ** 2))
        return jnp.stack([f, jnp.rot90(f), f * 0.8], axis=-1)
    if name == "oscillatory":
        f = jnp.sin(4.0 * math.pi * xx) * jnp.cos(3.0 * math.pi * yy)
        return jnp.stack([f, jnp.ones_like(f) - f, f], axis=-1)
    if name == "multiscale":
        f = (jnp.sin(6 * xx) * jnp.cos(5 * yy) * 0.4
             + jnp.exp(-(xx ** 2 + yy ** 2) / 0.1)
             - 0.3 * jnp.exp(-((xx - 0.6) ** 2 + (yy + 0.5) ** 2) / 0.05))
        f = (f - f.min()) / (f.max() - f.min())
        return jnp.stack([f, f, f], axis=-1)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Parameter init — identical geometric init across modes (fair comparison)
# ----------------------------------------------------------------------------
def init_params(target, grid, n_gaussians=250, mode="splat", init_sigma=0.06, seed=0):
    rng = np.random.default_rng(seed)
    H, W = target.shape[:2]

    flat_idx = rng.integers(0, H * W, size=n_gaussians)
    py, px = flat_idx // W, flat_idx % W
    grid_np = np.asarray(grid)                       # H,W,2
    centers = grid_np[py, px].copy() + 0.01 * rng.standard_normal((n_gaussians, 2))
    centers = centers.astype(np.float32)

    inv_sp = lambda z: float(math.log(math.expm1(max(z, 1e-8))))
    init_scale = np.full((n_gaussians, 2), init_sigma, dtype=np.float32)
    log_scales = np.array([inv_sp(s - 1e-3) for s in init_scale.ravel()],
                          dtype=np.float32).reshape(n_gaussians, 2)
    angles = (2 * math.pi * rng.random(n_gaussians)).astype(np.float32)

    params = {
        "centers": jnp.asarray(centers),
        "log_scales": jnp.asarray(log_scales),
        "angles": jnp.asarray(angles),
    }

    if mode == "splat":
        tgt_np = np.asarray(target)
        init_colors = tgt_np[py, px].clip(0.02, 0.98)
        params["color_logits"] = jnp.asarray(_logit(init_colors.astype(np.float32)))
        params["opacity_logits"] = jnp.asarray(np.full(
            (n_gaussians,), math.log(0.25 / 0.75), dtype=np.float32))
        params["bg_logits"] = jnp.asarray(_logit(
            np.asarray(target.mean(axis=(0, 1))).clip(0.02, 0.98).astype(np.float32)))
    else:
        params["weights"] = jnp.asarray((0.01 * rng.standard_normal((n_gaussians, 3))).astype(np.float32))
        params["bias"] = jnp.asarray(np.asarray(target.mean(axis=(0, 1)), dtype=np.float32))
    return params


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x) - np.log1p(-x)


def scales_of(params):
    return jax.nn.softplus(params["log_scales"]) + 1e-3


# ----------------------------------------------------------------------------
# Forward field — pure function of the param pytree
# ----------------------------------------------------------------------------
def field(params, grid, mode="splat"):
    """Return HxWx3 field for either mode. JIT-compatible (grid static buffer)."""
    X = grid[..., 0][None]
    Y = grid[..., 1][None]
    cx = params["centers"][:, 0, None, None]
    cy = params["centers"][:, 1, None, None]
    dx, dy = X - cx, Y - cy
    theta = params["angles"][:, None, None]
    c, s = jnp.cos(theta), jnp.sin(theta)
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    sx = scales_of(params)[:, 0, None, None]
    sy = scales_of(params)[:, 1, None, None]
    q = (xr / sx) ** 2 + (yr / sy) ** 2
    G = jnp.exp(-0.5 * q)                                     # N,H,W

    if mode == "rbf":
        image = jnp.einsum("nhw,nc->hwc", G, params["weights"])
        return image + params["bias"]

    # splat: alpha compositing with front-to-back transmittance
    colors = jax.nn.sigmoid(params["color_logits"])
    opacity = jax.nn.sigmoid(params["opacity_logits"])[:, None, None]
    alpha = (opacity * G).clip(max=0.995)
    om = (1.0 - alpha).clip(min=1e-6)
    prefix = jnp.concatenate([jnp.ones_like(om[:1]), om], axis=0)
    T = jnp.cumprod(prefix, axis=0)[:-1]
    image = jnp.einsum("nhw,nc->hwc", T * alpha, colors)
    T_end = jnp.prod(om, axis=0)
    image = image + T_end[..., None] * jax.nn.sigmoid(params["bg_logits"])
    return image


def mse_loss(params, grid, target, mode="splat"):
    pred = field(params, grid, mode=mode)
    return jnp.mean((pred - target) ** 2)


def make_loss_and_grad(mode):
    return jax.value_and_grad(functools.partial(mse_loss, mode=mode))


# ----------------------------------------------------------------------------
# In-python Adam (pure jax.grad; no optax dependency)
# ----------------------------------------------------------------------------
def adam_step(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    def upd(p, g, pm, pv):
        mnew = b1 * pm + (1 - b1) * g
        vnew = b2 * pv + (1 - b2) * g ** 2
        mhat = mnew / (1 - b1 ** t)
        vhat = vnew / (1 - b2 ** t)
        return p - lr * mhat / (jnp.sqrt(vhat) + eps), mnew, vnew

    new_params, new_m, new_v = {}, {}, {}
    for k in params:
        np_, nm, nv = upd(params[k], grads[k], m[k], v[k])
        new_params[k], new_m[k], new_v[k] = np_, nm, nv
    return new_params, new_m, new_v


# ----------------------------------------------------------------------------
# Closed-form additive residual correction probe (RBF only, basis FROZEN)
# ----------------------------------------------------------------------------
def rbf_residual_correction(params, grid, target):
    """Freeze geometry; solve linear least-squares for the weight correction.

    Coarse u = B w + b (B = frozen Gaussian fields). Target residual r = t - u
    lives in span(B) for RBF, so a one-shot correction adds cleanly.
    Returns (corrected_field, relL2_before, relL2_after, rank).
    """
    G = field_to_basis(params, grid)                      # N,H,W
    H, W = target.shape[:2]
    N = params["centers"].shape[0]

    u = jnp.einsum("nhw,nc->hwc", G, params["weights"]) + params["bias"]
    rel_before = jnp.linalg.norm(u - target) / (jnp.linalg.norm(target) + 1e-8)

    B = G.reshape(N, H * W).T                            # (HW,N)
    y = (target - params["bias"]).reshape(H * W, 3)
    w, residuals, rank, _ = jnp.linalg.lstsq(B, y)
    u_corr = jnp.einsum("nhw,nc->hwc", G, w) + params["bias"]
    rel_after = jnp.linalg.norm(u_corr - target) / (jnp.linalg.norm(target) + 1e-8)
    return u_corr, float(rel_before), float(rel_after), int(rank)


def field_to_basis(params, grid):
    """Expose the Gaussian fields G (N,H,W) for the linear probe (rbf gate)."""
    X = grid[..., 0][None]
    Y = grid[..., 1][None]
    cx = params["centers"][:, 0, None, None]
    cy = params["centers"][:, 1, None, None]
    dx, dy = X - cx, Y - cy
    theta = params["angles"][:, None, None]
    c, s = jnp.cos(theta), jnp.sin(theta)
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    sx = scales_of(params)[:, 0, None, None]
    sy = scales_of(params)[:, 1, None, None]
    q = (xr / sx) ** 2 + (yr / sy) ** 2
    return jnp.exp(-0.5 * q)