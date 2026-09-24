"""Faithful port of the ePIL VSD (K,6) anisotropic Gaussian RBF kernel.

Provenance: ePIL-RBF-1/src/models/ours/rbf_model.py (Dr. Torres's ePIL project, work
machine). The kernel MATH is preserved verbatim (see `_jax_precompute_params` /
`_jax_fn_evaluate`). Imports of jax are DEFERRED so this module parses even without jax
installed (lint/CI venv); call `build_jax_kernel_fns()` on the compute host to get real
jitted functions.

TWO DOCUMENTED DEVIATIONS (do not "fix" them away):
1. Weights are NOT tanh'd here (ePIL tanh clamps weights to (-1,1) because its weights are
   network outputs). We return raw weights so a Kansa base (O(1e3..1e6)) is inheritable at
   init; upstream we normalize w_norm to O(1) via a sigma_w factor.
2. The numerical-stability epsilon is caller-supplied and does not change the kernel form.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# ePIL VSD (K,6) anisotropic Gaussian kernel — faithful math
# ---------------------------------------------------------------------------

def _jax_precompute_params(jax, jnp, mus, log_sigmas, angles, weights, epsilon=1e-6):
    """ePIL precompute_params -> (mus, weights, inv_covs). Faithful math (no tanh)."""
    sigmas = jnp.exp(log_sigmas)                 # (K, 2)
    squared_sigmas = sigmas**2
    cos_a = jnp.cos(angles)
    sin_a = jnp.sin(angles)
    R = jnp.stack([
        jnp.stack([cos_a, -sin_a], axis=1),
        jnp.stack([sin_a, cos_a], axis=1),
    ], axis=2)                                    # (K, 2, 2)
    inv_diag = 1.0 / (squared_sigmas + epsilon)   # (K, 2)
    diag_inv = jnp.zeros((mus.shape[0], 2, 2))
    diag_inv = diag_inv.at[jnp.arange(mus.shape[0]), 0, 0].set(inv_diag[:, 0])
    diag_inv = diag_inv.at[jnp.arange(mus.shape[0]), 1, 1].set(inv_diag[:, 1])
    inv_covs = jnp.einsum("kij,kjl,klm->kim", R, diag_inv, R.transpose((0, 2, 1)))
    return mus, weights, inv_covs


def _jax_fn_evaluate(jnp, X, mus, weights, inv_covs):
    """ePIL fn_evaluate -> (N,). phi = exp(-0.5 quad); u = phi . w."""
    diff = X[:, None, :] - mus[None, :, :]        # (N, K, 2)
    quad = jnp.einsum("nki,kij,nkj->nk", diff, inv_covs, diff)
    phi = jnp.exp(-0.5 * quad)
    return jnp.dot(phi, weights)


def build_jax_kernel_fns():
    """Bound to THIS process's jax: return (jit_precompute_params, jit_fn_evaluate)."""
    import jax
    import jax.numpy as jnp

    precompute = jax.jit(lambda m, ls, a, w: _jax_precompute_params(jax, jnp, m, ls, a, w))
    evaluate = jax.jit(lambda X, m, w, ic: _jax_fn_evaluate(jnp, X, m, w, ic))
    return precompute, evaluate


# ---------------------------------------------------------------------------
# Init helper (numpy-only, testable without jax)
# ---------------------------------------------------------------------------

def vsd_from_isotropic_base(
    centers: np.ndarray,
    sigma_w: float,
    epsilon: float,
    weights_base: np.ndarray | None = None,
) -> dict:
    """VSD latent params whose field reproduces the isotropic Kansa base at init.

    Returns {'mus': (K,2), 'log_sigmas': (K,2), 'angles': (K,), 'w_norm': (K,) or None}.
    Isotropic base gaussian exp(-(eps r)^2) vs ePIL kernel exp(-0.5 r^2/sigma^2): match
    requires 0.5/sigma^2 = eps^2  =>  sigma = 1/(sqrt(2) eps).  w_norm = base/sigma_w (O(1)).
    """
    mus = np.asarray(centers, dtype=np.float64)
    sigma_iso = 1.0 / (np.sqrt(2.0) * epsilon)
    log_sigmas = np.log(np.full(mus.shape, sigma_iso))
    angles = np.zeros(mus.shape[0])
    w_norm = (np.asarray(weights_base, dtype=np.float64) / float(sigma_w)
              if weights_base is not None else None)
    return {"mus": mus, "log_sigmas": log_sigmas, "angles": angles, "w_norm": w_norm}


# ---------------------------------------------------------------------------
# OPERATOR BUILDING — implemented in common.py / arms.py, NOT here. Guidance:
# One-shot Kansa on the kappa-Poisson operator needs the PER-BASIS collocation matrix
#   A[i,j] = -kappa(x_i) * Delta phi_j(x_i)   (piecewise-constant kappa: -kappa Laplacian)
# plus interface rows. Build with JAX autograd over the single-kernel basis
#   phi_j(x) = evaluate(x, mus[j:j+1], ones(1), inv_covs[j:j+1])
# via jax.jacrev(jax.jacrev(...)) / jax.hessian summed for the Laplacian, vmap over j and x.
# VERIFY numerically against the isotropic closed form (gaussian_rbf Laplacian =
# (4 eps^4 r^2 - 4 eps^2) e^{-eps^2 r^2}). Report the check that passed.
# Gradient arms (rbf-grad / rbf-shape) use jax.grad of the residual loss w.r.t. params —
# the field building blocks above are sufficient.
# ---------------------------------------------------------------------------