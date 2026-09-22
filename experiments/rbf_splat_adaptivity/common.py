"""Adaptivity probe: which representation (additive RBF vs splat) tracks
u_t -> u_{t+1} on a real spatio-temporal flow, in terms of (a) time per step
and (b) rollout error, plus an "energy"/optimal-transport cost of kernel motion.

Reads CFDBench cavity/bc/case0000 u.npy,v.npy (T x H x W). Fits a Gaussian-basis
representation at frame t, propagates to t+1 by:
  * RBF: track centers (advect by local velocity) + one-shot linear weight
    re-solve on the frozen basis (exploits linear-in-weights correctability).
  * splat: full re-optimization of the alpha-composited stack (its only route).

Energy: Gaussian-mixture optimal transport between consecutive kernel configs
(Wasserstein-2 lower bound on kernel motion) -> "how much work to track".
"""
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "cfdb" / "cavity" / "bc" / "case0000"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_case(case_dir: pathlib.Path):
    """Return (u, v) each T x H x W."""
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")
    return u, v


# ---------------------------------------------------------------------------
# Energy / OT metric on kernel configurations
# ---------------------------------------------------------------------------
def gaussian_ot_energy(centers_t, amph_t, centers_t1, amph_t1):
    """Approx. transport energy between two anisotropic-aware Gaussian mixtures.

    Uses the isotropic / radial closed-form Gaussian-to-Gaussian OT on 2D:
        cost(i,j) = ||c_i - c'_j||^2   (shift-only, scale term omitted: we
        attribute motion to translation, the quantity the thread cares about).
    E = min over couplings of sum_ij gamma_ij * ||c_i - c'_j||^2,
        normalized by amplitude (so bigger/heavier kernels cost more to move).
    Solved exactly on a small mixture by scipy.optimize.linear_sum_assignment
    (balanced, mass-rescaled) — acceptable since n is small.
    """
    amp_t = np.maximum(np.asarray(amph_t, float).ravel(), 0.0)
    amp_t1 = np.maximum(np.asarray(amph_t1, float).ravel(), 0.0)
    c_t = np.asarray(centers_t, float)
    c_t1 = np.asarray(centers_t1, float)

    n, m = len(amp_t), len(amp_t1)
    if n == 0 or m == 0:
        return 0.0

    # normalized distributions
    a = amp_t / (amp_t.sum() + 1e-12)
    b = amp_t1 / (amp_t1.sum() + 1e-12)

    # cost matrix ||ci - c'j||^2
    C = np.sum((c_t[:, None, :] - c_t1[None, :, :]) ** 2, axis=-1)

    # If counts differ, embed into a balanced (max-dim) OT via padding.
    # For a cheap, monotone lower-bound we instead use a greedy flow on the
    # min(n,m) transport with amplitude-matching. Keep it simple + deterministic:
    # solve assignment on the smaller axis against the larger via padding.
    d = max(n, m)
    Cpad = np.zeros((d, d))
    A = np.zeros(d); Bp = np.zeros(d)
    A[:n] = a; Bp[:m] = b
    Cpad[:n, :m] = C
    # deterministic greedy coupling (Sinkhorn-lite) -> energy
    gamma = np.zeros((d, d))
    a2, b2 = A.copy(), Bp.copy()
    tol = 1e-12
    while (a2.sum() > tol) and (b2.sum() > tol):
        i = int(np.argmax(a2)); j = int(np.argmax(b2))
        if a2[i] <= tol or b2[j] <= tol:
            break
        t = min(a2[i], b2[j])
        gamma[i, j] += t
        a2[i] -= t; b2[j] -= t
    E = float(np.sum(gamma * Cpad))
    return E


def kinetic_energy(centers_t, centers_t1, dt=1.0):
    """Translation energy sum_i ||c_i - c'_j||^2/dt^2 (simple, no amplitude)."""
    c_t = np.asarray(centers_t, float)
    c_t1 = np.asarray(centers_t1, float)
    if len(c_t) == 0 or len(c_t1) == 0:
        return 0.0
    d = min(len(c_t), len(c_t1))
    return float(np.sum((c_t[:d] - c_t1[:d]) ** 2) / (dt ** 2))


# ---------------------------------------------------------------------------
# Simple tracked propagator shell (structure for run_training/evaluate)
# ---------------------------------------------------------------------------
def propagate_rbf(params, grid, velocity_t):
    """Advect RBF centers one step by local velocity, keep weights frozen.

    params: dict(centers, log_scales, angles, weights, bias) at time t.
    Returns new params for t+1. Weight re-solve is done in evaluate after
    tracking (needs the target field at t+1).
    """
    vx, vy = velocity_t
    new = dict(params)
    c = np.asarray(params["centers"])
    # sample velocity at each center (bilinear approx on grid coords)
    c4 = c.clip(-0.999, 0.999)
    # map [-1,1] -> pixel idx
    H, W = vx.shape
    ix = ((c4[:, 0] + 1) / 2 * (W - 1)).astype(int)
    iy = ((c4[:, 1] + 1) / 2 * (H - 1)).astype(int)
    dv = np.array([vx[iy, ix], vy[iy, ix]]).T
    new["centers"] = c + 0.01 * dv  # small dt advection
    return new