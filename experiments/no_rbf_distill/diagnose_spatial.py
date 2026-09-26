#!/usr/bin/env python3
"""Spatial correlation + map diagnostics for Phase 2 (stochastic-NO uncertainty).

Assesses, in SPACE, how the ensemble uncertainty (σ), the physics residual (|R|),
and the TRUE corrected error (|u_corr − u*|) relate to each other, and how the
per-arm kernel allocations look. Correlation (Pearson on interior points) is
reported FIRST, then full-map figures per seed.

Reuses the Phase-2 pipeline builders (run_pipeline_stoch) and shared kernel math.
numpy + matplotlib only (run with sibling venv that has matplotlib).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import (
    STOCH_ARM_NAMES,
    DEFAULT_KAPPA_JUMP,
    DEFAULT_KAPPA_M,
    DEFAULT_RESOLUTION,
    boundary_points,
    f_exact,
    fit_error_pde_corrector,
    flux_jump_of_field,
    gamma_normals,
    gamma_points,
    interface_band_mask,
    interior_away_from_gamma,
    kappa_of_points,
    make_grid,
    pde_residual_fd,
    rbf_eval,
    rel_l2,
    reshape_field,
    u_exact,
)
from run_pipeline_stoch import (
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_ensemble,
    place_centers_stoch,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ac = a - a.mean()
    bc = b - b.mean()
    d = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    return float(np.dot(ac, bc) / d) if d > 1e-30 else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    r_a = np.argsort(np.argsort(a)).astype(np.float64)
    r_b = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(r_a, r_b)


def analyze_seed(
    seed: int,
    resolution: int,
    k: int,
    epsilon: float,
    kappa_m: float,
    kappa_p: float,
    gamma: str,
    n_boundary: int,
    n_gamma: int,
    lambda_unc: float,
    ens_m: int,
    top_frac: float,
) -> dict:
    grid, X, Y = make_grid(resolution)
    h = float(X[0, 1] - X[0, 0])
    ue = u_exact(grid, kappa_m, kappa_p, gamma=gamma)
    fe = f_exact(grid, kappa_m, kappa_p, gamma=gamma)
    kap = kappa_of_points(grid, kappa_m, kappa_p, gamma=gamma)
    band = interface_band_mask(grid, band=0.05, gamma=gamma)

    ens = ensure_ensemble(
        seed=seed,
        data_dir=Path(__file__).resolve().parent / "data",
        force=False,
        m=ens_m,
        cfg=__import__("no_model").TinyNOConfig(),
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        gamma=gamma,
    )[0]

    u_no, sigma = ens.predict_mean_std(grid)
    model_predict = ens.predict  # used for boundary/flux of the mean solution
    u_grid = reshape_field(u_no, resolution)
    f_grid = reshape_field(fe, resolution)
    k_grid = reshape_field(kap, resolution)
    R_grid = pde_residual_fd(u_grid, f_grid, k_grid, h)
    R_flat = R_grid.ravel()
    sigma_flat = np.asarray(sigma, dtype=np.float64).ravel()

    interior_idx = interior_away_from_gamma(grid, gamma=gamma)
    interior = grid[interior_idx]

    # True pre-correction error of the ensemble mean
    err_pre = np.abs(u_no - ue).ravel()

    # Fields on interior points only (avoid boundary FD zeros in R)
    Ri = R_flat[interior_idx]
    Si = sigma_flat[interior_idx]
    Ei = err_pre[interior_idx]

    corr = {
        "pearson_sigma_absR": pearson(Si, np.abs(Ri)),
        "spearman_sigma_absR": spearman(Si, np.abs(Ri)),
        "pearson_sigma_errmean": pearson(Si, Ei),
        "spearman_sigma_errmean": spearman(Si, Ei),
        "pearson_absR_errmean": pearson(np.abs(Ri), Ei),
        "spearman_absR_errmean": spearman(np.abs(Ri), Ei),
    }

    # Top-set overlap
    n = len(interior_idx)
    kt = max(1, int(np.ceil(top_frac * n)))
    top_s = np.argpartition(Si, -kt)[-kt:]
    top_r = np.argpartition(np.abs(Ri), -kt)[-kt:]
    top_e = np.argpartition(Ei, -kt)[-kt:]
    overlap = {
        "top_frac": top_frac,
        "frac_top_sigma_in_top_err": float(np.mean(np.isin(top_s, top_e))),
        "frac_top_R_in_top_err": float(np.mean(np.isin(top_r, top_e))),
        "frac_top_sigma_in_top_R": float(np.mean(np.isin(top_s, top_r))),
    }

    # Per-arm corrected fields (reuse the same error-PDE corrector as the pipeline)
    bnd = boundary_points(n_boundary)
    gamma_pts = gamma_points(n_gamma, gamma=gamma)
    normals = gamma_normals(gamma_pts, gamma=gamma)
    u_no_bnd = model_predict(bnd)
    u_bnd_exact = u_exact(bnd, kappa_m, kappa_p, gamma=gamma)
    v_bnd = u_bnd_exact - u_no_bnd
    jump_no = flux_jump_of_field(model_predict, gamma_pts, normals, kappa_m, kappa_p)
    flux_rhs = -jump_no

    arms: dict[str, dict] = {}
    for arm in STOCH_ARM_NAMES:
        centers = place_centers_stoch(
            arm, k, seed, grid, R_flat, sigma_flat, interior_idx, lambda_unc
        )
        weights, _info = fit_error_pde_corrector(
            centers=centers,
            interior=interior,
            residual_int=R_flat[interior_idx],
            kappa_int=kap[interior_idx],
            gamma_pts=gamma_pts,
            boundary_pts=bnd,
            v_bnd=v_bnd,
            epsilon=epsilon,
            kappa_m=kappa_m,
            kappa_p=kappa_p,
            gamma=gamma,
            seed=seed,
            flux_jump_rhs=flux_rhs,
        )
        v = rbf_eval(grid, centers, weights, epsilon)
        u_corr = u_no + v
        e = np.abs(u_corr - ue).ravel()
        arms[arm] = {
            "centers": centers,
            "err": e,
            "rel_l2": float(rel_l2(u_corr, ue)),
            "pearson_sigma_err": float(pearson(sigma_flat[interior_idx], e[interior_idx])),
            "pearson_absR_err": float(pearson(np.abs(Ri), e[interior_idx])),
        }

    fig = make_figure(seed, X, Y, resolution, R_grid, sigma_flat, arms, corr, overlap, u_no, ue)
    out = FIGURES_DIR / f"spatial_corr_seed{seed}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rec = {"seed": seed, "u_no_rel_l2": float(rel_l2(u_no, ue)), "corr": corr,
           "overlap": overlap, "arms": {a: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else None)
                                            for kk, vv in arms[a].items() if kk != "centers"}
                                        for a in arms},
           "figure": str(out)}
    return rec


def make_figure(seed, X, Y, res, R_grid, sigma_flat, arms, corr, overlap, u_no, ue):
    S = reshape_field(sigma_flat, res)
    labels = {
        "residual_only": "residual-only",
        "residual_unc": "residual+σ",
        "shuffled_unc": "shuffled-σ",
    }
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), constrained_layout=True)
    # Row 0: physics residual, sigma, pre-correction error, then correlation summary
    ax = axes[0, 0]
    pcm = ax.pcolormesh(X, Y, np.abs(R_grid), shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("|physics residual|\n|R(u_NO)|")
    ax.set_aspect("equal")
    ax = axes[0, 1]
    pcm = ax.pcolormesh(X, Y, S, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"uncertainty σ (ens std)\ncorr(σ,|R|)={corr['pearson_sigma_absR']:.2f}")
    ax.set_aspect("equal")
    ax = axes[0, 2]
    pcm = ax.pcolormesh(X, Y, np.abs(u_no - ue).reshape(res, res), shading="auto", cmap="magma")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("|pre-corr error|\n|u_NO−u*|")
    ax.set_aspect("equal")
    ax = axes[0, 3]
    ax.axis("off")
    ax.text(0.0, 0.98, "Correlations (interior)", fontsize=10, va="top", fontweight="bold")
    txt = (
        f"σ↔|R|   P={corr['pearson_sigma_absR']:+.2f}  ρ={corr['spearman_sigma_absR']:+.2f}\n"
        f"σ↔err  P={corr['pearson_sigma_errmean']:+.2f}  ρ={corr['spearman_sigma_errmean']:+.2f}\n"
        f"|R|↔err P={corr['pearson_absR_errmean']:+.2f}  ρ={corr['spearman_absR_errmean']:+.2f}\n"
        f"\ntop-{int(overlap['top_frac']*100)}% overlap:\n"
        f"  σ∩err={overlap['frac_top_sigma_in_top_err']:.2f}\n"
        f"  |R|∩err={overlap['frac_top_R_in_top_err']:.2f}\n"
        f"  σ∩|R|={overlap['frac_top_sigma_in_top_R']:.2f}"
    )
    ax.text(0.0, 0.82, txt, fontsize=9, va="top", family="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Row 1: centers per arm on |R| (cols 0-2), then corrected-error maps + |R|+σ blend visually
    for j, arm in enumerate(STOCH_ARM_NAMES):
        ax = axes[1, j]
        pcm = ax.pcolormesh(X, Y, np.abs(R_grid), shading="auto", alpha=0.85)
        c = arms[arm]["centers"]
        ax.scatter(c[:, 0], c[:, 1], s=7, c="k", marker=".", linewidths=0)
        ax.set_title(f"{labels[arm]} centers on |R|")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if j == 2:
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    # cols 3 = residual_only corrected-error map (the arm that wins Phase 1)
    ax = axes[1, 3]
    e0 = arms["residual_only"]["err"].reshape(res, res)
    pcm = ax.pcolormesh(X, Y, e0, shading="auto", cmap="magma")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"|corr err| residual-only\nrel-L2={arms['residual_only']['rel_l2']:.3e}\n"
                 f"corr(σ,err)={arms['residual_only']['pearson_sigma_err']:+.2f}")
    ax.set_aspect("equal")

    fig.suptitle(
        f"Phase 2 seed={seed} — uncertainty vs residual vs error (spatial correlation)\n"
        f"u_NO rel-L2={rel_l2(u_no, ue):.3e}  "
        f"resid_only={arms['residual_only']['rel_l2']:.3e}  resid+σ={arms['residual_unc']['rel_l2']:.3e}  "
        f"shuff-σ={arms['shuffled_unc']['rel_l2']:.3e}",
        fontsize=11,
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--K", type=int, default=64, dest="k")
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--kappa-jump", type=float, default=10.0)
    ap.add_argument("--kappa-m", type=float, default=1.0)
    ap.add_argument("--gamma", type=str, default="vertical")
    ap.add_argument("--lambda-unc", type=float, default=1.0)
    ap.add_argument("--ensemble-m", type=int, default=3)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--n-boundary", type=int, default=40)
    ap.add_argument("--n-gamma", type=int, default=40)
    args = ap.parse_args()

    from common import default_epsilon
    eps = float(args.epsilon) if args.epsilon is not None else default_epsilon(args.k)
    kappa_p = float(args.kappa_m) * float(args.kappa_jump)

    records = []
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        rec = analyze_seed(s, args.resolution, args.k, eps, args.kappa_m, kappa_p,
                           args.gamma, args.n_boundary, args.n_gamma, args.lambda_unc,
                           args.ensemble_m, args.top_frac)
        records.append(rec)

    print("\n=== spatial correlation (per seed, interior) ===")
    hdr = f"{'seed':>4} {'σ↔|R|(P)':>10} {'σ↔|R|(ρ)':>10} {'σ↔err(P)':>10} {'σ↔err(ρ)':>10} {'|R|↔err(P)':>11} {'|R|↔err(ρ)':>11}"
    print(hdr)
    for r in records:
        c = r["corr"]
        print(f"{r['seed']:>4} {c['pearson_sigma_absR']:>10.3f} {c['spearman_sigma_absR']:>10.3f} "
              f"{c['pearson_sigma_errmean']:>10.3f} {c['spearman_sigma_errmean']:>10.3f} "
              f"{c['pearson_absR_errmean']:>11.3f} {c['spearman_absR_errmean']:>11.3f}")

    # Means across seeds
    import statistics as st
    mean = lambda kk: st.mean([r["corr"][kk] for r in records])
    print("\nmean over seeds:")
    for kk in ("pearson_sigma_absR", "spearman_sigma_absR", "pearson_sigma_errmean",
               "spearman_sigma_errmean", "pearson_absR_errmean", "spearman_absR_errmean"):
        print(f"  {kk}: {mean(kk):+.3f}")

    print("\n=== top-set overlap (mean) ===")
    for kk in ("frac_top_sigma_in_top_err", "frac_top_R_in_top_err", "frac_top_sigma_in_top_R"):
        print(f"  {kk}: {st.mean([r['overlap'][kk] for r in records]):.3f}")

    out = RESULTS_DIR / "spatial_corr.json"
    out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"\n[wrote] {out}")
    for r in records:
        print(f"[fig] {r['figure']}")


if __name__ == "__main__":
    main()