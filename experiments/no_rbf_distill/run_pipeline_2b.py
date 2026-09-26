#!/usr/bin/env python3
"""Phase 2b: complementary-σ controls (σ-only + residual/σ split subsets).

Does NOT modify Phase-1/2 paths. Reuses cached ensembles via
``run_pipeline_stoch.ensure_ensemble``. Figures default OFF (disk-safe).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_GAMMA,
    DEFAULT_K,
    DEFAULT_KAPPA_JUMP,
    DEFAULT_KAPPA_M,
    DEFAULT_RESOLUTION,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_ALPHA,
    STOCH_ARM_NAMES2,
    allocate_residual,
    allocate_residual_split_sigma,
    allocate_sigma,
    boundary_points,
    default_epsilon,
    f_exact,
    fit_error_pde_corrector,
    flux_jump_of_field,
    frac_centers_in_top_error,
    gamma_normals,
    gamma_points,
    interface_band_mask,
    interior_away_from_gamma,
    interior_for_allocation,
    kappa_of_points,
    make_grid,
    pde_residual_fd,
    rbf_eval,
    rel_l2,
    rel_l2_masked,
    reshape_field,
    sigma_residual_overlap_diagnostics,
    u_exact,
)
from no_model import DATA_DIR, ENSEMBLE_SIZE, TinyNOConfig
from run_pipeline_stoch import ensure_ensemble

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

_MPLSTYLE = Path(
    "/home/etorres/Documents/github/personal/research-project-pinns"
    "/style/pitayasmoothie-light.mplstyle"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    p.add_argument("--K", type=int, default=DEFAULT_K, dest="k")
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--kappa-jump", type=float, default=DEFAULT_KAPPA_JUMP)
    p.add_argument("--kappa-m", type=float, default=DEFAULT_KAPPA_M)
    p.add_argument("--gamma", type=str, default=DEFAULT_GAMMA)
    p.add_argument("--n-boundary", type=int, default=40)
    p.add_argument("--n-gamma", type=int, default=40)
    p.add_argument("--alpha", type=float, default=DEFAULT_SPLIT_ALPHA,
                   help="σ-budget fraction for residual_split_sigma (default 0.25)")
    p.add_argument("--ensemble-m", type=int, default=ENSEMBLE_SIZE)
    p.add_argument("--n-poly", type=int, default=2)
    p.add_argument("--n-sine", type=int, default=1)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--force",
        action="store_true",
        help="Retrain ensembles even if no_ens_seed*.npz cache exists",
    )
    # Figures default OFF: work host has a full disk and may lack matplotlib.
    p.add_argument(
        "--skip-figures",
        action="store_true",
        default=True,
        help="Skip diagnostic figures (default: on)",
    )
    p.add_argument(
        "--make-figures",
        action="store_false",
        dest="skip_figures",
        help="Write diagnostic figures (requires matplotlib)",
    )
    return p.parse_args()


def place_centers_2b(
    arm: str,
    k: int,
    seed: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    """Return (centers, subset_meta). subset_meta describes residual vs σ subsets."""
    if arm == "residual_only":
        centers = allocate_residual(k, grid, residual_flat, interior_idx, seed)
        return centers, {
            "alpha": 0.0,
            "n_residual": int(k),
            "n_sigma": 0,
            "split_index": int(k),
        }
    if arm == "sigma_only":
        centers = allocate_sigma(k, grid, sigma_flat, interior_idx, seed)
        return centers, {
            "alpha": 1.0,
            "n_residual": 0,
            "n_sigma": int(k),
            "split_index": 0,
        }
    if arm == "residual_split_sigma":
        return allocate_residual_split_sigma(
            k,
            grid,
            residual_flat,
            sigma_flat,
            interior_idx,
            seed,
            alpha=alpha,
        )
    raise ValueError(f"unknown Phase-2b arm {arm!r}")


def _subset_overlap(
    centers: np.ndarray,
    meta: dict,
    grid_cand: np.ndarray,
    err_cand: np.ndarray,
    top_frac: float = 0.1,
) -> dict[str, float]:
    """Fraction of residual- / σ-subset centers in top-frac true-error sites."""
    split = int(meta["split_index"])
    n_r = int(meta["n_residual"])
    n_s = int(meta["n_sigma"])
    c_r = centers[:split] if n_r > 0 else np.zeros((0, 2), dtype=np.float64)
    c_s = centers[split:] if n_s > 0 else np.zeros((0, 2), dtype=np.float64)
    return {
        "frac_residual_centers_in_top_err": frac_centers_in_top_error(
            c_r, grid_cand, err_cand, top_frac=top_frac
        )
        if n_r > 0
        else float("nan"),
        "frac_sigma_centers_in_top_err": frac_centers_in_top_error(
            c_s, grid_cand, err_cand, top_frac=top_frac
        )
        if n_s > 0
        else float("nan"),
        "frac_all_centers_in_top_err": frac_centers_in_top_error(
            centers, grid_cand, err_cand, top_frac=top_frac
        ),
        "top_err_frac": float(top_frac),
    }


def run_seed(
    seed: int,
    resolution: int,
    k: int,
    epsilon: float,
    kappa_m: float,
    kappa_p: float,
    gamma: str,
    n_boundary: int,
    n_gamma: int,
    alpha: float,
    ens_m: int,
    cfg: TinyNOConfig,
    data_dir: Path,
    force: bool,
    make_fig: bool,
) -> dict:
    ens, ens_meta = ensure_ensemble(
        seed=seed,
        data_dir=data_dir,
        force=force,
        m=ens_m,
        cfg=cfg,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        gamma=gamma,
    )

    grid, X, Y = make_grid(resolution)
    h = float(X[0, 1] - X[0, 0])
    ue = u_exact(grid, kappa_m, kappa_p, gamma=gamma)
    fe = f_exact(grid, kappa_m, kappa_p, gamma=gamma)
    kap = kappa_of_points(grid, kappa_m, kappa_p, gamma=gamma)

    u_no, sigma = ens.predict_mean_std(grid)
    u_no_rel = rel_l2(u_no, ue)
    abs_err_no = np.abs(u_no - ue)

    u_grid = reshape_field(u_no, resolution)
    f_grid = reshape_field(fe, resolution)
    k_grid = reshape_field(kap, resolution)
    R_grid = pde_residual_fd(u_grid, f_grid, k_grid, h)
    R_flat = R_grid.ravel()
    sigma_flat = np.asarray(sigma, dtype=np.float64).ravel()

    interior_idx = interior_away_from_gamma(grid, gamma=gamma)
    interior = grid[interior_idx]
    residual_int = R_flat[interior_idx]
    kappa_int = kap[interior_idx]
    alloc_idx = interior_for_allocation(grid)
    grid_cand = grid[alloc_idx]
    err_cand = abs_err_no[alloc_idx]

    overlap = sigma_residual_overlap_diagnostics(R_flat, sigma_flat, alloc_idx)

    bnd = boundary_points(n_boundary)
    gamma_pts = gamma_points(n_gamma, gamma=gamma)
    normals = gamma_normals(gamma_pts, gamma=gamma)
    u_no_bnd = ens.predict(bnd)
    u_bnd_exact = u_exact(bnd, kappa_m, kappa_p, gamma=gamma)
    v_bnd = u_bnd_exact - u_no_bnd

    jump_no = flux_jump_of_field(ens.predict, gamma_pts, normals, kappa_m, kappa_p)
    flux_rhs = -jump_no

    band = interface_band_mask(grid, band=0.05, gamma=gamma)
    abs_r_mean = float(np.mean(np.abs(residual_int)))
    abs_r_max = float(np.max(np.abs(residual_int)))

    print(
        f"[seed {seed}] u_NO(mean) rel-L2={u_no_rel:.4e}  "
        f"|R| mean={abs_r_mean:.3e} max={abs_r_max:.3e}  "
        f"σ mean={overlap['sigma_mean']:.3e} max={overlap['sigma_max']:.3e}  "
        f"corr(σ,|R|)={overlap['corr_sigma_abs_r']:.3f}  "
        f"K={k} ε={epsilon:.3f} α={alpha:.2f}",
        flush=True,
    )

    arm_payloads: dict[str, dict] = {}
    fig_bundle: dict[str, dict] = {
        "grid": grid,
        "X": X,
        "Y": Y,
        "resolution": resolution,
        "R_grid": R_grid,
        "sigma_grid": reshape_field(sigma_flat, resolution),
        "err_no": reshape_field(abs_err_no, resolution),
        "ue": ue,
        "u_no": u_no,
        "arms": {},
    }

    for arm in STOCH_ARM_NAMES2:
        centers, subset_meta = place_centers_2b(
            arm, k, seed, grid, R_flat, sigma_flat, alloc_idx, alpha
        )
        center_ov = _subset_overlap(centers, subset_meta, grid_cand, err_cand)

        weights, info = fit_error_pde_corrector(
            centers=centers,
            interior=interior,
            residual_int=residual_int,
            kappa_int=kappa_int,
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
        corr_rel = rel_l2(u_corr, ue)
        corr_band = rel_l2_masked(u_corr, ue, band)
        reduction = u_no_rel / max(corr_rel, 1e-30)

        payload = {
            "arm": arm,
            "phase": "2b",
            "seed": seed,
            "K": int(k),
            "epsilon": float(epsilon),
            "resolution": int(resolution),
            "gamma": gamma,
            "kappa_m": float(kappa_m),
            "kappa_p": float(kappa_p),
            "alpha": float(alpha),
            "ensemble_m": int(ens.m),
            "u_no_rel_l2": float(u_no_rel),
            "corrected_rel_l2": float(corr_rel),
            "corrected_rel_l2_band": float(corr_band),
            "rel_l2_reduction_vs_no": float(reduction),
            "abs_residual_mean": abs_r_mean,
            "abs_residual_max": abs_r_max,
            "sigma_mean": float(overlap["sigma_mean"]),
            "sigma_std": float(overlap["sigma_std"]),
            "sigma_max": float(overlap["sigma_max"]),
            "corr_sigma_abs_r": float(overlap["corr_sigma_abs_r"]),
            "n_residual_centers": int(subset_meta["n_residual"]),
            "n_sigma_centers": int(subset_meta["n_sigma"]),
            "frac_residual_centers_in_top_err": float(
                center_ov["frac_residual_centers_in_top_err"]
            ),
            "frac_sigma_centers_in_top_err": float(
                center_ov["frac_sigma_centers_in_top_err"]
            ),
            "frac_all_centers_in_top_err": float(
                center_ov["frac_all_centers_in_top_err"]
            ),
            "top_err_frac": float(center_ov["top_err_frac"]),
            "kansa_cond": float(info["cond"]),
            "kansa_rank": int(info["rank"]),
            "n_centers": int(len(centers)),
            "ensemble_cached": bool(ens_meta.get("cached", False)),
        }
        out_path = RESULTS_DIR / f"{arm}_2b_seed{seed}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"  [{arm:22s}] corrected rel-L2={corr_rel:.4e}  "
            f"(↓×{reduction:.2f} vs u_NO)  cond={info['cond']:.2e}  "
            f"top-err hit: all={center_ov['frac_all_centers_in_top_err']:.3f} "
            f"R-sub={center_ov['frac_residual_centers_in_top_err']:.3f} "
            f"σ-sub={center_ov['frac_sigma_centers_in_top_err']:.3f}",
            flush=True,
        )
        arm_payloads[arm] = payload
        fig_bundle["arms"][arm] = {
            "centers": centers,
            "subset_meta": subset_meta,
            "u_corr": u_corr,
            "err": np.abs(u_corr - ue),
            "rel_l2": corr_rel,
        }

    if make_fig:
        _save_diagnostic_figure(seed, fig_bundle, u_no_rel, overlap, alpha)

    return {
        "seed": seed,
        "u_no_rel_l2": float(u_no_rel),
        "overlap": overlap,
        "alpha": float(alpha),
        "ensemble": {
            "m": int(ens.m),
            "cached": bool(ens_meta.get("cached", False)),
            "path": ens_meta.get("path"),
        },
        "arms": arm_payloads,
    }


def _save_diagnostic_figure(
    seed: int,
    bundle: dict,
    u_no_rel: float,
    overlap: dict,
    alpha: float,
) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if _MPLSTYLE.exists():
        plt.style.use(str(_MPLSTYLE))

    res = int(bundle["resolution"])
    X, Y = bundle["X"], bundle["Y"]
    R = np.abs(bundle["R_grid"])
    S = bundle["sigma_grid"]
    E = bundle["err_no"]
    arms = bundle["arms"]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
    for j, arm in enumerate(STOCH_ARM_NAMES2):
        ax = axes[0, j]
        pcm = ax.pcolormesh(X, Y, R, shading="auto", alpha=0.85)
        c = arms[arm]["centers"]
        meta = arms[arm]["subset_meta"]
        split = int(meta["split_index"])
        if meta["n_residual"] > 0:
            ax.scatter(
                c[:split, 0], c[:split, 1], s=8, c="k", marker=".", linewidths=0,
                label="R",
            )
        if meta["n_sigma"] > 0:
            ax.scatter(
                c[split:, 0], c[split:, 1], s=10, c="C3", marker="x", linewidths=0.6,
                label="σ",
            )
        ax.set_title(arm)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if j == 2:
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    pcm = ax.pcolormesh(X, Y, S, shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"σ\ncorr(σ,|R|)={overlap['corr_sigma_abs_r']:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    pcm = ax.pcolormesh(X, Y, E, shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("|u_NO − u*|")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 2]
    arm = "residual_split_sigma"
    err = arms[arm]["err"].reshape(res, res)
    pcm = ax.pcolormesh(X, Y, err, shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"|err| split α={alpha:.2f}\nrel-L2={arms[arm]['rel_l2']:.3e}")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(
        f"Phase 2b seed={seed}  u_NO(mean) rel-L2={u_no_rel:.3e}  "
        f"residual_only={arms['residual_only']['rel_l2']:.3e}",
        fontsize=11,
    )
    out = FIGURES_DIR / f"diagnostic_2b_seed{seed}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  [fig] {out}", flush=True)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    kappa_p = float(args.kappa_m) * float(args.kappa_jump)
    eps = float(args.epsilon) if args.epsilon is not None else default_epsilon(args.k)
    cfg = TinyNOConfig(n_poly=args.n_poly, n_sine=args.n_sine)
    alpha = float(args.alpha)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "phase": "2b",
        "resolution": args.resolution,
        "K": args.k,
        "epsilon": eps,
        "kappa_m": float(args.kappa_m),
        "kappa_p": float(kappa_p),
        "gamma": args.gamma,
        "alpha": alpha,
        "ensemble_m": int(args.ensemble_m),
        "seeds": {},
    }
    for seed in seeds:
        summary["seeds"][str(seed)] = run_seed(
            seed=seed,
            resolution=args.resolution,
            k=args.k,
            epsilon=eps,
            kappa_m=float(args.kappa_m),
            kappa_p=kappa_p,
            gamma=args.gamma,
            n_boundary=args.n_boundary,
            n_gamma=args.n_gamma,
            alpha=alpha,
            ens_m=int(args.ensemble_m),
            cfg=cfg,
            data_dir=args.data_dir,
            force=bool(args.force),
            make_fig=not args.skip_figures,
        )

    summary_path = RESULTS_DIR / "phase2b_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary_path}", flush=True)

    print("\n=== Phase 2b arm preview (corrected rel-L2) ===", flush=True)
    hdr = f"{'arm':22s}" + "".join(f"  seed{s:<6d}" for s in seeds)
    print(hdr, flush=True)
    for arm in STOCH_ARM_NAMES2:
        row = f"{arm:22s}"
        for s in seeds:
            v = summary["seeds"][str(s)]["arms"][arm]["corrected_rel_l2"]
            row += f"  {v:<10.4e}"
        print(row, flush=True)
    print(
        f"{'u_NO mean':22s}"
        + "".join(
            f"  {summary['seeds'][str(s)]['u_no_rel_l2']:<10.4e}" for s in seeds
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
