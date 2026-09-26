#!/usr/bin/env python3
"""Phase 2: stochastic NO ensemble → residual±uncertainty allocation → v_RBF."""

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
    DEFAULT_LAMBDA_UNC,
    DEFAULT_RESOLUTION,
    DEFAULT_SEEDS,
    STOCH_ARM_NAMES,
    allocate_residual,
    allocate_residual_shuffledunc,
    allocate_residual_unc,
    boundary_points,
    default_epsilon,
    f_exact,
    fit_error_pde_corrector,
    flux_jump_of_field,
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
from no_model import (
    DATA_DIR,
    ENSEMBLE_SIZE,
    TinyNOConfig,
    ensemble_checkpoint_path,
    load_ensemble,
    save_ensemble,
    train_ensemble,
)

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
    p.add_argument("--lambda-unc", type=float, default=DEFAULT_LAMBDA_UNC)
    p.add_argument("--ensemble-m", type=int, default=ENSEMBLE_SIZE)
    p.add_argument("--n-poly", type=int, default=2)
    p.add_argument("--n-sine", type=int, default=1)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--force",
        action="store_true",
        help="Retrain ensembles even if no_ens_seed*.npz cache exists",
    )
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def ensure_ensemble(
    seed: int,
    data_dir: Path,
    force: bool,
    m: int,
    cfg: TinyNOConfig,
    kappa_m: float,
    kappa_p: float,
    gamma: str,
):
    path = ensemble_checkpoint_path(seed, data_dir)
    if path.exists() and not force:
        print(f"[cache hit] {path}", flush=True)
        return load_ensemble(path), {"cached": True, "path": str(path)}
    print(f"[train ensemble] seed_base={seed} → {path}", flush=True)
    ens, info = train_ensemble(
        u_exact_fn=u_exact,
        seed_base=seed,
        cfg=cfg,
        m=m,
        kappa_m=kappa_m,
        kappa_p=kappa_p,
        gamma=gamma,
        verbose=True,
    )
    save_ensemble(ens, path)
    info_out = {"cached": False, "path": str(path), **info}
    print(f"[saved] {path}", flush=True)
    return ens, info_out


def place_centers_stoch(
    arm: str,
    k: int,
    seed: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    sigma_flat: np.ndarray,
    interior_idx: np.ndarray,
    lambda_unc: float,
) -> np.ndarray:
    if arm == "residual_only":
        # Exact Phase-1 residual_alloc policy (same RNG offset + weights).
        return allocate_residual(k, grid, residual_flat, interior_idx, seed)
    if arm == "residual_unc":
        return allocate_residual_unc(
            k,
            grid,
            residual_flat,
            sigma_flat,
            interior_idx,
            seed,
            lambda_unc=lambda_unc,
        )
    if arm == "shuffled_unc":
        return allocate_residual_shuffledunc(
            k,
            grid,
            residual_flat,
            sigma_flat,
            interior_idx,
            seed,
            lambda_unc=lambda_unc,
        )
    raise ValueError(f"unknown Phase-2 arm {arm!r}")


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
    lambda_unc: float,
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
        f"K={k} ε={epsilon:.3f} λ_unc={lambda_unc:.2f}",
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
        "ue": ue,
        "u_no": u_no,
        "arms": {},
    }

    for arm in STOCH_ARM_NAMES:
        centers = place_centers_stoch(
            arm, k, seed, grid, R_flat, sigma_flat, alloc_idx, lambda_unc
        )
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
            "phase": 2,
            "seed": seed,
            "K": int(k),
            "epsilon": float(epsilon),
            "resolution": int(resolution),
            "gamma": gamma,
            "kappa_m": float(kappa_m),
            "kappa_p": float(kappa_p),
            "lambda_unc": float(lambda_unc),
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
            "frac_top_unc_with_low_r": float(overlap["frac_top_unc_with_low_r"]),
            "frac_top_r_with_low_unc": float(overlap["frac_top_r_with_low_unc"]),
            "kansa_cond": float(info["cond"]),
            "kansa_rank": int(info["rank"]),
            "n_centers": int(len(centers)),
            "ensemble_cached": bool(ens_meta.get("cached", False)),
        }
        out_path = RESULTS_DIR / f"{arm}_stoch_seed{seed}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"  [{arm:15s}] corrected rel-L2={corr_rel:.4e}  "
            f"(↓×{reduction:.2f} vs u_NO)  cond={info['cond']:.2e}",
            flush=True,
        )
        arm_payloads[arm] = payload
        fig_bundle["arms"][arm] = {
            "centers": centers,
            "u_corr": u_corr,
            "err": np.abs(u_corr - ue),
            "rel_l2": corr_rel,
        }

    if make_fig:
        _save_diagnostic_figure(seed, fig_bundle, u_no_rel, overlap)

    return {
        "seed": seed,
        "u_no_rel_l2": float(u_no_rel),
        "overlap": overlap,
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
) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if _MPLSTYLE.exists():
        plt.style.use(str(_MPLSTYLE))

    res = int(bundle["resolution"])
    X, Y = bundle["X"], bundle["Y"]
    R = np.abs(bundle["R_grid"])
    S = bundle["sigma_grid"]
    arms = bundle["arms"]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)
    titles = {
        "residual_only": "residual_only on |R|",
        "residual_unc": "residual_unc on |R|",
        "shuffled_unc": "shuffled_unc on |R|",
    }
    for j, arm in enumerate(STOCH_ARM_NAMES):
        ax = axes[0, j]
        pcm = ax.pcolormesh(X, Y, R, shading="auto", alpha=0.85)
        c = arms[arm]["centers"]
        ax.scatter(c[:, 0], c[:, 1], s=6, c="k", marker=".", linewidths=0)
        ax.set_title(titles[arm])
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if j == 2:
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    # Row 1: σ map, |err| residual_unc, |err| shuffled_unc (residual_only err in title bar)
    ax = axes[1, 0]
    pcm = ax.pcolormesh(X, Y, S, shading="auto")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"σ (ensemble std)\ncorr(σ,|R|)={overlap['corr_sigma_abs_r']:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for j, arm in enumerate(("residual_unc", "shuffled_unc")):
        ax = axes[1, j + 1]
        err = arms[arm]["err"].reshape(res, res)
        pcm = ax.pcolormesh(X, Y, err, shading="auto")
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"|err| {arm}\nrel-L2={arms[arm]['rel_l2']:.3e}")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"Phase 2 seed={seed}  u_NO(mean) rel-L2={u_no_rel:.3e}  "
        f"residual_only rel-L2={arms['residual_only']['rel_l2']:.3e}",
        fontsize=11,
    )
    out = FIGURES_DIR / f"diagnostic_stoch_seed{seed}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  [fig] {out}", flush=True)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    kappa_p = float(args.kappa_m) * float(args.kappa_jump)
    eps = float(args.epsilon) if args.epsilon is not None else default_epsilon(args.k)
    cfg = TinyNOConfig(n_poly=args.n_poly, n_sine=args.n_sine)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "phase": 2,
        "resolution": args.resolution,
        "K": args.k,
        "epsilon": eps,
        "kappa_m": float(args.kappa_m),
        "kappa_p": float(kappa_p),
        "gamma": args.gamma,
        "lambda_unc": float(args.lambda_unc),
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
            lambda_unc=float(args.lambda_unc),
            ens_m=int(args.ensemble_m),
            cfg=cfg,
            data_dir=args.data_dir,
            force=bool(args.force),
            make_fig=not args.skip_figures,
        )

    summary_path = RESULTS_DIR / "phase2_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary_path}", flush=True)

    print("\n=== Phase 2 arm preview (corrected rel-L2) ===", flush=True)
    hdr = f"{'arm':16s}" + "".join(f"  seed{s:<6d}" for s in seeds)
    print(hdr, flush=True)
    for arm in STOCH_ARM_NAMES:
        row = f"{arm:16s}"
        for s in seeds:
            v = summary["seeds"][str(s)]["arms"][arm]["corrected_rel_l2"]
            row += f"  {v:<10.4e}"
        print(row, flush=True)
    print(
        f"{'u_NO mean':16s}"
        + "".join(
            f"  {summary['seeds'][str(s)]['u_no_rel_l2']:<10.4e}" for s in seeds
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
