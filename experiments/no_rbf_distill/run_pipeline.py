#!/usr/bin/env python3
"""End-to-end Phase 1: NO → residual → allocate K centers → fit v_RBF → metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import (
    ARM_NAMES,
    DEFAULT_GAMMA,
    DEFAULT_K,
    DEFAULT_KAPPA_JUMP,
    DEFAULT_KAPPA_M,
    DEFAULT_RESOLUTION,
    DEFAULT_SEEDS,
    allocate_residual,
    allocate_shuffled,
    allocate_uniform,
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
    u_exact,
)
from no_model import DATA_DIR, checkpoint_path, load_model

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Optional research plotting style
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
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--skip-train", action="store_true", help="Do not auto-call run_train_no")
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def ensure_no_checkpoints(seeds: list[int], data_dir: Path, skip_train: bool) -> None:
    missing = [s for s in seeds if not checkpoint_path(s, data_dir).exists()]
    if not missing:
        return
    if skip_train:
        raise FileNotFoundError(
            f"Missing NO checkpoints for seeds {missing} in {data_dir}; "
            "run run_train_no.py first or drop --skip-train"
        )
    cmd = [
        sys.executable,
        str(ROOT / "run_train_no.py"),
        "--seeds",
        ",".join(str(s) for s in missing),
        "--data-dir",
        str(data_dir),
    ]
    print(f"[pipeline] training missing NO seeds via: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def place_centers(
    arm: str,
    k: int,
    seed: int,
    grid: np.ndarray,
    residual_flat: np.ndarray,
    interior_idx: np.ndarray,
) -> np.ndarray:
    if arm == "uniform":
        return allocate_uniform(k, seed)
    if arm == "residual_alloc":
        return allocate_residual(k, grid, residual_flat, interior_idx, seed)
    if arm == "shuffled_alloc":
        return allocate_shuffled(k, grid, residual_flat, interior_idx, seed)
    raise ValueError(f"unknown arm {arm!r}")


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
    data_dir: Path,
    make_fig: bool,
) -> dict:
    model = load_model(checkpoint_path(seed, data_dir))
    grid, X, Y = make_grid(resolution)
    h = float(X[0, 1] - X[0, 0])
    ue = u_exact(grid, kappa_m, kappa_p, gamma=gamma)
    fe = f_exact(grid, kappa_m, kappa_p, gamma=gamma)
    kap = kappa_of_points(grid, kappa_m, kappa_p, gamma=gamma)

    u_no = model.predict(grid)
    u_no_rel = rel_l2(u_no, ue)

    u_grid = reshape_field(u_no, resolution)
    f_grid = reshape_field(fe, resolution)
    k_grid = reshape_field(kap, resolution)
    R_grid = pde_residual_fd(u_grid, f_grid, k_grid, h)
    R_flat = R_grid.ravel()

    interior_idx = interior_away_from_gamma(grid, gamma=gamma)
    interior = grid[interior_idx]
    residual_int = R_flat[interior_idx]
    kappa_int = kap[interior_idx]

    # Allocation may sample near Γ (where |R| peaks); collocation stays away from Γ
    alloc_idx = interior_for_allocation(grid)

    bnd = boundary_points(n_boundary)
    gamma_pts = gamma_points(n_gamma, gamma=gamma)
    normals = gamma_normals(gamma_pts, gamma=gamma)
    u_no_bnd = model.predict(bnd)
    u_bnd_exact = u_exact(bnd, kappa_m, kappa_p, gamma=gamma)
    v_bnd = u_bnd_exact - u_no_bnd  # correction matches Dirichlet data

    # Error-field flux: [κ ∂ν v] = −[κ ∂ν u_NO] so corrected field has [κ ∂ν u]=0
    jump_no = flux_jump_of_field(
        model.predict, gamma_pts, normals, kappa_m, kappa_p
    )
    flux_rhs = -jump_no

    band = interface_band_mask(grid, band=0.05, gamma=gamma)
    abs_r_mean = float(np.mean(np.abs(residual_int)))
    abs_r_max = float(np.max(np.abs(residual_int)))

    print(
        f"[seed {seed}] u_NO rel-L2={u_no_rel:.4e}  "
        f"|R| mean={abs_r_mean:.3e} max={abs_r_max:.3e}  K={k} ε={epsilon:.3f}",
        flush=True,
    )
    if u_no_rel < 1e-3:
        print(
            f"  [warn] u_NO almost exact (rel-L2={u_no_rel:.2e}); "
            "distillation signal may be weak",
            flush=True,
        )
    if abs_r_max < 1e-8:
        print("  [warn] residual ~0 — nothing to correct", flush=True)

    arm_payloads: dict[str, dict] = {}
    fig_bundle: dict[str, dict] = {
        "grid": grid,
        "X": X,
        "Y": Y,
        "resolution": resolution,
        "R_grid": R_grid,
        "ue": ue,
        "u_no": u_no,
        "arms": {},
    }

    for arm in ARM_NAMES:
        centers = place_centers(arm, k, seed, grid, R_flat, alloc_idx)
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
            "seed": seed,
            "K": int(k),
            "epsilon": float(epsilon),
            "resolution": int(resolution),
            "gamma": gamma,
            "kappa_m": float(kappa_m),
            "kappa_p": float(kappa_p),
            "u_no_rel_l2": float(u_no_rel),
            "corrected_rel_l2": float(corr_rel),
            "corrected_rel_l2_band": float(corr_band),
            "rel_l2_reduction_vs_no": float(reduction),
            "abs_residual_mean": abs_r_mean,
            "abs_residual_max": abs_r_max,
            "kansa_cond": float(info["cond"]),
            "kansa_rank": int(info["rank"]),
            "n_centers": int(len(centers)),
        }
        out_path = RESULTS_DIR / f"{arm}_seed{seed}.json"
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
        _save_diagnostic_figure(seed, fig_bundle, u_no_rel)

    return {
        "seed": seed,
        "u_no_rel_l2": float(u_no_rel),
        "arms": arm_payloads,
    }


def _save_diagnostic_figure(seed: int, bundle: dict, u_no_rel: float) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if _MPLSTYLE.exists():
        plt.style.use(str(_MPLSTYLE))

    res = int(bundle["resolution"])
    X, Y = bundle["X"], bundle["Y"]
    R = np.abs(bundle["R_grid"])
    arms = bundle["arms"]

    # 2×4: |R| + three placements | three |error| maps (last col unused on row1 → drop)
    # Spec: residual map, three center placements, |error| per arm → use 2×3:
    # row0: each arm's centers overlaid on |R|
    # row1: |u_NO+v_RBF − u*| per arm
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True)

    titles = {
        "uniform": "uniform centers on |R|",
        "residual_alloc": "residual_alloc on |R|",
        "shuffled_alloc": "shuffled_alloc on |R|",
    }
    for j, arm in enumerate(ARM_NAMES):
        ax = axes[0, j]
        pcm = ax.pcolormesh(X, Y, R, shading="auto", alpha=0.9)
        c = arms[arm]["centers"]
        ax.scatter(c[:, 0], c[:, 1], s=6, c="k", marker=".", linewidths=0)
        ax.set_title(titles[arm])
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if j == 2:
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    for j, arm in enumerate(ARM_NAMES):
        ax = axes[1, j]
        err = arms[arm]["err"].reshape(res, res)
        pcm = ax.pcolormesh(X, Y, err, shading="auto")
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"|err| {arm}\nrel-L2={arms[arm]['rel_l2']:.3e}")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"seed={seed}  u_NO rel-L2={u_no_rel:.3e}  "
        f"(row0: |R|+centers / row1: |u_NO+v−u*|)",
        fontsize=11,
    )
    out = FIGURES_DIR / f"diagnostic_seed{seed}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  [fig] {out}", flush=True)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    kappa_p = float(args.kappa_m) * float(args.kappa_jump)
    eps = float(args.epsilon) if args.epsilon is not None else default_epsilon(args.k)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ensure_no_checkpoints(seeds, args.data_dir, args.skip_train)

    summary = {
        "resolution": args.resolution,
        "K": args.k,
        "epsilon": eps,
        "kappa_m": float(args.kappa_m),
        "kappa_p": float(kappa_p),
        "gamma": args.gamma,
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
            data_dir=args.data_dir,
            make_fig=not args.skip_figures,
        )

    summary_path = RESULTS_DIR / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {summary_path}", flush=True)

    # Quick arm table preview
    print("\n=== arm preview (corrected rel-L2) ===", flush=True)
    hdr = f"{'arm':16s}" + "".join(f"  seed{s:<6d}" for s in seeds)
    print(hdr, flush=True)
    for arm in ARM_NAMES:
        row = f"{arm:16s}"
        for s in seeds:
            v = summary["seeds"][str(s)]["arms"][arm]["corrected_rel_l2"]
            row += f"  {v:<10.4e}"
        print(row, flush=True)
    print(
        f"{'u_NO (no corr)':16s}"
        + "".join(
            f"  {summary['seeds'][str(s)]['u_no_rel_l2']:<10.4e}" for s in seeds
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
