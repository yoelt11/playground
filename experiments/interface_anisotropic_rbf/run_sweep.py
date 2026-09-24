#!/usr/bin/env python3
"""Multi-cell sweep driver for interface_anisotropic_rbf.

Example:
    .venv/bin/python run_sweep.py --seeds '0 1 2 3 4 5' \\
        --gammas vertical circle --kappa 10 100 --res 40 80

Writes:
    results/sweep/sweep_cells.json
    results/sweep/sweep_stats.json
and prints a consolidated table.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from common import default_n_centers
from run_pipeline import (
    ARM_ORDER,
    _jsonable,
    default_cfg,
    parse_seeds,
    run_seed,
)
from stats import (
    cell_statistics,
    consolidated_table_header,
    format_consolidated_row,
)

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "results" / "sweep"

CONVERGENCE_CENTERS = (32, 48, 64)
CONVERGENCE_REF = {"gamma": "vertical", "kappa": 10.0, "res": 40}


def _cell_key(gamma: str, kappa: float, res: int) -> str:
    return f"g={gamma}|k={int(kappa)}|r={res}"


def _cfg_for_cell(
    base: dict,
    gamma: str,
    kappa: float,
    res: int,
    n_centers: int | None = None,
) -> dict:
    cfg = deepcopy(base)
    cfg["gamma"] = gamma
    cfg["kappa_jump"] = float(kappa)
    cfg["resolution"] = int(res)
    cfg["n_gamma"] = max(32, int(res))
    cfg["n_boundary"] = max(32, int(res))
    cfg["n_rbf_centers"] = (
        int(n_centers) if n_centers is not None else default_n_centers(res)
    )
    return cfg


def run_cell(
    gamma: str,
    kappa: float,
    res: int,
    seeds: list[int],
    base_cfg: dict,
    n_centers: int | None = None,
) -> dict:
    cfg = _cfg_for_cell(base_cfg, gamma, kappa, res, n_centers=n_centers)
    print(
        f"\n######## CELL γ={gamma} κ={kappa:g} res={res} "
        f"centers={cfg['n_rbf_centers']} seeds={seeds} ########",
        flush=True,
    )
    per_seed = []
    for seed in seeds:
        payload, _preds, _results = run_seed(seed, cfg)
        # Slim for sweep JSON: keep metrics only
        slim_arms = {}
        for arm in ARM_ORDER:
            a = payload["arms"][arm]
            slim_arms[arm] = {
                "final_rel_l2": a["final_rel_l2"],
                "final_rel_l2_band": a["final_rel_l2_band"],
                "mean_step_time": a.get("mean_step_time"),
                "stability": a.get("stability"),
                "diverged": a.get("diverged", False),
                "kansa_cond": a.get("kansa_cond"),
                "aniso_cond": a.get("aniso_cond"),
                "shape_moved": a.get("shape_moved"),
                "mean_dlog_sigma": a.get("mean_dlog_sigma"),
                "mean_dangle": a.get("mean_dangle"),
            }
        per_seed.append({
            "seed": seed,
            "gamma": gamma,
            "kappa_jump": float(kappa),
            "resolution": int(res),
            "n_rbf_centers": cfg["n_rbf_centers"],
            "elapsed_s": payload["elapsed_s"],
            "interface_verify": payload.get("interface_verify"),
            "pde_verify": payload.get("pde_verify"),
            "kansa_info": payload.get("kansa_info"),
            "arms": slim_arms,
        })
    return {
        "gamma": gamma,
        "kappa_jump": float(kappa),
        "resolution": int(res),
        "n_rbf_centers": cfg["n_rbf_centers"],
        "seeds": list(seeds),
        "cfg": cfg,
        "per_seed": per_seed,
    }


def run_convergence_sweep(
    base_cfg: dict,
    seeds: list[int],
    centers_list: tuple[int, ...] = CONVERGENCE_CENTERS,
) -> dict:
    """Error vs N_centers at ref cell (vertical, κ=10, res=40)."""
    ref = CONVERGENCE_REF
    print(
        f"\n######## CONVERGENCE γ={ref['gamma']} κ={ref['kappa']:g} "
        f"res={ref['res']} centers={list(centers_list)} ########",
        flush=True,
    )
    by_n: dict[str, Any] = {}
    for n_cent in centers_list:
        cell = run_cell(
            ref["gamma"],
            ref["kappa"],
            ref["res"],
            seeds,
            base_cfg,
            n_centers=n_cent,
        )
        arm_means = {}
        for arm in ARM_ORDER:
            vals = [p["arms"][arm]["final_rel_l2"] for p in cell["per_seed"]]
            band = [p["arms"][arm]["final_rel_l2_band"] for p in cell["per_seed"]]
            arm_means[arm] = {
                "rel_l2_mean": float(np.nanmean(vals)),
                "rel_l2_std": float(np.nanstd(vals)),
                "band_l2_mean": float(np.nanmean(band)),
                "band_l2_std": float(np.nanstd(band)),
                "rel_l2_per_seed": vals,
                "band_l2_per_seed": band,
            }
        by_n[str(n_cent)] = {
            "n_centers": int(n_cent),
            "arms": arm_means,
            "per_seed": cell["per_seed"],
        }
        print(
            f"  N={n_cent}: base={arm_means['rbf-base']['rel_l2_mean']:.4e} "
            f"grad={arm_means['rbf-grad']['rel_l2_mean']:.4e} "
            f"shape={arm_means['rbf-shape']['rel_l2_mean']:.4e} "
            f"shape_band={arm_means['rbf-shape']['band_l2_mean']:.4e}",
            flush=True,
        )
    return {
        "ref_cell": ref,
        "centers": list(centers_list),
        "seeds": list(seeds),
        "by_n_centers": by_n,
    }


def main():
    parser = argparse.ArgumentParser(description="interface_anisotropic_rbf multi-cell sweep")
    parser.add_argument("--seeds", type=str, default="0 1 2 3 4 5")
    parser.add_argument("--gammas", nargs="+", default=["vertical", "circle"])
    parser.add_argument("--kappa", nargs="+", type=float, default=[10.0, 100.0])
    parser.add_argument("--res", nargs="+", type=int, default=[40, 80])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny budget")
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="skip N_centers convergence sweep",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="override results/sweep output directory",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    gammas = [str(g) for g in args.gammas]
    kappas = [float(k) for k in args.kappa]
    resolutions = [int(r) for r in args.res]

    base = default_cfg()
    if args.smoke:
        base.update({
            "steps": 30,
            "eval_every": 10,
            "colloc_batch": 64,
            "width": 32,
            "n_gamma": 12,
            "n_boundary": 12,
        })
    if args.steps is not None:
        base["steps"] = args.steps

    out_dir = Path(args.out_dir) if args.out_dir else SWEEP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={jax.devices()}; seeds={seeds}; gammas={gammas}; "
        f"kappa={kappas}; res={resolutions}; steps={base['steps']}",
        flush=True,
    )

    t0 = time.perf_counter()
    cells: dict[str, Any] = {}
    for gamma in gammas:
        for kappa in kappas:
            for res in resolutions:
                key = _cell_key(gamma, kappa, res)
                cells[key] = run_cell(gamma, kappa, res, seeds, base)

    # cond_ref = mean Kansa cond at vertical, κ=10, res=40 (or first matching)
    ref_key = _cell_key("vertical", 10.0, 40)
    cond_ref = None
    if ref_key in cells:
        conds = [
            p.get("kansa_info", {}).get("cond", float("nan"))
            for p in cells[ref_key]["per_seed"]
        ]
        cond_ref = float(np.nanmean(conds))
    else:
        # fallback: first cell at res=40
        for k, c in cells.items():
            if c["resolution"] == 40:
                conds = [
                    p.get("kansa_info", {}).get("cond", float("nan"))
                    for p in c["per_seed"]
                ]
                cond_ref = float(np.nanmean(conds))
                break
    print(f"\n[cond_ref] Kansa cond at vertical/κ10/res40 = {cond_ref}", flush=True)

    stats_cells: dict[str, Any] = {}
    for key, cell in cells.items():
        print(f"\n--- stats for {key} ---", flush=True)
        stats_cells[key] = cell_statistics(
            cell["per_seed"],
            cond_ref=cond_ref,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        stats_cells[key]["gamma"] = cell["gamma"]
        stats_cells[key]["kappa_jump"] = cell["kappa_jump"]
        stats_cells[key]["resolution"] = cell["resolution"]
        stats_cells[key]["n_rbf_centers"] = cell["n_rbf_centers"]

    convergence = None
    if not args.skip_convergence:
        # Use same seeds; for smoke keep centers list small-budget compatible
        centers = CONVERGENCE_CENTERS
        if args.smoke:
            centers = (16, 24, 32)
        convergence = run_convergence_sweep(base, seeds, centers_list=centers)

    elapsed = time.perf_counter() - t0

    cells_path = out_dir / "sweep_cells.json"
    stats_path = out_dir / "sweep_stats.json"
    with cells_path.open("w") as f:
        json.dump(_jsonable({
            "seeds": seeds,
            "gammas": gammas,
            "kappa": kappas,
            "resolutions": resolutions,
            "steps": base["steps"],
            "cond_ref": cond_ref,
            "cells": cells,
            "elapsed_s": elapsed,
        }), f, indent=2)
    with stats_path.open("w") as f:
        json.dump(_jsonable({
            "seeds": seeds,
            "cond_ref": cond_ref,
            "cells": stats_cells,
            "convergence": convergence,
            "elapsed_s": elapsed,
        }), f, indent=2)

    # Consolidated table
    header = consolidated_table_header()
    print("\n" + "=" * len(header))
    print("CONSOLIDATED SWEEP TABLE (mean rel-L2 + Welch p + kappa-norm shape)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = []
    for key in cells:
        row = format_consolidated_row(key, stats_cells[key])
        rows.append(row)
        print(row)
    if convergence is not None:
        print("-" * len(header))
        print("CONVERGENCE (rel-L2 mean vs N_centers) @ vertical/κ10/res40:")
        for n_s, block in convergence["by_n_centers"].items():
            a = block["arms"]
            print(
                f"  N={n_s:>3}: base={a['rbf-base']['rel_l2_mean']:.4e}  "
                f"grad={a['rbf-grad']['rel_l2_mean']:.4e}  "
                f"shape={a['rbf-shape']['rel_l2_mean']:.4e}  "
                f"shape_band={a['rbf-shape']['band_l2_mean']:.4e}"
            )
    print("=" * len(header))
    print(f"wrote {cells_path}")
    print(f"wrote {stats_path}")
    print(f"elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
