#!/usr/bin/env python3
"""Multi-cell compound-vstar-only sweep with paired-Welch / bootstrap gates.

Trains ONLY the 7th arm (compound-vstar) across a (gamma, kappa, res, lambda)
grid, then gates DOMAIN and BAND rel-L2 against compound-loss and rbf-grad
references pulled from an existing sweep_stats.json (same paired-Welch +
10k bootstrap CI as run_sweep.py / stats.py).

Example (full):
    .venv/bin/python run_sweep_cvstar.py \\
        --gammas vertical circle --kappa 10 100 --res 40 80 \\
        --lamdas 50 500 --seeds '0 1 2 3 4 5' --steps 1200

CPU smoke:
    .venv/bin/python run_sweep_cvstar.py --gammas vertical --kappa 10 \\
        --res 40 --lamdas 50 500 --seeds '0 1' --steps 60 \\
        --ref-stats results/sweep_validate/sweep_stats.json

Writes ONLY under results/sweep_cvstar/ (does not touch results/sweep/ or
per-seed compound-vstar outputs from run_pipeline).
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

from arms import train_compound_vstar
from common import build_experiment, default_n_centers
from run_pipeline import _jsonable, default_cfg, parse_seeds
from stats import (
    bootstrap_mean_diff_ci,
    summarize_arm_metric,
    welch_one_sided,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "sweep_cvstar"
DEFAULT_REF = ROOT / "results" / "sweep" / "sweep_stats.json"

# Arms default signature; pipeline cfg uses 128, but compound-vstar's own
# default (and the call requested for this driver) is 256.
CVSTAR_COLLOC_BATCH = 256


def _cell_key(gamma: str, kappa: float, res: int) -> str:
    return f"g={gamma}|k={int(kappa)}|r={res}"


def _cfg_for_cell(base: dict, gamma: str, kappa: float, res: int) -> dict:
    """Mirror run_sweep._cfg_for_cell so data-build matches the sweep."""
    cfg = deepcopy(base)
    cfg["gamma"] = gamma
    cfg["kappa_jump"] = float(kappa)
    cfg["resolution"] = int(res)
    cfg["n_gamma"] = max(32, int(res))
    cfg["n_boundary"] = max(32, int(res))
    cfg["n_rbf_centers"] = default_n_centers(res)
    return cfg


def _load_ref_stats(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference sweep stats not found: {path}. "
            "Pass --ref-stats to an existing sweep_stats.json "
            "(e.g. results/sweep/sweep_stats.json)."
        )
    with path.open() as f:
        return json.load(f)


def _ref_per_seed_metric(
    ref_cell: dict,
    arm: str,
    metric: str,
    ref_seeds: list[int],
    want_seeds: list[int],
) -> np.ndarray:
    """Pull per-seed rel_l2 / band_l2 values aligned to want_seeds."""
    key = "rel_l2" if metric == "rel_l2" else "band_l2"
    vals = ref_cell["arms"][arm][key]["values"]
    seed_to_val = {int(s): float(v) for s, v in zip(ref_seeds, vals)}
    missing = [s for s in want_seeds if s not in seed_to_val]
    if missing:
        raise KeyError(
            f"ref arm={arm} metric={key} missing seeds {missing}; "
            f"ref has {list(seed_to_val)}"
        )
    return np.asarray([seed_to_val[s] for s in want_seeds], dtype=float)


def _gate(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    arm_a: str,
    arm_b: str,
    n_resamples: int,
) -> dict[str, Any]:
    welch = welch_one_sided(a, b, alternative="less")
    boot = bootstrap_mean_diff_ci(a, b, n_resamples=n_resamples, seed=0)
    out = {
        "metric": metric,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "welch": welch,
        "bootstrap_mean_diff_95ci": boot,
        "beats_beyond_noise": welch["beats_beyond_noise"],
    }
    print(
        f"    [Welch {name}] t={welch['t']:.4g} df={welch['df']:.3g} "
        f"p={welch['pvalue']:.4g} beats={welch['beats_beyond_noise']}; "
        f"bootΔ={boot['mean_diff']:+.4e} "
        f"CI=[{boot['ci_low']:+.4e},{boot['ci_high']:+.4e}]",
        flush=True,
    )
    return out


def run_cell_lambda(
    gamma: str,
    kappa: float,
    res: int,
    lam: float,
    seeds: list[int],
    base_cfg: dict,
) -> dict:
    """Build data per seed (same as run_pipeline) and train compound-vstar only."""
    cfg = _cfg_for_cell(base_cfg, gamma, kappa, res)
    cfg["lambda_vstar"] = float(lam)
    print(
        f"\n######## CELL γ={gamma} κ={kappa:g} res={res} λ_v*={lam:g} "
        f"centers={cfg['n_rbf_centers']} seeds={seeds} ########",
        flush=True,
    )
    per_seed = []
    for seed in seeds:
        print(
            f"\n=== seed {seed} (γ={gamma}, κ-jump={kappa:g}, res={res}, "
            f"λ_v*={lam:g}) ===",
            flush=True,
        )
        t0 = time.perf_counter()
        data = build_experiment(
            seed=seed,
            resolution=cfg["resolution"],
            gt_fraction=cfg["gt_fraction"],
            n_rbf_centers=cfg["n_rbf_centers"],
            epsilon=cfg["epsilon"],
            kappa_m=cfg["kappa_m"],
            kappa_jump=cfg["kappa_jump"],
            n_gamma=cfg["n_gamma"],
            n_boundary=cfg["n_boundary"],
            band=cfg["band"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            gamma=cfg["gamma"],
        )
        print("  training compound-vstar ...", flush=True)
        result = train_compound_vstar(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_compound"],
            width=cfg["width"],
            lambda_vstar=float(lam),
            eval_every=cfg["eval_every"],
            colloc_batch=CVSTAR_COLLOC_BATCH,
        )
        elapsed = time.perf_counter() - t0
        entry = {k: v for k, v in result.items() if k not in ("u_pred", "e_hat", "model")}
        print(
            f"  compound-vstar: relL2={entry['final_rel_l2']:.4e} "
            f"band={entry['final_rel_l2_band']:.4e} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        per_seed.append({
            "seed": seed,
            "gamma": gamma,
            "kappa_jump": float(kappa),
            "resolution": int(res),
            "lambda_vstar": float(lam),
            "n_rbf_centers": cfg["n_rbf_centers"],
            "elapsed_s": elapsed,
            "kansa_info": data.kansa_info,
            "interface_verify": data.interface_verify,
            "arms": {
                "compound-vstar": {
                    "final_rel_l2": entry["final_rel_l2"],
                    "final_rel_l2_band": entry["final_rel_l2_band"],
                    "mean_step_time": entry.get("mean_step_time"),
                    "stability": entry.get("stability"),
                    "diverged": entry.get("diverged", False),
                    "mean_grad_cos": entry.get("mean_grad_cos"),
                }
            },
            # Diagnostic: Kansa base from the shared data-build (cross-check vs ref)
            "rbf_base_rel_l2": float(
                np.linalg.norm(data.u_base - data.u_exact)
                / max(float(np.linalg.norm(data.u_exact)), 1e-30)
            ),
        })
    return {
        "gamma": gamma,
        "kappa_jump": float(kappa),
        "resolution": int(res),
        "lambda_vstar": float(lam),
        "n_rbf_centers": cfg["n_rbf_centers"],
        "seeds": list(seeds),
        "cfg": cfg,
        "per_seed": per_seed,
    }


def stats_for_cell_lambda(
    cell: dict,
    ref_cell: dict,
    ref_seeds: list[int],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    seeds = list(cell["seeds"])
    v_rel = np.asarray(
        [p["arms"]["compound-vstar"]["final_rel_l2"] for p in cell["per_seed"]],
        dtype=float,
    )
    v_band = np.asarray(
        [p["arms"]["compound-vstar"]["final_rel_l2_band"] for p in cell["per_seed"]],
        dtype=float,
    )
    comp_rel = _ref_per_seed_metric(ref_cell, "compound-loss", "rel_l2", ref_seeds, seeds)
    comp_band = _ref_per_seed_metric(ref_cell, "compound-loss", "band_l2", ref_seeds, seeds)
    grad_rel = _ref_per_seed_metric(ref_cell, "rbf-grad", "rel_l2", ref_seeds, seeds)
    grad_band = _ref_per_seed_metric(ref_cell, "rbf-grad", "band_l2", ref_seeds, seeds)

    arms_out = {
        "compound-vstar": {
            "rel_l2": summarize_arm_metric(v_rel),
            "band_l2": summarize_arm_metric(v_band),
        },
        "compound-loss_ref": {
            "rel_l2": summarize_arm_metric(comp_rel),
            "band_l2": summarize_arm_metric(comp_band),
        },
        "rbf-grad_ref": {
            "rel_l2": summarize_arm_metric(grad_rel),
            "band_l2": summarize_arm_metric(grad_band),
        },
    }

    print(
        f"\n--- stats γ={cell['gamma']} κ={cell['kappa_jump']:g} "
        f"res={cell['resolution']} λ={cell['lambda_vstar']:g} ---",
        flush=True,
    )
    gates = {
        "cvstar_vs_compound_domain": _gate(
            "cvstar_vs_compound_domain",
            v_rel,
            comp_rel,
            "rel_l2",
            "compound-vstar",
            "compound-loss",
            bootstrap_resamples,
        ),
        "cvstar_vs_grad_domain": _gate(
            "cvstar_vs_grad_domain",
            v_rel,
            grad_rel,
            "rel_l2",
            "compound-vstar",
            "rbf-grad",
            bootstrap_resamples,
        ),
        "cvstar_vs_compound_band": _gate(
            "cvstar_vs_compound_band",
            v_band,
            comp_band,
            "band_l2",
            "compound-vstar",
            "compound-loss",
            bootstrap_resamples,
        ),
        "cvstar_vs_grad_band": _gate(
            "cvstar_vs_grad_band",
            v_band,
            grad_band,
            "band_l2",
            "compound-vstar",
            "rbf-grad",
            bootstrap_resamples,
        ),
    }
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "lambda_vstar": float(cell["lambda_vstar"]),
        "gamma": cell["gamma"],
        "kappa_jump": cell["kappa_jump"],
        "resolution": cell["resolution"],
        "n_rbf_centers": cell["n_rbf_centers"],
        "arms": arms_out,
        "gates": {
            **gates,
            "beats_compound_domain": gates["cvstar_vs_compound_domain"]["beats_beyond_noise"],
            "beats_grad_domain": gates["cvstar_vs_grad_domain"]["beats_beyond_noise"],
            "beats_compound_band": gates["cvstar_vs_compound_band"]["beats_beyond_noise"],
            "beats_grad_band": gates["cvstar_vs_grad_band"]["beats_beyond_noise"],
        },
        "ref_means": {
            "compound-loss": {
                "rel_l2": float(ref_cell["arms"]["compound-loss"]["rel_l2"]["mean"]),
                "band_l2": float(ref_cell["arms"]["compound-loss"]["band_l2"]["mean"]),
            },
            "rbf-grad": {
                "rel_l2": float(ref_cell["arms"]["rbf-grad"]["rel_l2"]["mean"]),
                "band_l2": float(ref_cell["arms"]["rbf-grad"]["band_l2"]["mean"]),
            },
        },
    }


def _fmt_bool(b: bool) -> str:
    return "Y" if b else "N"


def print_consolidated_table(stats_cells: dict[str, dict[str, Any]], lambdas: list[float]) -> None:
    """rows=cell; cols=λ=50 and 500 for domain/band/gates."""
    # Two-lambda layout as requested; if more λs, print all.
    lam_cols = lambdas
    header_parts = [f"{'cell':<28}"]
    for lam in lam_cols:
        header_parts.append(
            f"{'dom@'+str(int(lam)):<12}"
            f"{'band@'+str(int(lam)):<12}"
            f"{'bCdom@'+str(int(lam)):<11}"
            f"{'bCband@'+str(int(lam)):<12}"
            f"{'bGband@'+str(int(lam)):<11}"
        )
    header = "".join(header_parts)
    print("\n" + "=" * max(len(header), 80))
    print("CONSOLIDATED compound-vstar TABLE (mean rel-L2 + band gates)")
    print("=" * max(len(header), 80))
    print(header)
    print("-" * max(len(header), 80))
    for cell_key, by_lam in stats_cells.items():
        parts = [f"{cell_key:<28}"]
        for lam in lam_cols:
            lam_key = str(lam)
            # also try int-string keys
            block = by_lam.get(lam_key) or by_lam.get(str(float(lam))) or by_lam.get(str(int(lam)))
            if block is None:
                parts.append(f"{'n/a':<12}{'n/a':<12}{'n/a':<11}{'n/a':<12}{'n/a':<11}")
                continue
            dom = block["arms"]["compound-vstar"]["rel_l2"]["mean"]
            band = block["arms"]["compound-vstar"]["band_l2"]["mean"]
            g = block["gates"]
            parts.append(
                f"{dom:<12.3e}"
                f"{band:<12.3e}"
                f"{_fmt_bool(g['beats_compound_domain']):<11}"
                f"{_fmt_bool(g['beats_compound_band']):<12}"
                f"{_fmt_bool(g['beats_grad_band']):<11}"
            )
        print("".join(parts))
    print("=" * max(len(header), 80))


def print_verdict(stats_cells: dict[str, dict[str, Any]], target_lam: float = 500.0) -> None:
    """Explicit ALL/none/some verdict for λ=500 band claim + domain loss."""
    lam_key_candidates = (str(target_lam), str(float(target_lam)), str(int(target_lam)))
    n_cells = 0
    n_beat_comp_band = 0
    n_beat_grad_band = 0
    n_beat_both_band = 0
    n_lose_domain_comp = 0
    n_lose_domain_grad = 0
    per_cell_notes = []
    for cell_key, by_lam in stats_cells.items():
        block = None
        for k in lam_key_candidates:
            if k in by_lam:
                block = by_lam[k]
                break
        if block is None:
            continue
        n_cells += 1
        g = block["gates"]
        bc = bool(g["beats_compound_band"])
        bg = bool(g["beats_grad_band"])
        # "lose on DOMAIN" = does NOT beat refs (higher or not significantly lower)
        lose_c = not bool(g["beats_compound_domain"])
        lose_g = not bool(g["beats_grad_domain"])
        if bc:
            n_beat_comp_band += 1
        if bg:
            n_beat_grad_band += 1
        if bc and bg:
            n_beat_both_band += 1
        if lose_c:
            n_lose_domain_comp += 1
        if lose_g:
            n_lose_domain_grad += 1
        per_cell_notes.append(
            f"  {cell_key}: beats_compound_band={bc} beats_grad_band={bg} "
            f"beats_compound_domain={g['beats_compound_domain']} "
            f"beats_grad_domain={g['beats_grad_domain']}"
        )

    def _all_none_some(n_yes: int, n: int) -> str:
        if n == 0:
            return "none (0 cells)"
        if n_yes == n:
            return f"ALL ({n_yes}/{n})"
        if n_yes == 0:
            return f"none (0/{n})"
        return f"some ({n_yes}/{n})"

    print("\n" + "=" * 72)
    print(f"VERDICT (compound-vstar λ={target_lam:g})")
    print("=" * 72)
    print(
        f"BAND vs compound-loss beyond noise: "
        f"{_all_none_some(n_beat_comp_band, n_cells)}"
    )
    print(
        f"BAND vs rbf-grad beyond noise: "
        f"{_all_none_some(n_beat_grad_band, n_cells)}"
    )
    print(
        f"BAND beats BOTH compound-loss AND rbf-grad: "
        f"{_all_none_some(n_beat_both_band, n_cells)}"
    )
    print(
        f"DOMAIN does NOT beat compound-loss (loses / no sig.): "
        f"{n_lose_domain_comp}/{n_cells} cells"
    )
    print(
        f"DOMAIN does NOT beat rbf-grad (loses / no sig.): "
        f"{n_lose_domain_grad}/{n_cells} cells"
    )
    print("Per-cell gate flags:")
    for line in per_cell_notes:
        print(line)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="compound-vstar multi-cell sweep + Welch/bootstrap gates"
    )
    parser.add_argument("--seeds", type=str, default="0 1 2 3 4 5")
    parser.add_argument("--gammas", nargs="+", default=["vertical", "circle"])
    parser.add_argument("--kappa", nargs="+", type=float, default=[10.0, 100.0])
    parser.add_argument("--res", nargs="+", type=int, default=[40, 80])
    parser.add_argument(
        "--lamdas",
        "--lambdas",
        dest="lamdas",
        nargs="+",
        type=float,
        default=[50.0, 500.0],
        help="lambda_vstar grid (accepts --lamdas typo alias)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny budget")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--ref-stats",
        type=str,
        default=None,
        help="path to sweep_stats.json with compound-loss / rbf-grad refs "
        f"(default: {DEFAULT_REF})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="override results/sweep_cvstar output directory",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    gammas = [str(g) for g in args.gammas]
    kappas = [float(k) for k in args.kappa]
    resolutions = [int(r) for r in args.res]
    lambdas = [float(x) for x in args.lamdas]

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

    ref_path = Path(args.ref_stats) if args.ref_stats else DEFAULT_REF
    if not ref_path.is_absolute():
        ref_path = ROOT / ref_path
    ref = _load_ref_stats(ref_path)
    ref_seeds = [int(s) for s in ref.get("seeds", [])]
    ref_cells = ref["cells"]
    print(
        f"[ref] loaded {ref_path} seeds={ref_seeds} "
        f"cells={list(ref_cells.keys())}",
        flush=True,
    )

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={jax.devices()}; seeds={seeds}; gammas={gammas}; "
        f"kappa={kappas}; res={resolutions}; lamdas={lambdas}; "
        f"steps={base['steps']}; colloc_batch={CVSTAR_COLLOC_BATCH}; "
        f"out={out_dir}",
        flush=True,
    )

    t0 = time.perf_counter()
    # nested: cell_key -> lambda_key -> raw cell payload
    raw_cells: dict[str, dict[str, Any]] = {}
    stats_cells: dict[str, dict[str, Any]] = {}

    for gamma in gammas:
        for kappa in kappas:
            for res in resolutions:
                cell_key = _cell_key(gamma, kappa, res)
                if cell_key not in ref_cells:
                    raise KeyError(
                        f"Cell {cell_key} missing from ref stats {ref_path}; "
                        f"available={list(ref_cells.keys())}"
                    )
                ref_cell = ref_cells[cell_key]
                raw_cells[cell_key] = {}
                stats_cells[cell_key] = {}
                for lam in lambdas:
                    lam_key = str(lam)
                    cell = run_cell_lambda(gamma, kappa, res, lam, seeds, base)
                    raw_cells[cell_key][lam_key] = cell
                    stats_cells[cell_key][lam_key] = stats_for_cell_lambda(
                        cell,
                        ref_cell,
                        ref_seeds,
                        bootstrap_resamples=args.bootstrap_resamples,
                    )

    elapsed = time.perf_counter() - t0

    cells_path = out_dir / "sweep_cvstar_cells.json"
    stats_path = out_dir / "sweep_cvstar_stats.json"
    with cells_path.open("w") as f:
        json.dump(_jsonable({
            "seeds": seeds,
            "gammas": gammas,
            "kappa": kappas,
            "resolutions": resolutions,
            "lamdas": lambdas,
            "steps": base["steps"],
            "colloc_batch": CVSTAR_COLLOC_BATCH,
            "ref_stats": str(ref_path),
            "ref_seeds": ref_seeds,
            "cells": raw_cells,
            "elapsed_s": elapsed,
        }), f, indent=2)
    with stats_path.open("w") as f:
        json.dump(_jsonable({
            "seeds": seeds,
            "lamdas": lambdas,
            "ref_stats": str(ref_path),
            "ref_seeds": ref_seeds,
            "cells": stats_cells,
            "elapsed_s": elapsed,
        }), f, indent=2)

    print_consolidated_table(stats_cells, lambdas)
    # Prefer λ=500 verdict when present, else first lambda
    verdict_lam = 500.0 if 500.0 in lambdas else lambdas[-1]
    print_verdict(stats_cells, target_lam=verdict_lam)

    print(f"wrote {cells_path}")
    print(f"wrote {stats_path}")
    print(f"elapsed={elapsed:.1f}s")
    print(
        f"[sanity] wrote only under {out_dir} "
        "(results/sweep/ and other results/*.json untouched)"
    )


if __name__ == "__main__":
    main()
