#!/usr/bin/env python3
"""interface_anisotropic_rbf entrypoint (JAX).

Run (smoke):
    .venv/bin/python run_pipeline.py --seeds 0 --steps 50 --kappa-jump 10

Full single cell:
    .venv/bin/python run_pipeline.py --gamma vertical --kappa-jump 10 --resolution 40 \\
        --seeds '0 1 2 3 4 5'
    .venv/bin/python run_pipeline.py --gamma circle --kappa-jump 100 --resolution 80 \\
        --seeds '0 1 2 3 4 5'

Multi-cell sweep:
    .venv/bin/python run_sweep.py --seeds '0 1 2 3 4 5' --gammas vertical circle \\
        --kappa 10 100 --res 40 80

Writes results/ (per-seed JSON + results.json + kill_table.*) and figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from arms import (
    run_rbf_base,
    train_compound_loss,
    train_compound_vstar,
    train_correction_field,
    train_rbf_grad,
    train_rbf_shape,
    train_surrogate_target,
)
from common import build_experiment, default_n_centers
from stats import (
    GATE_SPECS,
    cell_statistics,
    median_iqr,
    mean_std,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ARM_ORDER = [
    "rbf-base",
    "rbf-grad",
    "rbf-shape",
    "correction-field",
    "surrogate-target",
    "compound-loss",
    "compound-vstar",
]


def parse_seeds(s: str) -> list[int]:
    """Accept '0,1,2' or '0 1 2'."""
    return [int(x) for x in s.replace(",", " ").split() if x.strip() != ""]


def parse_arms(s: str | None) -> list[str]:
    """Accept 'a,b' or 'a b'; default = full ARM_ORDER. Validate names."""
    if s is None or str(s).strip() == "":
        return list(ARM_ORDER)
    names = [x.strip() for x in str(s).replace(",", " ").split() if x.strip()]
    unknown = [n for n in names if n not in ARM_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown arm(s) {unknown}; valid: {ARM_ORDER}"
        )
    # Preserve ARM_ORDER ordering; drop duplicates
    selected = [a for a in ARM_ORDER if a in set(names)]
    if not selected:
        raise ValueError("(--arms) empty after parsing")
    return selected


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None:
        return None
    return obj


def _mean_std(vals: list[float], nan_as: str = "n/a") -> str:
    a = np.asarray(vals, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return nan_as
    return f"{np.nanmean(a):.4e}±{np.nanstd(a):.4e}"


def _noise_gate(a_arr: np.ndarray, b_arr: np.ndarray) -> dict:
    """Does a beat b beyond noise? (lower rel-L2 = better). sep = mean(b)-mean(a)."""
    a_m, a_s = float(np.nanmean(a_arr)), float(np.nanstd(a_arr))
    b_m, b_s = float(np.nanmean(b_arr)), float(np.nanstd(b_arr))
    sep = b_m - a_m
    pooled = float(np.sqrt(a_s**2 + b_s**2))
    return {
        "a_mean": a_m,
        "a_std": a_s,
        "b_mean": b_m,
        "b_std": b_s,
        "sep": sep,
        "pooled_std": pooled,
        "beats_beyond_noise": bool(sep > pooled),
    }


def print_kill_table(
    agg: dict,
    kappa_jump: float,
    cell_stats: dict | None = None,
    arm_order: list[str] | None = None,
) -> str:
    arms = arm_order if arm_order is not None else ARM_ORDER
    header = (
        f"{'arm':<20} {'relL2':<22} {'bandL2':<22} {'s/step':<14} "
        f"{'stability':<18} {'notes':<28}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        "=" * len(header),
        f"INTERFACE-Γ ANISOTROPIC RBF KILL TABLE (κ-jump={kappa_jump:g})",
        "=" * len(header),
        header,
        sep,
    ]
    for arm in arms:
        a = agg[arm]
        note = "n/a"
        if arm == "rbf-base":
            conds = a.get("kansa_conds", [])
            note = f"cond={np.nanmean(conds):.2e}" if conds else "Kansa"
        elif arm == "rbf-shape":
            moved = a.get("shape_moved", [])
            note = f"shape_moved={any(moved)}"
            conds = a.get("aniso_conds", [])
            if conds:
                note += f" anisoκ={np.nanmean(conds):.2e}"
        elif arm == "correction-field":
            note = "GT soft-anchor only"
        elif arm == "compound-vstar":
            note = "data→v* (not u*)"
        ms = mean_std(a["rel_l2"])
        mi = median_iqr(a["rel_l2"])
        bs = mean_std(a["band_l2"])
        bi = median_iqr(a["band_l2"])
        rel_str = f"{ms['mean']:.4e}±{ms['std']:.4e}"
        band_str = f"{bs['mean']:.4e}±{bs['std']:.4e}"
        row = (
            f"{arm:<20} {rel_str:<22} "
            f"{band_str:<22} "
            f"{_mean_std(a['step_time']):<14} "
            f"{a['stability_mode']:<18} "
            f"{note:<28}"
        )
        lines.append(row)
        lines.append(
            f"{'':<20} med±IQR {mi['median']:.4e}±{mi['iqr']:.4e}   "
            f"band med±IQR {bi['median']:.4e}±{bi['iqr']:.4e}"
        )
    lines.append(sep)

    have_gate_arms = all(a in agg for a in ("rbf-base", "rbf-grad", "rbf-shape"))
    if cell_stats is not None and have_gate_arms:
        for gname, _a, _b, _m in GATE_SPECS:
            g = cell_stats["gates"][gname]
            w = g["welch"]
            boot = g["bootstrap_mean_diff_95ci"]
            lines.append(
                f"{gname}: Welch t={w['t']:.4g} df={w['df']:.3g} "
                f"p={w['pvalue']:.4g} beats_beyond_noise={w['beats_beyond_noise']}; "
                f"bootΔ={boot['mean_diff']:+.4e} "
                f"95%CI=[{boot['ci_low']:+.4e},{boot['ci_high']:+.4e}]"
            )
        kn = cell_stats.get("kappa_normalized", {})
        lines.append(
            f"kappa-normalized (cond_ref={kn.get('cond_ref', float('nan')):.3e}): "
            + ", ".join(
                f"{arm}={kn.get(arm, {}).get('effective_error_mean', float('nan')):.4e}"
                for arm in ("rbf-base", "rbf-grad", "rbf-shape")
            )
        )
    elif have_gate_arms:
        # Fallback pooled-std gate (legacy) if stats not supplied
        base = np.asarray(agg["rbf-base"]["rel_l2"], dtype=float)
        grad = np.asarray(agg["rbf-grad"]["rel_l2"], dtype=float)
        shape = np.asarray(agg["rbf-shape"]["rel_l2"], dtype=float)
        base_b = np.asarray(agg["rbf-base"]["band_l2"], dtype=float)
        shape_b = np.asarray(agg["rbf-shape"]["band_l2"], dtype=float)
        for label, a, b in (
            ("rbf-grad beats_base", grad, base),
            ("rbf-shape beats_base", shape, base),
            ("rbf-shape beats_rbf-grad", shape, grad),
            ("rbf-shape band beats_base", shape_b, base_b),
        ):
            g = _noise_gate(a, b)
            lines.append(
                f"{label}_beyond_noise: {g['beats_beyond_noise']} "
                f"(sep={g['sep']:+.4e}, pooled_std={g['pooled_std']:.4e})"
            )

    # Shape movement summary
    if "rbf-shape" in agg:
        dlog = agg["rbf-shape"].get("mean_dlog_sigma", [])
        dangle = agg["rbf-shape"].get("mean_dangle", [])
        dw = agg["rbf-shape"].get("mean_weight_displacement", [])
        dmu = agg["rbf-shape"].get("mean_center_displacement", [])
        lines.append(
            f"SHAPES ACTUALLY MOVE (mean over seeds): "
            f"mean|Δw|={np.nanmean(dw) if dw else float('nan'):.4e} "
            f"mean|Δμ|={np.nanmean(dmu) if dmu else float('nan'):.4e} "
            f"mean|Δlogσ|={np.nanmean(dlog) if dlog else float('nan'):.4e} "
            f"mean|Δangle|={np.nanmean(dangle) if dangle else float('nan'):.4e}"
        )
    if "correction-field" in agg:
        lines.append(
            "Oracle-leak (correction-field): GT only as soft anchor "
            "‖(u_base+e_hat)−u*‖_GT; primary L[e_hat]=−(κ-residual)."
        )
    if "compound-vstar" in agg:
        lines.append(
            "Oracle-leak (compound-vstar): data term pins to analytic v* "
            "(not exact u*); smoothness regularizer, not GT leak."
        )
    lines.append("=" * len(header))
    text = "\n".join(lines)
    print(text)
    return text


def run_seed(seed: int, cfg: dict) -> dict:
    print(
        f"\n=== seed {seed} (γ={cfg['gamma']}, κ-jump={cfg['kappa_jump']:g}, "
        f"res={cfg['resolution']}) ===",
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

    active = cfg.get("arms", ARM_ORDER)
    results = {}
    if "rbf-base" in active:
        results["rbf-base"] = run_rbf_base(data)
        print(
            f"  rbf-base: relL2={results['rbf-base']['final_rel_l2']:.4e} "
            f"band={results['rbf-base']['final_rel_l2_band']:.4e}",
            flush=True,
        )

    if "rbf-grad" in active:
        print("  training rbf-grad ...", flush=True)
        results["rbf-grad"] = train_rbf_grad(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_rbf_grad"],
            lr_centers=cfg["lr_rbf_grad_centers"],
            lambda_bc=cfg["lambda_bc"],
            eval_every=cfg["eval_every"],
        )
        print(
            f"  rbf-grad: relL2={results['rbf-grad']['final_rel_l2']:.4e} "
            f"band={results['rbf-grad']['final_rel_l2_band']:.4e} "
            f"s/step={results['rbf-grad']['mean_step_time']:.4f}",
            flush=True,
        )

    if "rbf-shape" in active:
        print("  training rbf-shape ...", flush=True)
        results["rbf-shape"] = train_rbf_shape(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_rbf_shape"],
            lr_centers=cfg["lr_rbf_shape_centers"],
            lr_shape=cfg["lr_rbf_shape_shape"],
            lambda_bc=cfg["lambda_bc"],
            lambda_flux=cfg["lambda_flux"],
            eval_every=cfg["eval_every"],
        )
        print(
            f"  rbf-shape: relL2={results['rbf-shape']['final_rel_l2']:.4e} "
            f"band={results['rbf-shape']['final_rel_l2_band']:.4e} "
            f"s/step={results['rbf-shape']['mean_step_time']:.4f}",
            flush=True,
        )

    if "correction-field" in active:
        print("  training correction-field ...", flush=True)
        results["correction-field"] = train_correction_field(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_correction"],
            width=cfg["width"],
            lambda_anchor=cfg["lambda_anchor"],
            eval_every=cfg["eval_every"],
            colloc_batch=cfg["colloc_batch"],
        )
        print(
            f"  correction-field: relL2={results['correction-field']['final_rel_l2']:.4e} "
            f"band={results['correction-field']['final_rel_l2_band']:.4e}",
            flush=True,
        )

    if "surrogate-target" in active:
        print("  training surrogate-target ...", flush=True)
        results["surrogate-target"] = train_surrogate_target(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_surrogate"],
            width=cfg["width"],
            eval_every=cfg["eval_every"],
        )
        print(
            f"  surrogate-target: relL2={results['surrogate-target']['final_rel_l2']:.4e}",
            flush=True,
        )

    if "compound-loss" in active:
        print("  training compound-loss ...", flush=True)
        results["compound-loss"] = train_compound_loss(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_compound"],
            width=cfg["width"],
            lambda_data=cfg["lambda_data"],
            eval_every=cfg["eval_every"],
            colloc_batch=cfg["colloc_batch"],
        )
        print(
            f"  compound-loss: relL2={results['compound-loss']['final_rel_l2']:.4e} "
            f"grad-cos={results['compound-loss']['mean_grad_cos']:.3f}",
            flush=True,
        )

    if "compound-vstar" in active:
        print("  training compound-vstar ...", flush=True)
        results["compound-vstar"] = train_compound_vstar(
            data,
            steps=cfg["steps"],
            lr=cfg["lr_compound"],
            width=cfg["width"],
            lambda_vstar=cfg["lambda_vstar"],
            eval_every=cfg["eval_every"],
            colloc_batch=cfg["colloc_batch"],
        )
        print(
            f"  compound-vstar: relL2={results['compound-vstar']['final_rel_l2']:.4e} "
            f"grad-cos={results['compound-vstar']['mean_grad_cos']:.3f}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    # Strip bulky arrays for JSON (keep u_pred for figures via separate cache)
    slim = {}
    u_preds = {"u_exact": data.u_exact, "grid": data.grid, "band_mask": data.band_mask}
    for arm, r in results.items():
        u_preds[arm] = r["u_pred"]
        entry = {k: v for k, v in r.items() if k not in ("u_pred", "e_hat", "model")}
        # Trim history arrays already lists
        slim[arm] = entry

    payload = {
        "seed": seed,
        "gamma": cfg["gamma"],
        "kappa_jump": cfg["kappa_jump"],
        "resolution": cfg["resolution"],
        "n_rbf_centers": cfg["n_rbf_centers"],
        "elapsed_s": elapsed,
        "interface_verify": data.interface_verify,
        "pde_verify": data.pde_verify,
        "kansa_info": data.kansa_info,
        "surrogate_info": data.surrogate_info,
        "arms": _jsonable(slim),
        "cfg": cfg,
    }
    return payload, u_preds, results


def aggregate(seed_payloads: list[dict], arm_order: list[str] | None = None) -> dict:
    arms = arm_order if arm_order is not None else ARM_ORDER
    agg = {arm: {
        "rel_l2": [],
        "band_l2": [],
        "step_time": [],
        "grad_cos": [],
        "stabilities": [],
        "stability_mode": "n/a",
        "kansa_conds": [],
        "aniso_conds": [],
        "shape_moved": [],
        "mean_dlog_sigma": [],
        "mean_dangle": [],
        "mean_weight_displacement": [],
        "mean_center_displacement": [],
        "diverged": [],
    } for arm in arms}

    for pay in seed_payloads:
        for arm in arms:
            a = pay["arms"][arm]
            agg[arm]["rel_l2"].append(a["final_rel_l2"])
            agg[arm]["band_l2"].append(a["final_rel_l2_band"])
            agg[arm]["step_time"].append(a.get("mean_step_time", float("nan")))
            agg[arm]["grad_cos"].append(a.get("mean_grad_cos", float("nan")))
            agg[arm]["stabilities"].append(a.get("stability", "n/a"))
            if arm == "rbf-base":
                agg[arm]["kansa_conds"].append(
                    pay.get("kansa_info", {}).get("cond", float("nan"))
                )
            if arm == "rbf-shape":
                agg[arm]["aniso_conds"].append(a.get("aniso_cond", float("nan")))
                agg[arm]["shape_moved"].append(a.get("shape_moved", False))
                agg[arm]["mean_dlog_sigma"].append(a.get("mean_dlog_sigma", float("nan")))
                agg[arm]["mean_dangle"].append(a.get("mean_dangle", float("nan")))
                agg[arm]["mean_weight_displacement"].append(
                    a.get("mean_weight_displacement", float("nan"))
                )
                agg[arm]["mean_center_displacement"].append(
                    a.get("mean_center_displacement", float("nan"))
                )
            agg[arm]["diverged"].append(a.get("diverged", False))

    for arm in arms:
        modes = agg[arm]["stabilities"]
        # majority vote
        if modes:
            agg[arm]["stability_mode"] = max(set(modes), key=modes.count)
    return agg


def write_kill_csv(
    agg: dict,
    path: Path,
    kappa_jump: float,
    arm_order: list[str] | None = None,
) -> None:
    arms = arm_order if arm_order is not None else ARM_ORDER
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arm", "rel_l2_mean", "rel_l2_std", "band_l2_mean", "band_l2_std",
            "step_time_mean", "stability", "kappa_jump",
        ])
        for arm in arms:
            a = agg[arm]
            st = np.asarray(a["step_time"], dtype=float)
            st_mean = float(np.nanmean(st)) if np.any(np.isfinite(st)) else float("nan")
            w.writerow([
                arm,
                float(np.nanmean(a["rel_l2"])),
                float(np.nanstd(a["rel_l2"])),
                float(np.nanmean(a["band_l2"])),
                float(np.nanstd(a["band_l2"])),
                st_mean,
                a["stability_mode"],
                kappa_jump,
            ])


def default_cfg() -> dict:
    resolution = 40
    return {
        "gamma": "vertical",
        "resolution": resolution,
        "gt_fraction": 0.12,
        "n_rbf_centers": default_n_centers(resolution),
        "epsilon": 1.5,
        "kappa_m": 1.0,
        "kappa_jump": 10.0,
        "n_gamma": 32,
        "n_boundary": 32,
        "band": 0.05,
        "alpha": 1.0,
        "beta": 80.0,
        "steps": 1200,
        "eval_every": 50,
        "lambda_bc": 1e4,
        "lambda_flux": 1e3,
        "lambda_anchor": 5.0,
        "lambda_data": 500.0,
        "lambda_vstar": 500.0,
        "width": 64,
        "colloc_batch": 128,
        "lr_rbf_grad": 2e-6,
        "lr_rbf_grad_centers": 1e-8,
        "lr_rbf_shape": 2e-6,
        "lr_rbf_shape_shape": 5e-8,
        "lr_rbf_shape_centers": 1e-8,
        "lr_correction": 2e-3,
        "lr_surrogate": 5e-3,
        "lr_compound": 2e-3,
        "arms": list(ARM_ORDER),
    }


def main():
    parser = argparse.ArgumentParser(description="interface_anisotropic_rbf pipeline")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5",
        help="comma- or space-separated seeds (default: 0..5)",
    )
    parser.add_argument(
        "--kappa-jump",
        type=float,
        default=10.0,
        help="κ⁺/κ⁻ jump ratio (use 10 or 100)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="vertical",
        choices=["vertical", "circle"],
        help="interface geometry",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--resolution",
        "--res",
        dest="resolution",
        type=int,
        default=None,
        help="collocation grid per side (canonical: 40 or 80)",
    )
    parser.add_argument("--n-centers", type=int, default=None)
    parser.add_argument(
        "--lambda-vstar",
        type=float,
        default=None,
        help="data-term weight for compound-vstar (default: 500)",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=None,
        help=(
            "comma- or space-separated subset of arms to train/aggregate/print "
            f"(default: all of {ARM_ORDER})"
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="tiny budget for CI/debug")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    cfg = default_cfg()
    cfg["kappa_jump"] = float(args.kappa_jump)
    cfg["gamma"] = str(args.gamma)
    cfg["arms"] = parse_arms(args.arms)
    if args.lambda_vstar is not None:
        cfg["lambda_vstar"] = float(args.lambda_vstar)
    if args.smoke:
        cfg.update({
            "steps": 30,
            "resolution": 20,
            "n_rbf_centers": 16,
            "n_gamma": 12,
            "n_boundary": 12,
            "eval_every": 10,
            "colloc_batch": 64,
            "width": 32,
        })
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.resolution is not None:
        cfg["resolution"] = args.resolution
        # Scale interface/boundary sampling with grid unless smoke overrode centers
        if not args.smoke:
            cfg["n_gamma"] = max(32, args.resolution)
            cfg["n_boundary"] = max(32, args.resolution)
        if args.n_centers is None and not args.smoke:
            cfg["n_rbf_centers"] = default_n_centers(args.resolution)
    if args.n_centers is not None:
        cfg["n_rbf_centers"] = args.n_centers

    seeds = parse_seeds(args.seeds)
    active_arms = cfg["arms"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print(
        f"device={__import__('jax').devices()}; seeds={seeds}; "
        f"γ={cfg['gamma']}; κ-jump={cfg['kappa_jump']}; steps={cfg['steps']}; "
        f"centers={cfg['n_rbf_centers']}; res={cfg['resolution']}; "
        f"arms={active_arms}; λ_v*={cfg['lambda_vstar']}",
        flush=True,
    )

    seed_payloads = []
    last_preds = None
    last_results = None
    for seed in seeds:
        payload, u_preds, results = run_seed(seed, cfg)
        seed_payloads.append(payload)
        last_preds = u_preds
        last_results = results
        out = (
            RESULTS
            / f"seed_{seed}_g{cfg['gamma']}_k{int(cfg['kappa_jump'])}_r{cfg['resolution']}.json"
        )
        # Also keep legacy name for vertical default path
        legacy = RESULTS / f"seed_{seed}_k{int(cfg['kappa_jump'])}.json"
        with out.open("w") as f:
            json.dump(_jsonable(payload), f, indent=2)
        with legacy.open("w") as f:
            json.dump(_jsonable(payload), f, indent=2)
        print(f"  wrote {out}", flush=True)

    agg = aggregate(seed_payloads, arm_order=active_arms)
    # Build Welch/bootstrap/kappa-norm stats when gate arms are present
    gate_arms = ("rbf-base", "rbf-grad", "rbf-shape")
    cell_stats = None
    if all(a in active_arms for a in gate_arms):
        slim_for_stats = []
        for pay in seed_payloads:
            slim_for_stats.append({
                "seed": pay["seed"],
                "kansa_info": pay.get("kansa_info"),
                "arms": {
                    arm: {
                        "final_rel_l2": pay["arms"][arm]["final_rel_l2"],
                        "final_rel_l2_band": pay["arms"][arm]["final_rel_l2_band"],
                    }
                    for arm in active_arms
                    if arm in pay["arms"]
                },
            })
        # stats.cell_statistics uses its own 6-arm list; only call when those exist
        from stats import ARM_ORDER as STATS_ARMS

        if all(a in active_arms for a in STATS_ARMS):
            cell_stats = cell_statistics(
                slim_for_stats, cond_ref=None, bootstrap_resamples=10000
            )
    kill_txt = print_kill_table(
        agg, cfg["kappa_jump"], cell_stats=cell_stats, arm_order=active_arms
    )
    jump_tag = f"k{int(cfg['kappa_jump'])}"
    (RESULTS / "kill_table.txt").write_text(kill_txt)
    (RESULTS / f"kill_table_{jump_tag}.txt").write_text(kill_txt)
    write_kill_csv(agg, RESULTS / "kill_table.csv", cfg["kappa_jump"], arm_order=active_arms)
    write_kill_csv(
        agg, RESULTS / f"kill_table_{jump_tag}.csv", cfg["kappa_jump"], arm_order=active_arms
    )

    consolidated = {
        "cfg": cfg,
        "seeds": seeds,
        "aggregate": _jsonable(agg),
        "cell_stats": _jsonable(cell_stats) if cell_stats is not None else None,
        "kill_table": kill_txt,
        "gates": _jsonable(cell_stats["gates"]) if cell_stats is not None else None,
        "per_seed": _jsonable(seed_payloads),
    }
    with (RESULTS / "results.json").open("w") as f:
        json.dump(consolidated, f, indent=2)
    with (RESULTS / f"results_{jump_tag}.json").open("w") as f:
        json.dump(consolidated, f, indent=2)

    if not args.skip_figures and last_preds is not None and last_results is not None:
        try:
            from make_figures import make_all_figures

            make_all_figures(
                last_preds,
                last_results,
                agg,
                FIGURES,
                kappa_jump=cfg["kappa_jump"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [figures] skipped/failed: {exc}", flush=True)

    print(f"\nDone. results → {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
