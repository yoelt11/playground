#!/usr/bin/env python3
"""interface_anisotropic_rbf entrypoint (JAX).

Run (smoke):
    .venv/bin/python run_pipeline.py --seeds 0 --steps 50 --kappa-jump 10

Full:
    .venv/bin/python run_pipeline.py --seeds 0,1,2 --kappa-jump 10
    .venv/bin/python run_pipeline.py --seeds 0,1,2 --kappa-jump 100

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
    train_correction_field,
    train_rbf_grad,
    train_rbf_shape,
    train_surrogate_target,
)
from common import build_experiment

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
]


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


def print_kill_table(agg: dict, kappa_jump: float) -> str:
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
    for arm in ARM_ORDER:
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
        row = (
            f"{arm:<20} {_mean_std(a['rel_l2']):<22} "
            f"{_mean_std(a['band_l2']):<22} "
            f"{_mean_std(a['step_time']):<14} "
            f"{a['stability_mode']:<18} "
            f"{note:<28}"
        )
        lines.append(row)
    lines.append(sep)

    base = np.asarray(agg["rbf-base"]["rel_l2"], dtype=float)
    grad = np.asarray(agg["rbf-grad"]["rel_l2"], dtype=float)
    shape = np.asarray(agg["rbf-shape"]["rel_l2"], dtype=float)
    base_b = np.asarray(agg["rbf-base"]["band_l2"], dtype=float)
    shape_b = np.asarray(agg["rbf-shape"]["band_l2"], dtype=float)

    g_vs_base = _noise_gate(grad, base)
    s_vs_base = _noise_gate(shape, base)
    s_vs_grad = _noise_gate(shape, grad)
    s_band_vs_base = _noise_gate(shape_b, base_b)

    lines.append(
        f"rbf-grad beats_base_beyond_noise: {g_vs_base['beats_beyond_noise']} "
        f"(sep={g_vs_base['sep']:+.4e}, pooled_std={g_vs_base['pooled_std']:.4e})"
    )
    lines.append(
        f"rbf-shape beats_base_beyond_noise: {s_vs_base['beats_beyond_noise']} "
        f"(sep={s_vs_base['sep']:+.4e}, pooled_std={s_vs_base['pooled_std']:.4e})"
    )
    lines.append(
        f"rbf-shape beats_rbf-grad_beyond_noise: {s_vs_grad['beats_beyond_noise']} "
        f"(sep={s_vs_grad['sep']:+.4e}, pooled_std={s_vs_grad['pooled_std']:.4e})"
    )
    lines.append(
        f"rbf-shape beats_base on INTERFACE-BAND beyond noise: "
        f"{s_band_vs_base['beats_beyond_noise']} "
        f"(sep={s_band_vs_base['sep']:+.4e}, "
        f"pooled_std={s_band_vs_base['pooled_std']:.4e}; "
        f"band shape={np.nanmean(shape_b):.4e}±{np.nanstd(shape_b):.4e} "
        f"vs base={np.nanmean(base_b):.4e}±{np.nanstd(base_b):.4e})"
    )

    # Shape movement summary
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
    lines.append(
        "Oracle-leak (correction-field): GT only as soft anchor "
        "‖(u_base+e_hat)−u*‖_GT; primary L[e_hat]=−(κ-residual)."
    )
    lines.append("=" * len(header))
    text = "\n".join(lines)
    print(text)
    return text


def run_seed(seed: int, cfg: dict) -> dict:
    print(f"\n=== seed {seed} (κ-jump={cfg['kappa_jump']:g}) ===", flush=True)
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
    )

    results = {}
    results["rbf-base"] = run_rbf_base(data)
    print(
        f"  rbf-base: relL2={results['rbf-base']['final_rel_l2']:.4e} "
        f"band={results['rbf-base']['final_rel_l2_band']:.4e}",
        flush=True,
    )

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
        "kappa_jump": cfg["kappa_jump"],
        "elapsed_s": elapsed,
        "interface_verify": data.interface_verify,
        "kansa_info": data.kansa_info,
        "surrogate_info": data.surrogate_info,
        "arms": _jsonable(slim),
        "cfg": cfg,
    }
    return payload, u_preds, results


def aggregate(seed_payloads: list[dict]) -> dict:
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
    } for arm in ARM_ORDER}

    for pay in seed_payloads:
        for arm in ARM_ORDER:
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

    for arm in ARM_ORDER:
        modes = agg[arm]["stabilities"]
        # majority vote
        if modes:
            agg[arm]["stability_mode"] = max(set(modes), key=modes.count)
    return agg


def write_kill_csv(agg: dict, path: Path, kappa_jump: float) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arm", "rel_l2_mean", "rel_l2_std", "band_l2_mean", "band_l2_std",
            "step_time_mean", "stability", "kappa_jump",
        ])
        for arm in ARM_ORDER:
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
    return {
        "resolution": 36,
        "gt_fraction": 0.12,
        "n_rbf_centers": 48,
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
    }


def main():
    parser = argparse.ArgumentParser(description="interface_anisotropic_rbf pipeline")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--kappa-jump",
        type=float,
        default=10.0,
        help="κ⁺/κ⁻ jump ratio (use 10 or 100)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--n-centers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny budget for CI/debug")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    cfg = default_cfg()
    cfg["kappa_jump"] = float(args.kappa_jump)
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
    if args.n_centers is not None:
        cfg["n_rbf_centers"] = args.n_centers

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print(
        f"device={__import__('jax').devices()}; seeds={seeds}; "
        f"κ-jump={cfg['kappa_jump']}; steps={cfg['steps']}; "
        f"centers={cfg['n_rbf_centers']}; res={cfg['resolution']}",
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
        out = RESULTS / f"seed_{seed}_k{int(cfg['kappa_jump'])}.json"
        with out.open("w") as f:
            json.dump(_jsonable(payload), f, indent=2)
        print(f"  wrote {out}", flush=True)

    agg = aggregate(seed_payloads)
    kill_txt = print_kill_table(agg, cfg["kappa_jump"])
    jump_tag = f"k{int(cfg['kappa_jump'])}"
    (RESULTS / "kill_table.txt").write_text(kill_txt)
    (RESULTS / f"kill_table_{jump_tag}.txt").write_text(kill_txt)
    write_kill_csv(agg, RESULTS / "kill_table.csv", cfg["kappa_jump"])
    write_kill_csv(agg, RESULTS / f"kill_table_{jump_tag}.csv", cfg["kappa_jump"])

    consolidated = {
        "cfg": cfg,
        "seeds": seeds,
        "aggregate": _jsonable(agg),
        "kill_table": kill_txt,
        "gates": {
            "rbf_grad_vs_base": _noise_gate(
                np.asarray(agg["rbf-grad"]["rel_l2"]),
                np.asarray(agg["rbf-base"]["rel_l2"]),
            ),
            "rbf_shape_vs_base": _noise_gate(
                np.asarray(agg["rbf-shape"]["rel_l2"]),
                np.asarray(agg["rbf-base"]["rel_l2"]),
            ),
            "rbf_shape_vs_grad": _noise_gate(
                np.asarray(agg["rbf-shape"]["rel_l2"]),
                np.asarray(agg["rbf-grad"]["rel_l2"]),
            ),
            "rbf_shape_band_vs_base": _noise_gate(
                np.asarray(agg["rbf-shape"]["band_l2"]),
                np.asarray(agg["rbf-base"]["band_l2"]),
            ),
        },
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
