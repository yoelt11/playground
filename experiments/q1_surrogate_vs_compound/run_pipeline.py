#!/usr/bin/env python3
"""Q1 probe entrypoint: compound-loss vs surrogate-target vs correction-field.

Run:
    .venv/bin/python run_pipeline.py

Writes results/ (per-seed + consolidated JSON/CSV) and figures/, prints kill table.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from diagnostics import summarize_arm
from plotting import plot_error_maps, plot_solution_maps, plot_training_curves
from setup_data import build_experiment, rel_l2
from train_arms import train_arm1_compound, train_arm2_surrogate, train_arm3_correction

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Budget tuned for CPU wall-clock of a few minutes while still separating arms.
CFG = {
    "seeds": [0, 1, 2],
    "resolution": 40,
    "gt_fraction": 0.12,
    "n_rbf_centers": 48,
    "epsilon": 1.5,
    "alpha": 1.0,
    "beta": 80.0,
    "steps": 3000,  # shared step budget across arms
    "lambda_data": 500.0,
    "lambda_anchor": 5.0,
    "width": 128,
    "lr_arm1": 2e-3,
    "lr_arm2": 5e-3,
    "lr_arm3": 2e-3,
    "colloc_batch": 512,
    "eval_every": 50,
}


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
    return obj


def _mean_std(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    return f"{a.mean():.4f}±{a.std():.4f}"


def _fmt_span(vals_in: list[float], vals_out: list[float]) -> str:
    a = np.asarray(vals_in, dtype=float)
    b = np.asarray(vals_out, dtype=float)
    return f"in={a.mean():.2f}±{a.std():.2f} / out={b.mean():.2f}±{b.std():.2f}"


def print_kill_table(agg: dict) -> str:
    rows = []
    header = (
        f"{'arm':<20} {'relL2':<16} {'s/step':<12} {'grad-cos':<14} "
        f"{'in/out-span':<28} {'stability':<18}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        "=" * len(header),
        "Q1 FEASIBILITY KILL TABLE",
        "=" * len(header),
        header,
        sep,
    ]
    for arm in ["compound-loss", "surrogate-target", "correction-field"]:
        a = agg[arm]
        row = (
            f"{arm:<20} {_mean_std(a['rel_l2']):<16} "
            f"{_mean_std(a['step_time']):<12} "
            f"{_mean_std(a['grad_cos']):<14} "
            f"{_fmt_span(a['in_span'], a['out_span']):<28} "
            f"{a['stability_mode']:<18}"
        )
        lines.append(row)
        rows.append(row)
    lines.append(sep)

    # Feasibility declaration
    surr = np.mean(agg["surrogate-target"]["rel_l2"])
    comp = np.mean(agg["compound-loss"]["rel_l2"])
    surr_t = np.mean(agg["surrogate-target"]["step_time"])
    surr_stable = agg["surrogate-target"]["stability_mode"] == "smooth"
    comp_conflict = np.mean(agg["compound-loss"]["grad_cos"])
    # README: if >90% of surrogate-target gain is in-span, the claim collapses.
    surr_in_span = np.mean(agg["surrogate-target"]["in_span"])
    out_ok = surr_in_span <= 0.90

    feasible = (
        surr <= 0.012
        and surr < comp
        and surr_t < 1.0
        and surr_stable
        and comp_conflict < 0.25
        and out_ok
    )
    lines.append(
        f"Q1 feasible? {bool(feasible)}  "
        f"(surrogate relL2={surr:.4f} vs compound={comp:.4f}; "
        f"surr s/step={surr_t:.3f}; "
        f"compound grad-cos={comp_conflict:.3f}; "
        f"surr in-span={surr_in_span:.2f} (oracle-gate fail if >0.90); "
        f"surr stability={agg['surrogate-target']['stability_mode']}, "
        f"compound stability={agg['compound-loss']['stability_mode']})"
    )
    lines.append(
        "Oracle-leak (Arm3): GT is a soft constraint anchor only; "
        "primary target is L[e_hat]=residual — never e_hat←(u*-u_base)."
    )
    lines.append(
        "Note: common.py RBFKansaSolver sign bug (Δu=f) was verified and fixed "
        "(now L=-Δ) so the base/surrogate spans are physics-consistent."
    )
    lines.append("=" * len(header))
    text = "\n".join(lines)
    print(text)
    return text


def run_seed(seed: int) -> dict:
    print(f"\n=== seed {seed} ===", flush=True)
    t0 = time.perf_counter()
    data = build_experiment(
        seed=seed,
        resolution=CFG["resolution"],
        gt_fraction=CFG["gt_fraction"],
        n_rbf_centers=CFG["n_rbf_centers"],
        epsilon=CFG["epsilon"],
        alpha=CFG["alpha"],
        beta=CFG["beta"],
    )
    print(
        f"  GT subset={len(data.gt_idx)}/{len(data.interior)} "
        f"({100*len(data.gt_idx)/len(data.interior):.1f}%); "
        f"u_base relL2={rel_l2(data.u_base, data.u_exact):.4e}; "
        f"v* relL2={rel_l2(data.v_star, data.u_exact):.4e}",
        flush=True,
    )

    r1 = train_arm1_compound(
        data,
        steps=CFG["steps"],
        lr=CFG["lr_arm1"],
        lambda_data=CFG["lambda_data"],
        width=CFG["width"],
        eval_every=CFG["eval_every"],
        colloc_batch=CFG["colloc_batch"],
    )
    print(
        f"  Arm1 compound-loss: relL2={r1['final_rel_l2']:.4f} "
        f"grad-cos={r1['mean_grad_cos']:.3f} (late={r1['late_grad_cos']:.3f}) "
        f"s/step={r1['mean_step_time']:.3f}",
        flush=True,
    )

    r2 = train_arm2_surrogate(
        data,
        steps=CFG["steps"],
        lr=CFG["lr_arm2"],
        width=CFG["width"],
        eval_every=CFG["eval_every"],
    )
    print(
        f"  Arm2 surrogate-target: relL2={r2['final_rel_l2']:.4f} "
        f"(v*={r2['v_star_rel_l2']:.4f}) "
        f"s/step={r2['mean_step_time']:.3f}",
        flush=True,
    )

    r3 = train_arm3_correction(
        data,
        steps=CFG["steps"],
        lr=CFG["lr_arm3"],
        width=CFG["width"],
        lambda_anchor=CFG["lambda_anchor"],
        eval_every=CFG["eval_every"],
        colloc_batch=CFG["colloc_batch"],
    )
    print(
        f"  Arm3 correction-field: relL2={r3['final_rel_l2']:.4f} "
        f"(base={r3['base_rel_l2']:.4e}) "
        f"s/step={r3['mean_step_time']:.3f}",
        flush=True,
    )

    s1 = summarize_arm(r1, data, compute_span=True)
    s2 = summarize_arm(r2, data, compute_span=True)
    s3 = summarize_arm(r3, data, compute_span=True)

    arm_preds = {
        "compound-loss": r1["u_pred"],
        "surrogate-target": r2["u_pred"],
        "correction-field": r3["u_pred"],
        "u_base": data.u_base,
        "v_star": data.v_star,
    }
    plot_solution_maps(data, arm_preds, FIGURES, seed)
    plot_error_maps(
        data,
        {
            "compound-loss": r1["u_pred"],
            "surrogate-target": r2["u_pred"],
            "correction-field": r3["u_pred"],
        },
        FIGURES,
        seed,
    )

    elapsed = time.perf_counter() - t0
    print(f"  seed {seed} done in {elapsed:.1f}s", flush=True)

    # Drop non-serializable model handles
    for r in (r1, r2, r3):
        r.pop("model", None)

    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "gt_count": int(len(data.gt_idx)),
        "surrogate_info": data.surrogate_info,
        "arms": {
            "compound-loss": {**r1, "summary": s1},
            "surrogate-target": {**r2, "summary": s2},
            "correction-field": {**r3, "summary": s3},
        },
        "summaries": {
            "compound-loss": s1,
            "surrogate-target": s2,
            "correction-field": s3,
        },
    }


def aggregate(seed_results: list[dict]) -> dict:
    agg = {}
    for arm in ["compound-loss", "surrogate-target", "correction-field"]:
        rels, times, cosines, inns, outs, stabs = [], [], [], [], [], []
        for sr in seed_results:
            s = sr["summaries"][arm]
            rels.append(s["final_rel_l2"])
            times.append(s["mean_step_time"])
            cosines.append(s.get("late_grad_cos", s["mean_grad_cos"]))
            inns.append(s.get("in_span_frac", float("nan")))
            outs.append(s.get("out_span_frac", float("nan")))
            stabs.append(s["stability"])
        # majority stability label
        mode = max(set(stabs), key=stabs.count)
        agg[arm] = {
            "rel_l2": rels,
            "step_time": times,
            "grad_cos": cosines,
            "in_span": inns,
            "out_span": outs,
            "stability_labels": stabs,
            "stability_mode": mode,
        }
    return agg


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in CFG["seeds"]:
        sr = run_seed(seed)
        # Persist per-seed (strip large u_pred arrays to keep JSON lighter? keep them)
        per = {
            "seed": sr["seed"],
            "elapsed_s": sr["elapsed_s"],
            "gt_count": sr["gt_count"],
            "surrogate_info": sr["surrogate_info"],
            "summaries": sr["summaries"],
            "histories": {
                arm: {
                    k: v
                    for k, v in sr["arms"][arm]["history"].items()
                }
                for arm in sr["arms"]
            },
            "final_rel_l2": {
                arm: sr["arms"][arm]["final_rel_l2"] for arm in sr["arms"]
            },
        }
        path = RESULTS / f"seed_{seed}.json"
        path.write_text(json.dumps(_jsonable(per), indent=2))
        print(f"  wrote {path}", flush=True)
        seed_results.append(sr)

    plot_training_curves(seed_results, FIGURES)

    agg = aggregate(seed_results)
    kill_text = print_kill_table(agg)

    consolidated = {
        "config": CFG,
        "per_seed": [
            {
                "seed": sr["seed"],
                "elapsed_s": sr["elapsed_s"],
                "summaries": sr["summaries"],
                "final_rel_l2": {
                    arm: sr["arms"][arm]["final_rel_l2"] for arm in sr["arms"]
                },
            }
            for sr in seed_results
        ],
        "aggregate": {
            arm: {
                "rel_l2_mean": float(np.mean(v["rel_l2"])),
                "rel_l2_std": float(np.std(v["rel_l2"])),
                "step_time_mean": float(np.mean(v["step_time"])),
                "step_time_std": float(np.std(v["step_time"])),
                "grad_cos_mean": float(np.mean(v["grad_cos"])),
                "grad_cos_std": float(np.std(v["grad_cos"])),
                "in_span_mean": float(np.nanmean(v["in_span"])),
                "out_span_mean": float(np.nanmean(v["out_span"])),
                "stability_mode": v["stability_mode"],
                "stability_labels": v["stability_labels"],
            }
            for arm, v in agg.items()
        },
        "kill_table_text": kill_text,
        "oracle_leak_audit": (
            "Arm3: GT only as soft constraint anchor ‖(u_base+e_hat)-u*‖_GT; "
            "primary loss is ‖L[e_hat]-residual‖². Never fitted e_hat to (u*-u_base) alone."
        ),
        "common_py_fix": (
            "RBFKansaSolver.fit previously used +Delta (solved Δu=f → ≈−u_exact). "
            "Verified and patched to L=−Δ before running this probe."
        ),
    }
    (RESULTS / "results.json").write_text(json.dumps(_jsonable(consolidated), indent=2))

    # CSV kill table
    csv_path = RESULTS / "kill_table.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "arm",
                "relL2_mean",
                "relL2_std",
                "step_time_mean",
                "grad_cos_mean",
                "in_span_mean",
                "out_span_mean",
                "stability",
            ]
        )
        for arm, v in consolidated["aggregate"].items():
            w.writerow(
                [
                    arm,
                    v["rel_l2_mean"],
                    v["rel_l2_std"],
                    v["step_time_mean"],
                    v["grad_cos_mean"],
                    v["in_span_mean"],
                    v["out_span_mean"],
                    v["stability_mode"],
                ]
            )
    (RESULTS / "kill_table.txt").write_text(kill_text + "\n")
    print(f"\nwrote {RESULTS / 'results.json'}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"figures in {FIGURES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
