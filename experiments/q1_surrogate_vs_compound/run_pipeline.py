#!/usr/bin/env python3
"""Q1 probe entrypoint: 5-row honest kill table after oracle correctness fixes.

Rows (order): rbf-base, v-star, surrogate-target, compound-loss, correction-field.

Run:
    .venv/bin/python run_pipeline.py

Writes results/ (per-seed + consolidated JSON/CSV) and figures/, prints kill table.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from diagnostics import summarize_analytic_row, summarize_arm
from plotting import (
    plot_corrected_kill_table,
    plot_error_maps,
    plot_solution_maps,
    plot_training_curves,
)
from setup_data import build_experiment, rel_l2
from train_arms import train_arm1_compound, train_arm2_surrogate, train_arm3_correction

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Kill-table row order (required).
ARM_ORDER = [
    "rbf-base",
    "v-star",
    "surrogate-target",
    "compound-loss",
    "correction-field",
]

# Budget tuned for CPU wall-clock of a few minutes while still separating arms.
CFG = {
    "seeds": [0, 1, 2],
    "resolution": 40,
    "gt_fraction": 0.12,
    "n_rbf_centers": 48,
    "epsilon": 1.5,
    "alpha": 1.0,
    "beta": 80.0,
    "steps": 3000,  # shared step budget across neural arms
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
    if obj is None:
        return None
    return obj


def _mean_std(vals: list[float], nan_as: str = "n/a") -> str:
    a = np.asarray(vals, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return nan_as
    return f"{np.nanmean(a):.4f}±{np.nanstd(a):.4f}"


def _fmt_span(vals_in: list[float], vals_out: list[float]) -> str:
    a = np.asarray(vals_in, dtype=float)
    b = np.asarray(vals_out, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return "n/a"
    return f"in={np.nanmean(a):.2f}±{np.nanstd(a):.2f} / out={np.nanmean(b):.2f}±{np.nanstd(b):.2f}"


def _fmt_cond(ranks: list, conds: list, ridges: list) -> str:
    if not ranks:
        return "n/a"
    r = np.asarray(ranks, dtype=float)
    c = np.asarray(conds, dtype=float)
    e = np.asarray(ridges, dtype=float)
    return (
        f"ε={np.nanmean(e):.0e} rank={np.nanmean(r):.0f}/{CFG['n_rbf_centers']} "
        f"cond={np.nanmean(c):.2e}"
    )


def print_kill_table(agg: dict) -> str:
    header = (
        f"{'arm':<20} {'relL2':<16} {'s/step':<12} {'grad-cos':<14} "
        f"{'span(b) u-u_base':<28} {'stability':<18} {'ridge rank/cond':<28}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        "=" * len(header),
        "Q1 CORRECTED KILL TABLE (5 rows)",
        "=" * len(header),
        header,
        sep,
    ]
    for arm in ARM_ORDER:
        a = agg[arm]
        grad = _mean_std(a["grad_cos"]) if arm not in ("rbf-base", "v-star") else "n/a"
        step = _mean_std(a["step_time"]) if arm not in ("rbf-base", "v-star") else "n/a"
        span = _fmt_span(a["in_span"], a["out_span"])
        cond = (
            _fmt_cond(a.get("ranks", []), a.get("conds", []), a.get("ridges", []))
            if arm in ("v-star", "surrogate-target")
            else "n/a"
        )
        row = (
            f"{arm:<20} {_mean_std(a['rel_l2']):<16} "
            f"{step:<12} "
            f"{grad:<14} "
            f"{span:<28} "
            f"{a['stability_mode']:<18} "
            f"{cond:<28}"
        )
        lines.append(row)
    lines.append(sep)

    # Span (a) always reported once: v* − u_base is in-span by construction
    v_in = agg["v-star"].get("span_a_in", [])
    if v_in:
        lines.append(
            f"span (a) [v*-u_base] w.r.t. shared RBF: "
            f"{_fmt_span(agg['v-star']['span_a_in'], agg['v-star']['span_a_out'])}  "
            f"(~1.0 in-span BY CONSTRUCTION — shared centers; not non-vacuity)"
        )
    lines.append(
        "span (b) column above = [u_pred-u_base] for neural/analytic row; "
        "MLP ~0.85 in-span is regression residual, NOT out-of-span gain."
    )

    base_m = float(np.mean(agg["rbf-base"]["rel_l2"]))
    vstar_m = float(np.mean(agg["v-star"]["rel_l2"]))
    surr_m = float(np.mean(agg["surrogate-target"]["rel_l2"]))
    comp_m = float(np.mean(agg["compound-loss"]["rel_l2"]))
    corr_m = float(np.mean(agg["correction-field"]["rel_l2"]))
    neural = {"surrogate-target": surr_m, "compound-loss": comp_m, "correction-field": corr_m}
    best_neural_name = min(neural, key=neural.get)
    best_neural = neural[best_neural_name]
    if base_m <= best_neural:
        base_vs = (
            f"RBF base ({base_m:.4e}) already beats neural arms "
            f"(best={best_neural_name} {best_neural:.4e}) on this smooth toy."
        )
    else:
        base_vs = (
            f"RBF base ({base_m:.4e}); best neural={best_neural_name} "
            f"({best_neural:.4e}) — correction may beat base after sign fix."
        )
    lines.append(
        f"Framing: analytic projection is a stable distill target "
        f"(v*={vstar_m:.4e} → MLP={surr_m:.4e}); {base_vs}"
    )
    lines.append(
        "Oracle-leak (Arm3): GT is a soft constraint anchor only; "
        "primary target is L[e_hat]=−residual (sign-fixed) — never e_hat←(u*-u_base)."
    )
    lines.append(
        "Arm3 sign fix: residual=L[u_base]−f; train ‖−Δe + residual‖² so L[e]=−residual."
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
    sinfo = data.surrogate_info
    print(
        f"  GT subset={len(data.gt_idx)}/{len(data.interior)} "
        f"({100 * len(data.gt_idx) / len(data.interior):.1f}%); "
        f"u_base relL2={rel_l2(data.u_base, data.u_exact):.4e}; "
        f"v* relL2={rel_l2(data.v_star, data.u_exact):.4e}; "
        f"ridge ε={sinfo.get('ridge', float('nan')):.0e} "
        f"rank={sinfo.get('rank')}/{sinfo.get('n_centers')} "
        f"cond={sinfo.get('cond_est', float('nan')):.3e} "
        f"(unridged cond={sinfo.get('cond_unridged', float('nan')):.3e})",
        flush=True,
    )

    s0 = summarize_analytic_row("rbf-base", data.u_base, data)
    s_v = summarize_analytic_row("v-star", data.v_star, data)
    print(
        f"  Arm0 rbf-base: relL2={s0['final_rel_l2']:.4e}",
        flush=True,
    )
    print(
        f"  Arm2a v-star: relL2={s_v['final_rel_l2']:.4e} "
        f"span(a) in={s_v['span_vstar_minus_base']['in_span_frac']:.3f}",
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
    if r3["final_rel_l2"] > r3["base_rel_l2"]:
        print(
            f"  WARNING: arm3 still worse than frozen base "
            f"({r3['final_rel_l2']:.4e} > {r3['base_rel_l2']:.4e})",
            flush=True,
        )
    else:
        print(
            f"  arm3 check: does not degrade below base "
            f"({r3['final_rel_l2']:.4e} ≤ {r3['base_rel_l2']:.4e})",
            flush=True,
        )

    s1 = summarize_arm(r1, data, compute_span=True)
    s2 = summarize_arm(r2, data, compute_span=True)
    s3 = summarize_arm(r3, data, compute_span=True)

    arm_preds = {
        "rbf-base": data.u_base,
        "v-star": data.v_star,
        "compound-loss": r1["u_pred"],
        "surrogate-target": r2["u_pred"],
        "correction-field": r3["u_pred"],
    }
    plot_solution_maps(data, arm_preds, FIGURES, seed)
    plot_error_maps(
        data,
        {
            "rbf-base": data.u_base,
            "v-star": data.v_star,
            "compound-loss": r1["u_pred"],
            "surrogate-target": r2["u_pred"],
            "correction-field": r3["u_pred"],
        },
        FIGURES,
        seed,
    )

    elapsed = time.perf_counter() - t0
    print(f"  seed {seed} done in {elapsed:.1f}s", flush=True)

    for r in (r1, r2, r3):
        r.pop("model", None)

    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "gt_count": int(len(data.gt_idx)),
        "surrogate_info": data.surrogate_info,
        "arms": {
            "rbf-base": {
                "arm": "rbf-base",
                "u_pred": data.u_base,
                "final_rel_l2": s0["final_rel_l2"],
                "history": {"loss": [], "rel_l2": [], "grad_cos": [], "step_time": []},
                "summary": s0,
            },
            "v-star": {
                "arm": "v-star",
                "u_pred": data.v_star,
                "final_rel_l2": s_v["final_rel_l2"],
                "history": {"loss": [], "rel_l2": [], "grad_cos": [], "step_time": []},
                "summary": s_v,
            },
            "compound-loss": {**r1, "summary": s1},
            "surrogate-target": {**r2, "summary": s2},
            "correction-field": {**r3, "summary": s3},
        },
        "summaries": {
            "rbf-base": s0,
            "v-star": s_v,
            "compound-loss": s1,
            "surrogate-target": s2,
            "correction-field": s3,
        },
    }


def aggregate(seed_results: list[dict]) -> dict:
    agg = {}
    for arm in ARM_ORDER:
        rels, times, cosines, inns, outs, stabs = [], [], [], [], [], []
        ranks, conds, ridges = [], [], []
        span_a_in, span_a_out = [], []
        for sr in seed_results:
            s = sr["summaries"][arm]
            rels.append(s["final_rel_l2"])
            times.append(s["mean_step_time"])
            cosines.append(s.get("late_grad_cos", s["mean_grad_cos"]))
            inns.append(s.get("in_span_frac", float("nan")))
            outs.append(s.get("out_span_frac", float("nan")))
            stabs.append(s["stability"])
            sa = s.get("span_vstar_minus_base") or {}
            if sa:
                span_a_in.append(sa.get("in_span_frac", float("nan")))
                span_a_out.append(sa.get("out_span_frac", float("nan")))
            info = sr.get("surrogate_info") or {}
            if arm in ("v-star", "surrogate-target"):
                ranks.append(info.get("rank", float("nan")))
                conds.append(info.get("cond_est", float("nan")))
                ridges.append(info.get("ridge", float("nan")))
        mode = max(set(stabs), key=stabs.count)
        agg[arm] = {
            "rel_l2": rels,
            "step_time": times,
            "grad_cos": cosines,
            "in_span": inns,
            "out_span": outs,
            "stability_labels": stabs,
            "stability_mode": mode,
            "ranks": ranks,
            "conds": conds,
            "ridges": ridges,
            "span_a_in": span_a_in,
            "span_a_out": span_a_out,
        }
    return agg


def _safe_nanmean(vals: list) -> float | None:
    a = np.asarray(vals, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return None
    return float(np.nanmean(a))


def _safe_nanstd(vals: list) -> float | None:
    a = np.asarray(vals, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return None
    return float(np.nanstd(a))


def parse_args():
    p = argparse.ArgumentParser(description="Q1 corrected pipeline")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override shared neural-arm step budget (for smoke tests)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seed list (default: 0 1 2)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps is not None:
        CFG["steps"] = args.steps
    if args.seeds is not None:
        CFG["seeds"] = args.seeds

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in CFG["seeds"]:
        sr = run_seed(seed)
        per = {
            "seed": sr["seed"],
            "elapsed_s": sr["elapsed_s"],
            "gt_count": sr["gt_count"],
            "surrogate_info": sr["surrogate_info"],
            "summaries": sr["summaries"],
            "histories": {
                arm: {k: v for k, v in sr["arms"][arm]["history"].items()}
                for arm in sr["arms"]
                if sr["arms"][arm]["history"].get("loss") is not None
            },
            "final_rel_l2": {
                arm: sr["arms"][arm]["final_rel_l2"] for arm in ARM_ORDER
            },
        }
        path = RESULTS / f"seed_{seed}.json"
        path.write_text(json.dumps(_jsonable(per), indent=2))
        print(f"  wrote {path}", flush=True)
        seed_results.append(sr)

    plot_training_curves(seed_results, FIGURES)

    agg = aggregate(seed_results)
    kill_text = print_kill_table(agg)
    plot_corrected_kill_table(agg, FIGURES)

    consolidated = {
        "config": CFG,
        "arm_order": ARM_ORDER,
        "per_seed": [
            {
                "seed": sr["seed"],
                "elapsed_s": sr["elapsed_s"],
                "surrogate_info": sr["surrogate_info"],
                "summaries": sr["summaries"],
                "final_rel_l2": {
                    arm: sr["arms"][arm]["final_rel_l2"] for arm in ARM_ORDER
                },
            }
            for sr in seed_results
        ],
        "aggregate": {
            arm: {
                "rel_l2_mean": _safe_nanmean(v["rel_l2"]),
                "rel_l2_std": _safe_nanstd(v["rel_l2"]),
                "step_time_mean": _safe_nanmean(v["step_time"]),
                "step_time_std": _safe_nanstd(v["step_time"]),
                "grad_cos_mean": _safe_nanmean(v["grad_cos"]),
                "grad_cos_std": _safe_nanstd(v["grad_cos"]),
                "in_span_mean": _safe_nanmean(v["in_span"]),
                "out_span_mean": _safe_nanmean(v["out_span"]),
                "stability_mode": v["stability_mode"],
                "stability_labels": v["stability_labels"],
                "ridge_mean": _safe_nanmean(v["ridges"]),
                "rank_mean": _safe_nanmean(v["ranks"]),
                "cond_mean": _safe_nanmean(v["conds"]),
                "span_a_in_mean": _safe_nanmean(v["span_a_in"]),
            }
            for arm, v in agg.items()
        },
        "kill_table_text": kill_text,
        "framing": (
            "Q1 narrows to: an analytic function-space projection is a stable "
            "distillation target for a neural surrogate; the RBF base itself "
            "already beats the neural arms on this smooth toy."
        ),
        "oracle_leak_audit": (
            "Arm3: GT only as soft constraint anchor ‖(u_base+e_hat)-u*‖_GT; "
            "primary loss is ‖L[e_hat]+residual‖² with residual=L[u_base]−f "
            "(sign-fixed: L[e]=−residual). Never fitted e_hat to (u*-u_base) alone."
        ),
        "span_audit": (
            "Base and surrogate share the same RBF centers, so span(v*-u_base) "
            "is ~1.0 in-span by construction. Neural-arm span(u_mlp-u_base) "
            "in-span fractions reflect MLP regression residual, not "
            "out-of-span gain / surrogate non-vacuity."
        ),
        "arm3_sign_fix": (
            "Previously trained L[e]=residual; correct is L[e]=−residual so "
            "L[u_base+e]=f. Loss changed from (−Δe−residual)² to (−Δe+residual)²."
        ),
    }
    (RESULTS / "results.json").write_text(json.dumps(_jsonable(consolidated), indent=2))

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
                "in_span_mean_b",
                "out_span_mean_b",
                "stability",
                "ridge_mean",
                "rank_mean",
                "cond_mean",
            ]
        )
        for arm in ARM_ORDER:
            v = consolidated["aggregate"][arm]
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
                    v.get("ridge_mean"),
                    v.get("rank_mean"),
                    v.get("cond_mean"),
                ]
            )
    (RESULTS / "kill_table.txt").write_text(kill_text + "\n")
    print(f"\nwrote {RESULTS / 'results.json'}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"figures in {FIGURES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
