#!/usr/bin/env python3
"""Q1 probe entrypoint: 7-row honest kill table after oracle correctness fixes.

Rows (order): rbf-base, rbf-grad, rbf-shape, v-star, surrogate-target,
compound-loss, correction-field.

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
from train_arms import (
    train_arm1_compound,
    train_arm2_surrogate,
    train_arm3_correction,
    train_arm_rbf_grad,
    train_arm_rbf_shape,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Kill-table row order (required).
ARM_ORDER = [
    "rbf-base",
    "rbf-grad",
    "rbf-shape",
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
    "steps": 3000,  # shared step budget across neural + rbf-grad/shape arms
    "lambda_data": 500.0,
    "lambda_anchor": 5.0,
    "lambda_bc_rbf_grad": 1e4,  # match Kansa BC weight
    "width": 128,
    "lr_arm1": 2e-3,
    "lr_arm2": 5e-3,
    "lr_arm3": 2e-3,
    # O(1) reparam w̃=w/σ; Adam lr on w̃ is lr/σ so CFG lr = effective |Δw| scale.
    # 1e-2/5e-3/1e-3 on w̃ diverge (Δw_eff≈σ·lr). Largest stable effective lr ~2e-6.
    "lr_rbf_grad": 2e-6,
    "lr_rbf_grad_centers": 1e-8,
    # rbf-shape: same weight reparam + effective lr as rbf-grad; shape params
    # (logσ, angle) are O(1) but extremely stiff — 1e-2..1e-3 diverge hard;
    # largest stable lr_shape ~1e-7; use 5e-8 (best short-budget residual drop).
    "lr_rbf_shape": 2e-6,
    "lr_rbf_shape_shape": 5e-8,
    "lr_rbf_shape_centers": 1e-8,
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


def _rbf_grad_noise_gate(agg: dict) -> dict:
    """Honest significance: mean sep must exceed pooled std across seeds."""
    base_arr = np.asarray(agg["rbf-base"]["rel_l2"], dtype=float)
    rbfg_arr = np.asarray(agg["rbf-grad"]["rel_l2"], dtype=float)
    base_m = float(np.mean(base_arr))
    base_s = float(np.std(base_arr))
    rbfg_m = float(np.mean(rbfg_arr))
    rbfg_s = float(np.std(rbfg_arr))
    sep = base_m - rbfg_m  # >0 ⇒ rbf-grad mean better (lower rel-L2)
    pooled_std = float(np.sqrt(base_s**2 + rbfg_s**2))
    return {
        "rbf_base_mean": base_m,
        "rbf_base_std": base_s,
        "rbf_grad_mean": rbfg_m,
        "rbf_grad_std": rbfg_s,
        "sep": sep,
        "pooled_std": pooled_std,
        "beats_beyond_noise": bool(sep > pooled_std),
    }


def _rbf_shape_noise_gates(agg: dict) -> dict:
    """rbf-shape vs rbf-base and vs isotropic rbf-grad (sep > pooled_std)."""
    base_arr = np.asarray(agg["rbf-base"]["rel_l2"], dtype=float)
    rbfg_arr = np.asarray(agg["rbf-grad"]["rel_l2"], dtype=float)
    shape_arr = np.asarray(agg["rbf-shape"]["rel_l2"], dtype=float)
    base_m, base_s = float(np.mean(base_arr)), float(np.std(base_arr))
    rbfg_m, rbfg_s = float(np.mean(rbfg_arr)), float(np.std(rbfg_arr))
    shape_m, shape_s = float(np.mean(shape_arr)), float(np.std(shape_arr))
    sep_base = base_m - shape_m
    pooled_base = float(np.sqrt(base_s**2 + shape_s**2))
    sep_grad = rbfg_m - shape_m
    pooled_grad = float(np.sqrt(rbfg_s**2 + shape_s**2))
    return {
        "rbf_shape_mean": shape_m,
        "rbf_shape_std": shape_s,
        "vs_base": {
            "sep": sep_base,
            "pooled_std": pooled_base,
            "beats_beyond_noise": bool(sep_base > pooled_base),
        },
        "vs_rbf_grad": {
            "sep": sep_grad,
            "pooled_std": pooled_grad,
            "beats_beyond_noise": bool(sep_grad > pooled_grad),
        },
    }


def print_kill_table(agg: dict) -> str:
    header = (
        f"{'arm':<20} {'relL2':<16} {'s/step':<12} {'grad-cos':<14} "
        f"{'span(b) u-u_base':<28} {'stability':<18} {'ridge rank/cond':<28}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        "=" * len(header),
        "Q1 CORRECTED KILL TABLE (7 rows)",
        "=" * len(header),
        header,
        sep,
    ]
    analytic_like = ("rbf-base", "rbf-grad", "rbf-shape", "v-star")
    for arm in ARM_ORDER:
        a = agg[arm]
        grad = (
            _mean_std(a["grad_cos"])
            if arm not in analytic_like
            else "n/a"
        )
        step = (
            _mean_std(a["step_time"])
            if arm not in ("rbf-base", "v-star")
            else "n/a"
        )
        span = _fmt_span(a["in_span"], a["out_span"])
        if arm in ("v-star", "surrogate-target"):
            cond = _fmt_cond(
                a.get("ranks", []), a.get("conds", []), a.get("ridges", [])
            )
        elif arm == "rbf-shape":
            notes = a.get("shape_notes", [])
            cond = notes[0] if notes else "shape-refine"
        else:
            cond = "n/a"
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

    gate = _rbf_grad_noise_gate(agg)
    shape_gates = _rbf_shape_noise_gates(agg)
    base_m = gate["rbf_base_mean"]
    base_s = gate["rbf_base_std"]
    rbfg_m = gate["rbf_grad_mean"]
    rbfg_s = gate["rbf_grad_std"]
    sep_g = gate["sep"]
    pooled_std = gate["pooled_std"]
    beats_beyond_noise = gate["beats_beyond_noise"]
    shape_m = shape_gates["rbf_shape_mean"]
    shape_s = shape_gates["rbf_shape_std"]
    vb = shape_gates["vs_base"]
    vg = shape_gates["vs_rbf_grad"]
    vstar_m = float(np.mean(agg["v-star"]["rel_l2"]))
    surr_m = float(np.mean(agg["surrogate-target"]["rel_l2"]))
    comp_m = float(np.mean(agg["compound-loss"]["rel_l2"]))
    corr_m = float(np.mean(agg["correction-field"]["rel_l2"]))
    neural = {"surrogate-target": surr_m, "compound-loss": comp_m, "correction-field": corr_m}
    best_neural_name = min(neural, key=neural.get)
    best_neural = neural[best_neural_name]
    lines.append(
        f"rbf-grad beats base beyond noise: {beats_beyond_noise} "
        f"(sep={sep_g:+.4e}, pooled_std={pooled_std:.4e}; "
        f"rbf-grad={rbfg_m:.4e}±{rbfg_s:.4e} vs rbf-base={base_m:.4e}±{base_s:.4e})"
    )
    lines.append(
        f"rbf-shape beats rbf-base beyond noise: {vb['beats_beyond_noise']} "
        f"(sep={vb['sep']:+.4e}, pooled_std={vb['pooled_std']:.4e}; "
        f"rbf-shape={shape_m:.4e}±{shape_s:.4e} vs rbf-base={base_m:.4e}±{base_s:.4e})"
    )
    lines.append(
        f"rbf-shape beats rbf-grad beyond noise: {vg['beats_beyond_noise']} "
        f"(sep={vg['sep']:+.4e}, pooled_std={vg['pooled_std']:.4e}; "
        f"rbf-shape={shape_m:.4e}±{shape_s:.4e} vs rbf-grad={rbfg_m:.4e}±{rbfg_s:.4e})"
    )
    if beats_beyond_noise:
        grad_vs = (
            f"rbf-grad ({rbfg_m:.4e}±{rbfg_s:.4e}) beats one-shot base "
            f"({base_m:.4e}±{base_s:.4e}) beyond noise (sep={sep_g:+.4e} > "
            f"pooled_std={pooled_std:.4e})."
        )
    elif sep_g > 0:
        grad_vs = (
            f"rbf-grad mean ({rbfg_m:.4e}±{rbfg_s:.4e}) is numerically below base "
            f"({base_m:.4e}±{base_s:.4e}) but NOT beyond noise "
            f"(sep={sep_g:+.4e} ≤ pooled_std={pooled_std:.4e}) — within seed scatter."
        )
    else:
        grad_vs = (
            f"rbf-grad ({rbfg_m:.4e}±{rbfg_s:.4e}) does NOT beat one-shot base "
            f"({base_m:.4e}±{base_s:.4e}; sep={sep_g:+.4e}, "
            f"pooled_std={pooled_std:.4e}) — residual-gradient center refine "
            "is not free accuracy on this toy."
        )
    if vb["beats_beyond_noise"]:
        shape_vs = (
            f"rbf-shape ({shape_m:.4e}±{shape_s:.4e}) beats base beyond noise."
        )
    else:
        shape_vs = (
            f"rbf-shape ({shape_m:.4e}±{shape_s:.4e}) does NOT beat base beyond noise "
            f"(sep={vb['sep']:+.4e}, pooled_std={vb['pooled_std']:.4e})."
        )
    if vg["beats_beyond_noise"]:
        shape_vs += (
            f" Beats isotropic rbf-grad ({rbfg_m:.4e}±{rbfg_s:.4e}) beyond noise."
        )
    else:
        shape_vs += (
            f" Does NOT beat isotropic rbf-grad beyond noise "
            f"(sep={vg['sep']:+.4e}, pooled_std={vg['pooled_std']:.4e})."
        )
    if base_m <= best_neural and rbfg_m <= best_neural and shape_m <= best_neural:
        base_vs = (
            f"Classical RBF rows already beat neural arms "
            f"(best={best_neural_name} {best_neural:.4e}) on this smooth toy."
        )
    else:
        base_vs = (
            f"best neural={best_neural_name} ({best_neural:.4e}); "
            f"base={base_m:.4e}; rbf-grad={rbfg_m:.4e}; rbf-shape={shape_m:.4e}."
        )
    lines.append(
        f"Framing: analytic projection is a stable distill target "
        f"(v*={vstar_m:.4e} → MLP={surr_m:.4e}); {grad_vs} {shape_vs} {base_vs}"
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

    r_grad = train_arm_rbf_grad(
        data,
        steps=CFG["steps"],
        lr=CFG["lr_rbf_grad"],
        lr_centers=CFG["lr_rbf_grad_centers"],
        lambda_bc=CFG["lambda_bc_rbf_grad"],
        eval_every=CFG["eval_every"],
        colloc_batch=CFG["colloc_batch"],
    )
    print(
        f"  Arm0b rbf-grad: relL2={r_grad['final_rel_l2']:.4e} "
        f"(base={r_grad['base_rel_l2']:.4e}, "
        f"Δ={r_grad['rel_l2_delta_vs_base']:+.4e}, "
        f"lr_eff={r_grad['lr']:g}, lr_c={r_grad['lr_centers']:g}, "
        f"mean|Δw|={r_grad['mean_weight_displacement']:.3e}, "
        f"mean|Δc|={r_grad['mean_center_displacement']:.3e}) "
        f"s/step={r_grad['mean_step_time']:.3f}",
        flush=True,
    )

    r_shape = train_arm_rbf_shape(
        data,
        steps=CFG["steps"],
        lr=CFG["lr_rbf_shape"],
        lr_centers=CFG["lr_rbf_shape_centers"],
        lr_shape=CFG["lr_rbf_shape_shape"],
        lambda_bc=CFG["lambda_bc_rbf_grad"],
        eval_every=CFG["eval_every"],
    )
    sr = r_shape.get("shape_range") or {}
    print(
        f"  Arm0c rbf-shape: relL2={r_shape['final_rel_l2']:.4e} "
        f"(base={r_shape['base_rel_l2']:.4e}, "
        f"Δ={r_shape['rel_l2_delta_vs_base']:+.4e}, "
        f"lr_eff={r_shape['lr']:g}, lr_shape={r_shape['lr_shape']:g}, "
        f"mean|Δlogσ|={r_shape['mean_dlog_sigma']:.3e}, "
        f"mean|Δ∠|={r_shape['mean_dangle']:.3e}, "
        f"aspect_max={sr.get('aspect_max', float('nan')):.2e}, "
        f"collapsed={r_shape.get('shape_collapsed')}) "
        f"s/step={r_shape['mean_step_time']:.3f}",
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

    s_grad = summarize_arm(r_grad, data, compute_span=True)
    s_shape = summarize_arm(r_shape, data, compute_span=True)
    s1 = summarize_arm(r1, data, compute_span=True)
    s2 = summarize_arm(r2, data, compute_span=True)
    s3 = summarize_arm(r3, data, compute_span=True)

    arm_preds = {
        "rbf-base": data.u_base,
        "rbf-grad": r_grad["u_pred"],
        "rbf-shape": r_shape["u_pred"],
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
            "rbf-grad": r_grad["u_pred"],
            "rbf-shape": r_shape["u_pred"],
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

    for r in (r_grad, r_shape, r1, r2, r3):
        r.pop("model", None)
        r.pop("centers_final", None)
        r.pop("weights_final", None)
        r.pop("log_sigma_x_final", None)
        r.pop("log_sigma_y_final", None)
        r.pop("angles_final", None)

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
            "rbf-grad": {**r_grad, "summary": s_grad},
            "rbf-shape": {**r_shape, "summary": s_shape},
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
            "rbf-grad": s_grad,
            "rbf-shape": s_shape,
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
        shape_notes = []
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
            if arm == "rbf-shape":
                srng = s.get("shape_range") or {}
                collapsed = s.get("shape_collapsed", False)
                note = (
                    f"aspect≤{srng.get('aspect_max', float('nan')):.1f} "
                    f"∠∈[{srng.get('angle_min', float('nan')):.2f},"
                    f"{srng.get('angle_max', float('nan')):.2f}]"
                    + (" COLLAPSED" if collapsed else "")
                )
                shape_notes.append(note)
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
            "shape_notes": shape_notes,
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
            "distillation target for a neural surrogate; the classical RBF rows "
            "(one-shot Kansa + residual-gradient refine + anisotropic shape "
            "refine) already beat the neural arms on this smooth toy. rbf-grad "
            "asks whether Adam center/weight refinement beats the one-shot "
            "Kansa base; rbf-shape asks whether learnable anisotropic shape "
            "(ePIL/VSD) adds anything beyond isotropic rbf-grad from the same "
            "init. Claim 'beats X' only if mean sep > pooled std."
        ),
        "rbf_grad_note": (
            f"rbf-grad: O(1) reparam w̃=w_base/σ (σ=max(|w_base|) frozen; "
            f"u=Σ (σ·w̃)φ); Adam lr_eff={CFG['lr_rbf_grad']} on weights "
            f"(lr_w̃=lr_eff/σ), lr_centers={CFG['lr_rbf_grad_centers']}; "
            f"1e-2/5e-3/1e-3 on w̃ diverge because Δw_eff≈σ·lr. "
            f"Analytical Δ; lambda_bc={CFG['lambda_bc_rbf_grad']}, "
            f"steps={CFG['steps']}; float64; seeded from shared Kansa "
            "centers+weights (ε fixed); full-interior collocation. "
            "grad-cos / ridge columns are n/a. "
            "beats-base gated by sep > pooled_std across seeds."
        ),
        "rbf_shape_note": (
            f"rbf-shape: ePIL/VSD anisotropic kernel "
            f"(μ, logσ_x, logσ_y, angle, O(1) w̃); init σ=1/(ε√2) so "
            f"φ=exp(−ε²r²) matches Kansa at step 0. "
            f"Adam lr_eff={CFG['lr_rbf_shape']}, "
            f"lr_shape={CFG['lr_rbf_shape_shape']} "
            f"(1e-2..1e-3 diverge; largest stable ~1e-7), "
            f"lr_centers={CFG['lr_rbf_shape_centers']}; "
            f"autograd Δ; same λ_bc/steps as rbf-grad. "
            "Gates: beats base / beats rbf-grad only if sep > pooled_std."
        ),
        "rbf_grad_beats_base_beyond_noise": _rbf_grad_noise_gate(agg),
        "rbf_shape_beats_beyond_noise": _rbf_shape_noise_gates(agg),
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
