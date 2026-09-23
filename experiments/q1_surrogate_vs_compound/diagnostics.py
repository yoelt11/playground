"""Diagnostics: gradient conflict, in/out-of-span gain, stability."""

from __future__ import annotations

import numpy as np

from setup_data import ExperimentData, rel_l2


def in_out_span_fraction(
    u_pred: np.ndarray,
    u_base: np.ndarray,
    basis_phi: np.ndarray,
) -> dict:
    """Decompose (u_pred - u_base) into span(RBF) vs orthogonal.

    Returns fractions of ||delta||² explained by the RBF span projection.
    """
    improvement = u_pred - u_base
    norm2 = float(np.dot(improvement, improvement))
    if norm2 < 1e-16:
        return {
            "in_span_frac": 0.0,
            "out_span_frac": 0.0,
            "gain_rel_l2": 0.0,
            "improvement_norm": 0.0,
        }

    # proj = Φ pinv(Φ) improvement
    coeffs = np.linalg.lstsq(basis_phi, improvement, rcond=None)[0]
    proj = basis_phi @ coeffs
    orth = improvement - proj
    in_norm2 = float(np.dot(proj, proj))
    out_norm2 = float(np.dot(orth, orth))
    return {
        "in_span_frac": in_norm2 / norm2,
        "out_span_frac": out_norm2 / norm2,
        "improvement_norm": float(np.sqrt(norm2)),
        "in_span_norm": float(np.sqrt(in_norm2)),
        "out_span_norm": float(np.sqrt(out_norm2)),
    }


def span_decompositions(
    u_pred: np.ndarray | None,
    data: ExperimentData,
) -> dict:
    """Two separately labelled span decompositions (do not conflate).

    (a) span(w.r.t. base RBF basis) of [v* − u_base]:
        Base and surrogate share the SAME RBF centers, so this is ~1.0
        in-span BY CONSTRUCTION. Not evidence of non-vacuous correction.

    (b) span of [u_mlp − u_base] for a neural/correction prediction:
        Reflects MLP regression residual vs the shared RBF span — NOT
        'out-of-span gain' / surrogate non-vacuity.
    """
    span_vstar = in_out_span_fraction(data.v_star, data.u_base, data.basis_phi)
    out = {
        "span_vstar_minus_base": {
            **span_vstar,
            "label": "span(v*-u_base) w.r.t. shared RBF basis",
            "note": (
                "In-span by construction: v* and u_base share the same RBF "
                "centers. Do not treat this as non-vacuous out-of-span gain."
            ),
        },
    }
    if u_pred is not None:
        span_mlp = in_out_span_fraction(u_pred, data.u_base, data.basis_phi)
        out["span_mlp_minus_base"] = {
            **span_mlp,
            "label": "span(u_mlp-u_base) w.r.t. shared RBF basis",
            "note": (
                "In-span fraction reflects MLP regression residual onto the "
                "shared RBF span, not surrogate non-vacuity. Analytic v* is "
                "already in the base span."
            ),
        }
    return out


def stability_verdict(loss_hist: list[float], rel_hist: list[float]) -> dict:
    """Classify trajectory as smooth vs oscillating.

    Heuristic: fraction of upward loss steps + CV of finite differences.
    """
    loss = np.asarray(loss_hist, dtype=float)
    rel = np.asarray(rel_hist, dtype=float)
    if len(loss) < 4:
        return {"verdict": "unknown", "up_frac": 0.0, "loss_cv_diff": 0.0}

    dloss = np.diff(loss)
    up_frac = float(np.mean(dloss > 0))
    # Coefficient of variation of |Δloss| (high ⇒ jagged)
    abs_d = np.abs(dloss)
    loss_cv = float(abs_d.std() / (abs_d.mean() + 1e-12))

    # Rel-L2 late-window oscillation
    late = rel[len(rel) // 2 :]
    rel_osc = float(np.std(np.diff(late))) if len(late) > 2 else 0.0

    if up_frac > 0.35 and loss_cv > 1.0:
        verdict = "oscillating"
    elif up_frac > 0.28 or rel_osc > 5e-3:
        verdict = "mildly-oscillating"
    else:
        verdict = "smooth"

    return {
        "verdict": verdict,
        "up_frac": up_frac,
        "loss_cv_diff": loss_cv,
        "rel_l2_late_osc": rel_osc,
    }


def summarize_analytic_row(
    arm: str,
    u_pred: np.ndarray,
    data: ExperimentData,
) -> dict:
    """Summary for frozen analytic rows (rbf-base, v-star): no SGD metrics."""
    spans = span_decompositions(data.v_star if arm == "v-star" else None, data)
    if arm == "v-star":
        spans = span_decompositions(data.v_star, data)
        spans["span_mlp_minus_base"] = {
            **spans["span_vstar_minus_base"],
            "label": "span(v*-u_base) [= analytic; same as (a)]",
            "note": (
                "v* lies in the shared RBF span by construction; MLP not involved."
            ),
        }
    out = {
        "arm": arm,
        "final_rel_l2": rel_l2(u_pred, data.u_exact),
        "mean_step_time": float("nan"),
        "mean_grad_cos": float("nan"),
        "late_grad_cos": float("nan"),
        "stability": "analytic",
        "stability_detail": {"verdict": "analytic"},
        "base_rel_l2": rel_l2(data.u_base, data.u_exact),
        "in_span_frac": spans["span_vstar_minus_base"]["in_span_frac"]
        if arm == "v-star"
        else float("nan"),
        "out_span_frac": spans["span_vstar_minus_base"]["out_span_frac"]
        if arm == "v-star"
        else float("nan"),
        "span_vstar_minus_base": spans["span_vstar_minus_base"],
        "span_mlp_minus_base": spans.get("span_mlp_minus_base"),
        "span_note": (
            "Report (a) span(v*-u_base) and (b) span(u_mlp-u_base) separately. "
            "v*-u_base is in-span by construction (shared RBF centers)."
        ),
    }
    return out


def summarize_arm(
    result: dict,
    data: ExperimentData,
    compute_span: bool = True,
) -> dict:
    hist = result["history"]
    stab = stability_verdict(hist["loss"], hist["rel_l2"])
    out = {
        "arm": result["arm"],
        "final_rel_l2": result["final_rel_l2"],
        "mean_step_time": result["mean_step_time"],
        "mean_grad_cos": result["mean_grad_cos"],
        "late_grad_cos": result.get("late_grad_cos", result["mean_grad_cos"]),
        "stability": stab["verdict"],
        "stability_detail": stab,
        "base_rel_l2": rel_l2(data.u_base, data.u_exact),
    }
    if compute_span:
        spans = span_decompositions(result["u_pred"], data)
        out["span_vstar_minus_base"] = spans["span_vstar_minus_base"]
        out["span_mlp_minus_base"] = spans["span_mlp_minus_base"]
        # Table "in/out-span" for neural arms = (b) u_mlp − u_base
        out["in_span_frac"] = spans["span_mlp_minus_base"]["in_span_frac"]
        out["out_span_frac"] = spans["span_mlp_minus_base"]["out_span_frac"]
        out["span_note"] = (
            "(a) span(v*-u_base) is ~1 in-span by construction (shared RBF "
            "centers). (b) span(u_mlp-u_base) in-span reflects MLP regression "
            "residual — NOT out-of-span gain / surrogate non-vacuity."
        )
    else:
        out["in_span_frac"] = float("nan")
        out["out_span_frac"] = float("nan")

    if "oracle_leak_audit" in result:
        out["oracle_leak_audit"] = result["oracle_leak_audit"]
    if "v_star_rel_l2" in result:
        out["v_star_rel_l2"] = result["v_star_rel_l2"]
    if "debug_pde_check" in result:
        out["debug_pde_check"] = result["debug_pde_check"]
    # rbf-grad / rbf-shape train-moved + shape diagnostics (passthrough)
    for key in (
        "mean_weight_displacement",
        "mean_center_displacement",
        "mean_dlog_sigma",
        "mean_dangle",
        "shape_moved",
        "shape_range",
        "shape_collapsed",
        "trained_non_vacuous",
        "diverged",
        "rel_l2_delta_vs_base",
        "init_match_rel_err",
        "lr",
        "lr_shape",
        "lr_centers",
    ):
        if key in result:
            out[key] = result[key]
    return out
