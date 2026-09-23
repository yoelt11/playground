"""Diagnostics: gradient conflict, in/out-of-span gain, stability."""

from __future__ import annotations

import numpy as np

from setup_data import ExperimentData, rel_l2


def in_out_span_fraction(
    u_pred: np.ndarray,
    u_base: np.ndarray,
    basis_phi: np.ndarray,
) -> dict:
    """Decompose improvement (u_pred - u_base) into span(RBF) vs orthogonal.

    Returns fractions of ||improvement||² explained by the RBF span projection.
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
        span = in_out_span_fraction(result["u_pred"], data.u_base, data.basis_phi)
        out["in_span_frac"] = span["in_span_frac"]
        out["out_span_frac"] = span["out_span_frac"]
        out["span_detail"] = span
    else:
        out["in_span_frac"] = float("nan")
        out["out_span_frac"] = float("nan")

    if "oracle_leak_audit" in result:
        out["oracle_leak_audit"] = result["oracle_leak_audit"]
    if "v_star_rel_l2" in result:
        out["v_star_rel_l2"] = result["v_star_rel_l2"]
    return out
