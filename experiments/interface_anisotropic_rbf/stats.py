"""Cell-level statistics: mean/std, median/IQR, Welch, bootstrap, kappa-norm."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats

ARM_ORDER = [
    "rbf-base",
    "rbf-grad",
    "rbf-shape",
    "correction-field",
    "surrogate-target",
    "compound-loss",
]

# Key one-sided Welch gates (lower rel-L2 = better).
GATE_SPECS = (
    ("rbf_grad_vs_base", "rbf-grad", "rbf-base", "rel_l2"),
    ("rbf_shape_vs_base", "rbf-shape", "rbf-base", "rel_l2"),
    ("rbf_shape_vs_grad", "rbf-shape", "rbf-grad", "rel_l2"),
    ("rbf_shape_band_vs_base", "rbf-shape", "rbf-base", "band_l2"),
)


def mean_std(vals: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.nanmean(a)) if a.size else float("nan"),
        "std": float(np.nanstd(a)) if a.size else float("nan"),
        "n": int(np.sum(np.isfinite(a))),
    }


def median_iqr(vals: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"median": float("nan"), "iqr": float("nan"), "q25": float("nan"), "q75": float("nan")}
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return {
        "median": float(med),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
    }


def welch_one_sided(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "less",
) -> dict[str, Any]:
    """Paired one-sided t-test (scipy ttest_rel): H1 mean(a) < mean(b) when 'less'.

    Seeds are aligned across arms (same seed list per cell), so the correct
    widely-used test is paired Welch / related-samples ttest_rel rather than
    independent ttest_ind. beats_beyond_noise requires one-sided p < 0.05.
    df follows whatever scipy returns on the result object.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    out: dict[str, Any] = {
        "n": int(a.size),
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
        "mean_diff": float(np.mean(a) - np.mean(b)) if a.size else float("nan"),
        "t": float("nan"),
        "df": float("nan"),
        "pvalue": float("nan"),
        "alternative": alternative,
        "beats_beyond_noise": False,
    }
    if a.size < 2:
        return out
    res = scipy_stats.ttest_rel(a, b, alternative=alternative)
    p = float(res.pvalue)
    df = float(getattr(res, "df", float("nan")))
    out.update({
        "t": float(res.statistic),
        "df": df,
        "pvalue": p,
        "beats_beyond_noise": bool(np.isfinite(p) and p < 0.05),
    })
    return out


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap 95% CI on mean(a)-mean(b) (paired-by-seed indices)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n == 0:
        return {
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_resamples": n_resamples,
            "n": 0,
        }
    rng = np.random.default_rng(seed)
    diffs = a - b
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": lo,
        "ci_high": hi,
        "n_resamples": int(n_resamples),
        "n": int(n),
    }


def kappa_normalized_error(
    rel_l2: float,
    cond: float,
    cond_ref: float,
) -> float:
    """effective_error = rel-L2 * (cond / cond_ref)."""
    if not np.isfinite(rel_l2) or not np.isfinite(cond) or not np.isfinite(cond_ref):
        return float("nan")
    if cond_ref <= 0:
        return float("nan")
    return float(rel_l2 * (cond / cond_ref))


def summarize_arm_metric(vals: list[float] | np.ndarray) -> dict[str, Any]:
    return {
        **mean_std(vals),
        **median_iqr(vals),
        "values": [float(v) for v in np.asarray(vals, dtype=float)],
    }


def cell_statistics(
    per_seed: list[dict],
    cond_ref: float | None = None,
    bootstrap_resamples: int = 10000,
) -> dict[str, Any]:
    """Summarize one (gamma, kappa, res) cell over its seeds.

    per_seed entries must include arms[arm].final_rel_l2 / final_rel_l2_band
    and kansa_info.cond (or arms['rbf-base'] kansa).
    """
    arms_out: dict[str, Any] = {}
    for arm in ARM_ORDER:
        rel = [p["arms"][arm]["final_rel_l2"] for p in per_seed]
        band = [p["arms"][arm]["final_rel_l2_band"] for p in per_seed]
        arms_out[arm] = {
            "rel_l2": summarize_arm_metric(rel),
            "band_l2": summarize_arm_metric(band),
        }

    conds = []
    for p in per_seed:
        c = p.get("kansa_info", {}).get("cond")
        if c is None:
            c = p["arms"].get("rbf-base", {}).get("kansa_cond")
        conds.append(float(c) if c is not None else float("nan"))
    mean_cond = float(np.nanmean(conds)) if conds else float("nan")
    if cond_ref is None or not np.isfinite(cond_ref):
        cond_ref = mean_cond

    kappa_norm: dict[str, Any] = {"cond_ref": float(cond_ref), "mean_kansa_cond": mean_cond}
    for arm in ARM_ORDER:
        rel_mean = arms_out[arm]["rel_l2"]["mean"]
        # Per-seed effective errors then mean, using each seed's Kansa cond
        eff = []
        for p, c in zip(per_seed, conds):
            eff.append(
                kappa_normalized_error(p["arms"][arm]["final_rel_l2"], c, cond_ref)
            )
        kappa_norm[arm] = {
            "effective_error_mean": float(np.nanmean(eff)) if eff else float("nan"),
            "effective_error_std": float(np.nanstd(eff)) if eff else float("nan"),
            "effective_error_from_means": kappa_normalized_error(
                rel_mean, mean_cond, cond_ref
            ),
            "values": [float(v) for v in eff],
        }

    gates: dict[str, Any] = {}
    for name, arm_a, arm_b, metric in GATE_SPECS:
        key = "final_rel_l2" if metric == "rel_l2" else "final_rel_l2_band"
        a = np.asarray([p["arms"][arm_a][key] for p in per_seed], dtype=float)
        b = np.asarray([p["arms"][arm_b][key] for p in per_seed], dtype=float)
        welch = welch_one_sided(a, b, alternative="less")
        boot = bootstrap_mean_diff_ci(a, b, n_resamples=bootstrap_resamples, seed=0)
        gates[name] = {
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

    return {
        "n_seeds": len(per_seed),
        "arms": arms_out,
        "kansa_conds": conds,
        "kappa_normalized": kappa_norm,
        "gates": gates,
    }


def consolidated_table_header() -> str:
    cols = [f"{'cell':<28}"]
    short = {
        "rbf-base": "base",
        "rbf-grad": "grad",
        "rbf-shape": "shape",
        "correction-field": "corr",
        "surrogate-target": "surr",
        "compound-loss": "comp",
    }
    for a in ARM_ORDER:
        cols.append(f"{short.get(a, a):<10}")
    for c in ("p_g_b", "p_s_b", "p_s_g", "p_s_band", "knorm_s"):
        cols.append(f"{c:<10}")
    return "".join(cols)


def format_consolidated_row(cell_key: str, cell_stats: dict) -> str:
    """One-line table row: cell | per-arm mean relL2 | key p-values | kappa-norm."""
    arms = cell_stats["arms"]
    parts = [f"{cell_key:<28}"]
    for arm in ARM_ORDER:
        m = arms[arm]["rel_l2"]["mean"]
        parts.append(f"{m:<10.3e}")
    gates = cell_stats["gates"]
    for gname in (
        "rbf_grad_vs_base",
        "rbf_shape_vs_base",
        "rbf_shape_vs_grad",
        "rbf_shape_band_vs_base",
    ):
        p = gates[gname]["welch"]["pvalue"]
        parts.append(f"{p:<10.3g}" if np.isfinite(p) else f"{'n/a':<10}")
    kn = cell_stats["kappa_normalized"]
    kn_s = kn.get("rbf-shape", {}).get("effective_error_mean", float("nan"))
    parts.append(f"{kn_s:<10.3e}")
    return "".join(parts)
