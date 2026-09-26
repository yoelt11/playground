#!/usr/bin/env python3
"""Aggregate Phase-2 arm table + pre-registered go/no-go for uncertainty signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import DEFAULT_SEEDS, STOCH_ARM_NAMES

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# Pre-registered: residual_unc must beat residual_only by ≥1.2× rel-L2 reduction
# (corrected error ≤ residual_only / 1.2), AND shuffled_unc must NOT beat
# residual_only (destroying the σ map removes the benefit).
REDUCTION_MARGIN = 1.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--margin", type=float, default=REDUCTION_MARGIN)
    return p.parse_args()


def load_arm_seed(results_dir: Path, arm: str, seed: int) -> dict:
    path = results_dir / f"{arm}_stoch_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run run_pipeline_stoch.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    margin = float(args.margin)

    rows: dict[str, list[float]] = {arm: [] for arm in STOCH_ARM_NAMES}
    u_no_vals: list[float] = []
    corr_vals: list[float] = []
    frac_top_unc_low_r: list[float] = []
    frac_top_r_low_unc: list[float] = []
    per_seed: dict[str, dict] = {}

    for seed in seeds:
        per_seed[str(seed)] = {}
        for arm in STOCH_ARM_NAMES:
            payload = load_arm_seed(args.results_dir, arm, seed)
            rows[arm].append(float(payload["corrected_rel_l2"]))
            per_seed[str(seed)][arm] = payload
            if arm == STOCH_ARM_NAMES[0]:
                u_no_vals.append(float(payload["u_no_rel_l2"]))
                corr_vals.append(float(payload.get("corr_sigma_abs_r", float("nan"))))
                frac_top_unc_low_r.append(
                    float(payload.get("frac_top_unc_with_low_r", float("nan")))
                )
                frac_top_r_low_unc.append(
                    float(payload.get("frac_top_r_with_low_unc", float("nan")))
                )

    stats = {}
    print("\n=== Phase 2 arm table (corrected rel-L2) ===", flush=True)
    print(f"{'arm':16s}  {'mean':>10s}  {'std':>10s}  values", flush=True)
    for arm in STOCH_ARM_NAMES:
        vals = np.asarray(rows[arm], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        stats[arm] = {"mean": mean, "std": std, "values": vals.tolist()}
        vals_s = ", ".join(f"{v:.4e}" for v in vals)
        print(f"{arm:16s}  {mean:10.4e}  {std:10.4e}  [{vals_s}]", flush=True)

    u_no = np.asarray(u_no_vals, dtype=np.float64)
    u_no_mean = float(u_no.mean())
    u_no_std = float(u_no.std(ddof=1)) if len(u_no) > 1 else 0.0
    print(
        f"{'u_NO (mean ens)':16s}  {u_no_mean:10.4e}  {u_no_std:10.4e}  "
        f"[{', '.join(f'{v:.4e}' for v in u_no)}]",
        flush=True,
    )

    res_only = stats["residual_only"]["mean"]
    res_unc = stats["residual_unc"]["mean"]
    shuf_unc = stats["shuffled_unc"]["mean"]
    res_only_std = stats["residual_only"]["std"]
    res_unc_std = stats["residual_unc"]["std"]

    # Lower corrected rel-L2 is better.
    reduction_vs_residual_only = res_only / max(res_unc, 1e-30)
    unc_beats_residual = reduction_vs_residual_only >= margin

    # Gap vs seed noise: is the mean gap larger than ~1 pooled std?
    gap = res_only - res_unc
    pooled_std = float(np.sqrt(0.5 * (res_only_std**2 + res_unc_std**2)))
    gap_vs_noise = gap / max(pooled_std, 1e-30) if pooled_std > 0 else float("inf")
    gap_statistically_sensible = bool(gap > 0 and (pooled_std == 0 or gap >= pooled_std))

    # shuffled must NOT be better than residual_only (σ map is what matters)
    shuffled_not_better = shuf_unc >= res_only * 0.99

    # Extra red flag: shuffled matching residual_unc means σ spatial map is unused
    shuffled_matches_unc = abs(shuf_unc - res_unc) <= 0.05 * max(res_unc, 1e-30)

    go = bool(unc_beats_residual and shuffled_not_better and not shuffled_matches_unc)

    corr_arr = np.asarray(corr_vals, dtype=np.float64)
    corr_mean = float(np.nanmean(corr_arr))
    # High collinearity ⇒ uncertainty redundant with residual (expect NO-GO)
    sigma_redundant_hint = bool(corr_mean >= 0.85)

    print("\n=== σ vs |R| diagnostic ===", flush=True)
    print(
        f"  corr(σ, |R|) mean={corr_mean:.3f}  "
        f"values=[{', '.join(f'{v:.3f}' for v in corr_vals)}]",
        flush=True,
    )
    print(
        f"  frac top-σ with low-|R|: "
        f"{float(np.nanmean(frac_top_unc_low_r)):.3f}  "
        f"(high ⇒ σ adds locations residual misses)",
        flush=True,
    )
    print(
        f"  frac top-|R| with low-σ: "
        f"{float(np.nanmean(frac_top_r_low_unc)):.3f}",
        flush=True,
    )
    if sigma_redundant_hint:
        print(
            "  NOTE: σ≈collinear with |R| (corr≥0.85) — expect uncertainty "
            "to be redundant (valid scientific NO-GO).",
            flush=True,
        )

    print("\n=== go / no-go (Phase 2 uncertainty) ===", flush=True)
    print(
        f"  residual_unc vs residual_only reduction ×{reduction_vs_residual_only:.3f} "
        f"(need ≥{margin:.2f}): {'PASS' if unc_beats_residual else 'FAIL'}",
        flush=True,
    )
    print(
        f"  mean gap={gap:.4e}  pooled_std={pooled_std:.4e}  "
        f"gap/std={gap_vs_noise:.2f}  "
        f"(sensible if gap≥pooled_std): "
        f"{'YES' if gap_statistically_sensible else 'NO'}",
        flush=True,
    )
    print(
        f"  shuffled_unc mean={shuf_unc:.4e} vs residual_only={res_only:.4e} "
        f"(shuffled must NOT beat residual_only): "
        f"{'PASS' if shuffled_not_better else 'FAIL'}",
        flush=True,
    )
    print(
        f"  shuffled_unc ≈ residual_unc? "
        f"{'YES (bad — σ map unused)' if shuffled_matches_unc else 'NO'}",
        flush=True,
    )

    if go:
        verdict = "GO"
        reason = (
            "residual_unc beats residual_only by pre-registered margin AND "
            "shuffled_unc does not retain the benefit (σ spatial map matters)."
        )
    else:
        verdict = "NO-GO"
        if not unc_beats_residual:
            reason = (
                "residual_unc ties or fails to beat residual_only — "
                "uncertainty adds nothing over residual on this problem."
            )
        elif shuffled_matches_unc:
            reason = (
                "shuffled_unc matches residual_unc — destroying the σ map "
                "does not remove the benefit; spatial uncertainty is unused."
            )
        else:
            reason = (
                "shuffled_unc beats residual_only — control fails; "
                "cannot claim the uncertainty map is causal."
            )
        print(f"  NO-GO: {reason}", flush=True)

    print(f"  VERDICT: {verdict}", flush=True)

    table = {
        "phase": 2,
        "margin": margin,
        "seeds": seeds,
        "u_no_rel_l2": {"mean": u_no_mean, "std": u_no_std, "values": u_no.tolist()},
        "arms": stats,
        "reduction_residual_unc_vs_residual_only": float(reduction_vs_residual_only),
        "unc_beats_residual_only": unc_beats_residual,
        "gap_residual_only_minus_unc": float(gap),
        "pooled_std": float(pooled_std),
        "gap_vs_pooled_std": float(gap_vs_noise),
        "gap_statistically_sensible": gap_statistically_sensible,
        "shuffled_not_better_than_residual_only": shuffled_not_better,
        "shuffled_matches_residual_unc": shuffled_matches_unc,
        "corr_sigma_abs_r": {
            "mean": corr_mean,
            "values": corr_vals,
        },
        "frac_top_unc_with_low_r": {
            "mean": float(np.nanmean(frac_top_unc_low_r)),
            "values": frac_top_unc_low_r,
        },
        "frac_top_r_with_low_unc": {
            "mean": float(np.nanmean(frac_top_r_low_unc)),
            "values": frac_top_r_low_unc,
        },
        "sigma_redundant_hint": sigma_redundant_hint,
        "go": go,
        "verdict": verdict,
        "reason": reason,
    }
    out_json = args.results_dir / "phase2_arm_table.json"
    out_txt = args.results_dir / "phase2_arm_table.txt"
    verdict_path = args.results_dir / "phase2_verdict.json"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
    verdict_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "go": go,
                "reason": reason,
                "reduction_residual_unc_vs_residual_only": float(
                    reduction_vs_residual_only
                ),
                "margin": margin,
                "corr_sigma_abs_r_mean": corr_mean,
                "sigma_redundant_hint": sigma_redundant_hint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "Phase 2 arm table (corrected rel-L2 of u_NO_mean + v_RBF)",
        f"seeds={seeds}  margin={margin}",
        "",
        f"{'arm':16s}  {'mean':>10s}  {'std':>10s}",
    ]
    for arm in STOCH_ARM_NAMES:
        lines.append(
            f"{arm:16s}  {stats[arm]['mean']:10.4e}  {stats[arm]['std']:10.4e}"
        )
    lines += [
        f"{'u_NO (mean ens)':16s}  {u_no_mean:10.4e}  {u_no_std:10.4e}",
        "",
        f"residual_unc vs residual_only reduction: ×{reduction_vs_residual_only:.3f}",
        f"corr(σ,|R|) mean: {corr_mean:.3f}",
        f"verdict: {verdict}",
        f"reason: {reason}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote] {out_json}", flush=True)
    print(f"[wrote] {out_txt}", flush=True)
    print(f"[wrote] {verdict_path}", flush=True)


if __name__ == "__main__":
    main()
