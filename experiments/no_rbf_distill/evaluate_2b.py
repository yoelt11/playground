#!/usr/bin/env python3
"""Aggregate Phase-2b arm table + pre-registered go/no-go for complementary-σ.

Pre-registered claims (write them here so the verdict is not post-hoc):

1. GO for σ-only:
   ``sigma_only`` corrected rel-L2 beats ``residual_only`` by ≥1.2× reduction
   (corrected_σ ≤ residual_only / 1.2). NOT expected — σ is a weaker allocator —
   but if it does, that is a big win ("σ is usable at all for placement").

2. GO for the complementary claim (the arm that can rescue uncertainty):
   ``residual_split_sigma`` beats ``residual_only`` by ≥1.2× reduction AND the
   beat is stable across seeds (not a single-seed fluke: at least 2/3 seeds
   show a strict improvement, and mean reduction ≥1.2×).

3. If BOTH sigma_only and residual_split_sigma fail to beat residual_only,
   verdict = UNCONFIRMED/CLOSED for the uncertainty branch.

Also reports, per seed, the fraction of residual- vs σ-placed centers that land
in the top-10% true |u_NO−u*| sites (diagnostic: do σ's dedicated kernels hit
error the residual missed?).

Optional context: if Phase-2 ``residual_unc_stoch_seed*.json`` exist, their
means are printed for comparison — they are NOT the claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import DEFAULT_SEEDS, DEFAULT_SPLIT_ALPHA, STOCH_ARM_NAMES2

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

REDUCTION_MARGIN = 1.2
MIN_SEEDS_BEATING = 2  # stability: not a one-seed fluke (of 3 default seeds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--margin", type=float, default=REDUCTION_MARGIN)
    p.add_argument("--alpha", type=float, default=DEFAULT_SPLIT_ALPHA)
    return p.parse_args()


def load_arm_seed(results_dir: Path, arm: str, seed: int) -> dict:
    path = results_dir / f"{arm}_2b_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run run_pipeline_2b.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _try_load_phase2_residual_unc(results_dir: Path, seeds: list[int]) -> dict | None:
    vals = []
    for seed in seeds:
        path = results_dir / f"residual_unc_stoch_seed{seed}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        vals.append(float(payload["corrected_rel_l2"]))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": arr.tolist(),
        "note": "Phase-2 global blend (context only; NOT the Phase-2b claim)",
    }


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    margin = float(args.margin)
    n_seeds = len(seeds)
    min_beat = min(MIN_SEEDS_BEATING, max(1, n_seeds))

    rows: dict[str, list[float]] = {arm: [] for arm in STOCH_ARM_NAMES2}
    u_no_vals: list[float] = []
    per_seed: dict[str, dict] = {}
    overlap_rows: list[dict] = []

    for seed in seeds:
        per_seed[str(seed)] = {}
        for arm in STOCH_ARM_NAMES2:
            payload = load_arm_seed(args.results_dir, arm, seed)
            rows[arm].append(float(payload["corrected_rel_l2"]))
            per_seed[str(seed)][arm] = payload
            if arm == "residual_only":
                u_no_vals.append(float(payload["u_no_rel_l2"]))
        # Center-overlap diagnostic (prefer split arm; also record per-arm all-hit)
        split_p = per_seed[str(seed)]["residual_split_sigma"]
        sigma_p = per_seed[str(seed)]["sigma_only"]
        res_p = per_seed[str(seed)]["residual_only"]
        overlap_rows.append(
            {
                "seed": seed,
                "residual_only_frac_top_err": float(
                    res_p.get("frac_all_centers_in_top_err", float("nan"))
                ),
                "sigma_only_frac_top_err": float(
                    sigma_p.get("frac_all_centers_in_top_err", float("nan"))
                ),
                "split_residual_subset_frac_top_err": float(
                    split_p.get("frac_residual_centers_in_top_err", float("nan"))
                ),
                "split_sigma_subset_frac_top_err": float(
                    split_p.get("frac_sigma_centers_in_top_err", float("nan"))
                ),
                "split_all_frac_top_err": float(
                    split_p.get("frac_all_centers_in_top_err", float("nan"))
                ),
                "n_residual_centers": int(split_p.get("n_residual_centers", 0)),
                "n_sigma_centers": int(split_p.get("n_sigma_centers", 0)),
            }
        )

    stats = {}
    print("\n=== Phase 2b arm table (corrected rel-L2) ===", flush=True)
    print(f"{'arm':22s}  {'mean':>10s}  {'std':>10s}  values", flush=True)
    for arm in STOCH_ARM_NAMES2:
        vals = np.asarray(rows[arm], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        stats[arm] = {"mean": mean, "std": std, "values": vals.tolist()}
        vals_s = ", ".join(f"{v:.4e}" for v in vals)
        print(f"{arm:22s}  {mean:10.4e}  {std:10.4e}  [{vals_s}]", flush=True)

    u_no = np.asarray(u_no_vals, dtype=np.float64)
    u_no_mean = float(u_no.mean())
    u_no_std = float(u_no.std(ddof=1)) if len(u_no) > 1 else 0.0
    print(
        f"{'u_NO (mean ens)':22s}  {u_no_mean:10.4e}  {u_no_std:10.4e}  "
        f"[{', '.join(f'{v:.4e}' for v in u_no)}]",
        flush=True,
    )

    residual_unc_ctx = _try_load_phase2_residual_unc(args.results_dir, seeds)
    if residual_unc_ctx is not None:
        print(
            f"{'residual_unc (P2 ctx)':22s}  {residual_unc_ctx['mean']:10.4e}  "
            f"{residual_unc_ctx['std']:10.4e}  "
            f"[{', '.join(f'{v:.4e}' for v in residual_unc_ctx['values'])}]  "
            f"(context only)",
            flush=True,
        )

    res_only = stats["residual_only"]["mean"]
    sigma_only = stats["sigma_only"]["mean"]
    split = stats["residual_split_sigma"]["mean"]

    # Lower corrected rel-L2 is better.
    red_sigma = res_only / max(sigma_only, 1e-30)
    red_split = res_only / max(split, 1e-30)

    # Per-seed strict beats (for stability of the complementary claim)
    res_vals = np.asarray(rows["residual_only"], dtype=np.float64)
    sigma_vals = np.asarray(rows["sigma_only"], dtype=np.float64)
    split_vals = np.asarray(rows["residual_split_sigma"], dtype=np.float64)
    sigma_beats_per_seed = (sigma_vals < res_vals).tolist()
    split_beats_per_seed = (split_vals < res_vals).tolist()
    n_sigma_beat = int(sum(sigma_beats_per_seed))
    n_split_beat = int(sum(split_beats_per_seed))

    sigma_go = bool(red_sigma >= margin)
    split_mean_go = bool(red_split >= margin)
    split_stable = bool(n_split_beat >= min_beat)
    complementary_go = bool(split_mean_go and split_stable)

    # Closed if neither arm beats residual_only on the mean (≥1.2×) claim.
    # "Fail to beat" = mean reduction < margin (ties / worse both count as fail).
    both_fail = (not sigma_go) and (not complementary_go)

    print("\n=== Center overlap with top-10% true |u_NO−u*| ===", flush=True)
    print(
        f"{'seed':>4s}  {'R-only':>8s}  {'σ-only':>8s}  "
        f"{'split-R':>8s}  {'split-σ':>8s}  {'split-all':>9s}",
        flush=True,
    )
    for row in overlap_rows:
        print(
            f"{row['seed']:4d}  "
            f"{row['residual_only_frac_top_err']:8.3f}  "
            f"{row['sigma_only_frac_top_err']:8.3f}  "
            f"{row['split_residual_subset_frac_top_err']:8.3f}  "
            f"{row['split_sigma_subset_frac_top_err']:8.3f}  "
            f"{row['split_all_frac_top_err']:9.3f}",
            flush=True,
        )
    # Means
    def _nanmean(key: str) -> float:
        vals = [r[key] for r in overlap_rows]
        return float(np.nanmean(np.asarray(vals, dtype=np.float64)))

    print(
        f"{'mean':>4s}  "
        f"{_nanmean('residual_only_frac_top_err'):8.3f}  "
        f"{_nanmean('sigma_only_frac_top_err'):8.3f}  "
        f"{_nanmean('split_residual_subset_frac_top_err'):8.3f}  "
        f"{_nanmean('split_sigma_subset_frac_top_err'):8.3f}  "
        f"{_nanmean('split_all_frac_top_err'):9.3f}",
        flush=True,
    )
    if overlap_rows:
        print(
            f"  (split budget: {overlap_rows[0]['n_residual_centers']} residual + "
            f"{overlap_rows[0]['n_sigma_centers']} σ kernels; α≈"
            f"{overlap_rows[0]['n_sigma_centers'] / max(overlap_rows[0]['n_residual_centers'] + overlap_rows[0]['n_sigma_centers'], 1):.2f})",
            flush=True,
        )

    print("\n=== go / no-go (Phase 2b complementary-σ) ===", flush=True)
    print(
        f"  Pre-registered margin: ≥{margin:.2f}× rel-L2 reduction vs residual_only",
        flush=True,
    )
    print(
        f"  sigma_only vs residual_only: ×{red_sigma:.3f}  "
        f"seeds beating residual_only: {n_sigma_beat}/{n_seeds}  "
        f"→ {'GO' if sigma_go else 'FAIL'}",
        flush=True,
    )
    print(
        f"  residual_split_sigma vs residual_only: ×{red_split:.3f}  "
        f"seeds beating residual_only: {n_split_beat}/{n_seeds} "
        f"(need ≥{min_beat} for stability)  "
        f"→ {'GO' if complementary_go else 'FAIL'}",
        flush=True,
    )

    if complementary_go and sigma_go:
        verdict = "GO"
        reason = (
            "Both sigma_only and residual_split_sigma beat residual_only by the "
            "pre-registered margin; σ is usable and complementary split helps."
        )
    elif complementary_go:
        verdict = "GO (complementary)"
        reason = (
            "residual_split_sigma beats residual_only by ≥1.2× with multi-seed "
            "stability — complementary dedicated σ kernels rescue uncertainty. "
            "sigma_only alone did not clear the margin (expected if σ is weaker)."
        )
    elif sigma_go:
        verdict = "GO (sigma_only)"
        reason = (
            "sigma_only beats residual_only by ≥1.2× (unexpected — σ usable as "
            "sole allocator). residual_split_sigma did not clear the complementary "
            "claim."
        )
    elif both_fail:
        verdict = "UNCONFIRMED/CLOSED"
        reason = (
            "Both sigma_only and residual_split_sigma fail to beat residual_only "
            "by the pre-registered ≥1.2× margin. Uncertainty branch is CLOSED on "
            "this toy: σ does not help as a sole allocator nor as a dedicated "
            "split subset alongside residual-placed kernels."
        )
    else:
        # e.g. split mean improves but not stably, or beats but < margin
        verdict = "UNCONFIRMED"
        reason = (
            "Neither pre-registered GO fired cleanly (margin and/or seed "
            "stability). Uncertainty branch remains unconfirmed on this toy."
        )

    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  reason: {reason}", flush=True)

    table = {
        "phase": "2b",
        "margin": margin,
        "alpha": float(args.alpha),
        "seeds": seeds,
        "min_seeds_beating_for_stability": min_beat,
        "u_no_rel_l2": {"mean": u_no_mean, "std": u_no_std, "values": u_no.tolist()},
        "arms": stats,
        "residual_unc_phase2_context": residual_unc_ctx,
        "reduction_sigma_only_vs_residual_only": float(red_sigma),
        "reduction_split_vs_residual_only": float(red_split),
        "sigma_only_beats_per_seed": sigma_beats_per_seed,
        "split_beats_per_seed": split_beats_per_seed,
        "n_sigma_only_beats_residual_only": n_sigma_beat,
        "n_split_beats_residual_only": n_split_beat,
        "sigma_only_go": sigma_go,
        "complementary_go": complementary_go,
        "center_overlap_top_err": overlap_rows,
        "center_overlap_means": {
            "residual_only_frac_top_err": _nanmean("residual_only_frac_top_err"),
            "sigma_only_frac_top_err": _nanmean("sigma_only_frac_top_err"),
            "split_residual_subset_frac_top_err": _nanmean(
                "split_residual_subset_frac_top_err"
            ),
            "split_sigma_subset_frac_top_err": _nanmean(
                "split_sigma_subset_frac_top_err"
            ),
            "split_all_frac_top_err": _nanmean("split_all_frac_top_err"),
        },
        "verdict": verdict,
        "reason": reason,
        "pre_registered": {
            "sigma_only_go": (
                "sigma_only corrected rel-L2 beats residual_only by ≥1.2× reduction"
            ),
            "complementary_go": (
                "residual_split_sigma beats residual_only by ≥1.2× reduction "
                "AND beat is stable across seeds (not one seed)"
            ),
            "closed_if": (
                "BOTH sigma_only and residual_split_sigma fail → "
                "UNCONFIRMED/CLOSED for the uncertainty branch"
            ),
        },
    }
    out_json = args.results_dir / "phase2b_arm_table.json"
    out_txt = args.results_dir / "phase2b_arm_table.txt"
    verdict_path = args.results_dir / "phase2b_verdict.json"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
    verdict_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "sigma_only_go": sigma_go,
                "complementary_go": complementary_go,
                "reason": reason,
                "reduction_sigma_only_vs_residual_only": float(red_sigma),
                "reduction_split_vs_residual_only": float(red_split),
                "margin": margin,
                "alpha": float(args.alpha),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "Phase 2b arm table (corrected rel-L2 of u_NO_mean + v_RBF)",
        f"seeds={seeds}  margin={margin}  alpha={args.alpha}",
        "",
        f"{'arm':22s}  {'mean':>10s}  {'std':>10s}",
    ]
    for arm in STOCH_ARM_NAMES2:
        lines.append(
            f"{arm:22s}  {stats[arm]['mean']:10.4e}  {stats[arm]['std']:10.4e}"
        )
    lines += [
        f"{'u_NO (mean ens)':22s}  {u_no_mean:10.4e}  {u_no_std:10.4e}",
        "",
        f"sigma_only vs residual_only: ×{red_sigma:.3f}  go={sigma_go}",
        f"residual_split_sigma vs residual_only: ×{red_split:.3f}  "
        f"go={complementary_go}  seeds_beating={n_split_beat}/{n_seeds}",
        f"verdict: {verdict}",
        f"reason: {reason}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote] {out_json}", flush=True)
    print(f"[wrote] {out_txt}", flush=True)
    print(f"[wrote] {verdict_path}", flush=True)


if __name__ == "__main__":
    main()
