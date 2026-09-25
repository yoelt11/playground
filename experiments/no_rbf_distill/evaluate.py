#!/usr/bin/env python3
"""Aggregate arm table + go/no-go verdict for Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import ARM_NAMES, DEFAULT_SEEDS

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# Go/no-go: residual_alloc beats uniform by ≥1.5× rel-L2 *reduction*
# (i.e. corrected error is ≤ uniform_error / 1.5), AND shuffled is NOT better
# than uniform (spatial emphasis is what matters).
REDUCTION_MARGIN = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--margin", type=float, default=REDUCTION_MARGIN)
    return p.parse_args()


def load_arm_seed(results_dir: Path, arm: str, seed: int) -> dict:
    path = results_dir / f"{arm}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run run_pipeline.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    margin = float(args.margin)

    rows: dict[str, list[float]] = {arm: [] for arm in ARM_NAMES}
    u_no_vals: list[float] = []
    per_seed: dict[str, dict] = {}

    for seed in seeds:
        per_seed[str(seed)] = {}
        for arm in ARM_NAMES:
            payload = load_arm_seed(args.results_dir, arm, seed)
            rows[arm].append(float(payload["corrected_rel_l2"]))
            per_seed[str(seed)][arm] = payload
            if arm == ARM_NAMES[0]:
                u_no_vals.append(float(payload["u_no_rel_l2"]))

    stats = {}
    print("\n=== Phase 1 arm table (corrected rel-L2) ===", flush=True)
    print(f"{'arm':16s}  {'mean':>10s}  {'std':>10s}  values", flush=True)
    for arm in ARM_NAMES:
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
        f"{'u_NO (baseline)':16s}  {u_no_mean:10.4e}  {u_no_std:10.4e}  "
        f"[{', '.join(f'{v:.4e}' for v in u_no)}]",
        flush=True,
    )

    uni = stats["uniform"]["mean"]
    res = stats["residual_alloc"]["mean"]
    shuf = stats["shuffled_alloc"]["mean"]

    # Lower corrected rel-L2 is better. residual beats uniform by ≥margin means
    # uni / res ≥ margin  (reduction factor).
    reduction_vs_uniform = uni / max(res, 1e-30)
    residual_beats_uniform = reduction_vs_uniform >= margin
    # shuffled must NOT be better than uniform (shuf >= uni, within noise we use strict mean)
    shuffled_not_better = shuf >= uni * 0.99  # allow 1% slack for seed noise
    go = bool(residual_beats_uniform and shuffled_not_better)

    print("\n=== go / no-go ===", flush=True)
    print(
        f"  residual_alloc vs uniform reduction ×{reduction_vs_uniform:.3f} "
        f"(need ≥{margin:.2f}): {'PASS' if residual_beats_uniform else 'FAIL'}",
        flush=True,
    )
    print(
        f"  shuffled_alloc mean={shuf:.4e} vs uniform={uni:.4e} "
        f"(shuffled must NOT beat uniform): {'PASS' if shuffled_not_better else 'FAIL'}",
        flush=True,
    )
    print(f"  VERDICT: {'GO' if go else 'NO-GO'}", flush=True)

    table = {
        "margin": margin,
        "seeds": seeds,
        "u_no_rel_l2": {"mean": u_no_mean, "std": u_no_std, "values": u_no.tolist()},
        "arms": stats,
        "reduction_residual_vs_uniform": float(reduction_vs_uniform),
        "residual_beats_uniform": residual_beats_uniform,
        "shuffled_not_better_than_uniform": shuffled_not_better,
        "go": go,
        "verdict": "GO" if go else "NO-GO",
    }
    out_json = args.results_dir / "arm_table.json"
    out_txt = args.results_dir / "arm_table.txt"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")

    lines = [
        "Phase 1 arm table (corrected rel-L2 of u_NO + v_RBF)",
        f"seeds={seeds}  margin={margin}",
        "",
        f"{'arm':16s}  {'mean':>10s}  {'std':>10s}",
    ]
    for arm in ARM_NAMES:
        lines.append(
            f"{arm:16s}  {stats[arm]['mean']:10.4e}  {stats[arm]['std']:10.4e}"
        )
    lines += [
        f"{'u_NO (baseline)':16s}  {u_no_mean:10.4e}  {u_no_std:10.4e}",
        "",
        f"residual vs uniform reduction: ×{reduction_vs_uniform:.3f}",
        f"verdict: {'GO' if go else 'NO-GO'}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote] {out_json}", flush=True)
    print(f"[wrote] {out_txt}", flush=True)


if __name__ == "__main__":
    main()
