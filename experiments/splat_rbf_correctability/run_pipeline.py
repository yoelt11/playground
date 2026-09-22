"""End-to-end sweep: mode x N x target, trains + evaluates, prints/CSV table.

This is the actual experiment from the thread question:
"Are Gaussian Splats for Solution Approximations Correctable?"

Run on a GPU host inside a uv venv with jax[cuda]:
    uv run python3 run_pipeline.py --out results/sweep.csv

Produces a row per (mode, N, target) with recovery error and, for RBF, the
closed-form additive-correction gain.
"""
import argparse
import csv
import pathlib

import jax

from evaluate import evaluate

OUT = pathlib.Path("results")

SWEEP = {
    "n": [250, 1000],
    "targets": ["blob", "oscillatory", "multiscale"],
    "modes": ["rbf", "splat"],
    "steps": 600,
}


def main(out_path: pathlib.Path):
    rows = []
    for mode in SWEEP["modes"]:
        for n in SWEEP["n"]:
            for target in SWEEP["targets"]:
                print(f"--- {mode} n={n} {target} ---", flush=True)
                row = evaluate(mode=mode, n=n, target=target,
                               correct=True, steps=SWEEP["steps"])
                print(row, flush=True)
                rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["mode", "n", "target", "final_mse", "rel_l2", "psnr",
            "correct_rel_before", "correct_rel_after", "correct_gain",
            "basis_rank", "basis_rank_ratio"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "sweep.csv")
    args = ap.parse_args()
    jax.devices()
    main(args.out)