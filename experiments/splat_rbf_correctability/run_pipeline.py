"""End-to-end sweep: mode x N x target, trains + evaluates, prints/CSV table,
then synthesizes a brief NeurIPS-styled PDF report (make_report.py).

From the thread question: "Are Gaussian Splats for Solution Approximations Correctable?"

Run on a GPU host inside a uv venv with jax[cuda]:
    uv run python3 run_pipeline.py --out results/sweep.csv

Produces rows per (mode, N, target) with recovery rel-L2 plus, for RBF, the
closed-form additive-correction gain, then results/report.pdf via the bloated-neurips
typst template (requires `typst` on PATH — home has ~/.cargo/bin/typst; otherwise
the PDF step is skipped but the CSV is still written).
"""
import argparse
import csv
import pathlib
import sys

import jax

OUT = pathlib.Path("results")

SWEEP = {
    "n": [50, 250, 1000],
    "targets": ["blob", "oscillatory", "multiscale"],
    "modes": ["rbf", "splat"],
    "steps": 600,
}


def main(out_path: pathlib.Path, make_pdf: bool = True) -> int:
    from evaluate import evaluate

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
    print(f"\nwrote {out_path} ({len(rows)} runs)")

    if make_pdf:
        from make_report import main as report_main
        code = report_main(csv_path=out_path, out_typ=OUT / "report.typ")
        if code != 0:
            print("report step finished with nonzero status; sweep itself succeeded")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "sweep.csv")
    ap.add_argument("--no-pdf", action="store_true",
                    help="skip the typst PDF report step")
    args = ap.parse_args()
    jax.devices()
    sys.exit(main(args.out, make_pdf=not args.no_pdf))
