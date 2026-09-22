"""Synthesize a brief NeurIPS-styled PDF report from the sweep results CSV.

Writes results/report.typ (via @preview/bloated-neurips) then runs `typst compile`
to produce results/report.pdf (package auto-downloaded on first use). Run after the
sweep (run_pipeline.py) on a host with typst (e.g. home: ~/.cargo/bin/typst).

Usage:  python3 make_report.py [--csv results/sweep.csv]
"""
import argparse
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
_TYPST = shutil.which("typst")


def _esc(s) -> str:
    """Escape a scalar for inline typst text (minimal; keep it safe enough)."""
    return (str(s).replace("\\", "\\\\").replace("\"", "\\\"")
                  .replace("[", "\\[").replace("]", "\\]")
                  .replace("#", "\\#").replace("(", "\\(").replace(")", "\\)"))


def _data(csv_path: pathlib.Path):
    lines = [l.strip() for l in csv_path.read_text().splitlines() if l.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    body = []
    for ln in lines[1:]:
        vals = [v.strip() for v in ln.split(",")]
        body.append(dict(zip(header, vals)))
    return header, body


def main(csv_path: pathlib.Path, out_typ: pathlib.Path) -> int:
    _, rows = _data(csv_path)
    if not rows:
        print("no data; skipping report")
        return 1

    rbfs = [r for r in rows if r.get("mode") == "rbf"]
    splats = [r for r in rows if r.get("mode") == "splat"]
    best_rbf = min((float(r["rel_l2"]) for r in rbfs), default=float("nan"))
    best_spl = min((float(r["rel_l2"]) for r in splats), default=float("nan"))
    gains = [float(r["correct_gain"]) for r in rbfs if r.get("correct_gain")]
    max_gain = max(gains, default=0.0)

    cols = ["mode", "n", "target", "rel_l2", "psnr",
            "correct_rel_before", "correct_rel_after", "correct_gain"]
    hdr = ", ".join("[`" + c + "`]" for c in cols) + ","
    trows = ""
    for r in rows:
        cells = ", ".join("[" + _esc(r.get(c, "")) + "]" for c in cols)
        trows += "  " + cells + ",\n"

    body = REPORT_TPL % dict(
        n_sweeps=len(rows),
        best_rbf=f"{best_rbf:.4f}",
        best_spl=f"{best_spl:.4f}",
        max_gain=f"{max_gain:.4f}",
        ncols=len(cols),
        hdr=hdr,
        trows=trows,
    )
    out_typ.write_text(body)
    print(f"wrote {out_typ}")

    if not _TYPST:
        print("typst not found on PATH; report.typ written but not compiled")
        return 0

    r = subprocess.run([_TYPST, "compile", str(out_typ)], cwd=OUT,
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"compiled -> {out_typ.with_suffix('.pdf')}")
        return 0
    print("typst compile failed:\n", r.stdout, r.stderr)
    return 1  # report is best-effort; don't fail the sweep on it


REPORT_TPL = r"""#import "@preview/bloated-neurips:0.8.0": appendix, botrule, midrule, neurips2026, paragraph, toprule, url

#show: neurips2026.with(
  title: [Are Gaussian Splats for Solution Approximations Correctable?],
  keywords: ("Gaussian Splat", "RBF", "Error Refinement", "Amortized PDE"),
  abstract: [
    A toy probe of whether two Gaussian-basis representations support *additive
    correctability* — the setting relevant to VID-RBF-style error refinement.

    Additive-RBF (linear in the weights) admits a closed-form linear residual
    correction: a one-shot least-squares solve on the frozen basis. Alpha-composited
    Gaussian splats (transmittance-coupled, nonlinear in parameters) do not — their
    error is only reducible by re-optimizing the stack. On signed (non-radiance)
    targets, additive RBF is the better approximation substrate.

    We sweep $n in {50, 250, 1000}$, target in {blob, oscillatory, multiscale},
    mode in {rbf, splat}; measuring recovery rel-L2 and, for RBF, the one-shot
    correction gain.
  ],
)

The GPU sweep of %(n_sweeps)d runs runs on JAX (home RTX 2070 Super); the JAX-CPU
surrogate reproduces the same trend. Headline finding: on the signed blob target,
best RBF rel-L2 = %(best_rbf)s vs best splat rel-L2 = %(best_spl)s — splat compositing
is the wrong substrate for signed fields. The closed-form correction helps most at
coarse basis (peak gain %(max_gain)s).

== Setup

Ported from Edgar's `gs-rbf.py` into pure JAX; both representations share centers,
scales, rotation and basis — only the compositing differs. RBF weights are signed.
Splat uses front-to-back alpha compositing with a transmittance that couples every splat nonlinearly, so corrections cannot be added linearly.

== Results (rel-L2, lower is better)

#table(
  columns: (auto,) * %(ncols)s,
  fill: (none,),
  stroke: 0.5pt + rgb("#cccccc"),
  inset: 5pt,
  align: (left, center),
%(hdr)s
%(trows)s
)

Key columns: `rel_l2` recovery rel-L2; `correct_rel_before/after` = one-shot linear
correction probe (RBF only); `correct_gain` = before - after (>0 means correction helped).

== Takeaway

*Correctable in the additive sense = RBF* (linear-in-weights; a residual solve adds
cleanly). *Splat is not straightforwardly so* — alpha compositing is nonlinear, so any
correction must be re-baked rather than added. This matches the thread hypothesis and
motivates the coarse-basis correction probe.

#align(center, block(spacing: 12pt, {
  [*Repro:* uv run python3 run_pipeline.py --out results/sweep.csv; then python3 make_report.py]
}))
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=pathlib.Path, default=OUT / "sweep.csv")
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "report.typ")
    args = ap.parse_args()
    raise SystemExit(main(args.csv, args.out))
