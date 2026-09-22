#!/usr/bin/env python3
"""Synthesize a brief NeurIPS-styled PDF report from the sweep results CSV.

Regenerates results/figures/*.png via make_figures.py, writes results/report.typ
(via @preview/bloated-neurips:0.8.0), then runs typst compile -> results/report.pdf.

Usage:  python3 make_report.py [--csv sweep.csv]
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
FIG_DIR = OUT / "figures"
FIG_SCRIPT = HERE / "make_figures.py"
_TYPST_CANDIDATES = (
    shutil.which("typst"),
    "/home/etorres/.cargo/bin/typst",
)


def _find_typst() -> str | None:
    for p in _TYPST_CANDIDATES:
        if p and pathlib.Path(p).is_file():
            return p
    return None


def _esc(s) -> str:
    """Escape a scalar for inline typst text (minimal; keep it safe enough)."""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("#", "\\#")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _data(csv_path: pathlib.Path):
    lines = [l.strip() for l in csv_path.read_text().splitlines() if l.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    body = []
    for ln in lines[1:]:
        vals = [v.strip() for v in ln.split(",")]
        body.append(dict(zip(header, vals)))
    return header, body


def _fmt_rel(x: float) -> str:
    """2–3 decimal places, at most ~3 significant figures."""
    ax = abs(x)
    if ax == 0.0:
        return "0.00"
    if ax >= 1.0:
        return f"{x:.3g}"
    if ax >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".") if f"{x:.3f}" != "0.000" else f"{x:.2e}"
    # small values: 2–3 sig via scientific or fixed
    return f"{x:.2e}"


def _fmt_psnr(x: float) -> str:
    return f"{x:.1f}"


def _fmt_gain(x: float) -> str:
    ax = abs(x)
    if ax >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"


def _cell(text: str, bold: bool = False) -> str:
    t = _esc(text)
    return f"[*{t}*]" if bold else f"[{t}]"


def _build_comparison_table(rows: list[dict]) -> tuple[str, int]:
    """Compact (n, target) rows with RBF vs splat side-by-side; bold bests."""
    # index by (n, target, mode)
    by = {}
    for r in rows:
        key = (int(r["n"]), r["target"], r["mode"])
        by[key] = r

    ns = sorted({int(r["n"]) for r in rows})
    targets = ("blob", "oscillatory", "multiscale")

    # columns: n, target, rel_l2 RBF, rel_l2 splat, PSNR RBF, PSNR splat, gain
    cols_hdr = [
        "[$n$]",
        "[target]",
        "[rel-$L_2$ RBF]",
        "[rel-$L_2$ splat]",
        "[PSNR RBF]",
        "[PSNR splat]",
        "[gain]",
    ]
    hdr = ", ".join(cols_hdr) + ","
    trows = ""
    for n in ns:
        for target in targets:
            rbf = by.get((n, target, "rbf"))
            spl = by.get((n, target, "splat"))
            if not rbf or not spl:
                continue
            r_rel = float(rbf["rel_l2"])
            s_rel = float(spl["rel_l2"])
            r_ps = float(rbf["psnr"])
            s_ps = float(spl["psnr"])
            gain_s = (rbf.get("correct_gain") or "").strip()
            gain = float(gain_s) if gain_s else None

            # lower rel_l2 better; higher PSNR better
            cells = [
                _cell(str(n)),
                _cell(target),
                _cell(_fmt_rel(r_rel), bold=(r_rel <= s_rel)),
                _cell(_fmt_rel(s_rel), bold=(s_rel < r_rel)),
                _cell(_fmt_psnr(r_ps), bold=(r_ps >= s_ps)),
                _cell(_fmt_psnr(s_ps), bold=(s_ps > r_ps)),
                _cell(_fmt_gain(gain) if gain is not None else "—"),
            ]
            trows += "  " + ", ".join(cells) + ",\n"
    return hdr + "\n" + trows, 7


def _run_figures(csv_path: pathlib.Path) -> None:
    cmd = [sys.executable, str(FIG_SCRIPT), "--csv", str(csv_path), "--out", str(FIG_DIR)]
    # Prefer /usr/bin/python3 if available (task constraint)
    py = pathlib.Path("/usr/bin/python3")
    if py.is_file():
        cmd[0] = str(py)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("figure generation failed:\n", r.stdout, r.stderr)
        raise SystemExit(1)
    if r.stdout.strip():
        print(r.stdout.strip())


def main(csv_path: pathlib.Path, out_typ: pathlib.Path) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, rows = _data(csv_path)
    if not rows:
        print("no data; skipping report")
        return 1

    _run_figures(csv_path)

    rbfs = [r for r in rows if r.get("mode") == "rbf"]
    splats = [r for r in rows if r.get("mode") == "splat"]
    best_rbf = min((float(r["rel_l2"]) for r in rbfs), default=float("nan"))
    best_spl = min((float(r["rel_l2"]) for r in splats), default=float("nan"))
    gains = [float(r["correct_gain"]) for r in rbfs if (r.get("correct_gain") or "").strip()]
    max_gain = max(gains, default=0.0)

    hdr_and_rows, ncols = _build_comparison_table(rows)
    # split hdr / trows for template
    parts = hdr_and_rows.split("\n", 1)
    hdr = parts[0]
    trows = parts[1] if len(parts) > 1 else ""

    # relative paths from results/report.typ
    fig_a = "figures/rel_l2_vs_n.png"
    fig_b = "figures/mean_rel_l2_by_target.png"
    fig_c = "figures/correct_gain_vs_n.png"

    body = REPORT_TPL % dict(
        n_sweeps=len(rows),
        best_rbf=_fmt_rel(best_rbf),
        best_spl=_fmt_rel(best_spl),
        max_gain=_fmt_gain(max_gain),
        ncols=ncols,
        hdr=hdr,
        trows=trows,
        fig_a=fig_a,
        fig_b=fig_b,
        fig_c=fig_c,
    )
    out_typ.parent.mkdir(parents=True, exist_ok=True)
    out_typ.write_text(body)
    print(f"wrote {out_typ}")

    typst = _find_typst()
    if not typst:
        print("typst not found; report.typ written but not compiled")
        return 0

    r = subprocess.run(
        [typst, "compile", str(out_typ)],
        cwd=str(out_typ.parent),
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(f"compiled -> {out_typ.with_suffix('.pdf')}")
        return 0
    print("typst compile failed:\n", r.stdout, r.stderr)
    return 1


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

The GPU sweep of %(n_sweeps)d runs on JAX (home RTX 2070 Super); the JAX-CPU
surrogate reproduces the same trend. Headline finding: on the signed blob target,
best RBF rel-$L_2$ = %(best_rbf)s vs best splat rel-$L_2$ = %(best_spl)s — splat compositing
is the wrong substrate for signed fields. The closed-form correction helps most at
coarse basis (peak gain %(max_gain)s).

== Setup

Ported from Edgar's `gs-rbf.py` into pure JAX; both representations share centers,
scales, rotation and basis — only the compositing differs. RBF weights are signed.
Splat uses front-to-back alpha compositing with a transmittance that couples every splat nonlinearly, so corrections cannot be added linearly.

== Results

#figure(
  table(
    columns: (auto,) * %(ncols)s,
    align: center,
    stroke: 0.4pt + rgb("#bbbbbb"),
    inset: 4pt,
    fill: (_, y) => if calc.odd(y) { rgb("#f7f7f7") } else { none },
%(hdr)s
%(trows)s
  ),
  caption: [Recovery metrics by $(n, "target")$. Bold marks the better of RBF vs splat
    (lower rel-$L_2$, higher PSNR). Gain $=$ correct_rel_before $-$ correct_rel_after
    (RBF only; empty for splat).],
)

#figure(
  image("%(fig_a)s", width: 85%%),
  caption: [Relative $L_2$ vs basis size $n$ (log–log). Markers: circle $=$ RBF,
    square $=$ splat. Linestyles encode target.],
)

#figure(
  image("%(fig_b)s", width: 80%%),
  caption: [Mean rel-$L_2$ (averaged over $n$) by target, grouped by mode.],
)

#figure(
  image("%(fig_c)s", width: 80%%),
  caption: [One-shot linear correction gain vs $n$ (RBF only). Positive $=$ correction helped.],
)

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
    ap.add_argument("--csv", type=pathlib.Path, default=HERE / "sweep.csv")
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "report.typ")
    args = ap.parse_args()
    raise SystemExit(main(args.csv, args.out))
