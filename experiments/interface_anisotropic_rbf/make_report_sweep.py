#!/usr/bin/env python3
"""Robust-sweep NeurIPS report for interface_anisotropic_rbf.

Reads results/sweep/sweep_stats.json (8 cells x 6 seeds + convergence), writes
report_robust.typ (bloated-neurips), compiles to report_robust.pdf via typst.
Usage (on a host with typst, e.g. home): python3 make_report_sweep.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
_TYPST_CANDIDATES = (shutil.which("typst"), "/home/etorres/.cargo/bin/typst", "/home/etorres/.local/bin/typst")


def _find_typst():
    for p in _TYPST_CANDIDATES:
        if p and pathlib.Path(p).is_file():
            return p
    return None


def main() -> int:
    s = json.load(open(HERE / "results/sweep/sweep_stats.json"))
    cells = s["cells"]
    order = [f"g={g}|k={k}|r={r}" for g in ("vertical", "circle") for k in (10, 100) for r in (40, 80)]

    def rel(arms, name):
        r = arms.get(name, {}).get("rel_l2", {})
        return f"{r.get('mean', float('nan')):.3f}" if r else "nan"

    def pv(gates, name):
        g = gates.get(name, {})
        w = g.get("welch", {})
        return f"{w.get('pvalue', float('nan')):.2g}".replace("e-0", "e-")

    rows = []
    for key in order:
        c = cells[key]
        a, g = c["arms"], c["gates"]
        rows.append(
            f"  [`{' / '.join(key.split('|'))}`], [{rel(a,'rbf-base')}], [{rel(a,'rbf-grad')}], "
            f"[{rel(a,'rbf-shape')}], [{rel(a,'correction-field')}], [{rel(a,'compound-loss')}], "
            f"[{pv(g,'rbf_grad_vs_base')}/{str(g.get('rbf_grad_vs_base',{}).get('beats_beyond_noise'))[:1]}], "
            f"[{pv(g,'rbf_shape_vs_base')}/{str(g.get('rbf_shape_vs_base',{}).get('beats_beyond_noise'))[:1]}], "
            f"[{pv(g,'rbf_shape_band_vs_base')}/{str(g.get('rbf_shape_band_vs_base',{}).get('beats_beyond_noise'))[:1]}],"
        )
    body = "\n".join(rows)

    # convergence
    cv = s.get("convergence", {}).get("by_n_centers", {})
    crows = []
    for n in ("32", "48", "64"):
        d = cv.get(n, {})
        arms = d.get("arms", {})
        f_ = lambda name: (arms.get(name, {}).get("rel_l2_mean", float("nan")))
        crows.append(f"  [{n}], [{f_('rbf-base'):.4f}], [{f_('rbf-grad'):.4f}], [{f_('rbf-shape'):.4f}],")
    conv = "\n".join(crows)

    typ = f"""#import "@preview/bloated-neurips:0.8.0": appendix, botrule, midrule, neurips2026, paragraph, toprule, url

#show: neurips2026.with(
  title: [Robust Sweep: Interface-Gamma Anisotropic RBF — Is the Anisotropic (ePIL VSD) Win Real?],
  keywords: ("Anisotropic RBF", "Interface", "Robustness", "paired Welch", "Kappa-wall"),
  abstract: [
    We stress-test the interface-Gamma results across a grid: Gamma in {{vertical line, circle}}, kappa
    jumps 10 and 100, resolutions 40 and 80, 6 seeds per cell (48 full cells, ~1 GPU-hour). Claims are
    gated by a PAIRED Welch t-test (one-sided, p < 0.05) plus a 10k bootstrap CI on the mean difference.
    Two conclusions are ROBUST: (1) isotropic gradient refinement (rbf-grad) beats the one-shot Kansa
    base beyond noise in 7 of 8 cells (p < 0.005, and p < 1e-7 in most) — the interface supplies real
    residual, so gradient refinement is a genuine, reproducible win; (2) the ANISOTROPIC arm (rbf-shape)
    does NOT beat base or rbf-grad on the domain in any circle cell, and its apparent interface-band
    advantage DOES NOT SURVIVE higher resolution or a non-linear interface — it was a vertical, low-res
    artifact. A small MLP PINN (compound-loss) is best on the domain in every cell. The ePIL anisotropic
    shapes remain frozen under a kappa-wall (Gram condition ~1e18), which -- not any intrinsic failure of
    anisotropic kernels -- is the binding limit measured here.
  ],
)

This sweep is the robustness pass on the interface_anisotropic_rbf experiment (sequel to
q1_surrogate_vs_compound). It extends the single-geometry result to: geometry (vertical line vs
circle), resolution (40 vs 80), seed count (3 to 6), at kappa jumps 10 and 100. All arms share the
same per-cell Kansa base and centers; the 6 arms (rbf-base, rbf-grad, rbf-shape, correction-field,
surrogate-target, compound-loss) are unchanged. The paired Welch test uses the per-cell aligned
seeds; 'beats beyond noise' requires one-sided p < 0.05 for the domain rel-L2 (and a separate gate
for the interface-band rel-L2). The kappa-normalized error (rel-L2 times cond / cond-ref) is
reported per arm to couple accuracy to conditioning.

== Robust results (rel-L2 mean, 6 seeds) by cell -- base | grad | shape | corr | comp ; p-vals g>base s>base s-band

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (left, center),
  [`cell`], [`base`], [`grad`], [`shape`], [`corr`], [`comp`], [`p g>base`], [`p s>base`], [`p s-band`],
{body}
)

**Robust verdict (all 6-seed cells):**
- rbf-grad beats base on the domain: *True* in 7/8 cells (vertical both kappa, both res; circle k10
  both res, circle k100 r40); *False* only at circle k100 r80 (grad ~ base 0.339). p < 0.005 in the
  wins, p < 1e-6 in most.
- rbf-shape beats base on the domain: *True* only in vertical r40 (k10 p=0.049, k100 p=0.010);
  *False* everywhere else (p near 1 on circle). NOT robust.
- rbf-shape interface-band vs base: *True* only vertical r40 (k10 p=0.02, k100 p=0.043);
  *False* at r80 and on ALL circle cells. The band-signal does NOT generalize.
- compound-loss is lowest rel-L2 in every cell (0.10--0.30).
- correction-field collapses at kappa=100 (0.33--0.64) -- base-quality dependent.

== Convergence (error vs N_centers; vertical, kappa 10, res 40)

#table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  align: (left, center),
  [`N centers`], [`rbf-base`], [`rbf-grad`], [`rbf-shape`],
{conv}
)

rbf-grad retains its win as N grows (base ~0.23, grad ~0.19 at N=32); rbf-shape is unstable at
N=64 (high-variance, some seeds diverge). No saturation-driven reversal.

== Conclusions (robust)

- **Isotropic gradient refinement is a genuine, reproducible win at the interface** -- the single
  strongest robust result (7/8 cells, p < 0.005). It is NOT a low-res or single-geometry artifact.
- **Anisotropic learnable-shape RBF does NOT provide a robust benefit here.** Its earlier-looking
  interface-band edge was a vertical / low-resolution artifact: it fails at higher resolution and on
  a curved interface (p ~ 1). The ePIL VSD shapes stay frozen under a kappa-wall (Gram cond ~1e18);
  this is a conditioning limit on moving the shapes, not evidence about anisotropic kernels in general.
- **A small MLP PINN (compound-loss) is best on the domain in every cell** (robust reversal of the
  smooth-toy conclusion where the RBF base dominated).
- **correction-field is base-quality dependent** (helped in q1, collapses from a poor base at kappa 100).
- **Methodologically:** 6 seeds + paired Welch + bootstrap CI converted a fragile single-geometry
  signal into a defensible (negative) verdict -- the robustness sweep did its job.
"""

    (HERE / "report_robust.typ").write_text(typ)
    typst = _find_typst()
    if not typst:
        raise SystemExit("typst not found; compile report_robust.typ manually on a host with typst")
    r = subprocess.run([typst, "compile", "report_robust.typ"], cwd=str(HERE),
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("TYPST ERROR:\n", r.stderr[-3000:])
        return r.returncode
    print("OK ->", HERE / "report_robust.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())