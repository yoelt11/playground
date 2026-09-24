#!/usr/bin/env python3
"""Synthesize a NeurIPS-styled PDF report for interface_anisotropic_rbf.

Mirrors q1_surrogate_vs_compound/make_report.py (bloated-neurips format).
Reads results/kill_table_k{10,100}.csv, writes report.typ, compiles to report.pdf
via typst (on a host with typst, e.g. home at ~/.cargo/bin/typst).

Usage (on home): python3 make_report.py     # -> report.pdf
"""
from __future__ import annotations

import csv
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
_TYPST_CANDIDATES = (
    shutil.which("typst"),
    "/home/etorres/.cargo/bin/typst",
    "/home/etorres/.local/bin/typst",
)


def _find_typst():
    for p in _TYPST_CANDIDATES:
        if p and pathlib.Path(p).is_file():
            return p
    return None


def pm(m, s):
    return f"{float(m):.3g} (± {float(s):.1e})"


def load_table(k):
    rows = []
    for r in csv.DictReader(open(HERE / f"results/kill_table_k{k}.csv", newline="")):
        step = r.get("step_time_mean", "nan")
        s = "n/a" if not step or step == "nan" else f"{float(step):.3g}"
        rows.append(
            f"  [`{r['arm']}`], [{pm(r['rel_l2_mean'], r['rel_l2_std'])}], "
            f"[{pm(r['band_l2_mean'], r['band_l2_std'])}], [{s}], [{r['stability']}],"
        )
    return "\n".join(rows)


def main() -> int:
    t10 = load_table(10)
    t100 = load_table(100)

    typ = f"""#import "@preview/bloated-neurips:0.8.0": appendix, botrule, midrule, neurips2026, paragraph, toprule, url

#show: neurips2026.with(
  title: [Anisotropic Learnable-Shape RBF at an Interface: Kansa Base vs Gradient-Refined vs ePIL VSD],
  keywords: ("Anisotropic RBF", "Interface", "Kansa", "Gradient refinement", "Kappa-wall"),
  abstract: [
    On an interface-Gamma Poisson problem (piecewise kappa, jumps 1:10 and 1:100) where a smooth
    isotropic RBF basis genuinely struggles at the discontinuity, we ask whether an
    anisotropic, learnable-shape RBF (the ePIL VSD kernel) beats a one-shot isotropic Kansa base
    and an isotropic gradient-refined RBF *beyond noise*. The interface supplies the real
    residual the smooth toy lacked: the Kansa base sits at 21--25% rel-$L_2$, and isotropic
    gradient refinement (rbf-grad) beats it beyond noise at BOTH kappa settings (16% vs 21% at
    kappa=10; 24% vs 25% at kappa=100) -- the first clean classical-gradient win. The anisotropic
    arm (rbf-shape) does NOT beat base or rbf-grad on the whole domain, because its shapes stay
    isotropic under a kappa-wall (anisotropic Gram condition number ~1e18), but it does beat the
    base on the interface band beyond noise at both settings. A small MLP PINN (compound-loss) is
    best on the whole domain (10%, 16%); the error-PDE corrector collapses at kappa=100 (59%).
  ],
)

The experiment is the direct sequel to q1_surrogate_vs_compound, which could not test
anisotropic shapes on a smooth toy (base saturated; residual landscape kappa-stiff). The
interface-Gamma problem (vertical line Gamma = x = 0.5) has a genuine derivative kink, so a
low-resolution isotropic basis cannot represent it and the one-shot Kansa base struggles. We
compare, from the SAME shared RBF centers and Kansa base init: rbf-base (one-shot Kansa),
rbf-grad (isotropic gradient refinement), rbf-shape (ePIL VSD anisotropic, learnable shape),
correction-field (error-PDE), surrogate-target (MLP distilled onto the analytic projection),
and compound-loss (MLP PINN). 3 seeds per setting, 1200 steps, wall-clock + stability reported.
Beyond-noise claims require the arm separation to exceed the pooled cross-seed std.

The ePIL VSD kernel (rbf_kernel.py, ported from ePIL-RBF-1) is an anisotropic rotated Gaussian:
sigma = exp(log_sigma), inv_cov = R(angle) diag(1/sigma^2) R(angle)^T, phi = exp(-1/2 quad).
Weights are NOT tanh-clamped so the Kansa base is inherited exactly at init (init rel-L2 vs base
~5e-6). Build the divergence operator minus-kappa Laplacian per side via JAX autograd; interface
rows enforce continuity and the flux saltus.

== Results (kappa-jump = 10) -- rel-L2 and interface-band L2, mean +- std over seeds

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 5pt,
  align: (left, center),
  [`arm`], [`rel-$L_2$`], [`Gamma-band $L_2$`], [`s/step`], [`stability`],
{t10}
)

*Gates (kappa=10):* rbf-grad beats base beyond noise = *True* (0.158 vs 0.213).
rbf-shape beats base beyond noise = *False* on the whole domain (0.211 vs 0.213), but *True* on
the interface band (0.0814 vs 0.0906). rbf-shape beats rbf-grad = *False*. Shapes moved at the
noise floor (mean|Delta log-sigma| ~3e-8), aspect ~1.0; anisotropic Gram cond ~7e17 (kappa-wall).

== Results (kappa-jump = 100) -- rel-L2 and interface-band L2, mean +- std over seeds

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 5pt,
  align: (left, center),
  [`arm`], [`rel-$L_2$`], [`Gamma-band $L_2$`], [`s/step`], [`stability`],
{t100}
)

*Gates (kappa=100):* rbf-grad beats base beyond noise = *True* (0.237 vs 0.253).
rbf-shape beats base beyond noise = *False* on the whole domain, but *True* on the interface band
(0.0475 vs 0.0502). *correction-field collapses* (0.586 vs base 0.253). *compound-loss best on
domain* (0.162). Shapes remain isotropic (kappa-wall).

#figure(
  image("figures/rel_l2_by_arm.png", width: 82%),
  caption: [rel-$L_2$ by arm (log scale, mean over seeds).],
)

#figure(
  image("figures/training_curves.png", width: 82%),
  caption: [Training curves (loss / rel-$L_2$ vs step) per trained arm.],
)

#figure(
  image("figures/shape_evolution.png", width: 82%),
  caption: [Anisotropic shape-parameter evolution for rbf-shape (sigma, angle over training).],
)

#figure(
  image("figures/solution_error_maps.png", width: 82%),
  caption: [Solution / error maps per arm, showing the interface discontinuity.],
)

#figure(
  image("figures/u_exact.png", width: 70%),
  caption: [Manufactured solution u-star with the interface kink.],
)

== Conclusions

- **Gradient refinement finally wins at the interface.** Isotropic rbf-grad beats one-shot
  Kansa beyond noise at BOTH kappa jumps (0.158 vs 0.213 at kappa=10; 0.237 vs 0.253 at
  kappa=100) -- the first clean classical-gradient win, enabled by real residual at Gamma.
- **Anisotropic shape refinement stays kappa-limited.** rbf-shape does not beat base or rbf-grad
  on the whole domain; the ePIL shapes do not move (mean Delta log-sigma ~3e-8, aspect ~1.0) under
  an anisotropic Gram condition ~1e18. The anisotropic question remains condition-dominated here,
  not a clean win. It does beat base on the interface band beyond noise at both settings (a
  reproducible signal that is nonetheless fragile given the frozen shapes).
- **A MLP PINN (compound-loss) is best on the whole domain** (0.101 at kappa=10; 0.162 at
  kappa=100), reversing the smooth-toy conclusion where the RBF base dominated.
- **The error-PDE corrector is base-quality dependent**: it helped in q1 (good base, 1e-4) but
  collapses at kappa=100 from a 25%-error base (0.586).
- **Kappa-wall / conditioning is the recurring lever**: sharp residual landscapes cap how far any
  residual-gradient parameter refinement can move, echoing the q1 and RBF-FD kappa-wall findings.
"""

    (HERE / "report.typ").write_text(typ)

    typst = _find_typst()
    if not typst:
        raise SystemExit("typst not found; compile report.typ manually on a host with typst")
    print(f"compiling with {typst}")
    r = subprocess.run([typst, "compile", "report.typ"], cwd=str(HERE),
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-1500:] or "no stdout")
    if r.returncode != 0:
        print("TYPST ERROR:\n", r.stderr[-4000:])
        return r.returncode
    print(f"OK -> {(HERE/'report.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())