#!/usr/bin/env python3
"""Brief Typst/PDF report for interface_anisotropic_rbf (Q1-mirrored)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report.typ"


def main():
    results_path = RESULTS / "results.json"
    if not results_path.exists():
        raise SystemExit("results/results.json missing — run run_pipeline.py first")
    data = json.loads(results_path.read_text())
    kill = data.get("kill_table", "")
    gates = data.get("gates", {})
    cfg = data.get("cfg", {})

    body = f"""#set document(title: "interface_anisotropic_rbf")
#set page(margin: 1.5cm)
#set text(size: 10pt)

= interface_anisotropic_rbf

Interface-Γ Poisson (piecewise κ), anisotropic ePIL VSD (K,6) vs isotropic
Kansa / gradient refine. κ-jump = {cfg.get('kappa_jump', '?')}; steps = {cfg.get('steps', '?')}.

== Kill table

```
{kill}
```

== Gates (JSON)

```
{json.dumps(gates, indent=2)}
```

== Notes

- Manufactured u* has exact [u]=0 and [κ ∂ν u]=0 at Γ (x=0.5).
- rbf-shape init reproduces rbf-base via `vsd_from_isotropic_base`.
- Beyond-noise claims require sep > pooled_std across seeds.
- correction-field: GT soft anchor only; L[ê]=−(κ-residual).
"""
    REPORT.write_text(body)
    print(f"wrote {REPORT}")
    # Optional typst compile if available
    import shutil
    import subprocess

    if shutil.which("typst"):
        pdf = ROOT / "report.pdf"
        subprocess.run(["typst", "compile", str(REPORT), str(pdf)], check=False)
        print(f"compiled {pdf}")
    else:
        print("typst not found; left report.typ only")


if __name__ == "__main__":
    main()
