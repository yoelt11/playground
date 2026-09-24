#!/usr/bin/env python3
"""Γ-band rel-L2 figure: compound-vstar(λ=500) vs compound-loss / rbf-grad / surrogate per cell."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ref = json.loads((HERE/"results/sweep/sweep_stats.json").read_text())
cvs = json.loads((HERE/"results/sweep_cvstar/sweep_cvstar_stats.json").read_text())
LAM = "500.0"
CELLS = [f"g={g}|k={k}|r={r}" for g in ("vertical","circle") for k in (10,100) for r in (40,80)]
def ref_band(cell,arm): return ref["cells"][cell]["arms"].get(arm,{}).get("band_l2",{}).get("mean",float("nan"))
def cvs_band(cell): return cvs["cells"][cell][LAM]["arms"]["compound-vstar"]["band_l2"]["mean"]
def short(c): g,k,r=c.split("|"); return f"{g[2:][0]}{int(k.split('=')[1])}/{r.split('=')[1]}"

fig, ax = plt.subplots(figsize=(11,5.2))
w=0.19
labels=[short(c) for c in CELLS]
series=[("compound-vstar λ500","#9467bd", [cvs_band(c) for c in CELLS])]
# refs present in sweep
for name,color in [("compound-loss","#d62728"),("rbf-grad","#1f77b4"),("surrogate","#2ca02c"),("rbf-base","0.55")]:
    kk = "surrogate-target" if name=="surrogate" else name
    v=[ref_band(c,kk) for c in CELLS]
    if any(np.isfinite(x) for x in v): series.append((name,color,v))
x=np.arange(len(CELLS))
for i,(nm,color,v) in enumerate(series):
    off=(i-1.5)*w
    bars=ax.bar(x+off, v, w, color=color, label=nm, alpha=0.9, edgecolor="k", lw=0.4)
ax.set_yscale("log"); ax.set_ylim(0.03,0.6)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
ax.axvline(3.5, color="0.4", lw=1, ls=":")
ax.axvline(3.5, color="0.4", lw=0)
ax.text(1.9,0.32,"vertical",fontsize=9); ax.text(4.3,0.32,"circle",fontsize=9)
ax.set_ylabel("Γ-band rel-L2 (log)"); ax.set_xlabel("cell (g / κ / res)")
ax.set_title("Interface-band error by arm per cell — v*-regularized PINN wins on the circle (6 seeds)", fontsize=10)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out=HERE/"results/sweep_cvstar/band_vs_refs.png"; out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out, dpi=150); plt.close(fig); print("WROTE", out, out.stat().st_size//1024,"KB")