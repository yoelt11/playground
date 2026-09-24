#!/usr/bin/env python3
"""Robust-sweep figures for interface_anisotropic_rbf.

Reads results/sweep/sweep_stats.json, writes results/sweep/figures/*.png:
  1. rel_l2_by_cell.png      - 2x4 paned bar grid (mean +/- std) per arm per cell
  2. welch_pvalue_heatmap.png- -log10(p) heatmap, gates x cells (marks significance)
  3. convergence_curves.png  - rel-L2 vs n_centers {32,48,64} per arm (log-y)
  4. kappa_normalized_error.png - effective_error (rel-L2 * cond/cond_ref) bars
  5. band_vs_domain.png      - Gamma-band L2 vs domain rel-L2 scatter per arm
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STATS = json.loads((HERE / "results/sweep/sweep_stats.json").read_text())
OUT = HERE / "results/sweep/figures"; OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["rbf-base","rbf-grad","rbf-shape","correction-field","surrogate-target","compound-loss"]
CELLS = list(STATS["cells"].keys())
GATES = ["rbf_grad_vs_base","rbf_shape_vs_base","rbf_shape_vs_grad","rbf_shape_band_vs_base"]
CCOL = {"rbf-base":"0.55","rbf-grad":"#1f77b4","rbf-shape":"#ff7f0e","correction-field":"#d62728",
        "surrogate-target":"#2ca02c","compound-loss":"#9467bd"}

def short(cell):  # g=vertical|k=10|r=40 -> v|k10|r40
    g,k,r = cell.split("|")
    return f"{g[2:][0]}|k{int(k.split('=')[1])}|r{r.split('=')[1]}"

def arm(arms,name,key,stat): return arms.get(name,{}).get(key,{}).get(stat, float("nan"))

# ---- 1. rel-L2 by cell (2x4 paned bars) ----
fig, axes = plt.subplots(2,4, figsize=(16,7), sharey=True)
for ax,cell in zip(axes.ravel(), CELLS):
    a = STATS["cells"][cell]["arms"]
    means=[arm(a,n,"rel_l2","mean") for n in ARMS]; stds=[arm(a,n,"rel_l2","std") for n in ARMS]
    bars=ax.bar(range(len(ARMS)), means, yerr=stds, color=[CCOL[n] for n in ARMS],
                capsize=3, edgecolor="k", lw=0.4)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([n.split("-")[-1] for n in ARMS], rotation=45, fontsize=7)
    ax.set_title(short(cell), fontsize=9)
    ax.axhline(means[0], color="0.3", ls="--", lw=0.8)   # base reference
axes[0,0].set_ylabel("rel-L2"); axes[1,0].set_ylabel("rel-L2")
fig.suptitle("Interface-Gamma robustness sweep — mean rel-L2 per arm (6 seeds)", fontsize=13)
fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig(OUT/"rel_l2_by_cell.png", dpi=150); plt.close(fig)

# ---- 2. Welch p-value heatmap ----
M=np.full((len(GATES),len(CELLS)), np.nan); T=np.zeros_like(M,dtype=bool)
for j,g in enumerate(GATES):
    for i,cell in enumerate(CELLS):
        gd=STATS["cells"][cell]["gates"].get(g)
        if gd:
            p=max(gd["welch"]["pvalue"],1e-12); M[j,i]=-np.log10(p); T[j,i]=gd.get("beats_beyond_noise",False)
fig,ax=plt.subplots(figsize=(10,4))
im=ax.imshow(M, cmap="magma_r", aspect="auto", vmin=0, vmax=3)
ax.set_xticks(range(len(CELLS))); ax.set_xticklabels([short(c) for c in CELLS], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(GATES))); ax.set_yticklabels([g.replace("_","\n") for g in GATES], fontsize=8)
for j in range(len(GATES)):
    for i in range(len(CELLS)):
        if not np.isnan(M[j,i]):
            ax.text(i,j,f"{-M[j,i]:.1f}",ha="center",va="center",fontsize=7,
                    color="0.1" if M[j,i]<1.3 else "white",fontweight="bold" if T[j,i] else "normal")
ax.axvline(3.5,color="w",lw=1); ax.axhline(1.5,color="w",lw=1)
ax.set_title("Paired-Welch significance: -log10(p), one-sided; boxed bold => p<0.05 (beats beyond noise)", fontsize=10)
fig.colorbar(im,label="-log10(p)  (0.05 threshold = 1.30)"); fig.tight_layout()
fig.savefig(OUT/"welch_pvalue_heatmap.png", dpi=150); plt.close(fig)

# ---- 3. convergence curves ----
cv=STATS["convergence"]["by_n_centers"]; ns=[int(n) for n in cv]
fig,ax=plt.subplots(figsize=(7,5))
for name in ARMS:
    ys=[arm(cv[str(n)]["arms"],name,"rel_l2","mean") for n in ns]
    ax.plot(ns, ys, marker="o", color=CCOL[name], label=name)
    ax.set_xlabel("n RBF centers"); ax.set_ylabel("rel-L2 (vertical|kappa10|res40)"); ax.set_yscale("log")
ax.set_title("Convergence in basis width (6 seeds)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"convergence_curves.png", dpi=150); plt.close(fig)

# ---- 4. kappa-normalized error ----
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for bi,gamma in enumerate(["vertical","circle"]):
    ax=axes[bi]; ids=[c for c in CELLS if c.startswith(f"g={gamma}")]
    txt=[short(c).replace("|"," ") for c in ids]
    x=np.arange(len(ids)); w=0.13
    for ai,name in enumerate(ARMS):
        ys=[STATS["cells"][c]["kappa_normalized"][name]["effective_error_mean"] for c in ids]
        ax.bar(x+(ai-2.5)*w, ys, w, color=CCOL[name], label=name if bi==0 else None)
    ax.set_xticks(x); ax.set_xticklabels(txt, rotation=30, ha="right", fontsize=7)
    ax.set_title(f"{gamma} — effective error = rel-L2 x cond/cond_ref (kappa-wall cost)", fontsize=9)
    ax.set_yscale("log")
axes[0].set_ylabel("effective error (log)"); axes[0].legend(fontsize=7, ncol=3)
fig.suptitle("Accuracy weighted by conditioning (cond_ref = vertical|k10|r40 Kansa cond)", fontsize=12)
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(OUT/"kappa_normalized_error.png", dpi=150); plt.close(fig)

# ---- 5. band vs domain scatter ----
fig,ax=plt.subplots(figsize=(7.5,6))
for name in ARMS:
    xs=[arm(STATS["cells"][c]["arms"],name,"rel_l2","mean") for c in CELLS]
    ys=[arm(STATS["cells"][c]["arms"],name,"band_l2","mean") for c in CELLS]
    ax.scatter(xs,ys,s=34,color=CCOL[name],label=name,alpha=0.85,edgecolor="k",lw=0.4)
ax.plot([0.5,0.9],[0.5,0.9],"k--",lw=0.8)  # parity (band ~ domain)
ax.set_xlabel("domain rel-L2"); ax.set_ylabel("Gamma-band rel-L2"); ax.set_title("Interface-band vs domain error (8 cells)")
ax.legend(fontsize=8); ax.set_xlim(0.08,0.8); ax.set_ylim(0.03,0.5)
fig.tight_layout(); fig.savefig(OUT/"band_vs_domain.png", dpi=150); plt.close(fig)

print("WROTE:")
for p in sorted(OUT.glob("*.png")): print("  ", p.name, p.stat().st_size//1024, "KB")