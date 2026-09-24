"""Figures for interface_anisotropic_rbf."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARM_ORDER = [
    "rbf-base",
    "rbf-grad",
    "rbf-shape",
    "correction-field",
    "surrogate-target",
    "compound-loss",
]

STYLE = Path(
    "/home/etorres/Documents/github/personal/research-project-pinns/"
    "style/pitayasmoothie-light.mplstyle"
)


def _apply_style():
    if STYLE.exists():
        plt.style.use(str(STYLE))
    else:
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        })


def make_all_figures(preds: dict, results: dict, agg: dict, out_dir: Path, kappa_jump: float):
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(preds["grid"])
    u_exact = np.asarray(preds["u_exact"])
    n = int(np.sqrt(len(grid)))
    extent = [0, 1, 0, 1]

    # --- bar: rel-L2 + band ---
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(ARM_ORDER))
    means = [np.nanmean(agg[a]["rel_l2"]) for a in ARM_ORDER]
    stds = [np.nanstd(agg[a]["rel_l2"]) for a in ARM_ORDER]
    band_m = [np.nanmean(agg[a]["band_l2"]) for a in ARM_ORDER]
    band_s = [np.nanstd(agg[a]["band_l2"]) for a in ARM_ORDER]
    w = 0.38
    ax.bar(x - w / 2, means, w, yerr=stds, label="Ω rel-L2", capsize=3)
    ax.bar(x + w / 2, band_m, w, yerr=band_s, label="Γ-band rel-L2", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ARM_ORDER, rotation=25, ha="right")
    ax.set_ylabel("rel-L2")
    ax.set_yscale("log")
    ax.set_title(f"Kill bars (κ-jump={kappa_jump:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rel_l2_by_arm.png", dpi=140)
    plt.close(fig)

    # --- training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm in ARM_ORDER:
        if arm == "rbf-base":
            continue
        hist = results[arm]["history"]
        if not hist.get("rel_l2"):
            continue
        axes[0].plot(hist["step"], hist["rel_l2"], label=arm, alpha=0.85)
        if hist.get("loss"):
            axes[1].plot(hist["step"], hist["loss"], label=arm, alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_title("rel-L2 vs step")
    axes[0].set_xlabel("step")
    axes[1].set_yscale("log")
    axes[1].set_title("loss vs step")
    axes[1].set_xlabel("step")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=140)
    plt.close(fig)

    # --- shape / angle evolution ---
    if "rbf-shape" in results:
        hist = results["rbf-shape"]["history"]
        if hist.get("mean_aspect"):
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(hist["step"], hist["mean_aspect"], label="mean aspect")
            ax.plot(hist["step"], hist["mean_angle"], label="mean |angle|")
            ax.set_xlabel("step")
            ax.set_title("rbf-shape: aspect & |angle| evolution")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "shape_evolution.png", dpi=140)
            plt.close(fig)

    # --- solution / error maps ---
    show_arms = ["rbf-base", "rbf-grad", "rbf-shape", "correction-field"]
    fig, axes = plt.subplots(2, len(show_arms), figsize=(3.2 * len(show_arms), 6.2))
    ue = u_exact.reshape(n, n)
    for j, arm in enumerate(show_arms):
        up = np.asarray(preds[arm]).reshape(n, n)
        im0 = axes[0, j].imshow(up, origin="lower", extent=extent, aspect="equal")
        axes[0, j].axvline(0.5, color="w", lw=0.8, ls="--")
        axes[0, j].set_title(f"{arm}\nu")
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046)
        err = np.abs(up - ue)
        im1 = axes[1, j].imshow(err, origin="lower", extent=extent, aspect="equal")
        axes[1, j].axvline(0.5, color="w", lw=0.8, ls="--")
        axes[1, j].set_title("|u−u*|")
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046)
    fig.suptitle(f"Solution / error maps (Γ at x=0.5, κ-jump={kappa_jump:g})")
    fig.tight_layout()
    fig.savefig(out_dir / "solution_error_maps.png", dpi=140)
    plt.close(fig)

    # exact reference
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(ue, origin="lower", extent=extent, aspect="equal")
    ax.axvline(0.5, color="w", lw=1.0, ls="--")
    ax.set_title("u* (manufactured)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "u_exact.png", dpi=140)
    plt.close(fig)
    print(f"  figures → {out_dir}", flush=True)
