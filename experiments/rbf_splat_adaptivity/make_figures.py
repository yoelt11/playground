#!/usr/bin/env python3
"""Plot adaptivity sweep results into results/figures/*.png (numpy + matplotlib only).

Figures:
  (a) rel_l2_vs_arm        bar: rollout rel-L2 (mean over seeds) per arm
  (b) step_time_vs_arm     bar: per-step wall-clock (mean over seeds)
  (c) energy_vs_accuracy   scatter: E_kin vs relL2@K
  (d) rel_l2_vs_seed       line: rollout rel-L2 per arm vs seed
  (e) rollout_curves       rel-L2 vs propagation step, per arm (log-y) [needs preds]
  (f) prediction_panels    target | prediction | |error| at rollout horizon [needs preds]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "results" / "adaptivity_summary.json"
DEFAULT_SEEDED = HERE / "results" / "adaptivity.json"
DEFAULT_OUT = HERE / "results" / "figures"
DEFAULT_PREDS = HERE / "results" / "preds"

ARMS = ("rbf", "rbf-grad", "splat")
COLORS = {"rbf": "#1b9e77", "rbf-grad": "#d95f02", "splat": "#7570b3"}


def load_summary(path):
    with path.open() as f:
        return json.load(f)


def load_seeded(path):
    with path.open() as f:
        return json.load(f)


def order_arms(keys):
    return [k for k in ARMS if k in keys] + [k for k in keys if k not in ARMS]


def _bar(labels, means, stds, ylabel, title, out, log=False, fmt="{:.4f}"):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    x = np.arange(len(labels))
    colors = [COLORS.get(l, "#333333") for l in labels]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    for xi, m in zip(x, means):
        ax.text(xi, (m or 1e-6) * (3.0 if log else 1.02), fmt.format(m),
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_rel_l2_vs_arm(summary, out):
    arms = order_arms(summary.keys())
    means = [summary[a]["rel_l2_rollout"] for a in arms]
    stds = [summary[a]["rel_l2_rollout_std"] for a in arms]
    _bar(arms, means, stds, "rollout rel-$L_2$ (lower better)",
         "Rollout error by arm", out / "rel_l2_vs_arm.png", log=True)


def fig_step_time_vs_arm(summary, out):
    arms = order_arms(summary.keys())
    means = [summary[a]["mean_step_time_s"] for a in arms]
    stds = [summary[a]["mean_step_time_s_std"] for a in arms]
    _bar(arms, means, stds, "per-step wall-clock (s)",
         "Propagation cost per step", out / "step_time_vs_arm.png", log=True)


def fig_energy_vs_accuracy(summary, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    arms = order_arms(summary.keys())
    for a in arms:
        ax.scatter(summary[a]["mean_kinetic_energy"], summary[a]["rel_l2_rollout"],
                   s=120, color=COLORS.get(a, "#333"), label=a, edgecolors="black", zorder=3)
        ax.annotate(a, (summary[a]["mean_kinetic_energy"], summary[a]["rel_l2_rollout"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("mean kinetic transport energy")
    ax.set_ylabel("rollout rel-$L_2$ (lower better)")
    ax.set_yscale("log")
    ax.set_title("Energy vs accuracy", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "energy_vs_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_rel_l2_vs_seed(seeded, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    for a in ARMS:
        rows = sorted([r for r in seeded if r.get("mode") == a], key=lambda r: r["seed"])
        if not rows:
            continue
        xs = [r["seed"] for r in rows]
        ys = [r["rel_l2_rollout"] for r in rows]
        ax.plot(xs, ys, "o-", color=COLORS[a], label=a, linewidth=1.6, markersize=6)
    ax.set_xlabel("seed")
    ax.set_ylabel("rollout rel-$L_2$")
    ax.set_xticks(sorted({r["seed"] for r in seeded}))
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rel_l2_vs_seed.png", bbox_inches="tight")
    plt.close(fig)


def fig_rollout_curves(preds_dir, out, seed=0):
    """rel-L2 vs propagation step, per arm (from saved preds npz)."""
    if not preds_dir or not preds_dir.is_dir():
        print("  (no preds dir; skipping rollout curves)")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    plotted = False
    for mode in ARMS:
        f = preds_dir / f"{mode}_seed{seed}.npz"
        if not f.is_file():
            continue
        d = np.load(f)
        preds, tgt = d["preds"], d["target"]
        T = preds.shape[0]
        rels = [np.linalg.norm(preds[t] - tgt[t]) / (np.linalg.norm(tgt[t]) + 1e-8)
                for t in range(T)]
        ax.plot(range(T), rels, "o-", color=COLORS[mode], label=mode,
                linewidth=1.6, markersize=5)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("propagation step (frame index)")
    ax.set_ylabel("rel-$L_2$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rollout_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_prediction_panels(preds_dir, out, seed=0, horizon=None):
    """target | prediction | |error| at the rollout horizon; one row per arm."""
    if not preds_dir or not preds_dir.is_dir():
        print("  (no preds dir; skipping prediction panels)")
        return
    npzs = {}
    for mode in ARMS:
        f = preds_dir / f"{mode}_seed{seed}.npz"
        if f.is_file():
            d = np.load(f)
            npzs[mode] = (d["preds"], d["target"])
    if not npzs:
        print("  (no preds npz files; skipping prediction panels)")
        return
    T = next(iter(npzs.values()))[0].shape[0]
    h = min(horizon, T - 1) if horizon is not None else T - 1
    arms = [m for m in ARMS if m in npzs]
    fig, axes = plt.subplots(len(arms), 3, figsize=(10, 4.4 * len(arms)), dpi=130)
    axes = np.asarray(axes).reshape(len(arms), 3)
    vmin = min(tgt.min() for _, tgt in npzs.values())
    vmax = max(tgt.max() for _, tgt in npzs.values())
    for row, mode in enumerate(arms):
        preds, tgt = npzs[mode]
        p, t = preds[h], tgt[h]
        err = np.abs(p - t)
        axes[row][0].imshow(t, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row][0].set_title("target" if row == 0 else "", fontsize=9)
        axes[row][0].set_ylabel(f"{mode}\nt={h}", fontsize=9)
        axes[row][0].axis("off")
        axes[row][1].imshow(p, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row][1].set_title("prediction" if row == 0 else "", fontsize=9)
        axes[row][1].axis("off")
        im = axes[row][2].imshow(err, cmap="magma")
        axes[row][2].set_title("|error|" if row == 0 else "", fontsize=9)
        axes[row][2].axis("off")
    fig.suptitle(f"CFDBench cavity, rollout frame t={h} (seed {seed})", fontsize=11)
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.87, 0.12, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax, label="|error|")
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(out / "prediction_panels.png", bbox_inches="tight")
    plt.close(fig)


def main(summary_path, seeded_path, out_dir, preds_dir):
    summary = load_summary(summary_path)
    seeded = load_seeded(seeded_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_rel_l2_vs_arm(summary, out_dir)
    fig_step_time_vs_arm(summary, out_dir)
    fig_energy_vs_accuracy(summary, out_dir)
    fig_rel_l2_vs_seed(seeded, out_dir)
    fig_rollout_curves(preds_dir, out_dir)
    fig_prediction_panels(preds_dir, out_dir)
    print(f"wrote figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=pathlib.Path, default=DEFAULT_JSON)
    ap.add_argument("--seeded", type=pathlib.Path, default=DEFAULT_SEEDED)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--preds", type=pathlib.Path, default=DEFAULT_PREDS)
    args = ap.parse_args()
    raise SystemExit(main(args.summary, args.seeded, args.out, args.preds))