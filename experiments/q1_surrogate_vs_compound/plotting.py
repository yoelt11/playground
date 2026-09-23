"""Figures for the Q1 probe."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from setup_data import ExperimentData


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def plot_training_curves(
    seed_results: list[dict],
    out_dir: Path,
) -> None:
    """loss + relL2 + grad conflict vs step for each arm (mean over seeds)."""
    _apply_style()
    arms = ["compound-loss", "surrogate-target", "correction-field"]
    colors = {
        "compound-loss": "#c0392b",
        "surrogate-target": "#2980b9",
        "correction-field": "#27ae60",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [("loss", "Loss"), ("rel_l2", "relL2 (rollout)"), ("grad_cos", "grad-conflict cos")]

    for ax, (key, title) in zip(axes, metrics):
        for arm in arms:
            series = []
            for sr in seed_results:
                h = sr["arms"][arm]["history"]
                series.append(np.asarray(h[key], dtype=float))
            # Pad to common length
            mlen = min(len(s) for s in series)
            arr = np.stack([s[:mlen] for s in series], axis=0)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            steps = np.arange(mlen)
            ax.plot(steps, mean, color=colors[arm], label=arm, lw=1.6)
            ax.fill_between(steps, mean - std, mean + std, color=colors[arm], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("step")
        if key == "loss":
            ax.set_yscale("log")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_solution_maps(
    data: ExperimentData,
    arm_preds: dict[str, np.ndarray],
    out_dir: Path,
    seed: int,
) -> None:
    _apply_style()
    res = data.pde.res
    u_ex = data.u_exact.reshape(res, res)
    panels = [("exact", u_ex)]
    for name, u in arm_preds.items():
        panels.append((name, u.reshape(res, res)))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.8))
    if n == 1:
        axes = [axes]
    vmin, vmax = float(u_ex.min()), float(u_ex.max())
    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(f"solution maps (seed={seed})", fontsize=11)
    fig.savefig(out_dir / f"solutions_seed{seed}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(
    data: ExperimentData,
    arm_preds: dict[str, np.ndarray],
    out_dir: Path,
    seed: int,
) -> None:
    _apply_style()
    res = data.pde.res
    u_ex = data.u_exact.reshape(res, res)
    names = list(arm_preds.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 2.8))
    if len(names) == 1:
        axes = [axes]
    errs = [np.abs(arm_preds[n].reshape(res, res) - u_ex) for n in names]
    vmax = max(e.max() for e in errs) + 1e-12
    for ax, name, err in zip(axes, names, errs):
        im = ax.imshow(err, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"|err| {name}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(f"abs error (seed={seed})", fontsize=11)
    fig.savefig(out_dir / f"errors_seed{seed}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
