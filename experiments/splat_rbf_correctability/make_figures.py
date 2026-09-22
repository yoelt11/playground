#!/usr/bin/env python3
"""Plot sweep.csv metrics into results/figures/*.png (numpy + matplotlib only)."""
from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "sweep.csv"
DEFAULT_OUT = HERE / "results" / "figures"

TARGETS = ("blob", "oscillatory", "multiscale")
MODES = ("rbf", "splat")
# linestyle per target; marker per mode
LINESTYLES = {"blob": "-", "oscillatory": "--", "multiscale": ":"}
MARKERS = {"rbf": "o", "splat": "s"}
COLORS = {
    ("rbf", "blob"): "#1b9e77",
    ("rbf", "oscillatory"): "#d95f02",
    ("rbf", "multiscale"): "#7570b3",
    ("splat", "blob"): "#66c2a5",
    ("splat", "oscillatory"): "#fc8d62",
    ("splat", "multiscale"): "#8da0cb",
}


def load_rows(csv_path: pathlib.Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    return float(v)


def fig_rel_l2_vs_n(rows: list[dict], out: pathlib.Path) -> None:
    """(a) log-scale rel_l2 vs n; one line per (mode, target)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    for mode in MODES:
        for target in TARGETS:
            pts = [
                (int(r["n"]), _f(r, "rel_l2"))
                for r in rows
                if r["mode"] == mode and r["target"] == target
            ]
            pts = [(n, v) for n, v in pts if v is not None]
            pts.sort(key=lambda t: t[0])
            if not pts:
                continue
            ns, ys = zip(*pts)
            ax.plot(
                ns,
                ys,
                color=COLORS[(mode, target)],
                linestyle=LINESTYLES[target],
                marker=MARKERS[mode],
                markersize=6,
                linewidth=1.6,
                label=f"{mode}/{target}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$ (basis size)")
    ax.set_ylabel(r"rel-$L_2$ (lower better)")
    ax.set_xticks([50, 250, 1000])
    ax.set_xticklabels(["50", "250", "1000"])
    ax.grid(True, which="both", alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "rel_l2_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def fig_mean_rel_l2_bars(rows: list[dict], out: pathlib.Path) -> None:
    """(b) grouped bars: mean rel_l2 by mode, grouped by target."""
    means = {}
    for mode in MODES:
        for target in TARGETS:
            vals = [
                _f(r, "rel_l2")
                for r in rows
                if r["mode"] == mode and r["target"] == target
            ]
            vals = [v for v in vals if v is not None]
            means[(mode, target)] = float(np.mean(vals)) if vals else float("nan")

    x = np.arange(len(TARGETS), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    rbf_vals = [means[("rbf", t)] for t in TARGETS]
    splat_vals = [means[("splat", t)] for t in TARGETS]
    ax.bar(x - width / 2, rbf_vals, width, label="rbf", color="#1b9e77")
    ax.bar(x + width / 2, splat_vals, width, label="splat", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels(list(TARGETS))
    ax.set_ylabel(r"mean rel-$L_2$ over $n$")
    ax.set_xlabel("target")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "mean_rel_l2_by_target.png", bbox_inches="tight")
    plt.close(fig)


def fig_correct_gain_vs_n(rows: list[dict], out: pathlib.Path) -> None:
    """(c) correct_gain vs n for RBF only (one line per target)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    for target in TARGETS:
        pts = [
            (int(r["n"]), _f(r, "correct_gain"))
            for r in rows
            if r["mode"] == "rbf" and r["target"] == target
        ]
        pts = [(n, v) for n, v in pts if v is not None]
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        ns, ys = zip(*pts)
        ax.plot(
            ns,
            ys,
            color=COLORS[("rbf", target)],
            linestyle=LINESTYLES[target],
            marker=MARKERS["rbf"],
            markersize=6,
            linewidth=1.6,
            label=target,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$ (basis size)")
    ax.set_ylabel(r"correct gain (before $-$ after)")
    ax.set_xticks([50, 250, 1000])
    ax.set_xticklabels(["50", "250", "1000"])
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, title="target")
    fig.tight_layout()
    fig.savefig(out / "correct_gain_vs_n.png", bbox_inches="tight")
    plt.close(fig)


def main(csv_path: pathlib.Path, out_dir: pathlib.Path) -> int:
    rows = load_rows(csv_path)
    if not rows:
        print("no data; skipping figures")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_rel_l2_vs_n(rows, out_dir)
    fig_mean_rel_l2_bars(rows, out_dir)
    fig_correct_gain_vs_n(rows, out_dir)
    print(f"wrote figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    raise SystemExit(main(args.csv, args.out))
