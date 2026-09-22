"""Evaluation + the correctability probe.

Loads trained params (or trains on the fly) and reports:
  * recovery rel-L2 / PSNR after training
  * RBF corrective gain: rel-L2 before vs after a one-shot linear lstsq on the
    frozen basis (closed-form additive correction).

Usage: python3 evaluate.py --mode rbf --n 250 --target blob --correct
"""
import argparse
import json

import jax
import jax.numpy as jnp

from common import build_target, field, make_grid, rbf_residual_correction
from run_training import train

DST = __import__("pathlib").Path("results")


def rel_l2(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-8))


def evaluate(mode, n, target, correct=False, h=128, w=128, steps=600):
    params, hist, _ = train(mode, n, target, steps=steps, h=h, w=w, save=False)
    grid = make_grid(h, w)
    tgt = build_target(target, h, w)
    pred = field(params, grid, mode=mode)

    row = {
        "mode": mode, "n": n, "target": target,
        "final_mse": hist[-1]["loss"],
        "rel_l2": rel_l2(pred, tgt),
        "psnr": hist[-1]["psnr"],
    }

    if correct and mode == "rbf":
        u_corr, rel_before, rel_after, rank = rbf_residual_correction(
            params, grid, tgt)
        row.update({
            "correct_rel_before": rel_before,
            "correct_rel_after": rel_after,
            "correct_gain": rel_before - rel_after,   # >0 => correction helped
            "basis_rank": rank,
            "basis_rank_ratio": rank / n,
        })
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rbf", "splat"], default="rbf")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--target", choices=["blob", "oscillatory", "multiscale"], default="blob")
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    jax.devices()
    print(json.dumps(evaluate(args.mode, args.n, args.target,
                              correct=args.correct, steps=args.steps),
                     indent=2))