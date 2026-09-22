"""Train a single (mode, N, target) run; saves weights + loss curve.

Usage:  python3 run_training.py --mode rbf --n 250 --target blob
"""
import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import (adam_step, build_target, init_params, make_grid,
                    make_loss_and_grad)

OUT = pathlib.Path("results")


def train(mode, n, target, steps=600, lr=0.025, seed=0, h=128, w=128,
          save=True):
    grid = make_grid(h, w)
    tgt = build_target(target, h, w, seed=0)
    params = init_params(tgt, grid, n_gaussians=n, mode=mode, seed=seed)

    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(val) for k, val in m.items()}
    loss_and_grad = make_loss_and_grad(mode)

    history = []
    t0 = time.time()
    for step in range(steps):
        loss, grads = loss_and_grad(params, grid, tgt)
        params, m, v = adam_step(params, grads, m, v, step + 1, lr)
        if step % 50 == 0 or step == steps - 1:
            psnr = -10 * jnp.log10(loss)
            history.append({"step": step, "loss": float(loss),
                            "psnr": float(psnr)})
            print(f"{step:4d} | MSE {float(loss):.6f} | PSNR {float(psnr):.2f} dB",
                  flush=True)

    # save weights + curve
    if save:
        sub = OUT / f"{target}_{mode}_n{n}"
        sub.mkdir(parents=True, exist_ok=True)
        weights = {k: np.asarray(v) for k, v in params.items()}
        with open(sub / "params.json", "w") as fh:
            json.dump({k: v.tolist() for k, v in weights.items()}, fh)
        with open(sub / "curve.json", "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"saved -> {sub}/")

    return params, history, float(time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rbf", "splat"], default="rbf")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--target", choices=["blob", "oscillatory", "multiscale"], default="blob")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.025)
    args = ap.parse_args()
    jax.devices()  # init backend early
    params, hist, dt = train(args.mode, args.n, args.target,
                             steps=args.steps, lr=args.lr)
    print(f"final mse={hist[-1]['loss']:.6f} ({dt:.1f}s)")