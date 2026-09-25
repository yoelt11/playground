#!/usr/bin/env python3
"""Train / cache the cheap numpy NO surrogate per seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DEFAULT_GAMMA,
    DEFAULT_KAPPA_JUMP,
    DEFAULT_KAPPA_M,
    DEFAULT_SEEDS,
    u_exact,
)
from no_model import (
    DATA_DIR,
    TinyNOConfig,
    checkpoint_path,
    load_model,
    save_model,
    train_no_surrogate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 0,1,2)",
    )
    p.add_argument("--kappa-jump", type=float, default=DEFAULT_KAPPA_JUMP)
    p.add_argument("--kappa-m", type=float, default=DEFAULT_KAPPA_M)
    p.add_argument("--gamma", type=str, default=DEFAULT_GAMMA)
    p.add_argument("--n-poly", type=int, default=2, help="Polynomial degree count in x")
    p.add_argument("--n-sine", type=int, default=1, help="Sine modes in y")
    p.add_argument("--force", action="store_true", help="Retrain even if cache exists")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    kappa_p = float(args.kappa_m) * float(args.kappa_jump)
    cfg = TinyNOConfig(n_poly=args.n_poly, n_sine=args.n_sine)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "kappa_m": float(args.kappa_m),
        "kappa_p": float(kappa_p),
        "kappa_jump": float(args.kappa_jump),
        "gamma": args.gamma,
        "cfg": {
            "n_poly": cfg.n_poly,
            "n_sine": cfg.n_sine,
            "n_train": cfg.n_train,
            "ridge": cfg.ridge,
            "label_noise": cfg.label_noise,
        },
        "seeds": {},
    }

    for seed in seeds:
        path = checkpoint_path(seed, args.data_dir)
        if path.exists() and not args.force:
            print(f"[cache hit] {path}", flush=True)
            load_model(path)
            meta["seeds"][str(seed)] = {"path": str(path), "cached": True}
            continue
        print(f"[train] seed={seed} → {path}", flush=True)
        model, info = train_no_surrogate(
            u_exact_fn=u_exact,
            seed=seed,
            cfg=cfg,
            kappa_m=float(args.kappa_m),
            kappa_p=kappa_p,
            gamma=args.gamma,
            verbose=True,
        )
        save_model(model, path)
        meta["seeds"][str(seed)] = {"path": str(path), "cached": False, **info}
        print(f"[saved] {path}", flush=True)

    meta_path = args.data_dir / "no_train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] {meta_path}", flush=True)


if __name__ == "__main__":
    main()
