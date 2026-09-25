"""Cheap Neural-Operator-style surrogate expressed in numpy (float64).

A low-order separable spectral head: polynomials in x × sine modes in y.
Capacity is intentionally too small to resolve the κ-interface kink, so the PDE
residual concentrates near Γ — the heterogeneity residual allocation needs.
Deterministic per seed; no JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@dataclass
class TinyNOConfig:
    """n_poly / n_sine control capacity (keep small so residual stays interface-peaked)."""

    n_poly: int = 2          # powers x^0..x^{n_poly-1} (linear family = MS shape, one slope)
    n_sine: int = 1          # sin(π y) only — matches MS y-factor
    n_train: int = 2500
    ridge: float = 1e-8
    label_noise: float = 0.03


def _features(xy: np.ndarray, n_poly: int, n_sine: int) -> np.ndarray:
    """Φ = x^p · sin(q π y), p=0..P-1, q=1..Q. Shape (N, P·Q)."""
    xy = np.asarray(xy, dtype=np.float64)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    cols = []
    for q in range(1, n_sine + 1):
        s = np.sin(q * np.pi * y)
        xp = np.ones_like(x)
        for _p in range(n_poly):
            cols.append(xp * s)
            xp = xp * x
    return np.concatenate(cols, axis=1)


class TinyNO:
    """Low-capacity spectral surrogate (numpy float64)."""

    def __init__(self, cfg: TinyNOConfig | None = None, seed: int = 0):
        self.cfg = cfg or TinyNOConfig()
        self.seed = int(seed)
        n = self.cfg.n_poly * self.cfg.n_sine
        self.coeffs = np.zeros(n, dtype=np.float64)

    def forward(self, xy: np.ndarray) -> np.ndarray:
        phi = _features(xy, self.cfg.n_poly, self.cfg.n_sine)
        return (phi @ self.coeffs).reshape(-1)

    def predict(self, points: np.ndarray) -> np.ndarray:
        return self.forward(points)

    def fit_ls(self, xy: np.ndarray, u: np.ndarray) -> float:
        phi = _features(xy, self.cfg.n_poly, self.cfg.n_sine)
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        a = phi.T @ phi + self.cfg.ridge * np.eye(phi.shape[1])
        b = phi.T @ u
        self.coeffs = np.linalg.solve(a, b)
        pred = phi @ self.coeffs
        denom = float(np.linalg.norm(u) + 1e-15)
        return float(np.linalg.norm(pred - u) / denom)

    def to_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "cfg": {
                "n_poly": self.cfg.n_poly,
                "n_sine": self.cfg.n_sine,
                "n_train": self.cfg.n_train,
                "ridge": self.cfg.ridge,
                "label_noise": self.cfg.label_noise,
            },
            "coeffs": self.coeffs.copy(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TinyNO":
        cfg_raw = dict(state["cfg"])
        # Back-compat with earlier n_modes checkpoints
        if "n_modes" in cfg_raw and "n_poly" not in cfg_raw:
            m = int(cfg_raw.pop("n_modes"))
            cfg_raw["n_poly"] = m
            cfg_raw["n_sine"] = m
        cfg = TinyNOConfig(**cfg_raw)
        model = cls(cfg=cfg, seed=int(state["seed"]))
        model.coeffs = np.asarray(state["coeffs"], dtype=np.float64).reshape(-1)
        return model


def checkpoint_path(seed: int, data_dir: Path | None = None) -> Path:
    d = data_dir or DATA_DIR
    return d / f"no_seed{int(seed)}.npz"


def save_model(model: TinyNO, path: Path | None = None) -> Path:
    path = path or checkpoint_path(model.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, state=np.asarray(model.to_state(), dtype=object))
    return path


def load_model(path: Path) -> TinyNO:
    raw = np.load(path, allow_pickle=True)
    state = raw["state"].item()
    return TinyNO.from_state(state)


def train_no_surrogate(
    u_exact_fn,
    seed: int = 0,
    cfg: TinyNOConfig | None = None,
    kappa_m: float = 1.0,
    kappa_p: float = 10.0,
    gamma: str = "vertical",
    verbose: bool = True,
) -> tuple[TinyNO, dict[str, float]]:
    """Least-squares fit of spectral features to u* (seeded train cloud)."""
    cfg = cfg or TinyNOConfig()
    rng = np.random.default_rng(seed)
    model = TinyNO(cfg=cfg, seed=seed)

    xy = rng.uniform(0.0, 1.0, size=(cfg.n_train, 2))
    u = np.asarray(u_exact_fn(xy, kappa_m, kappa_p, gamma), dtype=np.float64)
    noise = cfg.label_noise * float(np.std(u) + 1e-12)
    u_noisy = u + rng.normal(0.0, noise, size=u.shape)

    train_rel_noisy = model.fit_ls(xy, u_noisy)
    pred_clean = model.predict(xy)
    train_rel = float(np.linalg.norm(pred_clean - u) / (np.linalg.norm(u) + 1e-15))

    info = {
        "train_rel_l2": train_rel,
        "train_rel_l2_noisy_fit": float(train_rel_noisy),
        "noise_std": float(noise),
        "n_poly": float(cfg.n_poly),
        "n_sine": float(cfg.n_sine),
        "n_train": float(cfg.n_train),
    }
    if verbose:
        print(
            f"  [NO spectral seed={seed}] poly={cfg.n_poly} sine={cfg.n_sine}  "
            f"train rel-L2 (clean)={train_rel:.4e}",
            flush=True,
        )
    return model, info
