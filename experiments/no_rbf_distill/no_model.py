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


# ---------------------------------------------------------------------------
# Phase 2: mini deep ensemble (epistemic disagreement)
# ---------------------------------------------------------------------------

ENSEMBLE_SIZE = 3
ENSEMBLE_SEED_STRIDE = 1000


def ensemble_member_seeds(seed_base: int, m: int = ENSEMBLE_SIZE) -> list[int]:
    """Independent member seeds: seed_base, seed_base+1000, seed_base+2000, …"""
    base = int(seed_base)
    return [base + i * ENSEMBLE_SEED_STRIDE for i in range(int(m))]


def ensemble_checkpoint_path(seed: int, data_dir: Path | None = None) -> Path:
    d = data_dir or DATA_DIR
    return d / f"no_ens_seed{int(seed)}.npz"


@dataclass
class TinyNOEnsemble:
    """M independent TinyNO members; predictive mean + epistemic std."""

    members: list[TinyNO]
    seed_base: int

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("ensemble needs ≥1 member")

    @property
    def m(self) -> int:
        return len(self.members)

    def predict_members(self, points: np.ndarray) -> np.ndarray:
        """Stack member predictions. Shape (M, N)."""
        preds = [m.predict(points) for m in self.members]
        return np.stack(preds, axis=0)

    def predict_mean_std(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u_no = mean over members, sigma = std (ddof=0 epistemic disagreement)."""
        stack = self.predict_members(points)
        u_mean = stack.mean(axis=0)
        # Population std over members — epistemic disagreement, not sample-of-population.
        sigma = stack.std(axis=0, ddof=0)
        return u_mean.astype(np.float64), sigma.astype(np.float64)

    def predict(self, points: np.ndarray) -> np.ndarray:
        u_mean, _ = self.predict_mean_std(points)
        return u_mean

    def to_state(self) -> dict[str, Any]:
        return {
            "seed_base": int(self.seed_base),
            "m": int(self.m),
            "members": [m.to_state() for m in self.members],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TinyNOEnsemble":
        members = [TinyNO.from_state(s) for s in state["members"]]
        return cls(members=members, seed_base=int(state["seed_base"]))


def save_ensemble(ens: TinyNOEnsemble, path: Path | None = None) -> Path:
    path = path or ensemble_checkpoint_path(ens.seed_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, state=np.asarray(ens.to_state(), dtype=object))
    return path


def load_ensemble(path: Path) -> TinyNOEnsemble:
    raw = np.load(path, allow_pickle=True)
    state = raw["state"].item()
    return TinyNOEnsemble.from_state(state)


def train_ensemble(
    u_exact_fn,
    seed_base: int = 0,
    cfg: TinyNOConfig | None = None,
    m: int = ENSEMBLE_SIZE,
    kappa_m: float = 1.0,
    kappa_p: float = 10.0,
    gamma: str = "vertical",
    verbose: bool = True,
) -> tuple[TinyNOEnsemble, dict[str, Any]]:
    """Train M independent TinyNO members on independent train-cloud draws.

    Same manufactured target and train-cloud *distribution*, but each member gets
    its own RNG subsample + label noise (seeds seed_base, +1000, +2000, …).
    Do NOT average training data across members — independence drives disagreement.
    """
    cfg = cfg or TinyNOConfig()
    member_seeds = ensemble_member_seeds(seed_base, m=m)
    members: list[TinyNO] = []
    member_infos: list[dict[str, float]] = []
    if verbose:
        print(
            f"  [ensemble seed_base={seed_base}] M={m} members "
            f"seeds={member_seeds}",
            flush=True,
        )
    for ms in member_seeds:
        model, info = train_no_surrogate(
            u_exact_fn=u_exact_fn,
            seed=ms,
            cfg=cfg,
            kappa_m=kappa_m,
            kappa_p=kappa_p,
            gamma=gamma,
            verbose=verbose,
        )
        members.append(model)
        member_infos.append(info)
    ens = TinyNOEnsemble(members=members, seed_base=int(seed_base))
    info_out: dict[str, Any] = {
        "seed_base": int(seed_base),
        "m": int(m),
        "member_seeds": member_seeds,
        "members": member_infos,
        "mean_train_rel_l2": float(
            np.mean([mi["train_rel_l2"] for mi in member_infos])
        ),
    }
    return ens, info_out
