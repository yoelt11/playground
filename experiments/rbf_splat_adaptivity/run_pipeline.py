"""Adaptivity probe on CFDBench cavity: tracking u_t -> u_{t+1} across three arms.

All three arms use the SAME representations and optimizers from this experiment
(gs-rbf common.py) — no hypernetwork, no GI-RBF. They differ only in how each
representation is propagated forward, holding compute budget per step equal for
the gradient-based arms:

  * rbf       : advect centers by local velocity, then ONE-SHOT linear lstsq on
                the frozen basis (closed-form, exploits linear-in-weights
                correctability; cheap but caps accuracy).
  * rbf-grad  : advect centers, then gradient-refine all params
                (centers/scales/angles/weights) against the next frame — same
                optimizer + step budget as splat. Fair-compute RBF arm.
  * splat     : advect centers, then gradient re-optimize the alpha-composited
                stack (its only route; non-additive so must re-bake).

Tracked scalar: speed magnitude |(u,v)| of the lid-driven cavity flow.

Metrics: per-step wall-clock, rollout rel-L2, and kernel-transport energy
(kinetic sum ||dc||^2/dt^2 and a Gaussian-mixture OT lower bound).

Seeds: pass --seeds "0 1 2" to repeat over kernel-init seeds; per-seed rows are
saved, and a mean +/- std aggregation is printed and written.

Usage: python3 run_pipeline.py [--seeds "0 1 2"]
"""
import argparse
import json
import sys
import time as _time
import pathlib
import numpy as np

import jax
import jax.numpy as jnp

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
import importlib.util
def _load_local(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(EXPERIMENTS / "splat_rbf_correctability"))
from common import (adam_step, field, field_to_basis, init_params, make_grid,  # noqa: E402
                    make_loss_and_grad)
_local = _load_local("adap_common", "common.py")
propagate_rbf = _local.propagate_rbf
load_case = _local.load_case
gaussian_ot_energy = _local.gaussian_ot_energy
kinetic_energy = _local.kinetic_energy

OUT = HERE / "results"
DATA_CASE = HERE / "data" / "cfdb" / "cavity" / "bc" / "case0000"

N_KERNELS = 250
H = W = 64
STEPS_PROP = 8             # propagation steps; cavity T=10 -> 9 transitions
ROOT_STEPS = 200           # initial fit steps at t=0 (all arms)
REFIT_STEPS = 120          # gradient-refine budget per step (rbf-grad and splat)
ROLLOUT_K = 8              # rel-L2 reported at this horizon (last frame)


def fit_params(params, grid, target, mode, steps, lr=0.02):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(val) for k, val in m.items()}
    loss_and_grad = make_loss_and_grad(mode)
    for s in range(steps):
        loss, grads = loss_and_grad(params, grid, target)
        params, m, v = adam_step(params, grads, m, v, s + 1, lr)
    return params


def rbf_propagate_and_resolve(params, grid, Vt, target_t1):
    """Cheap arm: advect centers; ONE-SHOT linear weight re-solve on frozen basis."""
    params2 = propagate_rbf(params, grid, Vt)
    G = field_to_basis(params2, grid)                          # N,H,W
    Hh, Ww = target_t1.shape[:2]
    N = len(params2["centers"])
    B = np.asarray(G.reshape(N, Hh * Ww).T, dtype=np.float64)
    y = np.asarray((target_t1 - params2["bias"]).reshape(Hh * Ww, 3), dtype=np.float64)
    w, *_ = np.linalg.lstsq(B, y, rcond=None)
    new = dict(params2)
    new["weights"] = jnp.asarray(w.astype(np.float32))
    resid = np.asarray(target_t1, float) - np.asarray(
        jnp.einsum("nhw,nc->hwc", G, new["weights"]) + new["bias"], float)
    new["bias"] = jnp.asarray(
        (np.asarray(new["bias"]) + resid.mean(axis=(0, 1))).astype(np.float32))
    return new


def propagate_step(mode, params, grid, Vt, target_t1):
    base = "rbf" if mode in ("rbf", "rbf-grad") else "splat"
    if mode == "rbf":
        return rbf_propagate_and_resolve(params, grid, Vt, target_t1)
    params2 = propagate_rbf(params, grid, Vt)
    return fit_params(params2, grid, target_t1, base, REFIT_STEPS)


def rel_l2(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-8))


def run_one(mode, u, v, grid, seed=0):
    T = u.shape[0]
    base = "rbf" if mode in ("rbf", "rbf-grad") else "splat"
    speed = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)
    targets = [jnp.stack([s, s, s], axis=-1)
               for s in (jnp.asarray(speed[t]) for t in range(T))]

    tgt0 = targets[0]
    params = init_params(tgt0, grid, n_gaussians=N_KERNELS, mode=base, seed=seed)
    t0 = _time.time()
    params = fit_params(params, grid, tgt0, base, ROOT_STEPS)
    times = [_time.time() - t0]
    rels = [rel_l2(field(params, grid, mode=base), tgt0)]
    energies = []
    preds = [np.asarray(field(params, grid, mode=base))]   # predicted grids (target-shape)

    for t in range(STEPS_PROP):
        tgt_t1 = targets[t + 1]
        Vt = (np.asarray(u[t]), np.asarray(v[t]))
        c_before = np.asarray(params["centers"]).copy()
        t0 = _time.time()
        params = propagate_step(mode, params, grid, Vt, tgt_t1)
        times.append(_time.time() - t0)
        pred_t = np.asarray(field(params, grid, mode=base))
        preds.append(pred_t)
        rels.append(rel_l2(pred_t, tgt_t1))

        c_after = np.asarray(params["centers"])
        E_ot = gaussian_ot_energy(c_before, np.ones(N_KERNELS),
                                  c_after, np.ones(N_KERNELS))
        E_kin = kinetic_energy(c_before, c_after)
        energies.append({"ot": E_ot, "kinetic": E_kin})
        if (t + 1) % 2 == 0 or t == STEPS_PROP - 1:
            print(f"  [{mode}|s{seed}] step {t+1} rel_l2={rels[-1]:.4f} "
                  f"time={times[-1]:.2f}s E_kin={E_kin:.4f}", flush=True)

    return {
        "mode": mode, "seed": seed, "n_kernels": N_KERNELS, "rollout_k": ROLLOUT_K,
        "t_frames": T,
        "fit_time_s": times[0],
        "mean_step_time_s": float(np.mean(times[1:])),
        "rel_l2_t1": rels[1],
        "rel_l2_rollout": rels[min(ROLLOUT_K, len(rels) - 1)],
        "final_rel_l2": rels[-1],
        "mean_ot_energy": float(np.mean([e["ot"] for e in energies])) if energies else 0.0,
        "mean_kinetic_energy": float(np.mean([e["kinetic"] for e in energies])) if energies else 0.0,
        "pred_grids": preds,
        "target_speed": [np.asarray(t[..., 0].astype(np.float32)) for t in targets],
    }


def _agg(rows):
    """Group by mode, aggregate mean +/- std over seeds."""
    out = {}
    for r in rows:
        m = r["mode"]
        out.setdefault(m, {"rows": []})["rows"].append(r)
    summary = {}
    for m, g in out.items():
        rs = g["rows"]
        d = {}
        for k in ("mean_step_time_s", "rel_l2_t1", "rel_l2_rollout",
                  "final_rel_l2", "mean_ot_energy", "mean_kinetic_energy"):
            vals = [r[k] for r in rs]
            d[k] = np.mean(vals)
            d[k + "_std"] = np.std(vals)
        summary[m] = d
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0",
                    help="space-separated seed list, e.g. '0 1 2'")
    ap.add_argument("--save-preds", type=str, default="",
                    help="directory to save per-(mode,seed) prediction grids (npz); "
                         "empty = don't save. npz keys: 'preds' (T,H,W), 'target' (T,H,W)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split()]
    pred_dir = pathlib.Path(args.save_preds) if args.save_preds else None

    print("loading CFDBench cavity case:", DATA_CASE)
    u, v = load_case(DATA_CASE)
    print(f"cavity case: u={u.shape} v={v.shape}")
    if u.ndim != 3:
        raise SystemExit(f"expected 3D (T,H,W); got {u.shape}")

    grid = make_grid(H, W)
    jax.devices()

    rows = []
    for mode in ("rbf", "rbf-grad", "splat"):
        for seed in seeds:
            print(f"=== {mode} seed={seed} ===", flush=True)
            r = run_one(mode, u, v, grid, seed=seed)
            print(json.dumps({
                k: v for k, v in r.items()
                if k not in ("pred_grids", "target_speed")
            }), flush=True)
            if pred_dir is not None:
                pred_dir.mkdir(parents=True, exist_ok=True)
                preds = np.stack(r["pred_grids"])[..., 0]   # T,H,W (drop channel)
                tgt = np.stack(r["target_speed"])
                np.savez_compressed(pred_dir / f"{mode}_seed{seed}.npz",
                                    preds=preds, target=tgt)
            rows.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    rows_json = [{k: v for k, v in r.items()
                  if k not in ("pred_grids", "target_speed")} for r in rows]
    with open(OUT / "adaptivity.json", "w") as fh:
        json.dump(rows_json, fh, indent=2)

    summary = _agg(rows)
    print("\n=== SUMMARY (mean over seeds %s) ===" % seeds)
    for m, d in summary.items():
        print(f"{m:9s} step={d['mean_step_time_s']:.3f}±{d['mean_step_time_s_std']:.3f}s "
              f"relL2@1={d['rel_l2_t1']:.4f}±{d['rel_l2_t1_std']:.4f} "
              f"relL2@K={d['rel_l2_rollout']:.4f}±{d['rel_l2_rollout_std']:.4f} "
              f"E_kin={d['mean_kinetic_energy']:.4f}±{d['mean_kinetic_energy_std']:.4f}")
    with open(OUT / "adaptivity_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", OUT / "adaptivity.json", "and", OUT / "adaptivity_summary.json")


if __name__ == "__main__":
    main()