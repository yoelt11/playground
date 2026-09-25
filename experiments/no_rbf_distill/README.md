# no_rbf_distill — residual-allocated RBF distillation of a (stochastic) Neural Operator

## About

Test whether we can distill a Neural Operator (NO) that generates an *approximate*
PDE solution at inference into a cheap RBF correction, where the NO's PDE **residual**
is used to **allocate** RBF kernels strategically (denser where residual is high).
Second phase: if the NO is **stochastic**, does predictive **uncertainty** add signal
over the raw residual for kernel allocation / trust-gated fallback to physics?

Codex-oracle validated framing: the interesting target is the **corrected surrogate**
`u_NO + v_RBF`, where `v_RBF` is an RBF field that corrects the NO's residual — NOT
pure RBF compression of the NO solution (which is a foregone positive). Shuffled-
allocation is a required control to separate "spatial emphasis matters" from "any
emphasis helps."

## Question

1. At **fixed kernel budget**, does residual-informed RBF kernel allocation beat
   uniform allocation at distilling an NO's approximate solution (measured as error of
   `u_NO + v_RBF`)?
2. Does the residual-informed allocation beat a *shuffled*-allocation control (i.e. is
   it the spatial emphasis, not just extra placement freedom, that helps)?
3. (Phase 2) Does predictive uncertainty of a **stochastic** NO add allocation signal
   over the raw residual, or is the residual already the stronger signal?

## Method (deterministic phase first — per oracle)

- **Toy PDE:** 2D elliptic interface-Γ Poisson, `−∇·(κ(x)∇u) = f` on Ω∖Γ, piecewise-κ,
  manufactured solution with a derivative kink at Γ (forces spatial heterogeneity so
  allocation matters). Reuse the shared benchmark from
  `../interface_anisotropic_rbf/common.py` (`u_exact`, `f_exact`, `kappa_of_points`,
  `make_grid`, `interface_band_mask`, `rel_l2`, Kansa helpers).
- **NO surrogate:** a small, cheap-trained Neural Operator (JAX FNO or DeepONet,
  laptop-scale) producing an approximate `u_NO` for a given κ instance.
- **Residual:** PDE residual `R(u_NO) = −∇·(κ∇u_NO) − f` (+ interface/BC mismatch).
- **Arms at FIXED kernel budget** (identical kernel count, identical solver steps;
  only placement varies):
  - `uniform`        — kernels on a uniform grid.
  - `residual-alloc` — kernel density ∝ |R(u_NO)| (e.g. weighted centroidal /
    importance-sampled placement from the residual field).
  - `shuffled-alloc` — same kernel count, placement shuffled / permuted (control).
- **Solve:** fit `v_RBF` weights by collocating `L[v_RBF] = −R(u_NO)` (Kansa / RBF-FD),
  i.e. the error-PDE corrector form.
- **Metric (single discriminative):** relative-L2 error of `u_NO + v_RBF` vs `u_exact`,
  at fixed total cost (kernel count + solver FLOPs tied across arms).
- **Go/no-go:** `residual_alloc` beats `uniform` by a pre-set margin (≥1.5× rel-L2
  reduction) AND `shuffled_alloc ≈ uniform` (shuffled does not match residual-alloc).
- **Implementation:** JAX, deterministic per seed, multi-seed (≥3). CPU fallback fine.

## Phase 2 (run only if phase 1 passes)

- **Stochastic NO:** 3-member tiny deep ensemble (MC-dropout acceptable only as a cheap
  preliminary, not the claim).
- **Arms:** (1) residual-only alloc, (2) residual+uncertainty alloc, (3) shuffled-
  uncertainty alloc.
- **Claim only if** `residual+uncertainty` beats `residual-only` AND
  `residual+shuffled-uncertainty ≈ residual-only`. Otherwise the uncertainty branch is a
  no-go (residual is the stronger signal for known elliptic problems).

## Directory Structure

```
├── README.md              # this file
├── common.py              # shared problem + RBF kernel (imports sibling benchmark)
├── no_model.py            # cheap JAX Neural Operator (FNO/DeepONet) → u_NO
├── run_train_no.py        # train the NO surrogate
├── run_pipeline.py        # end-to-end: residual → allocate → solve v_RBF → metric
├── evaluate.py            # metric computation (rel-L2, per-arm table)
├── results/               # per-arm metrics + tables
└── figures/               # error maps, allocation density, arm table
```

## How to run

```bash
python run_train_no.py          # train the cheap NO → u_NO (CPU/GPU)
python run_pipeline.py          # residual-allocated distillation + metric
python evaluate.py              # print arm table / go-no-go
```

## Status

Phase 1 (deterministic residual-allocated distillation) being implemented.
Phase 2 (stochastic NO + uncertainty) gated on phase 1.