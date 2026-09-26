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

## Method (Phase 1 — deterministic; PASSED)

- **Toy PDE:** 2D elliptic interface-Γ Poisson, `−∇·(κ(x)∇u) = f` on Ω∖Γ, piecewise-κ,
  manufactured solution with a derivative kink at Γ. Reuse
  `../interface_anisotropic_rbf/common.py`.
- **NO surrogate:** tiny spectral `TinyNO` (numpy float64; no JAX) — capacity too small
  to resolve the κ-interface kink, so residual concentrates near Γ.
- **Arms at FIXED kernel budget K:** `uniform` / `residual_alloc` / `shuffled_alloc`.
- **Solve:** fit `v_RBF` by collocating `L[v] = −R(u_NO)` (error-PDE corrector).
- **Go/no-go (passed):** residual_alloc 1.017e-01 vs uniform 2.022e-01 (×1.99);
  shuffled ≈ uniform.

## Phase 2 — stochastic NO + uncertainty allocation

- **Stochastic NO:** mini deep ensemble of M=3 independent `TinyNO` members
  (seeds `seed_base`, `+1000`, `+2000`), each with its own train-cloud draw + label
  noise. `u_NO = mean`, `σ = std` across members (epistemic disagreement).
  Cached as `data/no_ens_seed{s}.npz` (`--force` to retrain). Phase-1 single-model
  checkpoints remain untouched.
- **Arms (new; do not reuse Phase-1 uniform):**
  1. `residual_only` — density ∝ |R(u_NO)| (same policy as Phase-1 residual_alloc).
  2. `residual_unc` — density ∝ unit-mean(|R|) + λ_unc · unit-mean(σ)
     (default λ_unc=1.0).
  3. `shuffled_unc` — same blend with σ spatially permuted (control).
- **Claim (pre-registered):** residual_unc beats residual_only by ≥1.2× rel-L2
  reduction AND shuffled_unc ≉ residual_unc / does not beat residual_only.
  If residual_unc ties residual_only, or shuffled matches residual_unc → **NO-GO**
  (residual already carries the signal). High corr(σ,|R|) is a valid scientific NO-GO.

## Phase 2b — complementary-σ controls (σ-only + split subsets)

Global blend of |R| and σ NO-GO'd in Phase 2. Hypothesis: σ helps only if used
**directly** (σ-only) or to place a **separate subset** of kernels alongside
residual-placed ones — not blended into one global weight field.

- **Arms at fixed K** (default α=0.25 for the split):
  1. `residual_only` — reference (same policy as Phase 2).
  2. `sigma_only` — density ∝ σ alone.
  3. `residual_split_sigma` — (1−α)·K residual-placed + α·K σ-placed, concatenated
     (independent importance samples; NOT a blend).
- **Go/no-go:** σ-only ≥1.2× vs residual_only (unexpected win); split ≥1.2× and
  multi-seed stable (the claim that can rescue uncertainty). If both fail →
  **UNCONFIRMED/CLOSED** for the uncertainty branch.

## Directory Structure

```
├── README.md                 # this file
├── common.py                 # shared problem + RBF allocation (Phase 1+2+2b)
├── no_model.py               # TinyNO + TinyNOEnsemble (numpy float64)
├── run_train_no.py           # train Phase-1 single-model NO
├── run_pipeline.py           # Phase 1 end-to-end
├── evaluate.py               # Phase 1 arm table / go-no-go
├── run_pipeline_stoch.py     # Phase 2 ensemble + residual±σ allocation
├── evaluate_stoch.py         # Phase 2 arm table / go-no-go
├── run_pipeline_2b.py        # Phase 2b σ-only + residual/σ split
├── evaluate_2b.py            # Phase 2b arm table / go-no-go
├── results/                  # per-arm metrics + tables
└── figures/                  # diagnostic maps
```

## How to run

```bash
# Phase 1
python run_train_no.py
python run_pipeline.py
python evaluate.py

# Phase 2 (same GO cell: K=64, resolution=40, vertical, kappa_jump=10)
python run_pipeline_stoch.py --seeds 0,1,2 --K 64 --resolution 40 --skip-figures
python evaluate_stoch.py --seeds 0,1,2

# Phase 2b (complementary-σ; figures default off)
python run_pipeline_2b.py --seeds 0,1,2 --K 64 --resolution 40 --skip-figures
python evaluate_2b.py --seeds 0,1,2
```

Sibling venv: `../interface_anisotropic_rbf/.venv/bin/python`.

## Status

Phase 1 (deterministic residual-allocated distillation): **GO** (passed).
Phase 2 (stochastic NO + uncertainty blend): **NO-GO** (see `results/phase2_verdict.json`).
Phase 2b (complementary-σ): see `results/phase2b_verdict.json`.
