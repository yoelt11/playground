# interface_anisotropic_rbf

## Question
On a **heterogeneous-κ / interface-Γ** Poisson problem — where a smooth isotropic RBF
basis genuinely struggles (a derivative kink on the interface) — does an
**anisotropic, learnable-shape RBF** (ePIL VSD `(K,6)` kernel) beat a **one-shot isotropic
Kansa base** and an **isotropic gradient-refined** RBF **beyond noise**? Does the
anisotropic model help at the interface, or does it collapse on conditioning (the κ-wall)?

This is the direct sequel to `q1_surrogate_vs_compound`, which could not answer the
anisotropic-kernel question on the smooth `sin(πx)sin(πy)` toy (base saturated at 5e-4
and the residual landscape was κ-stiff, so shapes never moved). The interface-Γ problem
has real, spatially-localized residual at Γ where anisotropy has a chance to earn its keep.

## Problem (interface-Γ Poisson, piecewise-constant κ)
- Domain Ω = (0,1)², split by an interface curve Γ into Ω⁻ (κ=κ⁻) and Ω⁺ (κ=κ⁺).
  Use a simple Γ (vertical line x=0.5, and/or a centered disk) — one geometry in v1.
- PDE (divergence form): `−∇·(κ(x) ∇u) = f` on Ω∖Γ. Since κ is piecewise-constant,
  away from Γ the operator is `−κ Δu = f` (∇κ = 0) — use whichever is cleanest and
  state it. κ⁺/κ⁻ ∈ {1:10, 1:100} (two settings).
- Interface conditions across Γ: `[u] = 0` (continuity) and `[κ ∂_ν u] = g` (saltus/flux
  jump; usually g=0, i.e. `[κ ∂_ν u]=0`).
- Dirichlet BC on ∂Ω.
- **Manufactured solution**: pick an exact `u*` that has the correct kink at Γ (e.g. a
  piecewise-`x` solution matching u-continuity and the κ-flux condition, × a smooth factor
  in `y` so it is a valid non-trivial 2D test); derive `f` and interface data from it so the
  exact rel-L2 error is computable. Verify `u*` satisfies the interface conditions exactly.

## Framework / backend
- **JAX** (dr. Torres's preferred stack). Use `jax`/`optax`; small MLPs with `flax` (or
  hand-written `jax.nn`). Place tensors on GPUs via `jax.devices()` when available,
  CPU fallback otherwise. Auto-differentiate the RBF operator (`jax.jacrev`/`grad`,
  Ảnd-degree for Δu) rather than hand-deriving derivatives where it stays stable.
- Multi-seed (≥3), deterministic per seed. All arms inherit the SAME shared RBF centers
  and Kansa base (controlled).

## Arms (JAX; same budget + wall-clock reporting)
Order:
1. `rbf-base`   — one-shot Kansa collocation of `−∇·(κ∇u)=f` + interface + BC, isotropic RBF (ε).
2. `rbf-grad`   — isotropic; Adam gradient-refine weights (O(1) σ_w-normalized, per-param-group
   lr) + centers on residual+BC; honest beyond-noise gate vs base.
3. `rbf-shape`  — **anisotropic** (ePIL VSD `(K,6)` kernel, `rbf_kernel.py`): seeded from base
   with isotropic init (σ=ε, angle=0) so `Σ init == base`; trainable {weights, centers,
   log σ_x, log σ_y, angle}; Adam. This is the arm Q1 could not test.
4. `correction-field` — error-PDE corrector (u_base + ê, `L[ê] = −(κ-residual)`), GT as soft
   anchor (the winner in Q1).
5. `surrogate-target` — small JAX MLP distilled onto the analytic projection `v*` (optional if
   compute is a concern; keep for continuity with Q1).
6. `compound-loss`   — small JAX MLP, compound `‖L[u]−f‖² + λ‖u−u*‖²` (gradconflict diag).
(Keep parity with the Q1 7-arm table where meaningful; drop `v-star` — it's problem-independent
analytic, or keep as a row if trivial.)

## Mandatory diagnostics (the crux)
- **Beyond-noise gate** per RBF-refinement arm: report rel-L2 mean±std vs `rbf-base` (and for
  `rbf-shape` also vs `rbf-grad`) with an explicit `beats_base_beyond_noise: True/False
  (sep, pooled_std)` line. Only claim a win beyond noise.
- **Anisotropic shapes actually move?**: print per-seed mean|Δw|, mean|Δμ|, mean|Δlog σ|,
  mean|Δangle|, and the σ_x/σ_y/angle range after training. Do shapes adapt (aspect≠1,
  angle≠0 aligned to Γ) or collapse? This is the κ/conditioning question.
- **Interface error concentration**: report rel-L2 on Ω, and the error restricted to a
  band around Γ (does the corrector/the anisotropic arm fix the interface specifically?).
- **Conditioning**: report κ/cond of the Kansa collocation matrix and of the anisotropic
  normal-equation/Gauss-Newton system (rank, cond) per seed; track whether anisotropy
  raises κ (κ-wall). Report honestly if an arm diverges.
- **Oracle-leak** for correction-field: GT only as soft anchor; state how.
- Wall-clock per step, stability (smooth/oscillating), for all arms.

## Outcome the experiment decides
- anisotropic `rbf-shape` beats `rbf-base` (and `rbf-grad`) beyond noise at the interface →
  the anisotropic machinery (ePIL VSD) is justified for interface-κ problems.
- it ties / collapses on κ → representation is conditioning-dominated (a κ-wall result).

## Kernels / provenance
- `rbf_kernel.py` — faithful port of ePIL `src/models/ours/rbf_model.py`:
  `precompute_params` (sigmas=exp(log σ); inv_covs = R diag(1/σ²) Rᵀ; `R` from angle) and
  `fn_evaluate` (φ = exp(−½ (x−μ)ᵀ inv_cov (x−μ)); u = φ·w). TWO documented deviations from
  ePIL: (1) weight parametrization is `σ_w · w_norm` (raw, O(1)-normalized) so the Kansa base
  can be inherited at init — ePIL's `tanh(weights)` would clamp O(1e6) Kansa weights; (2) ε
  stability default kept small. Do NOT change the kernel math otherwise.

## Files / structure
Flat in this folder (playground convention):
`README.md`, `rbf_kernel.py` (ported), `common.py` (problem + shared base), `arms.py`,
`run_pipeline.py` (sweep; `--seeds`, `--kappa-jump`), `make_figures.py`, `make_report.py`
(brief Typst PDF, mirrored on q1).

## Reproducibility
Read the run command in README; commit the pipeline so a fresh clone reproduces the PDF
(`run_pipeline.py → make_figures.py → make_report.py`).

### Run commands (JAX / CPU via `.venv`)

```bash
# smoke
.venv/bin/python run_pipeline.py --smoke --seeds 0 --kappa-jump 10

# full (3 seeds), κ⁺/κ⁻ = 10 and 100
.venv/bin/python run_pipeline.py --seeds 0,1,2 --kappa-jump 10
.venv/bin/python run_pipeline.py --seeds 0,1,2 --kappa-jump 100

# figures/report from last results.json
.venv/bin/python make_figures.py   # also invoked by pipeline unless --skip-figures
.venv/bin/python make_report.py
```

Outputs: `results/seed_*_k{10,100}.json`, `results/results_k{10,100}.json`,
`results/kill_table_k{10,100}.{txt,csv}`, `figures/`.