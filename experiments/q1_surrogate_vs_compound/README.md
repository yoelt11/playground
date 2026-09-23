# q1_surrogate_vs_compound

## Question (thread Q1)

**Is a surrogate target (function-space projection) better than a compound
physics+data loss?**

The premise: a PDE residual is a *functional* value (operator range); ground
truth is a *function* value (solution space). Optimizing a compound loss
`ℒ = ‖L[u]−f‖² + λ‖u−u*‖²` under SGD wanders on the Pareto frontier — each step
improves one term and hurts the other, so the run oscillates and "deviates away
from both." The hypothesis is that replacing the two weighted terms with a
**single, consistent target** — the projection of the data onto the
physics-consistent subspace — removes the gradient conflict and reaches the
Pareto point directly.

This probe settles Q1 empirically on the existing 2D Poisson / RBF machinery
(`common.py`, reused from `discrepancy_rbf_refinement_poisson`).

### Corrected framing (post oracle review)

**Q1 narrows to: an analytic function-space projection is a stable distillation
target for a neural surrogate; the RBF base itself already beats the compound
and surrogate-MLP arms on this smooth toy.**

After the Arm-3 sign fix, the residual corrector can beat the frozen base
(as expected once `L[ê]=−residual`). The 5-row kill table separates (i) the
frozen RBF base, (ii) the analytic projection `v*`, and (iii) the MLP distill
of `v*`, so we do not credit the network for what the closed-form projection
already achieves — and we do not claim out-of-span gain from an in-span-by-
construction `v*−u_base` decomposition.

## Setup (shared by all arms — fair comparison)

- PDE: 2D Poisson `−Δu = f` on Ω = (0,1)².
- Manufactured solution: `u_exact(x,y) = sin(πx)sin(πy)`.
- Collocation points: interior grid (resolution ~40) where `L[u]=f` is imposed.
- **Sparse GT subset**: a fixed random subset of grid points (~10–15%) where
  `u*` (noisy or clean) is observed. This is the "data" the arms condition on.
  Same GT subset for every arm.
- Boundary: Dirichlet `u = u_exact` on ∂Ω.
- Same network architecture for arms 1 & 2 (arms 1 and 2 MUST use the same model
  class so the only difference is target/loss). Multi-seed (≥3 seeds), report means.
- Surrogate normal equations use a **ridge sweep**
  `ε ∈ {0, 1e-8, 1e-6, 1e-4, 1e-2}` (still closed-form); report chosen
  `ε`, rank, and condition number per seed.

## The arms (5-row honest kill table)

| order | arm | what it is |
|---|---|---|
| 0 | `rbf-base` | Frozen RBF-Kansa base solve `u_base` (required control) |
| 2a | `v-star` | Analytic surrogate projection (no MLP) |
| 2 | `surrogate-target` | Same net as Arm 1, distill onto `v*` with `‖u_θ − v*‖²` |
| 1 | `compound-loss` | `‖−Δu−f‖² + λ‖u−u*‖²` with per-step grad-conflict cosine |
| 3 | `correction-field` | Frozen `u_base` + `ê` on **sign-fixed** error PDE |

### Arm 1 — `compound-loss`  (baseline, the thing under suspicion)
Train the network `u_θ` with AdamW minimizing the compound loss
```
ℒ = ‖−Δu_θ − f‖²_collocation  +  λ‖u_θ(GT pts) − u*‖²_data
```
Record, per training step, the **gradient-conflict cosine**
`cos(grad_phys, grad_data)` between the physics term's gradient and the data
term's gradient. Expect anti-alignment / oscillation (this is the diagnosis).

### Arm 2a — `v-star` (analytic projection, no MLP)
Precompute a **single consistent target** in function space using the RBF closed
form: `v* = argmin_v ‖v − u*‖²  s.t. L[v] = f (soft)`, which is the penalized
least-squares minimizer
```
min_θ  α‖A_phys θ − f‖²  +  β‖A_data θ − u*(GT pts)‖²  + ε‖θ‖²
```
solved analytically via ridged normal equations (no SGD). Reported as its own
kill-table row so we can see whether the MLP adds anything over the analytic
projection.

### Arm 2 — `surrogate-target`  (MLP distill of v*)
Train the SAME network as Arm 1 with a **single loss**
```
ℒ = ‖u_θ − v*‖²
```
Gradient-conflict cosine for this arm is ≈ 0 (nothing to fight over).

### Arm 3 — `correction-field` (sign-fixed)
Freeze a base solve `u_base` (RBF-Kansa). Let
`residual = L[u_base] − f = (−Δu_base − f)`. The correct error PDE is
```
L[ê] = f − L[u_base] = −residual
```
so that `L[u_base + ê] = f`. Trained via `‖−Δê + residual‖²`, with GT used
**only as a constraint anchor** (never fitted directly). Final prediction
`u = u_base + ê`.

## Diagnostics (must be reported for the kill table)

1. **Gradient conflict** `cos(grad_phys, grad_data)`: time-averaged + trajectory.
   Arm 1 should show negative/oscillating values; Arm 2 ≈ 0. `n/a` for
   `rbf-base` and `v-star`.
2. **Two span decompositions (do not conflate)**:
   - **(a)** `span(v* − u_base)` w.r.t. the shared RBF basis: ~1.0 in-span
     **by construction** (base and surrogate use the same centers). Not evidence
     of non-vacuous correction.
   - **(b)** `span(u_mlp − u_base)` for each neural/correction arm: reflects MLP
     regression residual onto the shared span — **not** "out-of-span gain" /
     surrogate non-vacuity.
3. **Oracle-leak audit**: GT may be used as a *constraint anchor* only — never
   as the training target of the correction network. State how this is enforced.
4. **Stability**: loss / rel-L2 trajectory (smooth vs oscillating) for Arm 1 vs
   Arms 2/3; `analytic` for `rbf-base` / `v-star`.
5. **Surrogate conditioning**: chosen ridge `ε`, rank, and cond per seed.

## Feasibility / interpretation

| arm | target/loss | role |
|---|---|---|
| `rbf-base` | frozen Kansa solve | control — may beat every neural arm |
| `v-star` | ridged RBF projection (analytic) | upper bound on what distillation can match |
| `surrogate-target` | `‖u−v*‖²` MLP distill | tests whether the net recovers `v*` stably |
| `compound-loss` | physics+data SGD | gradient-conflict baseline |
| `correction-field` | `u_base+ê`, `L[ê]=−residual` | sign-fixed residual corrector |

On this smooth toy, expect the **RBF base** and **analytic `v*`** to dominate;
the interesting positive claim that survives is that `v*` is a **stable
distillation target** (Arm 2 smooth, no grad conflict), not that the MLP beats
the closed-form projection or invents out-of-span structure.

`relL2` = relative L2 error `‖u_pred − u_exact‖ / ‖u_exact‖` (rollout on the full
grid), reported at a fixed training budget (e.g. steps = epochs × batches).

## Outputs

- `results/` — per-seed metrics JSON/CSV + a consolidated `results.json` with the
  5-row kill-table summary (final rel-L2 per arm, mean±std, gradient-conflict
  stats, both span decompositions, wall-clock per step, ridge rank/cond).
- `figures/` — training curves, solution/error maps, `corrected_kill_table.png`.
- Print the kill-table to stdout on completion.

## Run

```bash
.venv/bin/python run_pipeline.py                  # full 3-seed, 3000 steps
.venv/bin/python run_pipeline.py --steps 200      # smoke / reduced budget
```
