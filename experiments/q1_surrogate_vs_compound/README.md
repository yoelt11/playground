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

## The three arms

### Arm 1 — `compound-loss`  (baseline, the thing under suspicion)
Train the network `u_θ` with AdamW minimizing the compound loss
```
ℒ = ‖−Δu_θ − f‖²_collocation  +  λ‖u_θ(GT pts) − u*‖²_data
```
Record, per training step, the **gradient-conflict cosine**
`cos(grad_phys, grad_data)` between the physics term's gradient and the data
term's gradient. Expect anti-alignment / oscillation (this is the diagnosis).

### Arm 2 — `surrogate-target`  (the proposed fix)
Precompute a **single consistent target** in function space using the RBF closed
form: `v* = argmin_v ‖v − u*‖²  s.t. L[v] = f (soft)`, which is the penalized
least-squares minimizer
```
min_θ  α‖A_phys θ − f‖²  +  β‖A_data θ − u*(GT pts)‖²
```
solved analytically via normal equations (no SGD). Then train the SAME network
as Arm 1 but with a **single loss**
```
ℒ = ‖u_θ − v*‖²
```
(regress onto the projected target). No gradient conflict by construction.
Gradient-conflict cosine for this arm should be ≈ 0 (nothing to fight over).

### Arm 3 — `correction-field`
Freeze a base solve `u_base` (RBF-Kansa or the trained base network). Learn a
correction `ê` on the **error PDE** `L[ê] = residual`, where
`residual = −Δu_base − f`, with GT used **only as a constraint anchor** (never
fitted directly). Final prediction `u = u_base + ê`.

## Diagnostics (must be reported for the kill table)

1. **Gradient conflict** `cos(grad_phys, grad_data)`: time-averaged + trajectory.
   Arm 1 should show negative/oscillating values; Arm 2 ≈ 0.
2. **In-span vs out-of-span gain**: for the corrector arms (2 & 3), measure how
   much of the improvement over the base solution lies in `span(RBF basis)`
   vs orthogonal to it. If >90% of the `surrogate-target` gain is in-span, the
   claim collapses (the oracle's gate).
3. **Oracle-leak audit**: GT may be used as a *constraint anchor* only — never
   as the training target of the correction network. State how this is enforced.
4. **Stability**: loss / rel-L2 trajectory (smooth vs oscillating) for Arm 1 vs
   Arms 2/3.

## Feasibility declaration (kill-table format)

| arm | target/loss | expectation |
|---|---|---|
| `compound-loss` | `‖−Δu−f‖² + λ‖u−u*‖²` (SGD/AdamW) | oscillates; anti-aligned gradient conflict |
| `surrogate-target` | RBF closed-form projection → single `‖u−v*‖²` | stable, hits Pareto point, beats Arm 1 |
| `correction-field` | `u_base + ê`, `L[ê]=residual` | out-of-span gain; no oracle leak |

**Q1 is feasible if** `surrogate-target` reaches `relL2 @ budget ≤ 0.012` at
`< 1s/step`, **beats** `compound-loss` on rollout error, and the diagnostics show
negative-to-mild gradient conflict for Arm 1 while Arm 2 is stable, with
predominantly out-of-span gain for the correctors. If `compound-loss` ties the
surrogate, the surrogate machinery is unjustified.

`relL2` = relative L2 error `‖u_pred − u_exact‖ / ‖u_exact‖` (rollout on the full
grid), reported at a fixed training budget (e.g. steps = epochs × batches).

## Outputs

- `results/` — per-seed metrics JSON/CSV + a consolidated `results.json` with the
  kill-table summary (final rel-L2 per arm, mean±std, gradient-conflict stats,
  in/out-of-span fractions, wall-clock per step, training-step budget).
- `figures/` — training curves (loss + relL2 + gradient conflict vs step),
  solution maps per arm, discrepancy/conflict visualization.
- Print the kill-table to stdout on completion.