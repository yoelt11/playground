# splat_rbf_correctability

## About

Toys with a **synthetic PDE-like target field**, comparing two Gaussian-basis
representations: additive RBF (`I = b + Σ w_i G_i(x)`, linear in weights) vs
alpha-composited Gaussian splat (`C = Σ T_i a_i c_i`, transmittance-coupled).
The point is to probe *correctability* — whether a coarse fit can be driven to
the target by an **additive residual correction**, or only by re-optimizing the
whole parameter stack.

## Question

Are Gaussian Splats for solution approximations *correctable*, in the
VID-RBF / error-refinement sense?

Concretely: for a signed, smooth target field, an additive-RBF fit is linear in
its weights, so the residual correction `δ` solves a linear least-squares /
Galerkin system and *adds cleanly* (`u + δ`). A splat fit is nonlinear in
opacity/color (via transmittance `T_i = Π_{j<i}(1-a_j)`) — a correction cannot
live in the same span, so it must be re-baked. This experiment makes that claim
testable and quantitative.

**Hypothesis:** RBF mode admits a cheap, closed-form linear residual correction
that converges to the target as basis count grows; splat mode does not — its
error can only be reduced by re-fitting the entire stack (densification + GD),
and it will stall or fit differently for signed / oscillatory targets.

## Expected Results

- With `N` fixed Gaussians, RBF recovery error (rel-L2) drops monotonically and
  a **one-shot linear correction after a coarse fit** recovers most of the gap.
- Splat mode on the same basis shows a much smaller / non-additive correction
  benefit; closing the gap requires full re-optimization with densification.
- On a signed (non-binary, non-radiance) target, additive RBF is strictly the
  better substrate; splat's non-negativity + transmittance bias hurts recovery.

## Concrete Implementation Plan

- **Data:** synthetic 2D scalar fields (Gaussian bump blob, an oscillatory mode,
  and a smooth multiscale field) rendered to `H x W` grids; signed values OK.
  No external data required (kept in `data/`, gitignored).
- **Method (both modes):** `N` trainable anisotropic Gaussians (center, `2`-dim
  scale, rotation, plus color/weights). Fixed seed + identical init for a fair
  comparison. Adam + cosine LR, 600 steps, LR 0.025 (mirrors `gs-rbf.py`).
- **Correctability probe:** after a coarse fit, freeze basis params and solve a
  linear least-squares for the RBF weight correction; measure rel-L2 before vs
  after. Splat mode gets the same basis, and we record its re-fit cost curve.
- **Metrics:** rel-L2 error, PSNR, residual curve, correction gain.
- **Sweep:** `N ∈ {250, 1000}`, target `∈ {blob, oscillatory, multiscale}`,
  `mode ∈ {rbf, splat}` → a small Δ.

## Directory Structure

```
splat_rbf_correctability/
├── README.md              # this file
├── common.py              # GaussianImageModel (rbf + splat modes), target builders
├── run_training.py        # train one (mode, N, target) run; saves weights + curves
├── evaluate.py            # rel-L2 / PSNR / correctability probe
├── run_pipeline.py        # sweep over mode/N/target, aggregate table
├── data/                  # (gitignored) input fields
└── results/               # (gitignored) synced out via sync-experiment-results
```

## Usage

```bash
# run the whole sweep (or single (mode,N,target) in run_training.py)
launch-experiment run playground \
  "cd experiments/splat_rbf_correctability && python3 run_pipeline.py" \
  --experiment splat_rbf_correctability
```

Prereq on the GPU host: a Python env with `torch + cuda` (`python3` system lacks
torch on work as of scaffold time — install into a venv before running).

## Results

*Pending.*

## Conclusion

*Pending.*

## Follow-up

This scaffolds cleanly onto the `#crazy-ideas` thread question — see AstroBot's
analysis: correctable = RBF linear-in-weights; not-correctable = splat
alpha-compositing (transmittance coupling is a nonlinear bake).