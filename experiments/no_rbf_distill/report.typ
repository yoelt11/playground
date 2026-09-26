// no_rbf_distill — Distillation of an approximate Neural-Operator solution
// via residual-allocated RBF kernels (Phase 1 = deterministic; Phase 2 =
// stochastic-NO uncertainty allocation; Phase 2b = complementary-σ controls).
#show: neurips2026.with(
  title: [Distilling a Neural Operator through Residual-Allocated RBF Kernels: Does Predictive Uncertainty Add Signal?],
  authors: (Authors: Edgar Torres, second participant, AstroBot AX-7)$,
  keywords: ("RBF", "domain decomposition", "error refinement", "neural operator", "uncertainty"),
  abstract: [
    We ask whether an approximate Neural-Operator solution can be distilled into a cheap
    RBF correction whose kernels are allocated by the PDE residual, and whether the
    predictive uncertainty of a *stochastic* operator adds allocation signal on top of
    the residual. On a 2D elliptic interface-$Gamma$ Poisson problem with a fixed kernel
    budget, residual-allocated kernels reduce corrected error by $approx 1.99 times$ over
    uniform placement (Phase 1: GO). Neither a fixed-priority global residual+uncertainty
    blend (Phase 2: NO-GO, $times 0.556$) nor cleaner sigma-only / residual-split controls
    (Phase 2b: CLOSED) beat the residual alone. Spatial diagnostics show the ensemble
    uncertainty ($sigma$) and the residual carry *partially complementary* error
    information ($symbolic$), yet none of the tested uncertainty-driven allocations exploit
    it on this problem.
  ],
)

== Introduction

A Neural Operator (NO) trained on a parametric PDE family yields an *approximate*
solution $u_NO$ at inference. We test whether we can improve it cheaply by fitting an RBF
correction $v_RBF$ through an error-PDE corrector, $cal(L)[v_RBF] = -R(u_NO)$, where the
collocation kernels are allocated from the solution's PDE residual $R(u_NO)$ rather than
uniformly. We further test whether, for a *stochastic* NO (deep ensemble), the predictive
uncertainty $sigma$ adds allocation signal over $R$.

== Problem setup

2D elliptic interface-$Gamma$ Poisson problem, vertical interface $x = 0.5$,
$kappa_- = 1$, $kappa_+ = 10$, manufactured solution, $40 times 40$ grid, fixed budget
$K = 64$ isotropic Gaussian RBF centers, $epsilon = 3.5$, ridged Kansa corrector.
Corrected field $u_NO + v_RBF$; metric = relative-L2 $norm(u_NO + v_RBF - u^*) slash norm(u^*)$.
Three seeds (0,1,2); deterministic per seed; numpy float64.

== Phase 1 — residual-allocated kernels (deterministic)

Arms at identical $K$: `uniform`, `residual_alloc` (density $propto |R|$), `shuffled_alloc`
(same weight histogram, spatially scrambled control).

#figure(
  cx.nonbreakingtable(
    columns: (auto, auto, auto, auto),
    table.header(arm, mean text(weight: "bold"), std, values),
    ["uniform", "2.022e-01", "6.3e-03", "1.961/2.018/2.086"],
    ["residual_alloc", "1.017e-01", "2.0e-02", "0.930/0.874/1.248"],
    ["shuffled_alloc", "2.034e-01", "4.3e-03", "1.989/2.038/2.076"],
    ["u_NO (no corr)", "1.575e-01", "~1e-4", "—"],
  ),
  caption: [Phase 1 corrected rel-L2 (values per seed 0/1/2). residual_alloc beats uniform
    $times 1.99$ (gate >= 1.5); shuffled ~ uniform.],
)

Verdict: **GO**. Residual-allocated placement beats uniform by $times 1.99$; the
shuffled control matches uniform, so it is the *spatial emphasis*, not placement freedom,
that helps. Notably uniform/shuffled correction is slightly *worse* than the bare $u_NO$
(2.02e-01 > 1.57e-01): a naive equal-cost corrector hurts, and only residual-localized
kernels recover.

== Phase 2 — stochastic NO and uncertainty allocation

An $M = 3$ deep ensemble yields a predictive mean $u_NO$ and epistemic std $sigma$. Arms:
`residual_only` ($propto |R|$), `residual_unc` (blend $+|R| + lambda sigma$, $lambda = 1$,
unit-mean normalized), `shuffled_unc` (sigma permuted spatially, control).

#figure(
  cx.nonbreakingtable(
    columns: (auto, auto, auto),
    table.header(arm, mean, std),
    ["residual_only", "1.032e-01", "2.18e-02"],
    ["residual_unc", "1.856e-01", "1.05e-02"],
    ["shuffled_unc", "1.447e-01", "1.13e-02"],
    ["u_NO (ens mean)", "1.575e-01", "~1e-4"],
  ),
  caption: [Phase 2 corrected rel-L2. residual_unc vs residual_only: $times 0.556$ (FAIL).],
)

Verdict: **NO-GO**. Blending $sigma$ into one global weight field dilutes residual-guided
placement in all three seeds. The shuffled control passes but cannot save the claim.

== Spatial correlation of uncertainty, residual, and error

To understand *why*, we measured how the uncertainty $sigma$, the physics residual $|R|$,
and the true corrected error $|u + v - u^*|$ relate *in space* (interior nodes, mean over
seeds).

#figure(
  cx.nonbreakingtable(
    columns: (auto, auto, auto),
    table.header(pair, Pearson, Spearman),
    [$sigma$ and $|R|$, "+0.207", "+0.221"],
    [$sigma$ and err, "+0.442", "+0.450"],
    [$|R|$ and err, "+0.598", "+0.657"],
  ),
  caption: [Spatial correlation (interior). The residual is the strongest predictor of where
    the correction is wrong; $sigma$ is genuinely informative but weakly related to $|R|$.],
)

Top-10% overlap (mean): $sigma cap$ err = 0.29, $|R| cap$ err = 0.49, $sigma cap |R|$ = 0.28.
The ensemble and the residual flag *partially different* high-error regions — yet the
fixed-priority blend dilutes placement rather than sharpening it.

== Phase 2b — complementary-$sigma$ controls

If $sigma$ flags error that the residual misses, then dedicated controls are: $sigma$-only
allocation, and a *split* allocation where $alpha K$ kernels are sigma-placed and
$(1 - alpha) K$ are residual-placed (no global blend), $alpha = 0.25$ ($K = 64$).

#figure(
  cx.nonbreakingtable(
    columns: (auto, auto, auto),
    table.header(arm, mean, values),
    ["residual_only", "1.03e-01", "0.957/0.862/1.28"],
    ["sigma_only", "1.92e-01", "2.16/1.81/1.80"],
    ["residual_split_sigma", "1.72e-01", "1.67/1.77/1.73"],
  ],
  caption: [Phase 2b corrected rel-L2 (values per seed x1e-1). sigma-only $times 0.54$,
    split $times 0.60$ vs residual_only (gate >= 1.2).],
)

Center-overlap with top-10% true error (mean): residual-placed 0.375, sigma-placed 0.219
(split-sigma subset 0.167). Sigma-dedicated kernels hit true high-error sites *less* often
than residual-placed ones.

Verdict: **UNCONFIRMED / CLOSED** for the uncertainty branch on this toy. $sigma$ is not a
usable sole allocator, and giving it a dedicated subset alongside the residual does not help.

== Discussion and disclaimer

On this 2D elliptic-interface benchmark the raw PDE residual is the dominant allocation
signal; the ensemble uncertainty, while spatially informative ($sigma leftrightarrow$ err
~ 0.44) and only weakly collinear with the residual ($sim 0.21$), does not translate into
better placement through any of the three mechanisms tested (global blend, $sigma$-only,
dedicated subset).

> *Disclaimer (Phase 2):* the negative result for uncertainty-guided allocation is
> specific to this toy problem, a single geometric interface (vertical $Gamma$), one
> residual-consistent corrector, and an $M = 3$ spectral ensemble. Further experiments are
> needed to fully discard the uncertainty signal: e.g. stronger/finer-grained stochastic
> operators (larger independent ensembles, variational posteriors), a metric that treats
> each kernel's *marginal* resolution benefit rather than a global priority, and interface
> geometries where the residual and uncertainty disagree more sharply. Until then, the
> uncertainty branch is best read as *not yet shown helpful here*, not *universally
> useless*.

== Conclusion

Residual-allocated kernel distillation of an approximate NO solution works and materially
beats uniform placement at fixed cost (Phase 1, GO). The predictive uncertainty of a
stochastic NO does not add allocation signal on this problem under the tested mechanisms
(Phase 2 NO-GO, Phase 2b CLOSED), warranting a caveat before generalizing the negative.
