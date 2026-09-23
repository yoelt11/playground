"""Training loops for the Q1 arms (neural + rbf-grad)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import RBFField, SolutionNet, flat_grads, grad_cosine, laplacian
from setup_data import ExperimentData, rel_l2, set_seeds, to_torch


def _eval_model(model: nn.Module, grid: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = to_torch(grid)
        return model(x).cpu().numpy().reshape(-1)


def _make_opt(model: nn.Module, lr: float, steps: int):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    return opt, sched


def _new_net(width: int, seed: int) -> SolutionNet:
    # Fourier buffer seed fixed; weight init follows torch seed from set_seeds.
    return SolutionNet(width=width, n_freq=32, seed=123)


def train_arm1_compound(
    data: ExperimentData,
    steps: int = 3000,
    lr: float = 2e-3,
    lambda_data: float = 80.0,
    width: int = 128,
    eval_every: int = 50,
    colloc_batch: int = 512,
) -> dict:
    """Arm 1: compound physics + data loss; record gradient-conflict cosine."""
    set_seeds(data.seed)
    model = _new_net(width, data.seed)
    opt, sched = _make_opt(model, lr=lr, steps=steps)
    rng = np.random.default_rng(data.seed + 7)

    x_gt = to_torch(data.gt_pts)
    u_gt = to_torch(data.u_gt).unsqueeze(-1)
    n_int = len(data.interior)

    hist = {
        "loss": [],
        "loss_phys": [],
        "loss_data": [],
        "rel_l2": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }

    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        if colloc_batch < n_int:
            idx = rng.choice(n_int, size=colloc_batch, replace=False)
        else:
            idx = np.arange(n_int)
        x_int = to_torch(data.interior[idx], requires_grad=True)
        f_int = to_torch(data.f_interior[idx]).unsqueeze(-1)

        opt.zero_grad(set_to_none=True)

        lap = laplacian(model, x_int)
        phys = ((-lap - f_int) ** 2).mean()
        data_term = ((model(x_gt) - u_gt) ** 2).mean()

        phys.backward(retain_graph=True)
        g_phys = flat_grads(model.parameters())
        opt.zero_grad(set_to_none=True)
        data_term.backward(retain_graph=True)
        g_data = flat_grads(model.parameters())
        opt.zero_grad(set_to_none=True)
        cos = grad_cosine(g_phys, g_data)

        loss = phys + lambda_data * data_term
        loss.backward()
        opt.step()
        sched.step()
        dt = time.perf_counter() - t0

        hist["loss"].append(float(loss.item()))
        hist["loss_phys"].append(float(phys.item()))
        hist["loss_data"].append(float(data_term.item()))
        hist["grad_cos"].append(cos)
        hist["step"].append(step)
        hist["step_time"].append(dt)

        if step % eval_every == 0 or step == steps - 1:
            hist["rel_l2"].append(rel_l2(_eval_model(model, data.grid), data.u_exact))
            model.train()
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))

    u_final = _eval_model(model, data.grid)
    return {
        "arm": "compound-loss",
        "u_pred": u_final,
        "history": hist,
        "final_rel_l2": rel_l2(u_final, data.u_exact),
        "mean_step_time": float(np.mean(hist["step_time"])),
        "mean_grad_cos": float(np.mean(hist["grad_cos"])),
        "late_grad_cos": float(np.mean(hist["grad_cos"][len(hist["grad_cos"]) // 2 :])),
        "model": model,
    }


def train_arm2_surrogate(
    data: ExperimentData,
    steps: int = 3000,
    lr: float = 5e-3,
    width: int = 128,
    eval_every: int = 50,
) -> dict:
    """Arm 2: regress onto precomputed surrogate target v* (single loss)."""
    set_seeds(data.seed)
    model = _new_net(width, data.seed)
    opt, sched = _make_opt(model, lr=lr, steps=steps)

    x_all = to_torch(data.grid)
    v_star = to_torch(data.v_star).unsqueeze(-1)

    hist = {
        "loss": [],
        "rel_l2": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }

    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = ((model(x_all) - v_star) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        dt = time.perf_counter() - t0

        hist["loss"].append(float(loss.item()))
        hist["grad_cos"].append(0.0)
        hist["step"].append(step)
        hist["step_time"].append(dt)

        if step % eval_every == 0 or step == steps - 1:
            hist["rel_l2"].append(rel_l2(_eval_model(model, data.grid), data.u_exact))
            model.train()
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))

    u_final = _eval_model(model, data.grid)
    return {
        "arm": "surrogate-target",
        "u_pred": u_final,
        "history": hist,
        "final_rel_l2": rel_l2(u_final, data.u_exact),
        "mean_step_time": float(np.mean(hist["step_time"])),
        "mean_grad_cos": 0.0,
        "late_grad_cos": 0.0,
        "model": model,
        "v_star_rel_l2": rel_l2(data.v_star, data.u_exact),
    }


def _debug_corrected_residual(
    e_net: nn.Module,
    data: ExperimentData,
    n_check: int = 256,
) -> dict:
    """Verify L[u_base + e_hat] - f is driven toward zero on interior collocation.

    residual_base = L[u_base] - f = (−Δu_base − f). With L[e_hat] = −residual,
    L[u_base + e_hat] − f = residual + L[e_hat] ≈ 0.
    """
    e_net.eval()
    rng = np.random.default_rng(data.seed + 99)
    n = min(n_check, len(data.interior))
    idx = rng.choice(len(data.interior), size=n, replace=False)
    x_int = to_torch(data.interior[idx], requires_grad=True)
    residual = to_torch(data.residual_base[idx]).unsqueeze(-1)

    lap_e = laplacian(e_net, x_int)
    # L[e] = −Δe; target L[e] = −residual ⇒ (−lap_e + residual) → 0
    err_pde = (-lap_e + residual).detach().cpu().numpy().reshape(-1)
    # Full corrected residual: L[u_base]+L[e]−f = residual + L[e]
    L_e = (-lap_e).detach().cpu().numpy().reshape(-1)
    corrected = data.residual_base[idx] + L_e
    base_abs = float(np.mean(np.abs(data.residual_base[idx])))
    corr_abs = float(np.mean(np.abs(corrected)))
    return {
        "mean_abs_residual_base": base_abs,
        "mean_abs_L_u_corrected_minus_f": corr_abs,
        "mean_abs_L_e_plus_residual": float(np.mean(np.abs(err_pde))),
        "reduction_ratio": corr_abs / max(base_abs, 1e-30),
        "n_check": int(n),
        "note": (
            "L[u_base+e_hat]-f should shrink vs |residual_base|; "
            "reduction_ratio < 1 means the sign-fixed error PDE is helping."
        ),
    }


def train_arm3_correction(
    data: ExperimentData,
    steps: int = 3000,
    lr: float = 2e-3,
    width: int = 128,
    lambda_anchor: float = 5.0,
    eval_every: int = 50,
    colloc_batch: int = 512,
) -> dict:
    """Arm 3: freeze u_base; learn e_hat on error PDE; GT only as constraint anchor.

    Oracle-leak enforcement:
      - residual := L[u_base] − f = (−Δu_base − f).
      - Correct error PDE: L[e_hat] = f − L[u_base] = −residual
        (so L[u_base + e_hat] = f). Trained via ‖−Δe_hat + residual‖².
      - GT enters ONLY via soft anchor ‖(u_base + e_hat) − u*‖ at GT pts.
      - We never regress e_hat onto (u* − u_base) as the sole/primary loss.
    """
    set_seeds(data.seed)
    e_net = _new_net(width, data.seed)
    opt, sched = _make_opt(e_net, lr=lr, steps=steps)
    rng = np.random.default_rng(data.seed + 11)

    x_gt = to_torch(data.gt_pts)
    u_base_gt = to_torch(data.u_base[data.gt_idx]).unsqueeze(-1)
    u_gt = to_torch(data.u_gt).unsqueeze(-1)
    u_base_grid = data.u_base
    n_int = len(data.interior)

    hist = {
        "loss": [],
        "loss_pde": [],
        "loss_anchor": [],
        "rel_l2": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }

    e_net.train()
    for step in range(steps):
        t0 = time.perf_counter()
        if colloc_batch < n_int:
            idx = rng.choice(n_int, size=colloc_batch, replace=False)
        else:
            idx = np.arange(n_int)
        x_int = to_torch(data.interior[idx], requires_grad=True)
        residual = to_torch(data.residual_base[idx]).unsqueeze(-1)

        opt.zero_grad(set_to_none=True)

        lap_e = laplacian(e_net, x_int)
        # Sign fix: fit L[e_hat] = −residual  ⇒  (−Δe + residual) → 0
        pde_loss = ((-lap_e + residual) ** 2).mean()
        e_gt = e_net(x_gt)
        anchor = ((u_base_gt + e_gt - u_gt) ** 2).mean()

        pde_loss.backward(retain_graph=True)
        g_pde = flat_grads(e_net.parameters())
        opt.zero_grad(set_to_none=True)
        anchor.backward(retain_graph=True)
        g_anc = flat_grads(e_net.parameters())
        opt.zero_grad(set_to_none=True)
        cos = grad_cosine(g_pde, g_anc)

        loss = pde_loss + lambda_anchor * anchor
        loss.backward()
        opt.step()
        sched.step()
        dt = time.perf_counter() - t0

        hist["loss"].append(float(loss.item()))
        hist["loss_pde"].append(float(pde_loss.item()))
        hist["loss_anchor"].append(float(anchor.item()))
        hist["grad_cos"].append(cos)
        hist["step"].append(step)
        hist["step_time"].append(dt)

        if step % eval_every == 0 or step == steps - 1:
            e_hat = _eval_model(e_net, data.grid)
            hist["rel_l2"].append(rel_l2(u_base_grid + e_hat, data.u_exact))
            e_net.train()
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))

    e_final = _eval_model(e_net, data.grid)
    u_final = u_base_grid + e_final
    debug_pde = _debug_corrected_residual(e_net, data)
    print(
        f"    [arm3 PDE check] |L[u_base]-f|={debug_pde['mean_abs_residual_base']:.4e} → "
        f"|L[u_base+e]-f|={debug_pde['mean_abs_L_u_corrected_minus_f']:.4e} "
        f"(ratio={debug_pde['reduction_ratio']:.3f})",
        flush=True,
    )
    return {
        "arm": "correction-field",
        "u_pred": u_final,
        "e_hat": e_final,
        "history": hist,
        "final_rel_l2": rel_l2(u_final, data.u_exact),
        "base_rel_l2": rel_l2(u_base_grid, data.u_exact),
        "mean_step_time": float(np.mean(hist["step_time"])),
        "mean_grad_cos": float(np.mean(hist["grad_cos"])),
        "late_grad_cos": float(np.mean(hist["grad_cos"][len(hist["grad_cos"]) // 2 :])),
        "model": e_net,
        "debug_pde_check": debug_pde,
        "oracle_leak_audit": (
            "GT used only as soft constraint anchor "
            "‖(u_base+e_hat)-u*‖_GT; primary target is L[e_hat]=−residual "
            "(residual=L[u_base]−f). Never trained e_hat ← (u*-u_base) as sole "
            "regression target."
        ),
    }


def train_arm_rbf_grad(
    data: ExperimentData,
    steps: int = 3000,
    lr: float = 2e-6,
    lr_centers: float | None = None,
    lambda_bc: float = 1e4,
    eval_every: int = 50,
    colloc_batch: int | None = None,
) -> dict:
    """Gradient-refined RBF (classical residual-driven center/weight refine).

    Seeds from the shared Kansa base (centers + weights, ε fixed) and Adam-
    refines centers + O(1) normalized weights against PDE residual + soft
    Dirichlet BC — same step budget as the neural arms. Faithful to splat
    `rbf-grad` / `fit_params`: all RBF params (centers + weights) are
    gradient-refined; ε and the frozen weight scale σ stay frozen.

    Reparameterization
    ------------------
    Kansa weights are O(10^6). Training raw w under Adam at ~1e-2 diverges;
    lr=1e-8 on raw w freezes params (vacuous). Instead: σ = max(|w_base|)
    (constant), w̃ = w_base / σ (O(1) trainable), u(x) = Σ_j (σ · w̃_j) φ_j(x).
    Init field == base.

    Adam step sizes are ~lr in *parameter* space, so an unscaled lr on w̃
    still yields Δw_eff ≈ σ·lr. Weight group therefore uses lr_w = lr / σ
    (CFG `lr` = effective |Δw| Adam scale). Centers use `lr_centers`.
    Empirically 1e-2 / 5e-3 / 1e-3 on w̃ all diverge; largest stable
    effective weight lr on this toy is ~2e-6 (lr_centers ~1e-8).

    Notes
    -----
    - float64 required for inheritance fidelity.
    - Analytical Δ (RBFField.laplacian) matches Kansa; not nested autograd.
    - Full interior collocation (no mini-batch). λ_bc=1e4 matches Kansa BC weight.
    """
    set_seeds(data.seed)
    model = RBFField(data.rbf_centers, data.u_base_weights, data.epsilon)
    sigma = float(model.weight_scale.item())
    if lr_centers is None:
        lr_centers = 1e-8
    # Plain Adam (default β1/β2), matching splat fit_params — not AdamW.
    # Scale weight lr by 1/σ so CFG `lr` is the effective |Δw| step scale.
    opt = optim.Adam(
        [
            {"params": [model.weights], "lr": lr / max(sigma, 1e-30)},
            {"params": [model.centers], "lr": lr_centers},
        ]
    )

    # Snapshot init for displacement checks (effective w = σ · w̃).
    with torch.no_grad():
        centers0 = model.centers.detach().clone()
        w_eff0 = model.effective_weights.detach().clone()
        w_norm0 = model.weights.detach().clone()
        rel_l2_init = rel_l2(_eval_model(model, data.grid), data.u_exact)

    # Full-batch collocation (see docstring); ignore colloc_batch for stability.
    x_int = torch.tensor(data.interior, dtype=torch.float64)
    f_int = torch.tensor(data.f_interior, dtype=torch.float64).unsqueeze(-1)
    x_bnd = torch.tensor(data.boundary_pts, dtype=torch.float64)
    u_bnd = torch.tensor(data.u_bnd, dtype=torch.float64).unsqueeze(-1)
    base_rel = rel_l2(data.u_base, data.u_exact)

    hist = {
        "loss": [],
        "loss_phys": [],
        "loss_bc": [],
        "rel_l2": [],
        "grad_cos": [],
        "step": [],
        "step_time": [],
    }

    model.train()
    loss0_phys = None
    diverged = False
    for step in range(steps):
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        lap = model.laplacian(x_int)
        phys = ((-lap - f_int) ** 2).mean()
        bc = ((model(x_bnd) - u_bnd) ** 2).mean()
        loss = phys + lambda_bc * bc
        if not torch.isfinite(loss):
            diverged = True
            hist["loss"].append(float("nan"))
            hist["loss_phys"].append(float("nan"))
            hist["loss_bc"].append(float("nan"))
            hist["grad_cos"].append(float("nan"))
            hist["step"].append(step)
            hist["step_time"].append(time.perf_counter() - t0)
            hist["rel_l2"].append(float("nan"))
            print(
                f"    [rbf-grad] non-finite loss at step {step}; aborting refine "
                f"(lr_eff={lr:g}, lr_c={lr_centers:g})",
                flush=True,
            )
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.centers.clamp_(0.0, 1.0)
        dt = time.perf_counter() - t0

        if step == 0:
            loss0_phys = float(phys.item())

        hist["loss"].append(float(loss.item()))
        hist["loss_phys"].append(float(phys.item()))
        hist["loss_bc"].append(float(bc.item()))
        # Single residual-based loss family; no phys-vs-data conflict column.
        hist["grad_cos"].append(float("nan"))
        hist["step"].append(step)
        hist["step_time"].append(dt)

        if step % eval_every == 0 or step == steps - 1:
            hist["rel_l2"].append(rel_l2(_eval_model(model, data.grid), data.u_exact))
            model.train()
        else:
            hist["rel_l2"].append(hist["rel_l2"][-1] if hist["rel_l2"] else float("nan"))

    with torch.no_grad():
        mean_w_disp = float(
            torch.mean(torch.abs(model.effective_weights - w_eff0)).item()
        )
        mean_w_norm_disp = float(
            torch.mean(torch.abs(model.weights - w_norm0)).item()
        )
        mean_c_disp = float(
            torch.mean(torch.linalg.norm(model.centers - centers0, dim=-1)).item()
        )

    u_final = _eval_model(model, data.grid)
    final_rel = rel_l2(u_final, data.u_exact)
    loss_final_phys = float(hist["loss_phys"][-1]) if hist["loss_phys"] else float("nan")
    delta = base_rel - final_rel
    beats_base = (not diverged) and final_rel < base_rel
    resid_decreased = (
        loss0_phys is not None
        and np.isfinite(loss_final_phys)
        and loss_final_phys < loss0_phys
    )
    # Non-vacuous: params moved and residual responded (not frozen at 1e-8-raw).
    trained = (mean_w_disp > 1e-10 or mean_c_disp > 1e-10) and resid_decreased
    lr_w_norm = lr / max(sigma, 1e-30)
    print(
        f"    [rbf-grad train-moved] mean|Δw|={mean_w_disp:.4e} "
        f"mean|Δw̃|={mean_w_norm_disp:.4e} mean|Δc|={mean_c_disp:.4e} "
        f"(non_vacuous={trained}); σ={sigma:.4e}; "
        f"lr_eff={lr:g} lr_w̃={lr_w_norm:.4e} lr_c={lr_centers:g}",
        flush=True,
    )
    print(
        f"    [rbf-grad sanity] phys-loss {loss0_phys:.4e} → {loss_final_phys:.4e} "
        f"(decreased={resid_decreased}); "
        f"relL2 before={rel_l2_init:.4e} (==base {base_rel:.4e}) → after={final_rel:.4e} "
        f"(Δ={delta:+.4e}, beats_base={beats_base}); diverged={diverged}",
        flush=True,
    )
    return {
        "arm": "rbf-grad",
        "u_pred": u_final,
        "history": hist,
        "final_rel_l2": final_rel,
        "base_rel_l2": base_rel,
        "rel_l2_init": float(rel_l2_init),
        "rel_l2_delta_vs_base": float(delta),
        "beats_base": bool(beats_base),
        "phys_loss_initial": float(loss0_phys) if loss0_phys is not None else float("nan"),
        "phys_loss_final": loss_final_phys,
        "phys_loss_decreased": bool(resid_decreased),
        "mean_weight_displacement": mean_w_disp,
        "mean_weight_norm_displacement": mean_w_norm_disp,
        "mean_center_displacement": mean_c_disp,
        "weight_scale": sigma,
        "trained_non_vacuous": bool(trained),
        "diverged": bool(diverged),
        "mean_step_time": float(np.nanmean(hist["step_time"]))
        if hist["step_time"]
        else float("nan"),
        "mean_grad_cos": float("nan"),
        "late_grad_cos": float("nan"),
        "lr": float(lr),
        "lr_weight_norm": float(lr_w_norm),
        "lr_centers": float(lr_centers),
        "lambda_bc": float(lambda_bc),
        "model": model,
        "centers_final": model.centers.detach().cpu().numpy(),
        "weights_final": model.effective_weights.detach().cpu().numpy(),
    }
