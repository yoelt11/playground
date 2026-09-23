"""Training loops for the three Q1 arms."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import SolutionNet, flat_grads, grad_cosine, laplacian
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
      - Primary training target is L[e_hat] = residual (= −Δu_base − f).
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
        pde_loss = ((-lap_e - residual) ** 2).mean()
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
        "oracle_leak_audit": (
            "GT used only as soft constraint anchor "
            "‖(u_base+e_hat)-u*‖_GT; primary target is L[e_hat]=residual. "
            "Never trained e_hat ← (u*-u_base) as sole regression target."
        ),
    }
