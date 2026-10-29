"""The ConnectionMiner solver (PyTorch, float64).

Model:  C ~ C_hat = P G beta G^T P^T

    C     (types x types)        binarised connectome, fitted where W = C_mask & observed
    P     (types x metacells)    soft assignment of metacells to types, zero outside the
                                 expression constraints D, rows and columns normalised
    G     (metacells x genes)    detection rate of each solver gene in each metacell
    beta  (genes x genes)        non-negative gene-gene interaction

Alternating minimisation for `num_iter` outer iterations:
  1. beta step: multiplicative non-negative least squares with an L1 penalty
     (`lambda_sparsity`), at most `regression_iterations` updates.
  2. P step: mirror descent on P with an entropy term (`optimal_transport_epsilon`)
     and Sinkhorn-style row/column projection onto D, once against C's rows and
     once against its columns.

Held-out blocks (`freeze_beta`): beta is loaded from the train solve and held
fixed up to one non-negative scalar, refit in closed form each iteration
(`fit_beta_scale`); only P is optimised.

float64 is required: beta's entries span down to ~1e-13, and float32 collapses
the train fit (r 0.67 -> 0.10).

The CUDA device is used when available, otherwise CPU. The two draw different
random streams for beta's initialisation, so the same seed gives a different
(equally valid) solve on CPU and GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

FLOAT = torch.float64
EPS = torch.finfo(FLOAT).eps


@dataclass
class SolveResult:
    P: np.ndarray
    beta: np.ndarray
    C_hat: np.ndarray
    loss: dict[str, np.ndarray]
    elapsed_sec: float


def solve(G_metacell: np.ndarray, P_constraints: np.ndarray, C: np.ndarray, C_mask: np.ndarray,
          cfg: dict, seed: int, beta_fixed: np.ndarray | None = None,
          device: str = "auto", log=print) -> SolveResult:
    """Run one solve. `beta_fixed` switches to the held-out (frozen beta) protocol."""
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    torch.manual_seed(seed)
    t0 = time.time()

    def _to(x):
        return torch.as_tensor(x, dtype=FLOAT, device=dev)

    G = _to(G_metacell)
    if cfg.get("use_complement", False):
        # Each gene also enters as its non-detection 1 - G, so absence of a
        # gene can carry wiring signal (motor: a blank sheet cell is "not reported").
        G = torch.cat([G, torch.clamp(1.0 - G, 0.0, 1.0)], dim=1)
    ng = G.shape[1]

    # Binary connectome. NaN compares False, so an unobserved entry inside the
    # mask is fitted as a 0 (the scorer, by contrast, skips NaN entries).
    C_np = (np.asarray(C, dtype=float) > 0).astype(float)
    W_np = (np.asarray(C_mask, dtype=float) > 0).astype(float)
    C_np[W_np == 0] = 0.0
    C_t, W = _to(C_np), _to(W_np)

    D = _to(np.asarray(P_constraints, dtype=float))
    with torch.no_grad():
        P = torch.nan_to_num(D / torch.clamp(D.sum(dim=1, keepdim=True), min=1e-16), 0.0)

    freeze = beta_fixed is not None
    if freeze:
        if beta_fixed.shape != (ng, ng):
            raise ValueError(f"fixed beta is {beta_fixed.shape}, expected {(ng, ng)}: "
                             "train and held-out solves must share the gene set")
        beta = _to(np.asarray(beta_fixed, dtype=float))
        beta0 = beta.clone()
        log("  beta FROZEN (from train) -- only P is optimised (+ scalar rescale)")
    else:
        beta = torch.rand(ng, ng, dtype=FLOAT, device=dev)
    beta_max = torch.full((ng, ng), float("inf"), dtype=FLOAT, device=dev)

    num_iter = int(cfg["num_iter"])
    lamb = float(cfg["lambda_sparsity"])
    epsilon = float(cfg["optimal_transport_epsilon"])
    step = float(cfg["optimal_transport_step"])
    ot_iter = int(cfg["optimal_transport_iterations"])
    reg_iter = int(cfg["regression_iterations"])
    tlim = float(cfg.get("time_limit_per_step", 60.0))

    obj_beta, obj_fit, obj_ent = (np.zeros(num_iter) for _ in range(3))
    log(f"  solver on {dev}: {num_iter} iterations, {P.shape[0]} types x {P.shape[1]} metacells, "
        f"{ng} gene features")

    for it in range(num_iter):
        t_it = time.time()
        with torch.no_grad():
            PG = P @ G
            if freeze:
                R = PG @ (beta0 @ PG.T)
                Wsq = W ** 2
                alpha = torch.sum(Wsq * R * C_t) / torch.clamp(torch.sum(Wsq * R * R), min=EPS)
                beta = torch.clamp(alpha, min=0.0) * beta0
                obj_beta[it] = _beta_obj(PG, PG.T, C_t, W ** 2, beta)
            else:
                beta, obj_beta[it] = _beta_update(PG, PG.T, C_t, W, beta, beta_max, lamb,
                                                  reg_iter, tlim, dev)

            Z = G @ beta @ G.T
            P = _P_update(P, Z, C_t, W, D, epsilon, step, ot_iter, tlim)

            recon = P @ G @ beta @ G.T @ P.T
            obj_fit[it] = float(torch.sum((W * (recon - C_t)) ** 2).item())
            pos = P > 0
            obj_ent[it] = float((epsilon * torch.sum(P[pos] * torch.log(P[pos]))).item()) if pos.any() else 0.0
        log(f"  iter {it + 1}/{num_iter}: obj_beta={obj_beta[it]:.6e} obj_P_fit={obj_fit[it]:.6e} "
            f"obj_P_ent={obj_ent[it]:.6e} ({time.time() - t_it:.1f}s)")

    with torch.no_grad():
        C_hat = P @ G @ beta @ G.T @ P.T
    loss = {"obj_beta": obj_beta, "obj_P_fit": obj_fit, "obj_P_ent": obj_ent,
            "total_loss": obj_fit + obj_ent}
    return SolveResult(P=P.cpu().numpy(), beta=beta.cpu().numpy(), C_hat=C_hat.cpu().numpy(),
                       loss=loss, elapsed_sec=time.time() - t0)


def _beta_update(A, B, C, W, beta, beta_max, lamb, max_iter, time_limit, dev):
    tol = 1e-6
    t0 = time.time()
    Wsq = W ** 2
    Numer = A.T @ (Wsq * C) @ B.T
    zero = torch.tensor(0.0, device=dev, dtype=FLOAT)
    beta = torch.clamp(beta, zero, beta_max)
    prev = _beta_obj(A, B, C, Wsq, beta)
    for _ in range(max_iter):
        Denom = A.T @ (Wsq * (A @ (beta @ B))) @ B.T
        beta = torch.clamp(beta * (Numer / (Denom + lamb)), zero, beta_max)
        curr = _beta_obj(A, B, C, Wsq, beta)
        if abs(curr - prev) / (prev + EPS) < tol or (time.time() - t0) > time_limit:
            break
        prev = curr
    # tiny positive jitter keeps zeroed entries able to grow again under the multiplicative update
    beta = beta + 100 * EPS * torch.rand(*beta.shape, dtype=FLOAT, device=dev)
    return beta, _beta_obj(A, B, C, Wsq, beta)


def _beta_obj(A, B, C, Wsq, X):
    return float(torch.sum(Wsq * ((A @ (X @ B) - C) ** 2)).item())


def _P_update(P, Z, C, W, D, epsilon, step, max_iter, time_limit):
    D_norm = torch.nan_to_num(D / torch.clamp(D.sum(dim=1, keepdim=True), min=1e-16), 0.0)
    row_c, col_c = D_norm.sum(dim=1), D_norm.sum(dim=0)
    # Two passes, against C's rows then its columns; the second sees the first's P.
    # Types whose row (column) of C is all zero keep their P row in that pass.
    for transpose in (False, True):
        A = (Z.T if transpose else Z) @ P.T
        B, Wd = (C.T, W.T) if transpose else (C, W)
        unfixed = ~torch.all(B == 0, dim=1)
        P_new = _entropic_mirror_descent(A, B, Wd, row_c, col_c, D, epsilon, step, max_iter,
                                         time_limit, P)
        P[unfixed, :] = P_new[unfixed, :]
    return P


def _entropic_mirror_descent(A, B, W, a, b, D, epsilon, step_size, max_iter, time_limit, P0):
    row_col_iters, max_backtrack, tol = 10, 20, 1e-6
    P = _normalize_clip(P0.clone(), a, b, D, row_col_iters)
    W2 = W ** 2
    prev = _ot_obj(P, A, B, W2, epsilon)
    t0 = time.time()
    for _ in range(max_iter):
        grad = 2.0 * ((W2 * (P @ A - B)) @ A.T) + epsilon * (1.0 + torch.log(torch.clamp(P, min=1e-30)))
        trial = step_size
        for bt in range(max_backtrack + 1):
            P_trial = torch.clamp(P * torch.exp(-trial * grad), torch.zeros_like(D), D)
            P_trial = _normalize_clip(P_trial, a, b, D, row_col_iters)
            new = _ot_obj(P_trial, A, B, W2, epsilon)
            if new <= prev or bt >= max_backtrack:
                break
            trial /= 2.0
        P = P_trial
        if abs(new - prev) / max(1.0, abs(prev)) < tol:
            P = _normalize_clip(P, a, b, D, 1000)
            break
        if (time.time() - t0) > time_limit:
            break
        prev = new
    return P


def _ot_obj(P, A, B, W2, epsilon):
    val = float(torch.sum(W2 * ((P @ A - B) ** 2)).item())
    pos = P > 0
    ent = float(torch.sum(P[pos] * torch.log(P[pos])).item()) if pos.any() else 0.0
    return val + epsilon * ent


def _normalize_clip(P, a, b, D, passes):
    zero = torch.zeros_like(D)
    for _ in range(passes):
        P = torch.clamp(P * (a[:, None] / torch.clamp(P.sum(dim=1, keepdim=True), min=1e-16)), zero, D)
        P = torch.clamp(P * (b[None, :] / torch.clamp(P.sum(dim=0, keepdim=True), min=1e-16)), zero, D)
    return P
