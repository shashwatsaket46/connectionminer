"""Type-axis k-fold split and the per-block solver inputs.

The held-out unit is a cell TYPE. Types are dealt round-robin into k folds
within each stratum (named / unnamed, plus family for motor), after one
permutation per stratum drawn from `default_rng(seed)`. Fold f is the test
block, fold f+1 (mod k) the validation block, the other k-2 folds train.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .data import Bundle


def kfold_types(strata: np.ndarray, k: int, fold: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = np.empty(len(strata), dtype=int)
    for s in np.unique(strata):          # sorted, so the RNG stream is fixed by the labels
        idx = rng.permutation(np.where(strata == s)[0])
        f[idx] = np.arange(len(idx)) % k
    te, va = fold % k, (fold + 1) % k
    return {"train": np.where((f != te) & (f != va))[0],
            "val": np.where(f == va)[0],
            "test": np.where(f == te)[0]}


@dataclass
class Problem:
    """What the solver sees for one block of types."""
    types: np.ndarray                 # indices into the bundle's type axis
    cells: np.ndarray                 # indices into the bundle's cell axis
    C: np.ndarray
    C_mask: np.ndarray
    P_constraints_metacell: np.ndarray   # types x metacells, 0/1
    G_metacell: np.ndarray               # metacells x solver genes, detection rate in [0, 1]
    gene_idx: np.ndarray                 # solver genes, as indices into the bundle's genes
    genes: np.ndarray
    cell_to_metacell: np.ndarray
    meta_sizes: np.ndarray


def build_problem(b: Bundle, types: np.ndarray, genes_cfg: dict, metacell_cfg: dict,
                  gene_idx: np.ndarray | None = None) -> Problem:
    """Cut the bundle to `types` and aggregate cells into metacell detection rates.

    Cells whose candidate set is emptied by the cut are dropped. The surviving
    metacells are renumbered densely in their original order.

    `gene_idx` fixes the solver genes (val/test reuse train's, so beta's axes
    line up). Without it they are chosen here by `genes_cfg["select"]`:
      all              every bundle gene, in bundle order
      binary_variance  genes detected in >= min_cells of THIS block's cells,
                       ranked by mu*(1-mu), top n_top
    """
    types = np.asarray(types, dtype=int)
    P = b.P_cells[types, :].tocsc()
    cells = np.where(np.diff(P.indptr) > 0)[0]
    P = P[:, cells].tocsr()

    X_bin = (b.X[cells].toarray() > 0).astype(float)
    if gene_idx is None:
        gene_idx = select_genes(X_bin, genes_cfg)
    gene_idx = np.asarray(gene_idx, dtype=int)
    # Column fancy-indexing returns a Fortran-ordered copy, which changes the
    # summation order below (~1e-13 in G). Skip it when every gene is used, as
    # the reference runs did, so G is bit-identical to theirs.
    if np.array_equal(gene_idx, np.arange(X_bin.shape[1])):
        X_feat = X_bin
    else:
        X_feat = X_bin[:, gene_idx]

    # Batch-normalised detection (CX). Each cell's 0/1 detection of gene g is
    # weighted by p_g / p_gb, the global rate over its batch's rate, with the
    # batch rate shrunk toward global by `batch_pseudo` pseudo-cells and the
    # weight capped to [1/cap, cap]. Rates are over this block's cells.
    batch_cfg = genes_cfg.get("batch_normalize")
    if batch_cfg:
        batch = b.batch[cells]
        a_ps, cap = float(batch_cfg.get("pseudo", 10.0)), float(batch_cfg.get("cap", 5.0))
        p_g = X_feat.mean(axis=0)
        weights = {}
        for lev in np.unique(batch):
            m = batch == lev
            p_gb = (X_feat[m].sum(axis=0) + a_ps * p_g) / (m.sum() + a_ps)
            w = np.where(p_gb > 0, p_g / np.maximum(p_gb, 1e-12), 1.0)
            weights[lev] = np.clip(w, 1.0 / cap, cap)
        X_feat = X_feat.astype(np.float64, copy=True)
        for lev, w in weights.items():
            X_feat[batch == lev] *= w

    old = b.metacell[cells]
    uniq = np.unique(old)
    c2m = np.searchsorted(uniq, old)
    K = len(uniq)
    meta_sizes = np.bincount(c2m, minlength=K).astype(float)
    M = sparse.csr_matrix((np.ones(c2m.size), (np.arange(c2m.size), c2m)), shape=(c2m.size, K))

    P_constraints_metacell = ((P @ M) > 0).astype(float).toarray()
    G = (M.T @ X_feat) / np.maximum(meta_sizes[:, None], 1.0)
    if batch_cfg:
        G = np.clip(np.asarray(G), 0.0, 1.0)

    # Metacells smaller than `min_samples_prior` are shrunk toward the block's
    # mean detection, as if padded with average cells up to that size.
    prior = int(metacell_cfg.get("min_samples_prior", 0) or 0)
    if prior > 0:
        p_global = np.mean(X_feat, axis=0)
        for k in np.where(meta_sizes < prior)[0]:
            n_k = meta_sizes[k]
            G[k, :] = (n_k * G[k, :] + (prior - n_k) * p_global) / prior

    return Problem(types=types, cells=cells, C=b.C[np.ix_(types, types)],
                   C_mask=b.C_mask[np.ix_(types, types)],
                   P_constraints_metacell=P_constraints_metacell, G_metacell=np.asarray(G),
                   gene_idx=gene_idx, genes=b.genes[gene_idx], cell_to_metacell=c2m,
                   meta_sizes=meta_sizes)


def select_genes(X_bin: np.ndarray, genes_cfg: dict) -> np.ndarray:
    method = genes_cfg.get("select", "all")
    if method == "all":
        return np.arange(X_bin.shape[1])
    if method == "binary_variance":
        n_top = int(genes_cfg["n_top"])
        min_cells = max(int(genes_cfg.get("min_cells", 5)), 1)
        cand = np.where(X_bin.sum(axis=0) >= min_cells)[0]
        if cand.size == 0:
            cand = np.arange(X_bin.shape[1])
        mu = X_bin[:, cand].mean(axis=0)
        order = np.argsort(-(mu * (1.0 - mu)))
        return cand[order[:min(n_top, cand.size)]]
    raise ValueError(f"unknown gene selection {method!r}")
