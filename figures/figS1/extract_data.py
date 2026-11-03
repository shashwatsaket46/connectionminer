#!/usr/bin/env python3
"""KIT NOTE: in this repository the output depth_stats.npz is stored in data/ (cluster-only provenance script)."""
"""Supplementary Fig S1 data: read depth of single cells vs metacells for the three best builds.

Per unit (cell or metacell): UMIs = total counts, genes detected = number of genes with count > 0.
Metacell profile = sum of the raw counts of its member cells (all genes of the count matrix, not only HVGs).

  visual : devstages/_h5ad/Adult.h5ad (X = raw counts, 12,028 genes), cells with MultiomeAnnotated "G_*" dropped
           (glia) -> 99,656 cells; partition devstages/Adult/grid/hvg3000/3_dispersion_binned__1_constrained_kmeans_local_pca
  CX     : cm_central_complex/data/atlas/counts.mtx (genes x cells, 21,771 genes), cells of
           grid/runs/g5_h2000_bn1_m1/cell_index.csv (48,014) matched by barcode
  motor  : cm_motor/output_tp3/G_cells.npz (11,547 shared genes; the source tables are already "cg_corrected",
           ~1/3 of entries non-integer -> no raw integer counts exist for motor)

Sparse throughout. Output: figS1/depth_stats.npz (per-unit vectors) + depth_summary.csv.
Usage: OMP_NUM_THREADS=3 python biorender_fig/figS1/s1_depth_stats.py
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
S = Path("/scratch/ss20308")


def agg(X, mc):
    """X cells x genes (csr), mc metacell index per cell -> per-cell and per-metacell (UMIs, genes detected)."""
    X = X.tocsr().astype(np.float32)
    n_mc = mc.max() + 1
    A = sp.csr_matrix((np.ones(len(mc), np.float32), (mc, np.arange(len(mc)))), shape=(n_mc, len(mc)))
    M = (A @ X).tocsr()
    M.eliminate_zeros(); X.eliminate_zeros()
    out = dict(cell_umi=np.asarray(X.sum(1)).ravel(), cell_genes=np.diff(X.indptr),
               mc_umi=np.asarray(M.sum(1)).ravel(), mc_genes=np.diff(M.indptr),
               mc_size=np.bincount(mc, minlength=n_mc), n_genes=np.array(X.shape[1]))
    return out


def visual():
    f = h5py.File(S / "cm_active/devstages/_h5ad/Adult.h5ad", "r")
    o = f["obs/MultiomeAnnotated"]
    lab = o["categories"][:].astype(str)[o["codes"][:]]
    keep = ~np.char.startswith(lab, "G_")
    g = f["X"]
    shape = tuple(g.attrs["shape"])
    X = sp.csc_matrix((g["data"][:].astype(np.float32), g["indices"][:], g["indptr"][:]), shape=shape)
    X = X.tocsr()[np.flatnonzero(keep)]
    p = pd.read_csv(S / "cm_active/devstages/Adult/grid/hvg3000/"
                    "3_dispersion_binned__1_constrained_kmeans_local_pca/cell_to_metacell.csv")
    assert len(p) == X.shape[0] and (p.cell_idx.values == np.arange(len(p))).all()
    return agg(X, p.metacell_idx.values)


def cx():
    a = S / "cm_central_complex/data/atlas"
    with open(a / "counts.mtx") as fh:
        fh.readline()
        ng, nc, nnz = map(int, fh.readline().split())
    t = pd.read_csv(a / "counts.mtx", sep=" ", skiprows=2, header=None, dtype=np.int32, engine="c").values
    X = sp.csr_matrix((t[:, 2].astype(np.float32), (t[:, 1] - 1, t[:, 0] - 1)), shape=(nc, ng))   # cells x genes
    del t
    bc = pd.read_csv(a / "cells.csv", usecols=["barcode"]).barcode.values
    run = S / "cm_central_complex/grid/runs/g5_h2000_bn1_m1"
    ci = pd.read_csv(run / "cell_index.csv", usecols=["col_idx", "cell_barcode"])
    pos = pd.Series(np.arange(len(bc)), index=bc).loc[ci.cell_barcode].values
    X = X[pos]
    c2m = pd.read_csv(run / "cell_to_metacell.csv")
    mi = pd.read_csv(run / "metacell_index.csv").set_index("metacell_id").metacell_idx
    assert (c2m.cell_idx.values == ci.col_idx.values).all()
    return agg(X, mi.loc[c2m.metacell_id].values)


def motor():
    d = S / "cm_motor/output_tp3"
    z = np.load(d / "G_cells.npz")
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    c2m = pd.read_csv(d / "cell_to_metacell.csv")
    mi = pd.read_csv(d / "metacell_index.csv").set_index("metacell_id").metacell_idx
    assert len(c2m) == X.shape[0] and (c2m.cell_idx.values == np.arange(len(c2m))).all()
    return agg(X, mi.loc[c2m.metacell_id].values)


if __name__ == "__main__":
    res, rows = {}, []
    for name, fn in (("visual", visual), ("cx", cx), ("motor", motor)):
        r = fn()
        for k, v in r.items():
            res[f"{name}__{k}"] = v
        rows.append(dict(system=name, n_cells=len(r["cell_umi"]), n_metacells=len(r["mc_umi"]),
                         n_genes_matrix=int(r["n_genes"]),
                         median_cells_per_mc=float(np.median(r["mc_size"])),
                         cell_umi_median=float(np.median(r["cell_umi"])), mc_umi_median=float(np.median(r["mc_umi"])),
                         cell_genes_median=float(np.median(r["cell_genes"])),
                         mc_genes_median=float(np.median(r["mc_genes"]))))
        print(rows[-1], flush=True)
    np.savez_compressed(HERE / "depth_stats.npz", **res)
    pd.DataFrame(rows).to_csv(HERE / "depth_summary.csv", index=False)
