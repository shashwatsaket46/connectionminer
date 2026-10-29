#!/usr/bin/env python3
"""Export a dataset bundle (see connectionminer/data.py) from the lab's build outputs.

This is the bridge from the original build pipelines (one repo per system) to
the solver-ready bundles this repository runs on. It reads the exact inputs the
paper's Fig 1 runs read, so those runs are reproducible from the bundles.

    python prepare/export_bundle.py visual --out data/visual
    python prepare/export_bundle.py motor  --out data/motor
    python prepare/export_bundle.py cx     --out data/cx

`--src` points at the build directories; defaults are their NYU HPC locations.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from connectionminer.data import save_bundle  # noqa: E402

SCRATCH = Path("/scratch/ss20308")
DEFAULT_SRC = {
    "visual": SCRATCH / "cm_active",
    "motor": SCRATCH / "cm_motor",
    "cx": SCRATCH / "cm_central_complex",
}


def named_type_mask(P_cells: sparse.spmatrix) -> np.ndarray:
    """A type is 'named' iff at least one cell is pinned to it alone."""
    Pc = sparse.csc_matrix(P_cells)
    pinned = np.where(np.diff(Pc.indptr) == 1)[0]
    return np.asarray(Pc[:, pinned].sum(axis=1)).ravel() > 0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c2m_from_ids(d: Path) -> np.ndarray:
    mi = pd.read_csv(d / "metacell_index.csv")
    c2m = pd.read_csv(d / "cell_to_metacell.csv")
    return c2m["metacell_id"].map(dict(zip(mi.metacell_id, mi.metacell_idx))).to_numpy(int)


def export_visual(src: Path, out: Path) -> None:
    """Adult visual: build at devstages/Adult/output, 3,000 dispersion_binned HVGs and
    metacells from the HVG grid, expression from the Adult h5ad."""
    import anndata as ad

    build = src / "devstages/Adult/output"
    grid = src / "devstages/Adult/grid/hvg3000/3_dispersion_binned__1_constrained_kmeans_local_pca"
    h5ad = src / "devstages/_h5ad/Adult.h5ad"

    Pc = sparse.load_npz(build / "P_constraints_cells.npz").tocsr()
    ti, ci = pd.read_csv(build / "type_index.csv"), pd.read_csv(build / "cell_index.csv")
    # Folds are stratified on the production adult build's named mask (as the Fig 1 runs were).
    named = named_type_mask(sparse.load_npz(src / "output/P_constraints_cells.npz"))

    adata = ad.read_h5ad(h5ad, backed="r")
    hvg = pd.read_csv(grid / "gene_index.csv")["gene_name"].astype(str).to_numpy()
    gidx = np.where(np.isin(adata.var_names.values, hvg))[0]       # h5ad column order, as the solver read it
    adata = adata[ci["cell_barcode"].astype(str).to_numpy()]
    X = adata[:, gidx].X
    X = sparse.csr_matrix(X if sparse.issparse(X) else np.asarray(X))
    c2m = (pd.read_csv(grid / "cell_to_metacell.csv").sort_values("cell_idx").metacell_idx.to_numpy())

    save_bundle(out, "visual", sparse.load_npz(build / "C_matrix.npz").toarray(), np.load(build / "C_mask.npy"),
                Pc, X, adata.var_names.values[gidx],
                pd.DataFrame({"type": ti["type"], "stratum": np.where(named, "named", "unnamed")}),
                pd.DataFrame({"cell_id": ci["cell_barcode"], "metacell": c2m}),
                {"build": str(build), "grid": str(grid), "h5ad": str(h5ad), "h5ad_sha256": sha256(h5ad),
                 "gene_selection": "3_dispersion_binned top 3000 (h5ad column order)",
                 "metacells": "1_constrained_kmeans_local_pca"})


def export_motor(src: Path, out: Path) -> None:
    """Adult motor: build output_tp3 (MN timepoint 3); solver genes = top 1,000
    3_dispersion_binned, as recorded in the Fig 1 runs' gene_index.csv."""
    build = src / "output_tp3"
    ref = SCRATCH / "cm_active/devstages/motor/tp3/typesplit_k10_f0/h1000_uniform_s0/rep0_train/gene_index.csv"
    gi = pd.read_csv(ref)
    assert gi.gene_name.tolist() == pd.read_csv(build / "gene_index.csv").gene_name.tolist()
    sel = np.where(gi["in_solver_set"].to_numpy(bool))[0]
    assert sel.size == 1000

    Pc = sparse.load_npz(build / "P_constraints_cells.npz").tocsr()
    ti, ci = pd.read_csv(build / "type_index.csv"), pd.read_csv(build / "cell_index.csv")
    named = named_type_mask(Pc)
    X = sparse.load_npz(build / "G_cells.npz").tocsr()[:, sel]

    save_bundle(out, "motor", sparse.load_npz(build / "C_matrix.npz").toarray(), np.load(build / "C_mask.npy"),
                Pc, X, gi.gene_name.to_numpy(str)[sel],
                pd.DataFrame({"type": ti["type"], "stratum": np.char.add(np.where(named, "named", "unnamed"),
                                                                         "_" + ti["family"].astype(str).to_numpy())}),
                pd.DataFrame({"cell_id": ci.get("cell_barcode", ci.index), "metacell": c2m_from_ids(build),
                              "source": ci["source"]}),
                {"build": str(build), "gene_index": str(ref),
                 "gene_selection": "3_dispersion_binned top 1000 on all tp3 cells",
                 "G_cells_sha256": sha256(build / "G_cells.npz")})


def export_cx(src: Path, out: Path) -> None:
    """Central complex: grid build g5_h2000_bn1_m1 rebuilt without the gene
    min-cells filter (grid_mc1); solver genes = its in_solver_set."""
    build = src / "grid_mc1/runs/g5_h2000_bn1_m1"
    gi = pd.read_csv(build / "gene_index.csv")
    sel = np.where(gi["in_solver_set"].to_numpy(bool))[0]
    assert sel.size == 2000

    Pc = sparse.load_npz(build / "P_constraints_cells.npz").tocsr()
    ti, ci = pd.read_csv(build / "type_index.csv"), pd.read_csv(build / "cell_index.csv")
    named = named_type_mask(Pc)
    X = sparse.load_npz(build / "G_cells.npz").tocsr()[:, sel]

    save_bundle(out, "cx", sparse.load_npz(build / "C_matrix.npz").toarray(), np.load(build / "C_mask.npy"),
                Pc, X, gi.gene_name.to_numpy(str)[sel],
                pd.DataFrame({"type": ti["type"], "stratum": np.where(named, "named", "unnamed")}),
                pd.DataFrame({"cell_id": ci.get("cell_barcode", ci.index), "metacell": c2m_from_ids(build),
                              "batch": ci["source"].astype(str)}),
                {"build": str(build.resolve()),
                 "gene_selection": "5_dispersion_binned_batch top 2000, gene min_cells 1",
                 "metacells": "1_constrained_kmeans_local_pca (grid/runs/g5_h2000_bn1_m1)",
                 "G_cells_sha256": sha256(build / "G_cells.npz")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["visual", "motor", "cx"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="root of that system's build repo")
    a = ap.parse_args()
    src = Path(a.src) if a.src else DEFAULT_SRC[a.dataset]
    {"visual": export_visual, "motor": export_motor, "cx": export_cx}[a.dataset](src, Path(a.out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
