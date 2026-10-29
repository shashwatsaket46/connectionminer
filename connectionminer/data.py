"""Dataset bundles: the solver-ready form of one system.

A bundle is a directory holding everything a type-split solve reads, already
aligned on one cell axis and one type axis:

    bundle.json              dataset name, counts, provenance
    C.npz                    (types x types) connectome, sparse; NaN = unobserved
    C_mask.npy               (types x types) 1 = entry is fitted and scored
    P_constraints_cells.npz  (types x cells) 1 = cell may be assigned to type
    X_cells.npz              (cells x genes) binary detection, CSR uint8
    genes.csv                gene_name, one row per column of X_cells
    types.csv                type, stratum   (stratum drives the fold split)
    cells.csv                cell_id, metacell[, batch]

`prepare/export_bundle.py` writes bundles from the lab's build outputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Bundle:
    name: str
    C: np.ndarray                     # float, NaN where unobserved
    C_mask: np.ndarray
    P_cells: sparse.csr_matrix        # types x cells
    X: sparse.csr_matrix              # cells x genes, 0/1
    genes: np.ndarray
    types: pd.DataFrame
    cells: pd.DataFrame

    @property
    def strata(self) -> np.ndarray:
        return self.types["stratum"].to_numpy(dtype=str)

    @property
    def metacell(self) -> np.ndarray:
        return self.cells["metacell"].to_numpy(dtype=int)

    @property
    def batch(self) -> np.ndarray | None:
        return self.cells["batch"].astype(str).to_numpy() if "batch" in self.cells else None


def load_bundle(path: str | Path) -> Bundle:
    d = Path(path)
    if not (d / "bundle.json").exists():
        raise FileNotFoundError(f"{d} is not a dataset bundle (no bundle.json); see README > Data")
    meta = json.loads((d / "bundle.json").read_text())
    b = Bundle(
        name=meta["dataset"],
        C=sparse.load_npz(d / "C.npz").toarray().astype(float),
        C_mask=np.load(d / "C_mask.npy"),
        P_cells=sparse.load_npz(d / "P_constraints_cells.npz").tocsr(),
        X=sparse.load_npz(d / "X_cells.npz").tocsr(),
        genes=pd.read_csv(d / "genes.csv")["gene_name"].to_numpy(dtype=str),
        types=pd.read_csv(d / "types.csv"),
        cells=pd.read_csv(d / "cells.csv"),
    )
    n_types, n_cells = b.P_cells.shape
    assert b.C.shape == b.C_mask.shape == (n_types, n_types), "C / C_mask / P disagree on types"
    assert len(b.types) == n_types, "types.csv length != number of types"
    assert b.X.shape == (n_cells, len(b.genes)) and len(b.cells) == n_cells, "cell or gene axis mismatch"
    return b


def save_bundle(d: str | Path, name: str, C, C_mask, P_cells, X, genes, types: pd.DataFrame,
                cells: pd.DataFrame, provenance: dict) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(d / "C.npz", sparse.csr_matrix(np.asarray(C, dtype=float)))
    np.save(d / "C_mask.npy", np.asarray(C_mask))
    sparse.save_npz(d / "P_constraints_cells.npz", sparse.csr_matrix(P_cells))
    sparse.save_npz(d / "X_cells.npz", sparse.csr_matrix((sparse.csr_matrix(X) > 0).astype(np.uint8)))
    pd.DataFrame({"gene_name": np.asarray(genes, dtype=str)}).to_csv(d / "genes.csv", index=False)
    types.to_csv(d / "types.csv", index=False)
    cells.to_csv(d / "cells.csv", index=False)
    meta = {"dataset": name, "n_types": int(len(types)), "n_cells": int(len(cells)),
            "n_genes": int(len(genes)), "n_metacells": int(cells["metacell"].max() + 1),
            "strata": types["stratum"].value_counts().sort_index().to_dict(),
            "provenance": provenance}
    (d / "bundle.json").write_text(json.dumps(meta, indent=2))
    return d
