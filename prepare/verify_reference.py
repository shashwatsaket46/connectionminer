#!/usr/bin/env python3
"""Check that this repo rebuilds the paper runs' solver inputs exactly.

For each fold, compares the split's type set, solver genes, P constraints and
metacell expression G against what the original Fig 1 runs saved
(split_matrices/ in each run dir). Cluster-only: needs the original run dirs.

    python prepare/verify_reference.py visual --folds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectionminer.data import load_bundle  # noqa: E402
from connectionminer.split import build_problem, kfold_types  # noqa: E402

S = Path("/scratch/ss20308")
REF = {
    "visual": lambda f, s: S / f"cm_active/devstages/Adult/typesplit_k10_f{f}_nofilt/h3000_uniform_s0/rep0_{s}",
    "motor": lambda f, s: S / f"cm_active/devstages/motor/tp3/typesplit_k10_f{f}/h1000_uniform_s0/rep0_{s}",
    "cx": lambda f, s: S / f"cm_central_complex/kfold_best_mc1/typesplit_k10_f{f}/h2000_uniform_s0/rep0_{s}",
}


def ref_types(name, d):
    if name == "cx":
        return np.asarray(json.loads((d / "split.json").read_text())["types"])
    if name == "visual":     # visual run dirs hold solver outputs; the prepared block lives in shared/
        d = d.parent.parent / "shared" / d.name
    return pd.read_csv(d / "type_index.csv")["orig_col_idx"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(REF))
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{a.dataset}.yaml").read_text())
    b = load_bundle(ROOT / "data" / a.dataset)
    ok = True
    for f in a.folds:
        blocks = kfold_types(b.strata, cfg["split"]["k"], f, cfg["split"]["seed"])
        gidx = None
        for s in ("train", "val", "test"):
            d = REF[a.dataset](f, s)
            p = build_problem(b, blocks[s], cfg["genes"], cfg["metacell"], gene_idx=gidx)
            if s == "train":
                gidx = p.gene_idx
            sm = d / "split_matrices"
            G_ref = np.load(sm / "G_metacell_p.npy")
            D_ref = sparse.load_npz(sm / "P_constraints_metacell.npz").toarray()
            genes_ref = np.load(sm / "genes_solver.npy", allow_pickle=True).astype(str)
            checks = {
                "types": np.array_equal(np.sort(blocks[s]), np.sort(ref_types(a.dataset, d)))
                and np.array_equal(blocks[s], ref_types(a.dataset, d)),
                "genes": np.array_equal(p.genes, genes_ref),
                "D": D_ref.shape == p.P_constraints_metacell.shape
                and np.array_equal(D_ref, p.P_constraints_metacell),
                "G": G_ref.shape == p.G_metacell.shape,
            }
            gdiff = float(np.max(np.abs(G_ref - p.G_metacell))) if checks["G"] else float("nan")
            checks["G"] = checks["G"] and gdiff == 0.0
            ok &= all(checks.values())
            print(f"{a.dataset} fold {f} {s:5s}: {len(p.types)} types {p.G_metacell.shape} "
                  + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items())
                  + f"  max|dG|={gdiff:.2e}", flush=True)
    print("ALL EXACT" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
