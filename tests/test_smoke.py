"""Smoke tests on a small synthetic bundle (no lab data needed): pytest tests/"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from connectionminer import build_problem, kfold_types, load_bundle, run_fold
from connectionminer.data import save_bundle

SOLVER = {"num_iter": 3, "lambda_sparsity": 1e-3, "optimal_transport_epsilon": 1e-12,
          "optimal_transport_step": 0.04, "optimal_transport_iterations": 10,
          "regression_iterations": 10, "use_complement": False}


@pytest.fixture
def bundle(tmp_path):
    rng = np.random.default_rng(0)
    n_types, n_cells, n_genes = 40, 400, 30
    true = rng.integers(0, n_types, n_cells)
    P = np.zeros((n_types, n_cells))
    P[true, np.arange(n_cells)] = 1
    named = np.arange(n_types) < 8
    open_cells = ~named[true]
    P[np.ix_(~named, np.where(open_cells)[0])] = 1        # unnamed cells may be any unnamed type
    C = (rng.random((n_types, n_types)) < 0.1).astype(float)
    X = (rng.random((n_cells, n_genes)) < 0.2).astype(np.uint8)
    cells = pd.DataFrame({"cell_id": np.arange(n_cells), "metacell": true,
                          "batch": np.where(np.arange(n_cells) % 2, "a", "b")})
    types = pd.DataFrame({"type": [f"t{i}" for i in range(n_types)],
                          "stratum": np.where(named, "named", "unnamed")})
    save_bundle(tmp_path / "b", "toy", C, np.ones_like(C), P, X, [f"g{i}" for i in range(n_genes)],
                types, cells, {})
    return load_bundle(tmp_path / "b")


def test_folds_partition_types(bundle):
    for f in range(5):
        blk = kfold_types(bundle.strata, 5, f)
        allidx = np.concatenate(list(blk.values()))
        assert sorted(allidx) == list(range(len(bundle.strata)))
    tests = [kfold_types(bundle.strata, 5, f)["test"] for f in range(5)]
    assert sorted(np.concatenate(tests)) == list(range(len(bundle.strata)))   # each type tested once


@pytest.mark.parametrize("genes", [{"select": "all"},
                                   {"select": "binary_variance", "n_top": 20, "min_cells": 1},
                                   {"select": "all", "batch_normalize": {"pseudo": 10, "cap": 5}}])
def test_problem_shapes(bundle, genes):
    p = build_problem(bundle, np.arange(20), genes, {"min_samples_prior": 10})
    assert p.P_constraints_metacell.shape == (20, len(p.meta_sizes))
    assert p.G_metacell.shape == (len(p.meta_sizes), len(p.gene_idx))
    assert np.all((p.G_metacell >= 0) & (p.G_metacell <= 1))


def test_run_fold_freezes_beta(bundle, tmp_path):
    cfg = {"split": {"k": 5, "seed": 0}, "seed_offset": 0, "genes": {"select": "all"},
           "metacell": {"min_samples_prior": 10}, "solver": SOLVER}
    df = run_fold(bundle, cfg, 0, [0], tmp_path / "runs", device="cpu")
    assert set(df.split) == {"train", "val", "test"} and df.r.notna().all()
    d = tmp_path / "runs" / "fold0" / "seed0"
    b_tr, b_te = np.load(d / "train" / "beta.npy"), np.load(d / "test" / "beta.npy")
    ratio = b_te[b_tr > 0] / b_tr[b_tr > 0]
    assert np.allclose(ratio, ratio[0])                     # held-out beta = alpha * train beta
    # rerunning skips finished splits and reproduces the same scores
    df2 = run_fold(bundle, cfg, 0, [0], tmp_path / "runs", device="cpu")
    assert np.allclose(df.r, df2.r)
