"""One fold of the type-split protocol: train -> frozen-beta val/test -> score."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

from .data import Bundle
from .solver import solve
from .split import build_problem, kfold_types

SPLITS = ("train", "val", "test")


def score(C_hat: np.ndarray, C: np.ndarray, C_mask: np.ndarray) -> float:
    """Pearson r between C_hat and binarised C over the block's observed, masked entries."""
    W = (C_mask > 0) & ~np.isnan(C)
    return float(pearsonr((np.nan_to_num(C) > 0).astype(float)[W], C_hat[W])[0])


def run_fold(b: Bundle, cfg: dict, fold: int, seeds: list[int], out: Path,
             device: str = "auto") -> pd.DataFrame:
    """Solve fold `fold` for each seed and write results to out/fold{f}/.

    Per seed: train solves beta and P on the train types; val and test load that
    beta, freeze it (one scalar refit) and solve only P on their own types.
    Finished splits are skipped, so a killed job can simply be resubmitted.
    """
    k = int(cfg["split"]["k"])
    blocks = kfold_types(b.strata, k, fold, int(cfg["split"].get("seed", 0)))
    fdir = Path(out) / f"fold{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"type": b.types["type"], "stratum": b.strata,
                  "split": _split_labels(blocks, len(b.strata))}).to_csv(fdir / "split.csv", index=False)
    print(f"[{b.name} fold {fold}] " + ", ".join(f"{s} {len(v)} types" for s, v in blocks.items()), flush=True)

    problems = {"train": build_problem(b, blocks["train"], cfg["genes"], cfg["metacell"])}
    gene_idx = problems["train"].gene_idx
    for s in ("val", "test"):
        problems[s] = build_problem(b, blocks[s], cfg["genes"], cfg["metacell"], gene_idx=gene_idx)
    for s, p in problems.items():
        print(f"  {s}: {len(p.types)} types, {len(p.cells):,} cells, {len(p.meta_sizes):,} metacells, "
              f"{len(p.gene_idx)} genes", flush=True)

    rows = []
    for seed in seeds:
        solver_seed = int(cfg.get("seed_offset", 0)) + seed
        beta = None
        for s in SPLITS:
            d = fdir / f"seed{seed}" / s
            if not (d / "C_hat.npy").exists():
                p = problems[s]
                print(f"--- fold {fold} seed {seed} {s} (solver seed {solver_seed}) ---", flush=True)
                r = solve(p.G_metacell, p.P_constraints_metacell, p.C, p.C_mask, cfg["solver"],
                          seed=solver_seed, beta_fixed=beta, device=device)
                _save(d, p, r, solver_seed)
            if s == "train":
                beta = np.load(fdir / f"seed{seed}" / "train" / "beta.npy")
            p = problems[s]
            rows.append({"dataset": b.name, "fold": fold, "seed": seed, "split": s,
                         "n_types": len(p.types), "n_metacells": len(p.meta_sizes),
                         "r": score(np.load(d / "C_hat.npy"), p.C, p.C_mask)})
    df = pd.DataFrame(rows)
    df.to_csv(fdir / "results.csv", index=False)
    print(df.groupby("split", sort=False).r.agg(["mean", "std", "count"]).round(4).to_string(), flush=True)
    return df


def _save(d: Path, p, r, solver_seed: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "beta.npy", r.beta)            # val/test: the frozen train beta times its fitted scalar
    sparse.save_npz(d / "P.npz", sparse.csr_matrix(r.P))
    np.savez_compressed(d / "block.npz", types=p.types, cells=p.cells, cell_to_metacell=p.cell_to_metacell,
                        genes=p.genes)
    pd.DataFrame(r.loss).rename_axis("iteration").to_csv(d / "loss.csv")
    (d / "solve.json").write_text(json.dumps({"solver_seed": solver_seed, "elapsed_sec": r.elapsed_sec,
                                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    np.save(d / "C_hat.npy", r.C_hat)          # written last: its presence marks the split done


def _split_labels(blocks: dict, n: int) -> np.ndarray:
    lab = np.empty(n, dtype=object)
    for s, idx in blocks.items():
        lab[idx] = s
    return lab


def summarize(out: Path) -> pd.DataFrame:
    """Mean r per split over every fold and seed found under `out`."""
    files = sorted(Path(out).glob("fold*/results.csv"))
    if not files:
        raise FileNotFoundError(f"no fold*/results.csv under {out}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = df.groupby("split", sort=False).r.agg(["mean", "std", "median", "count"])
    s["folds"] = df.groupby("split", sort=False).fold.nunique()
    return s.reindex([x for x in SPLITS if x in s.index])
