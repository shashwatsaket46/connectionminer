"""KIT NOTE: in this repository the three caches are stored in data/ in slimmed form: only the arrays the figure
draws are kept; cm.npz stores train_E = train_P @ train_G instead of G; both beta matrices are reduced to
beta_score (the gene ranking) and beta_top (the 14 x 14 drawn block); clusters.npz adds c2m (cell -> metacell).
"""
"""Cache the REAL intermediate objects of the gene-based baseline (Scanpy units + Connectome-Model rule + matching)
and of ConnectionMiner for the visual system, fold 0, test block, for figS_baseline_real.py.

Three stages, each run in the environment it needs (nothing is copied: the baseline / pbnull functions are imported):

  clusters  baselines/.env_tools/bin/python biorender_fig/figS_baseline_real_extract.py clusters
            stage 1 (indep_cluster.py) did NOT save per-cell cluster labels, so the scanpy Leiden partition at the
            selected resolution is recomputed with the same code path (load_cells -> HVG -> PCA(50) -> 15-NN graph
            -> Leiden, seed 0) and checked against the stored unit profiles in visual_scanpy_units.npz.
            Also caches the Fig-1 t-SNE (Adult_noglia.h5ad obsm X_tsne) and the annotated flag.
  baseline  env -u PYTHONPATH /scratch/ss20308/cm_latest/cm/bin/python biorender_fig/figS_baseline_real_extract.py baseline
            replays indep_connectome.main() for fold 0 only (same rng stream, so the matching is bit-identical), records
            the chosen ridge lambda, and saves X_train, C_train, beta, the matched unit of every test type, C_hat, C, r.
            Every fold-0 r_matched / r_unmatched is checked against visual_scanpy_scores.csv.
  cm        PBNULL_STAGE=Adult PBNULL_TAG=_nofilt (set here) ... figS_baseline_real_extract.py cm
            ConnectionMiner fold 0 seed 0: train P, G, beta; test P, G, C_hat, C; r via pbnull.common.score.

Output: biorender_fig/cache_baseline_real/{clusters,baseline,cm}.npz
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache_baseline_real"
CACHE.mkdir(exist_ok=True)
SYSTEM, TOOL, FOLD, SPLIT, RULE = "visual", "scanpy", 0, "test", "cm"
RUN = ROOT / "baselines" / "runs" / "indep"


def chosen_res():
    s = pd.read_csv(RUN / f"{SYSTEM}_{TOOL}_summary.csv")
    return s[(s.rule == RULE) & (s.arm == "real")].res.iloc[0]


# ---------------------------------------------------------------------------------------------------- clusters
def clusters():
    import anndata as ad
    import leidenalg
    import scanpy as sc
    sys.argv = ["indep_cluster.py", SYSTEM, TOOL]
    sys.path.insert(0, str(ROOT / "baselines"))
    import indep_cluster as icl
    from standard_tools import load_cells

    res = chosen_res()
    r = float(res[1:])
    a = load_cells(SYSTEM)                                   # same preprocessing as indep_cluster.main()
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    hv = a.var.highly_variable.to_numpy()
    sc.tl.pca(a, n_comps=50, mask_var="highly_variable", random_state=0)
    X = a.X[:, hv].tocsr()
    truth = a.obs.truth.astype(str).to_numpy()
    un = truth == "unannotated"
    g = icl.graph(icl.knn_edges(a.obsm["X_pca"], 15))       # partitions('scanpy', a), one resolution only
    cl = np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                           resolution_parameter=r, seed=0).membership)
    ks, n = np.unique(cl[un], return_counts=True)
    ks = ks[n >= icl.MIN_CELLS]
    lab = np.where(un, cl, -1)
    Kx = icl.mean_by(X, lab, ks)
    z = np.load(RUN / f"{SYSTEM}_{TOOL}_units.npz", allow_pickle=True)
    dX = np.abs(Kx - z[f"{res}_X"]).max() if Kx.shape == z[f"{res}_X"].shape else np.inf
    print(f"Leiden {res}: {len(set(cl))} clusters, {len(ks)} units; max |profile - stored| = {dX:.3g}; "
          f"genes equal: {(a.var_names[hv].to_numpy().astype(str) == z['genes']).all()}")
    unit = np.full(len(cl), -1)                             # unit index (row of <res>_X) per cell, -1 = none
    for i, k in enumerate(ks):
        unit[(lab == k)] = i
    e = ad.read_h5ad(ROOT / "data" / "Adult_noglia.h5ad", backed="r")
    assert (e.obs_names.values == a.obs_names.values).all()
    np.savez_compressed(CACHE / "clusters.npz", res=res, cluster=cl, unit=unit, annotated=~un, truth=truth,
                        tsne=np.asarray(e.obsm["X_tsne"]), barcodes=a.obs_names.to_numpy().astype(str),
                        max_profile_diff=dX)


# ---------------------------------------------------------------------------------------------------- baseline
def baseline():
    sys.argv = ["indep_connectome.py", SYSTEM, TOOL]
    sys.path.insert(0, str(ROOT / "baselines"))
    import indep_connectome as ic

    picked = []
    _tuned = ic.tuned

    def tuned_rec(fit, grid, X, C, W, fold):                # record the setting each tuned() call picks
        out = _tuned(fit, grid, X, C, W, fold)
        picked.append((out[1], out[2]))
        return out
    ic.tuned = tuned_rec

    res_pick = chosen_res()
    stored = pd.read_csv(RUN / f"{SYSTEM}_{TOOL}_scores.csv")
    stored = stored[stored.fold == FOLD]

    # ---- exactly indep_connectome.main() for f = FOLD (same rng consumption order)
    z = np.load(RUN / f"{SYSTEM}_{TOOL}_units.npz", allow_pickle=True)
    RES = z["settings"].astype(str)
    names = pd.read_csv(ic.TYPE_INDEX[SYSTEM]).type.astype(str).to_numpy().astype(str)
    ann_pos = {n: i for i, n in enumerate(z["ann_types"].astype(str))}
    f = FOLD
    rng = np.random.default_rng(f)
    idx = {s: ic.block_types(SYSTEM, f, s) for s in ("train", "val", "test")}
    C, W = ic.load_block(ic.BLOCKS[SYSTEM](f, "train"))
    tr = np.array([k for k, g in enumerate(idx["train"]) if names[g] in ann_pos])
    Xtr = z["ann_X"][[ann_pos[names[g]] for g in idx["train"][tr]]]
    Ctr, Wtr = C[np.ix_(tr, tr)], W[np.ix_(tr, tr)]
    rules = ic.fit_rules(Xtr, Ctr, Wtr, f)
    lam_cm, lam_scores = picked[0]                          # first tuned() call in fit_rules = cm ridge
    perm = rng.permutation(len(tr))
    nulls = ic.fit_rules(Xtr[perm], Ctr, Wtr, f)
    s4 = np.linalg.svd(Xtr, compute_uv=False).max() ** 4
    keep, checks = None, []
    for res in RES:
        K = z[f"{res}_X"]
        Uall = np.vstack([z["ann_X"], K, K.mean(0, keepdims=True)])
        cand = np.arange(len(z["ann_X"]), len(z["ann_X"]) + len(K))
        mean_unit = len(Uall) - 1
        for split in ("val", "test"):
            Cb, Wb = ic.load_block(ic.BLOCKS[SYSTEM](f, split))
            g = idx[split]
            a = np.array([ann_pos.get(names[x], mean_unit) for x in g])
            free = np.where(a == mean_unit)[0]
            for rule in ("cm", "bilinear"):
                for arm, beta in (("real", rules[rule]), ("null rule", nulls[rule])):
                    S = Uall @ beta @ Uall.T
                    r_nm = ic.r_full(S[np.ix_(a, a)], Cb, Wb)
                    am, r_m = ic.match(S, a, free, cand, Cb, Wb, rng)
                    q = stored[(stored.split == split) & (stored.res == res) & (stored.rule == rule) &
                               (stored.arm == arm)].iloc[0]
                    checks.append((res, split, rule, arm, r_m, q.r_matched, r_nm, q.r_unmatched))
                    if (res, split, rule, arm) == (res_pick, SPLIT, RULE, "real"):
                        keep = dict(S=S, a_matched=am, a_unmatched=a, free=free, cand=cand, Cb=Cb, Wb=Wb,
                                    g=g, r=r_m, r_stored=q.r_matched, r_unmatched=r_nm, beta=beta,
                                    nunits=len(K), ncells=z[f"{res}_ncells"])
    ck = pd.DataFrame(checks, columns=["res", "split", "rule", "arm", "r_matched", "stored_matched",
                                       "r_unmatched", "stored_unmatched"])
    ck["d"] = np.maximum((ck.r_matched - ck.stored_matched).abs(), (ck.r_unmatched - ck.stored_unmatched).abs())
    print(ck.round(4).to_string(index=False))
    print(f"max |recomputed - stored| over all {len(ck)} fold-{FOLD} rows: {ck.d.max():.3g}")
    k = keep
    S, am, free, cand, Cb, Wb = k["S"], k["a_matched"], k["free"], k["cand"], k["Cb"], k["Wb"]
    Chat = S[np.ix_(am, am)]
    # r of each free test type moved to each candidate unit (others at the matched assignment)
    gain = np.full((len(free), len(cand)), np.nan)
    for i, t in enumerate(free):
        for j, u in enumerate(cand):
            a2 = am.copy(); a2[t] = u
            gain[i, j] = ic.r_full(S[np.ix_(a2, a2)], Cb, Wb)
    nA = len(z["ann_X"])
    unit_names = np.array(list(z["ann_types"].astype(str)) + [f"U{i}" for i in range(k["nunits"])] + ["mean"])
    print(f"chosen res {res_pick}; cm lambda setting {lam_cm:g} (x s_max^4 = {lam_cm * s4:.4g}); inner-split r: "
          + ", ".join(f"{s:g}:{v:.3f}" for s, v in lam_scores.items()))
    print(f"test r recomputed {k['r']:.6f}  stored {k['r_stored']:.6f}; unmatched {k['r_unmatched']:.4f}")
    np.savez_compressed(
        CACHE / "baseline.npz", res=res_pick, lam=lam_cm, lam_abs=lam_cm * s4,
        lam_grid=np.array(list(lam_scores)), lam_inner_r=np.array(list(lam_scores.values())),
        genes=z["genes"].astype(str), Xtr=Xtr, Xtr_types=names[idx["train"][tr]], Ctr=Ctr, Wtr=Wtr,
        beta=k["beta"], test_types=names[k["g"]], test_matched=am, test_free=free, unit_names=unit_names,
        n_ann_units=nA, unit_ncells=k["ncells"], Chat=Chat, C=Cb, W=Wb, r=k["r"], r_stored=k["r_stored"],
        r_unmatched=k["r_unmatched"], move_r=gain, max_check_diff=ck.d.max())


# ---------------------------------------------------------------------------------------------------- cm
def cm():
    os.environ["PBNULL_STAGE"], os.environ["PBNULL_TAG"] = "Adult", "_nofilt"
    sys.path.insert(0, str(ROOT / "pbnull"))
    import common as pc
    tr = pc.load_split(FOLD, 0, "train", need_P=True)
    te = pc.load_split(FOLD, 0, SPLIT, need_P=True)
    btr = np.load(pc.train_beta_path(FOLD, 0))
    sc_ = pc.score(te["Chat"], te["C"], te["M"], te["named"])
    Chat_re = pc.chat(te["P"], te["G"], btr)
    rre = pc.score(Chat_re, te["C"], te["M"], te["named"])["pearson_all"]
    res = pd.read_csv(pc.real_dir(FOLD, 0, SPLIT).parents[1] / "results" / "results.csv")
    res = res[(res.split == SPLIT) & (res.solver_seed == 0)].pearson_all.iloc[0]
    print(f"CM test r: pc.score {sc_['pearson_all']:.6f}; from P G beta_train {rre:.6f}; results.csv {res:.6f}; "
          f"test beta == train beta: {np.array_equal(te['beta'], btr)}")
    sh = pc.real_dir(FOLD, 0, SPLIT).parents[1] / "shared"
    out = {}
    for s, d in (("train", tr), ("test", te)):
        ti = pd.read_csv(sh / f"rep0_{s}" / "type_index.csv")
        cmb = pd.read_csv(sh / f"rep0_{s}" / "cell_membership.csv",
                          usecols=["cell_barcode", "orig_metacell_idx", "metacell_idx"])
        out.update({f"{s}_P": d["P"], f"{s}_C": d["C"], f"{s}_M": d["M"], f"{s}_G": d["G"], f"{s}_D": d["D"], f"{s}_named": d["named"],
                    f"{s}_types": ti.type.to_numpy().astype(str), f"{s}_bc": cmb.cell_barcode.to_numpy().astype(str),
                    f"{s}_mc": cmb.metacell_idx.to_numpy(), f"{s}_mc_orig": cmb.orig_metacell_idx.to_numpy()})
    genes = np.load(pc.real_dir(FOLD, 0, "train") / "split_matrices" / "genes_solver.npy", allow_pickle=True)
    np.savez_compressed(CACHE / "cm.npz", beta=btr, genes=genes.astype(str), test_Chat=te["Chat"],
                        r=sc_["pearson_all"], r_results=res, r_recomputed=rre, **out)


if __name__ == "__main__":
    {"clusters": clusters, "baseline": baseline, "cm": cm}[sys.argv[1]]()
