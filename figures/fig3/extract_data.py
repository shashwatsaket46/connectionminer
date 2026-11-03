#!/usr/bin/env python3
"""CLUSTER-ONLY: extract the minimal inputs of Figure 3 into ./data/ so fig3.py runs anywhere.

You do NOT need this script to regenerate or edit the figure; it documents where every number in data/ comes from
and rebuilds data/ on the cluster. Run from anywhere:
    PYTHONPATH=/scratch/ss20308/python-user-lib/python3.13/site-packages python3 extract_data.py
(needs anndata for the visual t-SNE; everything else numpy / scipy / pandas).

Provenance (all paths on the NYU HPC cluster, root CMA = /scratch/ss20308/cm_active):
  type table, soft cell assignment  CMA/biorender_fig/fig3/cache/<sys>.npz   (fig3_data.py: mean P share per type
                                    and cell over the 50 real solves; contrib = soft cell count; named = annotated type)
  P.G on all genes                  CMA/pbnull/pg_allgenes/<label>/pg_allgenes.npz  ("mean" = mean P.G over solves)
                                    labels: visual Adult_nofilt, cx CXbestmc1, motor motor_tp3
  dot sizes (cells detecting gene)  CMA/biorender_fig/fig3/cache/dot_<sys>.npz  (fig3_dot_data.py: cellP @ (X > 0))
  embedding                         visual: CMA/data/Adult_noglia.h5ad obsm X_tsne (== build cell order, asserted);
                                    cx / motor: <build>/cell_index.csv umap_x, umap_y
  skeletons (x, z segments)         visual / cx: CMA/biorender_fig/fig3/cache/skel_<sys>.npz (fig3_skel.py, <= 25
                                    FlyWire neurons per type); motor: CMA/pbnull/dash/data/_morph/motor/geom.json (FANC,
                                    one neuron per type, z pre-flipped)
  connectome C (types x types)      <build>/C_matrix.npz, build = visual CMA/output, cx
                                    /scratch/ss20308/cm_central_complex/grid_mc1/runs/g5_h2000_bn1_m1, motor
                                    /scratch/ss20308/cm_motor/output_tp3   (used only for "wiring r" of a pair)
  sister-pair screen                CMA/biorender_fig/fig3/cache/pairs_<sys>.csv (fig3_sister_pairs.py), restricted to
                                    pairs of offline types -> data/pairs_<sys>.csv (browse it to pick a new pair)

Subsetting (keeps data/ small):
  * offline TYPES (cell probabilities + dot sizes kept): the top-20 inferred types (same rule as fig3.py) + the
    default pair + the next NEXTRA (100) most confident inferred types (visual / cx); motor: every inferred type with
    >= 20 soft cells that has a FANC skeleton. Any two offline types that have skeletons can be drawn as the pair.
  * GENES: named genes with P.G >= 0.02 in any offline type (covers the panel-b scatter, the >= 0.3 markers and
    pair genes) + the shared genes; P.G is kept for ALL types on those genes (panel c colours every EM neuron).
  * skeletons: all types (the panel-c background is every neuron).
"""
import json, base64, re
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

OUT = Path(__file__).resolve().parent / "data"
CMA = Path("/scratch/ss20308/cm_active")
F3 = CMA / "biorender_fig/fig3"
PGLAB = {"visual": "Adult_nofilt", "cx": "CXbestmc1", "motor": "motor_tp3"}
BUILD = {"visual": CMA / "output", "cx": Path("/scratch/ss20308/cm_central_complex/grid_mc1/runs/g5_h2000_bn1_m1"),
         "motor": Path("/scratch/ss20308/cm_motor/output_tp3")}
PAIRS = {"visual": ("Dm8a", "Dm8b"), "cx": ("PEN_a/PEN1", "PEN_b/PEN2"), "motor": ("Fe10", "Fe11")}
EXTRA_TYPES = {"cx": ["PFNa", "PFNd", "PFNp", "PFNv", "PFNm"], "motor": ["LN393", "LN432", "Fe6", "Fe7", "Fe8"],
               "visual": []}                                      # earlier candidate pairs, always kept
SHARED = ["hig", "kirre", "CanA-14F"]
NTOP, MIN_CELLS, NEXTRA, PG_KEEP = 20, 20, 100, 0.02
ANON = re.compile(r"\d{4,}|^(CG|CR)\d|:CR\d|^FBti")


def skeleton(k):
    if k != "motor":
        z = np.load(F3 / f"cache/skel_{k}.npz")
        return z["segs"].astype(np.float32), z["type"].astype(np.int32), z["types"].astype(str)
    j = json.load(open(CMA / "pbnull/dash/data/_morph/motor/geom.json"))
    segs, typ = [], []
    for t, l in enumerate(j["per_type"]):
        for q in l:
            s = j["skel"][q]
            p = np.frombuffer(base64.b64decode(s["p"]), np.int16).reshape(-1, 3).astype(float)
            e = np.frombuffer(base64.b64decode(s["e"]), np.uint16).reshape(-1, 2)
            segs.append(p[e][:, :, [0, 2]]); typ.append(np.full(len(e), t))
    return np.concatenate(segs).astype(np.float32), np.concatenate(typ).astype(np.int32), np.array(j["types"], str)


def embedding(k):
    if k == "visual":
        import anndata as ad
        a = ad.read_h5ad(CMA / "data/Adult_noglia.h5ad", backed="r")
        ci = pd.read_csv(BUILD[k] / "cell_index.csv", usecols=["cell_barcode"])
        assert (a.obs_names.values == ci.cell_barcode.values).all()
        return np.asarray(a.obsm["X_tsne"], np.float32), "t-SNE"
    ci = pd.read_csv(BUILD[k] / "cell_index.csv", usecols=["umap_x", "umap_y"])
    return ci[["umap_x", "umap_y"]].to_numpy(np.float32), "UMAP"


def main(k):
    z = np.load(F3 / f"cache/{k}.npz")
    types, named_t, contrib, P = z["types"].astype(str), z["named"], z["contrib"], z["cellP"]
    conf = (P.astype(float) ** 2).sum(1) / np.maximum(contrib, 1e-12)
    segs, styp, stypes = skeleton(k)
    has_sk = np.isin(types, stypes[np.unique(styp)])
    ok = (~named_t) & (contrib >= MIN_CELLS)
    by_conf = [i for i in np.argsort(-conf) if ok[i]]
    if k == "motor":
        cand = [i for i in by_conf if has_sk[i]]
    else:
        cand = by_conf[:NTOP + NEXTRA]
    cand = set(cand) | set(by_conf[:NTOP])
    cand |= {list(types).index(t) for t in (*PAIRS[k], *EXTRA_TYPES[k]) if t in types}
    cand = np.array(sorted(cand, key=lambda i: -conf[i]))

    g = np.load(CMA / f"pbnull/pg_allgenes/{PGLAB[k]}/pg_allgenes.npz", allow_pickle=True)
    assert (g["types"].astype(str) == types).all()
    genes, PG = g["genes"].astype(str), g["mean"].astype(np.float32)
    named_g = np.array([not ANON.search(x) for x in genes])
    keep = (named_g & (PG[cand].max(0) >= PG_KEEP)) | np.isin(genes, SHARED)
    gi = np.where(keep)[0]                                          # original gene order preserved

    dz = np.load(F3 / f"cache/dot_{k}.npz")
    assert (dz["types"].astype(str) == types).all() and (dz["genes"].astype(str) == genes).all()
    C = sp.load_npz(BUILD[k] / "C_matrix.npz").toarray().astype(np.float32)
    ti = pd.read_csv(BUILD[k] / "type_index.csv")
    assert (ti.type.astype(str).to_numpy() == types).all() and C.shape == (len(types),) * 2
    xy, kind = embedding(k)
    assert len(xy) == P.shape[1]

    np.savez_compressed(OUT / f"{k}.npz",
                        types=types, type_named=named_t, contrib=contrib.astype(np.float32), conf=conf,
                        cand=cand.astype(np.int32), cellP=P[cand].astype(np.float32),
                        genes=genes[gi], gene_named=named_g[gi], PG=PG[:, gi], n_expr=dz["n_expr"][np.ix_(cand, gi)],
                        xy=xy, emb_kind=kind, C=C,
                        skel_segs=segs, skel_type=styp, skel_types=stypes)
    p = pd.read_csv(F3 / f"cache/pairs_{k}.csv")
    ct = set(types[cand][has_sk[cand]])
    p = p[p.a.isin(ct) & p.b.isin(ct)].sort_values("mass_ovl", ascending=False)
    p.to_csv(OUT / f"pairs_{k}.csv", index=False, float_format="%.4g")
    print(f"{k}: {len(cand)} offline types ({has_sk[cand].sum()} with skeleton), {len(gi)}/{len(genes)} genes, "
          f"{len(p)} screened pairs among them, {len(segs)} skeleton segments")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for k in ("visual", "cx", "motor"):
        main(k)
