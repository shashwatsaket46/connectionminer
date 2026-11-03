"""CLUSTER-ONLY: pre-extract every input Figure 1 draws into figure_kit/fig1/data/ (fig1.py then needs nothing else).

Provenance (the paper build is biorender_fig/fig1_compose_v5.py -> figures/fig1_v5.png and
biorender_fig/fig1_svg_v5.py -> figures/editable_svg/fig1_editable.svg; this script reuses their loaders read-only):

  panelA_base.png    BioRender drawing biorender_fig/figure_v5SL_real.png, crop (14, 14, 1390, 744), widened by 420 px
                     of white on the right, old count text / legend / column titles blanked (rectangles copied from
                     fig1_compose_v5.py). Native resolution; fig1.py scales it by UP_A = 1.22.
  panelB_tiles.npz   panel B (panelB_v5.panel_D): for each system the fold-0 / seed-0 TRAIN solve
                     (panelsCDE.RUN[k](0, 0)/rep0_train: C_matrix.npz, split_matrices/G_metacell_p.npy,
                     connectionMiner_solve/P_refined.npz, beta_learned.npy, C_reconstructed.npy,
                     split_matrices/P_constraints_metacell.npz), rows/columns ordered exactly as panel_D, then
                     block-pooled to the pixel grid panel_D pools to (same pool sizes, same pooling function per tile:
                     mean for P/G/G^T/P^T, nanmax for beta, max for C_hat/C). Also the full-matrix colour limits
                     (G 99th pct, beta min/max, C_hat floor/top) and each tile's full matrix shape (tile geometry).
                     The P / P^T tiles keep the pooled constraint mask D (light tint) next to the pooled P values.
  panelB_scores.json train r / AP / F1 of that block (as panel_D computes them) + matrix shapes.
  panelC.npz         panel C (fig1_panelC_v5.prepost_panel): tf_blocks.collect_cached (biorender_fig/tiles/tf40_*.npz,
                     40 train solves per type), rows / 24 TFs selected by fig1_output.prep; stored: type-relative
                     PRE (P_init.G) and POST (P.G) heatmap values in drawn row/column order, colour limit, family
                     block boundaries, row labels, TF names, highlighted type rows, and the 2-D skeleton segments of the
                     two highlighted types (panelsCDE.closest_pair_any: FlyWire SWCs for visual, viewer JSON for CX /
                     motor), x/y only (the 3-D z is never drawn).

Run:  PYTHONPATH=/scratch/ss20308/python-user-lib/python3.13/site-packages OMP_NUM_THREADS=3 \
      python3 figure_kit/fig1/extract_data.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import scipy.sparse as sp
from PIL import Image, ImageDraw

KIT = Path(__file__).resolve().parent
OUT = KIT / "data"
OUT.mkdir(exist_ok=True)
BF = Path("/scratch/ss20308/cm_active/biorender_fig")
sys.path.insert(0, str(BF))

# ------------------------------------------------------------------ geometry shared with the composer
UP, GAP, M, TITLE_H = 1.5, 24, 10, 92
UP_A, CW, W_TOT, H_LEFT = 1.22, 1300, 3600, 2440
EXTRA_A = 420
ROW_Y = {"visual": 185, "cx": 358, "motor": 531}

# ------------------------------------------------------------------ A
src = Image.open(BF / "figure_v5SL_real.png").convert("RGB")
crop = src.crop((14, 14, 1390, 744))
wide = Image.new("RGB", (crop.width + EXTRA_A, crop.height), "white"); wide.paste(crop, (0, 0))
dw = ImageDraw.Draw(wide)
for k, yc in ROW_Y.items():
    dw.rectangle([862, yc - 50, 1028, yc + 52], fill="white")
    dw.rectangle([1028, yc - 14, 1078 if k == "visual" else 1082, yc + 14], fill="white")
    dw.rectangle([1250, yc - 52, crop.width - 1, yc + 52], fill="white")
dw.rectangle([680, 622, crop.width - 1, 662], fill="white")
dw.rectangle([440, 662, crop.width - 1, crop.height - 1], fill="white")
wide.save(OUT / "panelA_base.png", optimize=True)
print("panelA_base.png", wide.size)

AH = int(wide.height * UP_A)
LW = W_TOT - 2 * M - GAP - CW
dataH = AH + 30
bh = H_LEFT - dataH - GAP
WB, HB = LW - 40, bh - TITLE_H - 20                   # panel_D(w, h) as the composer calls it

# ------------------------------------------------------------------ B (mirrors panelB_v5.panel_D up to imshow)
import panelsCDE  # noqa: E402
from panelsCDE import SYS, MASK, pool, solve_mats, shared_gene_order  # noqa: E402
from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
import panelB_v5  # noqa: E402

panelsCDE.D_GCMAP, panelsCDE.D_GLOG = panelB_v5.D_GCMAP, panelB_v5.D_GLOG
assert panelB_v5.D_GCMAP == "system" and panelB_v5.D_GLOG
D_TYPE_UP, X_MAT = panelB_v5.D_TYPE_UP, panelB_v5.X_MAT
LBL = ["P", "G", "beta", "GT", "PT", "Chat", "C"]
FUN = [np.mean, np.mean, np.nanmax, np.mean, np.mean, np.max, np.max]

rows_, scores, extra = [], {}, {}
for k in SYS:
    C, G, P, B, Ch = solve_mats(k)
    r = np.corrcoef(C.ravel(), Ch.ravel())[0, 1]
    y_, s_ = (C.ravel() > 0).astype(int), Ch.ravel()
    ap = average_precision_score(y_, s_)
    pr_ = s_ >= np.quantile(s_, 1 - y_.mean()); f1 = 2 * (pr_ & (y_ == 1)).sum() / (pr_.sum() + y_.sum())
    Dm = sp.load_npz(panelsCDE.RUN[k](0, 0) / "rep0_train/split_matrices/P_constraints_metacell.npz").toarray() > 0
    n_cand = Dm.sum(0)
    named = np.zeros(Dm.shape[0], bool); named[Dm[:, n_cand == 1].argmax(0)] = True
    tiers = []
    for sel in (named, ~named):
        idx = np.where(sel)[0]
        if len(idx) >= 3 and C[np.ix_(idx, idx)].std() > 0:
            idx = idx[leaves_list(linkage(pdist(C[np.ix_(idx, idx)], metric="jaccard"), "average"))]
        tiers.append(idx)
    to = np.concatenate(tiers)
    rank = np.empty(len(to), int); rank[to] = np.arange(len(to))
    n_un = (~named).sum()
    pinned = n_cand == 1; full = n_cand >= n_un; narrowed = ~pinned & ~full
    mo = np.concatenate([np.where(m)[0][np.argsort(key[m], kind="stable")] for m, key in
                         ((pinned, rank[Dm.argmax(0)]), (narrowed, rank[P.argmax(0)]), (full, rank[P.argmax(0)]))])
    panelsCDE.CUR_SYS = k
    ga = shared_gene_order(k, G); bo = ga
    Bb = np.where(B[np.ix_(bo, bo)] > 1e-6, B[np.ix_(bo, bo)], np.nan)
    mats = [P[np.ix_(to, mo)], G[np.ix_(mo, ga)], Bb, G[np.ix_(mo, ga)].T, P[np.ix_(to, mo)].T,
            Ch[np.ix_(to, to)], C[np.ix_(to, to)]]
    mask = Dm[np.ix_(to, mo)].astype(float)
    extra[k] = dict(mask=mask, G99=float(np.percentile(G, 99)), Bmin=float(np.nanmin(Bb)), Bmax=float(np.nanmax(Bb)),
                    Chlo=float(np.quantile(Ch, 1 - C.mean())), Chhi=float(np.percentile(Ch, 99.9)))
    rows_.append((k, mats))
    scores[k] = dict(train_r=float(r), train_AP=float(ap), train_F1=float(f1), edge_density=float(y_.mean()),
                     shapes={l_: list(m_.shape) for l_, m_ in zip(LBL, mats)})
    print(f"{k}: r {r:.3f} AP {ap:.3f} F1 {f1:.3f}")

# same geometry as panel_D -> same pool sizes
w, h = WB, HB
L = lambda n: np.sqrt(n)
Lr = lambda j, n: L(n) * (D_TYPE_UP if j in (0, 5, 6) else 1)
Lc = lambda j, n: L(n) * (D_TYPE_UP if j in (4, 5, 6) else 1)
gap, sym = 0.04 * w, 0.03 * w
slot = [max(Lc(j, t[j].shape[1]) for _, t in rows_) for j in range(7)]
max_h = max(Lr(j, Mx.shape[0]) for _, t in rows_ for j, Mx in enumerate(t))
a_ = min(((0.99 - X_MAT) * w - 6 * gap - 2 * sym) / sum(slot), 0.255 * h / max_h)
arrs = {}
for k, mats in rows_:
    for j, (lbl, Mx) in enumerate(zip(LBL, mats)):
        tw, th = a_ * Lc(j, Mx.shape[1]) / w, a_ * Lr(j, Mx.shape[0]) / h
        if lbl in ("G", "GT"):
            Mp = pool(Mx, max(2, int(th * h * 2)), max(2, int(tw * w * 2)), np.mean)
        else:
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Mp = pool(Mx, max(2, int(th * h * 1.5)), max(2, int(tw * w * 1.5)), FUN[j])
        arrs[f"{k}_{lbl}"] = Mp.astype(np.float64)
        if lbl in ("P", "PT"):
            mk = extra[k]["mask"] if lbl == "P" else extra[k]["mask"].T
            arrs[f"{k}_{lbl}_mask"] = pool(mk, *Mp.shape, np.max).astype(np.float64)
    for q in ("G99", "Bmin", "Bmax", "Chlo", "Chhi"):
        arrs[f"{k}_{q}"] = np.array(extra[k][q])
np.savez_compressed(OUT / "panelB_tiles.npz", **arrs)
(OUT / "panelB_scores.json").write_text(json.dumps(dict(
    note="fold 0 / seed 0 training block; shapes = full matrix (rows, cols) of each tile, used for tile geometry; "
         "the npz holds the block-pooled values actually drawn",
    panel_D_size_px=[WB, HB], **scores), indent=1))
print("panelB_tiles.npz", {k_: v.shape for k_, v in arrs.items() if v.ndim})

# ------------------------------------------------------------------ C (mirrors fig1_panelC_v5.prepost_panel)
from tf_blocks import collect_cached  # noqa: E402
from fig1_output import prep, HL_EDGE  # noqa: E402
from panelsCDE import closest_pair_any  # noqa: E402

# highlighted (pre, post) type pair per system; None = the paper's choice (fig1_output.HL_EDGE:
# visual Li02/Li28, CX FS1A/FB4Y, motor LN314/LN191). If you change it, set fig1.py's HIGHLIGHT to the same pair.
HIGHLIGHT = None
HL = HIGHLIGHT or HL_EDGE

cz, meta = {}, {}
for k in SYS:
    genes, Dc, ro, gsel, labels, _ = prep(k)
    types, genes_c, I, F = collect_cached(k)
    Ic, Fc = I - I.mean(1, keepdims=True), F - F.mean(1, keepdims=True)
    lim = np.percentile(np.abs(np.r_[Ic[:, gsel].ravel(), Fc[:, gsel].ravel()]), 98)
    pre, post = HL[k]
    Sa, Sb = closest_pair_any(k, pre, post)
    cz[f"{k}_pre"] = Ic[np.ix_(ro, gsel)]
    cz[f"{k}_post"] = Fc[np.ix_(ro, gsel)]
    cz[f"{k}_lim"] = np.array(lim)
    cz[f"{k}_block_ends"] = np.array([end for *_, end in labels[:-1]], float)
    cz[f"{k}_skel_pre"] = np.asarray(Sa, np.float64)
    cz[f"{k}_skel_post"] = np.asarray(Sb, np.float64)
    meta[k] = dict(row_types=[str(t) for t in types[ro]], tfs=[str(g) for g in genes_c[gsel]],
                   blocks=[str(b) for b, *_ in labels], pre_type=pre, post_type=post)
    print(k, pre, post, cz[f"{k}_pre"].shape, f"lim {lim:.3f}", "skel", Sa.shape, Sb.shape)
np.savez_compressed(OUT / "panelC.npz", **cz)
(OUT / "panelC_labels.json").write_text(json.dumps(meta, indent=1))
for f in sorted(OUT.iterdir()):
    print(f"{f.name:22s} {f.stat().st_size / 1e6:7.2f} MB")
