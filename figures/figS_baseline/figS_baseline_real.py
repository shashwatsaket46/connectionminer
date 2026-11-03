"""Real-matrix version of figS_baseline_schematic: how each step of the gene-based baseline (top: Scanpy units +
Connectome-Model ridge rule + connectome-guided matching) and of ConnectionMiner (bottom) reconstructs the held-out
types' connectome. Visual system, fold 0, test block (75 types), ConnectionMiner solver seed 0.

Inputs: biorender_fig/cache_baseline_real/{clusters,baseline,cm}.npz (written by figS_baseline_real_extract.py) and
the Adult build's global cell -> metacell map.
    python biorender_fig/figS_baseline_real.py  ->  figures/figS_baseline_real.{svg,pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.cluster.hierarchy import leaves_list, linkage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "data"
for f in (HERE.parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Liberation Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                     "font.size": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                     "xtick.major.size": 1.5, "ytick.major.size": 1.5, "xtick.major.pad": 1, "ytick.major.pad": 1,
                     "xtick.labelsize": 4.6, "ytick.labelsize": 4.6, "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold"})

INK, GREY, MUTED = "#222222", "#b8b8b8", "#555555"
RED, BASE_BG, CM_BG, SEP = "#C0392B", "#F4F4F4", "#EAF2FB", "#2b8cbe"
QUAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
HEAT = "magma_r"
rng = np.random.default_rng(0)

cl = np.load(CACHE / "clusters.npz", allow_pickle=True)      # truth is an object array (own cache)
bl = np.load(CACHE / "baseline.npz")
cm = np.load(CACHE / "cm.npz")
c2m = cl["c2m"]                                              # cell -> metacell (Adult hvg3000 cell_to_metacell.csv, by cell_idx)
# (the cluster-side check that split metacells match the global cell_to_metacell.csv ran in extract_data.py)

# ------------------------------------------------------------------------------------------------ frame + helpers
FW, RH, M0, GAP = 180 / 25.4, 4.45, 0.08, 0.075
FH = 2 * RH + 0.45
fig = plt.figure(figsize=(FW, FH))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, FW); bg.set_ylim(0, FH); bg.axis("off")
WS = np.array([1.12, 1.12, 1.78, 1.45, 1.27])
WS = WS * (FW - 2 * M0 - 0.1 - 4 * GAP) / WS.sum()
XS = M0 + 0.05 + np.concatenate([[0], np.cumsum(WS + GAP)[:-1]])
ROWS = {"base": FH - 0.06 - RH, "cm": FH - 0.06 - 2 * RH - 0.1}
BH = RH - 0.3


def ax_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


def box(x, y, w, h, title, cap, held=False):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white",
                                ec=RED if held else "#555555", lw=1.6 if held else 0.6))
    bg.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=7.5, weight="bold", color=INK)
    bg.text(x + w / 2, y + 0.07, cap, ha="center", va="bottom", fontsize=5.0, color=INK, linespacing=1.3)
    if held:
        bg.text(x + w / 2, y + h - 0.21, "uses held-out C", ha="center", va="top", fontsize=5.6, color=RED,
                weight="bold")


def arrow(x0, x1, y):
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#444444"))


def note(x, y, s, **kw):
    kw = {"ha": "center", "va": "top", "fontsize": 4.7, "color": MUTED, "linespacing": 1.25, **kw}
    bg.text(x, y, s, **kw)


def heat(ax, M, cmap=HEAT, vmin=None, vmax=None, xl="", yl="", title=None, interp="nearest"):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    if xl:
        ax.set_xlabel(xl, fontsize=5.0, labelpad=1.5, color=MUTED)
    if yl:
        ax.set_ylabel(yl, fontsize=5.0, labelpad=1.5, color=MUTED)
    if title:
        ax.set_title(title, fontsize=5.6, pad=2, color=INK)
    return im


def cbar(im, x, y, w, h, label, ticks=None, fmt=None, vertical=False):
    c = fig.colorbar(im, cax=ax_in(x, y, w, h), orientation="vertical" if vertical else "horizontal", ticks=ticks,
                     format=fmt)
    c.ax.tick_params(labelsize=4.3, length=1.2, pad=0.8); c.outline.set_linewidth(0.4)
    c.set_label(label, fontsize=4.7, labelpad=1, color=MUTED)
    return c


def lowhigh(im, x, y, w):
    cbar(im, x, y, w, 0.035, "", ticks=[])
    note(x, y - 0.03, "low", ha="left", fontsize=4.3); note(x + w, y - 0.03, "high", ha="right", fontsize=4.3)


CBIN = ListedColormap(["white", INK])


def binC(ax, C, W, xl="", yl="", title=None):
    cmap = CBIN.copy(); cmap.set_bad("#d0d0d0")
    return heat(ax, np.where(W, C, np.nan), cmap=cmap, vmin=0, vmax=1, xl=xl, yl=yl, title=title)


def gene_ticks(ax, genes, axis="both", right=False):
    n = len(genes)
    if axis in ("x", "both"):
        ax.set_xticks(range(n)); ax.set_xticklabels(genes, rotation=90, fontsize=4.1, fontstyle="italic")
    if axis in ("y", "both"):
        ax.set_yticks(range(n)); ax.set_yticklabels(genes, fontsize=3.9, fontstyle="italic")
        if right:
            ax.tick_params(axis="y", labelleft=False, labelright=True, left=False, right=True)


def emb(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", "datalim")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("t-SNE 1", fontsize=4.8, labelpad=1, color=MUTED)
    ax.set_ylabel("t-SNE 2", fontsize=4.8, labelpad=1, color=MUTED)


def small(ax, xl, yl, title):
    ax.tick_params(labelsize=4.2)
    ax.set_xlabel(xl, fontsize=4.8, labelpad=1, color=MUTED); ax.set_ylabel(yl, fontsize=4.8, labelpad=1, color=MUTED)
    ax.set_title(title, fontsize=5.6, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def type_sep(ax, n):
    ax.axhline(n - 0.5, color=SEP, lw=0.6); ax.axvline(n - 0.5, color=SEP, lw=0.6)


for key, title, bgc in (("base", "Gene-based baseline: Scanpy clusters + Connectome Model rule + connectome-guided "
                                 "matching", BASE_BG), ("cm", "ConnectionMiner", CM_BG)):
    y0 = ROWS[key]
    bg.add_patch(FancyBboxPatch((M0, y0), FW - 2 * M0, RH, boxstyle="round,pad=0,rounding_size=0.1", fc=bgc, ec="none"))
    bg.text(M0 + 0.08, y0 + RH - 0.05, title, ha="left", va="top", fontsize=8.5, weight="bold", color=INK)

# ------------------------------------------------------------------------------------------------ shared data
xy, ann, truth = cl["tsne"], cl["annotated"], cl["truth"]
sub = rng.choice(len(xy), 30000, replace=False)
names, free = bl["test_types"], bl["test_free"]
nt = len(names)
named_t = ~np.isin(np.arange(nt), free)
n_ann_t = int(named_t.sum())
Cte, Wte = bl["C"], bl["W"].astype(bool)
assert np.array_equal(Cte, ((np.nan_to_num(cm["test_C"]) > 0) & (cm["test_M"] > 0)).astype(float))
assert (cm["test_types"] == names).all() and (cm["test_named"] == named_t).all()
# one test-type order for both rows: annotated test types, then the rest by clustering of their true C rows + cols
un = np.where(~named_t)[0]
order = np.concatenate([np.where(named_t)[0],
                        un[leaves_list(linkage(np.hstack([Cte[un], Cte[:, un].T]), "average", metric="jaccard"))]])
SQ = WS[4] - 0.36                                             # step-5 matrix side


def step5(x, y0, w, Ch, label, r, sub_txt):
    ytop = y0 + BH
    a = ax_in(x + 0.22, ytop - 0.27 - SQ, SQ, SQ)
    im = heat(a, Ch, vmin=np.percentile(Ch, 1), vmax=np.percentile(Ch, 99), yl=f"{nt} test types (pre)", title=label)
    type_sep(a, n_ann_t)
    lowhigh(im, x + 0.22, ytop - 0.27 - SQ - 0.1, SQ)
    a = ax_in(x + 0.22, ytop - 0.27 - 2 * SQ - 0.4, SQ, SQ)
    binC(a, Cte[np.ix_(order, order)], Wte[np.ix_(order, order)], yl=f"{nt} test types (pre)",
         xl="post (same order)", title="C (held-out, true)")
    type_sep(a, n_ann_t)
    yr = ytop - 0.27 - 2 * SQ - 0.68
    bg.text(x + w / 2, yr, f"r = {r:.3f}", ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
    note(x + w / 2, yr - 0.2, sub_txt)


# ============================================================================================ BASELINE ROW
y0 = ROWS["base"] + 0.1
ytop = y0 + BH
types_ann = sorted(set(truth[ann]))
pal = {t: QUAL[i % len(QUAL)] for i, t in enumerate(types_ann)}
n_un = int((~ann).sum())
unit = cl["unit"]
n_units = int(unit.max()) + 1
n_clu = len(np.unique(cl["cluster"]))

# 1 cells -----------------------------------------------------------------------------------------------------
x, w = XS[0], WS[0]
side = w - 0.14
box(x, y0, w, BH, "1  Cells", f"{ann.sum():,} annotated cells keep\ntheir label ({len(pal)} types, coloured);\n"
    f"{n_un:,} unannotated cells\n(grey) have no cell type")
a = ax_in(x + 0.09, ytop - 0.24 - side, side, side)
s_ = sub[~ann[sub]]; a.scatter(*xy[s_].T, s=0.3, c=GREY, lw=0, rasterized=True)
s_ = sub[ann[sub]]; a.scatter(*xy[s_].T, s=0.3, c=[pal[t] for t in truth[s_]], lw=0, rasterized=True)
emb(a)
a = ax_in(x + 0.27, ytop - 0.24 - side - 0.95, w - 0.38, 0.62)
nc = pd.Series(truth[ann]).value_counts().to_numpy()
a.bar(np.arange(len(nc)), nc, width=1.0, color=QUAL[0], lw=0)
a.set_yscale("log"); a.set_xlim(-0.5, len(nc) - 0.5)
small(a, f"{len(nc)} annotated types (ranked)", "cells", "cells per annotated type")
note(x + w / 2, ytop - 0.24 - side - 1.3, f"(subsample of {len(sub):,} of\n{len(xy):,} cells shown)")

# 2 clustering ------------------------------------------------------------------------------------------------
x, w = XS[1], WS[1]
box(x, y0, w, BH, "2  Clustering", f"Scanpy Leiden (res {str(bl['res'])[1:]}): {n_clu}\nclusters; {n_units} with ≥20 "
    f"unannotated\ncells become units U0–U{n_units - 1}\n(cell type unknown)")
a = ax_in(x + 0.09, ytop - 0.24 - side, side, side)
upal = plt.get_cmap("tab20")(np.arange(20))
perm_u = rng.permutation(n_units)
s_ = sub[unit[sub] < 0]; a.scatter(*xy[s_].T, s=0.3, c="#e6e6e6", lw=0, rasterized=True)
s_ = sub[unit[sub] >= 0]; a.scatter(*xy[s_].T, s=0.3, c=upal[perm_u[unit[s_]] % 20], lw=0, rasterized=True)
emb(a)
am = bl["test_matched"]
cand0 = int(bl["n_ann_units"])
mu = am[free] - cand0
cnt = np.bincount(mu, minlength=n_units)
for u in np.argsort(-cnt)[:6]:
    a.text(*np.median(xy[unit == u], 0), f"U{u}", fontsize=4.0, ha="center", va="center", color=INK, weight="bold",
           bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
a = ax_in(x + 0.27, ytop - 0.24 - side - 0.95, w - 0.38, 0.62)
ncu = np.sort(bl["unit_ncells"])[::-1]
a.bar(np.arange(len(ncu)), ncu, width=1.0, color="#7f7f7f", lw=0)
a.set_yscale("log"); a.set_xlim(-0.5, len(ncu) - 0.5)
small(a, f"{len(ncu)} units (ranked)", "cells", "unannotated cells per unit")
note(x + w / 2, ytop - 0.24 - side - 1.3, "annotated cells light grey;\nlabels: the 6 units matched\nto most test "
     "types (step 4)")

# 3 profiles + rule -------------------------------------------------------------------------------------------
x, w = XS[2], WS[2]
lam = float(bl["lam_abs"])
box(x, y0, w, BH, "3  Profiles + gene rule", f"profile = mean log-expression of the type's cells\n(2,000 HVGs); ridge β "
    f"fitted on the {len(bl['Xtr'])} ANNOTATED\ntrain types only; λ = {lam:.0f} (= 10$^{{{int(np.log10(float(bl['lam'])))}}}$"
    f"·s$_{{max}}^4$, chosen on an\ninner 80/20 split of those types)")
Xtr, genes, Ctr, Wtr = bl["Xtr"], bl["genes"], bl["Ctr"], bl["Wtr"].astype(bool)
gsel = np.argsort(-Xtr.var(0))[:20]
to = leaves_list(linkage(Xtr[:, gsel], "average", metric="correlation"))
HX = 0.9
a = ax_in(x + 0.2, ytop - 0.37 - HX, w - 0.5, HX)
im = heat(a, Xtr[to][:, gsel], cmap="Blues", yl=f"{len(Xtr)} annotated\ntrain types",
          title="X$_{train}$: types × 20 top-variance genes")
gene_ticks(a, genes[gsel], "x")
cbar(im, x + w - 0.25, ytop - 0.37 - HX, 0.045, HX, "log expr.", vertical=True)
S3 = 0.62
yb = ytop - 0.37 - HX - 0.52 - S3
a = ax_in(x + 0.14, yb, S3, S3)
binC(a, Ctr[np.ix_(to, to)], Wtr[np.ix_(to, to)], xl="post", yl="pre", title="C among them")
gb = np.argsort(-bl["beta_score"])[:14]                       # beta_score = |B|.sum(0) + |B|.sum(1), full B on cluster
Bgb = bl["beta_top"]                                           # = B[np.ix_(gb, gb)]
v = float(np.abs(Bgb).max())
a = ax_in(x + 0.14 + S3 + 0.08, yb, S3, S3)
im = heat(a, Bgb, cmap="RdBu_r", vmin=-v, vmax=v, title="β (14 top-|β| genes)")
gene_ticks(a, genes[gb], right=True)
cbar(im, x + 0.14 + S3 + 0.08, yb - 0.5, S3, 0.035, "β (genes × genes)", ticks=[-0.006, 0, 0.006], fmt="%g")
bg.text(x + w / 2, yb - 0.8, "Ĉ(a→X) = x$_a$ β x$_X^{\\top}$", ha="center", va="center", fontsize=6.5, color=INK)

# 4 matching --------------------------------------------------------------------------------------------------
x, w = XS[3], WS[3]
uord = np.lexsort((np.arange(n_units), -cnt))
rank_u = np.empty(n_units, int); rank_u[uord] = np.arange(n_units)
tord = np.lexsort((np.arange(len(free)), rank_u[mu]))
D = bl["move_r"] - float(bl["r"])
used = int((cnt > 0).sum())
box(x, y0, w, BH, "4  Matching", f"each of the {len(free)} unannotated test types\ngets ONE unit, chosen to maximize r\nwith "
    f"the held-out C → {used} distinct units;\n{n_ann_t} annotated test types use their own profile", held=True)
HM = 1.7
a = ax_in(x + 0.2, ytop - 0.47 - HM, w - 0.28, HM)
lo = float(np.percentile(D, 2))
im = heat(a, D[np.ix_(tord, uord)], cmap="Greys_r", vmin=lo, vmax=0, xl=f"{n_units} units (by #types matched)",
          yl=f"{len(free)} unannotated test types", title="Δr if the type moved to that unit")
a.scatter(rank_u[mu[tord]], np.arange(len(free)), s=1.5, c=RED, lw=0)
yc = ytop - 0.47 - HM - 0.3
cbar(im, x + 0.2, yc, (w - 0.28) * 0.5, 0.035, "Δr vs matched", ticks=[-0.02, 0], fmt="%g")
note(x + 0.2 + (w - 0.28) * 0.56, yc + 0.03, "● matched unit", ha="left", va="center", color=RED, fontsize=4.7)
ylist = yc - 0.33
note(x + w / 2, ylist, f"types sharing one unit (max {cnt.max()}):", color=INK, fontsize=5.0)
for k, u in enumerate(uord[:6]):
    ts = names[free[mu == u]]
    txt = ", ".join(ts[:3]) + (f" +{len(ts) - 3}" if len(ts) > 3 else "")
    yy = ylist - 0.15 - 0.135 * k
    bg.text(x + 0.12, yy, f"U{u}", fontsize=4.8, color=RED, va="top", weight="bold")
    bg.text(x + 0.36, yy, f"← {txt}", fontsize=4.5, color=INK, va="top")

# 5 predict + score -------------------------------------------------------------------------------------------
x, w = XS[4], WS[4]
box(x, y0, w, BH, "5  Predict + score", f"Ĉ = x β xᵀ for every test pair\n(rows, columns: the same {nt}\ntest types)"
    f"; r(Ĉ, C) over\nall {Wte.sum():,} entries")
step5(x, y0, w, bl["Chat"][np.ix_(order, order)], "Ĉ (baseline)", float(bl["r"]),
      f"(no matching: r = {float(bl['r_unmatched']):.3f})")

# ============================================================================================ CONNECTIONMINER ROW
y0 = ROWS["cm"] + 0.1
ytop = y0 + BH
Ptr, Pte = cm["train_P"], cm["test_P"]
named_tr, named_te = cm["train_named"], cm["test_named"]
ntr_mc, nte_mc = Ptr.shape[1], Pte.shape[1]
n_mc = int(c2m.max()) + 1
LOGP = dict(cmap="Blues", vmin=-4, vmax=0)


def logP(P):
    return np.log10(np.maximum(P, 1e-12))


# 1 metacells -------------------------------------------------------------------------------------------------
x, w = XS[0], WS[0]
box(x, y0, w, BH, "1  Metacells", f"cells aggregated into {n_mc:,}\nmetacells (train block {ntr_mc:,},\ntest block "
    f"{nte_mc:,}); annotated\nmetacells pinned to their type")
a = ax_in(x + 0.09, ytop - 0.24 - side, side, side)
mperm = rng.permutation(n_mc)
a.scatter(*xy[sub].T, s=0.3, c=plt.get_cmap("tab20")(mperm[c2m[sub]] % 20), lw=0, rasterized=True)
emb(a)
a = ax_in(x + 0.27, ytop - 0.24 - side - 0.95, w - 0.38, 0.62)
sz = np.bincount(c2m)
mc_ann = np.bincount(c2m, weights=ann.astype(float)) / sz > 0.5
bins = np.arange(0, np.percentile(sz, 99.5) + 5, 5)
a.hist([sz[~mc_ann], sz[mc_ann]], bins=bins, stacked=True, color=["#8a8a8a", QUAL[0]], lw=0)
small(a, "cells per metacell", "metacells", "metacell sizes")
note(x + w / 2, ytop - 0.24 - side - 1.3, "colour = metacell (20 colours\nrecycled); blue bars: mostly-\nannotated "
     "metacells, grey: rest")

# 2 soft assignment P (train) ---------------------------------------------------------------------------------
x, w = XS[1], WS[1]
r_ = np.random.default_rng(1)
rows2 = np.concatenate([np.sort(r_.choice(np.where(named_tr)[0], 8, replace=False)),
                        np.sort(r_.choice(np.where(~named_tr)[0], 22, replace=False))])
na2 = int(named_tr[rows2].sum())
cols2 = np.unique(np.concatenate([np.argsort(-Ptr[i])[:6] for i in rows2]))
Q = Ptr[np.ix_(rows2, cols2)]
cols2 = cols2[np.lexsort((-Q.max(0), Q.argmax(0)))]
box(x, y0, w, BH, "2  Soft assignment P", f"each type's row of P spreads\nover its candidate metacells\n(train block: "
    f"{Ptr.shape[0]} types ×\n{ntr_mc:,} metacells; subset shown)")
HP = 1.75
a = ax_in(x + 0.3, ytop - 0.24 - HP, w - 0.36, HP)
im = heat(a, logP(Ptr[np.ix_(rows2, cols2)]), xl=f"{len(cols2)} metacells", title="P (train), 30 types", **LOGP)
a.axhline(na2 - 0.5, color=SEP, lw=0.6)
a.set_yticks(range(len(rows2))); a.set_yticklabels(cm["train_types"][rows2], fontsize=3.5)
for tl, nmd in zip(a.get_yticklabels(), named_tr[rows2]):
    tl.set_color(QUAL[0] if nmd else INK)
cbar(im, x + 0.3, ytop - 0.24 - HP - 0.33, w - 0.36, 0.035, "log$_{10}$ P", ticks=[-4, -2, 0])
a = ax_in(x + 0.3, ytop - 0.24 - HP - 1.12, w - 0.4, 0.5)
k = np.arange(1, 201)
for msk, col, ls, lab in ((~named_tr, INK, "-", "unannotated"), (named_tr, QUAL[0], "--", "annotated")):
    pm = np.cumsum(np.sort(Ptr[msk], 1)[:, ::-1][:, :200], 1)
    a.plot(k, np.median(pm, 0), color=col, lw=0.8, ls=ls, label=lab)
a.set_xscale("log"); a.set_ylim(0, 1.03)
small(a, "top-k metacells of a type", "P mass", "how soft is P? (median type)")
a.legend(fontsize=3.9, frameon=False, loc="lower right", handlelength=1.6)
note(x + w / 2, ytop - 0.24 - HP - 1.32, "blue names: annotated types; P\nfixed on their own metacells",
     color=QUAL[0])

# 3 E = P G and beta ------------------------------------------------------------------------------------------
x, w = XS[2], WS[2]
genes_cm = cm["genes"]
E = cm["train_E"]                                              # = train_P @ train_G (precomputed)
box(x, y0, w, BH, "3  Profiles + gene rule β", f"profile E = P G: metacell detection rates weighted\nby P; β learned "
    f"jointly with P on ALL {Ptr.shape[0]} train\ntypes ({named_tr.sum()} annotated), {E.shape[1]:,} genes; same "
    f"30 types\nas in 2")
gsel = np.argsort(-E.var(0))[:20]
a = ax_in(x + 0.2, ytop - 0.37 - HX, w - 0.5, HX)
im = heat(a, E[np.ix_(rows2, gsel)], cmap="Blues", yl="30 train types", title="E = P G: types × 20 top-variance genes")
gene_ticks(a, genes_cm[gsel], "x")
a.axhline(na2 - 0.5, color=SEP, lw=0.6)
cbar(im, x + w - 0.25, ytop - 0.37 - HX, 0.045, HX, "detection", vertical=True)
yb = ytop - 0.37 - HX - 0.52 - S3
a = ax_in(x + 0.14, yb, S3, S3)
Wtr_cm = (cm["train_M"] > 0) & ~np.isnan(cm["train_C"])
Ctr_cm = (np.nan_to_num(cm["train_C"]) > 0).astype(float)
binC(a, Ctr_cm[np.ix_(rows2, rows2)], Wtr_cm[np.ix_(rows2, rows2)], xl="post", yl="pre", title="C among them")
type_sep(a, na2)
gb = np.argsort(-cm["beta_score"])[:14]                       # beta_score = B.sum(0) + B.sum(1)
Bs = np.log10(cm["beta_top"] + 1e-16)
a = ax_in(x + 0.14 + S3 + 0.08, yb, S3, S3)
im = heat(a, Bs, cmap="viridis", vmin=-1, vmax=float(Bs.max()), title="train β (14 top genes)")
gene_ticks(a, genes_cm[gb], right=True)
cbar(im, x + 0.14 + S3 + 0.08, yb - 0.5, S3, 0.035, "log$_{10}$ β (β ≥ 0)", ticks=[0, 1, 2, 3])
bg.text(x + w / 2, yb - 0.8, "Ĉ = E β Eᵀ = P G β Gᵀ Pᵀ", ha="center", va="center", fontsize=6.5, color=INK)

# 4 test-block P ----------------------------------------------------------------------------------------------
x, w = XS[3], WS[3]
Q = Pte[order]
cols4 = np.lexsort((-Q.max(0), Q.argmax(0)))
effn = 1 / (Pte[~named_te] ** 2).sum(1)
box(x, y0, w, BH, "4  Test types", f"β frozen (train β); P solved for the\n{nt} test types on the held-out C;\nsoft: "
    f"many metacells per type\n(median effective #metacells {np.median(effn):.0f})", held=True)
a = ax_in(x + 0.2, ytop - 0.47 - HM, w - 0.28, HM)
im = heat(a, logP(Q[:, cols4]), xl=f"all {nte_mc:,} test metacells (by top type)",
          yl=f"{nt} test types (order as in 5)", title="P (test), β frozen", interp="antialiased", **LOGP)
a.axhline(n_ann_t - 0.5, color=SEP, lw=0.6)
cbar(im, x + 0.2, ytop - 0.47 - HM - 0.3, w - 0.28, 0.035, "log$_{10}$ P", ticks=[-4, -2, 0])
note(x + w / 2, ytop - 0.47 - HM - 0.62, f"first {n_ann_t} rows: annotated types, pinned\nto their own metacells.\n"
     f"Effective #metacells (1/Σ P²) of the {len(free)}\nunannotated test types: median {np.median(effn):.0f}\n"
     f"(range {effn.min():.0f}–{effn.max():.0f}; baseline: ONE unit per type)", color=INK)

# 5 predict + score -------------------------------------------------------------------------------------------
x, w = XS[4], WS[4]
box(x, y0, w, BH, "5  Predict + score", f"Ĉ = P G β Gᵀ Pᵀ on the test\ntypes (same {nt} types and\norder as above); "
    f"r(Ĉ, C) over\nall {Wte.sum():,} entries")
step5(x, y0, w, cm["test_Chat"][np.ix_(order, order)], "Ĉ (ConnectionMiner)", float(cm["r"]), "(solver seed 0)")

for key in ROWS:
    yy = ROWS[key] + 0.1 + BH / 2
    for i in range(4):
        arrow(XS[i] + WS[i] + 0.005, XS[i + 1] - 0.005, yy)
bg.text(FW / 2, 0.05, f"Visual system, fold 0, test block ({nt} types: {n_ann_t} annotated, {len(free)} unannotated); "
        "blue lines separate annotated (first) from unannotated types. Red frame: the step uses the held-out connectome.",
        ha="center", va="bottom", fontsize=5.0, color=MUTED)

out = HERE / "out" / "figS_baseline_real"
for ext in ("svg", "pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=400 if ext == "png" else 300, facecolor="white")
print("saved", out)
print(f"baseline r {float(bl['r']):.6f} (stored {float(bl['r_stored']):.6f}); CM r {float(cm['r']):.6f} "
      f"(results.csv {float(cm['r_results']):.6f})")
