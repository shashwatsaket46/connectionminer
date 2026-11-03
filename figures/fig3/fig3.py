#!/usr/bin/env python3
"""Figure 3 (portable): a pseudo-spatial transcriptome on the connectome, with one sister-type case study per system.

    python3 fig3.py            -> out/fig3.svg (editable, live text), out/fig3.pdf, out/fig3.png
    python3 fig3.py --list     -> print the types / pairs / shared genes available offline in data/

Dependencies: Python 3.9+, numpy, scipy, matplotlib, pandas, pillow. Nothing else; no cluster access needed.
Reads ONLY ./data/ (next to this file) and ../fonts/ (Liberation Sans). Run from any working directory.

Panels (183 x 214 mm, Nature Methods style):
  a  predicted expression (mean P.G over 50 solves) of the top-20 inferred types, one named marker gene per type;
     dot area = cells of the type detecting the gene; the sister pair's two rows are outlined in the pair colours
  b  top: the pair's cells in the scRNA-seq embedding (zoom on their cluster, whole embedding inset), coloured by which
     of the two types a cell is assigned to; bottom: predicted expression of type A vs type B over the named genes,
     with the separation statistics
  c  every EM neuron receives the predicted expression of its type: row 1 the same SHARED_GENES in all systems,
     rows 2-3 the three genes that best tell the pair apart, type A (row 2) or type B (row 3) drawn on top, outlined

WHAT CAN BE CHANGED OFFLINE (see the constants below)
  * PAIRS: any two of the "offline types" of a system (python3 fig3.py --list prints them; data/pairs_<sys>.csv lists
    every pair among them with the screen statistics: mass_ovl = cell-mass overlap, conn_r = wiring similarity,
    max_dpg = largest P.G difference). Offline types:
      visual  the top-20 inferred types + the next 100 most confident inferred types (120; e.g. Dm8a/Dm8b, Tm5a/b/d/e/f,
              Dm3p/q/v, LC14a1/a2/b, LC20a/b, MeTu*, LPi*, Li*), all with FlyWire skeletons
      cx      the top-20 + the next 100 (120; e.g. PEN_a/PEN1, PEN_b/PEN2, PFNa/d/m/p/v, FB2H_a/b, hDelta*, vDelta*,
              PFL1/2/3, ExR*), all with skeletons
      motor   every inferred type with >= 20 soft cells and a FANC skeleton (86; Fe*, Ti*, LN*), + 4 without skeleton
              (cannot be the drawn pair)
    A pair type outside the top 20 is added as an extra row of panel a (noted under the panel), as in the paper.
  * SHARED_GENES: any gene present in data/<sys>.npz of all three systems (--list prints the candidates).
  * titles, labels, colours, sizes: constants below; or edit out/fig3.svg directly (text is live).
NOT changeable offline: the solves themselves (P.G, cell probabilities), types outside the offline set, genes with
P.G < 0.02 in every offline type (they are never drawn by the current rules anyway).
Numbers not drawn but printed: pg_r (r of P.G over the named genes kept in data/, not all genes as on the cluster).
"""
import re, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DATA, FONTS, OUT = HERE / "data", HERE.parent / "fonts", HERE / "out"

# ====================================================================================================== EDIT HERE
SYS = ["visual", "cx", "motor"]                                     # row order
NAME = {"visual": "Visual", "cx": "Central complex", "motor": "Motor"}   # row labels (left margin)
PAIRS = {"visual": ("Dm8a", "Dm8b"),                                # sister pair drawn per system: (type A, type B)
         "cx": ("PEN_a/PEN1", "PEN_b/PEN2"),                        # must be offline types with skeletons (--list)
         "motor": ("Fe10", "Fe11")}
SHARED_GENES = ["hig", "kirre", "CanA-14F"]                         # panel c row 1: same genes in every system
COL = {"visual": "#0072B2", "cx": "#009E73", "motor": "#E69F00"}   # system colour = type A colour (Okabe-Ito)
COLB = "#CC79A7"                                                    # type B colour (all systems)
GREY = "#555555"                                                    # secondary text
DOT_CMAP, MAP_CMAP = "RdBu", "viridis"                              # panel a dots / panel c neuron colours
TITLE_A = "Predicted expression of inferred types"
TITLE_B = "Sister types: assigned cells"
TITLE_C = "Pseudo-spatial transcriptome on the connectome"
SUBTITLE_C = "every EM neuron receives its type's predicted expression"
LABEL_A_CBAR = "log$_2$ predicted expr. / 20-type mean"
LABEL_B_SCATTER = "predicted expression, named genes"
LABEL_B_KEY = "cell's share of the pair"
LABEL_C_ROW0 = "same genes\nin all systems"
LABEL_C_CBAR = "Predicted expression\n(fraction of cells detecting the gene)"
EXTRA_NOTE = {}       # optional grey note under a system's panel-b statistics, e.g. {"motor": "top |Δ|: ribosomal /\n..."}
ALSO = {}             # optional second pair reported (not drawn) under panel b, e.g. {"cx": ("PFNa", "PFNd")}
WMM, HMM = 183, 214                                                 # figure size, mm
FS, FT = 5, 6                                                       # base / title font size, pt
OUTNAME = "fig3"
# rules (as in the paper build)
NTOP, MIN_CELLS, PG_MIN = 20, 20, 0.3   # top-20 inferred types with >= 20 soft cells; marker / pair genes P.G >= 0.3
FLIPZ = {"visual": False, "cx": False, "motor": True}               # z down on the page (motor geometry pre-flipped)
# housekeeping genes (ribosomal, respiratory chain, glycolysis, ferritin, chaperones): never DISPLAYED as pair genes
HK = re.compile(r"^(Rp[LS]\d|RpLP|mRp|ND-|COX\d|Cox\d|UQCR|ATPsyn|Cyt-c|RFeSP|Chchd|sta$|Fer\dLCH|Fer\dHCH|Pglym|Tpi$|"
                r"Eno$|PyK$|Gapdh|Pgk$|Ald\d|SdhB|Mpcp|Hsc70|Hsp|eEF|Ef1|Tctp|Rack1|blw$|levy$|awd$|Srrm234$)")
# ==================================================================================================================


def load(k):
    """everything panel a-c need for system k, from data/<k>.npz (see extract_data.py for provenance)."""
    z = np.load(DATA / f"{k}.npz")
    types, contrib, conf = z["types"].astype(str), z["contrib"], z["conf"]
    cand = list(z["cand"]); crow = {t: r for r, t in enumerate(cand)}
    ok = (~z["type_named"]) & (contrib >= MIN_CELLS)
    top = [i for i in np.argsort(-conf) if ok[i]][:NTOP]
    A, B = PAIRS[k]
    for t in (A, B):
        if t not in types or list(types).index(t) not in crow:
            sys.exit(f"{k}: type {t!r} is not available offline; run  python3 fig3.py --list")
    ia, ib = list(types).index(A), list(types).index(B)
    added = [i for i in (ia, ib) if i not in top]
    top = top[:NTOP - len(added)] + added                          # pair types always shown
    genes, PG = z["genes"].astype(str), z["PG"]
    sk_types = list(z["skel_types"].astype(str))
    for t in (A, B):
        if t not in sk_types or not (z["skel_type"] == sk_types.index(t)).any():
            sys.exit(f"{k}: type {t!r} has no skeleton; pick another pair (python3 fig3.py --list)")
    return dict(types=types[top], top=top, genes=genes, PG=PG[top], PGall=PG, named=z["gene_named"],
                ia=ia, ib=ib, A=A, B=B, pA=z["cellP"][crow[ia]].astype(float), pB=z["cellP"][crow[ib]].astype(float),
                added=[types[i] for i in added], crow=crow, cellP=z["cellP"], alltypes=types, contrib=contrib,
                n_expr=z["n_expr"], xy=z["xy"].astype(float), kind=str(z["emb_kind"]), C=z["C"],
                segs=z["skel_segs"].astype(float), stype_seg=z["skel_type"], stypes=sk_types)


def conn_r(d, a, b):
    """wiring similarity: Pearson r of the two types' concatenated C row + column, log1p (fig3_sister_pairs.py)."""
    C = d["C"].astype(float)
    return float(np.corrcoef(np.log1p(np.r_[C[a], C[:, a]]), np.log1p(np.r_[C[b], C[:, b]]))[0, 1])


def stats(d, a=None, b=None):
    """separation of a pair: overlap of cell-probability mass, P.G difference, connectivity similarity."""
    a = d["ia"] if a is None else a; b = d["ib"] if b is None else b
    pa, pb = d["cellP"][d["crow"][a]].astype(float), d["cellP"][d["crow"][b]].astype(float)
    pa, pb = pa / pa.sum(), pb / pb.sum()
    PG, nm = d["PGall"], d["named"]
    D = PG[a] - PG[b]; ok = nm & (np.maximum(PG[a], PG[b]) >= PG_MIN)
    return dict(A=d["alltypes"][a], B=d["alltypes"][b], ovl=np.minimum(pa, pb).sum(), maxd=np.abs(D[ok]).max(),
                n02=int((np.abs(D[ok]) >= 0.2).sum()), n01=int((np.abs(D[ok]) >= 0.1).sum()),
                upA=(D[ok & (np.abs(D) >= 0.05)] > 0).mean() if (ok & (np.abs(D) >= 0.05)).any() else np.nan,
                pg_r=np.corrcoef(PG[a, nm], PG[b, nm])[0, 1], conn_r=conn_r(d, a, b),
                cells=(d["contrib"][a], d["contrib"][b]))


def pair_genes(d, n=3):
    """genes that tell A from B: largest |P.G_A - P.G_B|, named, not housekeeping, P.G >= PG_MIN in one of them;
    the best gene higher in A, the best higher in B, then the next largest |difference|."""
    PG = d["PGall"]; a, b = d["ia"], d["ib"]
    D = PG[a] - PG[b]
    ok = d["named"] & np.array([not HK.search(g) for g in d["genes"]]) & (np.maximum(PG[a], PG[b]) >= PG_MIN)
    o = [j for j in np.argsort(-np.abs(D)) if ok[j]]
    first = [j for j in (next((j for j in o if D[j] > 0), None), next((j for j in o if D[j] < 0), None)) if j is not None]
    pick = sorted(first, key=lambda j: -abs(D[j])) + [j for j in o if j not in first][:n - len(first)]
    return pick, D


def list_options():
    shared = None
    for k in SYS:
        z = np.load(DATA / f"{k}.npz")
        types, cand = z["types"].astype(str), z["cand"]
        st = z["skel_types"].astype(str)
        has = set(st[np.unique(z["skel_type"])])
        print(f"\n{k}: {len(cand)} offline types (* = no skeleton, cannot be drawn as the pair), most confident first")
        print("  " + ", ".join(types[i] + ("" if types[i] in has else "*") for i in cand))
        print(f"  all screened pairs among them: data/pairs_{k}.csv")
        g = set(z["genes"].astype(str)); shared = g if shared is None else shared & g
    print(f"\ngenes usable as SHARED_GENES (in all 3 systems): {len(shared)}, e.g.", ", ".join(sorted(shared)[:40]), "...")


if "--list" in sys.argv:
    list_options(); sys.exit()

for f in FONTS.glob("LiberationSans-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.5,
                     "ytick.major.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False})
MM = 1 / 25.4
TOPMM, ROWMM = 9, 68


def ax_mm(x, y, w, h):
    return fig.add_axes([x / WMM, 1 - (y + h) / HMM, w / WMM, h / HMM])


def text_mm(x, y, s, **kw):
    return fig.text(x / WMM, 1 - y / HMM, s, **kw)


def letter(x, y, s):
    text_mm(x, y, s, fontsize=8, weight="bold", va="top", ha="left")


fig = plt.figure(figsize=(WMM * MM, HMM * MM))
text_mm(21, 2.2, TITLE_A, fontsize=FT, weight="bold", va="top")
text_mm(67, 2.2, TITLE_B, fontsize=FT, weight="bold", va="top")
text_mm(106, 2.2, TITLE_C, fontsize=FT, weight="bold", va="top")
text_mm(106, 5.0, SUBTITLE_C, fontsize=FS, color=GREY, va="top")
letter(2, 1.5, "a"); letter(62, 1.5, "b"); letter(100, 1.5, "c")

for i, k in enumerate(SYS):
    d = load(k)
    s = stats(d)
    CA, CB = COL[k], COLB
    y0 = TOPMM + i * ROWMM
    rows_t = list(d["types"])
    ra, rb = rows_t.index(d["A"]), rows_t.index(d["B"])

    # ---- a: dot plot, one named marker gene per type
    cols = []
    for r in range(len(rows_t)):
        enr = np.log2((d["PG"][r] + 1e-4) / (np.delete(d["PG"], r, 0).mean(0) + 1e-4))
        enr[(d["PG"][r] < PG_MIN) | ~d["named"]] = -np.inf
        cols.append(next(j for j in np.argsort(-enr) if j not in cols and np.isfinite(enr[j])))
    M = d["PG"][:, cols]; M = np.log2((M + 1e-4) / (M.mean(0, keepdims=True) + 1e-4))
    lim = np.percentile(np.abs(M), 98)
    hx, hy, hw, hhm = 21, y0 + 2, 34, 40
    ax = ax_mm(hx, hy, hw, hhm)
    N = np.nan_to_num(d["n_expr"][np.ix_([d["crow"][t] for t in d["top"]], cols)])
    smax = max(N.max(), 1.0)
    area = lambda n: 0.4 + 9.0 * np.sqrt(n / smax)
    yy, xx = np.meshgrid(range(M.shape[0]), range(M.shape[1]), indexing="ij")
    ax.scatter(xx.ravel(), yy.ravel(), s=area(N.ravel()), c=M.ravel(), cmap=DOT_CMAP, vmin=-lim, vmax=lim, lw=0.15,
               edgecolors="#555555")
    ax.set_xlim(-0.6, M.shape[1] - 0.4); ax.set_ylim(M.shape[0] - 0.4, -0.6); ax.grid(False)
    ticks = [v for v in (10, 100, 1000) if v <= smax] or [round(smax)]
    for q, v in enumerate(ticks):
        text_mm(hx + hw + 2.6, hy + 3 + q * 2.6, f"{v:g}", fontsize=FS - 0.5, va="center")
        fig.add_artist(matplotlib.lines.Line2D([(hx + hw + 1.7) / WMM], [1 - (hy + 3 + q * 2.6) / HMM], marker="o",
                                               ms=np.sqrt(area(v)), mfc="#bbbbbb", mec="#555555", mew=0.15, ls="",
                                               transform=fig.transFigure))
    text_mm(hx + hw + 1.2, hy + 0.6, "cells", fontsize=FS - 0.5)
    ax.set_yticks(range(len(rows_t)), rows_t, fontsize=FS)
    ax.set_xticks(range(len(cols)), d["genes"][cols], rotation=90, fontsize=FS, style="italic")
    ax.tick_params(length=1, pad=0.8)
    for sp_ in ax.spines.values(): sp_.set_visible(True)
    for r_, c_ in ((ra, CA), (rb, CB)):
        ax.add_patch(Rectangle((-0.5, r_ - 0.5), len(cols), 1, fill=False, ec=c_, lw=0.9, clip_on=False))
    for t in ax.get_yticklabels():
        if t.get_text() in (d["A"], d["B"]):
            t.set_color(CA if t.get_text() == d["A"] else CB); t.set_fontweight("bold")
    text_mm(1.5, hy + hhm / 2, NAME[k], rotation=90, va="center", ha="center", fontsize=FT + 1, weight="bold",
            color=COL[k])
    cax = ax_mm(hx + hw + 1.2, hy + hhm * 0.4, 1.2, hhm * 0.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(-lim, lim), DOT_CMAP), cax=cax, ticks=[-lim, 0, lim],
                      format="%.1f")
    cb.ax.tick_params(labelsize=FS, length=1, pad=0.6); cb.outline.set_linewidth(0.4)
    cb.set_label(LABEL_A_CBAR, fontsize=FS - 0.5, labelpad=1)
    if d["added"]:
        text_mm(hx - 14, hy + hhm + 15.5, f"{' and '.join(d['added'])} added (not in the top 20)", fontsize=FS - 0.5,
                color=GREY, va="top")

    # ---- b (top): embedding zoom, colour = share of A in the pair
    xy, kind = d["xy"], d["kind"]
    pa, pb = d["pA"], d["pB"]
    w = pa + pb; f = pa / np.maximum(w, 1e-12)
    ex, ew, eh = 67, 23.5, 27
    core = xy[w >= 0.5 * w.max()]; med = np.median(core, 0); dist = np.abs(core - med).max(1)
    core = core[dist <= np.percentile(dist, 90) * 1.2]                       # the pair's main cluster
    lo, hi = core.min(0), core.max(0)
    c0, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.3
    half = np.array([half * ew / eh, half])
    eb = ax_mm(ex, hy, ew, eh); eb.set_xticks([]); eb.set_yticks([])
    for sp_ in eb.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.4); sp_.set_color("#999999")
    inview = (np.abs(xy - c0) <= half * 1.05).all(1)
    eb.scatter(*xy[inview].T, s=0.6, c="#e3e3e3", lw=0, rasterized=True)
    sel = (w > 0.05 * w.max()) & inview; o = np.argsort(w[sel])
    cmap = LinearSegmentedColormap.from_list("pair", [CB, "#bdbdbd", CA])
    eb.scatter(*xy[sel][o].T, s=0.5 + 3.5 * (w[sel][o] / w.max()), c=f[sel][o], cmap=cmap, vmin=0, vmax=1, lw=0,
               rasterized=True)
    eb.set_xlim(c0[0] - half[0], c0[0] + half[0]); eb.set_ylim(c0[1] - half[1], c0[1] + half[1])
    ins = ax_mm(ex + ew + 0.8, hy, 8, 8); ins.axis("off")                    # whole embedding, zoom box
    ins.scatter(*xy.T, s=0.02, c="#cfcfcf", lw=0, rasterized=True)
    ins.scatter(*xy[w >= 0.5 * w.max()].T, s=0.05, c="black", lw=0, rasterized=True)
    ins.add_patch(Rectangle(c0 - half, *(2 * half), fill=False, ec="black", lw=0.4))
    ins.set_aspect("equal", "datalim")
    text_mm(ex + ew + 0.8, hy + 8.6, "whole\nembedding", fontsize=FS - 1, color=GREY, va="top", linespacing=1.1)
    mass_in = (w * inview).sum() / w.sum()
    text_mm(ex + ew + 0.8, hy + 13, f"{100 * mass_in:.0f}% of\nthe pair's\ncell mass\nin zoom", fontsize=FS - 1,
            color=GREY, va="top", linespacing=1.1)
    text_mm(ex, hy + eh + 0.5, f"{kind} (zoom)", fontsize=FS, color="#666666", va="top")
    kx = ax_mm(ex + 10.5, hy + eh + 3.2, 13, 1.1)                         # A <-> B colour key
    kx.imshow(np.linspace(0, 1, 64)[None], cmap=cmap, aspect="auto", extent=(0, 1, 0, 1)); kx.set_xticks([])
    kx.set_yticks([]); [sp_.set_linewidth(0.3) for sp_ in kx.spines.values()]
    kx.text(-0.04, 0.5, d["B"], transform=kx.transAxes, fontsize=FS - 0.5, ha="right", va="center", color=CB,
            weight="bold")
    kx.text(1.04, 0.5, d["A"], transform=kx.transAxes, fontsize=FS - 0.5, ha="left", va="center", color=CA,
            weight="bold")
    kx.text(0.5, -0.5, LABEL_B_KEY, transform=kx.transAxes, fontsize=FS - 1, ha="center", va="top", color=GREY)

    # ---- b (bottom): predicted expression A vs B, named genes
    pick, D = pair_genes(d)
    sx, sy, sw, shh = ex + 4, hy + eh + 9.0, 16, 16
    sc = ax_mm(sx, sy, sw, shh)
    PA, PB = d["PGall"][d["ia"]], d["PGall"][d["ib"]]
    nm = d["named"] & (np.maximum(PA, PB) >= 0.02)
    sc.scatter(PA[nm], PB[nm], s=0.15, c="#9a9a9a", lw=0, rasterized=True)
    sc.plot([0, 1], [0, 1], color="black", lw=0.4, ls=(0, (2, 1.5)))
    nA = nB = 0
    for j in pick:
        if D[j] > 0: off = (0.97, 0.30 - 0.13 * nA); nA += 1          # labels in two columns, axes fraction
        else: off = (0.08, 0.95 - 0.13 * nB); nB += 1
        sc.scatter(PA[j], PB[j], s=4, c=CA if D[j] > 0 else CB, lw=0.3, edgecolors="black", zorder=3)
        sc.annotate(d["genes"][j], (PA[j], PB[j]), xytext=off, textcoords="axes fraction", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=GREY, shrinkA=0, shrinkB=1.5),
                    fontsize=FS - 0.5, style="italic", zorder=4, ha="right" if D[j] > 0 else "left")
    sc.set_xlim(0, 1); sc.set_ylim(0, 1)
    sc.set_xticks([0, 0.5, 1], ["0", "0.5", "1"]); sc.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    sc.tick_params(labelsize=FS - 0.5, length=1, pad=0.6)
    sc.set_xlabel(d["A"], fontsize=FS, color=CA, labelpad=0.5, weight="bold")
    sc.set_ylabel(d["B"], fontsize=FS, color=CB, labelpad=0.5, weight="bold")
    sc.set_title(LABEL_B_SCATTER, fontsize=FS - 0.5, pad=1.5, color=GREY)
    lines = [f"cells {s['cells'][0]:.0f} / {s['cells'][1]:.0f}", f"cell overlap {s['ovl']:.2f}",
             f"max |Δ| {s['maxd']:.2f}", f"wiring r {s['conn_r']:.2f}",
             (f"|Δ|≥0.05 higher in\n{d['A']}: {100 * s['upA']:.0f}%" if s['upA'] == s['upA'] else "no gene with |Δ|≥0.05")]
    text_mm(sx + sw + 1.0, sy + 0.3, "\n".join(lines), fontsize=FS - 0.5, va="top", linespacing=1.25)
    if k in ALSO:
        t = list(d["alltypes"]); s2 = stats(d, t.index(ALSO[k][0]), t.index(ALSO[k][1]))
        text_mm(sx + sw + 1.0, sy + 12.6, f"{ALSO[k][0]}/{ALSO[k][1]}:\noverlap {s2['ovl']:.2f}\nmax |Δ| {s2['maxd']:.2f}\n"
                f"wiring r {s2['conn_r']:.2f}\n(not separated)", fontsize=FS - 1, va="top", color=GREY,
                linespacing=1.2)
    if k in EXTRA_NOTE:
        text_mm(sx + sw + 1.0, sy + 12.6, EXTRA_NOTE[k], fontsize=FS - 1, va="top", color=GREY, linespacing=1.2)

    # ---- c: gene maps
    segs, stype = d["segs"], d["stype_seg"]
    selA, selB = stype == d["stypes"].index(d["A"]), stype == d["stypes"].index(d["B"])
    # P.G per skeleton type (skeleton type order may differ from the build's)
    tpos = {t: q for q, t in enumerate(d["alltypes"])}
    smap = np.array([tpos.get(t, -1) for t in d["stypes"]])
    pts = segs.reshape(-1, 2)
    q_ = 0.5 if k == "motor" else 0.02
    lo_, hi_ = np.percentile(pts, q_, 0), np.percentile(pts, 100 - q_, 0)
    mx, mw, mg, mh = 106, 21.0, 0.9, 17.0
    hm = min(mh, mw * (hi_[1] - lo_[1]) / (hi_[0] - lo_[0]))
    TG = 4.2
    blockh = 3 * hm + 2 * TG
    top = y0 + 3.5 + (3 * mh + 2 * TG - blockh) / 2
    rows_ = [(list(SHARED_GENES), [(selA, CA), (selB, CB)], LABEL_C_ROW0, GREY),
             ([d["genes"][j] for j in pick], [(selA, CA)], d["A"], CA),
             ([d["genes"][j] for j in pick], [(selB, CB)], d["B"], CB)]
    glist = list(d["genes"])
    for r_, (genes_, hls, lab, lc) in enumerate(rows_):
        my = top + r_ * (hm + TG)
        text_mm(mx - 2.6, my + hm / 2, lab, rotation=90, va="center", ha="center", fontsize=FS, color=lc,
                weight="normal" if r_ == 0 else "bold")
        for g_, G in enumerate(genes_):
            if G not in glist:
                sys.exit(f"{k}: gene {G!r} not in data/{k}.npz (python3 fig3.py --list shows usable shared genes)")
            gi = glist.index(G)
            v = np.append(d["PGall"][:, gi], np.nan)[smap]                 # per skeleton type; NaN = type not in build
            ma = ax_mm(mx + g_ * (mw + mg), my, mw, hm); ma.axis("off")
            sv = v[stype]; o = np.argsort(sv, kind="stable")
            ma.add_collection(LineCollection(segs[o], array=sv[o], cmap=MAP_CMAP, norm=plt.Normalize(0, 1),
                                             linewidths=0.08, alpha=0.6, rasterized=True))
            for hsel, hc in hls:
                ma.add_collection(LineCollection(segs[hsel], colors=hc, linewidths=0.75, rasterized=True))
                ma.add_collection(LineCollection(segs[hsel], array=np.full(hsel.sum(), v[stype[hsel][0]]),
                                                 cmap=MAP_CMAP, norm=plt.Normalize(0, 1), linewidths=0.3,
                                                 rasterized=True))
            ma.set_xlim(lo_[0], hi_[0]); ma.set_ylim(lo_[1], hi_[1]) if FLIPZ[k] else ma.set_ylim(hi_[1], lo_[1])
            ma.set_aspect("equal", "box", anchor="S")
            if r_ == 0:
                ma.set_title(G, fontsize=FS + 0.5, style="italic", pad=1)
            elif r_ == 1:
                ma.set_title(f"{G}", fontsize=FS + 0.5, style="italic", pad=1, loc="left")
                ma.text(1.0, 1.0, f"{PA[gi]:.2f} vs {PB[gi]:.2f}", transform=ma.transAxes, fontsize=FS - 1,
                        ha="right", va="bottom", color=GREY)
            if i == 0 and g_ == 0 and r_ == 2:
                ma.annotate("", (0.14, 0.0), (0.0, 0.0), "axes fraction", arrowprops=dict(arrowstyle="->", lw=0.4))
                ma.annotate("", (0.0, -0.25), (0.0, 0.0), "axes fraction", arrowprops=dict(arrowstyle="->", lw=0.4))
                ma.text(0.16, 0.0, "x", transform=ma.transAxes, fontsize=FS, va="center")
                ma.text(0.03, -0.27, "z", transform=ma.transAxes, fontsize=FS, ha="left", va="top")
    cb_h = max(blockh * 0.8, 14)
    cax = ax_mm(mx + 3 * (mw + mg) + 0.4, top + (blockh - cb_h) / 2, 1.2, cb_h)
    cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 1), MAP_CMAP), cax=cax, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=FS, length=1, pad=0.6); cb.outline.set_linewidth(0.4)
    cb.set_label(LABEL_C_CBAR, fontsize=FS - 0.5, labelpad=1.5)
    print(k, s["A"], s["B"], f"ovl {s['ovl']:.3f} maxd {s['maxd']:.3f} n>=0.1 {s['n01']} n>=0.2 {s['n02']} "
          f"pg_r {s['pg_r']:.4f} conn_r {s['conn_r']:.3f}", "genes", [d["genes"][j] for j in pick])

OUT.mkdir(exist_ok=True)
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"{OUTNAME}.{ext}", dpi=600, facecolor="white")
plt.close(fig)
print("saved", OUT / OUTNAME, "(.svg .pdf .png)", f"{WMM} x {HMM} mm")
