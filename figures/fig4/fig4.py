"""Figure 4 (portable build): gene-pair evidence behind the connectome model.

    python fig4.py            ->  out/fig4.svg (editable, live text), out/fig4.pdf, out/fig4.png

Reads ONLY ./data/ (made on the cluster by extract_data.py) and ../fonts/ (Liberation Sans, metric-compatible
with Arial). Runs from any working directory. Dependencies: numpy, scipy, matplotlib, pandas, pillow.

a  Volcano per system: every TESTED directed gene pair. x = 10-fold mean log2(1+beta) - log2(1+beta_null)
   (real solves vs refit null), y = -log10 paired-t P over the folds. Significant = Benjamini-Hochberg FDR over all
   tested pairs of the system. Pairs whose two genes both belong to a gene family are coloured by family class.
   Combined = Fisher's method on one-sided P over the systems holding both genes (named genes, >= 2 systems).
b  R^2(k): squared Pearson r between C and C_hat restricted to the global top-k gene pairs of each train solve;
   median and IQR over 50 train solves; dashed = the full model.
c  Share of C_hat per family gene pair (columns) on synaptic type pairs (rows). Columns = the best-share supported
   pair (volcano x > 0, P < SUPPORT_P) of every family class, then the remaining supported pairs by mean share,
   filled up by share. Rows = top total share, at most ROWS_PER_COLUMN per dominant column; rows and columns
   biclustered (cosine, average linkage, optimal leaf order). Colour strip = family class; marks = volcano evidence.

Everything you are likely to want to edit is in the CONSTANTS block below.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # silence missing-cursive-font chatter
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FONTS = HERE.parent / "fonts"
OUT = HERE / "out"

# ===================================================================================================== CONSTANTS
# --- statistics ------------------------------------------------------------------------------------------------
FDR = 0.05                 # BH false-discovery rate for "significant" (volcano points, threshold line, * marks)
SUPPORT_P = 0.05           # heatmap: a column is "supported" if volcano x > 0 and P < SUPPORT_P (filled dot)
                           # (the header number "N up in volcano (P < 0.05)" is pre-computed at 0.05)
LABEL_MAX_P = 0.01         # volcano: only label pairs with x > 0 and P below this

# --- how much to show ------------------------------------------------------------------------------------------
N_LABELS = 3               # number of labelled top pairs per volcano (named genes, x > 0, P < LABEL_MAX_P)
N_COLUMNS = 20             # heatmap gene-pair columns per system (max: visual 200, motor 177, CX 200 candidates)
N_ROWS = 20                # heatmap synaptic-type-pair rows per system
ROWS_PER_COLUMN = 4        # at most this many rows may have the same dominant column

# --- systems: (data key, panel title, title colour, volcano subtitle) ------------------------------------------
SYSTEMS = [
    ("visual", "Visual system", "#3B5BA9", "{g:,} solver genes"),
    ("motor", "Motor system", "#D9822B", "{h:,} genes + {h:,} complements"),
    ("cx", "Central complex", "#3E9A57", "{g:,} genes (CX reference build)"),
]
FISHER_TITLE = "Combined (Fisher)"
FISHER_COLOUR = "#333333"
FISHER_SUBTITLE = "named genes in ≥ 2 systems\n{n:,} pairs"

# --- family classes: colour and legend name ---------------------------------------------------------------------
FAMILY_COLOURS = {"TF–TF": "#7B3294", "CAM–CAM": "#1B9E77", "NP–NPR": "#E7298A", "cross": "#E6AB02"}
FAMILY_NAMES = {"TF–TF": "TF–TF", "CAM–CAM": "CAM–CAM", "NP–NPR": "Neuropeptide–receptor", "cross": "Cross-family"}

# --- other colours ----------------------------------------------------------------------------------------------
OTHER_PAIR_COLOUR = "#d4d4d4"      # volcano: pairs not in a family
DARK = "#3A3A3A"                   # secondary text, marks
BAR_COLOUR = "#7A7A7A"             # heatmap bar plot (mean share)
HEAT_CMAP = LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#C6D6EA", "#3775BA", "#0F4D92", "#06213F"])
HEAT_GAMMA, HEAT_MAX = 0.5, 80     # colour scale: PowerNorm(gamma, 0, max %) ; colour-bar ticks below
HEAT_TICKS = [0, 5, 20, 40, 80]

# --- titles / axis labels ---------------------------------------------------------------------------------------
VOLCANO_YLABEL = r"$-\log_{10}$ $P$ (paired t, 10 folds)"
VOLCANO_XLABEL = (r"$\beta$ effect: 10-fold mean $\log_2(1+\beta) - \log_2(1+\beta_{\mathrm{null}})$, "
                  "every tested pair")
R2_TITLE = r"Variance of $C$ explained"
R2_SUBTITLE = "median and IQR over 50 train solves"
R2_XLABEL = "Top-k gene pairs (global, per solve)"
R2_YLABEL = r"Variance of $C$ explained, $R^2$ (train)"
R2_YMAX = 0.56
HEAT_YLABEL = "Synaptic type pair (pre → post)"
HEAT_BAR_LABEL = r"mean share of $\hat{C}$" + "\nover synaptic\ntype pairs (%)"
HEAT_CBAR_LABEL = r"Share of $\hat{C}$ (%)"
HEAT_FOOTER = ("Family gene pair (presynaptic-type gene–postsynaptic-type gene)   ·   marks: * BH q < {fdr:g}   "
               "● volcano x > 0, P < {sp:g}   ○ tested, not supported   – not tested in the volcano build")
FAMILY_LEGEND_TITLE = "Family class (colour strip)"

# --- fonts / output ---------------------------------------------------------------------------------------------
FONT_FAMILY = "Liberation Sans"    # bundled in ../fonts; put "Arial" first in style() if you prefer Arial
PNG_DPI, PDF_DPI = 600, 300
FIG_WIDTH_MM = 180

# gene names treated as anonymous loci (never used as labels / heatmap columns)
ANON = re.compile(r"\d{4,}|^(CG|CR)\d|:CR\d|^FBti|^~")
# ================================================================================================================

MM = 1 / 25.4


def style():
    for f in FONTS.glob("LiberationSans-*.ttf"):
        fm.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.labelpad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY, "mathtext.it": f"{FONT_FAMILY}:italic", "mathtext.bf": f"{FONT_FAMILY}:bold",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def bh(p):
    """Benjamini-Hochberg adjusted P values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.clip(q, 0, 1)
    return out


def letter(fig, ax, s, dx=-6, dy=1.5):
    bb = ax.get_position()
    fig.text(bb.x0 + dx * MM / fig.get_figwidth(), bb.y1 + dy * MM / fig.get_figheight(), s,
             fontsize=8, fontweight="bold", va="bottom", ha="left")


def leaf_order(X):
    D = np.nan_to_num(pdist(X, "cosine"), nan=1.0)
    return leaves_list(optimal_leaf_ordering(linkage(D, "average"), D))


def icons():
    """Split the BioRender icon row into its three CNS drawings at the blank columns between them."""
    a = np.asarray(Image.open(DATA / "cns_icons.png").convert("RGB")).astype(int)
    ink = a.min(2) < 235
    cols = ink.any(0)
    runs, x = [], 0
    while x < len(cols):
        if cols[x]:
            x0 = x
            while x < len(cols) and cols[x]:
                x += 1
            runs.append((x0, x))
        x += 1
    runs = sorted(runs, key=lambda r: r[0] - r[1])[:3]     # the three widest ink runs, left to right
    out = []
    for x0, x1 in sorted(runs):
        ys = np.where(ink[:, x0:x1].any(1))[0]
        sub = a[ys.min():ys.max() + 1, x0:x1]
        out.append(np.where((sub.min(2) > 235)[..., None], 255, sub).astype(np.uint8))
    return out


# ------------------------------------------------------------------------------------------------------ panel a
def volcano(ax, key, name, col, subtitle):
    meta = json.loads((DATA / f"volcano_{key}_meta.json").read_text())
    d = pd.read_csv(DATA / f"volcano_{key}.csv.gz", keep_default_na=False)
    cls = d.family_class.to_numpy(str)
    isf = cls != ""
    p = d.p.to_numpy(float)
    y = -np.log10(p)
    x = d.effect.to_numpy(float)
    sig = bh(p) < FDR
    ax.scatter(x[~isf & ~sig], y[~isf & ~sig], s=1.2, c=OTHER_PAIR_COLOUR, lw=0, rasterized=True, zorder=1)
    for c, cc in FAMILY_COLOURS.items():
        m = (cls == c) & ~sig
        ax.scatter(x[m], y[m], s=2.5, c=cc, lw=0, alpha=0.85, rasterized=True, zorder=2)
    fc = [FAMILY_COLOURS[c] if c else col for c in cls[sig]]
    ax.scatter(x[sig], y[sig], s=9, c=fc, lw=0.35, edgecolors="black", zorder=4)
    nsig = int(sig.sum())
    thr = -np.log10(p[sig].max()) if nsig else -np.log10(FDR / len(d))   # none pass: BH's first step FDR / m
    ax.axhline(thr, ls=(0, (1.5, 1.5)), lw=0.5, c="#555555", zorder=0)
    ax.axvline(0, c="#e0e0e0", lw=0.5, zorder=0)
    ok = d.named.to_numpy(bool)
    sel = np.flatnonzero(ok & (x > 0) & (p < LABEL_MAX_P))
    for k, t in enumerate(sel[np.argsort(-y[sel])][:N_LABELS]):
        ax.annotate(f"{d.gene_i.iloc[t]}–{d.gene_j.iloc[t]}", (x[t], y[t]), (0.04, 0.62 - 0.085 * k),
                    textcoords="axes fraction", fontsize=5, style="italic", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="#9a9a9a", shrinkA=1, shrinkB=1.5))
    ax.set_ylim(-0.2, max(y.max(), thr) * 1.08)
    ax.set_title(name, color=col, fontsize=7, weight="bold", loc="left", pad=13)
    g = meta["n_genes"]
    ax.text(0, 1.015, f"{subtitle.format(g=g, h=g // 2)}\n{meta['tested']:,} pairs tested", transform=ax.transAxes,
            fontsize=4.8, color=DARK, va="bottom", linespacing=1.15)
    ax.text(0.98, 0.04, f"{nsig} significant", transform=ax.transAxes, ha="right", fontsize=5.5, color=col)
    up, dn = int((sig & (x > 0)).sum()), int((sig & (x < 0)).sum())
    if FDR == 0.05:                       # cross-check against the counts of the source build
        assert (up, dn) == (meta["bh_q05_up"], meta["bh_q05_down"]), (key, up, dn, meta)
    print(f"{key}: {len(d):,} tested, {nsig} BH-significant ({up} up / {dn} down), "
          f"family among significant: {list(zip(d.gene_i[sig], d.gene_j[sig], cls[sig]))}")


def fisher_panel(ax):
    c = pd.read_csv(DATA / "fisher_combined.csv.gz", keep_default_na=False)
    p = c.p.to_numpy(float)
    x = c.effect.to_numpy(float)
    y = -np.log10(p)
    sig = bh(p) < FDR
    ax.scatter(x[~sig], y[~sig], s=1.5, c=OTHER_PAIR_COLOUR, lw=0, rasterized=True)
    ax.scatter(x[sig], y[sig], s=7, c=FISHER_COLOUR, lw=0.25, edgecolors="black", zorder=3)
    nsig = int(sig.sum())
    thr = -np.log10(p[sig].max()) if nsig else -np.log10(FDR / len(c))
    ax.axhline(thr, ls=(0, (1.5, 1.5)), lw=0.5, c="#555555", zorder=1)
    ax.axvline(0, c="#e0e0e0", lw=0.5, zorder=0)
    top = c[(x > 0) & (p < LABEL_MAX_P)].nsmallest(N_LABELS, "p")
    for k, (_, r) in enumerate(top.iterrows()):
        ax.annotate(f"{r.gene_i}–{r.gene_j}", (r.effect, -np.log10(r.p)), (0.04, 0.82 - 0.09 * k),
                    textcoords="axes fraction", fontsize=5, style="italic", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="#9a9a9a", shrinkA=1, shrinkB=1.5))
    ax.set_ylim(-0.2, max(y.max(), thr) * 1.08)
    ax.text(0.98, 0.04, f"{nsig} significant", transform=ax.transAxes, ha="right", fontsize=5.5, color=FISHER_COLOUR)
    ax.set_title(FISHER_TITLE, color=FISHER_COLOUR, fontsize=7, weight="bold", loc="left", pad=13)
    ax.text(0, 1.015, FISHER_SUBTITLE.format(n=len(c)), transform=ax.transAxes, fontsize=4.8, color=DARK,
            va="bottom", linespacing=1.15)
    print(f"fisher: {len(c):,} pairs, {nsig} BH-significant")


# ------------------------------------------------------------------------------------------------------ panel b
def r2_panel(ax):
    for key, name, col, _ in SYSTEMS:
        z = np.load(DATA / f"r2k_{key}.npz")
        k, r2, full = z["k"], z["r2"], z["r2_full"]
        lo, med, hi = np.percentile(r2, [25, 50, 75], axis=0)
        ax.fill_between(k, lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(k, med, color=col, lw=0.9, label=name.replace(" system", ""))
        ax.axhline(np.median(full), color=col, lw=0.6, ls=(0, (2.5, 1.5)))
        print(f"{key}: R2_full median {np.median(full):.3f}; median R2(10) {med[list(k).index(10)]:.3f}")
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0, R2_YMAX)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(R2_XLABEL)
    ax.set_ylabel(R2_YLABEL)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="#555555", lw=0.6, ls=(0, (2.5, 1.5))))
    l.append("Full model")
    ax.legend(h, l, loc="center left", bbox_to_anchor=(0.0, 0.52), handlelength=1.4, fontsize=5.2, borderaxespad=0.2)
    ax.set_title(R2_TITLE, loc="left", fontsize=7, fontweight="bold", pad=13)
    ax.text(0, 1.015, R2_SUBTITLE, transform=ax.transAxes, fontsize=4.8, color=DARK, va="bottom")


# ------------------------------------------------------------------------------------------------------ panel c
def cls_group(c):
    return c if c in ("TF–TF", "CAM–CAM", "NP–NPR") else "cross"


def pick_columns(d):
    cls = np.array([cls_group(c) for c in d["col_class"]])
    sup = (d["col_vdiff"] > 0) & (d["col_vp"] < SUPPORT_P)
    ok = np.array([not (ANON.search(str(a)) or ANON.search(str(b))) for a, b in zip(d["col_pre"], d["col_post"])])
    ms = d["col_mean_share"]
    cols = []
    for c in FAMILY_COLOURS:                             # the best supported pair of every class present
        m = np.flatnonzero(sup & ok & (cls == c))
        if len(m):
            cols.append(m[np.argmax(ms[m])])
    for idx in (np.flatnonzero(sup & ok), np.flatnonzero(~sup & ok)):
        for j in idx[np.argsort(-ms[idx], kind="stable")]:
            if len(cols) == N_COLUMNS:
                break
            if j not in cols:
                cols.append(j)
    return np.array(cols), cls, sup


def heat(ax, bar, strip, key, name, col, first, cax):
    d = dict(np.load(DATA / f"heat_{key}.npz"))
    cols, cls, sup = pick_columns(d)
    share = d["share"][:, cols]
    nc = len(cols)
    arg, rows, cnt = share.argmax(1), [], np.zeros(nc, int)
    for r in np.argsort(-share.sum(1), kind="stable"):
        if cnt[arg[r]] < ROWS_PER_COLUMN:
            rows.append(r)
            cnt[arg[r]] += 1
        if len(rows) == N_ROWS:
            break
    rows = np.array(rows)
    ro, co = leaf_order(share[rows]), leaf_order(share[rows].T)
    rows = rows[ro]
    cj = cols[co]
    M = share[np.ix_(rows, co)] * 100
    im = ax.imshow(M, aspect="auto", cmap=HEAT_CMAP, norm=PowerNorm(HEAT_GAMMA, 0, HEAT_MAX), interpolation="nearest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{d['row_pre'][r]} → {d['row_post'][r]}" for r in rows], fontsize=4.8)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{d['col_pre'][j]}–{d['col_post'][j]}" for j in cj], rotation=90, fontsize=5, style="italic")
    ax.tick_params(length=0, pad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    if first:
        ax.set_ylabel(HEAT_YLABEL)
    strip.imshow([[to_rgb(FAMILY_COLOURS[cls[j]]) for j in cj]], aspect="auto", interpolation="nearest")
    strip.set_axis_off()
    ms = d["col_mean_share"][cj] * 100
    bar.bar(range(nc), ms, width=0.78, color=BAR_COLOUR, lw=0)
    bar.set_xlim(-0.5, nc - 0.5)
    bar.set_xticks([])
    bar.spines["bottom"].set_visible(False)
    top = ms.max() * 1.45
    bar.set_ylim(0, top)
    bar.set_yticks([0, float(f"{ms.max():.2g}")])
    bar.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for k, j in enumerate(cj):
        q, p = d["col_vq"][j], d["col_vp"][j]
        yk = ms[k] + top * 0.09
        if q < FDR:
            bar.scatter([k], [yk], s=9, marker="*", c=DARK, lw=0, clip_on=False)
        elif sup[j]:
            bar.scatter([k], [yk], s=3, marker="o", c=DARK, lw=0, clip_on=False)
        elif np.isfinite(p):
            bar.scatter([k], [yk], s=3, marker="o", facecolors="white", edgecolors=DARK, lw=0.35, clip_on=False)
        else:
            bar.plot([k - 0.25, k + 0.25], [yk, yk], c=DARK, lw=0.4, clip_on=False)
    if first:
        bar.text(-0.2, 0.5, HEAT_BAR_LABEL, transform=bar.transAxes, ha="right", va="center", fontsize=5,
                 linespacing=1.1)
    bar.set_title(name, loc="left", fontsize=7, fontweight="bold", pad=15, color=col)
    bar.text(0, 1.04, f"{int(d['n_fam_genes'])} family genes · {int(d['n_fam_pairs']):,} family pairs\n"
                      f"{float(d['fam_total_share']):.1%} of $\\hat{{C}}$ · {int(d['n_fam_up'])} up in volcano (P < 0.05)",
             transform=bar.transAxes, fontsize=5, color=DARK, va="bottom", linespacing=1.1)
    if cax is not None:
        cb = plt.colorbar(im, cax=cax, orientation="vertical", ticks=HEAT_TICKS)
        cb.set_label(HEAT_CBAR_LABEL, fontsize=5.5, labelpad=2)
        cb.ax.tick_params(labelsize=5, length=1.5, width=0.4)
        cb.outline.set_linewidth(0.4)


# ------------------------------------------------------------------------------------------------------- layout
def main():
    style()
    W = FIG_WIDTH_MM
    vy, vh = 13, 36                                    # all positions in mm from the top-left corner
    hy, bh, sh, hh = vy + vh + 30, 8, 1.6, 35
    H = hy + bh + 1 + sh + 0.6 + hh + 27
    fig = plt.figure(figsize=(W * MM, H * MM))

    def box(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    vx0, vw, vsp = 11, 24, 7
    vaxes = [box(vx0 + k * (vw + vsp), vy, vw, vh) for k in range(4)]
    for ax, (key, name, col, sub) in zip(vaxes, SYSTEMS):
        volcano(ax, key, name, col, sub)
    fisher_panel(vaxes[3])
    vaxes[0].set_ylabel(VOLCANO_YLABEL)
    fig.text((vx0 + 2 * vw + 1.5 * vsp) / W, 1 - (vy + vh + 5) / H, VOLCANO_XLABEL, ha="center", va="top", fontsize=6)
    hand = [Line2D([], [], ls="", marker="o", ms=2, c=OTHER_PAIR_COLOUR)] + \
           [Line2D([], [], ls="", marker="o", ms=2.4, c=c) for c in FAMILY_COLOURS.values()] + \
           [Line2D([], [], ls="", marker="o", ms=3, mfc="white", mec="black", mew=0.4),
            Line2D([], [], ls=(0, (1.5, 1.5)), lw=0.6, c="#555555")]
    labs = ["Other pair"] + [FAMILY_NAMES[c] for c in FAMILY_COLOURS] + \
           [f"BH FDR {FDR * 100:g} % (black edge)", "BH threshold"]
    fig.legend(hand, labs, loc="upper center", bbox_to_anchor=((vx0 + 2 * vw + 1.5 * vsp) / W, 1 - (vy + vh + 9.5) / H),
               ncol=7, fontsize=5.2, handletextpad=0.3, columnspacing=1.0, frameon=False)
    letter(fig, vaxes[0], "a", dx=-11, dy=7)

    cx = vx0 + 4 * vw + 3 * vsp + 14
    axb = box(cx, vy, W - 2 - cx, vh)
    r2_panel(axb)
    letter(fig, axb, "b", dx=-11, dy=7)

    hw, hx0, hsp = 35, 21, 21
    ims = icons()
    for k, (key, name, col, _) in enumerate(SYSTEMS):
        x = hx0 + k * (hw + hsp)
        bar = box(x, hy, hw, bh)
        strip = box(x, hy + bh + 0.5, hw, sh)
        ax = box(x, hy + bh + 0.5 + sh + 0.6, hw, hh)
        cax = box(x + hw + 1.5, hy + bh + 1 + hh * 0.3, 1.6, hh * 0.4) if k == len(SYSTEMS) - 1 else None
        heat(ax, bar, strip, key, name, col, k == 0, cax)
        img = ims[k]
        ih = 12
        ia = box(x + hw + 0.8, hy - 4, ih * img.shape[1] / img.shape[0], ih)
        ia.imshow(img, interpolation="lanczos")
        ia.set_axis_off()
        if k == 0:
            letter(fig, bar, "c", dx=-21, dy=8)
    fig.text(0.5, 1 - (H - 1.5) / H, HEAT_FOOTER.format(fdr=FDR, sp=SUPPORT_P), ha="center", va="bottom", fontsize=5.5)
    fig.legend([Patch(color=c) for c in FAMILY_COLOURS.values()], [FAMILY_NAMES[c] for c in FAMILY_COLOURS],
               loc="lower center", bbox_to_anchor=(0.5, 1 - (hy - 7.5) / H), ncol=4, fontsize=5.2, frameon=False,
               handlelength=1.2, title=FAMILY_LEGEND_TITLE, title_fontsize=5.2)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig4.png", dpi=PNG_DPI)
    fig.savefig(OUT / "fig4.pdf", dpi=PDF_DPI)
    fig.savefig(OUT / "fig4.svg", dpi=PDF_DPI)          # live text (svg.fonttype none): edit in Inkscape/Illustrator
    plt.close(fig)
    print("wrote", OUT, f"{W} x {H:.0f} mm")


if __name__ == "__main__":
    main()
