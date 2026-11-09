"""Figure 4 (portable build): gene-pair evidence behind the connectome model.

    python fig4.py            ->  out/fig4.svg (editable, live text), out/fig4.pdf, out/fig4.png

Reads ONLY ./data/ (made on the cluster by extract_data.py) and ../fonts/ (Liberation Sans, metric-compatible
with Arial). Runs from any working directory. Dependencies: numpy, scipy, matplotlib, pandas, pillow.

a  Volcano per system: every TESTED directed gene pair. x = 10-fold mean log2(1+beta) - log2(1+beta_null)
   (real solves vs refit null), y = -log10 paired-t P over the folds. Significant = Benjamini-Hochberg FDR over all
   tested pairs of the system. Pairs whose two genes both belong to a gene family are coloured by family class.
   (The cross-system Fisher combination is reported in the Results text only, not drawn.)
b  R^2(k): squared Pearson r between C and C_hat restricted to the global top-k gene pairs of each train solve;
   median and IQR over 50 train solves; dashed = the full model.
c  Share of C_hat per family gene pair (columns) on synaptic type pairs (rows). Columns = the best-share supported
   pair (volcano x > 0, P < SUPPORT_P) of every family class, then the remaining supported pairs by mean share,
   filled up by share. Rows = top total share, at most ROWS_PER_COLUMN per dominant column; rows and columns
   biclustered (cosine, average linkage, optimal leaf order). Colour strip = family class (legend shared with a).

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

# --- systems: (data key, panel title, title colour) -------------------------------------------------------------
SYSTEMS = [
    ("visual", "Visual system", "#3B5BA9"),
    ("motor", "Motor system", "#D9822B"),
    ("cx", "Central complex", "#3E9A57"),
]

# --- font sizes (pt); 6 pt is the smallest size used, as in Fig 3 -------------------------------------------------
FS_MIN, FS, FS_TITLE, FS_LETTER = 6, 6.5, 8, 10
FS_LEGEND = 7.5            # family-class legend under panel a (also the key for the colour strips in c)
FS_GENE = 7                # gene-pair call-outs in the volcanoes

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
VOLCANO_YLABEL = r"$-\log_{10}$ $P$"
VOLCANO_XLABEL = r"$\beta$ effect, $\log_2(1+\beta) - \log_2(1+\beta_{\mathrm{null}})$"
R2_TITLE = r"Variance of $C$ explained"
R2_XLABEL = "Top-k gene pairs"
R2_YLABEL = r"$R^2$ (train)"
R2_YMAX = 0.56
HEAT_YLABEL = "Synaptic type pair (pre → post)"
HEAT_CBAR_LABEL = r"Share of $\hat{C}$ (%)"

# --- fonts / output ---------------------------------------------------------------------------------------------
FONT_FAMILY = "Liberation Sans"    # bundled in ../fonts; put "Arial" first in style() if you prefer Arial
PNG_DPI, PDF_DPI = 600, 300
FIG_WIDTH_MM = 180

# gene names treated as anonymous loci (never used as labels / heatmap columns)
ANON = re.compile(r"\d{4,}|^(CG|CR)\d|:CR\d|^FBti|^~")
# ================================================================================================================

MM = 1 / 25.4
# panel c layout (mm): heatmap width and left edges (room for each system's row labels)
HEAT_W = 35
HEAT_X0 = [27, 80, 144]


def style():
    for f in FONTS.glob("LiberationSans-*.ttf"):
        fm.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": FS, "axes.titlesize": FS_TITLE, "axes.labelsize": FS, "xtick.labelsize": FS_MIN,
        "ytick.labelsize": FS_MIN, "legend.fontsize": FS, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
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


def letter(fig, x, y, s):
    """Panel letter with its top-left corner at (x, y) mm from the figure's top-left corner."""
    fig.text(x * MM / fig.get_figwidth(), 1 - y * MM / fig.get_figheight(), s,
             fontsize=FS_LETTER, fontweight="bold", va="top", ha="left")


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
def volcano(ax, key, name, col):
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
    ax.scatter(x[sig], y[sig], s=11, c=fc, lw=0.4, edgecolors="black", zorder=4)
    nsig = int(sig.sum())
    thr = -np.log10(p[sig].max()) if nsig else -np.log10(FDR / len(d))   # none pass: BH's first step FDR / m
    ax.axhline(thr, ls=(0, (1.5, 1.5)), lw=0.5, c="#555555", zorder=0)
    ax.axvline(0, c="#e0e0e0", lw=0.5, zorder=0)
    ok = d.named.to_numpy(bool)
    sel = np.flatnonzero(ok & (x > 0) & (p < LABEL_MAX_P))
    for k, t in enumerate(sel[np.argsort(-y[sel])][:N_LABELS]):
        ax.annotate(f"{d.gene_i.iloc[t]}–{d.gene_j.iloc[t]}", (x[t], y[t]), (0.03, 0.6 - 0.12 * k),
                    textcoords="axes fraction", fontsize=FS_GENE, style="italic", va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=0.3, color="#9a9a9a", shrinkA=1, shrinkB=1.5))
    ax.set_ylim(-0.2, max(y.max(), thr) * 1.08)
    ax.set_title(name, color=col, fontsize=FS_TITLE, weight="bold", loc="left", pad=3)
    ax.text(0.98, 0.04, f"{nsig} significant", transform=ax.transAxes, ha="right", fontsize=FS, color=col)
    up, dn = int((sig & (x > 0)).sum()), int((sig & (x < 0)).sum())
    if FDR == 0.05:                       # cross-check against the counts of the source build
        assert (up, dn) == (meta["bh_q05_up"], meta["bh_q05_down"]), (key, up, dn, meta)
    print(f"{key}: {len(d):,} tested, {nsig} BH-significant ({up} up / {dn} down), "
          f"family among significant: {list(zip(d.gene_i[sig], d.gene_j[sig], cls[sig]))}")


# ------------------------------------------------------------------------------------------------------ panel b
def r2_panel(ax):
    for key, name, col in SYSTEMS:
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
    ax.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 0.8), handlelength=1.4, fontsize=FS, borderaxespad=0.2,
              labelspacing=0.3)
    ax.set_title(R2_TITLE, loc="left", fontsize=FS_TITLE, fontweight="bold", pad=3)


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


def heat(ax, strip, key, name, col, first, cax):
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
    ax.set_yticklabels([f"{d['row_pre'][r]}\u2009→\u2009{d['row_post'][r]}" for r in rows], fontsize=FS_MIN)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{d['col_pre'][j]}–{d['col_post'][j]}" for j in cj], rotation=90, fontsize=FS_MIN,
                       style="italic")
    ax.tick_params(length=0, pad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    if first:
        ax.set_ylabel(HEAT_YLABEL, fontsize=FS)
    strip.imshow([[to_rgb(FAMILY_COLOURS[cls[j]]) for j in cj]], aspect="auto", interpolation="nearest")
    strip.set_axis_off()
    strip.set_title(name, loc="left", fontsize=FS_TITLE, fontweight="bold", pad=2.5, color=col)
    if cax is not None:
        cb = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=HEAT_TICKS)
        cax.set_title(HEAT_CBAR_LABEL, fontsize=FS, pad=2)
        cb.ax.tick_params(labelsize=FS_MIN, length=1.5, width=0.4, pad=1)
        cb.outline.set_linewidth(0.4)


# ------------------------------------------------------------------------------------------------------- layout
def main():
    style()
    W = FIG_WIDTH_MM
    vy, vh = 6, 34                                     # all positions in mm from the top-left corner
    hy, sh, hh = vy + vh + 24, 1.8, 44                 # heatmap block: colour strip, then the heatmap
    H = hy + sh + 0.6 + hh + 20
    fig = plt.figure(figsize=(W * MM, H * MM))

    def box(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    # ---- a: volcanoes
    vx0, vw, vsp = 12, 31, 8
    vaxes = [box(vx0 + k * (vw + vsp), vy, vw, vh) for k in range(3)]
    for ax, (key, name, col) in zip(vaxes, SYSTEMS):
        volcano(ax, key, name, col)
    vaxes[0].set_ylabel(VOLCANO_YLABEL)
    vmid = vx0 + 1.5 * vw + vsp
    fig.text(vmid / W, 1 - (vy + vh + 5.5) / H, VOLCANO_XLABEL, ha="center", va="top", fontsize=FS)
    hand = [Line2D([], [], ls="", marker="o", ms=3, c=OTHER_PAIR_COLOUR)] + \
           [Line2D([], [], ls="", marker="s", ms=5, c=c) for c in FAMILY_COLOURS.values()] + \
           [Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="black", mew=0.5),
            Line2D([], [], ls=(0, (1.5, 1.5)), lw=0.7, c="#555555")]
    labs = ["Other pair"] + [FAMILY_NAMES[c] for c in FAMILY_COLOURS] + [f"FDR < {FDR * 100:g}%", "FDR threshold"]
    fig.legend(hand, labs, loc="upper left", bbox_to_anchor=(2 / W, 1 - (vy + vh + 10) / H),
               ncol=7, fontsize=FS_LEGEND, handletextpad=0.25, columnspacing=1.1, frameon=False, handlelength=1.2)
    letter(fig, 0.5, 0.5, "a")

    # ---- b: R^2(k)
    bx = vx0 + 3 * vw + 2 * vsp + 13
    axb = box(bx, vy, W - 2 - bx, vh)
    r2_panel(axb)
    letter(fig, bx - 12, 0.5, "b")

    # ---- c: heatmaps
    hw, hx0 = HEAT_W, HEAT_X0
    ims = icons()
    for k, (key, name, col) in enumerate(SYSTEMS):
        x = hx0[k]
        strip = box(x, hy, hw, sh)
        ax = box(x, hy + sh + 0.6, hw, hh)
        cax = box(W - 22, vy + vh + 13, 18, 1.6) if k == len(SYSTEMS) - 1 else None   # in the legend row of a
        heat(ax, strip, key, name, col, k == 0, cax)
        img = ims[k]
        ih = 9
        iw = ih * img.shape[1] / img.shape[0]
        ia = box(x - iw - 1.5, hy - ih + sh + 0.4, iw, ih)
        ia.imshow(img, interpolation="lanczos")
        ia.set_axis_off()
    letter(fig, 0.5, hy - 6, "c")

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig4.png", dpi=PNG_DPI)
    fig.savefig(OUT / "fig4.pdf", dpi=PDF_DPI)
    fig.savefig(OUT / "fig4.svg", dpi=PDF_DPI)          # live text (svg.fonttype none): edit in Inkscape/Illustrator
    plt.close(fig)
    print("wrote", OUT, f"{W} x {H:.0f} mm")


if __name__ == "__main__":
    main()
