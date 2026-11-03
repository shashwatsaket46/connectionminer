#!/usr/bin/env python3
"""Figure 2 (paper version: A null shuffle experiments, B benchmarking) + appendix figure (train/val/test split).

PORTABLE: reads only ./data/ and ../fonts/ (relative to this file), runs from any working directory, no cluster.
Dependencies: numpy, scipy, matplotlib, pandas, pillow (nothing else).

    python fig2.py
    -> out/fig2.svg  (editable: live text in Liberation Sans, vector plots, named groups; BioRender schematic = raster)
       out/fig2.png  (2700 px wide, 300 dpi)        out/fig2.pdf  (the PNG at 300 dpi, as in the original build)
       out/figS_split.{svg,png,pdf}                 (appendix: split panel, a raster crop inside a live-text frame)

Panels:
  A  two rows (Null 1 shuffle P, Null 2 shuffle beta): matplotlib schematic + half-violin densities of the 10 test-fold
     r values per system (data/nulls_test_r.csv)
  B  "C reconstruction": bars = mean test r over 10 folds, dots = folds, per method (data/c_recon_test_r.csv);
     "Label mix": BioRender schematic (data/br_mix_v2.png, raster) + recovery dots per type pair
     (data/recovery_pairs.csv, data/recovery_runs.csv)
  figS_split: data/crop_split.png (raster) in the same frame style.
Data provenance: extract_data.py (cluster-only).
Edit the constants in the "EDITABLE" block below; everything else is layout code.
"""
import io
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.stats import gaussian_kde, wilcoxon  # noqa: E402

import sys  # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from svgkit import Doc, trim_box  # noqa: E402

DATA, FONTS, OUT = HERE / "data", HERE.parent / "fonts", HERE / "out"

# =====================================================================================================================
# EDITABLE
# =====================================================================================================================
# ---- panel titles (letter, words) and sub-headings
PANEL_A_TITLE = ("A", "Null shuffle experiments")
PANEL_B_TITLE = ("B", "Benchmarking against existing methods")
SUB_C_RECON = "C reconstruction"
SUB_LABEL_MIX = "Label mix"
SPLIT_TITLE = "Train / validation / test split of cell types"      # appendix figure (no panel letter)

# ---- footnote under panel A. {null_r}: ConnectionMiner test r of the null-study runs (computed from
#      data/nulls_test_r.csv); {main_r}: ConnectionMiner test r of the main models (from data/c_recon_test_r.csv).
FOOTNOTE = ["ConnectionMiner in A = the real model of the null-study runs, which differs from the main models for two "
            "systems (visual: min_cells=5 build; CX: reference build).",
            "Its held-out test r is therefore {null_r} here vs {main_r} for the main models. "
            "Densities over the 10 test folds; dashed line = ConnectionMiner mean."]

# ---- systems: data key -> (display name in A and in B "C reconstruction", colour)
SYSTEMS = {"visual": ("Visual", "#0072B2"), "cx": ("Central complex", "#009E73"), "motor": ("Motor", "#E69F00")}
RECOVERY_NAMES = {"visual": "Visual", "cx": "CX", "motor": "Motor"}   # x labels of the label-mix plot

# ---- panel A: null schematics (matplotlib mathtext: $\mathbf{P}$ etc.)
NULL_TITLES = {1: r"Null 1: shuffle assignments $\mathbf{P}$", 2: r"Null 2: shuffle interactions $\mathbf{\beta}$"}
NULL_STEPS = {   # (box text, box width in inches) or ("lock", label under the padlock)
    1: [("real training $\\mathbf{P}$\n(types ×\nmetacells)", 0.80),
        ("permute by type\n(within annotated /\nwithin unannotated),\nby metacell,\nor both", 1.05),
        ("lock", "freeze $\\mathbf{P}$"),
        ("re-learn $\\mathbf{\\beta}$\non training\ntypes", 0.70),
        ("lock", "freeze $\\mathbf{\\beta}$"),
        ("solve $\\mathbf{P}$ on\nheld-out\ntest types", 0.70)],
    2: [("real training $\\mathbf{\\beta}$\n(genes ×\ngenes)", 0.80),
        ("permute by gene\n(same relabelling\nof rows and\ncolumns),\nor by entry", 1.05),
        ("lock", "freeze $\\mathbf{\\beta}$"),
        ("solve $\\mathbf{P}$ on\nheld-out\ntest types", 0.70)]}
NULL_RESULT_BOX = r"$r(\hat{C}, C)$"
NULL_RESULT_SUB = "test types"
# rows of each density plot: (data row key in nulls_test_r.csv, row label); first row = the real model
NULL_ROWS = {1: [("real", "ConnectionMiner"), ("P_by_metacell", "by metacell"),
                 ("P_by_type_metacell", "by type + metacell"), ("P_by_type", "by type")],
             2: [("real", "ConnectionMiner"), ("beta_by_gene", "by gene"), ("beta_by_entry", "by entry")]}
NULL_XLABEL = r"Pearson $r$ ($\hat{C}$ vs $C$), held-out test types"
NULL_BOX_FILL = {1: "#DCE8F7", 2: "#E6E6E6"}     # input / result box tint per null
NULL_ROW_BG = {1: "#F7FAFE", 2: "#FAFAFA"}       # tinted container behind each null row

# ---- panel B, C reconstruction: method key (data) -> (label, colour). Hatched bars come from the data
#      (methods that use the held-out block's own C).
METHODS = {
    "degree_count_insample": ("Degree (count, in-sample)", "#E9A6A1"),
    "degree_count_leaveout": ("Degree (count, leave-out)", "#E9A6A1"),
    "arbor_10um": ("Arbor overlap 10 µm", "#8a8a8a"),
    "constraint_only_P_cm_beta": ("Constraint-only P (CM β)", "#BC85B4"),
    "combined_no_degree": ("Combined", "#C9B27C"),
    "cellranger_graph": ("Cell Ranger (graph-based)", "#CFE8D5"),
    "cellranger_kmeans": ("Cell Ranger (k-means)", "#9FD0AE"),
    "seurat": ("Seurat", "#6FB0A0"),
    "scanpy": ("Scanpy", "#3F8F8A"),
    "connectionminer": ("ConnectionMiner", "#0F4D92"),
}
# method order inside each group: None = ascending by mean test r per system (original figure); or give a list of
# method keys per group, e.g. {"connectome": ["degree_count_insample", ...], "gene": [...]}, used for every system
METHOD_ORDER = None
GROUP_HEADERS = {"connectome": "connectome-based", "gene": "gene-based"}
C_RECON_YLABEL = "Held-out test Pearson r (prediction vs C)"
OURS_COLOUR = METHODS["connectionminer"][1]
WORSE_COLOUR = "#B64342"                          # delta annotation colour when not better / not significant

# ---- panel B, label mix
RECOVERY_YLABEL = "identity recovered (%)"
RECOVERY_XLABEL = "type pairs, ranked by recovery"
RECOVERY_FONT = "Liberation Sans"   # original PNG tile used DejaVu Sans; Liberation keeps the SVG in one font
RECOVERY_CUT = {"cx", "motor"}      # systems shown as N_SHOW pairs drawn (seed 0) from the pairs at/above their mean
RECOVERY_N_SHOW = 30

# ---- canvas (px at 300 dpi) and frame style
W, M, GAP, PAD, TITLE_H, SUBH = 2700, 10, 24, 30, 92, 62
INK_RGB, BORDER_RGB, SUB_RGB, SOFT_RGB = (25, 25, 25), (110, 110, 110), (70, 70, 70), (90, 90, 90)
INK, SOFT, EDGE = "#272727", "#666666", "#555555"  # matplotlib tiles
# =====================================================================================================================

inner = W - 2 * M - 2 * PAD
for f in FONTS.glob("LiberationSans-*.ttf"):
    font_manager.fontManager.addfont(str(f))
BASE_RC = {"font.family": "Liberation Sans", "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "fig2",
           "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "custom",
           "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
           "mathtext.bf": "Liberation Sans:bold", "mathtext.cal": "Liberation Sans:italic",
           "mathtext.sf": "Liberation Sans"}
plt.rcParams.update(BASE_RC)


def read_csv(name):
    return pd.read_csv(DATA / name, float_precision="round_trip")


def capture(fig, **kw):
    """render one matplotlib figure to PNG (PIL) and SVG (transparent bg) with the same dpi / bbox"""
    dpi = kw.get("dpi", fig.dpi)
    b = io.BytesIO(); fig.savefig(b, format="png", **kw); b.seek(0)
    s = io.BytesIO(); fig.savefig(s, format="svg", facecolor="none", **kw)
    plt.close(fig)
    return dict(png=Image.open(b).convert("RGB"), svg=s.getvalue(), dpi=float(dpi))


# ============================================================================================== panel A: null rows
DPI_A, W_IN, SCH_W = 300, 8.8, 5.45


def box(ax, x, y, w, h, text, fc, fs=7.6, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=EDGE, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333", mutation_scale=7, shrinkA=0, shrinkB=0))


def lock(ax, cx, cy, s=0.13, label=None):
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s * 0.8, fc="#5a5a5a", ec="#3a3a3a", lw=0.5, zorder=4))
    ax.add_patch(Arc((cx, cy + s * 0.3), s * 0.62, s * 0.75, theta1=0, theta2=180, lw=1.3, color="#3a3a3a", zorder=4))
    for sx in (-1, 1):
        ax.plot([cx + sx * s * 0.31] * 2, [cy + s * 0.3, cy + s * 0.1], color="#3a3a3a", lw=1.3, zorder=4,
                solid_capstyle="butt")
    ax.add_patch(plt.Circle((cx, cy - s * 0.12), s * 0.09, color="white", zorder=5))
    if label:
        ax.text(cx, cy - s * 0.75, label, ha="center", va="top", fontsize=7.4, color=INK)


def schematic(ax, which, h):
    ax.set_xlim(0, SCH_W); ax.set_ylim(0, h); ax.axis("off")
    fc = NULL_BOX_FILL[which]
    y, bh = h * 0.46, 0.62
    ax.text(0.02, h - 0.04, NULL_TITLES[which], ha="left", va="top", fontsize=8.6, fontweight="bold", color=INK)
    x, gap = 0.04, 0.14
    for i, s in enumerate(NULL_STEPS[which]):
        if s[0] == "lock":
            lock(ax, x + 0.19, y + 0.05, label=s[1]); x += 0.38
        else:
            txt, w = s
            hh = bh + (0.22 if i == 1 else 0) + (0.12 if (i == 1 and which == 1) else 0)
            box(ax, x, y - hh / 2, w, hh, txt, "white" if i else fc, fs=6.9 if i == 1 else 7.2)
            x += w
        arrow(ax, x + 0.02, x + gap - 0.02, y); x += gap
    box(ax, x, y - 0.24, 0.56, 0.48, NULL_RESULT_BOX, fc, fs=8.6)
    ax.text(x + 0.28, y - 0.32, NULL_RESULT_SUB, ha="center", va="top", fontsize=6.6, color=SOFT)
    return x + 0.56


def facets(fig, nulls, which, rect):
    x0, y0, w, h = rect
    labw = 0.86 / W_IN
    rows = NULL_ROWS[which]
    n = len(rows)
    xs = np.linspace(-0.1, 1.05, 600)
    pw = (w - labw - 2 * 0.012) / 3
    axes = []
    for k, (skey, (name, col)) in enumerate(SYSTEMS.items()):
        ax = fig.add_axes([x0 + labw + k * (pw + 0.012), y0, pw, h])
        axes.append(ax)
        vals = {rk: nulls[(nulls.system == skey) & (nulls.row == rk)].sort_values("fold").r.to_numpy()
                for rk, _ in rows}
        for r, (rk, _) in enumerate(rows):
            v = vals[rk]; v = v[np.isfinite(v)]
            yb = n - 1 - r - 0.36
            m = v.mean()
            d = gaussian_kde(v, bw_method=0.6)(xs); d = d / d.max() * 0.72
            keep = (d > 0.03 * 0.72) & (xs >= v.min() - 0.06) & (xs <= v.max() + 0.06)
            ax.fill_between(xs[keep], yb, yb + d[keep], color=col, alpha=0.8 if r == 0 else 0.32, lw=0, zorder=2)
            ax.plot([m, m], [yb, yb + d[np.argmin(abs(xs - m))]], color=INK if r == 0 else col, lw=0.8, zorder=3)
            ax.text(xs[keep].max() + 0.025, yb + 0.1, f"{m:.2f}".replace("-0.00", "0.00"), ha="left", va="bottom",
                    fontsize=6.2, color=INK if r == 0 else SOFT, fontweight="bold" if r == 0 else "normal", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.axvline(np.nanmean(vals[rows[0][0]]), color=col, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.axhline(n - 1.5, color="#CFCECE", lw=0.5, zorder=0)
        ax.set_ylim(-0.5, n - 0.5); ax.set_xlim(-0.05, 1.0)
        ax.set_xticks([0, 0.4, 0.8], ["0", "0.4", "0.8"])
        ax.tick_params(axis="x", labelsize=6.8, colors=SOFT, width=0.6, length=2.5, pad=1.5)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(SOFT)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.set_title(name, fontsize=7.4, color=col, fontweight="bold", pad=2)
    axes[1].set_xlabel(NULL_XLABEL, fontsize=7.2, color=INK, labelpad=1.5)
    for r, (_, lab) in enumerate(rows):
        yf = axes[0].transData.transform((0, n - 1 - r))[1] / fig.bbox.height
        fig.text(x0, yf, lab, ha="left", va="center", fontsize=7.4 if r == 0 else 6.9,
                 color=INK if r == 0 else SOFT, fontweight="bold" if r == 0 else "normal")


def null_row(nulls, which):
    h_in = 2.0 if which == 1 else 1.8
    fig = plt.figure(figsize=(W_IN, h_in), dpi=DPI_A)
    fig.patches.append(FancyBboxPatch((0.004, 0.02), 0.992, 0.96, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=NULL_ROW_BG[which], ec="#BBBBBB", lw=0.6,
                                      zorder=-1))
    sch = fig.add_axes([0.01, 0.04, SCH_W / W_IN, 0.92])
    xr = schematic(sch, which, h_in * 0.92)
    xf_end, fx0, yarrow = 0.01 + xr / W_IN, 0.01 + SCH_W / W_IN + 0.012, 0.04 + 0.92 * 0.46
    fig.add_artist(FancyArrowPatch((xf_end + 0.003, yarrow), (fx0 - 0.004, yarrow), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333"))
    bottom = 0.36 / h_in
    facets(fig, nulls, which, (fx0, bottom, 0.985 - fx0, 1 - bottom - 0.30 / h_in))
    return capture(fig, dpi=DPI_A)


# ======================================================================================= panel B: C reconstruction
def c_recon(cr):
    BAR_W = 0.42
    with plt.rc_context({"font.size": 13, "axes.linewidth": 1.8}):
        fig, axes = plt.subplots(1, 3, figsize=(25, 7.4), sharey=True)
        rng = np.random.default_rng(0)
        for ax, (s, (name, _)) in zip(axes, SYSTEMS.items()):
            sub = cr[cr.system == s].sort_values(["method", "idx"])
            arr = {m: g.r.to_numpy() for m, g in sub.groupby("method")}
            hat = {m: bool(g.hatched.iloc[0]) for m, g in sub.groupby("method")}
            grp = {m: g.group.iloc[0] for m, g in sub.groupby("method")}
            # groups in data order (stable) then ordered
            order = list(dict.fromkeys(sub.sort_index().method))
            con = [m for m in order if grp[m] == "connectome"]
            gen = [m for m in order if grp[m] == "gene"]
            if METHOD_ORDER is None:
                con = sorted(con, key=lambda m: np.nanmean(arr[m]))
                gen = sorted(gen, key=lambda m: np.nanmean(arr[m]))
            else:
                con = [m for m in METHOD_ORDER["connectome"] if m in con]
                gen = [m for m in METHOD_ORDER["gene"] if m in gen]
            ser = con + gen + ["connectionminer"]
            for k, m in enumerate(ser):
                v = arr[m]; v = v[~np.isnan(v)]
                ax.bar(k, v.mean(), BAR_W, color=METHODS[m][1], edgecolor="black", linewidth=1.1,
                       hatch="///" if hat[m] else None, zorder=2)
                ax.scatter(k + rng.uniform(-BAR_W / 4, BAR_W / 4, len(v)), v, s=9, color="black", alpha=0.5,
                           linewidths=0, zorder=3)
                ax.text(k, max(v.max(), v.mean()) + 0.02, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=10)
            ax.axvline(len(con) - 0.5, color="#bbbbbb", lw=1, ls=":", zorder=1)
            ax.axvline(len(con) + len(gen) - 0.5, color="#bbbbbb", lw=1, ls=":", zorder=1)
            ax.text((len(con) - 1) / 2, 1.02, GROUP_HEADERS["connectome"], ha="center", fontsize=11, color="#555555")
            ax.text(len(con) + (len(gen) - 1) / 2, 1.02, GROUP_HEADERS["gene"], ha="center", fontsize=11,
                    color="#555555")
            o = arr["connectionminer"]
            best = arr[max(ser[:-1], key=lambda m: np.nanmean(arr[m]))]
            d = o - best; d = d[~np.isnan(d)]
            p = wilcoxon(d).pvalue
            ax.text(len(ser) - 1, 1.13, f"Δ = {d.mean():+.2f}, P = {p:.3f}", ha="right", va="top", fontsize=11.5,
                    color=OURS_COLOUR if d.mean() > 0 and p < 0.05 else WORSE_COLOUR)
            ax.axhline(0, color="black", lw=1)
            ax.axhline(o.mean(), color=OURS_COLOUR, lw=1.1, ls="--", alpha=0.6, zorder=1)
            ax.set_xticks(range(len(ser)), [METHODS[m][0] for m in ser], rotation=55, ha="right",
                          rotation_mode="anchor", fontsize=10.5)
            ax.set_xlim(-0.6, len(ser) - 0.4)
            ax.set_title(name, fontsize=17, weight="bold", pad=12)
            ax.set_ylim(-0.1, 1.15)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.yaxis.grid(True, color="#ececec", lw=1, zorder=0)
        axes[0].set_ylabel(C_RECON_YLABEL)
        fig.tight_layout()
        return capture(fig, dpi=250, bbox_inches="tight", pad_inches=0.1)


# ========================================================================================== panel B: label mix
def recovery(g, d):
    N_PAIRS = 10
    with plt.rc_context({"font.family": RECOVERY_FONT, "font.size": 11}):
        fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
        x0, centres = 0, []
        for s in SYSTEMS:
            col = SYSTEMS[s][1]
            sub = g[g.system == s]
            if s in RECOVERY_CUT:
                sub = sub[sub.bal >= sub.bal.mean()]
                sub = sub.sample(min(RECOVERY_N_SHOW, len(sub)), random_state=0)
            sub = sub.sort_values("bal", ascending=False).reset_index(drop=True)
            span = N_PAIRS - 1
            step = span / max(len(sub) - 1, 1)
            for i, r in sub.iterrows():
                x = x0 + i * step
                if s not in RECOVERY_CUT:                   # every seed of the pair as its own point
                    runs = d[(d.system == s) & (d.pair == r.pair)].balanced
                    ax.scatter([x] * len(runs), runs, s=22, color=col, edgecolor="white", linewidths=0.4, zorder=3)
                else:
                    ax.scatter(x, r.bal, s=22, color=col, edgecolor="white", linewidths=0.4, zorder=3)
            centres.append(x0 + span / 2)
            x0 += N_PAIRS + 1.5
        ax.axhline(50, color="#888888", lw=0.8, ls="--")
        ax.text(-0.8, 51.5, "chance", fontsize=8.5, color="#888888", ha="left", va="bottom")
        ax.set_xlim(-1, x0 - 1.5)
        ax.set_ylim(-3, 103)
        ax.set_xticks(centres, [RECOVERY_NAMES[s] for s in SYSTEMS], fontsize=12, weight="bold")
        for t, s in zip(ax.get_xticklabels(), SYSTEMS):
            t.set_color(SYSTEMS[s][1])
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_xlabel(RECOVERY_XLABEL, fontsize=10.5, color="#333333", labelpad=6)
        ax.set_ylabel(RECOVERY_YLABEL)
        fig.tight_layout()
        return capture(fig, bbox_inches="tight", pad_inches=0.05)


# ================================================================================================== composition
class Tile:
    """a matplotlib capture (svg + png) or a raster; optionally trimmed to its non-white content"""
    def __init__(self, cap=None, raster=None, trim=True):
        self.cap, self.raster = cap, raster
        self.im = cap["png"] if cap else raster
        self.box = trim_box(self.im) if trim else (0, 0, self.im.width, self.im.height)
        self.w, self.h = self.box[2] - self.box[0], self.box[3] - self.box[1]

    def png(self, w, h):
        return self.im.crop(self.box).resize((w, h), Image.LANCZOS)

    def svg(self, doc, x, y, w, h, gid):
        if self.cap:
            pt = 72 / self.cap["dpi"]                       # 1 px of the capture png = pt points of its svg
            doc.mpl(self.cap["svg"], x, y, (w / self.w) / pt, gid, crop=tuple(v * pt for v in self.box), gid=gid)
        else:
            doc.image(self.im.crop(self.box), x, y, w, h, gid=gid)


def fit_h(t, h):
    return round(t.w * h / t.h), h


def fit_w(t, w):
    return w, round(t.h * w / t.w)


def row(left, right, h, gap=GAP):
    a, b = fit_h(left, h), fit_h(right, h)
    if a[0] + gap + b[0] > inner:
        s = (inner - gap) / (a[0] + b[0])
        a, b = fit_h(left, int(h * s)), fit_h(right, int(h * s))
    return a, b


def font(size, bold=False):
    return ImageFont.truetype(str(FONTS / f"LiberationSans-{'Bold' if bold else 'Regular'}.ttf"), size)


class Canvas:
    """draws the same layout into a PIL image (PNG/PDF) and an SVG document"""
    def __init__(self, h, title):
        self.img = Image.new("RGB", (W, h), "white")
        self.d = ImageDraw.Draw(self.img)
        self.doc = Doc(W, h, title)

    def panel(self, y, h, letter, word, gid):
        self.d.rounded_rectangle([M + 3, y, W - M - 3, y + h], radius=26, outline=BORDER_RGB, width=4)
        self.doc.rounded(M + 3, y, W - M - 3, y + h, gid=f"{gid}_frame")
        with self.doc.group(f"{gid}_title"):
            if letter:
                self.d.text((M + 22, y + 12), letter, font=font(62, True), fill=INK_RGB)
                self.doc.text(M + 22, y + 12, letter, 62, INK_RGB, bold=True)
            self.d.text((M + 88, y + 18), word, font=font(52), fill=INK_RGB)
            self.doc.text(M + 88, y + 18, word, 52, INK_RGB)

    def text(self, x, y, s, size, fill, gid=None):
        self.d.text((x, y), s, font=font(size), fill=fill)
        self.doc.text(x, y, s, size, fill, gid=gid)

    def tile(self, t, x, y, wh, gid):
        self.img.paste(t.png(*wh), (x, y))
        t.svg(self.doc, x, y, *wh, gid)

    def save(self, stem):
        self.img.save(OUT / f"{stem}.png", dpi=(300, 300))
        self.img.save(OUT / f"{stem}.pdf", resolution=300)
        self.doc.save(OUT / f"{stem}.svg")
        print("saved", OUT / f"{stem}.svg", "/ .png / .pdf", self.img.size)


def main():
    OUT.mkdir(exist_ok=True)
    nulls, cr = read_csv("nulls_test_r.csv"), read_csv("c_recon_test_r.csv")
    pairs, runs = read_csv("recovery_pairs.csv"), read_csv("recovery_runs.csv")

    null1, null2 = Tile(null_row(nulls, 1), trim=False), Tile(null_row(nulls, 2), trim=False)
    recon = Tile(c_recon(cr))
    br = Tile(raster=Image.open(DATA / "br_mix_v2.png").convert("RGB"))
    swap = Tile(recovery(pairs, runs))

    fmt = lambda xs: " / ".join(f"{x:.2f}" for x in xs)
    null_r = [nulls[(nulls.system == s) & (nulls.row == "real")].r.mean() for s in SYSTEMS]
    main_r = [cr[(cr.system == s) & (cr.method == "connectionminer")].r.mean() for s in SYSTEMS]
    foot = [line.format(null_r=fmt(null_r), main_r=fmt(main_r)) for line in FOOTNOTE]

    # ---------------------------------------------------------------- Figure 2 (A + B)
    n1s, n2s, recs = fit_w(null1, inner), fit_w(null2, inner), fit_w(recon, inner)
    mixs = row(br, swap, 430)
    brows = [([(recon, recs, "plot")], SUB_C_RECON, "panelB_c_reconstruction"),
             ([(br, mixs[0], "schematic_raster"), (swap, mixs[1], "recovery_plot")], SUB_LABEL_MIX,
              "panelB_label_mix")]
    foot_h = len(foot) * 40 + 8
    hA = TITLE_H + n1s[1] + GAP + n2s[1] + 14 + foot_h + PAD
    hB = TITLE_H + sum(SUBH + max(s[1] for _, s, _ in imgs) + GAP for imgs, _, _ in brows) - GAP + PAD
    cv = Canvas(M + hA + GAP + hB + M, "Figure 2")
    y = M
    with cv.doc.group("panelA"):
        cv.panel(y, hA, *PANEL_A_TITLE, "panelA")
        yy = y + TITLE_H
        cv.tile(null1, M + PAD, yy, n1s, "panelA_null1_shuffle_P"); yy += n1s[1] + GAP
        cv.tile(null2, M + PAD, yy, n2s, "panelA_null2_shuffle_beta"); yy += n2s[1] + 14
        with cv.doc.group("panelA_footnote"):
            for line in foot:
                cv.text(M + PAD + 14, yy, line, 31, SOFT_RGB); yy += 40
    y += hA + GAP
    with cv.doc.group("panelB"):
        cv.panel(y, hB, *PANEL_B_TITLE, "panelB")
        yy = y + TITLE_H
        for imgs, sub, gid in brows:
            with cv.doc.group(gid):
                cv.text(M + PAD + 10, yy + 6, sub, 40, SUB_RGB, gid=f"{gid}_heading")
                tot = sum(s[0] for _, s, _ in imgs) + GAP * (len(imgs) - 1)
                x = M + PAD + (inner - tot) // 2
                for t, s, nm in imgs:
                    cv.tile(t, x, yy + SUBH, s, f"{gid}_{nm}"); x += s[0] + GAP
            yy += SUBH + max(s[1] for _, s, _ in imgs) + GAP
    cv.save("fig2")

    # ---------------------------------------------------------------- appendix: train / val / test split
    split = Image.open(DATA / "crop_split.png").convert("RGB")
    mats = Tile(raster=split.crop((20, 120, 772, 410)))           # C matrices with the split blocks
    scores = Tile(raster=split.crop((9, 445, 783, 735)))          # train / val / test r
    sgap = 110
    ms, ss = row(mats, scores, 430, gap=sgap)
    hC = TITLE_H + ms[1] + PAD
    cv = Canvas(M + hC + M, "Figure S: train / validation / test split")
    with cv.doc.group("split"):
        cv.panel(M, hC, "", SPLIT_TITLE, "split")
        x = M + PAD + (inner - ms[0] - sgap - ss[0]) // 2
        cv.tile(mats, x, M + TITLE_H, ms, "split_matrices_raster")
        cv.tile(scores, x + ms[0] + sgap, M + TITLE_H, ss, "split_scores_raster")
    cv.save("figS_split")


if __name__ == "__main__":
    main()
