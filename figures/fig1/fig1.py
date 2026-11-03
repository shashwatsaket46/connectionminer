"""Figure 1 (ConnectionMiner paper), portable build.

    python3 fig1.py            ->  out/fig1.svg  (editable: live text, named groups)
                                    out/fig1.png  (3600 x 2460 px, 300 dpi)
                                    out/fig1.pdf  (the PNG at 300 dpi, i.e. a raster PDF, same as the paper build)

Reads ONLY  ./data/  (pre-extracted arrays, made on the cluster by extract_data.py) and  ../fonts/  (Liberation Sans).
Dependencies (exactly these):  numpy, matplotlib, pillow   (+ Python standard library).  Tested with numpy 2.5,
matplotlib 3.11, pillow 12.3.  No scipy / pandas needed: all clustering / ordering was done during extraction.

Layout: the canvas is 3600 x 2460 px.  A (top-left): BioRender drawing (one raster) with live text on top.
B (bottom-left): the solver matrices, drawn by matplotlib from the pooled arrays in data/panelB_tiles.npz.
C (right): PRE / POST TF heatmaps + EM skeletons of the two highlighted types, from data/panelC.npz.
In the SVG every label is <text> (font "Liberation Sans"; install fonts/*.ttf on your machine so an editor shows them);
heatmaps / matrices stay embedded bitmaps (as in any matplotlib SVG); skeletons and wiring are vector lines.

Everything a non-author is likely to change is a named constant in the EDITABLE block below.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FONTS = HERE.parent / "fonts"
OUT = HERE / "out"
sys.dont_write_bytecode = True                 # keep the kit folder clean (no __pycache__)
sys.path.insert(0, str(HERE))
from svgkit import Doc  # noqa: E402

# =====================================================================================================================
# EDITABLE  --  text, colours, highlighted types, sizes
# =====================================================================================================================
# ---- colours (RGB 0-255 for the PIL/SVG layer; hex for matplotlib) --------------------------------------------------
INK = (25, 25, 25)                 # main text
GREY = (95, 95, 95)                # secondary text
BORDER = (110, 110, 110)           # rounded panel frames
UNANNOTATED_DOT = (176, 176, 176)  # grey dot in panel A's legend
SYS_COLOR = {"visual": "#0072B2", "cx": "#009E73", "motor": "#E69F00"}   # one colour per system, used in A, B and C
POST_TYPE_COLOR = "#7E57C2"        # panel C: colour of the second (post-synaptic) highlighted type
HEATMAP_CMAP = "RdBu"              # panel C heatmap colour map (red = below type mean, blue = above)
BETA_CMAP = "viridis"              # panel B beta tile
P_CMAP = "Greens"                  # panel B P / P^T tiles

# ---- panel titles ---------------------------------------------------------------------------------------------------
TITLE_A = ("A", "Drosophila melanogaster")     # (letter, title); A's title is italic
TITLE_B = ("B", "ConnectionMiner solver")
SUBTITLE_B = "train r / AP / F1: one training block (fold 0, seed 0) · test r: held-out types"
TITLE_C = ("C", "Derived cell type assignment")
SUBTITLE_C = "TF detection per type, before (PRE) and after (POST) the solve"

# ---- panel A: per-row counts (checked 2026-10-30 against each build's indices / READMEs) ---------------------------
ROWS = {
    "visual": dict(cells="99,656", mc="4,748", red="21×",
                   types=["741 types", "81 annotated · 660 unannotated"],
                   em=["FlyWire · 97,105 neurons"]),
    "cx": dict(cells="48,014", mc="2,490", red="19×",
               types=["853 types", "21 annotated · 832 unannotated"],
               em=["FlyWire · 4,944 typed neurons", "(type II lineages innervating the CX)"]),
    "motor": dict(cells="11,292", mc="2,036", red="5.5×",
                  types=["730 types (701 preMN + 29 MN)", "12 annotated · 718 unannotated"],
                  em=["FANC · curated preMN–MN connection table", "skeletons: 707 of 730 types"]),
}
LEGEND_A = ("annotated", "unannotated", "annotated type: ≥1 cell uniquely labelled with that type")
COLUMN_TITLES_A = (("EM neurons", 591), ("scRNA-seq", 820), ("Connectome (EM)", 1290))  # (text, x in base-image px)
FS_A = dict(cells=33, metacells=27, reduction=25, types=31, split=27, em=25, legend=27, legend_note=25, column=44)

# ---- panel B ---------------------------------------------------------------------------------------------------------
SYS_LABEL_B = {"visual": "Visual", "cx": "CX", "motor": "Motor"}
TEST_R = {"visual": 0.60, "cx": 0.81, "motor": 0.37}         # held-out test r (10 folds x 5 seeds; Fig 1 / Fig 2)
TILE_LABELS = ["P", "G", "β", "Gᵀ", "Pᵀ", "Ĉ", "C"]           # column headers over the matrices
TILE_AXES = [("types", "metacells"), ("metacells", "genes"), ("genes", "genes"), ("genes", "metacells"),
             ("metacells", "types"), ("types", "types"), ("types", "types")]   # (y label, x label), visual row only
UPDATE_BOXES = [("β-update", "multiplicative NNLS"), ("P-update", "mirror descent + Sinkhorn")]
N_ITER_LABEL = "×100"
B_TYPE_STRETCH = 1.6               # every "types" axis drawn 1.6x longer than the sqrt(n) rule
B_X_MATRICES = 0.15                # left edge of the matrix row (fraction of panel width; room for the scores)
B_ROW_Y = (0.80, 0.505, 0.275)     # vertical centre of the visual / CX / motor rows (fraction of panel height)

# ---- panel C ---------------------------------------------------------------------------------------------------------
SYS_LABEL_C = {"visual": "Visual", "cx": "CX", "motor": "Motor"}
EM_SOURCE = {"visual": "EM neurons (FlyWire)", "cx": "EM neurons (FlyWire)", "motor": "EM neurons (FANC)"}
HEATMAP_TITLES = ("PRE  (P$_{init}$·G)", "POST  (P·G)")
ROW_AXIS_LABEL = "cell types"
COLUMN_NOTE = "{n} transcription factors (same columns in both)"
COLORBAR_LABEL = "P·G − type mean"
# Highlighted (pre, post) type pair per system. Rows can be any type in data/panelC_labels.json "row_types", but the
# skeletons are only extracted for the pair below; to show another pair, set HIGHLIGHT in extract_data.py and re-run it
# on the cluster (fig1.py stops with a clear message if the pair and the data disagree).
HIGHLIGHT = {"visual": ("Li02", "Li28"), "cx": ("FS1A", "FB4Y"), "motor": ("LN314", "LN191")}

# ---- canvas / sizes (px at 300 dpi). Changing these re-flows the panels; the pooled matrices stay as extracted ------
W_TOT, H_LEFT = 3600, 2440          # figure width; height of the left column (= panel C height)
M, GAP, TITLE_H = 10, 24, 92        # outer margin, gap between panels, title band
CW = 1300                           # panel C width
UP_A = 1.22                         # scale of panel A's base drawing
DPI_B = 300                         # panel B render dpi
DPI_C = 330                         # panel C render dpi (drawn at 2x then downsampled, as the paper build)
FS_TITLE, FS_LETTER, FS_SUB_B, FS_SUB_C = 52, 62, 30, 32
PNG_DPI = 300
# =====================================================================================================================

SYS = ["visual", "cx", "motor"]
ROW_Y = {"visual": 185, "cx": 358, "motor": 531}    # row centres in the base drawing (native px) -- tied to the raster

# ---- fonts / matplotlib defaults (independent of the user's matplotlibrc) -------------------------------------------
matplotlib.rcdefaults()
for f in FONTS.glob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "font.family": "Liberation Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "none"})


def pil_font(size, bold=False, italic=False):
    s = {(False, False): "Regular", (True, False): "Bold", (False, True): "Italic", (True, True): "BoldItalic"}
    return ImageFont.truetype(str(FONTS / f"LiberationSans-{s[(bold, italic)]}.ttf"), size)


def hex2rgb(h):
    return tuple(int(round(255 * v)) for v in to_rgb(h))


def sys_cmap(k):
    """white -> light system colour -> system colour -> darker shade."""
    c = np.array(to_rgb(SYS_COLOR[k]))
    return LinearSegmentedColormap.from_list(f"sys_{k}", ["#FFFFFF", tuple(1 - 0.45 * (1 - c)), tuple(c), tuple(c * 0.5)])


# ---- data ------------------------------------------------------------------------------------------------------------
BZ = np.load(DATA / "panelB_tiles.npz")
BS = json.loads((DATA / "panelB_scores.json").read_text())
CZ = np.load(DATA / "panelC.npz")
CL = json.loads((DATA / "panelC_labels.json").read_text())
BASE_A = Image.open(DATA / "panelA_base.png").convert("RGB")
for k in SYS:
    if tuple(HIGHLIGHT[k]) != (CL[k]["pre_type"], CL[k]["post_type"]):
        sys.exit(f"HIGHLIGHT[{k!r}] = {HIGHLIGHT[k]} but data/ holds skeletons for "
                 f"{(CL[k]['pre_type'], CL[k]['post_type'])}: re-run extract_data.py with the new pair.")

# ---- derived geometry --------------------------------------------------------------------------------------------------
LW = W_TOT - 2 * M - GAP - CW                       # left column width
AW, AH = int(BASE_A.width * UP_A), int(BASE_A.height * UP_A)
assert AW <= LW - 10, (AW, LW)
DATA_H = AH + 30                                    # panel A height
BH = H_LEFT - DATA_H - GAP                          # panel B height
WB, HB = LW - 40, BH - TITLE_H - 20                 # panel B plot size (px)
WC, HC = CW - 40, H_LEFT - TITLE_H - 60             # panel C plot size (px)
S = lambda v: int(round(v * UP_A))


# =====================================================================================================================
# panel B: the solver matrices
# =====================================================================================================================
def panel_B(w=WB, h=HB, dpi=DPI_B):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    keys = ["P", "G", "beta", "GT", "PT", "Chat", "C"]
    rows_ = []
    for k in SYS:
        sh = BS[k]["shapes"]
        gcm = sys_cmap(k).copy(); gcm.set_bad("white"); gcm.set_under("white")
        ccm = sys_cmap(k).copy(); ccm.set_under("white")
        g99 = float(BZ[f"{k}_G99"])
        norms = ["auto99", (gcm, LogNorm(1e-3, g99)), (BETA_CMAP, LogNorm(float(BZ[f"{k}_Bmin"]), float(BZ[f"{k}_Bmax"]))),
                 (gcm, LogNorm(1e-3, g99)), "auto99",
                 (ccm, plt.Normalize(float(BZ[f"{k}_Chlo"]), float(BZ[f"{k}_Chhi"]))),
                 (ListedColormap(["white", SYS_COLOR[k]]), plt.Normalize(0, 1))]
        rows_.append((k, [(TILE_LABELS[j], keys[j], BZ[f"{k}_{keys[j]}"], sh[keys[j]], norms[j]) for j in range(7)]))

    # tile side = a * sqrt(n) (one scale for all systems, so shared axes have one length); types axes stretched
    L = lambda n: np.sqrt(n)
    Lr = lambda j, n: L(n) * (B_TYPE_STRETCH if j in (0, 5, 6) else 1)
    Lc = lambda j, n: L(n) * (B_TYPE_STRETCH if j in (4, 5, 6) else 1)
    gap, sym = 0.04 * w, 0.03 * w
    slot = [max(Lc(j, t[j][3][1]) for _, t in rows_) for j in range(7)]
    max_h = max(Lr(j, t[j][3][0]) for _, t in rows_ for j in range(7))
    a_ = min(((0.99 - B_X_MATRICES) * w - 6 * gap - 2 * sym) / sum(slot), 0.255 * h / max_h)
    x0s = [B_X_MATRICES]
    for j in range(6):
        x0s.append(x0s[-1] + (a_ * slot[j] + gap + (sym if j in (4, 5) else 0)) / w)
    for i, (k, tiles) in enumerate(rows_):
        yc = B_ROW_Y[i]
        fig.text(0.005, yc + 0.075, SYS_LABEL_B[k], color=SYS_COLOR[k], fontsize=10, weight="bold", va="center")
        for j, (lbl, key, Mp, shape, style) in enumerate(tiles):
            tw, th = a_ * Lc(j, shape[1]) / w, a_ * Lr(j, shape[0]) / h
            xl = x0s[j] + (a_ * slot[j] / w - tw) / 2
            ax = fig.add_axes([xl, yc - th / 2, tw, th])
            if key in ("G", "GT"):
                Mp = np.where(Mp > 0, Mp, 1e-12)            # below the 1e-3 floor -> white
            if key in ("P", "PT"):
                # constraint mask D as a light tint of the system colour, P on top in green (alpha follows P)
                norm = plt.Normalize(0, np.percentile(Mp[Mp > 0], 99))
                Dp = BZ[f"{k}_{key}_mask"]
                tint = 1 - 0.28 * (1 - np.array(to_rgb(SYS_COLOR[k])))
                rgb = np.ones(Mp.shape + (3,)) * (1 - Dp[..., None]) + tint * Dp[..., None]
                Pn = np.clip(norm(Mp), 0, 1)
                pc = plt.get_cmap(P_CMAP)(0.35 + 0.65 * Pn)[..., :3]
                al = np.clip(Pn * 1.6, 0, 1)[..., None]
                ax.imshow(rgb * (1 - al) + pc * al, aspect="auto", interpolation="nearest")
            else:
                cmap, norm = style
                ax.imshow(Mp, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                fig.text(x0s[j] + a_ * slot[j] / w / 2, B_ROW_Y[0] + a_ * max_h / h / 2 + 0.012, lbl,
                         fontsize=11, ha="center", va="bottom")
                ylab, xlab = TILE_AXES[j]
                ax.set_xlabel(xlab, fontsize=8, labelpad=4); ax.set_ylabel(ylab, fontsize=8, labelpad=4)
        for j, sy in ((4, "="), (5, "≈")):
            fig.text(x0s[j] + (a_ * slot[j] + 0.2 * gap) / w, yc, sy, fontsize=13, va="center")
        sc = BS[k]
        for q_, (lab_, v_) in enumerate((("r", sc["train_r"]), ("AP", sc["train_AP"]), ("F1", sc["train_F1"]))):
            fig.text(0.005, yc + 0.03 - q_ * 0.04, f"train {lab_} {v_:.2f}", fontsize=8.5,
                     va="center", weight="bold" if q_ == 0 else "normal")
        fig.text(0.005, yc - 0.098, f"test r {TEST_R[k]:.2f}", fontsize=8.5, va="center", color=SYS_COLOR[k],
                 weight="bold")
    # alternating updates
    for x0, (title, sub) in zip((0.08, 0.56), UPDATE_BOXES):
        fig.add_artist(FancyBboxPatch((x0, 0.03), 0.36, 0.13, boxstyle="round,pad=0.005,rounding_size=0.02",
                                      transform=fig.transFigure, fc="#F2F2F2", ec="#888888", lw=0.7))
        fig.text(x0 + 0.18, 0.115, title, fontsize=10.5, weight="bold", ha="center", va="center")
        fig.text(x0 + 0.18, 0.06, sub, fontsize=9.5, ha="center", va="center")
    fig.add_artist(FancyArrowPatch((0.445, 0.12), (0.555, 0.12), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=11, lw=1.3, color="#333"))
    fig.add_artist(FancyArrowPatch((0.555, 0.065), (0.445, 0.065), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=11, lw=1.3, color="#333"))
    fig.text(0.5, 0.0925, N_ITER_LABEL, ha="center", va="center", fontsize=9)
    return fig


# =====================================================================================================================
# panel C: PRE / POST heatmaps + EM skeletons
# =====================================================================================================================
def panel_C(w=WC, h=HC, dpi=DPI_C):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi * 2)
    bh = 1 / 3
    HX, HW = (0.075, 0.305), 0.215                  # left edges of PRE and POST heatmaps, their width
    for i, k in enumerate(SYS):
        pre_m, post_m, lim = CZ[f"{k}_pre"], CZ[f"{k}_post"], float(CZ[f"{k}_lim"])
        ends = CZ[f"{k}_block_ends"]
        row_types = CL[k]["row_types"]
        nrow, ncol = pre_m.shape
        pre, post = HIGHLIGHT[k]
        rows = {t: row_types.index(t) for t in (pre, post)}
        cols = {pre: SYS_COLOR[k], post: POST_TYPE_COLOR}
        upper, lower = sorted((pre, post), key=lambda t: rows[t])
        yb = 1 - (i + 1) * bh
        hy0, hh = yb + 0.17 * bh, 0.69 * bh
        fig.text(HX[0] + (HX[1] + HW - HX[0]) / 2, hy0 + hh + 0.085 * bh, SYS_LABEL_C[k], color=SYS_COLOR[k],
                 fontsize=9, weight="bold", ha="center", va="center")
        axes = []
        for j, (Mx, ttl) in enumerate(((pre_m, HEATMAP_TITLES[0]), (post_m, HEATMAP_TITLES[1]))):
            ax = fig.add_axes([HX[j], hy0, HW, hh]); axes.append(ax)
            ax.imshow(Mx, cmap=HEATMAP_CMAP, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
            for end in ends:
                ax.axhline(end - 0.5, color="#444444", lw=0.4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=7, pad=2.5)
            if j == 0:
                ax.set_ylabel(ROW_AXIS_LABEL, fontsize=7, labelpad=3)
            for t in (pre, post):
                ax.axhline(rows[t], color=cols[t], lw=0.8)
            ax.set_xlim(-0.5, ncol - 0.5); ax.set_ylim(nrow - 0.5, -0.5)
        lab_ = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
        for t in (pre, post):
            axes[0].plot(-0.9, rows[t], marker=">", color=cols[t], ms=4, clip_on=False, mec="white", mew=0.4)
            axes[1].plot(ncol - 0.1, rows[t], marker="<", color=cols[t], ms=4.5, clip_on=False, mec="white", mew=0.4)
            up = t == upper
            axes[1].text(ncol - 1.2, rows[t] + (-0.008 if up else 0.008) * nrow, t, ha="right",
                         va="bottom" if up else "top", fontsize=6, weight="bold", color=cols[t], bbox=lab_)
        fig.text(HX[0] + (HX[1] + HW - HX[0]) / 2, hy0 - 0.004, COLUMN_NOTE.format(n=ncol),
                 fontsize=6, ha="center", va="top")
        cax = fig.add_axes([HX[0] + 0.17, hy0 - 0.095 * bh, HX[1] + HW - HX[0] - 0.24, 0.012 * bh])
        cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(-lim, lim), HEATMAP_CMAP), cax=cax,
                          orientation="horizontal", ticks=[-lim, 0, lim], format="%+.2f")
        cb.ax.tick_params(labelsize=5, length=1.2, pad=0.8)
        fig.text(HX[0] + 0.16, hy0 - 0.089 * bh, COLORBAR_LABEL, fontsize=5.5, ha="right", va="center")
        # skeletons + wiring from the POST heatmap rows to each neuron
        na = fig.add_axes([0.66, yb + 0.08 * bh, 0.33, 0.83 * bh]); na.axis("off")
        fig.text(0.825, yb + 0.035 * bh, EM_SOURCE[k], ha="center", fontsize=6.5, color="#444444")
        Sa, Sb = CZ[f"{k}_skel_pre"], CZ[f"{k}_skel_post"]
        na.add_collection(LineCollection(Sb, colors=POST_TYPE_COLOR, linewidths=0.45, alpha=0.85))
        na.add_collection(LineCollection(Sa, colors=SYS_COLOR[k], linewidths=0.45, alpha=0.95))
        na.autoscale(); na.set_aspect("equal", "datalim"); na.invert_yaxis()
        fig.canvas.draw()
        to_fig = lambda axx, xy: fig.transFigure.inverted().transform(axx.transData.transform(xy))
        S_of = {pre: Sa, post: Sb}
        sx = HX[1] + HW + 0.022
        for t, gx in ((upper, 0.645), (lower, 0.625)):
            pts = S_of[t].reshape(-1, 2)
            ym = np.median(np.r_[Sa.reshape(-1, 2)[:, 1], Sb.reshape(-1, 2)[:, 1]])
            half = pts[pts[:, 1] <= ym] if t == upper else pts[pts[:, 1] > ym]
            if len(half) == 0:
                half = pts
            target = half[np.argmin(half[:, 0])]
            _, sy = to_fig(axes[1], (ncol - 0.5, rows[t]))
            tx, ty = to_fig(na, tuple(target))
            path = [(sx, sy), (gx, sy), (gx, ty), (tx, ty)]
            fig.add_artist(plt.Line2D(*zip(*path), transform=fig.transFigure, color=cols[t], lw=1.0, alpha=0.9))
            na.plot(*target, "o", color=cols[t], ms=3.5, mec="white", mew=0.4)
            na.annotate(t, target, xytext=(0, 7 if t == upper else -7), textcoords="offset points", ha="center",
                        va="bottom" if t == upper else "top", fontsize=6.3, weight="bold", color=cols[t],
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    return fig


def render(fig, dpi, size=None):
    """matplotlib figure -> (PIL PNG at dpi [resized to size], SVG bytes with a transparent figure background)."""
    b = io.BytesIO(); fig.savefig(b, dpi=dpi, facecolor="white"); b.seek(0)
    png = Image.open(b).convert("RGB")
    if size is not None:
        png = png.resize(size, Image.LANCZOS)
    s = io.BytesIO(); fig.savefig(s, dpi=dpi, format="svg", facecolor="none", transparent=False)
    plt.close(fig)
    return png, s.getvalue()


# =====================================================================================================================
# panel A pieces shared by the PNG and SVG writers
# =====================================================================================================================
def panel_A_items():
    """(kind, args) list in base-image px * UP_A coordinates (origin = top-left of the scaled base drawing)."""
    it = []
    for k, yc in ROW_Y.items():
        r, y = ROWS[k], S(yc)
        tip = 1076 if k == "visual" else 1082        # visual: stop short of the "presynaptic" label
        it.append(((k, "arrow"), "line", [(S(1036), y), (S(tip - 10), y)], INK, 5))
        it.append(((k, "arrow"), "poly", [(S(tip), y), (S(tip) - 16, y - 11), (S(tip) - 16, y + 11)], INK))
        xc = S(955)
        it.append(((k, "cells"), "text", xc, y - 50, f"{r['cells']} cells", FS_A["cells"], INK, False, "mm"))
        it.append(((k, "cells"), "text", xc, y - 8, "↓", FS_A["metacells"], GREY, False, "mm"))
        it.append(((k, "cells"), "text", xc, y + 26, f"{r['mc']} metacells", FS_A["metacells"], INK, False, "mm"))
        it.append(((k, "cells"), "text", xc, y + 60, f"({r['red']} fewer)", FS_A["reduction"], GREY, False, "mm"))
        x0 = S(1262)
        lines = [(r["types"][0], FS_A["types"], True, INK), (r["types"][1], FS_A["split"], False, INK)] + \
                [(t, FS_A["em"], False, GREY) for t in r["em"]]
        hs = [38, 34] + [30] * len(r["em"])
        yy = y - sum(hs) / 2 + hs[0] / 2
        for (t, sz, b, c), h_ in zip(lines, hs):
            it.append(((k, "types"), "text", x0, yy, t, sz, c, b, "lm")); yy += h_
    lx, ly = S(700), S(640)
    for i, k in enumerate(SYS):
        it.append(("legend", "ellipse", lx + i * 20, ly - 8, lx + i * 20 + 16, ly + 8, hex2rgb(SYS_COLOR[k])))
    it.append(("legend", "text", lx + 66, ly, LEGEND_A[0], FS_A["legend"], INK, False, "lm"))
    gx = lx + 66 + Doc.textlength(LEGEND_A[0], FS_A["legend"]) + 22
    it.append(("legend", "ellipse", gx, ly - 8, gx + 16, ly + 8, UNANNOTATED_DOT))
    it.append(("legend", "text", gx + 24, ly, LEGEND_A[1], FS_A["legend"], INK, False, "lm"))
    gx2 = gx + 24 + Doc.textlength(LEGEND_A[1], FS_A["legend"]) + 30
    it.append(("legend", "text", gx2, ly, LEGEND_A[2], FS_A["legend_note"], GREY, False, "lm"))
    for txt, xc in COLUMN_TITLES_A:
        it.append(("columns", "text", S(xc), S(695), txt, FS_A["column"], INK, False, "mm"))
    return it


def draw_A(doc, ax0, ay0, items):
    """draw panel A overlay items into the SVG document, offset to the base drawing's top-left (ax0, ay0)."""
    for _, kind, *a in items:
        if kind == "line":
            doc.line([(ax0 + x, ay0 + y) for x, y in a[0]], a[1], a[2])
        elif kind == "poly":
            doc.polygon([(ax0 + x, ay0 + y) for x, y in a[0]], a[1])
        elif kind == "ellipse":
            doc.ellipse(ax0 + a[0], ay0 + a[1], ax0 + a[2], ay0 + a[3], a[4])
        else:
            x, y, t, sz, c, b, anc = a
            doc.text(ax0 + x, ay0 + y, t, sz, c, bold=b, anchor=anc)


def main():
    OUT.mkdir(exist_ok=True)
    W, H = M + LW + GAP + CW + M, M + H_LEFT + M
    offA = ((LW - AW) // 2, 12 + (DATA_H - 12 - AH) // 2)        # where panel A's drawing sits inside its frame
    wipe = (int(300 * UP_A) + offA[0], int(95 * UP_A))            # blanks the old "A Data" title baked in the drawing
    xA, yA = M, M
    xB, yB = M, M + DATA_H + GAP
    xC, yC = M + LW + GAP, M

    print("rendering panel B ..."); tileB, svgB = render(panel_B(), DPI_B)
    print("rendering panel C ..."); tileC, svgC = render(panel_C(), DPI_C * 2, size=(WC, HC))
    itemsA = panel_A_items()

    # ---------------------------------------------------------------- PNG / PDF (PIL composition, as the paper build)
    A = BASE_A.resize((AW, AH), Image.LANCZOS)
    d = ImageDraw.Draw(A)
    for _, kind, *a in itemsA:
        if kind == "line":
            d.line(a[0], fill=a[1], width=a[2])
        elif kind == "poly":
            d.polygon(a[0], fill=a[1])
        elif kind == "ellipse":
            d.ellipse(a[:4], fill=a[4])
        else:
            x, y, t, sz, c, b, anc = a
            d.text((x, y), t, font=pil_font(sz, bold=b), fill=c, anchor=anc)

    def frame(img, w, h, top=TITLE_H):
        out = Image.new("RGB", (w, h), "white")
        ImageDraw.Draw(out).rounded_rectangle([3, 3, w - 4, h - 4], radius=26, outline=BORDER, width=4)
        out.paste(img, ((w - img.width) // 2, top + (h - top - img.height) // 2))
        return out
    fig = Image.new("RGB", (W, H), "white"); dd = ImageDraw.Draw(fig)
    fig.paste(frame(A, LW, DATA_H, top=12), (xA, yA))
    fig.paste(frame(tileB, LW, BH), (xB, yB))
    fig.paste(frame(tileC, CW, H_LEFT, top=TITLE_H + 30), (xC, yC))
    dd.rectangle([xA + 12, yA + 10, xA + wipe[0], yA + wipe[1]], fill="white")
    for (x0, y0), (letter, word), it in (((xA, yA), TITLE_A, True), ((xB, yB), TITLE_B, False),
                                         ((xC, yC), TITLE_C, False)):
        dd.text((x0 + 22, y0 + 10), letter, font=pil_font(FS_LETTER, bold=True), fill=INK)
        dd.text((x0 + 88, y0 + 16), word, font=pil_font(FS_TITLE, italic=it), fill=INK)
    dd.text((xB + 88 + dd.textlength(TITLE_B[1], font=pil_font(FS_TITLE)) + 30, yB + 40), SUBTITLE_B,
            font=pil_font(FS_SUB_B), fill=GREY, anchor="lm")
    dd.text((xC + CW / 2, yC + 100), SUBTITLE_C, font=pil_font(FS_SUB_C), fill=GREY, anchor="mm")
    fig.save(OUT / "fig1.png", dpi=(PNG_DPI, PNG_DPI))
    fig.save(OUT / "fig1.pdf", resolution=PNG_DPI)

    # ---------------------------------------------------------------- SVG (same coordinates, live text)
    doc = Doc(W, H, "Figure 1")
    baseA = BASE_A.copy()                                        # same title wipe, in base-image px
    ImageDraw.Draw(baseA).rectangle([0, 0, int((wipe[0] - offA[0]) / UP_A + 1),
                                     int((wipe[1] - offA[1]) / UP_A + 1)], fill="white")
    with doc.group("panelA"):
        doc.frame(xA, yA, LW, DATA_H, gid="panelA_frame")
        ax0, ay0 = xA + offA[0], yA + offA[1]
        doc.image(baseA, ax0, ay0, AW, AH, gid="panelA_base_raster")
        with doc.group("panelA_overlay"):
            for grp in [*SYS, "legend", "columns"]:
                gname = {"legend": "panelA_legend", "columns": "panelA_column_titles"}.get(grp, f"panelA_row_{grp}")
                with doc.group(gname):
                    subs = [None] if grp in ("legend", "columns") else ["arrow", "cells", "types"]
                    for sub in subs:
                        with (doc.group(f"panelA_{grp}_{sub}") if sub else contextlib.nullcontext()):
                            draw_A(doc, ax0, ay0, [x for x in itemsA if x[0] == ((grp, sub) if sub else grp)])
        with doc.group("panelA_title"):
            doc.text(xA + 22, yA + 10, TITLE_A[0], FS_LETTER, INK, bold=True)
            doc.text(xA + 88, yA + 16, TITLE_A[1], FS_TITLE, INK, italic=True)
    with doc.group("panelB"):
        doc.frame(xB, yB, LW, BH, gid="panelB_frame")
        px, py = xB + (LW - tileB.width) // 2, yB + TITLE_H + (BH - TITLE_H - tileB.height) // 2
        doc.mpl(svgB, px, py, DPI_B / 72, "fig1B", gid="panelB_solver_plot")
        with doc.group("panelB_title"):
            doc.text(xB + 22, yB + 10, TITLE_B[0], FS_LETTER, INK, bold=True)
            doc.text(xB + 88, yB + 16, TITLE_B[1], FS_TITLE, INK)
            doc.text(xB + 88 + doc.textlength(TITLE_B[1], FS_TITLE) + 30, yB + 40, SUBTITLE_B, FS_SUB_B, GREY,
                     anchor="lm", gid="panelB_subtitle")
    with doc.group("panelC"):
        doc.frame(xC, yC, CW, H_LEFT, gid="panelC_frame")
        top = TITLE_H + 30
        px, py = xC + (CW - tileC.width) // 2, yC + top + (H_LEFT - top - tileC.height) // 2
        doc.mpl(svgC, px, py, DPI_C / 72, "fig1C", gid="panelC_prepost_plot")   # 1 pt = DPI_C/72 px
        with doc.group("panelC_title"):
            doc.text(xC + 22, yC + 10, TITLE_C[0], FS_LETTER, INK, bold=True)
            doc.text(xC + 88, yC + 16, TITLE_C[1], FS_TITLE, INK)
            doc.text(xC + CW / 2, yC + 100, SUBTITLE_C, FS_SUB_C, GREY, anchor="mm", gid="panelC_subtitle")
    doc.save(OUT / "fig1.svg")
    for f in ("fig1.svg", "fig1.png", "fig1.pdf"):
        print("wrote", OUT / f, f"{(OUT / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
