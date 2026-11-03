"""Schematic: how the gene-based baselines (standard single-cell tools + connectome-guided matching) and
ConnectionMiner reconstruct the connectivity of held-out cell types. Steps that use the held-out connectome are
marked in red.

    python biorender_fig/figS_baseline_schematic.py  ->  figures/figS_baseline_schematic.{svg,pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
for f in (HERE.parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Liberation Sans", "svg.fonttype": "none", "pdf.fonttype": 42})

INK, GREY, LIGHT = "#222222", "#8a8a8a", "#d9d9d9"
ANN = ["#0072B2", "#E69F00", "#009E73"]          # annotated types
RED, BASE_BG, CM_BG = "#C0392B", "#F4F4F4", "#EAF2FB"
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(13.5, 7.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 135); ax.set_ylim(0, 70); ax.axis("off")


def box(x, y, w, h, title, body, held=False, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc,
                                ec=RED if held else "#555555", lw=2.2 if held else 1.0))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + 1.2, body, ha="center", va="bottom", fontsize=8.3, color=INK, linespacing=1.25)
    if held:
        ax.text(x + w - 0.6, y + h + 0.9, "uses held-out C", ha="right", va="bottom", fontsize=7.8, color=RED,
                weight="bold")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#444444"))


def cells(cx, cy, n_ann=3, clusters=False, metacells=False):
    """little scatter: annotated cells coloured, unannotated grey (optionally ringed into clusters/metacells)."""
    for k in range(n_ann):
        p = rng.normal([cx - 4 + 4 * k, cy + 3], 0.7, (9, 2))
        ax.scatter(*p.T, s=10, color=ANN[k], zorder=3, lw=0)
    centers = [(cx - 4.5, cy - 2.5), (cx, cy - 3), (cx + 4.5, cy - 2.2)]
    for i, c in enumerate(centers):
        p = rng.normal(c, 0.8, (10, 2))
        ax.scatter(*p.T, s=10, color=GREY, zorder=3, lw=0)
        if clusters:
            ax.add_patch(plt.Circle(c, 2.0, fill=False, ec="#555555", lw=0.9, ls="--", zorder=2))
            ax.text(c[0], c[1] - 2.9, f"U{i + 1}", ha="center", va="top", fontsize=7, color="#555555")
        if metacells:
            for dx in (-0.8, 0.8):
                ax.add_patch(plt.Circle((c[0] + dx, c[1]), 0.95, fill=False, ec="#555555", lw=0.7, zorder=2))


def strip(x, y, w=9.0, h=1.4, cmap="Blues", seed=0):
    """a gene-expression profile drawn as a 1 x n heat strip (genes along x)."""
    v = np.random.default_rng(seed).random(12)
    for j, val in enumerate(v):
        ax.add_patch(Rectangle((x + j * w / 12, y), w / 12, h, fc=plt.get_cmap(cmap)(0.15 + 0.8 * val), ec="white", lw=0.3))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#555555", lw=0.7))


def dots(cx, cy, color, n=6, ring=None):
    p = rng.normal([cx, cy], 0.55, (n, 2))
    ax.scatter(*p.T, s=8, color=color, zorder=3, lw=0)
    if ring:
        ax.add_patch(plt.Circle((cx, cy), 1.5, fill=False, ec="#555555", lw=0.8, ls="--", zorder=2))
        ax.text(cx, cy - 2.0, ring, ha="center", va="top", fontsize=6.5, color="#555555")


def matrix(x, y, s, n=6, fill=None, ec="#555555"):
    m = rng.random((n, n)) < 0.3 if fill is None else fill
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * s / n, y + (n - 1 - i) * s / n), s / n, s / n,
                                   fc=INK if m[i, j] else "white", ec=LIGHT, lw=0.3))
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=ec, lw=1.0))


# ---------------------------------------------------------------- row backgrounds and titles
for y0, h, bg, title in ((37, 31, BASE_BG, "Gene-based baselines: standard single-cell tools + connectome-guided matching"),
                         (2, 31, CM_BG, "ConnectionMiner")):
    ax.add_patch(FancyBboxPatch((1, y0), 133, h, boxstyle="round,pad=0,rounding_size=2", fc=bg, ec="none"))
    ax.text(3, y0 + h - 1.5, title, ha="left", va="top", fontsize=12.5, weight="bold", color=INK)

W, H = 23.5, 22
xs = [3, 29.5, 56, 82.5, 109]
yb, yc = 39.5, 4.5

# ---------------------------------------------------------------- baseline row
box(xs[0], yb, W, H, "1  Cells", "annotated cells keep their label;\nunannotated cells (grey)\nhave no cell type")
cells(xs[0] + W / 2, yb + 12.5)
box(xs[1], yb, W, H, "2  Clustering", "Seurat / Scanpy / Cell Ranger group\nunannotated cells into clusters\nU1, U2, … (type unknown)")
cells(xs[1] + W / 2, yb + 12.5, clusters=True)
box(xs[2], yb, W, H, "3  Profiles + gene rule", "profile = mean log-expression of the\nunit's cells (2,000 HVGs); β fitted on\nANNOTATED training types (~65 / 16 / 9)")
x0 = xs[2] + 2.5
dots(x0 + 1.6, yb + 15.6, ANN[0]); arrow(x0 + 3.4, yb + 15.6, x0 + 6.2, yb + 15.6)
strip(x0 + 6.6, yb + 14.9, seed=1)
ax.text(x0 + 11.1, yb + 17.0, "annotated type a: its labelled cells", ha="center", fontsize=6.6, color="#555555")
dots(x0 + 1.6, yb + 10.9, GREY, ring="U2"); arrow(x0 + 3.4, yb + 10.9, x0 + 6.2, yb + 10.9)
strip(x0 + 6.6, yb + 10.2, cmap="Greys", seed=2)
ax.text(x0 + 11.1, yb + 12.3, "unannotated type X: its matched cluster", ha="center", fontsize=6.6, color="#555555")
ax.text(xs[2] + W / 2, yb + 7.4, "Ĉ(a→X) = profile$_a$ β profile$_X^{\\top}$", ha="center", va="center", fontsize=8.8, color=INK)
box(xs[3], yb, W, H, "4  Matching", "each unannotated test type gets\nONE cluster, chosen to maximize\nr with the held-out connectome", held=True)
for i, (t, u) in enumerate((("type X", "U2"), ("type Y", "U1"), ("type Z", "U2"))):
    yy = yb + 15.5 - 3.0 * i
    ax.text(xs[3] + 6.5, yy, t, ha="right", va="center", fontsize=8.5, color=INK)
    arrow(xs[3] + 7.2, yy, xs[3] + 13.0, yy)
    ax.text(xs[3] + 13.6, yy, u, ha="left", va="center", fontsize=8.5, color="#555555")
box(xs[4], yb, W, H, "5  Predict + score", "Ĉ for every test pair from their\nprofiles; score r(Ĉ, C) on test types")
matrix(xs[4] + 4.5, yb + 7.5, 7.0); matrix(xs[4] + 12.5, yb + 7.5, 7.0)
ax.text(xs[4] + 11.75, yb + 11, "~", ha="center", va="center", fontsize=13)
ax.text(xs[4] + 8, yb + 15.6, "Ĉ", ha="center", fontsize=9); ax.text(xs[4] + 16, yb + 15.6, "C", ha="center", fontsize=9)

# ---------------------------------------------------------------- ConnectionMiner row
box(xs[0], yc, W, H, "1  Metacells", "cells aggregated into metacells\nwithin clusters; annotated\nmetacells pinned to their type", fc="white")
cells(xs[0] + W / 2, yc + 12.5, metacells=True)
box(xs[1], yc, W, H, "2  Soft assignment P", "every metacell spreads over its\ncandidate types (transport plan);\nannotated stay with their type")
for i in range(3):
    for j in range(5):
        v = rng.random() if j > 1 else (1.0 if i == j else 0.0)
        ax.add_patch(Rectangle((xs[1] + 6 + j * 2.4, yc + 9.5 + (2 - i) * 2.4), 2.4, 2.4, fc=plt.cm.Blues(0.15 + 0.8 * v),
                               ec="white", lw=0.5))
ax.text(xs[1] + 5.5, yc + 13.1, "types", rotation=90, ha="right", va="center", fontsize=7.5, color="#555555")
ax.text(xs[1] + 12, yc + 8.6, "metacells", ha="center", va="top", fontsize=7.5, color="#555555")
box(xs[2], yc, W, H, "3  Profiles + gene rule β", "profile E$_t$ = P$_t$G: detection of the metacells\nweighted by P; β learned jointly with P on\nALL training types (~590 / 680 / 580)")
x0 = xs[2] + 2.5
for k, dx in enumerate((0.0, 2.2, 4.4)):
    ax.add_patch(plt.Circle((x0 + 1.2 + dx, yc + 15.6), 0.9, fill=False, ec="#555555", lw=0.7))
    ax.text(x0 + 1.2 + dx, yc + 17.2, ("0.6", "0.3", "0.1")[k], ha="center", fontsize=6.3, color="#1f4e8c")
arrow(x0 + 6.4, yc + 15.6, x0 + 8.5, yc + 15.6)
strip(x0 + 8.8, yc + 14.9, w=8.6, seed=3)
ax.text(x0 + 4.2, yc + 13.0, "metacells, weights P$_{tk}$", ha="center", fontsize=6.6, color="#555555")
ax.text(x0 + 13.1, yc + 13.0, "profile E$_t$ of type t", ha="center", fontsize=6.6, color="#555555")
ax.text(xs[2] + W / 2, yc + 9.0, "Ĉ = E β Eᵀ = P G β Gᵀ Pᵀ", ha="center", va="center", fontsize=9.5, color=INK)
box(xs[3], yc, W, H, "4  Test types", "β frozen; P solved for the test\ntypes by Sinkhorn-projected\nmirror descent on the held-out C", held=True)
ax.text(xs[3] + W / 2, yc + 12.5, "soft, metacell-level\n(many metacells per type)", ha="center", va="center", fontsize=8.5,
        color="#555555")
box(xs[4], yc, W, H, "5  Predict + score", "Ĉ = P G β Gᵀ Pᵀ on test types\nscore r(Ĉ, C)")
matrix(xs[4] + 4.5, yc + 7.5, 7.0); matrix(xs[4] + 12.5, yc + 7.5, 7.0)
ax.text(xs[4] + 11.75, yc + 11, "~", ha="center", va="center", fontsize=13)
ax.text(xs[4] + 8, yc + 15.6, "Ĉ", ha="center", fontsize=9); ax.text(xs[4] + 16, yc + 15.6, "C", ha="center", fontsize=9)

for y in (yb + H / 2, yc + H / 2):
    for i in range(4):
        arrow(xs[i] + W + 0.4, y, xs[i + 1] - 0.4, y)

ax.text(67.5, 0.6, "Red frame: the step uses the held-out connectome. In the baselines, a rule fitted on shuffled profiles "
        "scores almost the same, so their r comes mostly from matching.", ha="center", va="bottom", fontsize=8.3,
        color="#555555")

out = HERE / "out" / "figS_baseline_schematic"
for ext in ("svg", "pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=300 if ext == "png" else None, facecolor="white")
print("saved", out)
