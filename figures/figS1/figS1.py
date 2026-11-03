#!/usr/bin/env python3
"""Supplementary Figure S1: preprocessing (moved out of Fig 2C) and why metacells help (read depth).

a  Preprocessing per system: mask D, metacells, HVGs, G (the panel that was Fig 2C left; fig2/crop_prep.png).
b  Read depth of single cells vs metacells per system: UMIs per unit, genes detected per unit (metacell = sum of the
   raw counts of its member cells), and cells per metacell. Numbers from figS1/s1_depth_stats.py (depth_stats.npz).

Usage: python biorender_fig/figS1/s1_depth_stats.py   (once, ~20 s, sparse)
       python biorender_fig/figS1/figS1_preprocessing.py  -> figures/figS1_preprocessing.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
for f in (ROOT / "fonts").glob("LiberationSans-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.minor.width": 0.4, "xtick.minor.size": 1.5, "axes.edgecolor": "#444444"})
INK, SOFT, CELL = "#272727", "#666666", "#9A9A9A"
SYSTEMS = [("visual", "Visual", "#0072B2"), ("cx", "Central complex", "#009E73"), ("motor", "Motor", "#E69F00")]

z = np.load(DATA / "depth_stats.npz")
W, H = 7.2, 9.45
fig = plt.figure(figsize=(W, H), dpi=300)

# ---- a: preprocessing schematic (Fig 2 v3 panel C left, without its old frame and title)
prep = Image.open(DATA / "crop_prep.png").convert("RGB").crop((14, 92, 1320, 748))
ah = 3.25
axa = fig.add_axes([0.07, 1 - (0.30 + ah) / H, 0.90, ah / H])
axa.imshow(np.asarray(prep), interpolation="lanczos"); axa.axis("off")
fig.text(0.012, 1 - 0.12 / H, "a", fontsize=10, fontweight="bold", va="top")
fig.text(0.045, 1 - 0.135 / H, "Preprocessing: annotation mask D, metacells, highly variable genes, expression G",
         fontsize=8, va="top", color=INK)

# ---- b: read depth
top_b = 0.30 + ah + 0.22
fig.text(0.012, 1 - top_b / H, "b", fontsize=10, fontweight="bold", va="top")
fig.text(0.045, 1 - (top_b + 0.015) / H,
         "Metacells raise read depth: single cells vs metacells (summed counts of member cells)",
         fontsize=8, va="top", color=INK)
rows = [("umi", "UMIs per unit", True), ("genes", "genes detected per unit", True), ("size", "cells per metacell", False)]
x0, colw, cgap = 0.085, 0.27, 0.045
y_top = top_b + 0.77
rh, rgap = 1.05, 0.50
for k, (key, name, col) in enumerate(SYSTEMS):
    cu, mu = z[f"{key}__cell_umi"], z[f"{key}__mc_umi"]
    cg, mg = z[f"{key}__cell_genes"], z[f"{key}__mc_genes"]
    sz, ng = z[f"{key}__mc_size"], int(z[f"{key}__n_genes"])
    xl = x0 + k * (colw + cgap)
    fig.text(xl + colw / 2, 1 - (y_top - 0.50) / H, name, ha="center", va="bottom", fontsize=8.5,
             fontweight="bold", color=col)
    fig.text(xl + colw / 2, 1 - (y_top - 0.47) / H,
             f"{len(cu):,} cells → {len(mu):,} metacells", ha="center", va="top", fontsize=7, color=INK)
    fig.text(xl + colw / 2, 1 - (y_top - 0.32) / H, f"{ng:,} genes in the count matrix", ha="center", va="top",
             fontsize=6, color=SOFT)
    for r, (what, ylab, logx) in enumerate(rows):
        yb = y_top + r * (rh + rgap) + rh
        ax = fig.add_axes([xl, 1 - yb / H, colw, rh / H])
        if what == "size":
            b = np.arange(0.5, min(sz.max(), 80) + 2.5, 1 if sz.max() < 30 else 2)
            ax.hist(np.clip(sz, None, b[-1] - 0.01), bins=b, color=col, alpha=0.75, lw=0)
            ax.axvline(np.median(sz), color=INK, lw=0.7, ls=(0, (2, 1.5)))
            ax.set_title(f"median {np.median(sz):.0f}, mean {sz.mean():.1f} cells", fontsize=6.2, color=INK, pad=2)
            ax.set_xlabel("cells per metacell" + (f" (last bin ≥ {b[-1] - 0.5:.0f})" if sz.max() > b[-1] else ""),
                          fontsize=6.5, labelpad=1)
        else:
            c, m = (cu, mu) if what == "umi" else (cg, mg)
            lo = max(1, min(c[c > 0].min(), m[m > 0].min()))
            bins = np.logspace(np.log10(lo), np.log10(max(c.max(), m.max())), 60)
            # density per log-bin so cells (n large) and metacells (n small) share one axis
            for v, cc, lab, a in ((c, CELL, "single cells", 0.6), (m, col, "metacells", 0.7)):
                ax.hist(v[v > 0], bins=bins, color=cc, alpha=a, lw=0, density=False,
                        weights=np.full((v > 0).sum(), 1 / len(v)), label=lab)
                ax.axvline(np.median(v), color=cc if cc != CELL else "#555555", lw=0.8, ls=(0, (2, 1.5)))
            fold = np.median(m) / np.median(c)
            ax.set_title(f"median: cell {np.median(c):,.0f} → metacell {np.median(m):,.0f}  (×{fold:.1f})",
                         fontsize=6.2, color=INK, pad=2)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(ylab.replace(" per unit", ""), fontsize=6.5, labelpad=1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
            if k == 2 and r == 0:
                ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), fontsize=6, frameon=False, handlelength=1.0, borderaxespad=0.1)
        ax.tick_params(labelsize=6, pad=1.5)
        if k == 0:
            ax.set_ylabel({"umi": "% of units", "genes": "% of units", "size": "metacells"}[what], fontsize=6.5,
                          labelpad=2)

fig.text(0.012, 0.012,
         "UMIs = total counts per unit; genes detected = genes with count > 0. Metacell profile = sum of member-cell "
         "counts over all genes of the matrix (not only HVGs).\nVisual: Adult atlas, glia (G_*) removed; CX: T2 atlas, "
         "cells of the best build; motor: adult t3, preMN + MN source tables are already corrected "
         "(\"cg_corrected\", ~1/3 non-integer entries),\nso motor 'UMIs' are corrected counts. Gene universes differ between systems; "
         "compare cells vs metacells within a system.",
         fontsize=5.6, color=SOFT, va="bottom", linespacing=1.3)

out = HERE / "out" / "figS1_preprocessing"
fig.savefig(f"{out}.png", dpi=300); fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.svg")
print("saved", out)
