# ConnectionMiner figures

All paper figures (main and supplementary). Each figure has three things here:

- the **rendered figure** in `out/`, as SVG (editable), PDF and PNG;
- the **script** that draws it;
- the **pre-extracted data** it draws, in `data/`.

Scripts read only their own `data/` folder and `fonts/`. They need no cluster access, no GPU and no solver run,
and they reproduce the figures pixel for pixel.

## Figures

| Figure | Rendered | Script | Content |
|---|---|---|---|
| Fig 1 | [`fig1/out/fig1.pdf`](fig1/out/fig1.pdf) | `fig1/fig1.py` | Data, model and held-out reconstruction for visual / central complex / motor |
| Fig 2 | [`fig2/out/fig2.pdf`](fig2/out/fig2.pdf) | `fig2/fig2.py` | Nulls, connectome-only and gene-based baselines, label-mix recovery |
| Fig 3 | [`fig3/out/fig3.pdf`](fig3/out/fig3.pdf) | `fig3/fig3.py` | Inferred P·G of sister type pairs (Dm8a/b, PEN_a/b, Fe10/Fe11) |
| Fig 4 | [`fig4/out/fig4.pdf`](fig4/out/fig4.pdf) | `fig4/fig4.py` | β gene-pair volcano, R²(k), gene-family × synaptic-edge heatmaps |
| Fig S1 | [`figS1/out/figS1_preprocessing.pdf`](figS1/out/figS1_preprocessing.pdf) | `figS1/figS1.py` | Preprocessing per system; read depth of cells vs metacells |
| Fig S (split) | [`fig2/out/figS_split.pdf`](fig2/out/figS_split.pdf) | `fig2/fig2.py` | Type-level 10-fold train/val/test split |
| Fig S (baseline schematic) | [`figS_baseline/out/figS_baseline_schematic.pdf`](figS_baseline/out/figS_baseline_schematic.pdf) | `figS_baseline/figS_baseline_schematic.py` | How the gene-based baseline works (schematic only, no data) |
| Fig S (baseline, real data) | [`figS_baseline/out/figS_baseline_real.pdf`](figS_baseline/out/figS_baseline_real.pdf) | `figS_baseline/figS_baseline_real.py` | Baseline vs ConnectionMiner step by step on visual, fold 0, test block |

```
figures/
├── fonts/            Liberation Sans (metric-compatible with Arial), used by every script
├── requirements.txt
├── fig1/  fig1.py  data/  out/  extract_data.py  svgkit.py
├── fig2/  fig2.py  data/  out/  extract_data.py  svgkit.py   (also writes figS_split)
├── fig3/  fig3.py  data/  out/  extract_data.py
├── fig4/  fig4.py  data/  out/  extract_data.py
├── figS1/ figS1.py data/  out/  extract_data.py
└── figS_baseline/  figS_baseline_real.py  figS_baseline_schematic.py  data/  out/  extract_data.py
```

## Setup (once)

You need Python 3.10 or newer. Only numpy, scipy, pandas, matplotlib and pillow are used; the solver package is
not needed.

```bash
python -m venv cmfig
source cmfig/bin/activate          # Windows: cmfig\Scripts\activate
pip install -r figures/requirements.txt
```

Optionally, install the four `fonts/*.ttf` files on your system. Inkscape and Illustrator then show the same font
when you open the SVGs. Without them, they fall back to Arial, which has the same metrics.

## Make a figure

```bash
python figures/fig3/fig3.py
python figures/figS_baseline/figS_baseline_real.py
```

Each script writes `out/<name>.svg`, `.pdf` and `.png` next to itself. You can run it from any folder.

| Script | Run time | What you can change offline | Needs the cluster (`extract_data.py`) |
|---|---|---|---|
| `fig1/fig1.py` | ~15 s | colours, all texts, panel A counts/legend, panel B labels and test r, panel C labels, sizes | other highlighted cell types in panel C; panel A drawing (BioRender art) |
| `fig2/fig2.py` | ~17 s | titles, footnote, schematic texts, method labels/colours/order, recovery-plot settings | new runs; the BioRender label-mix drawing and the split panel are fixed images |
| `fig3/fig3.py` | ~5 min | the sister pair shown per system (`python fig3/fig3.py --list` prints the types available offline), shared genes, colours, texts | types outside the offline set (120 visual, 120 CX, 90 motor) |
| `fig4/fig4.py` | ~16 s | FDR level, number of labels, heatmap columns/rows, family colours, texts | family membership, Fisher rule, new solves |
| `figS1/figS1.py` | ~7 s | layout, colours, texts | depth statistics (`extract_data.py`, needs the h5ad files) |
| `figS_baseline/*.py` | ~15 s | layout, colours, texts, which train types are shown | other folds/systems, new baseline runs |

Editable settings are named constants near the top of each script.

## Two ways to change a figure

1. **Change the script** for anything that depends on data or should be reproducible: which cell types
   or genes are shown, colours, labels, panel titles, the number of heatmap columns, thresholds. Re-run the script.
2. **Edit the SVG** in Inkscape (free), Illustrator or Figma for last-minute layout and text changes.
   Every text is live text. In Figs 1–4, every panel is a named group. Re-running the script overwrites `out/`,
   so save hand-edited files under another name.

Do not edit data values (points, bars, heatmap cells) by hand in the SVG. Change the data or the script instead.

## Where the data comes from

`extract_data.py` in each folder is the provenance record. It rebuilds `data/` from the full results on the NYU
HPC cluster: it calls the same loaders as the original figure code and recomputes nothing. It will not run
elsewhere. The results themselves come from the pipelines in this repository (`prepare/`, `connectionminer/`,
`configs/`, `slurm/`; see the top-level README), run with the configurations listed there.

What each `data/` folder holds, in short:

- `fig1/data`: panel-B C / Ĉ tiles and test scores, the panel-C metacell embedding and labels, and the BioRender
  base drawing of panel A.
- `fig2/data`: per-fold test r of the real model and 5 null variants (`nulls_test_r.csv`), per-fold test r of
  every baseline (`c_recon_test_r.csv`), label-mix recovery per pair and per run, and two fixed raster panels.
- `fig3/data`: per system, P·G (all atlas genes) for the offline set of types, plus pair statistics.
- `fig4/data`: volcano tables (β gene pair vs relabelled null), R²(k) curves, gene-family heatmaps and the
  Fisher table.
- `figS1/data`: UMIs and genes per single cell vs per metacell (`depth_stats.npz`) and the preprocessing panel.
- `figS_baseline/data`: visual, fold 0, test block, solver seed 0.
  - `clusters.npz`: Scanpy Leiden units, t-SNE, cell → metacell.
  - `baseline.npz`: the gene-based baseline's training profiles, C, matching, Ĉ and r.
  - `cm.npz`: ConnectionMiner's train/test P, E = P·G, C, Ĉ and r.
  - The full β matrices (gene × gene, ~100 MB) are reduced to what is drawn: the per-gene ranking score and the
    14 × 14 top block.

### Notes

- Fig 2's label-mix recovery panel uses the label-mix runs that existed when Fig 2 v6 was made (2026-11-01). The
  later complete visual run (1,282 pairs) is not yet in this figure. To include it, re-run
  `fig2/extract_data.py` on the cluster.
- The recovery panel in `fig2.py` uses Liberation Sans like every other panel, so `fig2/out/fig2.png` differs
  from the original v6 export in that panel's font only.
