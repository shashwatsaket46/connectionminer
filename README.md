# ConnectionMiner

ConnectionMiner infers cell types in the *Drosophila* nervous system by combining
single-cell gene expression with the connectome. It fits two things jointly:

- **P**: a soft assignment of cells (pooled into metacells) to connectome cell
  types. P is restricted to the types each cell's expression annotation allows.
- **β**: a non-negative gene × gene interaction matrix.

The fit makes the predicted type-to-type wiring match the real connectome C:

```
C  ≈  Ĉ  =  P · G · β · Gᵀ · Pᵀ        G = metacell × gene detection rates
```

This repository runs the best model for each of three systems, using the
held-out-type protocol from the paper:

| dataset | system | cells | types | metacells | solver genes | test r (paper) |
|---|---|---:|---:|---:|---:|---:|
| `visual` | adult optic lobe | 99,656 | 741 | 4,748 | ≤3,000 | **0.65** (0.601 with the default config, see below) |
| `motor`  | adult VNC premotor + motor neurons | 11,292 | 730 | 2,036 | 1,000 (+ complements) | **0.375** |
| `cx`     | central complex | 48,014 | 853 | 2,490 | 2,000 | **0.815** |

## How a run works

1. **Split the types.** Each type goes into one of 10 folds. The deal is
   stratified by named/unnamed (motor also by preMN/MN) and drawn with seed 0.
   For fold *f*, fold *f* is **test**, fold *f+1* is **val**, and the other 8
   folds are **train**.
2. **Train.** Solve β and P on the train types. C, P and G are all cut to the
   train types and to the cells that can still belong to one of them.
3. **Val and test.** Load the train β and **freeze** it: only one scalar
   rescaling of β is refit. Solve P alone on each held-out block.
4. **Score.** Report the Pearson r between Ĉ and the binarised C over each
   block's masked, observed entries.

Every fold is run with 5 solver seeds. The figure numbers are means over the
10 folds × 5 seeds.

## Install

```bash
git clone https://github.com/shashwatsaket46/connectionminer.git
cd connectionminer
conda env create -f environment.yml
conda activate connectionminer
pytest tests/          # ~15 s, synthetic data, no download needed
```

`environment.yml` installs PyTorch with CUDA 13 wheels. On a CPU-only machine
the solver runs on CPU automatically, just more slowly.

## Data

**The data is not in this repository.** Each dataset ships as a solver-ready
*bundle*, one archive per dataset, distributed separately:

| archive | size |
|---|---:|
| `connectionminer_data_visual.tar.xz` | 26 MB |
| `connectionminer_data_motor.tar.xz` | 1.1 MB |
| `connectionminer_data_cx.tar.xz` | 3.9 MB |

The archives are available from the authors on request (ss20308@nyu.edu). Unpack them into `data/` and check them
against `DATA_SHA256SUMS`:

```bash
sha256sum -c DATA_SHA256SUMS            # run where the archives are
mkdir -p data
for d in visual motor cx; do tar -xJf connectionminer_data_$d.tar.xz -C data; done
ls data/                                # visual/  motor/  cx/
```

Each bundle directory holds:

| file | contents |
|---|---|
| `C.npz` | type × type connectome; NaN = unobserved |
| `C_mask.npy` | entries that are fitted and scored |
| `P_constraints_cells.npz` | type × cell: types each cell may be |
| `X_cells.npz` | cell × gene binary detection |
| `genes.csv`, `types.csv`, `cells.csv` | gene names; type names and fold strata; metacell (and batch) per cell |
| `bundle.json` | counts and provenance |

The format is documented in `connectionminer/data.py`.

## Run

The commands are the same for every dataset; only the config name changes.
Results go to `runs/<dataset>/fold<f>/`.

### Visual system

```bash
python -m connectionminer run visual --fold 0                # one fold, seeds 0-4
python -m connectionminer run visual                         # all 10 folds
python -m connectionminer summarize visual
```

- Config: `configs/visual.yaml`.
- The bundle carries the 3,000 dispersion-binned HVGs. On each train block the
  solver keeps the genes detected in at least `genes.min_cells` train cells,
  ordered by detection variance. Val and test reuse train's genes.
- **Paper model.** The paper's visual results (test r = 0.65) use
  `min_cells: 5`, which drops genes detected in fewer than 5 train cells. The
  config ships with `min_cells: 1` (test r = 0.601), the model used for the
  paper's null-shuffle and gene-pair (β) analyses. To reproduce the main
  visual result, set `min_cells: 5` in `configs/visual.yaml`; nothing else
  changes.
- About 25 min per seed per fold on a shared A100/H200 (train ≈ 12 s/iteration).
  Use a GPU: on CPU this dataset takes hours per seed.

### Motor system

```bash
python -m connectionminer run motor --fold 0
python -m connectionminer run motor
python -m connectionminer summarize motor
```

- Config: `configs/motor.yaml`.
- Uses the 1,000 dispersion-binned genes plus their complements (1 − G), so a
  gene's absence also carries signal.
- Solver seeds are 750 + seed.
- About 4 min per seed per fold on a GPU, or roughly 10× that on CPU. CPU is
  practical for this dataset.

### Central complex

```bash
python -m connectionminer run cx --fold 0
python -m connectionminer run cx
python -m connectionminer summarize cx
```

- Config: `configs/cx.yaml`.
- Uses 2,000 batch-aware dispersion genes. Each cell's detection is reweighted
  by the gene's global-to-batch detection ratio. Weights come from the block's
  own cells, with 10 pseudo-cells of shrinkage and capped to [1/5, 5].
- About 10 min per seed per fold on a GPU, or about 25 min on 4 CPU cores.

### Options and cluster use

- `--seeds 0 1`: run a subset of seeds.
- `--num-iter 5`: quick smoke run.
- `--device cpu` or `--device cuda`.
- `--data DIR` and `--out DIR`: non-default locations.
- Finished splits are skipped, so a killed job can be resubmitted as-is.

On SLURM, one fold per array task:

```bash
sbatch --array=0-9 slurm/run_fold.sbatch visual     # edit partition/GPU lines first
```

### Outputs

For each fold `runs/<dataset>/fold<f>/` holds:

- `results.csv`: r per seed and split.
- `split.csv`: which types went to train, val and test.
- `seed<s>/<split>/`, one directory per solve:

| file | contents |
|---|---|
| `C_hat.npy` | predicted type × type wiring for the block |
| `P.npz` | type × metacell assignment |
| `beta.npy` | train: the fitted β. val/test: the frozen β times its fitted scalar |
| `block.npz` | which types, cells and genes the block used, and each cell's metacell |
| `loss.csv` | objectives per iteration |

## Reproducibility

- **Precision.** The solver runs in float64. float32 is not supported: β's
  entries reach about 1e-13, and float32 collapses the train fit (r 0.67 → 0.10).
- **Inputs.** `prepare/verify_reference.py` rebuilds every fold's solver inputs
  (types, genes, constraints and the G matrix) and compares them with the
  paper's original runs. All 3 datasets × 10 folds × 3 splits match bit for bit.
- **Solver.** It produces bit-identical P, β and Ĉ to the original
  implementation on the same device and seed, for train solves and for
  frozen-β solves.
- **CPU vs GPU.** The two draw different random numbers for β's initial value,
  so the same seed gives a different but equally valid solve on each.
  Fold-level r varies across seeds by about 0.005.
- **Thread count.** The thread count changes floating-point summation order.
  `run` defaults to 4 BLAS threads; set `threads:` in the config to change it.
- **Time limit.** `time_limit_per_step` (60 s) caps each inner loop by wall
  clock. A GPU never reaches it. Raise it if you run the visual dataset on a
  slow CPU.

## Rebuilding the bundles (lab members)

`prepare/export_bundle.py <dataset> --out data/<dataset>` writes a bundle from
the original build outputs on the cluster. These are the inputs the Fig 1 runs
read:

| dataset | source |
|---|---|
| visual | `cm_active/devstages/Adult/output`, its `grid/hvg3000/…` gene set and metacells, and `devstages/_h5ad/Adult.h5ad` |
| motor | `cm_motor/output_tp3`, with the top-1,000 dispersion-binned gene set |
| cx | `cm_central_complex/grid_mc1/runs/g5_h2000_bn1_m1` |

Building those directories from the raw atlases and connectomes happens in the
per-system build pipelines. That covers annotation, metacell clustering and
gene selection, and it is not part of this repository.

## Figures

`figures/` holds every paper figure (Figs 1–4 and the supplementary figures),
rendered as SVG/PDF/PNG, together with the scripts and the small pre-extracted
data that redraw them on any laptop (numpy, scipy, pandas, matplotlib, pillow;
no solver, no GPU). See [`figures/README.md`](figures/README.md).

```bash
pip install -r figures/requirements.txt
python figures/fig1/fig1.py      # -> figures/fig1/out/fig1.{svg,pdf,png}
```

## Layout

```
connectionminer/
  data.py        bundle format, load/save
  split.py       type k-fold; per-block metacell aggregation (G, constraints)
  solver.py      the alternating P / β solver (PyTorch, float64)
  pipeline.py    train → frozen-β val/test → score, per fold; summarize
  cli.py         python -m connectionminer {run, summarize}
configs/         visual.yaml, motor.yaml, cx.yaml: the best settings per system
prepare/         export_bundle.py (lab builds → bundles), verify_reference.py
slurm/           run_fold.sbatch
tests/           synthetic smoke tests
figures/         paper figures: scripts, pre-extracted data, rendered outputs
```
