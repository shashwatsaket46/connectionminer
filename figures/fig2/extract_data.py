#!/usr/bin/env python3
"""CLUSTER-ONLY: pre-extract every input of the portable Figure 2 kit into figure_kit/fig2/data/.

Run on the cluster (needs the original run directories); fig2.py itself then reads ONLY data/ and ../fonts/.

Provenance (nothing is re-computed; the same loaders as the original build are called):
  nulls_test_r.csv     <- /scratch/ss20308/cm_central_complex/pbnull/plot_pub_three.py: load()
                          (as called by biorender_fig/fig2/fig2_nulls_v5.py: load_nulls()). Per system, per row
                          (real model + 5 re-solve null variants), per fold: mean test Pearson r over the test split.
                          Builds: visual = "Adult" (min_cells=5), CX = reference build, motor = t3.
  c_recon_test_r.csv   <- cm_active/baselines/plot_c_recon_combined.py (connectome(), pit.tool_r(.., "real"),
                          pit.ours()), data in cm_active/baselines/runs/. Per system, per method, per fold test r,
                          in the array order the original plot uses (rows sorted by fold; NaN kept).
  recovery_pairs.csv   <- cm_active/labelmix/plot_recovery.py: per (system, pair) mean recovery_a / recovery_b /
                          balanced and seed count, row order = the groupby order the original sample() acts on.
  recovery_runs.csv    <- same, every run's balanced recovery (visual plot draws each seed as its own point).
  br_mix_v2.png        <- cm_active/biorender_fig/fig2/br_mix_v2.png (BioRender export, raster)
  crop_split.png       <- cm_active/biorender_fig/fig2/crop_split.png (train/val/test split panel, raster)
CSV floats are written at full (round-trip) precision; fig2.py reads them with float_precision="round_trip" so the
recovery plot's ">= mean" selection and seeded sample() pick exactly the same pairs as the original.

Usage: python figure_kit/fig2/extract_data.py
"""
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
CMA = Path("/scratch/ss20308/cm_active")
PB = Path("/scratch/ss20308/cm_central_complex/pbnull")
SYS_KEY = {"Visual": "visual", "Central complex": "cx", "Motor": "motor"}


def nulls():
    ROWKEYS = ["real", "P_by_metacell", "P_by_type_metacell", "P_by_type", "beta_by_gene", "beta_by_entry"]
    sys.path.insert(0, str(PB))
    cwd = os.getcwd(); os.chdir(PB)
    import plot_pub_three as ppt
    data = ppt.load()
    os.chdir(cwd)
    rows = []
    for name, _col, vals in data:
        assert len(vals) == len(ROWKEYS)
        for key, v in zip(ROWKEYS, vals):
            for f, r in enumerate(np.asarray(v, float)):
                rows.append(dict(system=SYS_KEY[name], row=key, fold=f, r=r))
    pd.DataFrame(rows).to_csv(OUT / "nulls_test_r.csv", index=False)


def c_recon():
    bdir = CMA / "baselines"
    sys.path.insert(0, str(bdir))
    cwd = os.getcwd(); os.chdir(bdir)
    import plot_c_recon_combined as pcc
    import plot_indep_tools as pit
    import plot_like_for_like as lfl
    KEY = {"Degree (count, in-sample)": "degree_count_insample", "Degree (count, leave-out)": "degree_count_leaveout",
           "Arbor overlap 10 µm": "arbor_10um", "Constraint-only P (CM β)": "constraint_only_P_cm_beta",
           "Combined": "combined_no_degree"}
    rows = []
    for s, _name in lfl.SYSTEMS:
        ser = [("connectome", KEY[lab], held, v) for lab, _c, held, v in pcc.connectome(s)]
        ser += [("gene", t, False, pit.tool_r(s, t, "real").sort_index().to_numpy()) for t, _l, _c in pit.TOOLS]
        ser += [("ours", "connectionminer", False, pit.ours(s).sort_index().to_numpy())]
        for grp, key, held, v in ser:
            for i, r in enumerate(np.asarray(v, float)):
                rows.append(dict(system=s, group=grp, method=key, hatched=int(bool(held)), idx=i, r=r))
    os.chdir(cwd)
    pd.DataFrame(rows).to_csv(OUT / "c_recon_test_r.csv", index=False)


def recovery():
    runs = CMA / "labelmix" / "runs"
    rows = []
    for s in ("visual", "cx", "motor"):
        for f in glob.glob(str(runs / s / "res_*.csv")):
            rows.append(pd.read_csv(f).assign(system=s))
    d = pd.concat(rows, ignore_index=True)
    d["seed"] = d["seed"].fillna(0).astype(int) if "seed" in d else 0
    d["balanced"] = (d.recovery_a + d.recovery_b) / 2
    g = d.groupby(["system", "pair"], as_index=False).agg(a=("recovery_a", "mean"), b=("recovery_b", "mean"),
                                                          bal=("balanced", "mean"), n_seed=("seed", "nunique"))
    g.to_csv(OUT / "recovery_pairs.csv", index=False)
    d[["system", "pair", "seed", "recovery_a", "recovery_b", "balanced"]].to_csv(
        OUT / "recovery_runs.csv", index=False)


if __name__ == "__main__":
    nulls(); c_recon(); recovery()
    for f in ("br_mix_v2.png", "crop_split.png"):
        shutil.copy2(CMA / "biorender_fig" / "fig2" / f, OUT / f)
    for f in sorted(OUT.iterdir()):
        print(f"{f.stat().st_size:>10d}  {f.name}")
