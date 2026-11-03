"""CLUSTER-ONLY: extract the minimal inputs of paper Figure 4 into figure_kit/fig4/data/.

Run once on the cluster (needs the cm_active tree); fig4.py then needs only data/ and ../fonts/.

    PYTHONPATH=/scratch/ss20308/python-user-lib/python3.13/site-packages python3 figure_kit/fig4/extract_data.py

Provenance (reference build: biorender_fig/fig4/fig4_v5.py -> biorender_fig/figures/fig4_v5.{png,svg}):

panel a, per-system volcano  (data/volcano_<system>.csv.gz, data/volcano_<system>_meta.json)
    pbnull/beta_fold_paired_log2/<label>/pairs.csv.gz + meta.json  (pbnull/beta_fold_paired.py, log2 mode):
    every TESTED directed gene pair (pair in >= 3 folds' gene sets, real or null beta > 0 in some fold,
    non-constant fold differences). effect = 10-fold mean log2(1+beta) - log2(1+beta_null) ("diff"),
    p = two-sided paired t over the folds, q = BH over ALL tested pairs of the system (as in the file).
    family_class = class of the pair when BOTH genes are in a gene family
    (data/gene_list/families/atlas_map_<label>.tsv; classes TF-TF / CAM-CAM / NP-NPR / cross), else "".
    named = neither gene matches the anonymous-locus regex (CG/CR numbers, long numbers, FBti, '~' complements).
panel a, Combined (Fisher)   (data/fisher_combined.csv.gz)
    fig4_volcano_paired_full.fisher() over the three systems' NAMED tested pairs (named = its own regex),
    one-sided P (real > null) combined by Fisher over the systems whose solver genes hold both genes (k >= 2),
    effect = mean over those systems. q is recomputed in fig4.py (BH at the chosen FDR level).
panel b, R^2(k)              (data/r2k_<system>.npz)
    biorender_fig/fig4/cache/r2k_<label>.npz (fig4_r2k.py): k grid, r2 [50 train solves x k], r2_full [50].
panel c, heatmaps            (data/heat_<system>.npz)
    biorender_fig/fig4/cache/<label>_fam.npz (FIG4_GENESET=fam fig4_heatmap_data.py), with the 'cum' array
    and per-row extras dropped: share [synaptic type pairs x candidate family gene-pair columns] (float64, exact),
    row/column labels, column family class, mean share, volcano evidence (vdiff, vp, vq), header numbers.
    All candidate columns (177-200) are kept so the column count can be changed in fig4.py.
icons                        (data/cns_icons.png) = biorender_fig/fig4/br/icons.png (BioRender CNS drawings).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"
REPO = Path("/scratch/ss20308/cm_active")
F4 = REPO / "biorender_fig" / "fig4"
PB = REPO / "pbnull" / "beta_fold_paired_log2"
FAM = REPO / "data" / "gene_list" / "families"
SYSTEMS = [("visual", "Adult_nofilt", "Adult_nofilt"), ("motor", "motor_tp3", "motor_tp3"),
           ("cx", "CXbestmc1", "CX")]            # (key in figure_kit, heatmap/R2 build, volcano build)
ANON = re.compile(r"\d{4,}|^(CG|CR)\d|:CR\d|^FBti|^~")      # fig4_v5.ANON

sys.path.insert(0, str(F4))
from fig4_heatmap_data import pair_class  # noqa: E402
from fig4_volcano_paired_full import fisher, load as vload  # noqa: E402


def cls_group(c):
    return c if c in ("TF–TF", "CAM–CAM", "NP–NPR") else "cross"


def main():
    OUT.mkdir(exist_ok=True)
    per = {}
    for key, hlab, vlab in SYSTEMS:
        meta = json.loads((PB / vlab / "meta.json").read_text())
        d = pd.read_csv(PB / vlab / "pairs.csv.gz")
        d = d[d.tested].reset_index(drop=True)
        A = pd.read_csv(FAM / f"atlas_map_{vlab}.tsv", sep="\t", keep_default_na=False)
        fam = {g: set(f.split(",")) for g, f in zip(A.atlas_gene, A.families) if f}
        fi = [fam.get(g, set()) for g in d.gene_i]
        fj = [fam.get(g, set()) for g in d.gene_j]
        cls = [cls_group(pair_class(a, b)) if (a and b) else "" for a, b in zip(fi, fj)]
        named = [not (ANON.search(a) or ANON.search(b)) for a, b in zip(d.gene_i, d.gene_j)]
        t = pd.DataFrame({"gene_i": d.gene_i, "gene_j": d.gene_j, "effect": d["diff"], "p": d.p, "q": d.q,
                          "family_class": cls, "named": np.array(named, int)})
        t.to_csv(OUT / f"volcano_{key}.csv.gz", index=False, float_format="%.17g")
        (OUT / f"volcano_{key}_meta.json").write_text(json.dumps(meta, indent=1))
        per[vlab] = vload(vlab, "beta_fold_paired_log2")
        print(key, len(t), "tested;", int((t.q < 0.05).sum()), "BH q<0.05")

        z = np.load(F4 / "cache" / f"r2k_{hlab}.npz")
        np.savez_compressed(OUT / f"r2k_{key}.npz", k=z["k"], r2=z["r2"], r2_full=z["r2_full"])

        h = np.load(F4 / "cache" / f"{hlab}_fam.npz", allow_pickle=True)
        keep = ["share", "col_pre", "col_post", "col_class", "col_mean_share", "col_vdiff", "col_vp", "col_vq",
                "n_fam_genes", "n_fam_pairs", "n_fam_up", "fam_total_share"]
        o = {k: h[k] for k in keep}
        o["row_pre"] = h["row_pre"].astype(str)
        o["row_post"] = h["row_post"].astype(str)
        o["col_pre"], o["col_post"] = o["col_pre"].astype(str), o["col_post"].astype(str)
        np.savez_compressed(OUT / f"heat_{key}.npz", **o)
        print(key, "heatmap share", o["share"].shape)

    c = fisher(per)
    c[["gene_i", "gene_j", "k", "diff", "p"]].rename(columns={"diff": "effect"}).to_csv(
        OUT / "fisher_combined.csv.gz", index=False, float_format="%.17g")
    print("fisher", len(c), "pairs,", int((c.q < 0.05).sum()), "BH q<0.05")
    shutil.copy(F4 / "br" / "icons.png", OUT / "cns_icons.png")


if __name__ == "__main__":
    main()
