"""Command line:  python -m connectionminer {run,summarize} ..."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def load_config(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = REPO / "configs" / f"{name_or_path}.yaml"
    return yaml.safe_load(p.read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="connectionminer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="solve one or more folds (train -> frozen-beta val/test -> score)")
    r.add_argument("config", help="visual | motor | cx, or a path to a YAML config")
    r.add_argument("--fold", type=int, nargs="+", default=None, help="folds to run (default: all k)")
    r.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the config's seeds")
    r.add_argument("--data", default=None, help="bundle dir (default: data/<dataset>)")
    r.add_argument("--out", default=None, help="results dir (default: runs/<dataset>)")
    r.add_argument("--num-iter", type=int, default=None, help="override solver iterations (smoke tests)")
    r.add_argument("--device", default="auto", help="auto | cpu | cuda")

    s = sub.add_parser("summarize", help="mean r per split over all finished folds")
    s.add_argument("config")
    s.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.out or REPO / "runs" / cfg["dataset"])

    if a.cmd == "summarize":
        from .pipeline import summarize
        print(f"{cfg['dataset']}: {out}")
        print(summarize(out).round(4).to_string())
        return

    # Thread count changes BLAS reduction order; pin it so CPU runs are repeatable.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(cfg.get("threads", 4)))
    from .data import load_bundle
    from .pipeline import run_fold

    if a.num_iter is not None:
        cfg["solver"]["num_iter"] = a.num_iter
    b = load_bundle(a.data or REPO / "data" / cfg["dataset"])
    for f in (a.fold if a.fold is not None else range(int(cfg["split"]["k"]))):
        run_fold(b, cfg, f, a.seeds or cfg["seeds"], out, device=a.device)


if __name__ == "__main__":
    main()
