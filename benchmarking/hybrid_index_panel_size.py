#!/usr/bin/env python3
"""How large do the parental reference panels need to be for
`estimate-hybrid-index` to be accurate and its intervals trustworthy?

For each panel size (individuals per parental population, equal or
unequal) and seed, simulate_vcf.py builds a hybrid zone in which the
reference panels and the hybrids are all sampled from the parental
populations' frequencies (--hybrid-freqs population), as in real data:
the panels only estimate those frequencies. Each data set is analysed as
true genotypes (VCF) and as reads at --mean-depth (beagle), with two locus
sets:

  filtered   filter-aims --allele-freq-diff on the same panels, then
             estimate-hybrid-index on the retained loci (the usual workflow)
  true_aims  the simulator's own AIMs (a control: the same loci whatever
             the panels show, so only the panel frequencies' error remains)

Scores, for the hybrids (true hybrid index known): mean absolute error,
bias, slope of estimate on truth (1 = no compression or stretching
around 0.5), 95% interval coverage and mean width, and loci used. Writes
<outdir>/runs.tsv (one row per run) and <outdir>/summary.tsv (mean over
seeds), and prints the summary.

Example:
    python benchmarking/hybrid_index_panel_size.py --outdir hi_panel_size
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HYCLINE = [sys.executable, str(HERE.parent / "run_hycline.py")]
LABELS = ["--pop1-label", "pop1", "--pop2-label", "pop2"]


def parse_sizes(text: str) -> list[tuple[int, int]]:
    """'5,10,5x40' -> [(5, 5), (10, 10), (5, 40)]."""
    out = []
    for item in text.split(","):
        a, _, b = item.partition("x")
        out.append((int(a), int(b or a)))
    return out


def subset_true_aims(stem: Path, kind: str, out: Path) -> None:
    truth = pd.read_csv(f"{stem}_truth.tsv", sep="\t")
    aims = set(truth.loc[truth["is_aim"] == 1, "id"]) | set(truth.loc[truth["is_aim"] == 1, "marker"])
    with open(f"{stem}.{kind}") as src, open(out, "w") as dst:
        for i, line in enumerate(src):
            if kind == "vcf":
                keep = line.startswith("#") or line.split("\t", 3)[2] in aims
            else:
                keep = i == 0 or line.split("\t", 1)[0] in aims
            if keep:
                dst.write(line)


def score(est_path: Path, meta: pd.DataFrame) -> dict:
    est = pd.read_csv(est_path, sep="\t")
    j = est.merge(meta[meta["population"] == "popH"], on="individual")
    err = j["hybrid_index"] - j["true_h"]
    covered = (j["hybrid_index_lower"] <= j["true_h"]) & (j["true_h"] <= j["hybrid_index_upper"])
    return {"mae": err.abs().mean(), "bias": err.mean(),
            "slope": np.polyfit(j["true_h"] - 0.5, j["hybrid_index"] - 0.5, 1)[0],
            "coverage95": covered.mean(),
            "width95": (j["hybrid_index_upper"] - j["hybrid_index_lower"]).mean(),
            "n_loci": j["n_loci"].median()}


def run_one(job: dict) -> list[dict]:
    n1, n2, seed, a = job["n1"], job["n2"], job["seed"], job["args"]
    d = Path(a.outdir) / f"p{n1}x{n2}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / "sim"
    subprocess.run([sys.executable, str(HERE / "simulate_vcf.py"),
                    "--n-pop1", str(n1), "--n-pop2", str(n2), "--n-popH", str(a.n_hybrids),
                    "--n-snps", str(a.n_snps), "--aim-percent", str(a.aim_percent),
                    *(["--aim-freq-diff", str(a.sim_aim_freq_diff)] if a.sim_aim_freq_diff else []),
                    "--missing-global", str(a.missing), "--hybrid-freqs", "population",
                    "--format-beagle", "--beagle-output", f"{stem}.beagle",
                    "--mean-depth", str(a.mean_depth), "-o", f"{stem}.vcf",
                    "--seed", str(seed)], check=True, capture_output=True)
    meta_path = f"{stem}_metadata.tsv"
    meta = pd.read_csv(meta_path, sep="\t").rename(columns={"hybrid_index": "true_h"})
    rows = []
    for kind in ("vcf", "beagle"):
        inputs = {"filtered": d / f"filtered.{kind}", "true_aims": d / f"true_aims.{kind}"}
        subprocess.run(HYCLINE + ["filter-aims", "--input", f"{stem}.{kind}", "--metadata", meta_path,
                                  *LABELS, "--allele-freq-diff", str(a.allele_freq_diff),
                                  "--output", str(inputs["filtered"])],
                       check=True, capture_output=True)
        subset_true_aims(stem, kind, inputs["true_aims"])
        for loci, path in inputs.items():
            out = d / f"hi_{kind}_{loci}.tsv"
            subprocess.run(HYCLINE + ["estimate-hybrid-index", "--input", str(path),
                                      "--metadata", meta_path, *LABELS, "--output", str(out)],
                           check=True, capture_output=True)
            rows.append({"panel": f"{n1}+{n2}", "n_pop1": n1, "n_pop2": n2, "seed": seed,
                         "data": "genotypes" if kind == "vcf" else f"beagle_depth{a.mean_depth:g}",
                         "loci": loci, **score(out, meta)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="5,10,20,40,80,5x40,10x40",
                    help="Panel sizes: N for N + N, AxB for unequal panels (comma-separated).")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-hybrids", type=int, default=100)
    ap.add_argument("--n-snps", type=int, default=2000)
    ap.add_argument("--aim-percent", type=float, default=30)
    ap.add_argument("--sim-aim-freq-diff", type=float, default=None,
                    help="Simulated AIMs differ by at least this much between the parental "
                         "populations (default: the simulator's near-fixed AIMs, <= 0.05 in "
                         "one population and >= 0.95 in the other).")
    ap.add_argument("--missing", type=float, default=0.05)
    ap.add_argument("--mean-depth", type=float, default=5)
    ap.add_argument("--allele-freq-diff", type=float, default=0.5,
                    help="filter-aims rule for the 'filtered' locus set.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="hi_panel_size")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"n1": n1, "n2": n2, "seed": args.seed + r, "args": args}
            for n1, n2 in parse_sizes(args.sizes) for r in range(args.reps)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(run_one, jobs):
            rows.extend(result)
            print(f"done {result[0]['panel']} seed {result[0]['seed']}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.tsv", sep="\t", index=False)
    order = {f"{n1}+{n2}": i for i, (n1, n2) in enumerate(parse_sizes(args.sizes))}
    summary = (runs.drop(columns=["seed", "n_pop1", "n_pop2"])
               .groupby(["data", "loci", "panel"]).mean().reset_index()
               .sort_values(["data", "loci", "panel"], key=lambda c: c.map(order) if c.name == "panel" else c))
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
