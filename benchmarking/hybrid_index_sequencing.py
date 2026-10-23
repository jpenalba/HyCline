#!/usr/bin/env python3
"""How does `estimate-hybrid-index` hold up on realistic sequencing data?

Each scenario simulates the same kind of hybrid zone (simulate_vcf.py,
panels and hybrids sampled from the parental populations; by default
40 + 40 references, which with true genotypes gives near-nominal
intervals -- see hybrid_index_panel_size.py -- so the losses here come
from the sequencing) and then sequences it: read depth, uneven depth
across sites (--depth-dispersion) and individuals (--individual-depth-sd),
sequencing error, reference bias (--ref-bias) and allele dropout
(--allele-dropout). Each data set is analysed twice from the same reads:
as beagle genotype likelihoods and as called VCF genotypes
(--vcf-genotypes called). A 'true_genotypes' scenario is the reference.
Loci are chosen by filter-aims --allele-freq-diff on the same data, as in
the usual workflow.

Scores for the hybrids: mean absolute error, bias, slope of estimate on
truth, 95% interval coverage and width, loci used, and the same error and
coverage for the quarter of hybrids with the lowest sequencing depth.
Writes <outdir>/runs.tsv and <outdir>/summary.tsv.

Example:
    python benchmarking/hybrid_index_sequencing.py --outdir hi_sequencing
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

# name -> simulator read options (None: true genotypes, no reads)
SCENARIOS = {
    "true_genotypes": None,
    "depth5": ["--mean-depth", "5"],
    "depth2": ["--mean-depth", "2"],
    "depth1": ["--mean-depth", "1"],
    "uneven_sites": ["--mean-depth", "5", "--depth-dispersion", "1"],
    "uneven_individuals": ["--mean-depth", "5", "--individual-depth-sd", "1"],
    "seq_error_3pct": ["--mean-depth", "5", "--seq-error", "0.03"],
    "ref_bias_0.05": ["--mean-depth", "5", "--ref-bias", "0.05"],
    "allele_dropout_10pct": ["--mean-depth", "5", "--allele-dropout", "0.1"],
    "combined": ["--mean-depth", "3", "--depth-dispersion", "2", "--individual-depth-sd", "0.7",
                 "--ref-bias", "0.03", "--allele-dropout", "0.05"],
}


def score(est_path: Path, meta: pd.DataFrame) -> dict:
    est = pd.read_csv(est_path, sep="\t")
    j = est.merge(meta[meta["population"] == "popH"], on="individual")
    err = j["hybrid_index"] - j["true_h"]
    covered = (j["hybrid_index_lower"] <= j["true_h"]) & (j["true_h"] <= j["hybrid_index_upper"])
    out = {"mae": err.abs().mean(), "bias": err.mean(),
           "slope": np.polyfit(j["true_h"] - 0.5, j["hybrid_index"] - 0.5, 1)[0],
           "coverage95": covered.mean(),
           "width95": (j["hybrid_index_upper"] - j["hybrid_index_lower"]).mean(),
           "n_loci": j["n_loci"].median()}
    if "mean_depth" in j:
        low = j["mean_depth"] <= j["mean_depth"].quantile(0.25)
        out.update({"low_depth_mae": err[low].abs().mean(),
                    "low_depth_coverage95": covered[low].mean()})
    return out


def run_one(job: dict) -> list[dict]:
    name, seed, a = job["scenario"], job["seed"], job["args"]
    d = Path(a.outdir) / f"{name}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / "sim"
    reads = SCENARIOS[name]
    cmd = [sys.executable, str(HERE / "simulate_vcf.py"),
           "--n-pop1", str(a.n_parental), "--n-pop2", str(a.n_parental),
           "--n-popH", str(a.n_hybrids), "--n-snps", str(a.n_snps),
           "--aim-percent", str(a.aim_percent), "--aim-freq-diff", str(a.sim_aim_freq_diff),
           "--missing-global", str(a.missing), "--hybrid-freqs", "population",
           "-o", f"{stem}.vcf", "--seed", str(seed)]
    if reads is not None:
        cmd += ["--format-beagle", "--beagle-output", f"{stem}.beagle",
                "--vcf-genotypes", "called", *reads]
    subprocess.run(cmd, check=True, capture_output=True)
    meta_path = f"{stem}_metadata.tsv"
    meta = pd.read_csv(meta_path, sep="\t").rename(columns={"hybrid_index": "true_h"})
    rows = []
    for kind in (("vcf",) if reads is None else ("beagle", "vcf")):
        aims = d / f"aims.{kind}"
        subprocess.run(HYCLINE + ["filter-aims", "--input", f"{stem}.{kind}", "--metadata",
                                  meta_path, *LABELS, "--allele-freq-diff",
                                  str(a.allele_freq_diff), "--output", str(aims)],
                       check=True, capture_output=True)
        out = d / f"hi_{kind}.tsv"
        subprocess.run(HYCLINE + ["estimate-hybrid-index", "--input", str(aims),
                                  "--metadata", meta_path, *LABELS, "--output", str(out)],
                       check=True, capture_output=True)
        data = ("true genotypes" if reads is None
                else "genotype likelihoods" if kind == "beagle" else "called genotypes")
        rows.append({"scenario": name, "data": data, "seed": seed, **score(out, meta)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-parental", type=int, default=40)
    ap.add_argument("--n-hybrids", type=int, default=100)
    ap.add_argument("--n-snps", type=int, default=2000)
    ap.add_argument("--aim-percent", type=float, default=30)
    ap.add_argument("--sim-aim-freq-diff", type=float, default=0.5,
                    help="Simulated AIMs differ by at least this much between the parental "
                         "populations.")
    ap.add_argument("--missing", type=float, default=0.05)
    ap.add_argument("--allele-freq-diff", type=float, default=0.5,
                    help="filter-aims rule for choosing the loci.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="hi_sequencing")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"scenario": s, "seed": args.seed + r, "args": args}
            for s in args.scenarios for r in range(args.reps)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(run_one, jobs):
            rows.extend(result)
            print(f"done {result[0]['scenario']} seed {result[0]['seed']}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.tsv", sep="\t", index=False)
    order = {s: i for i, s in enumerate(args.scenarios)}
    summary = (runs.drop(columns=["seed"]).groupby(["scenario", "data"]).mean().reset_index()
               .sort_values(["scenario", "data"],
                            key=lambda c: c.map(order) if c.name == "scenario" else c))
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
