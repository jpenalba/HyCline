#!/usr/bin/env python3
"""Benchmark HyCline against gghybrid and bgchm on simulated hybrid zones.

For each replicate seed, simulate_vcf.py builds one data set (parental
panels, hybrids with known hybrid indices, and loci with known genomic
clines: neutral, barrier (v > 1), permeable (v < 1) and biased (centre
shifted)). The same truth is analysed in two scenarios:

  genotypes  true genotypes (VCF) for every tool
  lowdepth   reads at --mean-depth: HyCline and bgchm use the genotype
             likelihoods (beagle / glik), gghybrid, which takes genotypes
             only, the genotypes called from the same reads

make_tool_inputs.py writes each tool's inputs and run script (kept under
<outdir>/<scenario>_s<seed>/inputs for rerunning by hand), the tools are
run, and their estimates are scored against the truth:

  hybrid index (test individuals): mean absolute error, bias, Pearson r,
    95% interval coverage and width
  genomic clines (per locus, centre c and slope v; c = invlogit(-u / v) in
    HyCline's convention): median absolute error of logit(c) and log(v);
    Pearson r of log(v) with the truth; v outliers (95%
    interval excludes 1) among barrier/permeable loci (power) and neutral
    loci (false positives); centre outliers (interval excludes 0.5) among
    biased and neutral loci
  wall time per tool

HyCline is run as `fit-genomic` (MLE; Wald intervals on u and log v) and
`fit-hierarchical-genomic` (NumPyro; its per-locus output is a posterior
mean and SD, so its intervals here are normal approximations on u and
log v). gghybrid runs esth then ggcline; bgchm runs est_p, est_hi and the
hierarchical est_genocl.

Example:
    python benchmarking/compare_tools.py --reps 2 --outdir tool_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_tool_inputs import write_inputs  # noqa: E402

Z = 1.959964


def logit(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def invlogit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def run(cmd: list, cwd: Path, log: str) -> tuple[bool, float]:
    env = {**os.environ,
           "HYCLINE": f"{sys.executable} {HERE.parent / 'run_hycline.py'}"}
    t = time.perf_counter()
    with open(cwd / log, "w") as fh:
        ok = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                            env=env).returncode == 0
    return ok, time.perf_counter() - t


def simulate(stem: Path, args, seed: int, reads: bool) -> None:
    cmd = [sys.executable, str(HERE / "simulate_vcf.py"),
           "--n-pop1", str(args.n_parental), "--n-pop2", str(args.n_parental),
           "--n-popH", str(args.n_hybrids), "--n-snps", str(args.n_loci),
           "--aim-percent", "100", "--aim-freq-diff", str(args.aim_freq_diff),
           "--barrier-percent", str(args.class_percent), "--barrier-v", str(args.barrier_v),
           "--permeable-percent", str(args.class_percent),
           "--permeable-v", str(args.permeable_v),
           "--biased-percent", str(args.class_percent), "--biased-u", str(args.biased_u),
           "--seed", str(seed), "-o", f"{stem}.vcf"]
    if reads:
        cmd += ["--format-beagle", "--vcf-genotypes", "called",
                "--mean-depth", str(args.mean_depth)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def collect(inputs: Path, truth: pd.DataFrame, beagle: bool) -> tuple[dict, dict]:
    """Per tool: hybrid-index table (individual, h, lower, upper) and cline
    table (id, c, c_lo, c_hi, v, v_lo, v_hi), on the truth's locus IDs."""
    his, cls = {}, {}
    marker_to_id = dict(zip(truth["marker"], truth["id"]))

    hy = inputs / "hycline"
    if (hy / "hybrid_index.tsv").exists():
        h = pd.read_csv(hy / "hybrid_index.tsv", sep="\t")
        his["HyCline"] = h.rename(columns={"hybrid_index": "h", "hybrid_index_lower": "lo",
                                           "hybrid_index_upper": "hi"})[
            ["individual", "h", "lo", "hi"]]
    for name, path, cols in (
            ("HyCline MLE", hy / "fit_mle" / "hycline_summary.csv", ("u", "u_se", "v", "v_se")),
            ("HyCline hierarchical", hy / "fit_hier" / "hycline_hierarchical_locus_summary.csv",
             ("u_mean", "u_sd", "v_mean", "v_sd"))):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        u, use, v, vse = (d[c].to_numpy(float) for c in cols)
        lv, lvse = np.log(v), vse / v
        # Centre c = invlogit(-u / v); interval from the u interval at fixed v
        # (c = 0.5 exactly when u = 0, so the outlier call is the u test).
        ids = d["locus"].map(lambda l: marker_to_id.get(l, l))
        cls[name] = pd.DataFrame({
            "id": ids, "c": invlogit(-u / v),
            "c_lo": invlogit(-(u + Z * use) / v), "c_hi": invlogit(-(u - Z * use) / v),
            "v": v, "v_lo": np.exp(lv - Z * lvse), "v_hi": np.exp(lv + Z * lvse)})

    gg = inputs / "gghybrid"
    if (gg / "hybrid_index.tsv").exists():
        h = pd.read_csv(gg / "hybrid_index.tsv", sep="\t")
        his["gghybrid"] = h.rename(columns={"INDLABEL": "individual", "h_posterior_mode": "h",
                                            "h_cred_int_lower": "lo",
                                            "h_cred_int_upper": "hi"})[
            ["individual", "h", "lo", "hi"]]
    if (gg / "clines.tsv").exists():
        d = pd.read_csv(gg / "clines.tsv", sep="\t")
        cls["gghybrid"] = pd.DataFrame({
            "id": d["locus"], "c": d["invlogit_mean_logit_centre"],
            "c_lo": d["centre_lower_95"], "c_hi": d["centre_upper_95"],
            "v": d["exp_mean_log_v"], "v_lo": d["v_lower_95"], "v_hi": d["v_upper_95"]})

    bg = inputs / "bgchm"
    if (bg / "hybrid_index.tsv").exists():
        h = pd.read_csv(bg / "hybrid_index.tsv", sep="\t")
        his["bgchm"] = h.rename(columns={"hi_median": "h", "hi_q2.5": "lo",
                                         "hi_q97.5": "hi"})[["individual", "h", "lo", "hi"]]
    if (bg / "clines.tsv").exists():
        d = pd.read_csv(bg / "clines.tsv", sep="\t")
        cls["bgchm"] = pd.DataFrame({
            "id": d["locus"], "c": d["center"], "c_lo": d["center_q2.5"],
            "c_hi": d["center_q97.5"], "v": d["v"], "v_lo": d["v_q2.5"], "v_hi": d["v_q97.5"]})
    return his, cls


def score_hi(est: pd.DataFrame, meta: pd.DataFrame) -> dict:
    j = est.merge(meta[meta["population"] == "popH"], on="individual")
    err = j["h"] - j["hybrid_index"]
    return {"n": len(j), "mae": err.abs().mean(), "bias": err.mean(),
            "r": np.corrcoef(j["h"], j["hybrid_index"])[0, 1],
            "coverage95": ((j["lo"] <= j["hybrid_index"]) & (j["hybrid_index"] <= j["hi"])).mean(),
            "width95": (j["hi"] - j["lo"]).mean()}


def score_clines(est: pd.DataFrame, truth: pd.DataFrame) -> dict:
    t = pd.DataFrame({"id": truth["id"], "cline_class": truth["cline_class"],
                      "v_true": truth["v"], "c_true": invlogit(-truth["u"] / truth["v"])})
    j = est.merge(t, on="id")
    v_out = (j["v_lo"] > 1) | (j["v_hi"] < 1)
    c_out = (j["c_lo"] > 0.5) | (j["c_hi"] < 0.5)
    neutral = j["cline_class"] == "neutral"
    vclass = j["cline_class"].isin(["barrier", "permeable"])
    biased = j["cline_class"] == "biased"
    return {"n_loci": len(j),
            # Medians: a locus estimated as nearly flat (v -> 0) has no
            # identifiable centre, and a few such loci would dominate a mean.
            "mdae_logit_c": np.median(np.abs(logit(j["c"]) - logit(j["c_true"]))),
            "mdae_log_v": np.median(np.abs(np.log(j["v"]) - np.log(j["v_true"]))),
            "r_log_v": np.corrcoef(np.log(j["v"]), np.log(j["v_true"]))[0, 1],
            "v_power": v_out[vclass].mean(), "v_false_pos": v_out[neutral].mean(),
            "c_power": c_out[biased].mean(), "c_false_pos": c_out[neutral].mean()}


def score_run(run_dir, stem, inputs, reads, scen, seed, times, hi_rows, cl_rows) -> None:
    truth = pd.read_csv(f"{stem}_truth.tsv", sep="\t")
    meta = pd.read_csv(f"{stem}_metadata.tsv", sep="\t")
    his, cls = collect(inputs, truth, reads)
    for tool, est in his.items():
        hi_rows.append({"scenario": scen, "seed": seed, "tool": tool,
                        **score_hi(est, meta), "seconds": times.get(tool)})
    for tool, est in cls.items():
        cl_rows.append({"scenario": scen, "seed": seed, "tool": tool,
                        **score_clines(est, truth), "seconds": times.get(tool.split()[0])})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenarios", nargs="+", default=["genotypes", "lowdepth"],
                    choices=["genotypes", "lowdepth"])
    ap.add_argument("--tools", nargs="+", default=["hycline", "gghybrid", "bgchm"],
                    choices=["hycline", "gghybrid", "bgchm"])
    ap.add_argument("--n-parental", type=int, default=20)
    ap.add_argument("--n-hybrids", type=int, default=100)
    ap.add_argument("--n-loci", type=int, default=100)
    ap.add_argument("--aim-freq-diff", type=float, default=0.6)
    ap.add_argument("--class-percent", type=float, default=10,
                    help="Percent of loci in each of the barrier, permeable and biased classes.")
    ap.add_argument("--barrier-v", type=float, default=3.0)
    ap.add_argument("--permeable-v", type=float, default=0.33)
    ap.add_argument("--biased-u", type=float, default=1.5)
    ap.add_argument("--mean-depth", type=float, default=3.0)
    ap.add_argument("--gghybrid-hi-iters", type=int, nargs=2, default=[3000, 1000],
                    metavar=("NITT", "BURNIN"))
    ap.add_argument("--gghybrid-cline-iters", type=int, nargs=2, default=[5000, 2000],
                    metavar=("NITT", "BURNIN"))
    ap.add_argument("--bgchm-iters", type=int, default=2000)
    ap.add_argument("--outdir", default="tool_comparison")
    ap.add_argument("--score-only", action="store_true",
                    help="Re-score the runs already in --outdir without simulating or "
                         "running any tool (e.g. after an interrupted benchmark).")
    args = ap.parse_args(argv)

    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hi_rows, cl_rows = [], []
    for rep in range(args.reps):
        seed = args.seed + rep
        for scen in args.scenarios:
            run_dir = out / f"{scen}_s{seed}"
            reads = scen == "lowdepth"
            stem = run_dir / "sim" / "sim"
            inputs = run_dir / "inputs"
            if args.score_only:
                if not run_dir.exists():
                    continue
                times_file = run_dir / "times.json"
                times = json.loads(times_file.read_text()) if times_file.exists() else {}
                score_run(run_dir, stem, inputs, reads, scen, seed, times, hi_rows, cl_rows)
                continue
            if run_dir.exists():
                shutil.rmtree(run_dir)
            (run_dir / "sim").mkdir(parents=True)
            simulate(stem, args, seed, reads)
            write_inputs(Path(f"{stem}.vcf"), Path(f"{stem}_metadata.tsv"), inputs,
                         "pop1", "pop2", Path(f"{stem}.beagle") if reads else None,
                         hi_nitt=args.gghybrid_hi_iters[0], hi_burnin=args.gghybrid_hi_iters[1],
                         gc_nitt=args.gghybrid_cline_iters[0],
                         gc_burnin=args.gghybrid_cline_iters[1],
                         bgchm_iters=args.bgchm_iters)
            times = {}
            if "hycline" in args.tools:
                ok, t = run(["bash", "run_hycline.sh"], inputs / "hycline", "log.txt")
                times["HyCline"] = t if ok else np.nan
            if "gghybrid" in args.tools:
                ok, t = run(["Rscript", "run_gghybrid.R"], inputs / "gghybrid", "log.txt")
                times["gghybrid"] = t if ok else np.nan
            if "bgchm" in args.tools:
                ok, t = run(["Rscript", "run_bgchm.R", "glik" if reads else "genotype"],
                            inputs / "bgchm", "log.txt")
                times["bgchm"] = t if ok else np.nan
            (run_dir / "times.json").write_text(json.dumps(times))
            score_run(run_dir, stem, inputs, reads, scen, seed, times, hi_rows, cl_rows)
            print(f"{scen} seed {seed}: " + ", ".join(
                f"{k} {v:.0f} s" if v == v else f"{k} FAILED" for k, v in times.items()),
                flush=True)

    hi_df, cl_df = pd.DataFrame(hi_rows), pd.DataFrame(cl_rows)
    if hi_df.empty and cl_df.empty:
        print(f"No tool output to score under {out} -- run the tools' scripts first.")
        return 1
    hi_df.to_csv(out / "hybrid_index_scores.tsv", sep="\t", index=False)
    cl_df.to_csv(out / "cline_scores.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nHybrid index (test individuals; mean over seeds)")
        if not hi_df.empty:
            print(hi_df.drop(columns="seed").groupby(["scenario", "tool"]).mean().round(3))
        print("\nGenomic clines (mean over seeds)")
        if not cl_df.empty:
            print(cl_df.drop(columns="seed").groupby(["scenario", "tool"]).mean().round(3))
    print(f"\nWrote {out / 'hybrid_index_scores.tsv'} and {out / 'cline_scores.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
