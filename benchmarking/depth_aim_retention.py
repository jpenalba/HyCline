#!/usr/bin/env python3
"""How read depth affects which AIMs `filter-aims` retains.

For each replicate seed, simulates one dataset with simulate_vcf.py at each
requested mean depth. Reads come from their own random stream, so a seed
gives the same true genotypes at every depth; only the reads (and so the
genotype likelihoods) change. Each dataset is filtered from its beagle
output -- and, for comparison, from a VCF of genotypes called from the same
reads (--vcf-genotypes called) -- at each requested --minor-allele-freq
cutoff, and the retained loci are scored against the simulator's truth
file (is_aim):

  recall            true AIMs retained / true AIMs
  recall_fixed      the same, for AIMs fixed in both reference panels
  recall_not_fixed  the same, for AIMs with the minor allele present in a
                    panel (up to the simulator's --aim-cutoff)
  precision         true AIMs retained / loci retained
  false_pos         non-AIMs retained

Example:
    python benchmarking/depth_aim_retention.py --depths 1 2 4 8 16 \\
        --cutoffs 0.05 0.1 --reps 3 --outdir depth_run

Writes <outdir>/depth_aim_retention.tsv (one row per seed, depth, input,
cutoff) and prints a table averaged over seeds.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hycline.data.filter_aims import filter_aims_beagle, filter_aims_vcf  # noqa: E402

SIMULATOR = HERE / "simulate_vcf.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[1, 2, 3, 4, 6, 8, 10, 15, 20, 30],
                    help="Mean read depths to simulate.")
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.05, 0.1],
                    help="filter-aims --minor-allele-freq values to score (default: 0.05 0.1).")
    ap.add_argument("--reps", type=int, default=3, help="Replicate seeds (default: 3).")
    ap.add_argument("--seed", type=int, default=1, help="First seed; replicates use seed, seed+1, ...")
    ap.add_argument("--n-snps", type=int, default=5000)
    ap.add_argument("--aim-percent", type=float, default=20.0)
    ap.add_argument("--aim-cutoff", type=float, default=0.05,
                    help="Simulator AIM rule (default: 0.05).")
    ap.add_argument("--n-pop1", type=int, default=20)
    ap.add_argument("--n-pop2", type=int, default=20)
    ap.add_argument("--n-popH", type=int, default=20)
    ap.add_argument("--depth-dispersion", type=float, default=0.0,
                    help="Passed to the simulator (0 = Poisson depth).")
    ap.add_argument("--individual-depth-sd", type=float, default=0.0,
                    help="Passed to the simulator.")
    ap.add_argument("--seq-error", type=float, default=0.01, help="Passed to the simulator.")
    ap.add_argument("--no-vcf", action="store_true",
                    help="Score only the beagle output, not called-genotype VCFs.")
    ap.add_argument("--outdir", default="depth_aim_retention")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(args.reps):
        seed = args.seed + rep
        for depth in args.depths:
            stem = outdir / f"sim_s{seed}_d{depth:g}"
            subprocess.run(
                [sys.executable, str(SIMULATOR), "--n-pop1", str(args.n_pop1),
                 "--n-pop2", str(args.n_pop2), "--n-popH", str(args.n_popH),
                 "--n-snps", str(args.n_snps), "--aim-percent", str(args.aim_percent),
                 "--aim-cutoff", str(args.aim_cutoff), "--format-beagle",
                 "--vcf-genotypes", "called", "--mean-depth", str(depth),
                 "--depth-dispersion", str(args.depth_dispersion),
                 "--individual-depth-sd", str(args.individual_depth_sd),
                 "--seq-error", str(args.seq_error), "--seed", str(seed),
                 "-o", f"{stem}.vcf"],
                check=True, stdout=subprocess.DEVNULL)
            truth = pd.read_csv(f"{stem}_truth.tsv", sep="\t")
            is_aim = truth["is_aim"].to_numpy(dtype=bool)
            # An AIM's distance from a perfect fixed difference in the panels:
            # the larger of the low population's frequency and 1 minus the
            # high population's. 0 = fixed in both panels.
            lo = np.minimum(truth["freq_pop1"], truth["freq_pop2"]).to_numpy()
            hi = np.maximum(truth["freq_pop1"], truth["freq_pop2"]).to_numpy()
            fixed = is_aim & (np.maximum(lo, 1 - hi) < 1e-9)
            inputs = [("beagle", filter_aims_beagle, f"{stem}.beagle")]
            if not args.no_vcf:
                inputs.append(("vcf_called", filter_aims_vcf, f"{stem}.vcf"))
            for label, fn, path in inputs:
                for cutoff in args.cutoffs:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = fn(path, f"{stem}_metadata.tsv", str(outdir / "out.tmp"),
                                    pop1_label="pop1", pop2_label="pop2",
                                    low_threshold=cutoff, high_threshold=1 - cutoff)
                    kept = np.asarray(result["mask"], dtype=bool)
                    tp = int((kept & is_aim).sum())
                    rows.append({
                        "seed": seed, "mean_depth": depth, "input": label, "cutoff": cutoff,
                        "n_true_aims": int(is_aim.sum()), "n_retained": int(kept.sum()),
                        "true_pos": tp, "false_pos": int((kept & ~is_aim).sum()),
                        "recall": tp / is_aim.sum(),
                        "recall_fixed": (kept & fixed).sum() / fixed.sum(),
                        "recall_not_fixed": (kept & is_aim & ~fixed).sum() / (is_aim & ~fixed).sum(),
                        "precision": tp / kept.sum() if kept.any() else float("nan"),
                    })
            for path in outdir.glob(f"{stem.name}*"):
                path.unlink()
            print(f"seed {seed} depth {depth:g} done", flush=True)
    (outdir / "out.tmp").unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "depth_aim_retention.tsv", sep="\t", index=False)
    summary = (df.groupby(["input", "cutoff", "mean_depth"])
                 [["n_retained", "false_pos", "recall", "recall_fixed", "recall_not_fixed"]]
                 .mean().round(3))
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 160):
        print(summary)
    print(f"Wrote {outdir / 'depth_aim_retention.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
