#!/usr/bin/env python3
"""Time `hycline filter-aims` against the number of loci.

For each requested SNP count, simulates a dataset with simulate_vcf.py (VCF,
plus a beagle file when "beagle" is among --formats), then runs filter-aims
on it as a separate process --reps times and records the fastest wall-clock
time and the peak resident memory. Each run is a fresh process, so the
times include HyCline's fixed start-up cost; that cost is measured on its
own (importing hycline.cli) and reported alongside, along with a log-log
slope of the time beyond start-up against loci (1 = linear scaling).

Example:
    python benchmarking/time_filter_aims.py --n-snps 1000 10000 100000 \\
        --formats vcf beagle --outdir timing_run

Writes <outdir>/filter_aims_timing.tsv (format, n_loci, n_samples, seconds,
peak_rss_mb, input_mb) and prints the same table. Simulated inputs are
deleted after timing unless --keep-files is given. Peak memory comes from
os.wait4, so the script needs Linux or macOS.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SIMULATOR = HERE / "simulate_vcf.py"
RUN_HYCLINE = REPO / "run_hycline.py"


def run_timed(cmd: list, cwd: Path) -> tuple:
    """Run `cmd`, returning (wall seconds, peak RSS in MB). Raises on failure."""
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, status, usage = os.wait4(proc.pid, 0)
    seconds = time.perf_counter() - start
    stderr = proc.stderr.read().decode(errors="replace")
    proc.stderr.close()
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{stderr[-2000:]}")
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    rss_mb = usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return seconds, rss_mb


def loglog_slope(n: list, t: list) -> float:
    """Least-squares slope of log(t) on log(n); NaN with fewer than 2 points."""
    if len(n) < 2:
        return float("nan")
    return float(np.polyfit(np.log(n), np.log(t), 1)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-snps", type=int, nargs="+",
                    default=[1000, 3000, 10000, 30000, 100000],
                    help="SNP counts to time (default: 1000 3000 10000 30000 100000).")
    ap.add_argument("--formats", nargs="+", choices=["vcf", "beagle"],
                    default=["vcf", "beagle"], help="Input formats to time (default: both).")
    ap.add_argument("--n-pop1", type=int, default=20)
    ap.add_argument("--n-pop2", type=int, default=20)
    ap.add_argument("--n-popH", type=int, default=60)
    ap.add_argument("--aim-percent", type=float, default=20.0)
    ap.add_argument("--minor-allele-freq", type=float, default=0.1,
                    help="filter-aims --minor-allele-freq (default: 0.1).")
    ap.add_argument("--reps", type=int, default=3,
                    help="Runs per size and format; the fastest is reported (default: 3).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="filter_aims_timing",
                    help="Directory for simulated inputs and the results table.")
    ap.add_argument("--keep-files", action="store_true",
                    help="Keep the simulated inputs and filtered outputs.")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    n_samples = args.n_pop1 + args.n_pop2 + args.n_popH
    rows = []

    for n in sorted(args.n_snps):
        stem = f"sim_{n}"
        sim_cmd = [sys.executable, str(SIMULATOR), "--n-pop1", str(args.n_pop1),
                   "--n-pop2", str(args.n_pop2), "--n-popH", str(args.n_popH),
                   "--n-snps", str(n), "--aim-percent", str(args.aim_percent),
                   "--seed", str(args.seed), "-o", f"{stem}.vcf"]
        if "beagle" in args.formats:
            sim_cmd.append("--format-beagle")
        print(f"Simulating {n} SNPs x {n_samples} samples ...", flush=True)
        run_timed(sim_cmd, outdir)

        for fmt in args.formats:
            inp = outdir / f"{stem}.{fmt}"
            out = outdir / f"{stem}_aims.{fmt}"
            cmd = [sys.executable, str(RUN_HYCLINE), "filter-aims", "--input", str(inp),
                   "--metadata", f"{stem}_metadata.tsv", "--pop1-label", "pop1",
                   "--pop2-label", "pop2", "--minor-allele-freq", str(args.minor_allele_freq),
                   "--output", str(out)]
            runs = [run_timed(cmd, outdir) for _ in range(args.reps)]
            seconds = min(t for t, _ in runs)
            rss = max(m for _, m in runs)
            input_mb = inp.stat().st_size / 1e6
            rows.append({"format": fmt, "n_loci": n, "n_samples": n_samples,
                         "seconds": round(seconds, 3), "peak_rss_mb": round(rss, 1),
                         "input_mb": round(input_mb, 1)})
            print(f"  {fmt:6s} {n:>9d} loci  {seconds:8.2f} s  {rss:8.0f} MB peak  "
                  f"({input_mb:.0f} MB input)", flush=True)
            if not args.keep_files:
                out.unlink(missing_ok=True)

        if not args.keep_files:
            for path in outdir.glob(f"{stem}*"):
                path.unlink()

    startup = min(run_timed([sys.executable, "-c",
                             f"import sys; sys.path.insert(0, {str(REPO)!r}); import hycline.cli"],
                            outdir)[0] for _ in range(args.reps))

    table = outdir / "filter_aims_timing.tsv"
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nStart-up (import hycline.cli): {startup:.2f} s")
    for fmt in args.formats:
        sub = [r for r in rows if r["format"] == fmt]
        # Fit the time beyond start-up, over the sizes where that is at least
        # as large as start-up itself, so the slope reflects per-locus cost.
        net = [(r["n_loci"], r["seconds"] - startup) for r in sub
               if r["seconds"] - startup >= startup]
        if len(net) < 2:
            print(f"{fmt}: too few sizes run long enough past start-up to fit a "
                  f"scaling slope -- add larger --n-snps values")
            continue
        slope = loglog_slope([n for n, _ in net], [t for _, t in net])
        print(f"{fmt}: log-log slope of (time - start-up) on loci = {slope:.2f} "
              f"(over {len(net)} sizes; 1 = linear)")
    print(f"Wrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
