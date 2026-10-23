#!/usr/bin/env python3
"""Run time, peak memory and output size of generate-geographic-input and
generate-genomic-input as the input grows, and the cost of loading their
output into the fitting commands.

For each size (loci x individuals), simulate_vcf.py writes a VCF (true
genotypes) and a beagle file (reads at depth 4); every individual gets a
sampling site (--n-sites) and a distance, and the two parental panels are
labelled for generate-genomic-input. Each command is run on each file in
its own process, recording wall time, peak resident memory and the output
CSV's size and column count; the output is then loaded, again in its own
process, with the reader the fitting commands use
(hycline.data.io.read_*_multi_locus_wide_csv), recording time and peak
memory. Writes <outdir>/results.tsv.

Example:
    python benchmarking/conversion_scaling.py --sizes 1000x200 10000x200 --outdir conv_scaling
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Runs one command as a child process and reports its wall time and the
# child's peak resident memory (MB).
MEASURE = r"""
import json, resource, subprocess, sys, time
t = time.perf_counter()
r = subprocess.run(sys.argv[1:], capture_output=True, text=True)
print(json.dumps({"seconds": time.perf_counter() - t, "returncode": r.returncode,
                  "stderr": r.stderr[-500:],
                  "peak_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}))
"""

READ_BACK = r"""
import sys
sys.path.insert(0, {repo!r})
from hycline.data import io
reader, path = sys.argv[1], sys.argv[2]
kw = {{"individual_col": "individual"}} if reader != "read_multi_locus_wide_csv" else {{}}
data = getattr(io, reader)(path, **kw)
print(len(data))
"""


def measure(cmd: list) -> dict:
    out = subprocess.run([sys.executable, "-c", MEASURE, *cmd], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", nargs="+", default=["1000x200", "10000x200", "100000x200",
                                                     "10000x1000"],
                    help="LOCIxINDIVIDUALS (individuals include 20 + 20 parentals).")
    ap.add_argument("--n-sites", type=int, default=20)
    ap.add_argument("--mean-depth", type=float, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="conversion_scaling")
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in args.sizes:
        n_loci, n_ind = (int(x) for x in size.split("x"))
        d = out / size
        d.mkdir(exist_ok=True)
        stem = d / "sim"
        t = time.perf_counter()
        subprocess.run([sys.executable, str(HERE / "simulate_vcf.py"), "--n-pop1", "20",
                        "--n-pop2", "20", "--n-popH", str(n_ind - 40), "--n-snps", str(n_loci),
                        "--aim-percent", "30", "--missing-global", "0.05", "--format-beagle",
                        "--mean-depth", str(args.mean_depth), "-o", f"{stem}.vcf",
                        "--seed", str(args.seed)], check=True, capture_output=True)
        print(f"{size}: simulated in {time.perf_counter() - t:.0f} s", flush=True)
        meta = pd.read_csv(f"{stem}_metadata.tsv", sep="\t")
        site = (meta["hybrid_index"] * (args.n_sites - 1)).round().astype(int)
        meta["site"] = "site" + site.astype(str)
        meta["distance"] = site * 10.0
        meta_path = d / "metadata.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)
        for kind in ("vcf", "beagle"):
            inp = Path(f"{stem}.{kind}")
            for command in ("generate-geographic-input", "generate-genomic-input"):
                short = command.split("-")[1]
                csv = d / f"{short}_{kind}.csv"
                cmd = [sys.executable, str(REPO / "run_hycline.py"), command, "--input", str(inp),
                       "--metadata", str(meta_path), "--output", str(csv)]
                if short == "genomic":
                    cmd += ["--pop1-label", "pop1", "--pop2-label", "pop2"]
                m = measure(cmd)
                row = {"loci": n_loci, "individuals": n_ind, "input": kind, "command": short,
                       "input_mb": inp.stat().st_size / 1e6, "seconds": m["seconds"],
                       "peak_mb": m["peak_mb"], "ok": m["returncode"] == 0}
                if m["returncode"] == 0:
                    with open(csv) as fh:
                        n_cols = fh.readline().count(",") + 1
                    reader = {("geographic", "vcf"): "read_multi_locus_wide_csv",
                              ("geographic", "beagle"): "read_gl_multi_locus_wide_csv",
                              ("genomic", "vcf"): "read_genomic_cline_multi_locus_wide_csv",
                              ("genomic", "beagle"): "read_genomic_gl_multi_locus_wide_csv"}[(short, kind)]
                    r = measure([sys.executable, "-c", READ_BACK.format(repo=str(REPO)),
                                 reader, str(csv)])
                    row.update(output_mb=csv.stat().st_size / 1e6, output_columns=n_cols,
                               read_seconds=r["seconds"], read_peak_mb=r["peak_mb"],
                               read_ok=r["returncode"] == 0)
                    csv.unlink()
                else:
                    row["error"] = m["stderr"].strip().splitlines()[-1] if m["stderr"] else ""
                rows.append(row)
                print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()},
                      flush=True)
        for f in (f"{stem}.vcf", f"{stem}.beagle"):
            Path(f).unlink()
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(df.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
