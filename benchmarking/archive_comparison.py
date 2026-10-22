#!/usr/bin/env python3
"""Copy a compare_tools.py run into a compact, re-scorable archive.

For each <scenario>_s<seed> run folder under SOURCE, writes under DEST:

  <scenario>_s<seed>/sim/      the simulation truth (sim_truth.tsv,
                               sim_metadata.tsv, sim_params.tsv)
  <scenario>_s<seed>/inputs/   fresh inputs and run scripts for HyCline,
                               gghybrid, bgchm and bgc (make_tool_inputs.py),
                               plus the tool outputs that compare_tools.py
                               scores (HyCline hybrid_index.tsv,
                               fit_mle/hycline_summary.csv and the
                               fit_hier/ summaries; gghybrid and bgchm
                               hybrid_index.tsv and clines.tsv)
  <scenario>_s<seed>/times.json

and copies design.json. Large intermediate files (genomic_input.csv, fitted
curves, logs) are left out. `compare_tools.py --score-only --outdir DEST`
reproduces the scores from the archive.

Usage:
    python benchmarking/archive_comparison.py SOURCE DEST [--design NAME]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_tool_inputs import write_inputs  # noqa: E402

OUTPUTS = ["hycline/hybrid_index.tsv", "hycline/fit_mle/hycline_summary.csv",
           "hycline/fit_hier/hycline_hierarchical_locus_summary.csv",
           "hycline/fit_hier/hycline_hierarchical_population_summary.csv",
           "gghybrid/hybrid_index.tsv", "gghybrid/clines.tsv",
           "bgchm/hybrid_index.tsv", "bgchm/clines.tsv"]


def archive(source: Path, dest: Path, design: str | None = None) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    settings = {}
    if (source / "design.json").exists():
        settings = json.loads((source / "design.json").read_text())
    if design:
        settings["design"] = design
    if settings:
        (dest / "design.json").write_text(json.dumps(settings, indent=1))
    n = 0
    for run in sorted(p for p in source.iterdir() if p.is_dir() and "_s" in p.name):
        sim, target = run / "sim", dest / run.name
        if target.exists():
            shutil.rmtree(target)
        (target / "sim").mkdir(parents=True)
        for f in ("sim_truth.tsv", "sim_metadata.tsv", "sim_params.tsv"):
            shutil.copyfile(sim / f, target / "sim" / f)
        beagle = sim / "sim.beagle"
        write_inputs(sim / "sim.vcf", sim / "sim_metadata.tsv", target / "inputs", "pop1",
                     "pop2", beagle if run.name.startswith("lowdepth") else None,
                     hi_nitt=settings.get("gghybrid_hi_iters", [3000, 1000])[0],
                     hi_burnin=settings.get("gghybrid_hi_iters", [3000, 1000])[1],
                     gc_nitt=settings.get("gghybrid_cline_iters", [5000, 2000])[0],
                     gc_burnin=settings.get("gghybrid_cline_iters", [5000, 2000])[1],
                     bgchm_iters=settings.get("bgchm_iters", 2000))
        for rel in OUTPUTS:
            src = run / "inputs" / rel
            if src.exists():
                (target / "inputs" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target / "inputs" / rel)
        if (run / "times.json").exists():
            shutil.copyfile(run / "times.json", target / "times.json")
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--design", default=None)
    args = ap.parse_args(argv)
    n = archive(Path(args.source), Path(args.dest), args.design)
    print(f"Archived {n} runs to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
