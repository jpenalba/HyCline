#!/usr/bin/env python3
"""HyCline, gghybrid and bgchm on the Italian sparrow data (Hermansen et al.
2014), compared with each other and with BGC's published estimates.

Builds the VCF and metadata from a gghybrid checkout as
italy_sparrow_hybrid_index.py does (house sparrows = --pop1-label, Spanish
sparrows = --pop2-label; 77 loci; 433 Italian sparrows as test
individuals), writes every tool's inputs with make_tool_inputs.py, runs the
three pipelines (hybrid index, then genomic clines), and compares:

  hybrid index  each tool against BGC's posterior means (Pearson r, mean
                absolute difference, share inside BGC's 95% CI) and the
                tools against each other
  clines        the tools' centre (logit c) and log(v) against each other
                (Pearson r) and against BGC's alpha and beta (Spearman
                rho; BGC's cline function differs, so only rank agreement
                is meaningful)

Writes <outdir>/italy_hybrid_index_scores.tsv and
<outdir>/italy_cline_scores.tsv. --score-only re-scores existing outputs.

Example:
    git clone --depth 1 https://github.com/ribailey/gghybrid gghybrid
    python benchmarking/italy_sparrow_tools.py --gghybrid-dir gghybrid --outdir italy_tools
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from italy_sparrow_hybrid_index import bgc_reference, build_inputs  # noqa: E402
from make_tool_inputs import write_inputs  # noqa: E402
from compare_tools import collect, logit, run  # noqa: E402

REPO = HERE.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gghybrid-dir", required=True)
    ap.add_argument("--outdir", default="italy_sparrow_tools")
    ap.add_argument("--bgc-clines", default=str(REPO / "examples" / "hermansen_italy_bgc_reference.json"),
                    help="BGC alpha/beta per locus (JSON), from gghybrid's BGC comparison.")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)

    gg = Path(args.gghybrid_dir).resolve()
    out = Path(args.outdir).resolve()
    tools = out / "tools"
    if not args.score_only:
        out.mkdir(parents=True, exist_ok=True)
        vcf, meta, _ = build_inputs(gg, out, [])
        write_inputs(vcf, meta, tools, "house", "spanish")
        for name, cmd in (("hycline", ["bash", "run_hycline.sh"]),
                          ("gghybrid", ["Rscript", "run_gghybrid.R"]),
                          ("bgchm", ["Rscript", "run_bgchm.R", "genotype"])):
            ok, secs = run(cmd, tools / name, "log.txt")
            print(f"{name}: {'ok' if ok else 'FAILED'} in {secs:.0f} s", flush=True)

    # Locus IDs are the VCF IDs (the gghybrid column names), so the truth
    # table only needs id/marker for collect().
    loci = pd.read_csv(tools / "bgchm" / "G0.tsv", sep="\t", index_col=0).columns
    fake_truth = pd.DataFrame({"id": loci, "marker": loci})
    his, cls = collect(tools, fake_truth, False)
    bgc = bgc_reference(gg)

    rows = []
    for tool, est in his.items():
        j = est.set_index("individual").join(bgc, how="inner")
        d = j["h"] - j["bgc_mean"]
        rows.append({"comparison": f"{tool} vs BGC", "n": len(j),
                     "pearson_r": np.corrcoef(j["h"], j["bgc_mean"])[0, 1],
                     "mean_abs_diff": d.abs().mean(), "mean_diff": d.mean(),
                     "inside_bgc_ci95": ((j["h"] >= j["bgc_lower"])
                                         & (j["h"] <= j["bgc_upper"])).mean(),
                     "width95": (j["hi"] - j["lo"]).mean()})
    names = list(his)
    italians = bgc.index
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = his[a].set_index("individual")[["h"]].join(
                his[b].set_index("individual")[["h"]], rsuffix="_b", how="inner").loc[
                lambda x: x.index.isin(italians)]
            d = j["h"] - j["h_b"]
            rows.append({"comparison": f"{a} vs {b}", "n": len(j),
                         "pearson_r": np.corrcoef(j["h"], j["h_b"])[0, 1],
                         "mean_abs_diff": d.abs().mean(), "mean_diff": d.mean()})
    hi_df = pd.DataFrame(rows)

    bgc_cl = pd.DataFrame(json.loads(Path(args.bgc_clines).read_text())).T
    crow = []
    for tool, est in cls.items():
        e = est.set_index("id")
        j = e.join(bgc_cl, how="inner")
        crow.append({"comparison": f"{tool} vs BGC", "n_loci": len(j),
                     "spearman_logv_beta": spearmanr(np.log(j["v"]), j["bgc_beta"])[0],
                     "spearman_logitc_alpha": spearmanr(logit(j["c"]), j["bgc_alpha"])[0]})
    cnames = list(cls)
    for i, a in enumerate(cnames):
        for b in cnames[i + 1:]:
            j = cls[a].set_index("id").join(cls[b].set_index("id"), rsuffix="_b", how="inner")
            crow.append({"comparison": f"{a} vs {b}", "n_loci": len(j),
                         "pearson_logv": np.corrcoef(np.log(j["v"]), np.log(j["v_b"]))[0, 1],
                         "pearson_logitc": np.corrcoef(logit(j["c"]), logit(j["c_b"]))[0, 1]})
    cl_df = pd.DataFrame(crow)

    hi_df.to_csv(out / "italy_hybrid_index_scores.tsv", sep="\t", index=False)
    cl_df.to_csv(out / "italy_cline_scores.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(hi_df.round(4).to_string(index=False))
        print()
        print(cl_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
