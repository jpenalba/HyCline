#!/usr/bin/env python3
"""Benchmark `estimate-hybrid-index` on empirical data: the Italian sparrow
hybrid zone (Hermansen et al. 2014, Mol Ecol 23:5831; Dryad
doi:10.5061/dryad.v6f4d), as distributed with gghybrid
(github.com/ribailey/gghybrid).

77 loci (76 nuclear SNPs plus mitochondrial ND2, coded as a diploid
homozygote), 569 individuals: house sparrow (Passer domesticus) parentals
from Kralove and Oslo, Spanish sparrow (P. hispaniolensis) parentals from
LesinaSPANISH, Lauro_Lesina, MasseriaFontanarosa and MasseriaMustazzo, and
433 Italian sparrows (P. italiae). This is the setup of Bailey (2023, Mol
Ecol Resour, section 2.3.1), which compared gghybrid's hybrid indices with
the BGC estimates Hermansen et al. published for the 433 Italian sparrows
(gghybrid vs BGC: Pearson r = 0.991).

The script converts gghybrid's two-rows-per-individual allele table
(RB_Italy_ggcline_precol_headers_diploidND2.csv; allele codes 0/1, the VCF
ALT allele being "1") to a VCF and a metadata file, runs
`hycline estimate-hybrid-index` on them (house = --pop1-label, hybrid index
0; Spanish = --pop2-label, hybrid index 1, matching BGC's orientation), and
compares the 433 Italian sparrows' estimates with BGC's posterior means and
95% credible intervals (hi.out_italy.txt, matched to individuals through
id_italy.csv). If --gghybrid-results is given (a table with INDLABEL and
h_posterior_mode, e.g. from esth()), the estimates are compared with those
too.

Example:
    git clone --depth 1 https://github.com/ribailey/gghybrid gghybrid
    python benchmarking/italy_sparrow_hybrid_index.py --gghybrid-dir gghybrid \\
        --outdir italy_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HOUSE_SITES = ["Kralove", "Oslo"]
SPANISH_SITES = ["LesinaSPANISH", "Lauro_Lesina", "MasseriaFontanarosa", "MasseriaMustazzo"]
ALLELE_TABLE = "RB_Italy_ggcline_precol_headers_diploidND2.csv"


def build_inputs(gg: Path, outdir: Path, drop_loci: list[str]) -> tuple[Path, Path, list[str]]:
    """Write <outdir>/italy.vcf and <outdir>/italy_metadata.tsv; return
    their paths and the loci used."""
    d = pd.read_csv(gg / ALLELE_TABLE, encoding="utf-8-sig")
    loci = [c for c in d.columns[2:] if c not in drop_loci]
    order = list(dict.fromkeys(d["INDLABEL"]))
    pops = d.groupby("INDLABEL")["POPID"].first()
    # Two rows per individual, one allele each; dosage = copies of allele 1,
    # missing if either allele is.
    alleles = d.set_index("INDLABEL")[loci]
    dosage = alleles.groupby(level=0, sort=False).sum(min_count=2)
    dosage = dosage.where(alleles.isna().groupby(level=0, sort=False).sum() == 0)
    dosage = dosage.loc[order]

    gt = {0.0: "0/0", 1.0: "0/1", 2.0: "1/1"}
    lines = ["##fileformat=VCFv4.2",
             "##source=benchmarking/italy_sparrow_hybrid_index.py from gghybrid "
             + ALLELE_TABLE,
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(order)]
    for i, locus in enumerate(loci):
        calls = [gt.get(v, "./.") for v in dosage[locus]]
        lines.append(f"chr1\t{(i + 1) * 1000}\t{locus}\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(calls))
    vcf = outdir / "italy.vcf"
    vcf.write_text("\n".join(lines) + "\n")

    population = pops.loc[order].replace({s: "house" for s in HOUSE_SITES}
                                         | {s: "spanish" for s in SPANISH_SITES})
    meta = outdir / "italy_metadata.tsv"
    pd.DataFrame({"individual": order, "population": population.to_numpy(),
                  "site": pops.loc[order].to_numpy()}).to_csv(meta, sep="\t", index=False)
    return vcf, meta, loci


def bgc_reference(gg: Path) -> pd.DataFrame:
    """BGC posterior mean and 95% CI per Italian sparrow, by individual."""
    hi = pd.read_csv(gg / "hi.out_italy.txt")
    hi["BGC"] = hi["param"].str.split("_").str[2].astype(int)
    ids = pd.read_csv(gg / "id_italy.csv", encoding="utf-8-sig")
    out = ids.merge(hi, on="BGC", validate="one_to_one")
    return out.rename(columns={"mean": "bgc_mean", "0.950_CI_LB": "bgc_lower",
                               "0.950_CI_UB": "bgc_upper"}).set_index("individual")[
        ["site", "bgc_mean", "bgc_lower", "bgc_upper"]]


def compare(est: pd.Series, ref: pd.Series) -> dict:
    diff = est - ref
    return {"n": int(len(est)), "pearson_r": float(np.corrcoef(est, ref)[0, 1]),
            "mean_abs_diff": float(diff.abs().mean()), "mean_diff": float(diff.mean()),
            "max_abs_diff": float(diff.abs().max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gghybrid-dir", required=True,
                    help="Checkout of github.com/ribailey/gghybrid.")
    ap.add_argument("--outdir", default="italy_sparrow_hybrid_index")
    ap.add_argument("--drop-loci", nargs="*", default=[],
                    help="Loci to leave out (e.g. ND2, the mitochondrial locus).")
    ap.add_argument("--gghybrid-results", default=None,
                    help="Optional table (CSV or TSV) with INDLABEL and h_posterior_mode "
                         "columns from gghybrid's esth(), to compare against as well.")
    args = ap.parse_args(argv)

    gg = Path(args.gghybrid_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    vcf, meta, loci = build_inputs(gg, outdir, args.drop_loci)
    print(f"Wrote {vcf} ({len(loci)} loci) and {meta}")

    hi_path = outdir / "hycline_hybrid_index.tsv"
    result = subprocess.run(
        [sys.executable, str(REPO / "run_hycline.py"), "estimate-hybrid-index",
         "--input", str(vcf), "--metadata", str(meta),
         "--pop1-label", "house", "--pop2-label", "spanish", "--output", str(hi_path)],
        capture_output=True, text=True)
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    hy = pd.read_csv(hi_path, sep="\t").set_index("individual")
    metadata = pd.read_csv(meta, sep="\t").set_index("individual")
    for label in ("house", "spanish"):
        h = hy.loc[metadata.index[metadata["population"] == label], "hybrid_index"]
        print(f"{label} parentals (n = {len(h)}): mean h = {h.mean():.3f}, "
              f"range [{h.min():.3f}, {h.max():.3f}]")

    bgc = bgc_reference(gg)
    both = bgc.join(hy, how="inner")
    rows = [{"reference": "BGC posterior mean", **compare(both["hybrid_index"], both["bgc_mean"])}]
    inside = ((both["hybrid_index"] >= both["bgc_lower"])
              & (both["hybrid_index"] <= both["bgc_upper"])).mean()
    overlap = ((both["hybrid_index_lower"] <= both["bgc_upper"])
               & (both["hybrid_index_upper"] >= both["bgc_lower"])).mean()
    width_hy = (both["hybrid_index_upper"] - both["hybrid_index_lower"]).mean()
    width_bgc = (both["bgc_upper"] - both["bgc_lower"]).mean()

    if args.gghybrid_results:
        sep = "\t" if args.gghybrid_results.endswith((".tsv", ".txt")) else ","
        gh = pd.read_csv(args.gghybrid_results, sep=sep).set_index("INDLABEL")
        j = both.join(gh["h_posterior_mode"], how="inner")
        rows.append({"reference": "gghybrid posterior mode",
                     **compare(j["hybrid_index"], j["h_posterior_mode"])})
        rows.append({"reference": "gghybrid vs BGC (for scale)",
                     **compare(j["h_posterior_mode"], j["bgc_mean"])})

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(f"\nItalian sparrows (n = {len(both)}): HyCline against references")
        print(table.round(4).to_string(index=False))
    print(f"\nHyCline estimate inside BGC's 95% CI: {inside:.3f}")
    print(f"HyCline 95% interval overlaps BGC's 95% CI: {overlap:.3f}")
    print(f"Mean interval width: HyCline {width_hy:.3f}, BGC {width_bgc:.3f}")

    both.to_csv(outdir / "comparison.tsv", sep="\t")
    table.to_csv(outdir / "summary.tsv", sep="\t", index=False)
    print(f"\nWrote {outdir / 'comparison.tsv'} and {outdir / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
