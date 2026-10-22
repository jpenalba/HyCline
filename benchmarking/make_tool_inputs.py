#!/usr/bin/env python3
"""Write matching input files for HyCline, gghybrid, bgchm and bgc from one
data set, so the same hybrid zone can be analysed by all four.

Input: a VCF (biallelic SNPs; GT, and AD if read counts are wanted for
bgc's genotype-uncertainty mode), an optional ANGSD beagle file for the same
loci and samples, and a tab-delimited metadata file with sample names and
population labels. benchmarking/simulate_vcf.py writes all three
(<stem>.vcf, <stem>.beagle, <stem>_metadata.tsv); with --vcf-genotypes
called its VCF also carries AD.

Individuals labelled --pop1-label are the parental population at hybrid
index 0 (gghybrid S0, bgchm/bgc parental set 0), those labelled
--pop2-label the one at hybrid index 1 (S1, parental set 1), and every
other sample in the VCF is a test (admixed) individual. Allele counts are
of the VCF ALT allele.

Output, under --outdir:

  hycline/   input.vcf (and input.beagle), metadata.tsv, run_hycline.sh
  gghybrid/  gghybrid_input.csv (read.data format: INDLABEL, POPID, one
             column per locus, two rows per individual, alleles 0/1, NA
             missing) and run_gghybrid.R (read.data, data.prep, esth,
             ggcline)
  bgchm/     G0.tsv, G1.tsv, Gx.tsv (individuals x loci, ALT copies 0/1/2,
             NA missing), with --beagle also GL0_*.tsv/GL1_*.tsv/GL2_*.tsv
             genotype-likelihood matrices, and run_bgchm.R (est_p, est_hi,
             est_genocl)
  bgc/       bgc 1.03 input: genotypes/ (p0in.txt, p1in.txt: per-locus
             parental allele counts "REF ALT"; admixedin.txt: per-locus,
             per-individual counts, "0 0" missing) and, if the VCF has AD,
             reads/ (the same files with per-individual REF/ALT read counts,
             for -N 1); individuals.txt and loci.txt give the row orders
  README.md  what each file is and how the tools' parameters correspond

The run scripts write each tool's hybrid indices and genomic clines as
tab-delimited tables (hybrid_index.tsv, clines.tsv) in its folder.

Example:
    python benchmarking/simulate_vcf.py --n-snps 100 --aim-percent 100 \\
        --format-beagle --vcf-genotypes called --mean-depth 4 -o sim.vcf
    python benchmarking/make_tool_inputs.py --vcf sim.vcf --beagle sim.beagle \\
        --metadata sim_metadata.tsv --pop1-label pop1 --pop2-label pop2 \\
        --outdir tool_inputs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def read_vcf_gt_ad(path: Path) -> tuple[list, list, np.ndarray, np.ndarray | None]:
    """Locus IDs, sample names, ALT dosage (loci x samples, NaN missing) and,
    if every record has AD, REF/ALT read counts (loci x samples x 2)."""
    loci, dosage, ad, has_ad = [], [], [], True
    samples = None
    gt_code = {"0/0": 0, "0|0": 0, "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1,
               "1/1": 2, "1|1": 2}
    with open(path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = fields[9:]
                continue
            if len(fields[4].split(",")) != 1:
                continue
            fmt = fields[8].split(":")
            gi = fmt.index("GT")
            ai = fmt.index("AD") if "AD" in fmt else None
            has_ad &= ai is not None
            loci.append(fields[2] if fields[2] != "." else f"{fields[0]}:{fields[1]}")
            row, row_ad = [], []
            for cell in fields[9:]:
                parts = cell.split(":")
                row.append(gt_code.get(parts[gi], np.nan))
                if ai is not None:
                    c = parts[ai].split(",") if ai < len(parts) else ["0", "0"]
                    row_ad.append([int(x) if x not in (".", "") else 0 for x in c[:2]])
            dosage.append(row)
            ad.append(row_ad)
    dosage = np.array(dosage, dtype=float)
    return loci, samples, dosage, (np.array(ad, dtype=int) if has_ad else None)


RUN_HYCLINE = """#!/usr/bin/env bash
# HyCline: hybrid index, genomic-cline input, then genomic clines (MLE and
# hierarchical Bayesian). Run from this folder; HYCLINE defaults to the
# repository's run_hycline.py.
set -euo pipefail
HYCLINE=${{HYCLINE:-"python {run_hycline}"}}
INPUT={input}
$HYCLINE estimate-hybrid-index --input $INPUT --metadata metadata.tsv \\
    --pop1-label {pop1} --pop2-label {pop2} --output hybrid_index.tsv
$HYCLINE generate-genomic-input --input $INPUT --metadata hybrid_index.tsv \\
    --parental-metadata metadata.tsv --pop1-label {pop1} --pop2-label {pop2} \\
    --output genomic_input.csv
$HYCLINE fit-genomic --input genomic_input.csv --individual-col individual \\
    --no-plot --output-dir fit_mle
$HYCLINE fit-hierarchical-genomic --input genomic_input.csv --individual-col individual \\
    --no-plot --output-dir fit_hier
"""

RUN_GGHYBRID = """# gghybrid (Bailey 2023): hybrid index (esth) then per-locus genomic clines
# (ggcline). Run from this folder: Rscript run_gghybrid.R
suppressMessages({{library(gghybrid); library(data.table)}})
set.seed(1)
dat <- read.data("gghybrid_input.csv", nprecol = 2, MISSINGVAL = NA, NUMINDS = {n_ind})
prep <- data.prep(data = dat$data, loci = dat$loci, alleles = dat$alleles,
                  S0 = "{pop1}", S1 = "{pop2}", precols = dat$precols)
pdf(NULL)
hi <- esth(data.prep.object = prep$data.prep, read.data.precols = dat$precols,
           include.Source = TRUE, nitt = {hi_nitt}, burnin = {hi_burnin})
fwrite(hi$hi, "hybrid_index.tsv", sep = "\\t")
gc <- ggcline(data.prep.object = prep$data.prep, esth.object = hi,
              read.data.precols = dat$precols, nitt = {gc_nitt}, burnin = {gc_burnin})
fwrite(gc$gc, "clines.tsv", sep = "\\t")
"""

RUN_BGCHM = """# bgchm (Gompert, DeRaad & Buerkle 2024): parental allele frequencies,
# hybrid indexes, then a hierarchical genomic cline model. Run from this
# folder: Rscript run_bgchm.R [genotype|glik]
suppressMessages(library(bgchm))
args <- commandArgs(trailingOnly = TRUE)
model <- if (length(args)) args[1] else "genotype"
rd <- function(f) as.matrix(read.delim(f, row.names = 1, check.names = FALSE))
G0 <- rd("G0.tsv"); G1 <- rd("G1.tsv"); Gx <- rd("Gx.tsv")
set.seed(1)
if (model == "glik") {{
    gl <- function(set) lapply(0:2, function(k) rd(sprintf("GL%d_%s.tsv", k, set)))
    GL0 <- gl("G0"); GL1 <- gl("G1"); GLx <- gl("Gx")
    # HMC = TRUE is bgchm's default for genotype likelihoods; its analytical
    # shortcut (HMC = FALSE) shrinks parental frequencies toward 0.5 at low depth.
    p <- est_p(G0 = GL0, G1 = GL1, model = "glik", ploidy = "diploid", HMC = TRUE,
               n_iters = {n_iters}, full = FALSE)
    h <- est_hi(Gx = GLx, p0 = p$p0[, 1], p1 = p$p1[, 1], model = "glik",
                ploidy = "diploid", n_iters = {n_iters}, full = FALSE)
    cl <- est_genocl(Gx = GLx, p0 = p$p0[, 1], p1 = p$p1[, 1], H = h$hi[, 1],
                     model = "glik", ploidy = "diploid", hier = TRUE,
                     n_iters = {n_iters}, full = TRUE)
}} else if (anyNA(G0) || anyNA(G1) || anyNA(Gx)) {{
    # Missing genotypes: bgchm encodes them with ploidy 0 under ploidy = "mixed".
    pl <- function(G) {{ m <- ifelse(is.na(G), 0, 2); G[is.na(G)] <- 0; list(G = G, pl = m) }}
    a0 <- pl(G0); a1 <- pl(G1); ax <- pl(Gx)
    p <- est_p(G0 = a0$G, G1 = a1$G, model = "genotype", ploidy = "mixed",
               pldat = list(ax$pl, a0$pl, a1$pl))
    h <- est_hi(Gx = ax$G, p0 = p$p0[, 1], p1 = p$p1[, 1], model = "genotype",
                ploidy = "mixed", pldat = ax$pl, n_iters = {n_iters}, full = FALSE)
    cl <- est_genocl(Gx = ax$G, p0 = p$p0[, 1], p1 = p$p1[, 1], H = h$hi[, 1],
                     model = "genotype", ploidy = "mixed", pldat = ax$pl, hier = TRUE,
                     n_iters = {n_iters}, full = TRUE)
}} else {{
    p <- est_p(G0 = G0, G1 = G1, model = "genotype", ploidy = "diploid")
    h <- est_hi(Gx = Gx, p0 = p$p0[, 1], p1 = p$p1[, 1], model = "genotype",
                ploidy = "diploid", n_iters = {n_iters}, full = FALSE)
    cl <- est_genocl(Gx = Gx, p0 = p$p0[, 1], p1 = p$p1[, 1], H = h$hi[, 1],
                     model = "genotype", ploidy = "diploid", hier = TRUE,
                     n_iters = {n_iters}, full = TRUE)
}}
hi <- data.frame(individual = rownames(Gx), h$hi, check.names = FALSE)
names(hi)[-1] <- c("hi_median", "hi_q2.5", "hi_q5", "hi_q95", "hi_q97.5")
write.table(hi, "hybrid_index.tsv", sep = "\\t", quote = FALSE, row.names = FALSE)
q <- function(par) t(apply(rstan::extract(cl$gencline_hmc, par)[[1]], 2, quantile,
                           probs = c(.5, .025, .975)))
cc <- q("center"); vv <- q("v")
cl_out <- data.frame(locus = colnames(Gx), center = cc[, 1], center_q2.5 = cc[, 2],
                     center_q97.5 = cc[, 3], v = vv[, 1], v_q2.5 = vv[, 2], v_q97.5 = vv[, 3])
write.table(cl_out, "clines.tsv", sep = "\\t", quote = FALSE, row.names = FALSE)
cat("SDc", cl$SDc, "\\nSDv", cl$SDv, "\\n")
"""

README = """# Matched inputs for HyCline, gghybrid, bgchm and bgc

Source: `{vcf}`{beagle_note}, metadata `{metadata}`.
{n_loci} loci; {n_p0} `{pop1}` individuals (hybrid index 0), {n_p1} `{pop2}`
individuals (hybrid index 1) and {n_x} test individuals. Allele counts and
frequencies are of the VCF ALT allele; `individuals.txt` (in `bgc/`) and the
row names of the bgchm matrices give the individual order.

| Folder | Run | Writes |
|---|---|---|
| `hycline/` | `bash run_hycline.sh` | `hybrid_index.tsv`, `fit_mle/hycline_summary.csv`, `fit_hier/` |
| `gghybrid/` | `Rscript run_gghybrid.R` | `hybrid_index.tsv` (esth), `clines.tsv` (ggcline) |
| `bgchm/` | `Rscript run_bgchm.R genotype` (or `glik`) | `hybrid_index.tsv`, `clines.tsv` |
| `bgc/` | see below | bgc's HDF5/text output |

## Parameter correspondence

HyCline: logit(phi) = u + v logit(h). gghybrid and bgchm:
phi = h^v / (h^v + (1 - h)^v exp(u')), i.e. u' = -u, with cline centre
c = invlogit(-u / v) = invlogit(u' / v) (the h at which phi = 0.5). All three
model a locus's allele frequency as p0 + (p1 - p0) phi(h). bgc 1.03 uses a
different cline function (alpha and beta; Gompert & Buerkle 2011), so its
parameters are compared through outlier calls rather than directly.

## bgc 1.03

bgc is distributed from its authors' site (sites.google.com/site/bgcsoftware;
manual: bgcmanual.pdf). `genotypes/` holds known-genotype input: for each
locus, `p0in.txt`/`p1in.txt` give the parental REF and ALT allele counts, and
`admixedin.txt` gives, under a single `pop 0` line, each test individual's
REF and ALT allele copies (`0 0` = missing). {reads_note}A typical run (check
the flags against the manual for your bgc version):

    bgc -a genotypes/p0in.txt -b genotypes/p1in.txt -h genotypes/admixedin.txt \\
        -O 2 -x 50000 -n 25000 -t 10 -p 1 -q 1 -N 0 -o bgc_out

then `estpost` to summarize alpha, beta and the hybrid indexes.
"""


def write_inputs(vcf: Path, metadata: Path, outdir: Path, pop1: str, pop2: str,
                 beagle: Path | None = None, sample_col: str = "individual",
                 population_col: str = "population", hi_nitt: int = 3000,
                 hi_burnin: int = 1000, gc_nitt: int = 5000, gc_burnin: int = 2000,
                 bgchm_iters: int = 2000) -> dict:
    loci, samples, dosage, ad = read_vcf_gt_ad(vcf)
    meta = pd.read_csv(metadata, sep="\t", dtype={sample_col: str})
    pop_of = dict(zip(meta[sample_col], meta[population_col]))
    pops = np.array([pop_of.get(s, "") for s in samples], dtype=object)
    i0, i1 = np.where(pops == pop1)[0], np.where(pops == pop2)[0]
    ix = np.where((pops != pop1) & (pops != pop2))[0]
    if not len(i0) or not len(i1):
        raise ValueError(f"No samples labelled '{pop1}' or '{pop2}' in {metadata}.")
    for d in ("hycline", "gghybrid", "bgchm", "bgc/genotypes"):
        (outdir / d).mkdir(parents=True, exist_ok=True)

    # HyCline
    shutil.copyfile(vcf, outdir / "hycline" / "input.vcf")
    if beagle:
        shutil.copyfile(beagle, outdir / "hycline" / "input.beagle")
    pd.DataFrame({"individual": samples,
                  "population": [p if p in (pop1, pop2) else "test" for p in pops]}).to_csv(
        outdir / "hycline" / "metadata.tsv", sep="\t", index=False)
    (outdir / "hycline" / "run_hycline.sh").write_text(RUN_HYCLINE.format(
        run_hycline=REPO / "run_hycline.py", pop1=pop1, pop2=pop2,
        input="input.beagle" if beagle else "input.vcf"))

    # gghybrid: two rows per individual, allele 1 = ALT
    rows = []
    for j, s in enumerate(samples):
        g = dosage[:, j]
        a1 = np.where(np.isnan(g), np.nan, (g >= 1).astype(float))
        a2 = np.where(np.isnan(g), np.nan, (g >= 2).astype(float))
        label = pops[j] if pops[j] in (pop1, pop2) else "test"
        rows.append([s, label, *a1])
        rows.append([s, label, *a2])
    gg = pd.DataFrame(rows, columns=["INDLABEL", "POPID", *loci])
    gg.to_csv(outdir / "gghybrid" / "gghybrid_input.csv", index=False, na_rep="NA",
              float_format="%.0f")
    (outdir / "gghybrid" / "run_gghybrid.R").write_text(RUN_GGHYBRID.format(
        n_ind=len(samples), pop1=pop1, pop2=pop2, hi_nitt=hi_nitt, hi_burnin=hi_burnin,
        gc_nitt=gc_nitt, gc_burnin=gc_burnin))

    # bgchm
    def mat(idx, values):
        return pd.DataFrame(values[:, idx].T, index=[samples[i] for i in idx], columns=loci)
    for name, idx in (("G0", i0), ("G1", i1), ("Gx", ix)):
        mat(idx, dosage).to_csv(outdir / "bgchm" / f"{name}.tsv", sep="\t", na_rep="NA",
                                float_format="%.0f")
    if beagle:
        from hycline.data.beagle import read_beagle
        from hycline.data.filter_aims import beagle_missing_mask
        b = read_beagle(str(beagle))
        if list(b.sample_names) != list(samples) or len(b) != len(loci):
            raise ValueError("The beagle file must have the VCF's samples and loci, in order.")
        gls = [b.gl0.copy(), b.gl1.copy(), b.gl2.copy()]
        total = gls[0] + gls[1] + gls[2]
        miss = beagle_missing_mask(*gls)
        for k in range(3):
            gls[k] = np.where(miss, 1 / 3, gls[k] / np.where(total > 0, total, 1))
        for name, idx in (("G0", i0), ("G1", i1), ("Gx", ix)):
            for k in range(3):
                mat(idx, gls[k]).to_csv(outdir / "bgchm" / f"GL{k}_{name}.tsv", sep="\t",
                                        float_format="%.6g")
    (outdir / "bgchm" / "run_bgchm.R").write_text(RUN_BGCHM.format(n_iters=bgchm_iters))

    # bgc 1.03
    def bgc_files(folder: Path, per_ind_parental: bool, counts: np.ndarray):
        """counts: loci x samples x 2 (REF, ALT)."""
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "p0in.txt", "w") as f0, open(folder / "p1in.txt", "w") as f1, \
                open(folder / "admixedin.txt", "w") as fx:
            for l in range(len(loci)):
                for fh, idx in ((f0, i0), (f1, i1)):
                    fh.write(f"locus {l}\n")
                    if per_ind_parental:
                        fh.writelines(f"{counts[l, i, 0]} {counts[l, i, 1]}\n" for i in idx)
                    else:
                        fh.write(f"{counts[l, idx, 0].sum()} {counts[l, idx, 1].sum()}\n")
                fx.write(f"locus {l}\npop 0\n")
                fx.writelines(f"{counts[l, i, 0]} {counts[l, i, 1]}\n" for i in ix)

    allele_counts = np.stack([np.where(np.isnan(dosage), 0, 2 - dosage),
                              np.where(np.isnan(dosage), 0, dosage)], axis=-1).astype(int)
    bgc_files(outdir / "bgc" / "genotypes", False, allele_counts)
    if ad is not None:
        bgc_files(outdir / "bgc" / "reads", True, ad)
    (outdir / "bgc" / "individuals.txt").write_text("\n".join(samples[i] for i in ix) + "\n")
    (outdir / "bgc" / "loci.txt").write_text(
        "".join(f"{l}\t{n}\n" for l, n in enumerate(loci)))

    reads_note = ("`reads/` holds the same layout with per-individual REF/ALT read "
                  "counts in all three files, for the genotype-uncertainty model "
                  "(`-N 1`, with `-E` set to the sequencing error rate). " if ad is not None else "")
    (outdir / "README.md").write_text(README.format(
        vcf=vcf.name, beagle_note=f" and `{beagle.name}`" if beagle else "",
        metadata=metadata.name, n_loci=len(loci), n_p0=len(i0), n_p1=len(i1), n_x=len(ix),
        pop1=pop1, pop2=pop2, reads_note=reads_note))
    return {"loci": loci, "samples": samples, "n_p0": len(i0), "n_p1": len(i1),
            "n_test": len(ix), "has_reads": ad is not None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--beagle", default=None)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pop1-label", required=True, help="Parental population at hybrid index 0.")
    ap.add_argument("--pop2-label", required=True, help="Parental population at hybrid index 1.")
    ap.add_argument("--sample-col", default="individual")
    ap.add_argument("--population-col", default="population")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)
    info = write_inputs(Path(args.vcf), Path(args.metadata), Path(args.outdir),
                        args.pop1_label, args.pop2_label,
                        Path(args.beagle) if args.beagle else None,
                        args.sample_col, args.population_col)
    print(f"Wrote inputs for {len(info['loci'])} loci, {info['n_p0']} + {info['n_p1']} "
          f"parental and {info['n_test']} test individuals to {args.outdir}"
          + ("" if info["has_reads"] else " (no AD in the VCF, so no bgc read-count files)"))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())
