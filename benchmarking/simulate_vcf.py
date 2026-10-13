#!/usr/bin/env python3
"""Generate a fake multi-sample VCF for benchmarking HyCline commands.

Samples belong to three populations -- two parentals (pop1, pop2) and a
hybrid population (popH) -- and are named "sample<N>_<pop>", numbered
consecutively across the whole file (sample1_pop1, ..., sample11_pop2, ...).

A chosen fraction of SNPs are ancestry-informative markers (AIMs). For each
AIM, the realized alternate-allele frequency computed from the pop1 and pop2
individuals is exactly within [0, cutoff] in one parental population and
within [1 - cutoff, 1] in the other; which parental population carries the
low frequency is chosen at random per AIM. Non-AIM SNPs share a common
frequency across pop1 and pop2 and are re-drawn if their realized
frequencies happen to meet the AIM criterion, so the AIM count is exact.

Each popH individual gets a hybrid index h (fraction of pop2 ancestry), and
its genotypes are drawn from p = (1 - h) * p_pop1 + h * p_pop2 at every SNP.

Alongside the VCF the script writes a metadata CSV (columns: individual,
population) usable directly with HyCline's --metadata/--sample-col/
--population-col options, and a truth CSV recording each SNP's AIM status
and realized parental frequencies.

Example:
    python benchmarking/simulate_vcf.py --n-pop1 20 --n-pop2 20 --n-popH 100 \\
        --n-snps 10000 --aim-percent 10 --aim-cutoff 0.05 \\
        --output bench.vcf --seed 1
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Simulate a multi-sample VCF (pop1, pop2, popH) with a "
                    "controlled fraction of ancestry-informative markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-pop1", type=int, default=20,
                   help="Number of samples in parental population pop1.")
    p.add_argument("--n-pop2", type=int, default=20,
                   help="Number of samples in parental population pop2.")
    p.add_argument("--n-popH", type=int, default=50,
                   help="Number of samples in the hybrid population popH.")
    p.add_argument("--n-snps", type=int, default=1000,
                   help="Total number of SNPs in the VCF.")
    p.add_argument("--aim-percent", type=float, default=10.0,
                   help="Percentage (0-100) of SNPs that are ancestry-informative "
                        "markers. The AIM count is round(n_snps * percent / 100).")
    p.add_argument("--aim-cutoff", type=float, default=0.05,
                   help="AIM allele-frequency cutoff: an AIM has frequency in "
                        "[0, cutoff] in one parental population and in "
                        "[1 - cutoff, 1] in the other. Must be in [0, 0.5).")
    p.add_argument("--hybrid-index", choices=["uniform", "beta", "fixed"],
                   default="uniform",
                   help="Distribution of popH hybrid indices (fraction pop2 "
                        "ancestry): uniform(0,1), beta(--hi-alpha, --hi-beta), "
                        "or fixed at --hi-value for every hybrid.")
    p.add_argument("--hi-alpha", type=float, default=2.0,
                   help="Alpha for --hybrid-index beta.")
    p.add_argument("--hi-beta", type=float, default=2.0,
                   help="Beta for --hybrid-index beta.")
    p.add_argument("--hi-value", type=float, default=0.5,
                   help="Hybrid index for --hybrid-index fixed.")
    p.add_argument("--n-chrom", type=int, default=1,
                   help="Number of chromosomes to spread SNPs across.")
    p.add_argument("--snp-spacing", type=int, default=1000,
                   help="Base-pair spacing between consecutive SNPs on a chromosome.")
    p.add_argument("--missing-rate", type=float, default=0.0,
                   help="Fraction of genotypes set to missing (./.) at random. "
                        "Applied after AIM frequencies are fixed, so it does not "
                        "change which SNPs are AIMs but can shift realized "
                        "frequencies among non-missing calls slightly.")
    p.add_argument("--include-pl", action="store_true",
                   help="Also write a PL (Phred-scaled genotype likelihood) "
                        "FORMAT field, derived from the true genotype with "
                        "--pl-error per-genotype error.")
    p.add_argument("--pl-error", type=float, default=0.01,
                   help="Error probability used to build PL values.")
    p.add_argument("--output", "-o", required=True,
                   help="Output VCF path (.vcf or .vcf.gz).")
    p.add_argument("--metadata", default=None,
                   help="Output metadata CSV path (individual,population). "
                        "Default: <output stem>_metadata.csv.")
    p.add_argument("--truth", default=None,
                   help="Output per-SNP truth CSV path. "
                        "Default: <output stem>_truth.csv.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    args = p.parse_args(argv)

    for name in ("n_pop1", "n_pop2"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.n_popH < 0:
        p.error("--n-popH must be >= 0")
    if args.n_snps < 1:
        p.error("--n-snps must be >= 1")
    if not 0.0 <= args.aim_percent <= 100.0:
        p.error("--aim-percent must be between 0 and 100")
    if not 0.0 <= args.aim_cutoff < 0.5:
        p.error("--aim-cutoff must be in [0, 0.5)")
    if not 0.0 <= args.missing_rate < 1.0:
        p.error("--missing-rate must be in [0, 1)")
    if not 0.0 <= args.hi_value <= 1.0:
        p.error("--hi-value must be in [0, 1]")
    if args.n_chrom < 1 or args.snp_spacing < 1:
        p.error("--n-chrom and --snp-spacing must be >= 1")
    if not 0.0 < args.pl_error < 1.0:
        p.error("--pl-error must be in (0, 1)")
    return args


def _stem(path: str) -> Path:
    path = Path(path)
    name = path.name
    for suffix in (".vcf.gz", ".vcf"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return path.with_name(name)


def genotypes_with_exact_count(rng, n_ind: int, n_alt: int) -> np.ndarray:
    """Return n_ind diploid genotypes (0/1/2) carrying exactly n_alt alt alleles."""
    alleles = np.zeros(2 * n_ind, dtype=np.int8)
    alleles[:n_alt] = 1
    rng.shuffle(alleles)
    return alleles.reshape(n_ind, 2).sum(axis=1)


def is_aim(freq_a: float, freq_b: float, cutoff: float) -> bool:
    lo, hi = cutoff, 1.0 - cutoff
    return (freq_a <= lo and freq_b >= hi) or (freq_b <= lo and freq_a >= hi)


def simulate_parentals(rng, n1: int, n2: int, aim: bool, cutoff: float):
    """Genotypes for pop1 and pop2 at one SNP, plus the realized frequencies."""
    a1, a2 = 2 * n1, 2 * n2
    if aim:
        low_is_pop1 = rng.random() < 0.5
        # Allowed alt-allele counts so realized frequency is within range.
        def low_count(n_alleles):
            return int(rng.integers(0, int(np.floor(cutoff * n_alleles + 1e-9)) + 1))

        def high_count(n_alleles):
            return n_alleles - low_count(n_alleles)

        k1 = low_count(a1) if low_is_pop1 else high_count(a1)
        k2 = high_count(a2) if low_is_pop1 else low_count(a2)
        g1 = genotypes_with_exact_count(rng, n1, k1)
        g2 = genotypes_with_exact_count(rng, n2, k2)
        return g1, g2, k1 / a1, k2 / a2

    while True:
        q = rng.uniform(0.05, 0.95)
        g1 = rng.binomial(2, q, size=n1).astype(np.int8)
        g2 = rng.binomial(2, q, size=n2).astype(np.int8)
        f1, f2 = g1.sum() / a1, g2.sum() / a2
        if not is_aim(f1, f2, cutoff):
            return g1, g2, f1, f2


def pl_string(gt: int, err: float) -> str:
    # Probability (1 - err) on the true genotype, err/2 on each other.
    probs = np.full(3, err / 2.0)
    probs[gt] = 1.0 - err
    phred = -10.0 * np.log10(probs)
    phred = np.rint(phred - phred.min()).astype(int)
    return ",".join(map(str, phred))


GT_STR = {0: "0/0", 1: "0/1", 2: "1/1"}


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    pops = (["pop1"] * args.n_pop1 + ["pop2"] * args.n_pop2
            + ["popH"] * args.n_popH)
    samples = [f"sample{i + 1}_{pop}" for i, pop in enumerate(pops)]
    n_samples = len(samples)

    if args.hybrid_index == "uniform":
        hybrid_index = rng.uniform(0.0, 1.0, size=args.n_popH)
    elif args.hybrid_index == "beta":
        hybrid_index = rng.beta(args.hi_alpha, args.hi_beta, size=args.n_popH)
    else:
        hybrid_index = np.full(args.n_popH, args.hi_value)

    n_aims = int(round(args.n_snps * args.aim_percent / 100.0))
    aim_flags = np.zeros(args.n_snps, dtype=bool)
    aim_flags[rng.choice(args.n_snps, size=n_aims, replace=False)] = True

    out_vcf = args.output
    stem = _stem(out_vcf)
    meta_path = args.metadata or f"{stem}_metadata.csv"
    truth_path = args.truth or f"{stem}_truth.csv"

    snps_per_chrom = int(np.ceil(args.n_snps / args.n_chrom))
    fmt = "GT:PL" if args.include_pl else "GT"
    bases = np.array(list("ACGT"))

    opener = gzip.open if out_vcf.endswith(".gz") else open
    with opener(out_vcf, "wt") as vcf, open(truth_path, "w") as truth:
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write("##source=HyCline_benchmarking_simulate_vcf\n")
        for c in range(args.n_chrom):
            length = snps_per_chrom * args.snp_spacing + args.snp_spacing
            vcf.write(f"##contig=<ID=chr{c + 1},length={length}>\n")
        vcf.write('##INFO=<ID=AIM,Number=0,Type=Flag,'
                  'Description="Simulated ancestry-informative marker">\n')
        vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        if args.include_pl:
            vcf.write('##FORMAT=<ID=PL,Number=G,Type=Integer,'
                      'Description="Phred-scaled genotype likelihoods">\n')
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(samples) + "\n")

        truth.write("chrom,pos,id,is_aim,freq_pop1,freq_pop2,low_pop\n")

        for s in range(args.n_snps):
            chrom = f"chr{s // snps_per_chrom + 1}"
            pos = (s % snps_per_chrom + 1) * args.snp_spacing
            snp_id = f"snp{s + 1}"
            aim = bool(aim_flags[s])

            g1, g2, f1, f2 = simulate_parentals(
                rng, args.n_pop1, args.n_pop2, aim, args.aim_cutoff)
            if args.n_popH:
                p_h = (1.0 - hybrid_index) * f1 + hybrid_index * f2
                gh = rng.binomial(2, p_h).astype(np.int8)
                geno = np.concatenate([g1, g2, gh])
            else:
                geno = np.concatenate([g1, g2])

            missing = (rng.random(n_samples) < args.missing_rate
                       if args.missing_rate > 0 else np.zeros(n_samples, bool))

            ref, alt = rng.choice(bases, size=2, replace=False)
            fields = []
            for g, miss in zip(geno, missing):
                if miss:
                    fields.append("./.:." if args.include_pl else "./.")
                elif args.include_pl:
                    fields.append(f"{GT_STR[int(g)]}:{pl_string(int(g), args.pl_error)}")
                else:
                    fields.append(GT_STR[int(g)])

            info = "AIM" if aim else "."
            vcf.write(f"{chrom}\t{pos}\t{snp_id}\t{ref}\t{alt}\t.\tPASS\t{info}\t"
                      f"{fmt}\t" + "\t".join(fields) + "\n")

            low_pop = ("pop1" if f1 < f2 else "pop2") if aim else ""
            truth.write(f"{chrom},{pos},{snp_id},{int(aim)},{f1:.6f},{f2:.6f},{low_pop}\n")

    with open(meta_path, "w") as meta:
        meta.write("individual,population,hybrid_index\n")
        hi_iter = iter(hybrid_index)
        for name, pop in zip(samples, pops):
            if pop == "pop1":
                hi = "0"
            elif pop == "pop2":
                hi = "1"
            else:
                hi = f"{next(hi_iter):.6f}"
            meta.write(f"{name},{pop},{hi}\n")

    print(f"Wrote {out_vcf}: {n_samples} samples "
          f"({args.n_pop1} pop1, {args.n_pop2} pop2, {args.n_popH} popH), "
          f"{args.n_snps} SNPs ({n_aims} AIMs at cutoff {args.aim_cutoff}).")
    print(f"Wrote {meta_path} and {truth_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
