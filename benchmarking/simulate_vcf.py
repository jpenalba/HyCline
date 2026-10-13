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

Missing data (./.) comes in two forms, both exact counts rather than
per-genotype coin flips:
  --missing-global F       a fraction F of all genotype calls belonging to
                           the samples not named in --missing-individual is
                           set missing, placed uniformly at random across
                           those samples and SNPs.
  --missing-individual S=F sample S is missing at exactly round(F * n_snps)
                           randomly chosen SNPs. Repeatable / multi-valued.
                           A named sample's rate replaces the global rate
                           for that sample.
Missingness is decided before genotypes are drawn, and AIM status is
enforced on the called (non-missing) pop1 and pop2 genotypes, so missing
data never pushes an AIM's realized frequency outside the cutoff.

Alongside the VCF the script writes a metadata CSV (columns: individual,
population, hybrid_index, missing_fraction) usable directly with HyCline's
--metadata/--sample-col/--population-col options, and a truth CSV recording
each SNP's AIM status and realized parental frequencies (from called
genotypes).

Example:
    python benchmarking/simulate_vcf.py --n-pop1 20 --n-pop2 20 --n-popH 100 \\
        --n-snps 10000 --aim-percent 10 --aim-cutoff 0.05 \\
        --missing-global 0.05 \\
        --missing-individual sample3_pop1=0.5 sample45_popH=0.9 \\
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
    p.add_argument("--missing-global", type=float, default=0.0,
                   help="Fraction (0-1) of genotype calls set to missing (./.), "
                        "spread uniformly at random across all samples and SNPs. "
                        "Samples named in --missing-individual are excluded from "
                        "this pool and use their own rate instead.")
    p.add_argument("--missing-individual", nargs="+", action="extend",
                   default=[], metavar="SAMPLE=FRAC",
                   help="Per-sample missing fraction, e.g. "
                        "sample3_pop1=0.5 sample45_popH=0.9. The sample is "
                        "missing at exactly round(FRAC * n_snps) random SNPs. "
                        "Can be given several values and/or repeated.")
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
    if not 0.0 <= args.missing_global <= 1.0:
        p.error("--missing-global must be in [0, 1]")
    missing_individual = {}
    for item in args.missing_individual:
        name, sep, frac = item.rpartition("=")
        if not sep or not name:
            p.error(f"--missing-individual expects SAMPLE=FRAC, got {item!r}")
        try:
            frac = float(frac)
        except ValueError:
            p.error(f"--missing-individual: {frac!r} is not a number ({item!r})")
        if not 0.0 <= frac <= 1.0:
            p.error(f"--missing-individual: fraction must be in [0, 1] ({item!r})")
        if name in missing_individual:
            p.error(f"--missing-individual: {name} given more than once")
        missing_individual[name] = frac
    args.missing_individual = missing_individual
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
    """Genotypes for the n1 called pop1 and n2 called pop2 samples at one SNP,
    plus the realized frequencies (NaN for a population with no calls)."""
    a1, a2 = 2 * n1, 2 * n2
    if n1 == 0 or n2 == 0:
        # AIM status is undefined without calls in both parentals; draw
        # genotypes for whichever side has calls and report NaN frequencies.
        q = rng.uniform(0.0, 1.0)
        g1 = rng.binomial(2, q, size=n1).astype(np.int8)
        g2 = rng.binomial(2, q, size=n2).astype(np.int8)
        f1 = g1.sum() / a1 if n1 else np.nan
        f2 = g2.sum() / a2 if n2 else np.nan
        return g1, g2, f1, f2
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


class MissingMasks:
    """Per-SNP missingness masks with exact global and per-sample counts.

    Targeted samples get a precomputed mask over SNPs. The remaining samples
    share an exact global count of missing cells, allocated SNP by SNP with
    sequential hypergeometric draws (an exact uniform allocation). numpy's
    hypergeometric requires both counts < 1e9, so for very large files the
    early SNPs use the binomial approximation, clamped so the remaining draws
    can still hit the total exactly.
    """

    def __init__(self, rng, n_snps, n_samples, global_frac, individual):
        self.rng = rng
        self.n_samples = n_samples
        self.targeted = {}
        for idx, frac in individual.items():
            mask = np.zeros(n_snps, dtype=bool)
            k = int(round(frac * n_snps))
            mask[rng.choice(n_snps, size=k, replace=False)] = True
            self.targeted[idx] = mask
        self.pool = np.array([i for i in range(n_samples)
                              if i not in self.targeted], dtype=np.int64)
        self.cells_left = n_snps * len(self.pool)
        self.k_left = int(round(global_frac * self.cells_left))

    def next(self, snp: int) -> np.ndarray:
        mask = np.zeros(self.n_samples, dtype=bool)
        for idx, tmask in self.targeted.items():
            mask[idx] = tmask[snp]
        n_pool = len(self.pool)
        if n_pool and self.k_left:
            good, bad = self.k_left, self.cells_left - self.k_left
            if good < 10**9 and bad < 10**9:
                m = int(self.rng.hypergeometric(good, bad, n_pool))
            else:
                m = int(self.rng.binomial(n_pool, good / self.cells_left))
                m = min(m, good, n_pool)
                m = max(m, good - (self.cells_left - n_pool))
            if m:
                mask[self.rng.choice(self.pool, size=m, replace=False)] = True
            self.k_left -= m
        self.cells_left -= n_pool
        return mask


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
    pop_arr = np.array(pops)

    name_to_idx = {name: i for i, name in enumerate(samples)}
    unknown = [n for n in args.missing_individual if n not in name_to_idx]
    if unknown:
        print(f"ERROR: --missing-individual names not among the simulated "
              f"samples: {', '.join(unknown)}. Sample names look like "
              f"{samples[0]}.", file=sys.stderr)
        return 2

    if args.hybrid_index == "uniform":
        hybrid_index = rng.uniform(0.0, 1.0, size=args.n_popH)
    elif args.hybrid_index == "beta":
        hybrid_index = rng.beta(args.hi_alpha, args.hi_beta, size=args.n_popH)
    else:
        hybrid_index = np.full(args.n_popH, args.hi_value)

    n_aims = int(round(args.n_snps * args.aim_percent / 100.0))
    aim_flags = np.zeros(args.n_snps, dtype=bool)
    aim_flags[rng.choice(args.n_snps, size=n_aims, replace=False)] = True

    masks = MissingMasks(
        rng, args.n_snps, n_samples, args.missing_global,
        {name_to_idx[n]: f for n, f in args.missing_individual.items()})
    missing_per_sample = np.zeros(n_samples, dtype=np.int64)

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

            missing = masks.next(s)
            missing_per_sample += missing
            called1 = ~missing & (pop_arr == "pop1")
            called2 = ~missing & (pop_arr == "pop2")
            g1, g2, f1, f2 = simulate_parentals(
                rng, int(called1.sum()), int(called2.sum()), aim, args.aim_cutoff)

            # Missing samples get a placeholder 0; they are written as ./.
            geno = np.zeros(n_samples, dtype=np.int8)
            geno[called1] = g1
            geno[called2] = g2
            if args.n_popH:
                # Hybrids draw from the parental frequencies; fall back to the
                # other parental (or 0.5) if one side has no calls at this SNP.
                p1 = f1 if not np.isnan(f1) else (f2 if not np.isnan(f2) else 0.5)
                p2 = f2 if not np.isnan(f2) else p1
                p_h = (1.0 - hybrid_index) * p1 + hybrid_index * p2
                geno[pop_arr == "popH"] = rng.binomial(2, p_h).astype(np.int8)

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

            low_pop = ("pop1" if f1 < f2 else "pop2") if aim and called1.any() and called2.any() else ""
            truth.write(f"{chrom},{pos},{snp_id},{int(aim)},{f1:.6f},{f2:.6f},{low_pop}\n")

    with open(meta_path, "w") as meta:
        meta.write("individual,population,hybrid_index,missing_fraction\n")
        hi_iter = iter(hybrid_index)
        for name, pop, n_miss in zip(samples, pops, missing_per_sample):
            if pop == "pop1":
                hi = "0"
            elif pop == "pop2":
                hi = "1"
            else:
                hi = f"{next(hi_iter):.6f}"
            meta.write(f"{name},{pop},{hi},{n_miss / args.n_snps:.6f}\n")

    print(f"Wrote {out_vcf}: {n_samples} samples "
          f"({args.n_pop1} pop1, {args.n_pop2} pop2, {args.n_popH} popH), "
          f"{args.n_snps} SNPs ({n_aims} AIMs at cutoff {args.aim_cutoff}).")
    total_missing = int(missing_per_sample.sum())
    print(f"Missing genotypes: {total_missing} of {n_samples * args.n_snps} "
          f"({total_missing / (n_samples * args.n_snps):.4%}).")
    print(f"Wrote {meta_path} and {truth_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
