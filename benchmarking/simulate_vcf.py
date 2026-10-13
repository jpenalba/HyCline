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
population, hybrid_index, missing_fraction, plus mean_depth when reads are
simulated) usable directly with HyCline's --metadata/--sample-col/
--population-col options, and a truth CSV recording each SNP's AIM status
and realized parental frequencies (from the true, non-missing genotypes).

Genotype likelihoods: --format-beagle also writes an ANGSD-style beagle file
(`angsd -doGlf 2` layout) for the same samples and SNPs. Reads are simulated
for every non-missing genotype: a depth d (Poisson, or negative binomial with
--depth-dispersion, scaled per individual by --individual-depth-sd, and
redrawn if 0 so the missing-data settings stay exact), then d reads each
showing the ALT allele with probability (g/2)(1-e) + (1-g/2)e for true
dosage g and per-read error e (--seq-error). The three genotype likelihoods
are computed from the read counts and normalized to sum to 1; missing
genotypes are written as 0.333333/0.333333/0.333333, ANGSD's no-data
encoding. Beagle markers are named <chrom>_<pos>; allele1/allele2 are the
VCF's REF/ALT as ANGSD base codes (0=A, 1=C, 2=G, 3=T), so allele2 is the
focal allele in both files. --vcf-genotypes called writes the VCF GT as the
most likely genotype from those same reads instead of the true genotype,
giving matched hard-call and genotype-likelihood versions of one dataset.

Example:
    python benchmarking/simulate_vcf.py --n-pop1 20 --n-pop2 20 --n-popH 100 \\
        --n-snps 10000 --aim-percent 10 --aim-cutoff 0.05 \\
        --missing-global 0.05 \\
        --missing-individual sample3_pop1=0.5 sample45_popH=0.9 \\
        --output bench.vcf --seed 1

    # Low-coverage version with matched beagle output and called VCF genotypes
    python benchmarking/simulate_vcf.py --n-snps 10000 --aim-percent 10 \\
        --format-beagle --mean-depth 2 --seq-error 0.01 \\
        --vcf-genotypes called --output lowcov.vcf.gz --seed 1
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
    gl = p.add_argument_group(
        "genotype likelihoods / simulated reads",
        "Reads are simulated only when --format-beagle or --vcf-genotypes "
        "called is given; the depth and error flags have no effect otherwise.")
    gl.add_argument("--format-beagle", "--format_beagle", dest="format_beagle",
                    action="store_true",
                    help="Also write an ANGSD-style beagle genotype-likelihood "
                         "file for the same samples and SNPs.")
    gl.add_argument("--beagle-output", default=None,
                    help="Beagle output path. Default: <output stem>.beagle "
                         "(.beagle.gz when --output ends in .gz).")
    gl.add_argument("--mean-depth", type=float, default=10.0,
                    help="Mean read depth per non-missing genotype. Zero depths "
                         "are redrawn (missing data is set only by the missing-"
                         "data flags), so the realized mean is slightly higher "
                         "at very low depth, e.g. ~2.3 for 2 with Poisson depth.")
    gl.add_argument("--depth-dispersion", type=float, default=0.0,
                    help="Negative-binomial size parameter for read depth "
                         "(variance = mean + mean^2 / size; smaller = more "
                         "uneven depth across sites). 0 = Poisson depth.")
    gl.add_argument("--individual-depth-sd", type=float, default=0.0,
                    help="Log-scale SD of a per-individual depth multiplier "
                         "(mean 1), mimicking uneven sequencing effort across "
                         "samples. 0 = every individual has --mean-depth.")
    gl.add_argument("--seq-error", type=float, default=0.01,
                    help="Per-read probability of reading the wrong allele.")
    gl.add_argument("--vcf-genotypes", choices=["true", "called"], default="true",
                    help="VCF GT values: the true simulated genotypes, or the "
                         "most likely genotype given the simulated reads (hard "
                         "calls with realistic low-depth errors). With "
                         "'called', the VCF also gets a DP field. AIM "
                         "frequency guarantees apply to the true genotypes.")
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
    if args.mean_depth <= 0:
        p.error("--mean-depth must be > 0")
    if args.depth_dispersion < 0:
        p.error("--depth-dispersion must be >= 0")
    if args.individual_depth_sd < 0:
        p.error("--individual-depth-sd must be >= 0")
    if not 0.0 <= args.seq_error < 0.5:
        p.error("--seq-error must be in [0, 0.5)")
    if args.beagle_output and not args.format_beagle:
        p.error("--beagle-output requires --format-beagle")
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


def draw_depths(rng, lam: np.ndarray, dispersion: float) -> np.ndarray:
    """Read depths >= 1 with per-sample means `lam` (zero draws are redrawn,
    so a non-missing genotype always has at least one read)."""
    def draw(m):
        if dispersion > 0:
            return rng.negative_binomial(dispersion, dispersion / (dispersion + m))
        return rng.poisson(m)

    depth = draw(lam)
    zero = depth == 0
    while zero.any():
        depth[zero] = draw(lam[zero])
        zero = depth == 0
    return depth


def genotype_likelihoods(rng, geno: np.ndarray, depth: np.ndarray, err: float):
    """Simulate ALT read counts for true dosages `geno` at depths `depth` and
    return (n_samples, 3) normalized likelihoods for dosage 0/1/2."""
    q = np.array([err, 0.5, 1.0 - err])  # P(read shows ALT | dosage 0/1/2)
    alt_reads = rng.binomial(depth, q[geno])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q, log_1mq = np.log(q), np.log1p(-q)
        loglik = (alt_reads[:, None] * log_q[None, :]
                  + (depth - alt_reads)[:, None] * log_1mq[None, :])
    # err = 0 gives log(0) terms; 0 * -inf must count as 0, not NaN.
    loglik = np.where(np.isnan(loglik), 0.0, loglik)
    loglik -= loglik.max(axis=1, keepdims=True)
    lik = np.exp(loglik)
    return lik / lik.sum(axis=1, keepdims=True)


BASE_CODE = {"A": "0", "C": "1", "G": "2", "T": "3"}
GT_STR = {0: "0/0", 1: "0/1", 2: "1/1"}


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    # Reads get their own stream, so turning beagle output or called
    # genotypes on/off never changes the simulated genotypes for a given seed.
    read_rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0])

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
    beagle_path = None
    if args.format_beagle:
        beagle_path = args.beagle_output or (
            f"{stem}.beagle.gz" if out_vcf.endswith(".gz") else f"{stem}.beagle")

    simulate_reads = args.format_beagle or args.vcf_genotypes == "called"
    if simulate_reads:
        sd = args.individual_depth_sd
        depth_mult = (read_rng.lognormal(-0.5 * sd * sd, sd, size=n_samples)
                      if sd > 0 else np.ones(n_samples))
        sample_lam = args.mean_depth * depth_mult
        depth_sum = np.zeros(n_samples, dtype=np.int64)

    snps_per_chrom = int(np.ceil(args.n_snps / args.n_chrom))
    write_dp = args.vcf_genotypes == "called"
    fmt = "GT:DP" if write_dp else "GT"
    bases = np.array(list("ACGT"))

    def _open(path):
        return gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")

    beagle = _open(beagle_path) if beagle_path else None
    if beagle:
        beagle.write("marker\tallele1\tallele2\t"
                     + "\t".join(f"{n}\t{n}\t{n}" for n in samples) + "\n")
    missing_gl = "0.333333\t0.333333\t0.333333"

    with _open(out_vcf) as vcf, open(truth_path, "w") as truth:
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write("##source=HyCline_benchmarking_simulate_vcf\n")
        for c in range(args.n_chrom):
            length = snps_per_chrom * args.snp_spacing + args.snp_spacing
            vcf.write(f"##contig=<ID=chr{c + 1},length={length}>\n")
        vcf.write('##INFO=<ID=AIM,Number=0,Type=Flag,'
                  'Description="Simulated ancestry-informative marker">\n')
        vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        if write_dp:
            vcf.write('##FORMAT=<ID=DP,Number=1,Type=Integer,'
                      'Description="Simulated read depth">\n')
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(samples) + "\n")

        truth.write("chrom,pos,id,marker,is_aim,freq_pop1,freq_pop2,low_pop\n")

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
            marker = f"{chrom}_{pos}"

            if simulate_reads:
                called = ~missing
                depth = np.zeros(n_samples, dtype=np.int64)
                depth[called] = draw_depths(read_rng, sample_lam[called],
                                            args.depth_dispersion)
                depth_sum += depth
                lik = np.full((n_samples, 3), 1.0 / 3.0)
                lik[called] = genotype_likelihoods(
                    read_rng, geno[called], depth[called], args.seq_error)
                if beagle:
                    cells = [missing_gl if m else
                             f"{l0:.6f}\t{l1:.6f}\t{l2:.6f}"
                             for m, (l0, l1, l2) in zip(missing, lik)]
                    beagle.write(f"{marker}\t{BASE_CODE[ref]}\t{BASE_CODE[alt]}\t"
                                 + "\t".join(cells) + "\n")

            if args.vcf_genotypes == "called":
                vcf_geno = lik.argmax(axis=1)
                fields = ["./.:0" if m else f"{GT_STR[int(g)]}:{d}"
                          for g, m, d in zip(vcf_geno, missing, depth)]
            else:
                fields = ["./." if m else GT_STR[int(g)]
                          for g, m in zip(geno, missing)]

            info = "AIM" if aim else "."
            vcf.write(f"{chrom}\t{pos}\t{snp_id}\t{ref}\t{alt}\t.\tPASS\t{info}\t"
                      f"{fmt}\t" + "\t".join(fields) + "\n")

            low_pop = ("pop1" if f1 < f2 else "pop2") if aim and called1.any() and called2.any() else ""
            truth.write(f"{chrom},{pos},{snp_id},{marker},{int(aim)},"
                        f"{f1:.6f},{f2:.6f},{low_pop}\n")

    if beagle:
        beagle.close()

    with open(meta_path, "w") as meta:
        meta.write("individual,population,hybrid_index,missing_fraction"
                   + (",mean_depth" if simulate_reads else "") + "\n")
        hi_iter = iter(hybrid_index)
        for i, (name, pop, n_miss) in enumerate(zip(samples, pops, missing_per_sample)):
            if pop == "pop1":
                hi = "0"
            elif pop == "pop2":
                hi = "1"
            else:
                hi = f"{next(hi_iter):.6f}"
            row = f"{name},{pop},{hi},{n_miss / args.n_snps:.6f}"
            if simulate_reads:
                n_called = args.n_snps - n_miss
                row += f",{depth_sum[i] / n_called:.3f}" if n_called else ",0"
            meta.write(row + "\n")

    print(f"Wrote {out_vcf}: {n_samples} samples "
          f"({args.n_pop1} pop1, {args.n_pop2} pop2, {args.n_popH} popH), "
          f"{args.n_snps} SNPs ({n_aims} AIMs at cutoff {args.aim_cutoff}).")
    total_missing = int(missing_per_sample.sum())
    print(f"Missing genotypes: {total_missing} of {n_samples * args.n_snps} "
          f"({total_missing / (n_samples * args.n_snps):.4%}).")
    if simulate_reads:
        n_called = n_samples * args.n_snps - total_missing
        print(f"Mean simulated read depth per called genotype: "
              f"{depth_sum.sum() / max(n_called, 1):.2f}.")
    if beagle_path:
        print(f"Wrote {beagle_path}.")
    print(f"Wrote {meta_path} and {truth_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
