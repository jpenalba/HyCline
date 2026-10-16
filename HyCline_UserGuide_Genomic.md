# HyCline User Guide: Genomic Cline Analysis

**Document status: working draft, updated alongside HyCline's own
development.**

This guide walks through HyCline's genomic-cline pipeline: fitting the
logit-logistic genomic cline model (Fitzpatrick 2013) of per-locus ancestry
against genome-wide hybrid index, the standard way to find loci that
introgress more freely (or are more resistant to introgression) than the
genome-wide average.

Every command shown below was run for real against synthetic test data
while writing this guide -- flags, output messages, and file names are
exact.

**File formats.** Files you create yourself -- metadata (sample names,
population labels, distances, hybrid indices) and parental reference
labels -- are **tab-delimited** text with
one header row (`.tsv`). HyCline rejects a comma-separated file there
with a message saying so. The files HyCline writes for its own later
steps (`generate-*-input` output, fit summaries and curves) are CSV;
`estimate-hybrid-index` writes a tab-delimited table, since it is meant
to be used as metadata.

## Pipeline overview

```
genome-wide genotype data (VCF or ANGSD beagle)
        |
        v
[1] filter-aims  ------------------------------  (skip if you already have
        |                                          a curated AIM panel)
        v
[2] estimate-hybrid-index          --> genome-wide hybrid index per individual
        |
        v
[3] generate-genomic-input         --> VCF: hard counts, always
        |                              --> beagle: genotype likelihoods, always
        |                              (+ each locus's parental allele frequencies)
        v
[4] fit genomic clines
        |
        +-- fit-genomic               (independent per-locus fitting)
        +-- fit-hierarchical-genomic   (partial pooling across loci)
```

**Only two commands handle fitting, regardless of locus count or data
type.** `fit-genomic` and `fit-hierarchical-genomic` auto-detect
everything about the input -- hard counts vs. genotype likelihoods, one
locus vs. thousands -- from the file itself, with no flags needed to tell
them what they're looking at. `filter-aims` is shared with the
geographic-cline pipeline; see that guide for its own full flag reference
and the summary file (AIM count, missing data, warnings) it writes.

Once you have per-locus cline fits, a separate guide covers across-locus
coupling / concordance analysis (`coupling-genomic`), which measures how
concordant cline shapes are across your fitted loci and estimates
Firneno et al. (2023)'s coupling coefficient (theta).

---

## Step 1: Filter for ancestry-informative markers (AIMs)

```bash
hycline filter-aims \
    --input genomewide.vcf \
    --metadata parental_labels.tsv \
    --population-col population \
    --pop1-label parent0 --pop2-label parent1 \
    --output filtered.vcf
```

Real output:

```
Detected input format: vcf (from file extension; use --format to override)
Reading vcf data from 'genomewide.vcf' ...
  parent0: 6 individuals; parent1: 6 individuals
  7 loci checked, 3 retained (<= 0.1 in one parental population and >= 0.9 in the other)

Wrote filtered.vcf
Wrote filtered_summary.txt
Ready for: hycline generate-geographic-input / generate-genomic-input, using this file as --input.
```

Every retained VCF line is copied byte-for-byte from the input, so INFO
annotations and any FORMAT field beyond `GT` survive filtering exactly as
written.

---

## Step 2: Estimate hybrid index

Genomic clines need each individual's **genome-wide hybrid index** (0 =
pure parental population 0, 1 = pure parental population 1) as the x-axis
variable. `estimate-hybrid-index` computes it directly from a VCF or ANGSD
beagle file -- format auto-detected from the `--input` file extension, the
same pattern every other conversion command in HyCline uses -- and a
tab-delimited metadata file labelling the parental reference individuals,
which must be in the same VCF/beagle file. The labels use `filter-aims`'s
flags: `--pop1-label` is the population at hybrid index 0 (parental
population 0 below) and `--pop2-label` the one at hybrid index 1
(parental population 1). Every sample labelled with either must be in the
input file under exactly that name; samples with no row, or another label,
are simply estimated. Genotypes are diploid.

Run it on ancestry-informative markers from Step 1 (see "Which loci to
use" below):

```bash
hycline estimate-hybrid-index \
    --input hi_aims.vcf \
    --metadata parental_labels.tsv \
    --pop1-label parent0 --pop2-label parent1 \
    --output hybrid_index.tsv
```

Real output (on a simulated 90-sample VCF, 15 reference individuals per
parental population; absolute paths shortened):

```
Detected input format: vcf (from file extension; use --format to override)
Parsing 'hi_aims.vcf' ...
  395 biallelic SNPs parsed, 90 samples.
  395 loci detected: ['snp21', 'snp22', 'snp55', 'snp63', 'snp75', 'snp91', 'snp119', 'snp122', 'snp125', 'snp150'] ... and 385 more
  --pop1-label 'parent0' (hybrid index 0): 15 reference individuals
  --pop2-label 'parent1' (hybrid index 1): 15 reference individuals

Hybrid index estimated for 90 individuals (--method ml).
  Range: [0.000, 1.000]  (parental reference individuals should land near 0 or 1 -- a useful sanity check on the result)

Wrote /path/to/hybrid_index.tsv
Merge this into your metadata (e.g. via the sample-name column) to get a hybrid_index column any genomic-cline fitting command can use.
```

The output has one row per individual, parental references included so
you can check they land near 0 or 1:

```
individual	hybrid_index	hybrid_index_lower	hybrid_index_upper	n_loci
sample1_pop1	0.0	0.0	0.00912283000070602	371
sample33_popH	0.18347595310769976	0.13992310611065473	0.23034432947169986	374
```

`hybrid_index_lower`/`hybrid_index_upper` are a 95% likelihood-profile
interval and `n_loci` is the number of loci with a genotype for that
individual. The file works directly as Step 3's `--metadata`.

Parental reference individuals usually come out at exactly 0 or 1 --
under this estimator that is their most likely value. Step 3 leaves
them out of the cline-fitting input by default.

### How the estimate is made

The default, `--method ml`, is a maximum-likelihood hybrid index. In an
individual with hybrid index h, each locus's allele frequency is
`(1 - h) * p0 + h * p1`, where p0 and p1 are the two parental
populations' frequencies; genotypes follow Hardy-Weinberg proportions
given that frequency; and h is the value that makes the individual's
genotypes most likely. For VCF input the genotypes are the `GT` calls;
for beagle input the full genotype likelihoods are used, so low-depth
data isn't pulled toward 0.5. Missing genotypes (`./.`, or beagle's
`0.333333 0.333333 0.333333`) are left out. This needs no allele
polarization -- a VCF's ALT allele is just the non-reference base -- and
it models partially differentiated loci rather than treating every
locus as diagnostic, so a locus with frequencies 0.3 and 0.6 counts for
what it is and nearly undifferentiated loci carry almost no weight.

#### Step by step, with a worked example

The worked example is one individual at three loci, counting the VCF
ALT allele:

| Locus | p0 (ALT freq., parental pop. 0) | p1 (parental pop. 1) | Genotype |
|---|---|---|---|
| A | 0.0 | 1.0 (fixed) | 1 (heterozygote) |
| B | 0.9 | 0.3 (ALT is mostly pop. 0's allele) | 1 (heterozygote) |
| C | 0.5 | 0.6 (barely differentiated) | 2 (homozygous ALT) |

1. **Read genotypes and mark missing ones.** VCF `GT` calls become ALT
   counts (0/1/2) and `./.` is missing. Beagle keeps each individual's
   three genotype likelihoods; a row of three equal values (ANGSD's
   `0.333333` triple for an individual with no reads) is missing.
2. **Get each locus's parental frequencies p0 and p1** -- frequencies of
   the counted allele (VCF ALT, beagle allele2). They come from the
   labelled reference individuals: their sample frequency for VCF input,
   and for beagle input the pooled maximum-likelihood frequency from their
   genotype likelihoods (the estimator `filter-aims` uses), so low depth
   doesn't pull it toward 0.5. They are clipped to [0.001, 0.999] so that an
   allele absent from a finite reference panel can't make a locus
   impossible -- locus A is used as 0.001/0.999. Loci without parental
   frequencies are skipped.
3. **Express genotypes as likelihoods.** A VCF call becomes 1 for the
   called genotype and 0 for the other two; beagle likelihoods are
   normalized to sum to 1. Missing genotypes get 1/3 each, which adds the
   same constant at every h and so has no effect.
4. **Write down the likelihood of h.** In an individual with hybrid
   index h, each allele copy comes from parental population 1 with
   probability h, so the counted allele's frequency at a locus is
   p(h) = (1 - h) p0 + h p1, and under Hardy-Weinberg proportions the
   genotype probabilities are (1-p)^2, 2p(1-p), p^2. Each locus adds
   log[L0 (1-p)^2 + L1 2p(1-p) + L2 p^2] to the individual's
   log-likelihood. For the example:

   | h | p(h) at A, B, C | P(observed genotype) at A, B, C | log-likelihood |
   |---|---|---|---|
   | 0 | 0.001, 0.90, 0.50 | 0.002, 0.18, 0.25 | -9.32 |
   | 0.25 | 0.25, 0.75, 0.525 | 0.375, 0.375, 0.276 | -3.25 |
   | 0.5 | 0.50, 0.60, 0.55 | 0.50, 0.48, 0.30 | -2.62 |
   | **0.576** | 0.576, 0.554, 0.558 | 0.488, 0.494, 0.311 | **-2.59** |
   | 1 | 0.999, 0.30, 0.60 | 0.002, 0.42, 0.36 | -8.11 |

   Locus A, fixed, carries most of the information: a heterozygote is
   nearly impossible at h = 0 or 1. Locus B is handled without any
   flipping -- its ALT frequency simply falls as h rises. Locus C's term
   barely changes with h (0.25 to 0.36), so it barely moves the estimate;
   weakly differentiated loci down-weight themselves.
5. **Find the maximum.** Each locus's term is a quadratic in h, so its
   coefficients are computed once. The log-likelihood is evaluated on a
   grid of 21 values (0, 0.05, ..., 1), because it can have more than one
   peak; then, around the best grid point, bisection on the sign of the
   slope narrows h down to about 1e-10. A maximum at 0 or 1 is reached as
   that boundary, which is why parental reference individuals often come
   out at exactly 0 or 1. Example: **h = 0.576**.
6. **Compute the 95% interval**: every h whose log-likelihood is within
   1.92 of the maximum (the chi-square likelihood-profile rule). Each end
   is found by walking the grid outward to the first point below that
   level and bisecting; if no grid point on one side is below it, the
   interval reaches 0 or 1. Example: **[0.089, 0.964]** -- wide, because
   three loci say little; with hundreds of AIMs intervals are typically a
   few hundredths wide.
7. **Write one row per individual**, reference individuals included:
   `hybrid_index`, `hybrid_index_lower`, `hybrid_index_upper`, and
   `n_loci` (loci with a usable genotype; 3 in the example). Step 3 then
   leaves the reference individuals out of the cline-fitting input by
   default, so their exact 0 or 1 values don't enter the cline fit.

Assumptions: the parental frequencies are treated as exact (with many
weakly differentiated loci their errors accumulate -- see "Which loci to
use"); each individual's alleles are treated as independent draws given
h, so F1s, heterozygous at every fixed AIM, aren't modelled specially;
and h is passed on as a point estimate. This is the hybrid-index
likelihood gghybrid uses (Bailey 2023, section 2.1.1), which places a
Beta prior on h and reports the posterior mode; with a uniform prior
that mode equals this maximum-likelihood estimate.

Measured against the simulator's true hybrid indices (15 + 15 reference
individuals, 40 hybrids, 2000 SNPs, 10% missing data), mean absolute
error of the hybrids' estimates:

| Input | `--method ml` | `--method mean` (weighted) |
|---|---|---|
| beagle, depth 1, 100% AIMs | 0.007 | 0.075 |
| beagle, depth 2, 30% AIMs | 0.016 | 0.093 |
| beagle, depth 5, 30% AIMs | 0.011 | 0.064 |
| true genotypes, 30% AIMs | 0.012 | 0.044 |

and the 95% intervals contained the true value for 90-97% of hybrids.

`--method mean` is the simpler estimator: the mean, across loci, of each
individual's parental-population-1 allele dosage (each locus oriented by
the parental frequencies first), optionally weighted by
`--weight-by-differentiation`. It treats every locus as diagnostic, so
partially differentiated loci and, for beagle input, low depth pull its
estimates toward 0.5. That compression carries into Step 4: on the
simulated example below (where every locus is neutral, true v = 1), a
`mean` hybrid index made the fitted clines come out at a median v of
about 2, against about 1.05 with `ml`. Use `ml` for cline fitting.
`--weight-by-differentiation` applies to `--method mean` only.

Neither method carries the uncertainty in h into the genomic-cline fit
(Step 4 treats h as known, unlike the Bayesian hybrid-index models in
gghybrid and bgchm); the interval columns are there so you can see how
precise each individual's estimate is.

### Which loci to use

**Use ancestry-informative markers, not every SNP.** The parental
frequencies come from a finite reference panel, and the model treats
them as exact. Across thousands of barely differentiated loci those
small estimation errors add up: on a simulated 100,000-SNP beagle file
(30 + 30 reference individuals, depth 4, 5% AIMs) using every locus gave
a mean hybrid error of 0.0074 with 95% intervals containing the truth for
only 57% of hybrids, while the 5,000 AIMs alone gave 0.0039 with 94%
coverage -- and ran in under a second instead of about 20.

Within that, more loci beat stricter loci. On the 90-sample example
above:

| Hybrid-index loci (`filter-aims`) | Loci | Mean hybrid error | Mean interval width |
|---|---|---|---|
| `--minor-allele-freq 0.05` | 8 | 0.078 | 0.341 |
| `--allele-freq-diff 0.5` | 395 | 0.019 | 0.090 |
| `--allele-freq-diff 0.3` | 1000 | 0.014 | 0.071 |

The interval width is the practical guide: if it's wide for most
individuals, loosen the filter to bring in more loci.

### Using one AIM set for hybrid index and another for cline fitting

Hybrid index and the clines themselves don't have to use the same loci.
A common choice is a stricter set for hybrid index and a looser one for
cline fitting, so the hybrid index rests on well-differentiated markers
while more loci get their own cline. Run `filter-aims` twice on the same
input and reference panel, and feed each output to a different step:

```bash
# Two AIM sets from the same genome-wide file
hycline filter-aims --input genomewide.vcf --metadata parental_labels.tsv \
    --pop1-label parent0 --pop2-label parent1 \
    --allele-freq-diff 0.5 --output hi_aims.vcf
hycline filter-aims --input genomewide.vcf --metadata parental_labels.tsv \
    --pop1-label parent0 --pop2-label parent1 \
    --allele-freq-diff 0.3 --output cline_aims.vcf

# Hybrid index from the stricter set
hycline estimate-hybrid-index --input hi_aims.vcf \
    --metadata parental_labels.tsv \
    --pop1-label parent0 --pop2-label parent1 \
    --output hybrid_index.tsv

# Cline input from the looser set, using that hybrid index and the
# parental reference individuals for each locus's parental frequencies
hycline generate-genomic-input --input cline_aims.vcf \
    --metadata hybrid_index.tsv --parental-metadata parental_labels.tsv \
    --pop1-label parent0 --pop2-label parent1 \
    --output genomic_input.csv
```

Real output of the two filters and the conversion (the hybrid-index step
is shown above):

```
  4000 loci checked, 395 retained (allele-frequency difference >= 0.5 between the parental populations)
  4000 loci checked, 1000 retained (allele-frequency difference >= 0.3 between the parental populations)

  Parental frequencies from 15 'parent0' (hybrid index 0) and 15 'parent1' (hybrid index 1) reference individuals.
  30 parental reference individual(s) left out of the output (--include-parentals to keep them).

Wrote /path/to/genomic_input.csv
  60 individuals, 1000 loci
Ready for: hycline fit-genomic --input genomic_input.csv --individual-col individual
       or: hycline fit-hierarchical-genomic --input genomic_input.csv --individual-col individual
```

`estimate-hybrid-index`'s output has the `individual` and `hybrid_index`
columns `generate-genomic-input` looks for by default, so it can be
passed as `--metadata` directly; to combine it with other metadata,
merge on the individual column.

Two things to keep in mind:

- **Don't make the hybrid-index set so strict that it has few loci.** As
  the table under "Which loci to use" shows, 8 near-fixed loci gave a far
  noisier hybrid index (mean error 0.078) than 395 or 1000 moderately
  differentiated ones (0.019, 0.014). Check `filter-aims`'s retained
  count and the interval widths.
- **The stricter set is contained in the looser one,** so those loci are
  used both to estimate hybrid index and as loci with their own clines.
  Each makes up roughly 1/N of every individual's hybrid index, which
  pulls its own cline slightly toward the genome-wide average. With
  hundreds of hybrid-index loci this is negligible; with tens it is not.

---

## Step 3: Generate genomic-cline input

`generate-genomic-input --input <file>` takes a VCF or ANGSD beagle file
directly, plus a tab-delimited metadata file giving each individual's hybrid index
(Step 2's output works as is), and **the parental reference individuals,
so each locus's parental allele frequencies can be computed**:

```bash
hycline generate-genomic-input \
    --input cline_aims.vcf \
    --metadata hybrid_index.tsv \
    --parental-metadata parental_labels.tsv \
    --pop1-label parent0 --pop2-label parent1 \
    --output genomic_input.csv
```

Real output (the simulated example from Step 2; absolute paths
shortened):

```
Detected input format: vcf (from file extension; use --format to override)
Parsing 'cline_aims.vcf' ...
  1000 biallelic SNPs parsed, 90 samples.
  Parental frequencies from 15 'parent0' (hybrid index 0) and 15 'parent1' (hybrid index 1) reference individuals.
  30 parental reference individual(s) left out of the output (--include-parentals to keep them).

Wrote /path/to/genomic_input.csv
  60 individuals, 1000 loci
Ready for: hycline fit-genomic --input genomic_input.csv --individual-col individual
       or: hycline fit-hierarchical-genomic --input genomic_input.csv --individual-col individual
```

Each locus gets its counted allele's frequency (VCF ALT, beagle allele2)
in the two parental populations, written as `<locus>.p0` (the
`--pop1-label` population, hybrid index 0) and `<locus>.p1` (the
`--pop2-label` population, hybrid index 1) columns alongside the
genotype data:

```
   individual  hybrid_index  snp9.n  snp9.freq  snp9.p0  snp9.p1
sample31_popH      0.219337       2        1.0      0.5 0.857143
```

Step 4 uses them to model each locus's allele frequency as
`p0 + (p1 - p0) * phi(h)` -- the form gghybrid uses (Bailey 2023,
section 2.1.1) -- so loci needn't be fixed between the parental
populations, and it doesn't matter which population's allele is the
VCF's ALT.

| Flag | Meaning |
|---|---|
| `--pop1-label`, `--pop2-label` | Labels of the parental reference individuals at hybrid index 0 and 1, as in `filter-aims` and `estimate-hybrid-index`; give both. Frequencies are the references' sample frequencies for VCF input, and pooled maximum-likelihood estimates from their genotype likelihoods for beagle input |
| `--population-col` | Column holding the labels (default `population`) |
| `--parental-metadata` | Tab-delimited file holding `--sample-col` and `--population-col`, if the labels aren't in `--metadata` |
| `--include-parentals` | Keep the reference individuals in the output. By default they are left out, as gghybrid's `ggcline` does: clines are fitted to the test individuals |

Parental frequencies are clipped to [0.001, 0.999], so an individual at
hybrid index exactly 0 or 1 can't make a locus impossible, and loci
whose two parental frequencies are equal (or unknown) are left out, with
a count printed. **Without `--pop1-label`/`--pop2-label`** every locus
is treated as fixed for the counted allele in parental population 1, and
the command prints a note saying so. That is right only for fixed markers
whose ALT allele is parental population 1's: on the simulated data,
every locus whose ALT allele belonged to parental population 0 then
fitted at the lower bound of v.

Individuals present in the VCF/beagle file but absent from `--metadata`
are dropped, not an error. The reverse is an error: every sample the
metadata labels with `--pop1-label` or `--pop2-label` must match a sample
name in the VCF `#CHROM` line or beagle header exactly, and the error
lists the ones that don't. `--hybrid-index-col` (default
`hybrid_index`) names the hybrid-index column.

**The input format determines the output, with no flag to choose
otherwise.** VCF input produces hard counts from `GT` calls; beagle input
produces per-individual genotype likelihoods. For genotype-likelihood-
aware fitting from sequencing data, generate genotype likelihoods with
ANGSD (`angsd -doGlf 2`) and provide that beagle file here. There is no
VCF route to genotype-likelihood output, even via a VCF's own `PL`
field.

---

## Step 4: Fit genomic clines

`fit-genomic` reads any file `generate-genomic-input` produces -- hard
counts or genotype likelihoods, one locus or thousands -- and figures out
which it's looking at from the column names alone: `.n`/`.freq` suffix
pairs mean hard counts; `.gl0`/`.gl1`/`.gl2` triples mean genotype
likelihoods. `.p0`/`.p1` columns, when present, give each locus's
parental frequencies.

```bash
hycline fit-genomic --input genomic_input.csv --individual-col individual \
    --loci snp1602,snp548,snp9 --output-dir results/
```

Real output, for three loci from the Step 3 file -- a fixed locus whose
ALT allele is parental population 1's (`snp1602`, parental frequencies
0/1), a fixed locus whose ALT allele is parental population 0's
(`snp548`, 0.96/0), and a partially differentiated one (`snp9`,
0.5/0.86). Every simulated locus is neutral (true u = 0, v = 1):

```
Reading input data from 'genomic_input.csv' ...
  3 loci detected: ['snp1602', 'snp548', 'snp9']

--- Locus 'snp1602' (54 individuals) ---
  Done in 0.0s. Genomic cline fit: success=True, logL=-35.519, AICc=75.273, n=54, k=2
           u =     0.2667  (SE 0.2708, 95% CI [-0.2641, 0.7976])
           v =     1.1965  (SE 0.2237, 95% CI [0.7580, 1.6350])
  interpretation: steeper than genome-wide average (elevated barrier to introgression)

--- Locus 'snp548' (55 individuals) ---
  Done in 0.0s. Genomic cline fit: success=True, logL=-31.909, AICc=68.048, n=55, k=2
           u =    -0.1402  (SE 0.3164, 95% CI [-0.7603, 0.4798])
           v =     1.5971  (SE 0.3322, 95% CI [0.9459, 2.2482])
  interpretation: steeper than genome-wide average (elevated barrier to introgression)

--- Locus 'snp9' (58 individuals) ---
  Done in 0.0s. Genomic cline fit: success=True, logL=-52.328, AICc=108.875, n=58, k=2
           u =     0.1801  (SE 0.6187, 95% CI [-1.0325, 1.3927])
           v =     0.6882  (SE 0.3911, 95% CI [-0.0784, 1.4548])
  interpretation: shallower than genome-wide average (elevated introgression)

3/3 loci fit successfully.

All output written to: /path/to/results
  hycline_summary.csv        -- one row per locus: u, v, statistics
  hycline_fitted_curves.csv  -- fitted curve points, every locus (long format)
  hycline_overlay.png        -- one fitted-curve line per locus, shared axes
  hycline_run_config.json    -- the exact settings used (for reproducibility)
```

`u` (bias) and `v` (rate, steepness) describe phi(h), the probability
that an allele copy at the locus derives from parental population 1;
`v = 1, u = 0` is the neutral expectation phi(h) = h. The locus's allele
frequency is `p0 + (p1 - p0) * phi(h)`, so u and v mean the same thing
whichever allele is counted and however differentiated the locus is.
(HyCline writes the cline as logit(phi) = u + v logit(h); gghybrid's
u has the opposite sign.) The summary CSV records each locus's
`parental_freq_pop0`/`parental_freq_pop1`; the fitted-curves CSV has
both `fitted_freq` (the allele frequency) and `fitted_ancestry` (phi),
and the overlay plot shows the ancestry curves, which are comparable
across loci with different parental frequencies.

**Single-locus estimates are noisy.** All three loci above are neutral,
yet the "interpretation" lines read their point estimates as steeper or
shallower than average; every 95% CI includes v = 1. Across all 1000
neutral loci in this example, the median fit was u = 0.02, v = 1.07;
1.5% had a 95% interval for u excluding 0 and 6.5% one for v excluding 1
(5% expected for each), and 2 loci fitted at the lower bound of v. From
beagle input at depth 5 the intervals for v were narrower than they
should be: 9% of neutral loci had one excluding v = 1.
Part of that comes from the parental frequencies and hybrid indices
being estimates treated as exact, the same simplification gghybrid
makes. Interpret any single locus with caution unless it is
corroborated by pooling across loci (the hierarchical option below) or
replication.

The summary is always a per-locus table, even for a single-locus input.
The plot is always a single set of axes with one fitted-curve line per
locus (a legend for up to 15 loci, a colorbar beyond that), not a grid of
separate panels, so it stays readable at genome scale.

Key input-data flags:

| Flag | Applies to | Meaning |
|---|---|---|
| `--hybrid-index-col` | both | Column name for hybrid index (default: `hybrid_index`) |
| `--n-alleles-suffix` / `--freq-suffix` | hard counts | Column suffixes for per-individual sample size / observed genotype frequency (defaults: `.n`/`.freq`) |
| `--ploidy` | hard counts | Used when `--n-alleles-suffix` is empty (pass `''` to rely on ploidy instead of per-locus sample-size columns) |
| `--n-focal-suffix` | hard counts | Focal-allele counts instead of frequency |
| `--gl0-suffix` / `--gl1-suffix` / `--gl2-suffix` | genotype likelihoods | Column suffixes for natural-log genotype likelihoods (defaults: `.gl0`/`.gl1`/`.gl2`) |
| `--phred` | genotype likelihoods | Treat the GL columns as Phred-scaled PL values rather than natural-log values |
| `--individual-col` | both | Individual labels, for reporting |
| `--loci` | both | Comma-separated list of locus names to fit (default: all detected) -- also the natural way to fit just one locus out of a larger file |

`fit-genomic` is classic MLE only, for both hard-count and
genotype-likelihood input -- there's no Bayesian backend for independent
per-locus genomic-cline fitting. The one Bayesian genomic-cline path is
hierarchical fitting, below.

---

## Fitting hierarchically

`fit-hierarchical-genomic` fits every locus *jointly*, with `u` and `v`
partially pooled across loci via shared population-level priors --
sparse or noisy loci borrow statistical strength from data-rich ones,
rather than each locus being fit fully independently the way
`fit-genomic` does. Same auto-detection between hard counts and
genotype likelihoods as `fit-genomic`, and the same use of each locus's
`.p0`/`.p1` parental frequencies, which are recorded in the locus
summary. Every locus must have data for the same individuals.

```bash
hycline fit-hierarchical-genomic --input converted.csv --individual-col individual \
    --num-warmup 1000 --num-samples 1000 --num-chains 4 --output-dir results/
```

Real output on a 5-locus dataset with a known `v=5.0` barrier locus:

```
Per-locus posterior means:
     locus    u_mean     u_sd   v_mean     v_sd
  neutral1  0.216261 0.132107 0.900562 0.121704
  neutral2 -0.133320 0.130617 0.850936 0.118374
  barrier1 -0.073101 0.237422 5.046247 0.742758
permeable1 -0.071396 0.117578 0.235269 0.080755
   biased1  2.644593 0.292526 1.200249 0.181038

Convergence diagnostics (4 chains x 1000 draws):
  max R-hat        = 1.0000  (want < 1.01)
  min ESS (bulk)   = 2602  (want > 400)
  min ESS (tail)   = 1994  (want > 400)
  divergences      = 0
  overall: OK
```

`barrier1` correctly recovers `v≈5.05` against a true value of 5.0.
**Hierarchical pooling needs a reasonable number of loci to work well** --
the population-level variance parameters that pooling depends on are
themselves estimated from the spread across loci, so a handful of loci
(fewer than ~5) commonly gives poor convergence (high R-hat, many
divergences) rather than a useful fit. If you see that, either add more
loci or use `fit-genomic` (independent fitting) instead.

Requires every locus to have data for exactly the same individuals -- no
per-locus missing data. If your loci don't share individuals exactly, use
`fit-genomic` instead.

Model-specific flags: `--mu-u-scale`/`--sigma-u-scale` (bias
hyperpriors), `--mu-log-v-loc`/`--mu-log-v-scale`/`--sigma-log-v-scale`
(rate hyperpriors, `--sigma-log-v-scale` most directly answering "how
much does introgression rate vary across the genome"), and
`--pool-center` (pool `logit(u/v)` instead of raw `u` -- the
parameterization `coupling-genomic`'s own theta calibration expects; see
the dedicated coupling-analysis guide for the full detail on when this
matters).

Output mirrors `fit-genomic`'s naming, with `_hierarchical_` inserted:
`{prefix}_hierarchical_population_summary.csv`,
`{prefix}_hierarchical_locus_summary.csv`,
`{prefix}_hierarchical_fitted_curves.csv`, and
`{prefix}_hierarchical_overlay.png`.

---

## A note on where this guide's examples came from

Every command and every block of real output in this guide was generated
by actually running HyCline against synthetic test data while writing it.
