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
    --metadata parental_labels.csv \
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
same pattern every other conversion command in HyCline uses.

Run it on ancestry-informative markers from Step 1 (see "Which loci to
use" below):

```bash
hycline estimate-hybrid-index \
    --input hi_aims.vcf \
    --metadata parental_labels.csv \
    --population-col population --pop0-label parent0 --pop1-label parent1 \
    --method ml --output hybrid_index.csv
```

Real output (on a simulated 90-sample VCF, 15 reference individuals per
parental population; absolute paths shortened):

```
Detected input format: vcf (from file extension; use --format to override)
Parsing 'hi_aims.vcf' ...
  395 biallelic SNPs parsed, 90 samples.
  395 loci detected: ['snp21', 'snp22', 'snp55', 'snp63', 'snp75', 'snp91', 'snp119', 'snp122', 'snp125', 'snp150'] ... and 385 more
  Parental population 0 ('parent0'): 15 reference individuals
  Parental population 1 ('parent1'): 15 reference individuals

Hybrid index estimated for 90 individuals (--method ml).
  Range: [0.000, 1.000]  (parental reference individuals should land near 0 or 1 -- a useful sanity check on the result)

Wrote /path/to/hybrid_index.csv
Merge this into your metadata (e.g. via the sample-name column) to get a hybrid_index column any genomic-cline fitting command can use.
```

The output has one row per individual, parental references included so
you can check they land near 0 or 1:

```
individual,hybrid_index,hybrid_index_lower,hybrid_index_upper,n_loci
sample1_pop1,0.0,0.0,0.00912283000070602,371
sample33_popH,0.18347595310769976,0.13992310611065473,0.23034432947169986,374
```

`hybrid_index_lower`/`hybrid_index_upper` are a 95% likelihood-profile
interval and `n_loci` is the number of loci with a genotype for that
individual. The file works directly as Step 3's `--metadata`.

**Don't use `--method ml` output as Step 3's hybrid index yet.** It
places parental reference individuals at exactly 0 or 1. The genomic-
cline model in Step 4 fixes a locus's allele frequency at 0 when h = 0,
so a parental reference at h = 0 that carries the other population's
allele is treated as impossible, and the fit flattens the cline to
accommodate it: on simulated data where every locus is neutral (true
v = 1), near-fixed AIMs fitted at v = 0.18-0.28 with an `ml` hybrid
index and v = 0.71-1.19 with a `mean` one. Use the default,
`--method mean`, for cline fitting; `ml` is still the more accurate
estimate of the hybrid index itself. Without `--method`, the output has
just the `individual` and `hybrid_index` columns.

### How the estimate is made

`--method ml` is a maximum-likelihood hybrid index. In an
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

`--method mean`, the default, is the simpler estimator: the mean, across loci, of each
individual's parental-population-1 allele dosage (each locus oriented by
the parental frequencies first), optionally weighted by
`--weight-by-differentiation`. It treats every locus as diagnostic, so
partially differentiated loci and, for beagle input, low depth pull its
estimates toward 0.5. `--weight-by-differentiation` applies to
`--method mean` only, and `--method ml` requires diploid data
(`--ploidy 2`).

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

### Two ways to supply parental allele frequencies

**Mode 1** (shown above) -- `--metadata`/`--population-col`/`--pop0-label`/
`--pop1-label`: a separate metadata CSV maps sample names to which
parental population each is a reference for.

**Mode 2** -- `--parental-freq-csv`: a separate CSV of precomputed
per-locus parental frequencies (of the VCF ALT / beagle allele2 allele),
for when you already have a reference panel's frequencies summarized
rather than raw reference genotypes. No `--metadata` needed for this
mode.

```bash
hycline estimate-hybrid-index \
    --input genomewide.vcf \
    --parental-freq-csv parental_frequencies.csv \
    --freq-pop0-col freq_pop0 --freq-pop1-col freq_pop1 \
    --output hybrid_index.csv
```

Exactly one of the two modes must be given.

### Using one AIM set for hybrid index and another for cline fitting

Hybrid index and the clines themselves don't have to use the same loci.
A common choice is a stricter set for hybrid index and a looser one for
cline fitting, so the hybrid index rests on well-differentiated markers
while more loci get their own cline. Run `filter-aims` twice on the same
input and reference panel, and feed each output to a different step:

```bash
# Two AIM sets from the same genome-wide file
hycline filter-aims --input genomewide.vcf --metadata parental_labels.csv \
    --pop1-label parent0 --pop2-label parent1 \
    --allele-freq-diff 0.5 --output hi_aims.vcf
hycline filter-aims --input genomewide.vcf --metadata parental_labels.csv \
    --pop1-label parent0 --pop2-label parent1 \
    --allele-freq-diff 0.3 --output cline_aims.vcf

# Hybrid index from the stricter set
hycline estimate-hybrid-index --input hi_aims.vcf \
    --metadata parental_labels.csv \
    --population-col population --pop0-label parent0 --pop1-label parent1 \
    --output hybrid_index.csv

# Cline input from the looser set, using that hybrid index
hycline generate-genomic-input --input cline_aims.vcf \
    --metadata hybrid_index.csv --output genomic_input.csv
```

Real output of the two filters and the conversion (the hybrid-index step
is shown above):

```
  4000 loci checked, 395 retained (allele-frequency difference >= 0.5 between the parental populations)
  4000 loci checked, 1000 retained (allele-frequency difference >= 0.3 between the parental populations)

Wrote /path/to/genomic_input.csv
  90 individuals, 1000 loci
Ready for: hycline fit-genomic --input genomic_input.csv --individual-col individual
       or: hycline fit-hierarchical-genomic --input genomic_input.csv --individual-col individual
```

`estimate-hybrid-index`'s output has the `individual` and `hybrid_index`
columns `generate-genomic-input` looks for by default, so it can be
passed as `--metadata` directly; to combine it with other metadata,
merge on the individual column.

Three things to keep in mind:

- **Only near-fixed AIMs give interpretable clines with the current
  model.** Step 4's genomic cline describes the probability of carrying
  parental population 1's allele as rising from 0 at h = 0 to 1 at h = 1.
  A locus whose parental frequencies are, say, 0.57 and 0.87 can't do
  that, so even a neutral one fits as a strongly shallow, shifted cline:
  on simulated data where every locus is neutral (true u = 0, v = 1),
  loci with parental frequencies 0.57/0.87 and 0.63/0.93 fitted at
  v = 0.38 and 0.56 with u = 1.1 and 1.5, which reads as elevated
  introgression. Until the cline model includes the parental frequencies,
  restrict the cline-fitting set to near-fixed AIMs (for example
  `--minor-allele-freq 0.05`) rather than the looser
  `--allele-freq-diff 0.3` set shown above.
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

One command, one flag: `generate-genomic-input --input <file>` takes a
VCF or ANGSD beagle file directly, plus a metadata CSV giving each
individual's hybrid index (typically Step 2's own output, merged into
your broader metadata by individual name).

**The input format determines the output path, with no flag to choose
otherwise.** VCF input always produces hard-count output from `GT`
calls. Beagle input always produces per-individual genotype-likelihood
output.

```bash
hycline generate-genomic-input \
    --input filtered.vcf \
    --metadata test_individuals_with_hi.csv \
    --output converted.csv
```

Real output:

```
Detected input format: vcf (from file extension; use --format to override)
Parsing 'filtered.vcf' ...
  3 biallelic SNPs parsed, 72 samples.
  NOTE: 12 VCF sample(s) had no matching row in the metadata and were dropped.

Wrote converted.csv
  60 individuals, 3 loci
Ready for: hycline fit-genomic --input converted.csv --individual-col individual
       or: hycline fit-hierarchical-genomic --input converted.csv --individual-col individual
```

Individuals present in the VCF/beagle file but absent from `--metadata`
are simply dropped, not an error -- the note above shows exactly this
happening with the 12 parental reference individuals, which have no
`hybrid_index` entry in this particular metadata file. `--hybrid-index-col`
(default `hybrid_index`) names the column holding it.

**Want genotype-likelihood-aware fitting from sequencing data?** Generate
genotype likelihoods with ANGSD (`angsd -doGlf 2`) and provide that
beagle file here. There is no VCF route to genotype-likelihood output,
even via a VCF's own `PL` field.

---

## Step 4: Fit genomic clines

`fit-genomic` reads any file `generate-genomic-input` produces -- hard
counts or genotype likelihoods, one locus or thousands -- and figures out
which it's looking at from the column names alone: `.n`/`.freq` suffix
pairs mean hard counts; `.gl0`/`.gl1`/`.gl2` triples mean genotype
likelihoods.

**Check allele polarity before fitting.** The model describes the
frequency of parental population 1's allele and requires v > 0, but
`generate-genomic-input` writes the VCF ALT (or beagle allele2)
frequency as is. At a locus where ALT is parental population 0's
allele, the true cline decreases with hybrid index, and the fit ends at
the lower bound v = 0.001 (every such locus did, on simulated data). Fit
only loci whose ALT allele is more common in parental population 1, or
recode the others before fitting.

```bash
hycline fit-genomic --input converted.csv --output-dir results/
```

Real output:

```
  3 loci detected: ['aim_neutral', 'aim_barrier', 'aim_permeable']

--- Locus 'aim_neutral' (60 individuals) ---
  Done in 0.0s. Genomic cline fit: success=True, logL=-34.569, AICc=73.349, n=60, k=2
           u =     0.4896  (SE 0.2829, 95% CI [-0.0648, 1.0440])
           v =     2.1895  (SE 0.3857, 95% CI [1.4335, 2.9456])
  interpretation: steeper than genome-wide average (elevated barrier to introgression)

--- Locus 'aim_barrier' (60 individuals) ---
  Done in 0.0s. Genomic cline fit: success=True, logL=-31.505, AICc=67.221, n=60, k=2
           u =     0.0702  (SE 0.3054, 95% CI [-0.5283, 0.6687])
           v =     3.0283  (SE 0.5347, 95% CI [1.9804, 4.0762])
  interpretation: steeper than genome-wide average (elevated barrier to introgression)

--- Locus 'aim_permeable' (60 individuals) ---
  Done in 0.0s. Genomic cline fit: success=True, logL=-52.598, AICc=109.407, n=60, k=2
           u =    -0.2810  (SE 0.2037, 95% CI [-0.6802, 0.1181])
           v =     0.6239  (SE 0.2016, 95% CI [0.2288, 1.0191])
  interpretation: shallower than genome-wide average (elevated introgression)

3/3 loci fit successfully.

All output written to: results/
  hycline_summary.csv        -- one row per locus: u, v, statistics
  hycline_fitted_curves.csv  -- fitted curve points, every locus (long format)
  hycline_overlay.png        -- one fitted-curve line per locus, shared axes
  hycline_run_config.json    -- the exact settings used (for reproducibility)
```

`u` (bias, shifts the cline's center along the hybrid-index axis) and `v`
(rate, steepness -- `v=1, u=0` is the neutral expectation) are the two
fitted parameters. This particular `aim_neutral` result is a useful,
honest illustration of finite-sample variability worth keeping in mind:
the true simulated locus was exactly neutral (`u=0, v=1`), and at
`n`=60 individuals it still came back with a 95% CI that excludes `v=1`
-- genomic cline parameter estimates are noisier than they might look,
and any single locus's fit should be interpreted with real caution unless
corroborated by pooling across many loci (see Step 4's hierarchical
option below) or replication.

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
genotype likelihoods as `fit-genomic`.

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
