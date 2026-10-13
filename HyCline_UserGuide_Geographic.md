# HyCline User Guide: HZAR-Parity Geographic Cline Analysis

**Document status: working draft, updated alongside HyCline's own
development.**

This guide walks through the full geographic-cline pipeline in HyCline, from
raw genome-wide genotype data to a fitted, HZAR-comparable cline, covering
every major fork in the road: hard genotype calls vs. genotype likelihoods,
classic MLE vs. Bayesian inference, and HyCline's own tail model vs. HZAR's
verified tail parameterization.

Every command shown below was run for real against synthetic test data while
writing this guide, not copied from memory -- flags, output messages, and
file names are exact.

## Pipeline overview

```
genome-wide genotype data (VCF or ANGSD beagle)
        |
        v
[1] filter-aims  ------------------------------  (skip if you already have
        |                                          a curated AIM panel)
        v
[2] generate-geographic-input      --> VCF: hard counts, always
        |                              --> beagle: genotype likelihoods, always
        v
[3] fit clines
        |
        +-- fit-geographic               (independent per-locus fitting)
        +-- fit-hierarchical-geographic   (partial pooling across loci)
        |
        +-- --method classic   (fast point estimate, HZAR-comparable numbers)
        +-- --method bayesian  (full posterior, proper credible intervals)
        |
        +-- --tail none/mirror/both, optionally --hzar-tail for exact
        |   HZAR tail parity
        |
        v
[4] model selection & diagnostics
        |
        +-- compare-tails            (AICc across tail models)
        +-- convergence warnings     (Bayesian fits only)
```

**Only two commands handle fitting, regardless of locus count or data type.**
`fit-geographic` and `fit-hierarchical-geographic` auto-detect everything
about the input -- hard counts vs. genotype likelihoods, one locus vs.
thousands -- from the file itself, with no flags needed to tell them what
they're looking at.

---

## Step 1: Filter for ancestry-informative markers (AIMs)

Most genome-wide SNPs aren't informative for cline analysis -- you want loci
where the two parental populations are (nearly) fixed for different alleles.
`filter-aims` does this filtering directly on a VCF or an ANGSD beagle file,
**writing output in the same format it was given** -- a filtered VCF from VCF
input, a filtered beagle file from beagle input. Format is auto-detected
from the `--input` file extension.

```bash
hycline filter-aims \
    --input genomewide.vcf \
    --metadata parental_labels.csv \
    --population-col population \
    --pop1-label parent1 --pop2-label parent2 \
    --output filtered.vcf
```

Real output (here on a simulated 60-sample, 5000-SNP VCF with 10 individuals
per parental population; absolute paths shortened):

```
Detected input format: vcf (from file extension; use --format to override)
Reading vcf data from 'genomewide.vcf' ...
  parent1: 10 individuals; parent2: 10 individuals
  5000 loci checked, 250 retained (>= 0.9 in one population and <= 0.1 in the other, either direction)
WARNING: Parental individual 'sample4_pop1' is missing data at 60.00% of loci checked (more than 50%).
WARNING: 211 of 250 retained AIMs had fewer than 10 called individuals in at least one parental population at that locus, because of missing data.

Wrote /path/to/filtered.vcf
Wrote /path/to/filtered_summary.txt
Ready for: hycline generate-geographic-input / generate-genomic-input, using this file as --input -- or use it with any other tool that reads VCF directly.
```

### The summary file

Every run also writes a plain-text summary next to the output (by default
`--output` with its `.vcf`/`.vcf.gz`/`.beagle`/`.beagle.gz` extension replaced
by `_summary.txt`; set the path with `--summary`). It records how many AIMs
were retained, the global missing-data rate, a per-sample missing-data table
for the parental individuals (over all loci checked and over the retained
AIMs alone), and any warnings. From the run above (per-sample table
truncated):

```
AIMs
----
Loci checked:   5000
AIMs retained:  250 (5.00% of loci checked)
Sites skipped:  0 multiallelic, 0 indel, 0 with no GT field

Missing data (over loci checked)
--------------------------------
Global, all samples (60):             5.92%
Global, parental samples (20):        7.74%
Parental samples, retained AIMs only: 7.46%

Per-sample missing data (parental samples)
------------------------------------------
sample	population	missing_all_loci	missing_aims	flag
sample1_pop1	parent1	4.96%	4.00%	
sample4_pop1	parent1	60.00%	56.00%	HIGH_MISSING
...

Warnings
--------
WARNING: Parental individual 'sample4_pop1' is missing data at 60.00% of loci checked (more than 50%).
WARNING: 211 of 250 retained AIMs had fewer than 10 called individuals in at least one parental population at that locus, because of missing data.
```

The per-sample table is tab-separated, so it pastes straight into a
spreadsheet. A genotype counts as missing when it is `./.` in a VCF, or
when its three genotype likelihoods are equal or absent in a beagle file
(ANGSD writes `0.333333 0.333333 0.333333` for an individual with no reads
at a site). Missing beagle genotypes are also left out of the parental
allele-frequency calculation itself, the same way `./.` calls are for VCF
input -- an equal-likelihood row says nothing about the individual's
alleles. For beagle input, each parental population's allele frequency
at a locus is the maximum-likelihood estimate from all of its individuals'
genotype likelihoods together (the same EM model as ANGSD's `-doMaf`), so
low-depth data is not pulled toward 0.5: on simulated depth-2 data this
recovers 207 of 300 true AIMs at a 0.05/0.95 cutoff, with no false ones.
Warnings are printed to the terminal as well as written to the
summary, for three situations:

- a parental population with fewer than `--min-parental-n` individuals
  (default 10). The cost of a small panel is false AIMs: a locus with true
  frequencies 0.2/0.8 passes the default 0.1/0.9 filter by sampling chance
  about 14% of the time with 5 individuals per population, 4% with 10, and
  0.6% with 20;
- a parental individual missing data at more than `--max-sample-missing`
  of loci checked (default 0.5);
- retained AIMs where, because of missing data, fewer than
  `--min-parental-n` individuals of a parental population actually had a
  genotype call at that locus -- the AIM's frequency rests on fewer
  individuals than the panel size suggests. (Populations already below
  `--min-parental-n` overall are covered by the first warning instead.)
  With a panel of exactly `--min-parental-n` individuals, a single missing
  call at an AIM counts, as in the example above; lower `--min-parental-n`
  if that is more sensitive than you need.

Warnings never stop the run or change which loci are retained.

For VCF specifically: **every retained line is copied byte-for-byte from the
input, not re-derived from parsed fields** -- filtering only removes whole
lines (variants); it never rewrites one. Verified directly while building
this: a test VCF carrying an extra `DP` (read-depth) FORMAT field alongside
`GT` came through filtering with that field completely untouched. This means
INFO annotations, QUAL/FILTER values, and any FORMAT field beyond `GT` you
care about downstream survive filtering exactly as your variant caller wrote
them.

### Key flags

| Flag | Meaning |
|---|---|
| `--input` | VCF (`.vcf`/`.vcf.gz`) or ANGSD beagle-format file (`.beagle`/`.beagle.gz`) |
| `--format {vcf,beagle}` | Force the format instead of auto-detecting from the extension |
| `--metadata` | **Separate** CSV: one row per individual, a sample-name column and a population-label column. Individuals not labeled as either parental population (e.g. your actual hybrid-zone samples) are simply excluded from the frequency calculation -- not an error -- so parental references and hybrid-zone individuals can live in the same `--input` file |
| `--sample-col` | Column in `--metadata` holding individual names (default: `individual`). Must match the VCF's own `#CHROM` header row, or ANGSD's own beagle header row |
| `--population-col` | Column in `--metadata` holding each individual's population label (default: `population`) |
| `--pop1-label` / `--pop2-label` | Values identifying the two parental reference groups (default: `parent1`/`parent2`) |
| `--low-threshold` / `--high-threshold` | A locus is retained if one population's frequency is at or below `--low-threshold` (default 0.1) and the other's is at or above `--high-threshold` (default 0.9), in either direction |
| `--summary` | Path for the summary file (default: `--output` with its extension replaced by `_summary.txt`) |
| `--min-parental-n` | Warn when a parental population, or a retained AIM's called individuals in a parental population, number fewer than this (default 10) |
| `--max-sample-missing` | Warn about parental individuals missing data at more than this fraction of loci checked (default 0.5) |

For VCF input, sites are also silently checked against a biallelic-SNP-with-
GT scope (multiallelic sites, indels, and sites with no `GT` field are
skipped, not guessed at). When any are skipped, you'll see an extra line:

```
  (3 additional sites skipped: 1 multiallelic, 1 indel, 1 with no GT field -- not counted toward loci checked above)
```

If you already have a curated, pre-known diagnostic-marker panel, skip this
step altogether and go straight to Step 2.

---

## Step 2: Generate HyCline input

One command, one flag: `generate-geographic-input --input <file>` takes
your (optionally filtered) VCF or beagle file directly. Format is
auto-detected from the `--input` file extension (`--format vcf`/`beagle`
to override).

**The input format determines the output path, with no flag to choose
otherwise.** VCF input always produces site-aggregated hard-count output
from `GT` calls. Beagle input always produces per-individual genotype-
likelihood output. A VCF's `GT` field is a hard call regardless of which
variant caller produced it; ANGSD's beagle format exists specifically to
carry genotype uncertainty.

```bash
hycline generate-geographic-input \
    --input filtered.vcf \
    --metadata metadata.csv \
    --output hardcounts.csv
```

Real output:

```
Detected input format: vcf (from file extension; use --format to override)
Parsing 'filtered.vcf' ...
  3 biallelic SNPs parsed, 20 samples.

Wrote hardcounts.csv
  10 sites, 3 loci
Ready for: hycline fit-geographic --input hardcounts.csv
       or: hycline fit-hierarchical-geographic --input hardcounts.csv --site-col site
```

The exact same command, no flags added, works for beagle input -- but notice
the output shape and the suggested next step both change automatically:

```bash
hycline generate-geographic-input \
    --input mydata.beagle.gz \
    --metadata metadata.csv \
    --output gl.csv
```

```
Detected input format: beagle (from file extension; use --format to override)
Parsing 'mydata.beagle.gz' ...
  1 markers parsed, 2 samples.

Wrote gl.csv
  2 individuals, 1 loci
Ready for: hycline fit-geographic --input gl.csv --individual-col individual
       or: hycline fit-hierarchical-geographic --input gl.csv --individual-col individual
```

**Want genotype-likelihood-aware fitting from sequencing data?** Generate
genotype likelihoods with ANGSD (`angsd -doGlf 2`) and provide that beagle
file here. There is no VCF route to genotype-likelihood output, even via a
VCF's own `PL` field.

### Distance: pre-computed, or from coordinates

`--distance-col` (a distance you've already computed) or `--lat-col`/
`--lon-col` (raw GPS coordinates, converted to a 1D transect distance in
km automatically) -- the same two options `fit-geographic` itself offers.

```bash
hycline generate-geographic-input \
    --input filtered.vcf \
    --metadata metadata.csv \
    --lat-col lat --lon-col lon \
    --output hardcounts.csv
```

`--transect-method {reference,pca}` controls how 2D coordinates collapse to
1D distance -- `reference` (default) is great-circle distance from a single
reference site (matching HZAR's own convention; override with
`--reference-lat`/`--reference-lon`), `pca` projects onto the best-fit line
through all sites, better for zigzagging sampling with no natural
reference endpoint. If both `--distance-col` and `--lat-col`/`--lon-col`
are given, coordinates take priority.

**For VCF input, every individual assigned to the same site must share the
same coordinate** -- checked directly, before any distance is computed,
giving a clear error naming the site and both coordinate pairs (not
meaningful for beagle input, which never aggregates by site):

```
ERROR: Site 'S0' has inconsistent coordinates across its individuals in
the metadata ((40.0, -74.0) vs (999.0, -74.0)) -- every individual at the
same site must share the same latitude/longitude.
```

This is the coordinate-level analog of the distance-consistency check
(also in place for `--distance-col` input on VCF):

```
ERROR: Site 'S0' has inconsistent distance values across its individuals in
the metadata (0.0 vs 5.26...) -- every individual at the same site must
share the same distance value.
```

### `--site-col` and `--ploidy` are VCF-only

Both flags matter only for VCF's site-aggregation step; both are simply
ignored (not an error) if given with beagle input, since beagle output
always stays one row per individual with no aggregation at all.

---

## Step 3: Fit clines

`fit-geographic` reads any file `generate-geographic-input` produces --
hard counts or genotype likelihoods, one locus or thousands -- and figures
out which it's looking at from the column names alone: `.nSamples`/`.A`
suffix pairs mean hard counts; `.gl0`/`.gl1`/`.gl2` triples mean genotype
likelihoods.

```bash
hycline fit-geographic --input hardcounts.csv --output-dir results/
```

Real output (abbreviated, three loci):

```
  3 loci detected: ['snp1', 'snp2', 'snp3']

--- Locus 'snp1' (10 sites) ---
  Done in 0.0s. Cline fit (tail=none): success=True, logL=-6.108, AICc=28.216, n=10, k=4
      center =    27.9967
       width =    38.9930
        pmin =     0.0000
        pmax =     1.0000

--- Locus 'snp3' (10 sites) ---
  Done in 0.0s. Cline fit (tail=none): success=True, logL=-5.463, AICc=26.926, n=10, k=4
      center =    41.6150
       width =    36.7408
        pmin =     0.0000
        pmax =     1.0000

3/3 loci fit successfully.

All output written to: results/
  hycline_summary.csv        -- one row per locus: parameters + statistics
  hycline_fitted_curves.csv  -- fitted curve points for every locus (long format)
  hycline_observed_data.csv  -- your input data, parsed and echoed back (long format)
  hycline_overlay.png        -- one fitted-curve line per locus, shared axes
  hycline_run_config.json    -- the exact settings used (for reproducibility)
```

The summary is always a per-locus table, even for a single-locus input --
one row either way. The plot is always a single set of axes with one
fitted-curve line per locus (a legend for up to 15 loci, a colorbar beyond
that), not a grid of separate panels, so it stays readable at genome scale.

Key input-data flags:

| Flag | Applies to | Meaning |
|---|---|---|
| `--distance-col` | both | Pre-computed geographic distance column (default: `distance`) |
| `--lat-col` / `--lon-col` | both | Compute distance from coordinates instead -- see `--transect-method` |
| `--n-alleles-suffix` / `--freq-suffix` | hard counts | Column suffixes for total sampled alleles / observed frequency per locus (defaults: `.nSamples`/`.A`, matching HZAR's own convention) |
| `--n-ind-suffix` | hard counts | Individual (not allele) counts instead, combined with `--ploidy` |
| `--n-focal-suffix` | hard counts | Focal-allele counts instead of frequency |
| `--site-col` | hard counts | Site labels, for reporting |
| `--gl0-suffix` / `--gl1-suffix` / `--gl2-suffix` | genotype likelihoods | Column suffixes for natural-log genotype likelihoods (defaults: `.gl0`/`.gl1`/`.gl2`) |
| `--phred` | genotype likelihoods | Treat the GL columns as Phred-scaled PL values (GATK/bcftools/ANGSD convention) rather than natural-log values |
| `--individual-col` | genotype likelihoods | Individual labels, for reporting |
| `--loci` | both | Comma-separated list of locus names to fit (default: all detected) -- also the natural way to fit just one locus out of a larger file |

Genotype-likelihood input is classic MLE only, with independent per-locus
fitting; `--method bayesian`, `--scaling fixed`/`--pmin`/`--pmax`, and
`--hzar-tail` all give a clear, actionable error if used with
genotype-likelihood input rather than being silently ignored.

### Why fitting isn't just "estimate a frequency, then fit"

For genotype-likelihood input, HyCline doesn't collapse each individual's
genotype likelihoods to a point estimate before fitting. At every step of
the fit, for each individual, the cline's own predicted allele frequency
at that individual's location is converted to Hardy-Weinberg genotype
probabilities, and the individual's marginal likelihood is computed by
weighting their raw genotype likelihoods by those probabilities:

L_i = GL_0 * (1-p)^2 + GL_1 * 2p(1-p) + GL_2 * p^2

This matters beyond "it's more careful": a simpler alternative -- averaging
each individual's genotype likelihoods into a flat-prior expected dosage
first, then fitting on that as if it were ordinary count data -- has a
real, measured bias at low sequencing depth, worst exactly at
allele-frequency extremes, which is where a hybrid zone's parental
reference individuals usually sit. A direct simulation check (true
frequency 0.02, depth 1 read) showed the flat-prior collapse-first
estimate coming out at 0.19 -- off by nearly an order of magnitude -- while
a simple hard-call point estimate at the same depth was accurate to within
0.01. Proper marginalized fitting avoids this because its implicit "prior"
at each step is the cline's own prediction, not an uninformative flat one.
95% confidence interval coverage bears this out too: naive hard-calling
covered the true value only 67% of the time at 1x depth (vs. 95% nominal),
while marginalized fitting stayed close to nominal throughout, with both
converging by ~15x depth where there's little genotype uncertainty left to
matter.

---

## Choosing MLE or Bayesian inference

`--method {classic,bayesian}` (default: `classic`), for hard-count input.

**Classic** -- a direct maximum-likelihood optimizer. Fast, deterministic,
and gives numbers directly comparable to published HZAR-based studies. No
convergence diagnostics to check. This is the default and the recommended
starting point, especially for tail models (see below).

**Bayesian** -- full posterior via NUTS MCMC, giving proper credible
intervals rather than point estimates plus asymptotic standard errors.

```bash
hycline fit-geographic --input hardcounts.csv \
    --method bayesian --num-warmup 1000 --num-samples 1000 \
    --output-dir results/
```

Bayesian-specific flags:

| Flag | Meaning |
|---|---|
| `--backend {numpyro,pymc}` | Default `numpyro` -- faster, and supports the plate-vectorized multi-locus batch path (see below). `pymc` doesn't support `--scaling fixed` |
| `--num-warmup` / `--num-samples` | Per-chain tuning/post-warmup iterations (default 1000/1000) |
| `--num-chains` | Number of chains (default 4) -- more chains give a more reliable R-hat check |
| `--width-upper-bound` | Upper bound of the flat prior on cline width. Default: 10x the transect's geographic span, which can be far looser than sparse data can constrain |
| `--credible-prob` | Width of the credible interval to report/plot (default 0.95) |
| `--seed` | Reproducibility |

A real convergence warning, and how to read it:

```
  WARNING: Low effective sample size (bulk=38, tail=18). Often the `width`
  parameter specifically when sampling near the transition zone is sparse --
  consider more samples or a tighter/better-informed width prior.
  WARNING: 1 divergent transitions. A handful on this model usually
  indicates the width prior's boundary is being pushed against; many
  divergences suggest a real geometry problem worth investigating.

NOTE: convergence diagnostics flagged possible issues (see above). Try
--width-upper-bound with a tighter value, or more --num-samples.
```

In practice: **tighten `--width-upper-bound`** to something close to the
transect's actual span first -- this is the single most common fix for
these warnings, since the default prior is intentionally loose. If
warnings persist, increase `--num-samples`/`--num-warmup` before assuming
there's a real problem.

`fit-geographic --method bayesian` automatically uses a faster,
plate-vectorized batch path (one NUTS run across every locus at once)
whenever your data is dense -- every locus shares the same sites and
`--scaling` isn't `fixed`. It falls back automatically to one independent
run per locus for ragged data. You don't need to do anything to opt into
this; it's automatic.

The Bayesian summary CSV reports `center_mean`/`width_mean` as point
estimates, plus `{param}_credible_lower`/`{param}_credible_upper` for
every fitted parameter; the fitted-curve CSV carries a median line and a
credible band (`fitted_freq_median`/`_lower`/`_upper`) rather than a
single line.

---

## Tail models, and exact HZAR parity

By default, HyCline fits the plain sigmoid core (`--tail none`). Real cline
data sometimes needs exponential tails outside the sigmoid -- HZAR's classic
model-selection workflow.

```bash
hycline compare-tails --input hardcounts.csv \
    --n-alleles-col locus0.nSamples --freq-col locus0.A \
    --output-dir results/
```

```
 model  k      logL       AICc  delta_AICc  converged
  none  4 -7.642439  31.284878    0.000000       True
mirror  6 -7.229722  54.459444   23.174566       True
  both  8 -6.697047 173.394094  142.109215       True

Best model by AICc: none
```

`--candidates` controls which tail models to compare (default
`none,mirror,both`). `compare-tails` is classic MLE only.

Once you know which tail model you want, pass `--tail {none,mirror,both}` to
`fit-geographic`. Tail-model fitting works with both `--method classic` and
`--method bayesian` on hard-count input, but Bayesian tail-model fitting
has a real, documented convergence difficulty (a non-smooth log-density
where the sigmoid core meets the exponential tail) -- classic MLE is
recommended whenever you need a tail model. Genotype-likelihood input uses
several randomized restarts internally to guard against the same
non-smooth surface.

### `--hzar-tail`: exact parity with HZAR's own tail parameterization

HyCline's *default* tail model is a differently-parameterized (but
mathematically valid) exponential-tail extension, not a byte-for-byte match
to HZAR's own construction. For exact HZAR parity -- verified directly against
HZAR's R source -- add `--hzar-tail`:

```bash
hycline fit-geographic --input hardcounts.csv \
    --tail mirror --hzar-tail --output-dir results/
```

This works with both `--method classic` and `--method bayesian` on
hard-count input, but **not** combined with `--scaling fixed`/`--pmin`/
`--pmax`, and not with genotype-likelihood input at all -- both give a
clear error rather than silently ignoring the flag.

A real, named difference to be aware of: with `--hzar-tail`, the tail edge is
`center +/- delta` directly (a raw distance), not `center +/- width/2` the way
the default tail model defines it -- `delta_l`/`tau_l`/`delta_r`/`tau_r` mean
something different with this flag than without it.

### `--tail t-cline`: a continuous alternative that avoids the tripartite model's real convergence difficulty

`none`/`mirror`/`both` (with or without `--hzar-tail`) all build the tail
model the same underlying way: a smooth sigmoid core joined to exponential
tails at fixed points. That join is mathematically a genuine corner (Baird &
Daley 2025) -- not a numerical inconvenience, a structural discontinuity --
and it's the real, specific cause of the Bayesian convergence difficulty
mentioned above (gradient-based samplers can struggle badly with a kinked
log-density; more warmup usually doesn't fix it).

`--tail t-cline` sidesteps the problem at its source instead of working
around it: it's a *different* model family entirely, built from the
Student's-t distribution, continuous everywhere with no corners at all.

```bash
hycline fit-geographic --input hardcounts.csv \
    --tail t-cline --method classic --output-dir results/
```

```
Cline fit (tail=t-cline): success=True, logL=-39.484, AICc=91.468, n=30, k=5
      center =    51.2519  (SE 0.7094, ...)
       width =    17.1589  (SE 3.1966, ...)
        pmin =     0.0211  (SE 0.0114, ...)
        pmax =     0.9921  (SE 0.0086, ...)
     t_shape =     3.7396  (SE 3.7867, ...)
```

`t_shape` is the one new parameter -- it slides the cline's steepness
continuously from very "stepped" (thick-tailed, `t_shape` near the lower
bound of 0.05, close to a Cauchy distribution shape) through the ordinary
logistic/sigmoid shape (`t_shape` around 6.67) to effectively Normal-shaped
(`t_shape` above roughly 30). AICc against `--tail none` tells you whether
the extra steepness parameter is actually earning its keep on your data --
`compare-tails --candidates none,mirror,both,t-cline` includes it alongside
the tripartite models directly.

Add `--t-cline-asymmetric` to let the two sides of the cline steepen or
decay at different rates (an extra `t_skew` parameter), the T-cline's
analog of the tripartite model's separate left/right `delta`/`tau`:

```bash
hycline fit-geographic --input hardcounts.csv \
    --tail t-cline --t-cline-asymmetric --method classic --output-dir results/
```

Two things worth knowing before relying on this:
- **It's slower** -- an asymmetric fit runs several restart phases
  internally to avoid a real risk of converging to a badly wrong local
  optimum (confirmed directly during development, not a theoretical
  worry), so expect roughly an order of magnitude longer than a symmetric
  T-cline fit on the same data.
- **`t_skew` itself can come back only loosely constrained by your data**,
  the same way the tripartite model's own `delta`/`tau` tail parameters
  are already known to be weakly identified. If the fitted `t_skew` sits
  right at its search bound, HyCline prints a `UserWarning` explaining
  this -- `center`/`width`/`pmin`/`pmax` and the AICc comparison against
  the symmetric-only fit remain meaningful even then; treat the `t_skew`
  point estimate itself with real caution.

**Classic MLE only, for both the symmetric and asymmetric case** --
`--method bayesian` gives an immediate, clear error rather than attempting
this and failing deep inside NumPyro. This isn't a temporary gap expected to
close soon: checked directly, `jax.scipy.special.betainc` (which both a
hand-built JAX Student's-t CDF and NumPyro's own official `StudentT.cdf()`
are built on) does not support gradients with respect to its own shape
parameters -- confirmed against NumPyro's own code, not just a homemade
implementation. Since NUTS needs the gradient of the log-density with
respect to every parameter it samples, including `t_shape`, this is a real
capability gap in the JAX/NumPyro ecosystem itself as currently available,
not something more porting would close. See README.md's Tier 1 backlog for
the full investigation.

---

## Fixed vs. estimated scaling (`pmin`/`pmax`)

By default (`--scaling estimated`), `pmin`/`pmax` (the allele frequencies at
the two ends of the cline) are fit as free parameters. HZAR's
`scaling='fixed'` mode instead holds them at the observed frequencies at the
transect's geographic extremes:

```bash
hycline fit-geographic --input hardcounts.csv --scaling fixed --output-dir results/
```

`--pmin`/`--pmax` override the derived values directly (and imply
`--scaling fixed`). `--n-edge-sites` (default 1, matching HZAR exactly)
controls how many sites near each end are pooled when deriving pmin/pmax --
a single extreme site's observed frequency can be fragile with realistic
sample sizes (an unlucky all-zero draw locks the fixed asymptote at the
wrong value); increasing this to 3 or so is a reasonable fix if you suspect
this is happening. Not supported for genotype-likelihood input, which
always estimates pmin/pmax freely.

---

## Hierarchical (partial-pooling) fitting

`fit-hierarchical-geographic` fits every locus *jointly*, with center and
width partially pooled across loci via shared population-level priors --
sparse or noisy loci borrow statistical strength from data-rich ones,
rather than each locus being fit fully independently the way
`fit-geographic` does. Same auto-detection between hard counts and
genotype likelihoods as `fit-geographic`.

```bash
hycline fit-hierarchical-geographic --input hardcounts.csv --site-col site \
    --output-dir results/
```

Requires every locus to have data at exactly the same sites (hard counts)
or individuals (genotype likelihoods) -- no per-locus missing data. If your
loci don't share sites exactly, use `fit-geographic` instead (it handles
per-locus missing data by dropping only the affected rows for that locus).

Model-specific flags worth knowing: `--n-edge-sites` (hard counts,
default 3) / `--n-edge-individuals` (genotype likelihoods, default 3),
`--sigma-center-scale`, `--mu-log-width-loc`/`--mu-log-width-scale`,
`--sigma-log-width-scale` (hyperprior tuning, defaults are usually fine to
start with). Bayesian only -- there is no classic-MLE hierarchical path.

Output mirrors `fit-geographic`'s naming, with `_hierarchical_` inserted:
`{prefix}_hierarchical_population_summary.csv` (population-level
hyperparameters), `{prefix}_hierarchical_locus_summary.csv` (one row per
locus), `{prefix}_hierarchical_fitted_curves.csv`, and
`{prefix}_hierarchical_overlay.png`.

---

## Output files, in brief

Every fitting command writes a consistent set of files to `--output-dir`
(default `./hycline_output`, filename prefix configurable via `--prefix`):

- **`*_summary.csv`** -- one row per locus: fitted parameters, standard
  errors or credible intervals, log-likelihood, AICc
- **`*_fitted_curves.csv`** -- smooth curve points to replot the fit
  yourself, long format (one `locus` column identifying which curve each
  row belongs to)
- **`*_observed_data.csv`** -- your input, parsed and echoed back (a good
  sanity check that columns were read as intended)
- **`*_overlay.png`** -- one fitted-curve line per locus, shared axes
- **`*_run_config.json`** -- every setting used, for exact reproducibility

`--no-plot` skips PNG generation (CSV/JSON output still written) --
useful when scripting many runs. `--plot-grid`/`--no-plot-grid` control
the overlay plot specifically.

---

## Quick reference: full pipeline, start to finish

```bash
# 1. Filter to ancestry-informative markers
hycline filter-aims --input genomewide.vcf --metadata parental_labels.csv \
    --pop1-label parent1 --pop2-label parent2 --output filtered.vcf

# 2. Convert to HyCline's site-aggregated hard-count format
hycline generate-geographic-input --input filtered.vcf --metadata metadata.csv \
    --output hardcounts.csv

# 3a. Fit with classic MLE, no tail
hycline fit-geographic --input hardcounts.csv --output-dir results_classic/

# 3b. Compare tail models
hycline compare-tails --input hardcounts.csv \
    --n-alleles-col locus0.nSamples --freq-col locus0.A --output-dir results_tails/

# 3c. Refit the winning tail model with full HZAR parity, Bayesian
hycline fit-geographic --input hardcounts.csv \
    --tail mirror --hzar-tail --method bayesian \
    --width-upper-bound 150 --output-dir results_final/
```

---

## A note on where this guide's examples came from

Every command and every block of real output in this guide was generated by
actually running HyCline against synthetic test data while writing it -- not
reconstructed from memory.
