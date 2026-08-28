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
geographic-cline pipeline; see that guide for its own full flag reference.

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
  7 loci checked, 3 retained (>= 0.9 in one population and <= 0.1 in the other, either direction)

Wrote filtered.vcf
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

```bash
hycline estimate-hybrid-index \
    --input genomewide.vcf \
    --metadata parental_labels.csv \
    --population-col population --pop0-label parent0 --pop1-label parent1 \
    --weight-by-differentiation \
    --output hybrid_index.csv
```

Real output:

```
Detected input format: vcf (from file extension; use --format to override)
Parsing 'genomewide.vcf' ...
  7 biallelic SNPs parsed, 72 samples.
  7 loci detected: ['aim_neutral', 'aim_barrier', 'aim_permeable', 'background0', 'background1', 'background2', 'background3']
  Parental population 0 ('parent0'): 6 reference individuals
  Parental population 1 ('parent1'): 6 reference individuals

Hybrid index estimated for 72 individuals.
  Range: [0.062, 1.000]  (parental reference individuals should land near 0 or 1 -- a useful sanity check on the result)

Wrote hybrid_index.csv
Merge this into your metadata (e.g. via the sample-name column) to get a hybrid_index column any genomic-cline fitting command can use.
```

Hybrid index is computed from *every* locus in the file, not just AIMs --
run this on your unfiltered genome-wide data, before Step 1's filtering,
so the estimate draws on as much ancestry information as possible.
Parental reference individuals are included in the output alongside your
test individuals, specifically so you can sanity-check them landing near
0 or 1.

**Per-locus allele dosage is automatically re-oriented to a consistent
direction before averaging.** REF/ALT assignment in a VCF is arbitrary
with respect to which parental population carries which allele -- a
normal consequence of variant calling, not a data problem -- so some loci
will have population 1 carrying the higher-frequency allele and others
will have population 0 carrying it. `estimate-hybrid-index` detects this
per locus (comparing the two parental frequencies) and flips dosage for
any locus where population 0's frequency exceeds population 1's, so every
locus contributes in the same direction. When this happens, you'll see a
line like `Re-orienting 3/7 loci where population 0 (not population 1)
carries the higher-frequency allele` before the final estimate is
written; seeing 0 re-oriented loci is equally normal (it just means every
locus in this particular file already happened to be oriented the same
way).

**VCF input uses each individual's hard GT dosage directly. Beagle input
uses each individual's expected dosage under their own genotype
likelihoods** (posterior mean, not an argmax point estimate) -- there's
no proper marginalized hybrid-index estimator in HyCline yet (unlike
allele-frequency estimation, hybrid index has no model to marginalize
against), so this is the best available option for genotype-likelihood
data, not a claim of full rigor.

### Two ways to supply parental allele frequencies

**Mode 1** (shown above) -- `--metadata`/`--population-col`/`--pop0-label`/
`--pop1-label`: a separate metadata CSV maps sample names to which
parental population each is a reference for.

**Mode 2** -- `--parental-freq-csv`: a separate CSV of precomputed
per-locus parental frequencies, for when you already have a reference
panel's frequencies summarized rather than raw reference genotypes. No
`--metadata` needed for this mode.

```bash
hycline estimate-hybrid-index \
    --input genomewide.vcf \
    --parental-freq-csv parental_frequencies.csv \
    --freq-pop0-col freq_pop0 --freq-pop1-col freq_pop1 \
    --output hybrid_index.csv
```

Exactly one of the two modes must be given.

This is a standard, defensible baseline estimator -- the mean parental-
population-1 frequency across loci -- not the more sophisticated Bayesian
estimators some other tools (gghybrid, bgchm) use, which also propagate
uncertainty in the hybrid index estimate itself into downstream cline
fitting. HyCline's genomic-cline fitting treats hybrid index values as
directly observed, not as estimates carrying their own uncertainty.

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

### Ranking candidate loci by delta-AICc, not raw p-values

Whenever `fit-genomic` fits more than one locus, it also ranks every
locus by **delta-AICc against the fixed neutral null** (`u=0`, `v=1` --
no free parameters, not fit): Bailey (2024, the gghybrid paper)'s own
recommended alternative to using the reported significance/p-value to
pick candidate loci, since that value's relationship to true effect size
turned out non-monotonic in their own simulations ("use of [...] p
values to select candidate non-null loci is problematic and should be
avoided").

Real output, on the same 5-locus dataset used above for hierarchical
fitting:

```
Loci ranked by delta-AICc against the fixed neutral null (u=0, v=1) (Bailey 2024's recommended alternative to raw p-values for ranking candidate loci -- higher delta-AICc = stronger evidence against null):
  biased1              delta_AICc=  242.16  (u=2.766, v=1.264)
  barrier1             delta_AICc=  101.64  (u=-0.093, v=5.129)
  permeable1           delta_AICc=   65.06  (u=-0.076, v=0.218)
  neutral1             delta_AICc=   -0.70  (u=0.214, v=0.904)
  neutral2             delta_AICc=   -1.52  (u=-0.137, v=0.852)
  NOTE: delta-AICc near 0 or negative means no meaningful evidence of deviation from null -- the two extra free parameters (u, v) weren't worth their AICc penalty for that locus.
```

The three genuinely non-null loci (`biased1`, `barrier1`, `permeable1`)
land at the top with large positive `delta_aicc_from_null`; the two
truly neutral loci (`neutral1`, `neutral2`) correctly show values near
zero or slightly negative -- the simpler null model wasn't worth
improving on for those two. The same three columns
(`log_likelihood_null`, `aicc_null`, `delta_aicc_from_null`) are also
written into `*_summary.csv`, so you can re-sort or re-threshold without
re-running anything.

**What this doesn't do, stated plainly rather than left implicit:**
Bailey's own recommendation is a richer three-model comparison --
ranking separately by delta-AIC for a centre-only one-parameter model
and a rate-only one-parameter model, which additionally tells you
*which* parameter is driving a locus's deviation -- and found delta-WAIC
(needing Bayesian posterior samples) outperforms delta-AIC in their own
simulations. This ranking implements the simpler full-vs-fully-null
comparison, available from classic MLE alone with no Bayesian fit
required; it tells you *whether* a locus deviates from null, not
specifically *which* parameter drives it. See BACKEND_PHASE4.md
section 16 for the full account.

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

**A warning worth understanding, not dismissing, if it fires:**
`fit-hierarchical-genomic` checks, after fitting, whether any locus with
a steep fitted cline (`v > 20`) has its center sitting in a gap of your
sampled hybrid-index distribution (no sampled individual within 5% of
that center). Bailey (2024) found this specific combination -- not
uneven sampling generally, and not steep clines generally, but the two
together -- degrades genomic-cline accuracy: fitted centers become
artificially clustered and `v` estimates less accurate with higher
posterior uncertainty. Their own null/shallow-cline simulations (`v=1`)
were largely unaffected by the same uneven sampling, which is why this
check only fires for steep loci specifically. If you see it, the fitted
`v` for that locus is genuinely less trustworthy than its reported
credible interval alone would suggest -- denser sampling near its fitted
center, if you can get it, is the real fix; treating the point estimate
with extra caution is the fallback if you can't. See BACKEND_PHASE4.md
section 16 for the full investigation, including why this can't be
caught before fitting (it needs each locus's own fitted center, which
you don't have until after the fit).

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
