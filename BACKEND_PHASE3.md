# HyCline Backend Notes: Phase 3 (Genomic Clines)

Companion to `BACKEND_PHASE1.md` (classic geographic cline fitting) and
`BACKEND_PHASE2.md` (Bayesian geographic cline fitting). This document
covers a genuinely different model, not a variant of the geographic cline:
locus ancestry as a function of an individual's genome-wide hybrid index,
rather than as a function of geographic position.

---

## 1. Why there's no HZAR baseline here -- and what stands in for it

Every previous phase of this project was validated against HZAR's own
published numbers -- HZAR is a geographic cline tool, and matching its
output was the whole point of Phases 1-2. **HZAR has no genomic cline
functionality at all.** Geographic and genomic clines are complementary
but separate analyses in the hybrid-zone literature: geographic clines ask
"how does allele frequency change across space," genomic clines ask "how
does a locus's ancestry compare to an individual's genome-wide admixture
proportion, independent of where that individual was sampled." An
individual can be a 50/50 hybrid whether it was sampled at the center of a
geographic transect or, in a mosaic hybrid zone, somewhere with no clean
geographic pattern at all -- genomic clines are the tool for the latter
situation and for genome-wide outlier scans generally.

So Phase 3 doesn't have an "HZAR published value" to reproduce the way
`tests/test_manakin_validation.py` does for Phase 1. Two things stand in
for it instead:

- **Mathematical validation against synthetic data with known true
  parameters** (neutral, barrier, permeable, and biased-introgression
  loci), the same kind of check Phase 1 used *before* the real Manakin
  data was available -- see `tests/test_mle_genomic.py` and
  `tests/test_genomic_cline.py`.
- **Cross-validation against BGC (Gompert & Buerkle 2012) on a real,
  published dataset -- done.** See section 7 below for the full writeup:
  77 real loci from Hermansen et al. (2014)'s Italian sparrow hybrid zone,
  independently fit by HyCline and compared against BGC's own published
  parameter estimates on the same data. Strong, highly significant,
  same-direction agreement (Pearson r=0.85 and 0.69-0.85 depending on
  scale, both p<1e-10; 75-82% sign agreement) across every locus.

---

## 2. Why genomic clines are a sibling model, not a relabeled geographic one

It might look at first like the genomic cline is "the same sigmoid, just
with hybrid index instead of distance on the x-axis" -- it is *not*, and
conflating the two would be a real modeling error:

- **Different functional form.** The geographic cline
  (`hycline.models.cline`) is a sigmoid in raw distance:
  `p(x) = pmin + (pmax-pmin)/(1+exp(-4(x-c)/w))`, with free asymptotes
  `pmin`/`pmax` and an unbounded-magnitude x-axis (distance can be any
  positive number). The genomic cline (`hycline.models.genomic_cline`) is
  a **logit-linear regression on a logit-transformed, unit-bounded
  x-axis**: `logit(phi(h)) = u + v*logit(h)`, where h in [0,1] is hybrid
  index. There's no `pmin`/`pmax` -- the model is anchored so phi(0)=0 and
  phi(1)=1 always (a locus's ancestry must be 0 in a pure-parental-0
  individual and 1 in a pure-parental-1 individual, by construction of
  what hybrid index means), and the only free parameters are the bias `u`
  and rate `v`.
- **Different data shape.** Geographic cline data is per-*site*: at each
  sampled locality, some number of individuals were genotyped and their
  counts aggregated into one (n_alleles, n_focal) pair per site
  (`AlleleFreqTransect`). Genomic cline data is per-*individual*: each
  individual has their own continuous hybrid index and their own genotype
  at the locus, with no site-level aggregation (`HybridIndexLocusData`)
  -- two individuals at the same geographic site can have very different
  hybrid indices (that's the whole point of a mosaic or well-mixed hybrid
  zone), so aggregating by site would destroy exactly the signal genomic
  clines are meant to capture.
- **Different reference point for "no effect."** In the geographic model,
  a "flat" relationship (no cline) isn't really representable within the
  sigmoid family at all -- HZAR/HyCline handle that by comparing against a
  null model separately. In the genomic model, "no effect" has a precise,
  named parameter value: `u=0, v=1` reproduces `phi(h) = h` exactly (the
  locus tracks genome-wide admixture with no bias and no rate difference)
  -- see `tests/test_genomic_cline.py::
  test_neutral_locus_reproduces_identity_line`. This makes v and u
  directly interpretable as *deviations from neutral admixture*, which is
  the actual scientific question genomic cline analysis is usually asking
  (is this locus behaving like the rest of the genome, or not).

Because of these real differences, `hycline.api.fit_genomic_cline()` is a
separate top-level function from `fit_cline()`, not a `data_type=` flag on
the same one -- see `BACKEND_PHASE1.md`'s `fit_cline` discussion for the
cross-reference. The likelihood machinery (binomial, same idea as Phase
1's `hycline.likelihoods.binomial`) genuinely is shared in *spirit*
between `hycline.likelihoods.binomial` and
`hycline.likelihoods.genomic_binomial` -- both are "observed focal-allele
count out of total, binomial around a predicted probability" -- but kept
as separate modules rather than one shared function parameterized by which
cline function to call, since the two domains have different validity
ranges (arbitrary distance vs. [0,1] hybrid index) and, once tail models
and hierarchical pooling are added on the geographic side, different
extension paths. Revisit this decision if the duplication becomes a real
maintenance burden; for two structurally-similar-but-conceptually-distinct
models it reads clearer kept apart for now.

---

## 3. The model, precisely

```
logit(phi(h)) = u + v * logit(h)
phi(h) = inv_logit(u + v * logit(h))

for each individual i:
    n_focal_i ~ Binomial(n_alleles_i, phi(hybrid_index_i))
```

- **u** (bias/center): controls `phi(0.5) = inv_logit(u)`. u=0 means no
  bias at the 50% hybrid-index point. u != 0 is a signature of asymmetric
  introgression -- the locus's ancestry favors one parental genome more
  than the genome-wide average does.
- **v** (rate): v=1 (with u=0) is the neutral null. v>1 is a *steeper*
  transition than the genome-wide average -- interpreted in the
  literature as resistance to introgression / linkage to a barrier locus
  under selection against the minority ancestry near intermediate hybrid
  index. v<1 is *shallower* than average -- elevated introgression /
  reduced barrier effect at that locus relative to the rest of the genome.
  `v` is constrained positive in the fit (`_bounds()` in
  `hycline.inference.mle_genomic`, lower bound `1e-3`) -- a negative v
  would invert the direction of the relationship between locus ancestry
  and genome-wide admixture, which has no clean biological interpretation
  in the standard reading of this model and isn't a case the literature
  this is based on (Fitzpatrick 2013; Gompert & Buerkle 2011) treats as
  meaningful.

`GenomicFitResult.summary()` includes a one-line plain-language
interpretation of the fitted `v` (steeper/shallower/~neutral) specifically
because "v=4.39" means little to someone not steeped in this literature,
and getting the interpretation direction right by hand is an easy place to
make a sign error -- see the interpretation logic in
`hycline.inference.mle_genomic.GenomicFitResult.summary`.

---

## 4. Fitting: classic MLE only (no Bayesian backend yet)

Phase 3 currently only implements the classic MLE path
(`hycline.inference.mle_genomic.fit_genomic_cline_mle`), following the
exact same approach as Phase 1's geographic MLE (L-BFGS-B via
scipy.optimize, data-driven initial guess, numerical-Hessian standard
errors -- see `BACKEND_PHASE1.md` sections 3 and 5 for the general
reasoning, not repeated here).

**A Bayesian NUTS backend for genomic clines is a natural, tracked
extension, not built in this pass.** The pattern from Phase 2 would carry
over directly: a NumPyro model with `u ~ Normal(0, wide)` and
`v ~ LogNormal(0, wide)` priors (LogNormal for v specifically, since v
must be positive -- a log-normal-ish prior in interpretable units is one
of the more common choices in the genomic-cline literature itself for
this exact parameter, e.g. the priors used in `bgchm`/`bgc-hm`'s Stan
models). This would matter most for the Phase 4 hierarchical multi-locus
extension, where genome-wide outlier detection benefits directly from
proper posterior uncertainty per locus rather than point estimates and
Wald-style CIs.

---

## 5. Hybrid index estimation: a real simplification, stated plainly

`hycline.data.hybrid_index.estimate_hybrid_index` computes, for each
individual, the mean (optionally differentiation-weighted) proportion of
"parental population 1" alleles across a set of loci. This is a standard,
defensible baseline -- it's literally "what fraction of this individual's
genome, sampled at these markers, looks like parental population 1" -- and
it's adequate for feeding into the genomic cline fit, which (per
Fitzpatrick 2013's original description) treats hybrid index as a known
covariate.

**It is not what the current state-of-the-art tools do.** `gghybrid` and
`bgchm` both fit hybrid index with an explicit Bayesian model and
propagate *uncertainty in h itself* into the genomic cline fit -- an
individual's hybrid index estimated from 20 markers is far less certain
than one estimated from 2000, and the more sophisticated tools account for
that. HyCline's estimator treats every individual's h as a fixed, known
number once computed, which will understate uncertainty when the number
of ancestry-informative markers per individual is small. This is flagged
here rather than quietly matched to a standard it doesn't meet -- it's a
legitimate scope decision for a first cut, not an oversight, but anyone
using this on a marker-sparse dataset should know the uncertainty in `h`
itself isn't being carried through to the genomic cline's reported CIs.

**Revisited: allele orientation.** The estimator assumed `genotypes`
already counted population 1's allele at every locus, and the CLI passed
raw VCF ALT dosage, which counts population 0's allele at roughly half of
real AIMs. Found while benchmarking with `benchmarking/simulate_vcf.py`
(ALT population randomized per AIM): every individual, parental
references included, came out at 0.487-0.518, with correlation -0.14 to
the true hybrid index. The existing CLI test always put population 1's
allele in ALT, which is why it passed. `estimate_hybrid_index` now
orients each locus from the parental frequencies whenever both are given
(flipping loci where population 1's frequency is lower), which leaves
already-oriented input unchanged; on the same simulated data the
correlation is 1.000 with mean absolute error 0.009. For beagle input the
CLI now also drops ANGSD's equal-likelihood no-data rows (previously
counted as heterozygotes) and, in Mode 1, uses `filter_aims.em_allele_freq`
for the parental frequencies. Regression tests:
`test_cli_estimate_hybrid_index_orients_alleles_by_parental_frequency`,
`test_cli_estimate_hybrid_index_beagle_missing_genotypes_not_counted`,
`test_loci_are_oriented_by_parental_frequencies`.

**Revisited: a maximum-likelihood estimator, now the default.** Even
oriented, the mean-dosage estimator is pulled toward 0.5 by partially
differentiated loci (it treats every locus as diagnostic) and, for
beagle input, by low depth (per-individual flat-prior expected dosage).
`estimate_hybrid_index_ml` maximizes, per individual,
sum_l log(GL0 (1-p)^2 + GL1 2p(1-p) + GL2 p^2) with p = (1-h) p0 + h p1,
parental frequencies clipped to [0.001, 0.999] so an allele unseen in a
finite panel can't make a locus impossible. Each locus's term is a
quadratic in h, so its coefficients are precomputed; the maximum is
found from a 21-point grid then bisection on the derivative's sign, and
the 95% interval (log-likelihood drop 1.92) by bisection bracketed from
the grid. Checks: matches a 20,001-point brute-force search to within
1e-4 for the estimate and both interval ends, including individuals at
h = 0 and 1; invariant to allele polarity; missing genotypes equivalent
to dropping them. On simulator data (15 + 15 references, 40 hybrids,
2000 SNPs, 10% missing) mean hybrid error was 0.005-0.016 (ml) vs
0.010-0.106 (mean, weighted) from depth-1 beagle to true genotypes, with
95% interval coverage 0.90-0.97.

Two things found while validating it. (1) On a 100,000-SNP, depth-4
beagle file with 5% AIMs and 30 + 30 references, using every locus gave
interval coverage 0.57 and estimates compressed toward 0.5 (correlation
-0.79 between error and h - 0.5). With the simulator's true panel
frequencies coverage was 0.93; with estimated frequencies on the AIMs
alone it was 0.94 (mean error 0.0039 vs 0.0074, 0.4 s vs 18 s): small
errors in the estimated frequencies of thousands of barely differentiated
loci accumulate, because the model treats p0/p1 as exact. The Genomic
guide therefore recommends running it on AIMs. (2) The CLI took 104 s on
that file, half of it an O(n^2) `list.index` lookup of each locus's row
in `cmd_estimate_hybrid_index`, replaced with a dict (49 s, identical
output). Regression tests: `test_cli_estimate_hybrid_index_ml_recovers_hybrid_index_at_low_depth`
(simulator-generated depth-2 beagle) and the `test_ml_*` tests in
`tests/test_hybrid_index.py`.

**Revisited: `ml` is no longer the default -- it broke genomic-cline
fits.** Fitting clines against the `ml` hybrid index showed near-fixed,
neutral AIMs (true v = 1) fitting at v ~ 0.17. Cause: `ml` puts parental
references at exactly h = 0 or 1, and the logit-logistic cline forces
phi(0) = 0, so a parental at h = 0 carrying the population-1 allele has
likelihood ~0 (log ~ -23 after clipping) and the optimizer flattens v to
escape it. With the `mean` hybrid index (parentals ~0.02) the same loci
fit at v = 0.71-1.19. `--method mean` is the default again. The same
investigation showed two problems that predate this work: loci whose ALT
allele belongs to parental population 0 always fit at the v lower bound
(0.001), and partially differentiated neutral loci fit as shallow,
shifted clines (0.57/0.87 parental frequencies: v = 0.38, u = 1.1),
because the model has no parental frequencies. Bailey (2023, gghybrid,
`references/`) fits allele frequency = S0 + phi(h) (S1 - S0) with each
locus oriented to the allele more common in S1, which addresses all
three. Regression test:
`test_cli_default_hybrid_index_gives_unflattened_genomic_clines`.

**Revisited: parental frequencies in the genomic-cline model; `ml`
default restored.** Bailey (2023, gghybrid, section 2.1.1, read
directly from `references/`) models "the allele j1 with higher frequency
in source S1 than S0" as A = S0 + phi(h)(S1 - S0), with phi the
logit-logistic term (written there as h^v / (h^v + (1-h)^v e^u), so its
u has the opposite sign to HyCline's logit(phi) = u + v logit(h)); it
estimates parental frequencies from the reference samples, does not
propagate their uncertainty, and by default leaves the reference
individuals out of cline estimation (`ggcline`, `include.Source =
FALSE`). HyCline now does the same: `genomic_cline_allele_freq` gives
p0 + (p1 - p0) phi(h), carried as `parental_freq_pop0/1` on
`HybridIndexLocusData`/`GenotypeLikelihoodLocusData` (defaults 0/1 =
the previous model exactly) through the binomial and genotype-likelihood
MLE likelihoods and both hierarchical NUTS models. Orientation is not
needed: with p1 < p0 the allele frequency simply decreases as phi
rises. `generate-genomic-input` computes p0/p1 from labelled references
(VCF sample frequency, beagle `em_allele_freq`) or reads them from
`--parental-freq-csv`, clips them to [0.001, 0.999] (an individual at
h = 0 can't make a locus impossible), drops loci with equal or unknown
frequencies, writes `<locus>.p0`/`.p1` columns, and leaves the
references out unless `--include-parentals`.

Validation, simulator data where every locus is neutral (true u = 0,
v = 1), 15 + 15 references, 60 hybrids, 1000 AIMs with parental
frequency difference >= 0.3 of both polarities. Old pipeline (no
parental frequencies, `mean` hybrid index): every reversed locus at
v = 0.001; 86% of partially differentiated forward loci flagged
non-neutral. New model with the `ml` hybrid index: median u = 0.02,
v = 1.07; 95% intervals excluded u = 0 for 1.5% and v = 1 for 6.5% of
loci; 2 of 1000 at the v bound (inspected one: its sampled genotypes
really are near-flat -- log-likelihood -52.4 at v ~ 0 vs -58.5 at v = 1).
New model with the `mean` hybrid index: median v ~ 2 for every locus
class, because that hybrid index is compressed (0.14-0.82) on partially
differentiated hybrid-index loci -- so `--method ml` is the default
again: the problem that made it unusable (references at exactly 0/1
under phi(0) = 0) is gone once references are excluded and p0 > 0.
Beagle input at depth 5 (genotype-likelihood MLE): median u ~ 0,
v = 1.08-1.10, but the v interval excluded 1 for 9.2% of neutral loci;
with the simulator's true panel frequencies 8.7%, and with true
frequencies and true hybrid index 7.7% -- so most of the excess is the
Wald interval for v at n = 60, not the new model, with estimated
frequencies adding a little. Hierarchical NUTS (hard counts and GLs, 50
hybrids, ~100 loci, short chains): mu_u ~ 0.00, mu_log_v ~ 0.00-0.02.
Regression tests: `test_cli_genomic_pipeline_with_parental_frequencies_recovers_neutral_clines`,
`test_cli_genomic_pipeline_without_parental_frequencies_pins_reversed_loci`,
`test_neutral_loci_recovered_for_any_parental_frequencies`,
`test_cli_generate_genomic_input_parental_options`.

**Revisited: first benchmarks with realistic reference panels and
non-neutral loci.** `benchmarking/simulate_vcf.py` gained
`--hybrid-freqs population` (true population frequencies; reference
panels and hybrids sampled from them, so panel frequencies carry real
sampling error) and per-locus genomic clines (barrier / permeable /
biased classes among the AIMs, random u / log v elsewhere). Setup: 15 +
15 references, 100 hybrids, 3000 SNPs, 30% AIMs (population frequency
difference >= 0.5), filter-aims at 0.5, `ml` hybrid index on the AIMs,
generate-genomic-input with parental labels, fit-genomic.

| Panels, outliers | HI error | HI 95% coverage | v: barrier / neutral / permeable (estimated inputs) | same, true h and frequencies |
|---|---|---|---|---|
| panel, none | 0.011 | 0.93 | - / 1.03 / - | - / 1.03 / - |
| panel, 45% of AIMs | 0.026 | 0.58 | 3.89 / 1.16 / 0.33 | 4.20 / 1.04 / 0.25 |
| population, none | 0.020 | 0.68 | - / 1.04 / - | - / 1.02 / - |
| population, 45% of AIMs | 0.037 | 0.54 | 3.27 / 1.15 / 0.37 | 4.26 / 1.02 / 0.26 |

(true v: barrier 4, neutral 1, permeable 0.25; biased loci |u| = 2 was
recovered at a median of 1.97 with every sign correct.) Cline fitting
is accurate given true inputs; the attenuation toward neutral comes from
the estimated hybrid index, whose likelihood assumes every locus is
neutral, and from treating panel frequencies as exact (coverage 0.93 ->
0.68 with neutral loci alone). In population mode filter-aims missed
13-20% of true AIMs (near the threshold) and with 5 references per
population 6% of retained loci were not AIMs. Open: realistic outlier
fractions, and a second pass that drops flagged outliers from the
hybrid-index set.

**Read depth and AIM retention.** `benchmarking/depth_aim_retention.py`
simulates the same true genotypes at a range of mean depths (reads come
from their own random stream) and scores `filter_aims_beagle`, and
`filter_aims_vcf` on genotypes called from the same reads, against the
simulator's `is_aim`. 20 + 20 parental and 20 hybrid individuals, 5,000
SNPs, 1,000 AIMs simulated with `--aim-cutoff 0.05`, 3 seeds, Poisson depth:

| Mean depth | 1 | 2 | 3 | 4 | 6 | 8 | 10+ |
|---|---|---|---|---|---|---|---|
| Recall, beagle, cutoff 0.10 | 0.936 | 0.964 | 0.982 | 0.993 | 0.999 | 1.000 | 1.000 |
| Recall, called VCF, cutoff 0.10 | 0.950 | 0.955 | 0.964 | 0.972 | 0.984 | 0.994 | 0.999-1.000 |
| Recall of AIMs fixed in both panels, beagle, cutoff 0.05 | 0.925 | 0.977 | 0.981 | 0.992 | 0.997 | 1.000 | 1.000 |

With uneven depth (`--depth-dispersion 2 --individual-depth-sd 0.5`) the
beagle numbers are within 0.01 of these. No non-AIM was retained in any
run, but that says little about specificity: the simulator gives non-AIMs
the same source frequency in both panels, so none is near the cutoff.
Zero depths are redrawn rather than made missing, so low mean depths here
have no missing data, unlike real data.

At cutoff 0.05 -- equal to the simulator's AIM rule -- beagle recall of
AIMs that are not fixed in both panels falls with depth (0.55 at depth 1,
0.42 at depth 30) while called-VCF recall rises to 0.999. Those AIMs have
panel frequencies exactly at the cutoff (2 minor alleles of 40). The EM
estimate is the maximum-likelihood frequency, and homozygous individuals'
genotype likelihoods carry small non-zero heterozygote values (sequencing
error, 6-decimal rounding), which put it slightly past the cutoff (e.g.
0.0500128 for a panel frequency of 0.05, confirmed by direct likelihood
maximization; the EM had converged). Higher depth sharpens the estimate
around that value, so about half such loci fail. A called-genotype
frequency is exactly 0.05 and passes. Any locus whose panel frequency is
exactly a `--minor-allele-freq` cutoff is therefore retained from a VCF but
retained about half the time from a beagle file.

**File formats.** User-created tables -- metadata, parental labels,
parental frequencies (the flag is now `--parental-freq-file`, earlier
`--parental-freq-csv`) and trait data -- are read as tab-delimited via
`hycline.data.io.read_user_table` / `_read_csv(..., sep="\t")`, which
reject a comma-separated file with an explicit message;
`estimate-hybrid-index` writes a tab-delimited table. HyCline's own
intermediate files stay CSV. `benchmarking/simulate_vcf.py` writes
`_metadata.tsv`, `_truth.tsv` and a `_params.tsv` record of every option,
the command line, git commit, versions and realized counts (a seed is
drawn and recorded when none is given).

---

## 6. CLI: `fit-genomic` and `fit-genomic-multi`

Following the same `fit-geographic`/`fit-geographic-multi` split established for geographic
clines (`BACKEND_PHASE1.md`, README's "Command-line usage" section):

- **`fit-genomic`**: one locus, one CSV (one row per individual, a
  hybrid-index column, and that locus's genotype data).
- **`fit-genomic-multi`**: many loci at once from a wide-format CSV (one
  row per individual, one column-pair per locus) -- this is the more
  realistic use case in practice, since genomic cline analysis is
  essentially always about comparing many loci to find outliers, not
  examining one locus in isolation. Loci are auto-detected from column
  suffixes (default `.n`/`.freq`), the same mechanism as
  `hycline.data.io.detect_loci` for geographic multi-locus files but with
  its own default suffixes -- there's no HZAR naming convention to match
  here (section 1), so these are HyCline's own choices, not borrowed ones.

Both commands accept `--n-alleles-col`/`--n-alleles-suffix` as *optional*
-- if omitted, every individual is assumed to contribute `ploidy` alleles
at every locus (the normal case for complete, non-missing diploid
genotype calls), which keeps the common case simple without requiring a
redundant sample-size column when it would just be a constant.

**Hybrid-index estimation now has CLI wiring**: `hycline
estimate-hybrid-index` (section 9). The CLI subcommands here still expect
a `--hybrid-index-col` already present in the input CSV -- estimation is
a separate, prior step, not folded into `fit-genomic`/`fit-genomic-multi`
themselves -- but it no longer requires writing Python to produce that
column in the first place.

---

## 7. The gghybrid/BGC cross-validation, in detail

`examples/validate_gghybrid_italy.py` is the full analysis;
`tests/test_gghybrid_cross_validation.py` locks in the quantitative
thresholds as a regression test. This section covers what the data is,
how it was obtained and reshaped, and exactly what was and wasn't
compared -- worth reading before citing this validation, since the
"cross-validation" label could oversell it without these caveats attached.

### 7.1 The dataset

Hermansen et al. (2014) studied the Italian sparrow (*Passer italiae*), a
homoploid hybrid species formed from house sparrow (*P. domesticus*) and
Spanish sparrow (*P. hispaniolensis*) ancestry. The `gghybrid` R package
(Bailey 2024) bundles this dataset -- genotypes at 77 loci (mostly
autosomal SNPs plus the mtDNA marker ND2, coded as a diploid homozygote to
match BGC's own convention) for 569 individuals, of which 433 are
non-reference "test" individuals from Italian hybrid-zone populations and
the remainder are parental reference samples (house sparrow: Kralove,
Oslo; Spanish sparrow: LesinaSPANISH) -- specifically to reproduce a
BGC-vs-gghybrid comparison the package author had already done (see
`RBailey_231_232a_bgc_introgress_comparison.R` in the gghybrid repo). That
comparison script also bundles **BGC's own precomputed output** for this
exact dataset: per-locus `alpha`/`beta` posterior means
(`a.out_italy.txt`/`b.out_italy.txt`) and per-individual hybrid index
estimates (`hi.out_italy.txt`), all cross-referenced to locus/individual
identities via `loci.csv`/`id_italy.csv`.

This is what made real cross-validation possible without needing to run R
or BGC's own C++ binary: the *inputs* (raw genotypes) and BGC's *outputs*
(alpha, beta, hybrid index) were already sitting in the same repository,
put there for exactly this kind of comparison by the gghybrid author.

### 7.2 Reshaping the data (`examples/hermansen_italy_sparrows.csv`)

gghybrid/BGC's native format is one row per allele copy (two rows per
diploid individual), 0/1/NA-coded relative to a fixed reference allele,
with a separate precol-header convention. HyCline's wide-format multi-locus
reader expects one row per individual with `{locus}.n`/`{locus}.freq`
column pairs. The conversion (done once, as a small standalone script, not
part of the HyCline package itself) sums each individual's two allele-copy
rows per locus to get `n_focal`/`n_alleles`, joins in BGC's own hybrid
index estimate via the `id_italy.csv` <-> `hi.out_italy.txt` index
mapping, and writes the result as a standard CSV. All 433 test individuals
matched cleanly with zero individuals dropped in the join --
`examples/hermansen_italy_sparrows.csv` is the result, and
`examples/hermansen_italy_bgc_reference.json` holds BGC's own alpha/beta
per locus for comparison.

**Using BGC's own hybrid index (not HyCline's `estimate_hybrid_index`, and
not gghybrid's) was a deliberate choice**, not an oversight: it isolates
the comparison to "does an independent implementation of a closely related
genomic cline model recover a similar per-locus signal from the same
data and the same covariate," rather than compounding it with a second,
unrelated comparison of hybrid-index estimators.

### 7.3 Why correlation/sign-agreement, not exact numerical match

BGC's alpha and beta are described in the literature (e.g. the Asclepias
hybrid-zone paper citing Gompert & Buerkle 2012) as: positive alpha =
excess ancestry from one parental species, negative = excess from the
other (this maps directly onto Fitzpatrick's u); beta's sign indicates
direction of deviation from random introgression (mapping onto some
transform of v, plausibly log-scale given beta values are centered near
zero and span both signs, unlike v itself which must stay positive).
**This mapping was inferred from the literature and the shape of BGC's
own output values, not confirmed against BGC's technical documentation
line by line** -- BGC's original description (Gompert & Buerkle 2011) is
for a model in the same family as Fitzpatrick's logit-logistic form but
not guaranteed to use an identical link function throughout (Fitzpatrick's
2013 paper itself frames the logit-logistic model as an *alternative* to
Gompert & Buerkle's original parameterization, not a straight
reimplementation of it).

Given that real uncertainty, the validation deliberately checks two things
that are robust to getting the exact alpha/beta<->u/v transform slightly
wrong:

- **Correlation** (Pearson, both under a raw-`v` and a `ln(v)` hypothesis
  for how `v` maps to `beta`, plus Spearman rank correlation, which is
  invariant to *which* monotonic transform is right) -- both interpretations
  give strong, highly significant correlations (Pearson r=0.85 for u vs.
  alpha; r=0.69-0.85 for v vs. beta depending on the assumed scale;
  Spearman rho=0.66-0.68 either way).
- **Sign agreement** across all 77 loci (75.3% for u/alpha, 81.8% for
  ln(v)/beta) -- a coarser but more robust check than exact correlation,
  since it only requires the two implementations to agree on *direction*
  (which parental ancestry is favored; whether a locus is more or less
  permeable than the genome-wide average), not on magnitude or scale.

**What this establishes**: two independent implementations of closely
related genomic cline models, run on the same real 77-locus dataset,
recover the same qualitative per-locus signal with high, statistically
significant consistency. **What this does not establish**: that HyCline
reproduces BGC's exact reported numbers -- that would require confirming
BGC's precise link function and replicating it bit-for-bit, which wasn't
done here and is a reasonable follow-up if exact numerical parity (rather
than validated agreement) becomes a priority.

---

## 8. Known rough edges / follow-ups

- The gghybrid/BGC cross-validation (section 7) checks correlation and
  sign agreement, not exact numerical reproduction of BGC's alpha/beta --
  see section 7.3 for why, and what confirming BGC's exact link function
  would take if exact parity becomes a priority.
- No Bayesian backend yet (section 4).
- Hybrid index treated as a fixed known covariate, not itself estimated
  with propagated uncertainty (section 5).
- No hierarchical multi-locus pooling (the direct genomic-cline analog of
  the Phase 4 geographic extension already planned) -- `fit-genomic-multi`
  fits every locus with fully independent priors/optimization, the same
  scope limit `fit-geographic-multi` has on the geographic side
  (`BACKEND_PHASE2.md` section 9 covers the geographic version of this
  same decision).
- The genomic cline model as implemented assumes biallelic loci with a
  clearly designated "parental population 1" allele -- multiallelic loci
  would need the same kind of biallelic reduction Phase 1's geographic
  multi-locus reader already applies to HZAR's own multiallelic example
  loci (`BACKEND_PHASE1.md` section 9.3), not yet formalized as a shared
  utility between the geographic and genomic readers.

---

## 9. CLI wiring for hybrid-index estimation

`hycline estimate-hybrid-index` exposes `estimate_hybrid_index` (section
5) from the command line -- closing the CLI gap flagged there and in the
project README's "Not yet implemented" list.

### 9.1 Design: two modes, not one, after finding the original design left a real gap

The function itself takes pre-computed parental allele frequencies as
direct arguments (`parental_freq_pop0`/`parental_freq_pop1`). The CLI
was originally built with a single mode: `estimate-hybrid-index` takes
one wide genotype-dosage CSV containing *both* the parental reference
individuals and the hybrid-zone individuals to estimate, identifies
which rows are which reference population via `--population-col`/
`--pop0-label`/`--pop1-label`, and computes the parental frequencies
internally (the mean dosage across each reference group, per locus)
before calling `estimate_hybrid_index`. This covers the common case --
the only thing the user needs to prepare is raw genotype data with
population labels, not a second, hand-derived frequency table -- but it
left a real gap for users who *already have* a precomputed per-locus
frequency table (e.g. from a separate reference panel, or a published
source) rather than raw reference genotypes sitting in the same file as
their test individuals.

**Mode 2** (`--parental-freq-csv`) closes that gap: a separate CSV of
per-locus frequencies, matched to the main input's locus columns by
name (`--freq-locus-col`, default `locus`), read directly rather than
computed from reference genotypes. The two modes are mutually exclusive
at the CLI level -- `cmd_estimate_hybrid_index` validates that exactly
one is given, with a clear error for either "neither" or "both" rather
than an ambiguous default. Locus columns are still auto-detected using
the same `hycline.data.converter.DEFAULT_RESERVED_COLUMNS` convention
introduced for the genotype-to-GL converter (`BACKEND_PHASE5.md` section
7) regardless of which mode is active -- both commands share the same
"wide genotype-dosage CSV with some reserved covariate columns" input
shape, so sharing the detection logic keeps behavior consistent.

**Verified as an exact equivalence, not just "runs and looks plausible"**:
`tests/test_cli.py::
test_cli_estimate_hybrid_index_precomputed_frequencies_matches_mode1`
computes Mode 1's own internally-derived frequencies, writes them out as
a Mode 2 input file, and confirms the two modes produce numerically
identical hybrid-index estimates (`np.testing.assert_allclose`, default
tolerance) -- Mode 2 isn't a separate implementation that happens to
agree approximately, it feeds the exact same numbers into the exact same
underlying `estimate_hybrid_index` call.

### 9.2 Real-data validation, not just "the command runs"

Checked against known ground truth, not just confirmed to execute
without error: a synthetic dataset was built with 15 pure-parental
reference individuals per population (20 ancestry-informative loci, true
parental frequencies drawn near 0 and near 1) and 30 hybrid-zone
individuals simulated from a *known* true hybrid index. Running
`estimate-hybrid-index` on this data:

- Parental reference individuals correctly recovered near their expected
  extremes: pop0 mean estimated h = 0.045, pop1 mean estimated h = 0.955
  (true values 0 and 1).
- Hybrid-zone individuals: correlation between estimated and true hybrid
  index = 0.977; mean absolute error = 0.047.

`tests/test_cli.py::test_cli_estimate_hybrid_index_recovers_known_true_hybrid_index`
locks in both checks (correlation > 0.9, parental means within 0.1 of
their expected extreme) as a regression test, not just a one-off
manual check. A further end-to-end test
(`test_cli_estimate_hybrid_index_output_chains_into_genomic_fitting`)
confirms the command's output isn't just numerically close to ground
truth in isolation -- it was fed directly into a real genomic-cline fit
(a simulated `v=3.0` locus, using the *estimated* hybrid index as the
covariate) and correctly recovered `v > 1.5`, confirming the full
realistic workflow (estimate hybrid index -> fit a genomic cline using
it) works end to end, not just each piece separately.

### 9.3 Known rough edges

- **No uncertainty propagation**, same limitation as the underlying
  function itself (section 5) -- the CLI doesn't add anything beyond what
  `estimate_hybrid_index` already does or doesn't do. This applies to
  both modes equally; Mode 2's frequencies are used as given, with no
  way to express uncertainty in a precomputed frequency itself (e.g. from
  a small reference panel) even if the user has it.
- **Mode 2 requires an exact locus-name match** between the main input
  file's columns and `--parental-freq-csv`'s locus column -- no fuzzy
  matching or case-insensitivity, so a naming mismatch (e.g. `Locus1` vs
  `locus1`) produces the same "no row for locus" error as a genuinely
  missing locus, rather than a more specific "did you mean" suggestion.
