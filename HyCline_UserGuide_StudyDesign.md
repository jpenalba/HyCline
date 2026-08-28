# HyCline User Guide: Study Design Recommendations

**Document status: working draft, updated alongside HyCline's own
development.**

This guide collects design recommendations for planning a hybrid-zone
study around HyCline -- reference panel sizes, sequencing depth,
individuals-per-locus, and loci-per-study, organized by which pipeline
they affect. It's meant to be read *before* data collection, not after.

**A note on how to read this guide.** Most of the numbers here come
directly from real, verified behavior found while building and testing
HyCline -- a specific dataset, a specific result, reported honestly
either way. Where a recommendation is instead general population-
genetics practice rather than something tested in this project, it's
labeled as such. The two are kept clearly separate throughout.

This guide assumes familiarity with the Geographic, Genomic, Coupling,
and Trait cline guides -- see those for command-level detail on anything
referenced here.

---

## Geographic clines

### Parental reference panel size

`filter-aims` (shared by the geographic and genomic pipelines) retains
loci where each parental population's allele frequency clears
`--high-threshold` (default 0.9, with the low side always `1 -
--high-threshold`). **Small reference panels make this threshold noisy,
not just imprecise.** In direct testing, a locus with a true parental
frequency of 0.02/0.98 -- genuinely informative -- came back at an
*empirical* frequency of 0.875 in one population with only 8 reference
individuals, narrowly failing the default 0.9 cutoff by sampling chance
alone, not because the locus wasn't real.

**Recommendation**: more reference individuals per parental population
reduces this risk directly (the standard error of a frequency estimate
shrinks with sample size). If you're constrained to a small reference
panel, expect to lose some genuinely informative loci to threshold noise,
and consider loosening `--high-threshold` slightly, or switching to
`--min-freq-diff` (retains any locus with a large-enough frequency
*difference* between populations, without requiring either one to reach
a fixed extreme) rather than assuming every failed locus is actually
uninformative.

### Sequencing depth and the hard-call vs. genotype-likelihood decision

This is the single highest-leverage design decision in the geographic
pipeline, and the numbers behind it are stark, not a mild preference.

At 1x depth, naive hard-genotype-calling's nominal 95% confidence
intervals covered the true cline parameter only **67% of the time** --
badly overconfident. Separately, the *point estimate* itself is biased
at low coverage for frequencies near the parental extremes: a true
frequency of 0.02 came back estimated at 0.19 under a flat-prior
dosage collapse at 1x depth -- off by nearly an order of magnitude. Both
problems shrink as depth increases, converging to essentially no
difference by around 15x.

**Recommendation**: if your expected per-individual, per-locus depth is
below roughly 15x, plan to sequence for genotype likelihoods (ANGSD,
`-doGlf 2`) and use the beagle/GL fitting path
(`generate-geographic-input` auto-detects this from beagle input; see
the Geographic guide's "Why fitting isn't just 'estimate a frequency,
then fit'" section for the full mechanism). This matters most exactly
where it's easy to overlook: parental reference individuals sitting near
the frequency extremes, which is precisely where the bias above is
worst. If your depth is reliably high (15x+), hard calls are simpler and
the difference becomes negligible.

### Bayesian fitting: plan for a two-step workflow, not one

The default width prior (10x the transect's geographic span) is
deliberately loose, and in practice this produces real convergence
warnings -- low effective sample size, divergent transitions -- unless
tightened. This isn't a sign of a data problem; it's the default being
intentionally uninformative.

**Recommendation**: run classic MLE first (fast, no convergence
diagnostics to check), then use its width estimate to set a realistic
`--width-upper-bound` for the Bayesian fit. Treat this two-step sequence
as the default workflow rather than something to reach for only after a
Bayesian fit fails to converge.

### Tail models and Bayesian fitting: plan for classic MLE if you expect to need both

The join between the sigmoid core and the exponential tail is
non-smooth, and this is a genuine, documented NUTS sampler difficulty --
not simply a matter of more samples or a better prior. If your study
design anticipates clines with real tail structure (steep transitions
with asymmetric spread beyond the core, common in some hybrid zones),
budget for classic MLE plus `compare-tails` for model selection as the
primary analysis path. Attempt Bayesian tail fitting only with real
convergence checking in hand, not as a default.

---

## Genomic clines

### Individuals per locus, for single-locus precision

Genomic-cline parameter estimates (`u`, `v`) are noisier at moderate
sample sizes than intuition from geographic clines might suggest. In
direct testing, a locus simulated as exactly neutral (`u=0, v=1`) came
back with a 95% CI that *excluded* `v=1` at n=60 individuals -- a
genuine, if unlucky, finite-sample result, not a bug (see the Genomic
guide's Step 4 for the real numbers).

**Recommendation**: if your study's conclusions depend on individual
outlier loci rather than genome-wide patterns, plan for more individuals
per locus than 60, or lean on corroboration across loci and/or
hierarchical pooling rather than trusting any single locus's point
estimate in isolation.

### Number of loci, for hierarchical fitting specifically

`fit-hierarchical-genomic`'s population-level variance parameters are
themselves estimated from the spread across loci -- this requires a real
number of loci to be estimable at all, not just "more than one." In
direct testing, on the same underlying data and MCMC settings: 2 loci
gave R-hat 1.01, ESS as low as 245, and 442 divergent transitions (a
genuinely flagged, unreliable fit); the full 5 loci converged cleanly
(R-hat 1.00, minimum ESS 1994, zero divergences) and correctly recovered
known parameters.

**Recommendation**: if hierarchical pooling is central to your design (as
it often is -- the whole point is usually borrowing strength across
loci), don't plan around just a handful of loci. 5 converged cleanly and
2 didn't in this project's own matched-settings testing; published
genomic-cline studies using hierarchical/Bayesian pooling typically use
many dozens to thousands of loci, well beyond the point where this is a
concern. If your locus panel is necessarily small, `fit-genomic`
(independent per-locus fitting, no pooling) avoids this failure mode
entirely, at the cost of not borrowing strength across loci.

---

## Coupling analysis

### Independent vs. hierarchical input: pick one and stay consistent

`coupling-geographic`/`coupling-genomic` accept either independent
(`fit-geographic`/`fit-genomic`) or hierarchical
(`fit-hierarchical-geographic`/`fit-hierarchical-genomic`) per-locus
summaries. In direct testing on the same underlying data, independent
fits gave SD(slope) = 0.67 while hierarchical fits gave SD(slope) = 0.08
-- an 8-fold difference driven entirely by hierarchical shrinkage, not by
anything biological (see the Coupling guide's dedicated section on this).

**Recommendation**: this matters most when *comparing* coupling strength
across populations, species, or studies -- decide up front which fitting
mode you'll use throughout, and don't mix independent and hierarchical
inputs within a single comparison. A difference in apparent coupling
could otherwise just reflect a difference in which fitting mode
happened to be used, not a real biological difference.

### Number of loci

A coupling estimate from 2-3 loci isn't really answering the "genome-
wide concordance" question it's meant to answer -- it's dominated by
those specific loci's own sampling noise. Beyond that, `coupling-
genomic`'s theta estimate is a linear calibration fit over Firneno et
al.'s own limited published reference range; data whose `sigma_c`/
`sigma_v` fall outside that range (which happened directly in this
project's own small-dataset testing) can produce a theoretically
impossible negative theta, flagged by `within_calibration_range: false`
rather than hidden.

**Recommendation**: plan for a genuinely genome-wide locus panel if
coupling/theta is a primary goal of the study, not an add-on analysis
run on whatever handful of loci happen to be available. If working with
a small locus panel, treat the raw `sigma_c`/`sigma_v` variance metrics
as the reliable output and check `within_calibration_range` before
reporting theta at all.

---

## Trait clines

### Individuals per site, for the variance-cline model specifically

HyCline's trait-cline model fits each site's *observed* variance as real
data (`var_l`/`var_r`/`var_h`), not just its mean -- this is a real
strength (it captures genuinely elevated within-site variance near a
hybrid zone's center, verified directly in the Trait guide's hybrid-index
example, where `var_h` came back ~40x larger than `var_l`/`var_r` exactly
where it should). But a variance estimated from very few individuals is
itself noisy.

**Recommendation** (general statistical practice, not a number tested
specifically in this project): aim for enough individuals per site that
the site-level SD itself is reasonably stable -- as a rough starting
point, fewer than ~5 individuals per site should be expected to give
unstable `var_h` estimates (wide or zero-crossing confidence intervals),
though the exact threshold depends on the trait's true variance
structure. If you only have per-site means with no way to compute a
real SD, this model cannot be fit at all -- there's no single-shared-
variance fallback in HyCline, matching HZAR's own requirement.

### Hybrid index as a trait: uncertainty compounds

When fitting hybrid index itself as a geographic trait cline (see the
Trait guide's dedicated section), two layers of sampling noise stack:
`estimate-hybrid-index`'s own precision depends on how many informative
loci fed into it, and that already-noisy per-individual value then gets
treated as directly observed (no further uncertainty propagation) by the
trait-cline fit.

**Recommendation**: if hybrid-index-as-trait fitting is a planned part of
your design, invest in enough ancestry-informative markers for the
hybrid-index estimation step specifically -- not just enough for whatever
genomic-cline loci you're separately analyzing. A handful of AIMs may be
enough to screen individuals for a genomic-cline study, but a noisier
per-individual hybrid index will show up directly as inflated apparent
variance in the downstream trait-cline fit, potentially confounded with
the real biological `var_h` signal the model is trying to detect.

---

## Quick reference

| Question | Where the evidence lives |
|---|---|
| How many parental reference individuals for `filter-aims`? | Geographic guide, Step 1 |
| Hard calls or genotype likelihoods? | Geographic guide, "Why fitting isn't just 'estimate a frequency, then fit'" |
| How to set `--width-upper-bound` for Bayesian fits? | Geographic guide, "Choosing MLE or Bayesian inference" |
| Individuals per locus for genomic clines? | Genomic guide, Step 4 |
| Loci needed for `fit-hierarchical-genomic` to converge? | Genomic guide, "Fitting hierarchically" |
| Independent vs. hierarchical input for coupling? | Coupling guide, "Independent vs. hierarchical input" |
| Loci needed for a meaningful theta estimate? | Coupling guide, `coupling-genomic` section |
| Individuals per site for trait-cline variance estimates? | Trait guide, `fit-trait` |
| Planning hybrid-index-as-trait fitting? | Trait guide, "Fitting hybrid index as a geographic trait cline" |

---

## A note on where this guide's numbers came from

Every specific figure in this guide (the 67% coverage rate, the 0.19 vs.
0.02 bias, the R-hat/ESS/divergence comparison for hierarchical genomic
fitting, the 0.67 vs. 0.08 SD comparison, the 0.875-vs-0.9 threshold
miss) came from real commands run against real or synthetic test data
while building and testing HyCline, not from generic expectation. The
hierarchical-genomic comparison specifically was re-verified with
matched data and MCMC settings while writing this guide, rather than
quoted from an earlier, less directly comparable run. Recommendations
explicitly marked as general statistical practice, rather than a tested
finding, are labeled as such above.
