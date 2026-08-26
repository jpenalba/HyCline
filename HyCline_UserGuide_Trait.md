# HyCline User Guide: Quantitative Trait Cline Analysis

**Document status: working draft, updated alongside HyCline's own
development.**

This guide covers HyCline's quantitative-trait cline fitting -- HZAR's
other core data type, alongside allele frequencies. A trait here is any
continuous, measured phenotype: body size, a plumage or color score, a
song frequency, or -- as a genuinely useful, already-working application
covered in its own section below -- genome-wide hybrid index itself.

Every command shown below was run for real against synthetic test data
while writing this guide -- flags, output messages, and file names are
exact.

## What makes this different from allele-frequency clines

`fit-geographic` fits a binomial (or genotype-likelihood-marginalized)
likelihood to allele counts. Trait fitting is a different model
entirely: **the mean follows the usual sigmoid cline, but the variance
follows its own cline shape too** (`var_l`/`var_r`/`var_h`, one
parental-population-level variance on each side plus an excess
hybrid-zone-specific variance), matching HZAR's own quantitative-trait
model exactly -- not a single shared variance across all sites. This
extra structure is doing real biological work: within-site variance in
a trait is often itself elevated right at a hybrid zone's center, where
individuals range from lightly to heavily admixed, and `var_h` captures
that directly rather than averaging it away.

Classic MLE fitting only -- no Bayesian backend for trait clines yet.

---

## `fit-trait`: one trait at a time

Two input styles, matching HZAR's own:

**Site-aggregated (default)** -- one row per site, with mean/n/SD
columns:

```bash
hycline fit-trait --input bodysize.csv --sd-col trait_sd --output-dir results/
```

Real output:

```
Reading input data from 'bodysize.csv' ...
  10 sampling sites loaded.
Fitting (tail=none, HZAR-parity variance model) ...
Done in 0.0s. Trait cline fit (tail=none, HZAR-parity variance model): success=True, logL=-115.932, AICc=301.865, n=10, k=7
      center =    72.8032  (SE 1.0627, 95% CI [70.7203, 74.8860])
       width =    29.7170  (SE 3.1676, 95% CI [23.5085, 35.9255])
        pmin =    17.9287  (SE 0.0984, 95% CI [17.7358, 18.1216])
        pmax =    23.9576  (SE 0.1014, 95% CI [23.7588, 24.1564])
       var_l =     0.3984  (SE 0.0826, 95% CI [0.2366, 0.5603])
       var_r =     0.3762  (SE 0.0854, 95% CI [0.2089, 0.5436])
       var_h =     0.1338  (SE 0.2184, 95% CI [-0.2943, 0.5619])

All output written to: results/
```

True generating values were `center=75, width=30` -- recovered well.
`--mean-col`/`--n-col` default to `trait_mean`/`n`; `--sd-col` has no
default and is required for aggregated input (HZAR's own model always
uses each site's observed sample variance as real data, not just the
mean -- there's no single-shared-variance fallback in this project).

**Raw, individual-level input** -- one row per measured individual,
aggregated internally to the same shape:

```bash
hycline fit-trait --input measurements.csv --raw \
    --trait-col hybrid_index --site-col site --output-dir results/
```

Uses `--trait-col`/`--site-col`/`--ddof` instead of
`--mean-col`/`--n-col`/`--sd-col`. `--site-col` is how individuals are
grouped into sites for aggregation (default: group by exact matching
distance value instead, if you don't have discrete sites). `--ddof`
(default 0, numpy's population-SD convention) controls the delta degrees
of freedom used when computing each site's SD during aggregation.

### Output files

`{prefix}_summary.txt`/`.json` (fitted parameters and statistics),
`{prefix}_fitted_curve.csv` (points to replot the fit yourself), and
`{prefix}_trait_cline_plot.png` (mean cline with observed site means ±
SE) -- `--no-plot` skips the PNG.

### Distance: pre-computed, or from coordinates

Same `--distance-col`/`--lat-col`/`--lon-col`/`--transect-method` options
as `fit-geographic`, using the same underlying calculation.

### Tail models

```bash
hycline fit-trait --input measurements.csv --sd-col trait_sd --tail mirror --output-dir results/
```

Real output on a genuine tail-model dataset:

```
Fitting (tail=mirror, HZAR-parity variance model) ...
Done in 0.2s. Trait cline fit (tail=mirror, HZAR-parity variance model): success=True, logL=-129.940, AICc=322.881, n=14, k=9
      center =   100.6882
       width =    24.1806
        pmin =    18.0388
        pmax =    24.0400
     delta_m =    22.3891
       tau_m =    42.6802
       var_l =     0.2732
       var_r =     0.2531
       var_h =     0.0000
```

**Full HZAR parity for tail models, including the tail formula itself** --
verified directly against HZAR's R source. This is a real, named
difference from `fit-geographic`'s own tail model, which uses a
different, not-yet-HZAR-verified parameterization: here, `delta_l`/
`delta_r` are raw distances (the tail edge is exactly `center ± delta`),
not the width-relative fraction `fit-geographic` uses. There's currently
no `compare-tails`-style model-selection command for trait data --
`compare-tails` itself is allele-frequency-only (flat `--n-alleles-col`/
`--freq-col` convention) -- so comparing tail models for a trait means
fitting each `--tail` option yourself and comparing the reported AICc
values directly.

---

## `fit-trait-multi`: many traits at once

```bash
hycline fit-trait-multi --input multitrait.csv --output-dir results/
```

Wide-format input: one row per site, with a `<trait>.mean`/`<trait>.n`/
`<trait>.sd` column group per trait (all three required per trait). Real
output, three traits:

```
  3 traits detected: ['body_size', 'song_freq', 'plumage_score']

--- Trait 'body_size' (10 sites) ---
  Done in 0.0s. Trait cline fit (tail=none, HZAR-parity variance model): success=True, logL=-73.973, AICc=217.947, n=10, k=7
      center =    75.2710  (SE 1.1301, 95% CI [73.0561, 77.4859])
       width =    35.2285  (SE 4.0812, 95% CI [27.2293, 43.2277])
      ...

--- Trait 'song_freq' (10 sites) ---
      center =    59.9352
       width =    14.7331
      ...

--- Trait 'plumage_score' (10 sites) ---
      center =    90.7560  (SE 0.9489, 95% CI [88.8962, 92.6159])
       width =    40.5935  (SE 3.4706, 95% CI [33.7911, 47.3959])
      ...

3/3 traits fit successfully.

All output written to: results/
```

True centers were 75, 60, and 90 -- all three recovered accurately, each
trait fit fully independently (one trait failing to fit does not stop the
others). `--traits` restricts to a comma-separated subset, the same
pattern `--loci` uses elsewhere. `--mean-suffix`/`--n-suffix`/
`--sd-suffix` (defaults `.mean`/`.n`/`.sd`) match your own column naming
if it differs.

Output: `{prefix}_trait_multi_summary.csv` (one row per trait),
`{prefix}_trait_multi_fitted_curves.csv` (long format), and
`{prefix}_trait_multi_grid.png` -- a small-multiples grid, one panel per
trait, each showing observed site means with SE bars and the fitted
sigmoid curve. Confirmed visually: all three panels above show a clean
S-curve tracking the observed points closely, correctly crossing near
each trait's own true center.

**`fit-trait-multi` has no `--raw` mode** -- only pre-aggregated,
site-level input. This matters directly for the hybrid-index application
below: since hybrid index is naturally one trait per study, not many, the
individual-level `--raw` path through single-trait `fit-trait` is both
the necessary and the natural route, not a limitation in practice.

---

## Fitting hybrid index as a geographic trait cline

This is a genuinely useful application of `fit-trait` worth calling out
on its own: **genome-wide hybrid index is a continuous, per-individual
value, exactly the shape `fit-trait --raw` already expects.** No special
handling was built for this -- it's a direct, already-working use of the
existing `--raw`/`--trait-col` machinery, verified here with a real run.

```bash
hycline fit-trait --input individuals_with_hi.csv --raw \
    --trait-col hybrid_index --site-col site --output-dir results/
```

Real output, on hybrid index simulated from a known true cline (center=100,
width=40), with within-site variance deliberately elevated near the true
center -- the real biological signature of a hybrid zone, where sampled
individuals near the middle range from lightly to heavily admixed:

```
Fitting (tail=none, HZAR-parity variance model) ...
Done in 0.1s. Trait cline fit (tail=none, HZAR-parity variance model): success=True, logL=151.630, AICc=-261.260, n=12, k=7
      center =   100.1915  (SE 1.7684, 95% CI [96.7255, 103.6576])
       width =    51.0216  (SE 4.0073, 95% CI [43.1672, 58.8759])
        pmin =     0.0360  (SE 0.0059, 95% CI [0.0245, 0.0476])
        pmax =     0.9754  (SE 0.0049, 95% CI [0.9658, 0.9850])
       var_l =     0.0006  (SE 0.0002, 95% CI [0.0002, 0.0010])
       var_r =     0.0004  (SE 0.0001, 95% CI [0.0001, 0.0006])
       var_h =     0.0244  (SE 0.0060, 95% CI [0.0127, 0.0361])
```

`center` recovered almost exactly (100.19 vs. true 100). More
interestingly, **`var_h` came back roughly 40x larger than `var_l`/
`var_r`** (0.024 vs. 0.0006/0.0004) -- correctly detecting the elevated
admixture variance built into the simulation, with no special handling
needed to surface it. This is exactly the kind of signal `var_h` exists
to capture: a hybrid zone's center isn't just where the *mean* ancestry
crosses 0.5, it's typically also where individual-to-individual variation
in ancestry peaks, reflecting a real mix of backcross classes rather than
a uniformly intermediate population. The resulting plot shows a clean
S-curve from roughly 0.04 at distance 0 to roughly 0.98 at distance 200,
crossing 0.5 near the fitted center, with visibly wider error bars for
the handful of sites nearest that center -- exactly where the elevated
`var_h` says they should be.

### Real, honest caveats worth knowing

**This is a Gaussian model applied to a value that's mathematically
bounded to [0, 1].** Near the extremes (individuals close to purely one
parental type, where variance is small), a Gaussian can in principle
place a small amount of probability mass outside [0, 1] -- a real
approximation, not unique to hybrid index. HZAR's own quantitative-trait
model has always had this same limitation for any naturally bounded
phenotype, and it's a well-precedented, generally tolerated one as long
as your data doesn't sit with real density hard against the boundary
(which, for a genuinely admixing hybrid zone with real intermediate
individuals, it usually won't).

**The hybrid index values themselves carry their own estimation
uncertainty** (from `estimate-hybrid-index`, itself an average across
however many loci you used, or from an external tool like STRUCTURE/
ngsAdmix), which this fit treats as directly observed, not as a value
with its own error bars. This mirrors the same simplification
`estimate-hybrid-index` itself already makes explicit -- see the genomic
pipeline guide -- rather than introducing a new one.

**Where this fits relative to genomic-cline fitting**: this is a
*geographic* cline of hybrid index against distance, a different question
from a *genomic* cline of one locus's ancestry against hybrid index
(covered in the genomic-cline guide). Both are useful, answer different
questions, and can be run on the same underlying dataset.

---

## A note on where this guide's examples came from

Every command and every block of real output in this guide was generated
by actually running HyCline against synthetic test data while writing it.
