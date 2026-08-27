# BACKEND_TRAIT.md: quantitative trait clines (full HZAR parity)

HZAR doesn't only fit clines to molecular-marker allele frequencies --
it also fits the same sigmoid cline shape to continuous, measured
phenotypes (body size, a color/plumage score, a song frequency, a
morphological index) that vary smoothly across a hybrid zone, using a
Gaussian likelihood in place of the binomial one. This document covers
that capability: `hycline.data.schema.TraitTransect`,
`hycline.likelihoods.gaussian`, `hycline.inference.mle_trait`, and the
`fit-trait`/`fit-trait-multi` CLI commands (both still real, current
command names -- unlike the geographic-cline comparison point this
document repeatedly draws below).

**Correction, found during a later doc-consistency sweep: `fit-geographic-multi`,
referenced throughout this document as the geographic-cline comparison
point, no longer exists as a separate command** -- it was later merged
into `fit-geographic` itself, which handles any locus count with the
same command. See BACKEND_PHASE5.md section 6's correction note for the
full account. The structural comparisons below (this document was
written when `fit-geographic-multi` was the real name) are unaffected.

This is the project's second pass at this feature. The first pass built
a principled but simpler model (a single shared variance, either
estimated or given); once HZAR's actual R source was checked directly
(https://github.com/cran/hzar -- accessible via GitHub even though CRAN
itself is blocked in this project's network environment, a real, useful
finding in its own right), it turned out to genuinely diverge from
HZAR's own model in two substantive ways, not just naming. That simpler
model has been removed entirely -- this document describes the current,
sole implementation, which is full HZAR parity, including tail models.

## 1. The model, in full, verified directly against HZAR's own R source

Verified against `https://github.com/cran/hzar`'s actual R source
(`25-guassianPrep.R`, `15-modelExpressionFunctions.R`), not a principled
reimplementation from the paper or general trait-cline literature.

**The mean** follows a sigmoid cline (`muL` to `muR`), exactly as for
allele frequencies -- reusing `ClineParams`'s `pmin`/`pmax` fields to
play the role of `muL`/`muR`.

**The variance is not a single constant.** It follows a cline of its
own, sharing the same center/width/tail shape as the mean, with its own
boundary parameters (`var_l`, `var_r`) plus one extra term (`var_h`)
representing excess variance specifically in the hybrid-zone center --
a real, well-documented biological phenomenon (segregation variance
among admixed individuals, absent in either pure parental population),
not an ad hoc addition. For the no-tail case:

```
u(x) = 4*(x - center)/width
frac(x) = 1/(1+exp(-u(x)))                    -- same shape as the mean
bump(x) = 1/(2+exp(u(x))+exp(-u(x)))          -- peaks at x=center
var(x) = var_l + (var_r - var_l)*frac(x) + 4*var_h*bump(x)
```

`bump(x)` peaks at `x=center` (value 1/4, so the `4*var_h*bump(x)` term
contributes exactly `var_h` there) and decays to 0 away from the center,
so `var(x) -> var_l` far to the left and `var(x) -> var_r` far to the
right -- verified directly (`tests/test_mle_trait.py::
test_variance_cline_boundary_conditions_match_hzar`).

**The log-likelihood uses each site's observed sample variance as
data**, not just the mean -- a sufficient statistic alongside the sample
mean and n:

```
LL = -sum( n * (log(2*pi*var(x)) + (sampleVar + (sampleMean-mu(x))^2)/var(x)) ) / 2
```

This is exactly the log-likelihood of every individual raw observation
at a site (assuming individuals within a site are iid
`Normal(mu(x), var(x))`), expressed via sufficient statistics rather
than requiring the raw data itself -- confirmed two ways: algebraically,
and by direct numerical comparison against summing raw individual
observations' own log-densities, matching to `1e-6`
(`tests/test_mle_trait.py::
test_log_likelihood_matches_raw_individual_data_likelihood`).

**`transect.trait_sd` is required, not optional.** There is no "single
shared variance" fallback in this project -- HZAR's own model always
uses per-site sample variance as data. If you only have site means
(no measure of within-site spread), this model cannot be fit.

## 2. Tail models: also full HZAR parity, including a real, named structural difference from the allele-frequency tail model

HZAR's own tail construction (`pTailExpF`/`qTailExpF` for the mean,
`vTailExpF` for the variance) was re-derived from the R source and
verified directly, not approximated:

```
alpha(delta, tau, width) = exp((4*delta/width)*(tau-1)) / (1+exp(-4*delta/width))
```

**Left tail** (`x < center - delta_l`):
```
u = 4*(x-center)/width
frac(x) = alpha(delta_l, tau_l, width) * exp(u*tau_l)
mu(x) = muL + (muR-muL)*frac(x)
var(x) = var_l + (var_r-var_l)*frac(x) + 4*var_h*frac(x)*(1-frac(x))
```

**Right tail** (`x > center + delta_r`) is the mirror-image construction
(`frac_r` built the same way using `-4*(x-center)/width`, then
`frac = 1 - frac_r`), and `var_h`'s tail-region contribution is the same
`frac*(1-frac)` form -- HZAR's own construction, not this project's
invention; it has the same shape as the familiar binomial `p*(1-p)`
variance, applied to the interpolation fraction as if it were a
probability.

**Continuity at the tail/center boundary was verified directly, not
assumed** -- both the mean and variance formulas match the plain
center-region formula exactly at `x = center ± delta`, confirmed to
better than `1e-4` (`tests/test_mle_trait.py::
test_hzar_tail_formula_is_continuous_with_center_formula_at_boundary`).
This is the derivation the whole tail implementation depends on being
correct, so it's checked directly rather than only inferred from clean
-looking fitted curves.

**A real, named structural difference from `fit-geographic`/`fit-geographic-multi`'s own
allele-frequency tail model, stated plainly rather than left as an
implicit trap**: HZAR's tail edges are `center - delta_l` / `center +
delta_r` -- the delta parameters ARE the raw distance from the center to
the tail edge. The allele-frequency `cline_function`'s own (separately
parameterized, already self-documented as "several parameterizations
exist ... not verified bit-for-bit against HZAR") tail model instead
uses `center - width/2` / `center + width/2` as fixed edges, with delta
as a `[0, 1]`-bounded height fraction. These are genuinely different
parameterizations of "a tail," not a bug in either -- but `delta_l`/
`tau_l`/`delta_r`/`tau_r` mean something different in the trait-cline
context than they do for allele-frequency fitting, even though the same
`ClineParams` container is reused for convenience. Confirmed directly,
not just asserted:
`tests/test_mle_trait.py::test_tail_edge_uses_hzar_own_definition_not_width_over_2`.

**`MIRROR`** reuses `delta_l`/`tau_l` for both tail edges (a single
shared value, matching this project's own existing convention for the
allele-frequency tail model, where the same field pair plays the role of
HZAR's separate `deltaM`/`tauM`) -- not a HZAR-source-verified choice in
itself, but a reasonable, consistent convention given the rest of this
project already uses it.

## 3. Validation performed

Real recovery against known synthetic ground truth, simulated from
HZAR's own actual model (not the earlier, removed simpler model):

- **Boundary conditions**: `variance_cline`/`mean_cline` checked directly
  against the derived formula at the cline center and far from it, for
  both tail and no-tail cases.
- **Likelihood correctness**: matched exactly against summing raw
  individual observations' own log-densities (the strongest available
  check -- not just internal self-consistency).
- **Parameter recovery, no tail**: `fit_mle_trait` recovers all seven
  parameters (center, width, pmin, pmax, var_l, var_r, var_h) from data
  simulated with HZAR's actual variance-cline model, including correctly
  recovering *asymmetric* parental variances (`var_r != var_l`) -- a
  real capability the earlier, removed single-sigma model couldn't
  represent at all.
- **Parameter recovery, tail models**: `LEFT`, `BOTH`, and `MIRROR` all
  fit without error on data simulated from HZAR's own tail formula, with
  core (non-tail) parameters recovering well. `delta_l`/`tau_l`
  themselves show the same weak identifiability already documented for
  the allele-frequency case (`BACKEND_PHASE2.md` section 12.3) -- not
  re-litigated here, and not a sign of a bug in this implementation
  specifically.
- **End to end through the real CLI**: both `fit-trait` and
  `fit-trait-multi` tested via actual subprocess invocations for both
  no-tail and left-tail cases -- correct recovery, correct output file
  structure (`var_l`/`var_r`/`var_h` in the JSON/CSV output, not the
  removed `sigma`), correct plots (error bars visibly widening toward
  the cline center, directly showing the `var_h` term's effect), and
  correct handling of a trait missing its `.sd` column in a multi-trait
  run (skipped with a clear message, doesn't stop the rest).
- **A real bug found and fixed during this validation, not glossed
  over**: the CLI's shared `_tail_type()` helper didn't recognize the
  new `left`/`right` choices added for trait fitting (`fit-geographic`/`fit-geographic-multi`
  themselves only ever exposed `none`/`mirror`/`both`) -- caught
  immediately by the first tail-model CLI test (`KeyError: 'left'`),
  fixed in one line, confirmed fixed by re-running the same test.

## 4. CLI commands, and two input styles

`hycline fit-trait` (single trait) and `hycline fit-trait-multi` (many
traits from one wide-format CSV, `<trait>.mean`/`<trait>.n`/`<trait>.sd`
column groups, mirroring `fit-geographic-multi`'s own `.nSamples`/`.A` convention)
-- both classic MLE only in this pass (see section 6). `--sd-col`/
`--sd-suffix` are required for site-aggregated input, matching section
1's "no fallback" design; `fit-trait-multi` skips (not silently
downgrades) a trait missing its `.sd` column, printing a clear message
and continuing with the rest -- the same "one failure doesn't stop the
others" convention `fit-geographic-multi` already uses for allele-frequency data.
`--tail` supports `none`/`left`/`right`/`both`/`mirror` (the allele-
frequency `fit-geographic`/`fit-geographic-multi` commands themselves only expose
`none`/`mirror`/`both`, since `LEFT`/`RIGHT` alone are less commonly
requested there, but the underlying `TailType` enum and fitting
machinery support all five uniformly).

**`fit-trait --raw` accepts HZAR's other supported input style**: raw,
individual-level measurements (one row per individual) instead of pre-
aggregated site summaries -- `hycline.data.io.read_raw_trait_csv`
aggregates internally to the exact same `TraitTransect` shape (mean, n,
SD per site) the sufficient-statistics likelihood needs, grouping
individuals by `--site-col` if given or by exact matching distance
value otherwise, with the same "every individual at a site must share
the same distance" check `hycline.data.vcf`'s hard-count writers already
enforce elsewhere in this project. Verified two ways: the aggregation
itself matches hand-computed values exactly
(`tests/test_data_io.py::test_read_raw_trait_csv_aggregates_correctly_with_site_col`),
and fitting from raw input recovers the same known true cline parameters
as fitting from an equivalent pre-aggregated file
(`tests/test_cli.py::test_cli_fit_trait_raw_and_aggregated_give_consistent_results`
-- the two paths agree to floating-point-round-trip precision, not
exactly bit-for-bit, since the aggregated-file comparison target itself
passes through a CSV write/read cycle that truncates precision slightly;
this is expected CSV-format noise, not a logical discrepancy between the
two aggregation paths).

**One real, stated scope limit carried over from `read_raw_trait_csv`'s
own docstring**: the per-site SD computed during raw-data aggregation
uses `ddof=0` (numpy's default, population/biased SD) by default.
Whether HZAR's own raw-data aggregation uses `ddof=0` or `ddof=1` (R's
`var()` default) wasn't independently confirmed while building this
reader -- a real, honestly-stated gap, not a claim of exact parity the
way the core sufficient-statistics likelihood formula itself was
verified (section 1). `--ddof 1` is available if a user specifically
needs to match a `var()`-based computation done elsewhere.

## 5. `--hzar-tail`: the same verified tail construction, now also available for allele-frequency fitting

Building the trait-cline tail model against HZAR's real R source
(section 2) revealed that `fit-geographic`/`fit-geographic-multi`'s own allele-frequency tail
model is a *different* parameterization from HZAR's real one -- not a
bug in either, but a real, previously-undocumented-as-precisely
divergence. Rather than leave that as a flagged-but-unaddressed finding,
the verified HZAR tail formula was ported to
`hycline.models.cline.hzar_cline_function` -- bit-identical to
`hycline.likelihoods.gaussian.mean_cline` (the same formula, confirmed
by direct comparison, not re-derived from scratch) -- and wired into
`fit_mle`/`fit_mle_fixed_extremes`'s CLI path as an opt-in
`--hzar-tail` flag.

**Additive by design, the same pattern as `n_edge_sites`
(BACKEND_PHASE1.md section 4.2) and `pool_center`
(BACKEND_PHASE4.md section 7.4)**: `hzar_tail` defaults to `False`,
preserving `cline_function`'s original tail behavior *exactly* -- the
entire existing MLE and Manakin cross-validation test suite was re-run
unmodified after this change and confirmed to still pass, so no
previously-fitted result changes unless `--hzar-tail` is explicitly
requested. `FitResult` is self-describing (an `hzar_tail` field, and
`summary()` labels the tail type as "HZAR-parameterized" only when
active) rather than requiring the caller to track which parameterization
was used separately.

**Validated against known ground truth simulated from HZAR's own real
tail formula**, not just "the flag doesn't crash": allele-frequency data
simulated with `hzar_cline_function` recovers correctly with
`hzar_tail=True` (center, pmin, pmax all close to truth, clean
convergence) -- and, as a genuinely informative side effect of this
validation, the *mismatched* default parameterization fit to that same
data reported `success=False`, while the correctly-matched HZAR
parameterization converged cleanly on it. Confirmed end to end through
the real CLI, and confirmed zero regression across the full `fit-geographic`/
`fit-geographic-multi`/`compare-tails` CLI test group (54 tests) after the shared
`add_model_args` change this required.

**Extended to the Bayesian path** (`fit_mcmc`/`fit_mcmc_batch`, wired
through `fit-geographic`/`fit-multi --method bayesian --hzar-tail`), in a follow-up
pass -- see BACKEND_PHASE2.md section 14 for the full story, including
why the scope turned out smaller than expected (Components A/D don't
support *any* tail model, HZAR-parameterized or not) and six real
curve-export bugs found and fixed along the way, one of which (the
classic-MLE single-locus fitted-curve CSV) had been present, undetected,
since `--hzar-tail` was first shipped in this section.

**One real gap found and closed with a clear error, not a silent
ignore**: `fit_mle_fixed_extremes` (the classic-MLE path for
`--scaling fixed`) has no HZAR-parameterized tail option at all -- only
`fit_mle` (estimated extremes) does. Combining `--hzar-tail` with
`--scaling fixed`/`--pmin`/`--pmax` previously would have silently
dropped `--hzar-tail` with no warning; it now gives a clear, actionable
error instead (`tests/test_cli.py::
test_cli_fit_hzar_tail_with_fixed_scaling_gives_actionable_error`). The
Bayesian path doesn't have this gap -- `fit_mcmc` handles fixed extremes
and `hzar_tail` within the same function, not a separate one, so both
can be combined freely there.

## 6. What's not built (deliberate scope, not oversight)

- **Bayesian/hierarchical trait-cline fitting.** Classic MLE only, for
  the same reason this project's own history took Phase 1 (classic)
  before Phase 2 (Bayesian) and Phase 4 (hierarchical) for allele
  frequencies -- a deliberate, staged scope, not a gap discovered after
  the fact.
- **A genomic-cline analog** (a quantitative trait as a function of
  hybrid index rather than geographic distance). Discussed explicitly
  before this build started, and re-examined directly afterward:
  Fitzpatrick (2013)'s original genomic-cline derivation (the basis for
  this project's own genomic-cline machinery) is built specifically for
  two bounded proportions, using the identity `S = E(p_i)` across loci --
  a real mathematical dependency, not an incidental choice, and one with
  no natural analog for an arbitrary continuous trait. A literature
  search found no established precedent for extending the logit-logistic
  form to continuous traits directly; real practice instead uses
  geographic trait clines (this section's own subject), simple
  regression of trait on hybrid index, or genomic-cline analysis of the
  trait's underlying markers. Explicitly out of scope, not attempted
  here, and -- on reflection -- there's a real case for *not* building
  this speculatively (the mathematical elegance that justifies the
  original model doesn't survive the generalization, and a sigmoid isn't
  even the right null expectation for a purely additive polygenic
  trait); build only for a specific, motivated hypothesis, not by
  default.
- **`hzar_tail` support for `fit-geographic`/`fit-geographic-multi`'s Bayesian and
  hierarchical paths -- now done for the Bayesian (NUTS) path**
  (`fit_mcmc`/`fit_mcmc_batch`, `--method bayesian --hzar-tail`); see
  BACKEND_PHASE2.md section 14 for the full story, including the real
  scope-narrowing discovery that Components A and D (Phase 4's
  hierarchical geographic models) don't support *any* tail model at all,
  HZAR-parameterized or otherwise -- a separate, larger capability gap
  (not a parameterization choice) that remains genuinely unbuilt, listed
  in the README's "Not yet implemented" alongside this document.
- **A real, separate naming inconsistency found but not fixed**: for
  `TailType.MIRROR`, classic MLE's JSON output labels the shared tail
  parameter `delta_m`/`tau_m` (via `hycline.models.cline.param_names`),
  while the Bayesian path's JSON always labels it `delta_l`/`tau_l`
  (since NumPyro's model samples under that literal site name
  regardless of tail type). Found while writing MIRROR-tail regression
  tests for the Bayesian extension above; not introduced by this pass,
  and not fixed here since it's an unrelated pre-existing inconsistency,
  not part of the `hzar_tail` work itself.
- **A real gap in `fit_mle_fixed_extremes` found and closed with a clear
  error, not silently left broken**: see the note in section 5 above --
  `--hzar-tail` combined with `--scaling fixed` now errors actionably
  rather than silently ignoring the flag. Actually adding HZAR-tail
  support to fixed-extremes fitting itself remains unbuilt.
