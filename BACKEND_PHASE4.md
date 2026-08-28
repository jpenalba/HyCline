# HyCline Backend Notes: Phase 4 (Hierarchical Multi-Locus Models)

Companion to `BACKEND_PHASE1.md`/`BACKEND_PHASE2.md` (geographic clines)
and `BACKEND_PHASE3.md` (genomic clines). Phase 4 adds partial pooling
across loci -- letting loci with sparse data borrow statistical strength
from the rest of a multi-locus batch, rather than being fit fully
independently the way `fit_mcmc_batch` (Phase 2) and `fit-genomic-multi`
(Phase 3) currently do.

---

## 1. Why Phase 4 is broken into components, and what's in this pass

Phase 4 isn't one model -- it's three separable pieces:

- **Component A (done): hierarchical geographic cline model.** Partial
  pooling of cline center and width across loci sharing a transect.
  Implemented in `hycline.inference.mcmc_hierarchical`. Sections 2-5 below.
- **Component B (done): hierarchical genomic cline model.** The direct
  analog of A for `u`/`v` across loci against hybrid index. Implemented in
  `hycline.inference.mcmc_genomic_hierarchical`. Sections 6-8 below --
  this was flagged as a tracked extension in `BACKEND_PHASE3.md` section 8
  before being built here.
- **Component C (not started): coupling/concordance analysis** (Firneno
  et al. 2023), computed from A's and/or B's across-locus variance once
  they exist -- naturally sequenced after A/B since it needs their output.

These were separated because they're independently useful, independently
testable, and don't block each other -- A doesn't need B to exist, and C
needs *either* A or B, not both. Building all three at once would also
have meant validating all three at once, which given the next point was
worth avoiding.

**There is no HZAR-equivalent gold standard for any of Phase 4.** HZAR
doesn't do hierarchical pooling at all -- Phases 1/2's real-data
validation (matching HZAR's own published Manakin numbers) has no
counterpart here. So validation throughout Phase 4 means synthetic data
with a known population-level structure, the same category of check
Phase 1 used *before* real data was available, and the same category used
for Phase 3's synthetic (neutral/barrier/permeable/biased) validation.
Section 4 below covers what that means concretely for Component A, and
why "recovers the right numbers" isn't actually the most important thing
to check for a hierarchical model -- **shrinkage** is. Section 7 repeats
the same exercise for Component B.

---

## 2. The model

Extends the non-hierarchical batch model from
`hycline.inference.mcmc_numpyro.fit_mcmc_batch` (BACKEND_PHASE2.md
section 4), which gives every locus fully independent flat priors. This
model instead puts a shared population-level prior over center and (log)
width:

```
mu_center      ~ Uniform(transect extremes +/- span)
sigma_center   ~ HalfNormal(span / 2)
mu_log_width   ~ Normal(log(span / 4), 1.5)
sigma_log_width ~ HalfNormal(1.0)

for each locus j:
    center_j    ~ Normal(mu_center, sigma_center)
    log_width_j ~ Normal(mu_log_width, sigma_log_width)
    width_j = exp(log_width_j)

for each locus j, site i:
    n_focal_ji ~ Binomial(n_alleles_ji, cline(distance_i; center_j, width_j, pmin_j, pmax_j))
```

**Width uses a log-normal hierarchy, not the flat prior Phase 2's
non-hierarchical model uses.** BACKEND_PHASE2.md section 2 flagged this
tradeoff explicitly at the time ("not used here so Phase 2 posteriors are
a clean, comparable extension of Phase 1's MLE fit") -- for a
*hierarchical* model there's no such constraint to preserve, and a
log-normal population-level distribution over a strictly-positive
parameter is both the natural choice and (per that same section) likely a
better-behaved one geometrically. This is Phase 4 actually adopting the
alternative Phase 2 chose not to.

**`pmin`/`pmax` are NOT pooled across loci.** They're derived per-locus
from each locus's own data (see section 3) and held fixed, the same
"fixed scaling" convention Phase 1/2 use. Pooling them would assume
different loci's allele-frequency asymptotes are drawn from a shared
distribution, which isn't obviously true or useful the way pooling
center/width is (center/width variation across loci is the actual
biological signal of interest -- concordance or lack of it in *where* and
*how sharply* clines occur; asymptote variation is mostly about which
allele happens to be more common in which parental population, which
isn't comparable across loci in the same way).

**Missing-site masking is not supported, inherited from `fit_mcmc_batch`.**
Every locus must have data at every site in the shared `distance` vector
-- a "sparse" locus means small sample sizes at every site, not missing
data at most sites. This is documented directly in
`fit_hierarchical_batch`'s docstring and was the source of a real bug
during this component's own testing (see section 4.1) before being
understood and worked around correctly in the validation design, not the
model code.

---

## 3. A real bug found and fixed during this component's own validation

While building the synthetic validation for Component A (section 4), a
locus's hierarchical width estimate came back roughly 2x the true
generating value -- consistently, across most loci in the batch, not just
one. Before assuming this was expected "hierarchical model behavior" or
shrugging it off as sampling noise, it was traced to ground truth:

1. **First check**: was it specific to the hierarchical code? No --
   fitting the same locus's data independently with the already-validated
   Phase 1 MLE (`fit_mle_fixed_extremes`) gave the *same* inflated width.
   Ruled out a hierarchical-model-specific bug.
2. **Second check**: was the synthetic data generation itself wrong (a
   mismatch between the claimed "true" parameters and what was actually
   simulated)? No -- plotting the true generating curve against the raw
   observed frequencies at each site showed a close match; the data really
   did look like a narrower cline than what got fit.
3. **Third check (the actual answer)**: comparing the log-likelihood at
   the *true* parameters against the log-likelihood at the *fitted*
   parameters, using the same `pmin`/`pmax` the fitting procedure actually
   used, showed the true parameters had a **much higher** likelihood
   (better fit) than what the optimizer converged to. That ruled out "the
   data genuinely supports a wider cline" and pointed at the `pmin`/`pmax`
   values themselves being wrong.

The root cause: `pmin`/`pmax` were being derived from a **single** extreme
site's observed frequency (the site with minimum/maximum distance) --
matching Phase 1/2's own convention. With realistic sample sizes, a true
`pmin` around 0.05 can easily draw exactly 0/n at one site (a few-percent
event, not a rare one). When that happens, the fixed `pmin` gets locked at
0.0 instead of 0.05, and the optimizer has to substantially widen the
fitted cline to reconcile the rest of the data (which shows nonzero
frequencies near that end) with an asymptote pinned at the wrong value.

**The fix**: `_derive_pmin_pmax` in this module averages the `n_edge_sites`
(default 3) sites nearest each end of the transect -- pooling their counts
(`sum(n_focal) / sum(n_alleles)`) rather than reading a single site's
frequency -- before deriving `pmin`/`pmax`. Re-running the same synthetic
scenario with this fix brought width estimates back in line with the true
generating values (see section 4 for the actual before/after numbers).

**This fix is scoped to the hierarchical model only, in this pass.**
Phase 1's `fit_mle_fixed_extremes` and the Phase 2 CLI's
`_resolve_fixed_extremes` still use the single-site convention --
unchanged here because (a) it's what matches HZAR's own convention and
what was validated against HZAR's real published Manakin numbers (where
this fragility apparently wasn't a dominant issue, likely because that
dataset's sample sizes and pmin/pmax values didn't land in the
unlucky-zero-draw regime), and (b) changing shared Phase 1/2 code was out
of scope for a Phase 4 component. Worth revisiting as a small, targeted
fix to `hycline.inference.mle`/`hycline/cli.py` later -- flagged in
section 6 rather than done silently or left undocumented.

---

## 4. Validation: hyperparameter recovery and shrinkage

`examples/demo_hierarchical_geographic.py` is the full worked example;
`tests/test_mcmc_hierarchical.py` locks in the quantitative checks as
regression tests.

### 4.1 Setup

8 loci sharing one 40-site transect (span ~150 units), true population-level
structure `mu_center=75, sigma_center=10, mu_width=25`. Per-locus true
centers/widths drawn from that population distribution. Sample sizes
deliberately varied: locus 0 gets 4 alleles/site ("sparse"), locus 1 gets
60 alleles/site ("data-rich"), the remaining 6 loci get 20 alleles/site
("moderate").

### 4.2 Hyperparameter recovery

| Parameter | True | Recovered (posterior mean) |
|---|---|---|
| mu_center | 75.0 | 76.0 |
| sigma_center | 10.0 | 9.9 |
| mu_width (= exp(mu_log_width)) | 25.0 | 27.5 |

Center-related hyperparameters recovered essentially exactly; width
recovery is good but not as tight, consistent with width being the
harder-to-identify parameter throughout this project (the same theme
documented in BACKEND_PHASE2.md section 9's width-prior case study, now
showing up again at the population level with only 8 loci to estimate a
population SD from).

### 4.3 Shrinkage: the result that actually matters

The signature of a working hierarchical model isn't "recovers the right
numbers" (an independent per-locus fit can do that too, given enough
data) -- it's that a sparse locus's estimate gets pulled toward the
population mean more than a data-rich locus's does, because the sparse
locus has less of its own information to resist that pull.

| | independent (unpooled) | hierarchical (pooled) | shrunk toward pop. mean by |
|---|---|---|---|
| locus_0 center (sparse, n=4) | 80.82 | 80.38 | 9.3% |
| locus_1 center (rich, n=60) | 74.44 | 74.52 | 5.3% |
| locus_0 width (sparse, n=4) | 36.05 | 34.35 | 19.8% |
| locus_1 width (rich, n=60) | 32.87 | 31.95 | 17.1% |

Correctly directioned in both cases (sparse shrinks more than rich), and
the data-rich locus's hierarchical estimate stays close to its own
independent one (center moved by only 0.08 units; the model correctly
"gets out of the way" when a locus has enough of its own data rather than
imposing unwanted pooling bias).

**Reported honestly, not oversold**: the magnitude of the effect here is
modest, not dramatic -- n=4 at 40 sites (160 total binomial draws) is
sparse relative to n=60, but not so sparse that the locus's own likelihood
is negligible. A more extreme sparsity contrast would show a larger
shrinkage differential; this scenario was chosen to stay within the
model's actual constraints (no missing-site masking -- section 2) rather
than to manufacture a more dramatic-looking number. `tests/
test_mcmc_hierarchical.py::test_sparse_locus_shrinks_more_than_data_rich_locus`
locks in the *direction* of the effect (sparse shift >= rich shift, with a
small tolerance for Monte Carlo noise) rather than a specific magnitude,
since the exact magnitude is expected to vary with the specific sparsity
contrast chosen.

---

## 5. API surface

```python
from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch

result = fit_hierarchical_batch(distance, n_alleles, n_focal, locus_names=names)
print(result.summary())                # population-level + per-locus posterior means
result.population_summary()            # DataFrame: mu_center, sigma_center, mu_log_width, sigma_log_width
result.locus_summary()                 # DataFrame: one row per locus, center/width posterior mean+SD
result.idata                           # full ArviZ InferenceData, for anything else (HDIs, trace plots, ...)
```

Diagnostics: `hycline.diagnostics.convergence.check(result.idata)` works
unchanged -- it operates generically on any ArviZ `InferenceData`, so no
hierarchical-specific diagnostics module was needed.

---

## 6. Component B: the hierarchical genomic cline model

Extends the non-hierarchical genomic cline model
(`hycline.models.genomic_cline`, Phase 3 MLE-only) the same way Component
A extends the geographic one -- shared population-level priors over the
logit-logistic parameters:

```
mu_u        ~ Normal(0, mu_u_scale)
sigma_u     ~ HalfNormal(sigma_u_scale)
mu_log_v    ~ Normal(mu_log_v_loc, mu_log_v_scale)
sigma_log_v ~ HalfNormal(sigma_log_v_scale)

for each locus j:
    u_j     ~ Normal(mu_u, sigma_u)
    log_v_j ~ Normal(mu_log_v, sigma_log_v)
    v_j = exp(log_v_j)

for each locus j, individual i:
    n_focal_ji ~ Binomial(n_alleles_ji, genomic_cline(hybrid_index_i; u_j, v_j))
```

**This is the first Bayesian backend for genomic clines at all** --
`BACKEND_PHASE3.md` section 4 shipped classic MLE only and flagged a
Bayesian extension as a natural next step. Built directly as the
hierarchical version here rather than a non-hierarchical Bayesian batch
model first (the geographic side got both -- `fit_mcmc_batch` in Phase 2,
then `fit_hierarchical_batch` in Component A -- but for genomic clines the
hierarchical version was the requested and more broadly useful increment,
and there was no strong reason to build and validate a throwaway
intermediate step).

**No `pmin`/`pmax` fragility to guard against here.** The genomic cline
has no free asymptotes -- `phi(0)=0` and `phi(1)=1` always, by
construction of what hybrid index means (`BACKEND_PHASE3.md` section 2).
Component A's single-site anchor bug (section 3 above) doesn't have an
analog in this model; `_genomic_cline_core_jax` needed no equivalent
robustness fix.

**Missing-individual masking is inherited here too** (same limitation as
Component A, section 2) -- every locus must have data for every
individual in `hybrid_index`. But this constraint maps onto real genomic
data more naturally than it did for the geographic case: a "low-quality"
locus in real sequencing data is typically one with reduced per-individual
confidence at every individual (e.g. `n_alleles=1` from a single
low-coverage read instead of a confident diploid call, or a locus
genotyped via genotype likelihood rather than a hard call), not one
missing from most individuals outright. Section 7's validation uses
exactly this shape: a sparse locus with `n_alleles=1` everywhere, not
`n_alleles=0` at most individuals.

## 7. Component B validation: hyperparameter recovery and shrinkage

`examples/demo_hierarchical_genomic.py` is the full worked example;
`tests/test_mcmc_genomic_hierarchical.py` locks in the quantitative
checks. Same structure as Component A's validation (section 4).

### 7.1 Setup

8 loci, 120 individuals with hybrid index drawn `Uniform(0.02, 0.98)`.
True population-level structure: `mu_u=0, sigma_u=1.0, mu_v=1.0,
sigma_log_v=0.5` -- **centered at the neutral null** (`u=0, v=1`
reproduces `phi(h)=h`, a locus that tracks genome-wide admixture exactly
with no bias and no rate difference; see `BACKEND_PHASE3.md` section 3).
This is a deliberate, biologically motivated choice, not an arbitrary
default -- see section 8's discussion of why `mu_u=0`/`mu_log_v=0` is the
meaningful reference point here, tying back to the same tension-zone
theory that motivates Component A.

Per-locus true `u`/`v` drawn from that population distribution. Sample
sizes deliberately varied to represent two realistic real-data scenarios:
locus 0 gets `n_alleles=1` at every individual (a single low-confidence
SNP call), locus 1 gets `n_alleles=20` (a resolved multi-SNP ancestry
block, pooling evidence from several tightly linked markers into one
"locus" -- a realistic aggregation when multiple SNPs within a haplotype
block are treated as ancestry-informative markers for the same underlying
introgression signal), the remaining 6 loci get `n_alleles=2` (a normal
diploid call).

### 7.2 Hyperparameter recovery

With only 8 loci, a hierarchical fit's estimate of the population mean
tracks the *sample* mean of the 8 true per-locus values used in this
particular simulation, not necessarily the *theoretical* population value
(0 loci drawn from a distribution rarely have a sample mean of exactly 0)
-- this is correct behavior, not bias, and `tests/
test_mcmc_genomic_hierarchical.py::
test_hierarchical_genomic_fit_recovers_population_level_hyperparameters`
checks against the sample mean accordingly, the same way Component A's
equivalent test does.

| Parameter | Recovered (posterior mean) | Sample mean of the 8 true values |
|---|---|---|
| mu_u | 0.255 | ~0.25 |
| mu_log_v | -0.446 | (matches within MC noise) |
| sigma_u | 1.05 | true sigma_u = 1.0 |
| sigma_log_v | 0.33 | true sigma_log_v = 0.5 |

### 7.3 Shrinkage

| | independent | hierarchical | shrunk toward population mean by |
|---|---|---|---|
| locus_0 u (sparse, n=1) | -0.201 | -0.189 | 2.6% |
| locus_1 u (rich, n=20) | -0.854 | -0.854 | 0.0% |
| locus_0 v (sparse, n=1) | 0.458 | 0.529 | 38.9% |
| locus_1 v (rich, n=20) | 0.488 | 0.495 | 4.0% |

A cleaner, more decisive shrinkage differential than Component A's
geographic validation showed (section 4.3's 9-20% range) -- particularly
for `v` (38.9% vs. 4.0%). This isn't a coincidence: the sparsity contrast
here (`n=1` vs. `n=20`, a 20x difference in per-individual information) is
much larger than what Component A's validation could use (`n=4` vs.
`n=60` at 40 *sites*, but the *aggregate* information gap was smaller
relative to typical geographic transect sizes). Finding a contrast this
size took two failed attempts first (an `n=1` vs. `n=2` contrast showed
almost no shrinkage signal at all -- too little information difference to
matter; a version with 400 individuals also washed out the effect, since
even `n=1` per individual across 400 individuals is already a lot of
aggregate data) -- reported here rather than silently discarded, since the
failed attempts are informative about how much sparsity contrast is
actually needed to see this effect clearly, not just that the effect
exists in principle.

### 7.4 The `pool_center` extension: pooling `logit(center)` directly, matching Firneno et al.'s own quantity

Found while verifying Component C's theta calibration against Firneno et
al.'s actual source code and Supplemental Material (section 9.3): the
original Component B model (above) hierarchically pools the raw bias
parameter `u`, giving a genuine, jointly-estimated `sigma_u` -- a real
hyperparameter, the same *kind* of quantity Firneno et al.'s own Stan
model produces, confirmed directly from their
`FitClineModel_Agalychnis.R`. But `sigma_u` isn't the quantity their
paper's own coupling-coefficient methodology actually uses --
`logit(center) = u/v` is (section 9.3), and pooling `u` and pooling `u/v`
are genuinely different statistical claims (whether the raw bias is
what's exchangeable across loci, versus whether the hybrid-index-scale
cline center is).

**Design: additive, not a silent change to the default.** Changing
`fit_hierarchical_genomic_batch`'s default parameterization would have
risked silently altering the already-published-comparable BGC/gghybrid
cross-validation (section 7's own validation, and BACKEND_PHASE3.md
section 7's cross-validation against real Italian sparrow data) without
direct evidence the change was warranted there. Instead, a new
`pool_center: bool = False` parameter was added -- default `False`
reproduces the original model *exactly* (confirmed by re-running the
existing Component B test suite unchanged, all 6 tests passing with no
modification); `pool_center=True` switches to sampling
`mu_center, sigma_center` (a hierarchical Normal prior on
`logit(center)`, centered at 0 -- center=0.5 on the hybrid-index scale,
the same neutral-null convention section 8 already establishes for
`mu_u`) and `mu_log_v, sigma_log_v` (unchanged from before), deriving
`u = v * logit(center)` per posterior draw for use in the same
likelihood function. This derivation is algebraically exact by
construction (`logit(c) = u/v` implies `u = v * logit(c)`), verified
directly rather than just asserted:
`tests/test_mcmc_genomic_hierarchical.py::
test_pool_center_jax_and_numpy_reparameterization_are_exact` confirms
`genomic_cline_function`'s output is bit-identical (`atol=1e-12`)
whether `u` is supplied directly or derived this way.

**`HierarchicalGenomicFitResult` is self-describing, not configured
separately from its own posterior.** `.pool_center` is a property that
checks for `"logit_center"` in the posterior rather than being stored as
a separate flag the caller has to track alongside the result --
`population_summary()`/`locus_summary()` both branch on this internally,
so calling code doesn't need an `if pool_center:` of its own.
`locus_summary()` always includes `u_mean`/`v_mean` regardless of which
parameterization was used (derived from `logit_center * v` when
`pool_center=True`), so downstream code that only cares about u/v (e.g.
the CLI's fitted-curve plotting, which constructs `GenomicClineParams`
directly from these columns) works unchanged either way -- only
`population_summary()`'s column set actually differs
(`sigma_center`/`mu_center` vs. `sigma_u`/`mu_u`).

**Validated against known ground truth, not just "the model runs."** A
synthetic dataset (15 loci, 80 individuals, true `sigma_center=0.8`,
true `sigma_log_v=0.3`) was fit with `pool_center=True`: clean
convergence (R-hat=1.00 across all four hyperparameters), both true SDs
within or very close to their 89% credible intervals (appropriately
imprecise given only 15 loci, not a red flag), and per-locus recovery of
the individual true centers/v values with correlation 0.965/0.940
respectively across all 15 loci --
`tests/test_mcmc_genomic_hierarchical.py::
test_pool_center_true_recovers_known_sigma_center`, `::
test_pool_center_true_locus_summary_recovers_known_per_locus_values`.

**CLI wiring**: `hycline fit-hierarchical-genomic --pool-center` (plus
`--mu-center-scale`/`--sigma-center-scale` for the new hyperpriors),
tested end to end including confirming the convergence-diagnostic tuning
hint correctly switches to suggesting `--sigma-center-scale` rather than
`--sigma-u-scale` when this mode is active
(`tests/test_cli.py::test_cli_fit_hierarchical_genomic_pool_center_runs_and_reports_sigma_center`,
`::test_cli_fit_hierarchical_genomic_default_still_reports_sigma_u` for
the default-unchanged guarantee at the CLI level too).

**What this doesn't close**: Component C's `genomic_cline_variance` still
computes sigma_c from per-locus point estimates by default -- it hasn't
been changed to automatically detect and prefer a `pool_center=True`
fit's own `sigma_center` when one is available, since that would require
Component C to accept a `HierarchicalGenomicFitResult` object directly
rather than plain arrays, a real (if likely small) API change not
attempted in this pass. For now, a user with a `pool_center=True` fit
should read `sigma_center` from `population_summary()` directly and pass
it to `estimate_coupling_coefficient` themselves, rather than calling
`genomic_cline_variance` at all for that quantity.

## 8. Why the neutral null (`u=0, v=1`) is the biologically meaningful center, not an arbitrary default

Worth stating explicitly, since it's easy to read `mu_u ~ Normal(0, ...)`
as just a generic weakly-informative prior choice: it's centered at 0
specifically because `u=0, v=1` is the neutral expectation under standard
hybrid-zone theory (Barton & Hewitt 1985; see the README's "Why
hierarchical clines" section for the fuller citation list and the
geographic-cline version of this same argument). A locus with `u=0, v=1`
tracks genome-wide admixture exactly -- it's doing nothing special
relative to the rest of the genome. Departures from that (`sigma_u`,
`sigma_log_v` estimated as meaningfully greater than zero; individual loci
with `v` well above or below 1, or `u` well away from 0) are the
biological signal genomic cline analysis exists to detect: barrier loci
resistant to introgression (`v > 1`), loci with elevated introgression
(`v < 1`), or loci with asymmetric ancestry bias (`u != 0`) -- candidates
for linkage to genes under selection or involved in reproductive
isolation. Centering the hyperpriors at the neutral point means the
model's own population-level posterior for `sigma_u`/`sigma_log_v`
directly answers "how much does the genome actually vary in introgression
around that neutral expectation" -- which is the central question
`bgchm` (Gompert, DeRaad & Buerkle 2024) was built to quantify, and the
core motivation for building Component B at all rather than continuing to
rely on `fit-genomic-multi`'s fully independent per-locus fits (Phase 3).

---

---

## 9. Component C: coupling/concordance analysis (Firneno et al. 2023)

`hycline.analysis.coupling` quantifies how concordant fitted cline
parameters are across a set of loci, and connects that to Barton's (1983)
theoretical coupling coefficient theta = s/r (the ratio of total
selection to recombination among barrier loci), following Firneno,
Semenov, Dopman, Taylor, Larson & Gompert (2023, "Quantitative Analyses
of Coupling in Hybrid Zones," *Cold Spring Harb Perspect Biol* 15:a041434).

### 9.1 The underlying theory, briefly

Barton (1983) showed that when multiple "barrier loci" (loci under
selection against hybrids) are linked, linkage disequilibrium among them
causes selection to spill over from each locus onto the others -- so the
*total* selection a locus experiences (call it s*) can exceed its own
direct selection coefficient s. Whether this spillover is strong enough
to make barrier loci behave as an effectively coupled, genome-wide unit
(rather than a set of independent, locus-specific barriers) is governed
by theta = s/r: above theta = 1, total selection approaches the sum
across all L barrier loci; well below theta = 1, loci behave largely
independently. This is the same tension-zone/barrier-locus framework
underlying this project's "Why hierarchical clines?" README section --
Component C is the piece that turns "loci should be concordant under a
shared barrier" from a qualitative expectation into a number.

Firneno et al.'s key insight, confirmed via extensive individual-based
simulation (`dfuse`, Lindtke & Buerkle 2015): **theta is not directly
observable, but the *variance* in fitted cline parameters across loci
is** -- and that variance declines as theta increases (more coupling ->
more concordant clines -> lower across-locus variance). They calibrate
this relationship by simulating hybrid zones across a range of known
theta values and fitting a regression relating the observed cline
parameter variance to the known theta.

### 9.2 What HyCline implements directly (on firm ground)

The raw variance metrics themselves are a direct, faithful implementation
of Firneno et al.'s definitions, not an approximation:

- **Geographic**: `geographic_cline_variance(locus_widths)` computes the
  across-locus mean and SD of cline *slope*, `beta = 4/width` -- exactly
  the quantity Firneno et al. fit a linear model to logit allele
  frequency to estimate ("The slope in this model is expected to be four
  times the allele frequency gradient... at the center of the cline").
  The `4/width` relationship holds automatically here because this
  project's own cline parameterization (`hycline.models.cline`) already
  uses the same width convention.
- **Genomic**: `genomic_cline_variance(locus_u, locus_v)` computes the
  across-locus SD of `logit(center) = u/v` and `log(v)` -- Firneno et
  al.'s `sigma_c` and `sigma_v`. Note carefully: `logit(center)` is
  **not** the same as this project's own `u` parameter
  (`hycline.models.genomic_cline`) -- it's Bailey (2022)/gghybrid's
  reparameterization `logit(c) = u/v`, a nonlinear function of both `u`
  and `v` together. Computing `SD(u)` instead of `SD(u/v)` would be a
  real error, not a minor approximation -- the function computes `u/v`
  explicitly for exactly this reason.

Both of these are useful and fully defensible outputs on their own, with
no calibration or extrapolation risk: lower variance always means more
concordant clines, regardless of whether the theta conversion below is
trusted.

### 9.3 What's approximated: the theta calibration

Converting sigma_c/sigma_v into an actual theta *estimate* requires
Firneno et al.'s simulation-calibrated regression. Their full simulation
output (thousands of simulated hybrid zones across theta = 0.05 to 2)
isn't reproducible from the published paper alone -- their code lives at
https://github.com/zgompert/ClineCoupling, and while that repository and
the paper's own Supplemental Material were both checked directly (not
just the paper's main text) as part of confirming this section, neither
publishes the ~1140 raw simulated datasets or the unstandardized
regression coefficients fitted to them -- only standardized coefficients
(which need the training data's own summary statistics to convert back
to raw scale) and the simulation-derived r² values.

Instead, `estimate_coupling_coefficient` **reverse-engineers their
applied model from its own outputs**: the paper's Table 1 reports
(sigma_c, sigma_v, theta) for 25 real hybrid zones (31 rows counting
multiple transects), and states these theta values were computed by
"estimat[ing] coupling coefficients based on the values of sigma_c and
sigma_v and the parameterized linear model... from the simulated data
sets." In other words, Table 1's theta column is already the *output* of
their fitted linear model applied to real data -- so fitting
`theta ~ sigma_c + sigma_v + sigma_c:sigma_v` to these 31 (input, output)
pairs isn't really a fresh 31-point regression, it's solving for what
their original coefficients must have been. `FIRNENO_2023_TABLE1` in the
module holds this reference data -- **confirmed digit-for-digit against
the actual published Table 1** (not the informal summary table in the
GitHub repository's README, which has at least one clear transcription
error -- Neotoma listed there as theta=64 rather than the published
theta=0.64; `FIRNENO_2023_TABLE1` was built from the correct source and
was never affected by that error).

**Two things that were previously open questions are now resolved, not
just argued:**

- **The model form is exactly linear-with-interaction, not polynomial.**
  The paper's main text is explicit for the specific model actually used
  to generate Table 1's theta column: "standardized regression
  coefficients from linear regression: sigma_c = -0.0167 ... sigma_v =
  -0.24 ... sigma_c:sigma_v = 0.097 ... model P-value < 0.0001" -- the
  "polynomial regression" mentioned elsewhere in the paper (and in
  Tables S1/S2 of the Supplemental Material) describes a *different,
  exploratory* analysis used only for the smooth-curve figures (2 and 3)
  showing how sigma_c/sigma_v trend with theta in the simulations, not
  the model applied to produce Table 1's actual theta estimates. This
  removes what was previously flagged as a real source of uncertainty.
- **The log/logit scale convention matches exactly.** The paper states
  plainly: "we measure the variance in clines here as the standard
  deviation (SD) in log(vi) and logit(ci)" -- natural log for the slope
  parameter, logit for center. `genomic_cline_variance` uses `np.log`
  (natural log, not log10) for `sd_log_v` and the logit-scale `u/v` for
  `sd_logit_center` -- confirmed to match exactly, resolving an earlier
  uncertainty about whether the slope variability should be computed on
  a log10 or natural-log scale (an informally-commented analysis script
  in the GitHub repository used ambiguous language that could have been
  read either way; the published paper's own methods text is
  unambiguous, and this implementation was already on the correct side
  of that ambiguity).

**New, genuinely independent validation, not just a stronger restatement
of the same 31-point fit**: the paper's Supplemental Material includes
Table S4, reporting sigma_c/sigma_v/theta separately for autosomes and
sex chromosomes for 8 of the 25 hybrid zones (16 data points) -- real,
published values that were never part of the 31-point calibration set
above. `estimate_coupling_coefficient` was checked against all 16
without any refitting: mean absolute error 0.0081, max 0.0171 (`tests/
test_coupling.py::test_calibration_holds_out_of_sample_on_table_s4`).
One point (Poecile's sex-chromosome subset, sigma_c=2.58) falls well
outside the 31-point calibration's bounding box (max sigma_c=1.73 in
that set) and still predicted almost exactly (0.819 vs. published 0.82)
-- a genuine, if single, data point of evidence that extrapolation
outside the bounding box isn't automatically unreliable, though this
shouldn't be over-generalized from one case; `within_calibration_range`
still flags such points as outside the checked region rather than
silently treating them as equally trustworthy.

**Caveats that remain genuinely open, stated as plainly as the
resolutions above:**
- The exact unstandardized coefficients from Firneno et al.'s *original*
  regression (fit on their ~1140 simulated datasets) are still not
  available -- what's implemented here is confirmed to reproduce their
  *applied model's outputs* faithfully (both on the original 31-point
  set and on 16 new held-out points), which is the practically relevant
  guarantee, but is a different claim from "these are their exact
  original coefficients."
- The calibration reflects one specific simulation study's assumptions
  (a one-dimensional stepping-stone model, 110 demes, migration rate 0.1
  or 0.2, multiplicative underdominance, a single 1-Morgan chromosome) --
  applying it to a system with very different demography or genetic
  architecture is a real extrapolation, which Firneno et al. themselves
  caution about repeatedly for their own estimates ("we caution that our
  estimates of theta for the empirical data sets should be taken as
  estimates given the specific simulation conditions and linear model
  used here").
- **A genuine, documented limitation found while building this**: the
  linear+interaction form can produce a negative theta (theoretically
  impossible, since theta = s/r is a ratio of two positive quantities)
  for some (sigma_c, sigma_v) combinations *even within the bounding box*
  of the 31 reference points -- see section 10 ("Known rough edges").
  This isn't hidden;
  `GenomicClineVariance.summary()` flags it explicitly, and
  `within_calibration_range` reports whether the query point falls inside
  the reference data's bounding box at all (a coarser but always-computable
  check than the true, irregular convex hull).
- **The ancestry-informative-marker threshold this project's own
  `filter-aims` command defaults to (0.9/0.1) is stricter than what
  Firneno et al. actually used (allele frequency difference > 0.3
  between parentals, confirmed directly from both their Supplemental
  Material and the `FitClineModel_*.R` scripts)** -- worth knowing if
  comparing directly against their per-organism loci counts, though not
  a correctness issue for either tool (each is a defensible, stated
  default, not a hidden inconsistency, and `filter-aims`'s threshold was
  set to match this project's own stated preference, not to reproduce
  Firneno et al.'s exact pipeline).
- **sigma_c/sigma_v as computed by `genomic_cline_variance` (the SD of
  already-extracted per-locus point estimates) is methodologically close
  to, but not identical to, Firneno et al.'s own quantity** (a
  hierarchical Bayesian model's own posterior-estimated SD hyperparameter,
  fit jointly across all loci via Stan/HMC -- confirmed directly from
  `FitClineModel_Agalychnis.R`). For **sigma_v specifically, a better,
  directly-available alternative already exists**: Component B's own
  hierarchical model (`hycline.inference.mcmc_genomic_hierarchical.
  fit_hierarchical_genomic_batch`) already jointly estimates a genuine
  `sigma_log_v` hyperparameter as part of its own fit (reported in
  `population_summary()`), the same kind of quantity Firneno et al.'s
  Stan model produces -- feeding that directly into
  `estimate_coupling_coefficient` (rather than computing
  `np.std(v_mean, ddof=1)` from Component B's per-locus point estimates
  first) is the more faithful choice going forward, not a hypothetical
  future improvement. **For sigma_c, no equivalent exists yet**:
  Component B's model hierarchically pools raw `u` (giving `sigma_u`,
  reported in the same `population_summary()`), not `logit(center) =
  u/v` -- the actual quantity Firneno et al.'s definition needs -- so
  `genomic_cline_variance`'s SD-of-point-estimates approach remains the
  practical option for sigma_c specifically, with the same caveat that
  shrinkage in the per-locus `u`/`v` posterior means has already partly
  pulled them toward the group center before this SD is taken, unlike a
  jointly-estimated hyperparameter that accounts for that directly.
  Extending Component B's model to hierarchically pool `logit(u/v)`
  itself (closing this gap for sigma_c too) is a real, scoped follow-up,
  not attempted in this pass -- see the README's "Not yet implemented"
  list.

### 9.4 A real limitation found while validating this component

`examples/demo_coupling.py`'s "discordant" genomic-cline scenario
(sigma_c=0.797, sigma_v=0.575, both near but within the reference data's
observed ranges of [0.29, 1.73] and [0.08, 0.56]) produces theta = -0.49
-- a nonsensical negative coupling coefficient. This isn't a bug in the
arithmetic; it's a real property of fitting a linear+interaction surface
to 31 irregularly-scattered points and then querying it somewhere the
data doesn't densely support, even within the coordinate-wise bounding
box (the true region where 31 scattered 2D points provide good support is
their convex hull, which is smaller and more irregular than the box each
coordinate's min/max implies). Rather than silently clip negative results
to zero (which would hide the extrapolation problem) or fail loudly with
an exception (which would make the raw sigma_c/sigma_v values -- which
remain valid and useful -- inaccessible), `GenomicClineVariance.summary()`
prints an explicit warning and the raw variance metrics stay available on
the object regardless of whether theta comes back sensible. Concretely:
the demo's "concordant" scenario (sigma_c=0.105, sigma_v=0.064, *below*
the reference data's minimum sigma_c) still returns a sensible theta =
1.845 -- so the failure mode isn't simply "extrapolation always breaks
it," it's specific to certain regions of the (sigma_c, sigma_v) plane,
which is exactly why a bounding-box check alone (`within_calibration_range`)
is necessary but not sufficient, and why the negative-value check exists
as a second, independent safeguard.

---

## 10. Known rough edges / follow-ups

**Component A:**
- **No missing-site masking** (section 2) -- every locus needs data
  (however sparse) at every shared site. Also inherited by, and a real
  limitation of, Phase 2's `fit_mcmc_batch`.
- **The single-site `pmin`/`pmax` fragility (section 3) is only fixed
  here**, not in Phase 1's `fit_mle_fixed_extremes` or the Phase 2 CLI's
  `_resolve_fixed_extremes`. A small, targeted fix applying the same
  multi-site averaging there is a reasonable next step, not done in this
  pass to keep Component A's scope focused.
- **`pmin`/`pmax` not pooled** (section 2) -- a deliberate scope choice,
  not obviously wrong, but worth revisiting if a real use case wants it.

**Component B:**
- **No missing-individual masking** (section 6), same underlying
  limitation as Component A's missing-site masking, though a better fit
  for realistic genomic data shapes (section 6 explains why).
- **This is the first Bayesian genomic cline backend at all** -- there's
  no non-hierarchical Bayesian genomic cline model to compare against the
  way Component A could be checked against Phase 2's `fit_mcmc_batch`
  (independent-priors batch fitting). Cross-checking `fit_hierarchical_
  genomic_batch`'s per-locus posteriors against Phase 3's classic MLE
  (`fit_genomic_cline_mle`) on the same locus's data, as section 7's
  validation does for the shrinkage comparison, is the closest available
  substitute.

**Component C:**
- **The theta calibration is a reverse-engineered approximation** of
  Firneno et al. (2023)'s applied model, not their original code -- see
  section 9.3 for the full methodology and confidence level.
- **The linear+interaction calibration can return a negative (theoretically
  impossible) theta** for some (sigma_c, sigma_v) combinations, even
  within the reference data's bounding box -- flagged explicitly in
  `GenomicClineVariance.summary()` rather than silently clipped or hidden;
  see section 9.4 for a worked real example found while validating this.
- **No geographic theta estimate.** Firneno et al. calibrated theta
  against real empirical data using genomic clines only ("We thus focus
  exclusively on genomic clines for natural hybrid zones"), so
  `geographic_cline_variance` deliberately provides only the raw slope
  variance metrics (mean, SD, CV), with no attempt at a theta conversion
  for geographic clines -- there's no real-data calibration to base one on.
- **No support for the sex-chromosome/autosome split** Firneno et al.
  themselves report as biologically informative (sex chromosomes showing
  higher coupling than autosomes in most of their 8 tested species pairs)
  -- computing coupling separately for pre-labeled subsets of loci would
  need to be done by calling `genomic_cline_variance`/
  `geographic_cline_variance` separately on each subset, not a single
  built-in split.

**All components:**
- **CLI wiring is now done for all of Components A-E** (`hycline
  fit-hierarchical`/`fit-hierarchical-genomic` -- section 13;
  `coupling-geographic`/`coupling-genomic`, `fit-hierarchical-geographic-gl`/
  `fit-hierarchical-genomic-gl` -- section 14). Only Phase 5's own
  standalone single-locus GL fitting (`fit_mle_geographic_gl`/
  `fit_mle_genomic_gl`) remains Python-API-only.

---

## 11. Component D: hierarchical geographic clines with genotype-likelihood marginalization

Combines Component A (partial pooling across loci) with Phase 5
(marginalizing over genotype uncertainty from sequencing data) into one
model, implemented in `hycline.inference.mcmc_hierarchical_gl`. This
section covers why the combination needed real new work, not just wiring
two existing pieces together, and the actual empirical question it exists
to answer.

### 11.1 Why this needed genuine new work

Naively, one might expect "just feed genotype-likelihood data into
Component A" to work. It doesn't, for two concrete reasons:

- **A new JAX-native likelihood was needed.** Component A's model
  (`_hierarchical_batch_model`) is built around the binomial likelihood
  over hard allele counts. Phase 5's genotype-likelihood marginalization
  (`hycline.likelihoods.genotype_likelihood`) is implemented in NumPy/
  SciPy, not JAX, so it isn't traceable inside a NumPyro model -- the same
  reason `mcmc_numpyro`/`mcmc_genomic_hierarchical` each carry their own
  JAX-native reimplementation of their respective NumPy cline cores.
  `_marginal_log_likelihood_jax` in this module is that reimplementation
  for the genotype-likelihood marginalization specifically, verified to
  match the NumPy/SciPy version exactly (`tests/test_mcmc_hierarchical_gl.py::
  test_jax_marginal_likelihood_matches_numpy_version`, `atol=1e-10`).
- **`pmin`/`pmax` derivation needed a different approach.** Component A's
  `_derive_pmin_pmax` averages *hard allele counts* over several sites
  near each end of the transect (the fix for its own single-site fragility
  bug, section 3). There's no hard count to average when the input is
  genotype likelihoods. `_derive_pmin_pmax_gl` generalizes the same idea:
  treat each individual's genotype likelihood as if it were a genotype
  posterior under a flat prior (via softmax normalization), take that
  individual's expected focal-allele dosage, and average over several
  individuals near each end -- the natural genotype-likelihood analog of
  "average several observations instead of anchoring on one."
  `tests/test_mcmc_hierarchical_gl.py::
  test_derive_pmin_pmax_gl_hard_call_matches_direct_computation` confirms
  this reduces to the expected hard-count-based answer for deterministic
  (hard-called) genotype likelihoods.

The `numpyro.factor` mechanism (rather than `numpyro.sample(..., obs=...)`)
is used to inject the marginal log-likelihood into the model, since it
isn't naturally expressible as a single distribution's `.log_prob()` call
-- this is NumPyro's standard, documented way to add an arbitrary
log-density term.

### 11.2 The actual question: does GL-awareness add value on top of pooling?

This is the open empirical question Component D exists to answer, not
assumed going in. Hierarchical pooling already has its own stabilizing
effect on sparse loci (Component A, section 4) -- so it's a real
possibility that pooling alone already fixes most of what Phase 5 fixes
for *independent* per-locus fits, making genotype-likelihood-awareness
redundant once you're pooling anyway.

**Setup**: 6 loci sharing 100 individuals, one locus (`locus_0`) at
sequencing depth 2 (sparse), the rest at depth 8 (moderate). Two fitting
approaches compared on the same simulated data each time: (a) the full
GL-aware hierarchical model (`fit_hierarchical_batch_gl`), and (b) "hard-
call then hierarchically pool" -- take each individual's maximum-
likelihood genotype and feed the resulting hard counts to Component A's
existing `fit_hierarchical_batch`. 12 independent replicate simulations,
each refit both ways (`examples/demo_hierarchical_gl.py`;
`tests/test_mcmc_hierarchical_gl.py::
test_gl_aware_hierarchical_beats_hard_call_hierarchical_for_sparse_locus`
locks in the same result on one representative seed as a fast regression
check).

**Result: pooling alone is not enough.**

| | GL-aware | Hard-call |
|---|---|---|
| `locus_0` (sparse) center coverage, 95% CI | 100% (12/12) | 75% (9/12) |
| `locus_0` (sparse) width coverage, 95% CI | 100% (12/12) | 8.3% (1/12) |
| mean center CI width | 13.91 | 13.77 |
| mean width CI width | 32.53 | 31.18 |

### 11.3 The mechanism is more consequential than expected

The original expectation, based on Phase 5's own finding (section 3.2 of
`BACKEND_PHASE5.md`: hard-calling is *overconfident*, meaning narrower
CIs that miss the true value more often), was that hard-call's CIs here
would be visibly narrower than GL-aware's. **They aren't** -- the mean CI
widths above are nearly identical between the two methods (13.91 vs.
13.77 for center; 32.53 vs. 31.18 for width). So the coverage failure
isn't primarily an overconfidence-width story here; it's a **location**
story: hard-call's intervals are about the same size, but centered in the
wrong place.

Investigating why (checking actual posterior means, not just coverage
booleans, the same "look at the real numbers before concluding" discipline
used for Component A's own bug investigation in section 3): on one
representative replicate, `locus_0`'s true width is 31.1. The GL-aware
fit's posterior mean for that locus is 29.3 (close). The hard-call fit's
posterior mean is 70.3 -- badly inflated, consistent with the same kind of
width non-identifiability documented throughout this project when data is
sparse (section 3; `BACKEND_PHASE2.md` section 9; `BACKEND_PHASE1.md`
section 5). **Critically, the population-level `mu_width` hyperparameter
is also distorted**: 33.4 under the GL-aware fit (close to the true
population mean of 25) versus 60.7 under the hard-call fit -- more than
double. The hard-called sparse locus's bad estimate isn't just wrong on
its own; the hierarchical model treats it as real signal and pools it
into the *shared* hyperparameter that every other locus's prior is centered
on. This is visible directly in the per-replicate `mu_width` trace in
`examples/demo_hierarchical_gl.py`'s output and right-hand plot panel.

**This reframes the motivation for Component D**, compared to the original
framing going into this build. It's not just "does GL-awareness help
stabilize one sparse locus's own estimate" (Phase 5's finding, still true
and still relevant) -- it's "does a poorly-handled sparse locus contaminate
the shared population-level estimate that *every* locus in the batch gets
pooled toward." The latter is a real risk specific to combining
hierarchical pooling with imperfect low-coverage data, not present at all
in independent per-locus fitting (where each locus's error stays
contained to itself), and arguably the more important reason to use
genotype-likelihood-aware fitting whenever hierarchical pooling and
low-coverage sequencing data are both in play.

### 11.4 Known rough edges specific to Component D

- **Genomic-cline analog not built.** `mcmc_genomic_hierarchical` (Component
  B) has no genotype-likelihood-aware counterpart yet -- the same
  combination (Component B + Phase 5, for genomic clines against hybrid
  index rather than geographic clines against distance) is structurally
  a very similar build to this one, not started in this pass.
- **The 12-replicate study (section 11.2) is a meaningful sample, not an
  exhaustive one.** All 12 replicates used the same underlying scenario
  (one depth-2 locus among five depth-8 loci); the result's robustness to
  different depth contrasts, numbers of loci, or population-level
  variance settings hasn't been separately characterized.
- **No missing-individual masking**, inherited from both Component A and
  Phase 5's own limitations (section 2, and `BACKEND_PHASE5.md` section 5).
- **No CLI wiring, Bayesian-vs-classic choice, or tail-model support**
  beyond what Component A and Phase 5 individually already lack (see
  section 10 and `BACKEND_PHASE5.md` section 5) -- this component doesn't
  add any new scope-limiting decisions beyond what its two parent pieces
  already have.

---

## 12. Component E: hierarchical genomic clines with genotype-likelihood marginalization

The genomic-cline analog of Component D -- combines Component B (partial
pooling across loci for `u`/`v`) with Phase 5's genotype-likelihood
marginalization, implemented in `hycline.inference.mcmc_genomic_hierarchical_gl`.

**This section is worth reading carefully precisely because its finding
does not just repeat Component D's.** The natural (and, going in,
expected) assumption was that the same dramatic result -- hard-calling a
sparse locus badly distorts both its own estimate and the shared
population-level hyperparameter -- would reproduce here. It does not, at
least not at the same magnitude or by the same mechanism, and that
difference is itself informative about what actually drove Component D's
result.

### 12.1 Reuse, not reimplementation

Very little new code was needed, by design: `_marginal_log_likelihood_jax`
(the JAX-native genotype-likelihood marginalization) is fully generic --
it takes a predicted probability and three genotype-likelihood arrays,
with no assumption about where that probability came from -- so it's
imported directly from `mcmc_hierarchical_gl` (Component D) rather than
duplicated. `_genomic_cline_core_jax` is imported directly from
`mcmc_genomic_hierarchical` (Component B), already validated there. The
only genuinely new code is the hierarchical model definition itself
(population-level `mu_u`/`sigma_u`/`mu_log_v`/`sigma_log_v` hyperpriors,
per-locus `u`/`log_v` drawn from them, `numpyro.factor` injecting the
marginal likelihood) -- a small, low-risk amount of new surface area,
which is reflected in this component needing no debugging iteration the
way Component D's `pmin`/`pmax` derivation did.

**No `pmin`/`pmax` derivation needed at all**, unlike Component D --
consistent with Component B's own simplicity relative to Component A
(section 6: "no `pmin`/`pmax` fragility to guard against here," since the
genomic cline has no free asymptotes). This removes an entire class of
possible instability that Component D had to work around.

### 12.2 Moderate depth contrast: no detectable difference

Using the same setup as Component D's own validation (6 loci, one at
depth 2, the rest at depth 8, 100 shared individuals), across 8
independent replicates:

| | GL-aware | Hard-call |
|---|---|---|
| Sparse-locus `v` coverage, 95% CI | 100% (8/8) | 100% (8/8) |
| Sparse-locus `u` coverage, 95% CI | 87.5% (7/8) | 87.5% (7/8) |

**Identical performance.** At this depth contrast -- the exact setup that
produced Component D's headline 100% vs. 8.3% coverage gap for geographic
clines -- genotype-likelihood-awareness makes no detectable difference
for genomic clines under hierarchical pooling. This was checked directly,
not assumed to transfer, and it's reported as the real result rather than
adjusted or reframed to manufacture a difference.

### 12.3 A modest difference does emerge at a more extreme contrast, via a different mechanism

Pushing to a more extreme setup (depth 1 vs. depth 20, fewer individuals
-- 60 instead of 100, to make the contrast more decisive) across 8 further
replicates:

| | GL-aware | Hard-call |
|---|---|---|
| Sparse-locus `v` coverage, 95% CI | 100% (8/8) | 87.5% (7/8) |
| Mean absolute error, point estimate | 0.410 | 0.176 |

A real difference now appears in coverage, but the mechanism is
**different from Component D's**, not a smaller version of the same
thing. Component D's problem was fundamentally about *location*: hard-call's
point estimates were badly wrong (width inflated 2x+), and that wrongness
propagated into the shared population-level hyperparameter. Here, the
opposite pattern shows up in the point estimates: **hard-call's mean
absolute error (0.176) is smaller than GL-aware's (0.410)** -- hard-call
is, on average, closer to the true value in raw point-estimate terms.
GL-aware still achieves better *coverage* despite this, which means the
mechanism here is closer to Phase 5's own standalone finding
(`BACKEND_PHASE5.md` section 3.2: hard-calling is overconfident, with
intervals too narrow relative to their actual error) than to Component
D's contamination story. Hard-call's interval is tight around a
reasonably good point estimate, but not tight in a way that's honest
about how much that estimate could plausibly be wrong given only one read
per individual at that locus.

### 12.4 Why the two components' findings genuinely differ

Two structural differences between the geographic and genomic cline
models plausibly explain this, though neither is proven here beyond the
empirical pattern itself:

- **No `pmin`/`pmax` fragility source** (section 12.1) -- Component D's
  dramatic finding traced directly to a specific failure mode (a single
  noisy read distorting a derived asymptote, then propagating into a
  wildly wrong width estimate). The genomic cline model has no analogous
  free parameter for sparse data to corrupt.
- **Hybrid index always spans the full [0,1] range by construction**,
  unlike geographic distance, where a sparse or clustered sampling
  pattern can leave the cline's actual transition zone poorly covered.
  Every simulated genomic-cline dataset here has individuals spread
  across the entire hybrid-index range regardless of depth, which likely
  gives the rate parameter `v` inherently better-conditioned identifiability
  even when each individual's genotype is uncertain.

**Practical implication**: the geographic and genomic recommendations are
not the same. For geographic clines with hierarchical pooling, genotype-
likelihood-awareness (Component D) is worth using whenever any locus in
the batch has meaningfully lower depth than the others -- the risk of
population-level contamination is real even at a moderate contrast. For
genomic clines with hierarchical pooling, Component E's benefit is real
but more marginal, and mainly relevant at genuinely extreme depth
contrasts (closer to 1x than 2x) -- pooling's own stabilizing effect
already does most of the work at moderate contrasts.

### 12.5 API surface

```python
from hycline.inference.mcmc_genomic_hierarchical_gl import fit_hierarchical_genomic_batch_gl

fit = fit_hierarchical_genomic_batch_gl(hybrid_index, log_gl0, log_gl1, log_gl2)
print(fit.summary())               # same HierarchicalGenomicFitResult as Component B
```

Returns the exact same `HierarchicalGenomicFitResult` type as Component
B's `fit_hierarchical_genomic_batch` (`.population_summary()`,
`.locus_summary()`), for the same reason Component D reuses Component A's
`HierarchicalFitResult` -- the two functions differ only in likelihood,
not output shape.

### 12.6 Known rough edges

- **The 8-replicate studies (sections 12.2-12.3) are meaningful samples,
  not exhaustive ones** -- same caveat as Component D's own 12-replicate
  study (section 11.4). The specific depth values and number of
  individuals used here were chosen to demonstrate the two regimes
  clearly, not swept systematically.
- **The structural explanation in section 12.4 is a plausible account of
  *why* the components differ, not independently verified** -- e.g. by
  constructing a geographic scenario with denser, evenly-distributed
  sampling (removing the "sparse local coverage of the transition zone"
  factor) to see if that alone closes the gap with the genomic case.
  Flagged as a natural follow-up, not done here.
- No CLI wiring, Bayesian-vs-classic choice, or tail-model considerations
  beyond what Components B and Phase 5 individually already lack (genomic
  clines have no tail-model concept at all, so that particular limitation
  doesn't apply here the way it does for Component D).

---

## 13. CLI wiring for Components A and B

`hycline fit-hierarchical` and `hycline fit-hierarchical-genomic` expose
Components A and B (sections 2 and 6) from the command line, using the
same wide-format multi-locus CSV input `fit-geographic-multi`/`fit-genomic-multi`
already accept -- no new input format, no new readers.

### 13.1 The one real design decision: enforcing the dense-data requirement

Components A/B's batch-fitting functions require every locus to share an
identical covariate vector (`distance` or `hybrid_index`) -- no per-locus
missing-site/individual masking (sections 2 and 6). The existing
multi-locus readers (`read_multi_locus_wide_csv`/
`read_genomic_cline_multi_locus_wide_csv`), by contrast, were built for
`fit-geographic-multi`/`fit-genomic-multi`'s independent-per-locus fitting and
deliberately drop missing rows *per locus* -- appropriate there, but it
means two loci in the same file can silently end up with different
site/individual subsets if the file has any missing values anywhere.

Rather than let a shape mismatch surface as a cryptic error deep inside
NumPyro (or, worse, silently misalign data if array lengths happened to
coincide), `_stack_for_hierarchical` in `hycline/cli.py` checks that every
locus's covariate vector matches exactly before fitting, and fails with an
actionable message naming the offending locus and suggesting
`fit-geographic-multi`/`fit-genomic-multi` as the fallback for genuinely incomplete
data. Verified directly: `tests/test_cli.py::
test_cli_fit_hierarchical_gives_actionable_error_on_missing_data` and its
genomic-cline counterpart construct a real incomplete CSV, confirming the
command exits cleanly (code 2) with the intended message rather than a
traceback.

### 13.2 A narrower `--bayes-args` helper, not the shared one

`fit-geographic`/`fit-geographic-multi`'s existing `add_bayes_args` helper includes `--backend`
(choices `numpyro`/`pymc`) and `--width-upper-bound`, neither of which
apply here: Components A/B are NumPyro-only, and use population-level
log-normal hyperpriors for width/rate rather than a single flat prior
with an upper bound. Reusing that helper would have shown the user
options that don't work, or work differently than their names imply.
`add_hierarchical_bayes_args` is a smaller, accurate sibling (just
`--num-warmup`/`--num-samples`/`--num-chains`/`--seed`), with the
actual prior controls (`--sigma-center-scale`, `--mu-log-width-loc`,
etc.) exposed separately as a "hierarchical model" argument group.

### 13.3 Output files and diagnostics

Both commands write: a population-level summary CSV, a per-locus summary
CSV (posterior means/SDs for center/width or u/v -- `fit-hierarchical-geographic`
also includes each locus's fixed `pmin`/`pmax`, since those aren't part
of `HierarchicalFitResult.locus_summary()` itself and are recomputed
independently in the CLI via the same `_derive_pmin_pmax` the fit itself
used, for consistency), a long-format fitted-curves CSV, a small-multiples
grid plot (new `plot_hierarchical_geographic_grid`/
`plot_hierarchical_genomic_grid` functions in `hycline/viz/plotting.py`,
built following the same pattern as the existing non-hierarchical grid
plots but using each locus's posterior-mean point estimate rather than a
per-locus independent fit), and a JSON summary including the
`hycline.diagnostics.convergence.check()` report. **Convergence warnings
are surfaced to the user, not swallowed** -- confirmed on real data: a CLI
run on the bundled Manakin multi-locus dataset (`examples/
manakin_multilocus.csv`) with default priors reproduced the exact kind of
identifiability trouble already documented for this dataset elsewhere in
the project (`BACKEND_PHASE2.md` section 9's width-prior case study) --
R-hat 1.08, 252 divergences -- and the CLI printed the resulting
`ISSUES FLAGGED` warnings and a suggestion to tighten the priors, rather
than reporting a falsely-clean result.

### 13.4 Real-data validation

Both commands were run end-to-end on real bundled example data, not just
synthetic test fixtures. `fit-hierarchical-geographic` on the Manakin multi-locus
dataset (7 loci, 11 sites) produced a grid plot with sensible,
recognizable cline shapes for most loci (sharp transitions around
distance ~200-210 for `ak2`/`l5`/`pscn3`/`mtDNA`; flat near-zero curves
for `gsr`/`pgm2`, consistent with those loci carrying little clinal
signal at these frequencies; a shallower, messier fit for `ada`,
consistent with that locus's own documented identifiability struggles
elsewhere in this project). `fit-hierarchical-genomic` on the Phase 3
synthetic multi-locus dataset (`examples/synthetic_genomic_clines.csv`)
correctly recovered the known true parameter values baked into that
dataset when it was first created (`barrier1` v=5.06 vs. true ~5;
`permeable1` v=0.23 vs. true ~0.3; `biased1` u=2.66 vs. true ~2.5) with
clean convergence (R-hat=1.00, zero divergences).

### 13.5 Known rough edges

- Only Components A and B are wired; Component C (coupling analysis),
  Components D/E (GL-aware hierarchical fitting), and Phase 5's
  standalone GL fitting remain Python-API-only.
- No option to feed a coupling-analysis summary directly from these
  commands' output -- computing `hycline.analysis.coupling.
  geographic_cline_variance`/`genomic_cline_variance` on the resulting
  locus-summary CSV is currently a separate manual (Python) step.
- The dense-data requirement (section 13.1) means these commands are less
  forgiving of messy real-world data than `fit-geographic-multi`/`fit-genomic-multi`
  -- appropriate given the underlying model's real constraint, but worth
  knowing before reaching for `fit-hierarchical-geographic` on a dataset with
  scattered missingness.

---

## 14. CLI wiring for Components C, D, and E

**Correction, found during a later doc-consistency sweep: `fit-
hierarchical-geographic-gl`/`fit-hierarchical-genomic-gl` don't exist in
the CLI as it stands today** -- checked directly against the current
parser (only `fit-hierarchical-geographic`/`fit-hierarchical-genomic`
exist, auto-detecting genotype-likelihood vs. hard-count input from the
column names present, same command either way). Read this section's own
command names as describing this feature's original CLI surface before
that later merge -- the findings/validation numbers are unaffected. See
BACKEND_PHASE5.md section 6's own correction note for the parallel issue
on the independent-fitting path, and README.md's "Validation: CLI wiring
for Components C, D, and E" section for the corrected, current-command-
name version of the validation narrative below.

`hycline coupling-geographic`/`coupling-genomic` (Component C),
`hycline fit-hierarchical-geographic-gl` (Component D), and `hycline
fit-hierarchical-genomic-gl` (Component E) close the remaining CLI gap
flagged in section 13's rough edges.

### 14.1 Component C: analysis, not fitting -- a different command shape

Unlike every other subcommand so far, `coupling-geographic`/
`coupling-genomic` don't fit anything -- they compute a derived statistic
from *already-fitted* per-locus results. The natural input is therefore
not raw data but a locus-summary CSV, and the natural source of that CSV
is the output of commands already built: `fit-geographic-multi`/`fit-genomic-multi`
(independent per-locus fits) or `fit-hierarchical-geographic`/`fit-hierarchical-genomic`
(recommended, per section 9's own docstring on why hierarchical estimates
are the intended input for this analysis).

**A real inconsistency in existing column-naming conventions had to be
handled, not just documented around.** `fit-geographic-multi`'s classic-method
summary CSV uses a bare `width` column; its bayesian-method summary uses
`width_mean`; `fit-hierarchical-geographic`'s summary always uses `width_mean`. Same
pattern for `u`/`v` vs. `u_mean`/`v_mean` on the genomic side. Rather than
force the user to know which convention their particular upstream command
used, `--width-col`/`--u-col`/`--v-col` default to auto-detecting the
right column (`width_mean` then `width`; `u_mean`/`v_mean` then `u`/`v`),
falling back to a clear, actionable error listing the columns actually
present if neither candidate is found. This is directly exercised, not
just described: `tests/test_cli.py::test_cli_coupling_genomic_no_theta_flag`
uses a summary with bare `u`/`v` columns specifically to confirm the
fallback path (not just the preferred `_mean` path) works.

### 14.2 Components D/E needed new CSV-reading infrastructure Phase 5 itself never built

Phase 5 shipped `fit_mle_geographic_gl`/`fit_mle_genomic_gl` as pure
Python functions, taking `GenotypeLikelihoodTransect`/
`GenotypeLikelihoodLocusData` objects the caller had to construct
directly -- no CSV reader existed for genotype-likelihood data at all,
single-locus or multi-locus, at the time Components D/E were built. Wiring
Components D/E into the CLI therefore required building that
infrastructure first: `read_gl_multi_locus_wide_csv` (geographic) and
`read_genomic_gl_multi_locus_wide_csv` (genomic) in `hycline/data/io.py`,
mirroring the existing multi-locus readers' structure (auto-detected
locus columns via a suffix convention, per-locus missing-row dropping,
optional individual labels) but for three GL columns per locus
(`.gl0`/`.gl1`/`.gl2` by default) instead of a count/frequency pair.

**A `--phred` flag handles the format genotype likelihoods actually come
in from real tools.** GATK, bcftools, and ANGSD all report genotype
likelihoods Phred-scaled (`PL = -10*log10(likelihood)`), not as the
natural-log values this project's internal math uses
(`BACKEND_PHASE5.md` section 2). Rather than require users to convert
their own files before running HyCline, `--phred` converts via
`hycline.data.genotype_likelihood.phred_to_log_likelihood` at read time.
Verified as an exact round-trip, not just "runs without erroring":
`tests/test_data_io.py::test_read_gl_multi_locus_wide_csv_phred_conversion`
and a standalone check comparing a Phred-encoded file's read-in values
against the same data's natural-log-encoded equivalent confirm bit-for-bit
agreement (`atol=1e-6`).

### 14.3 Reusing the dense-data enforcement, not rebuilding it

Components D/E's CLI commands call the *exact same*
`_stack_for_hierarchical` helper Components A/B's commands use (section
13.1), just with `count_attrs=["log_gl0", "log_gl1", "log_gl2"]` instead
of `["n_alleles", "n_focal"]`. Since that helper only cares about the
covariate vector (`distance`/`hybrid_index`) matching across loci, not
what the per-locus payload actually is, no new enforcement logic was
needed -- the same actionable "locus X has different data than locus Y"
error, and the same `fit-geographic-multi`/`fit-genomic-multi`-as-fallback
suggestion, apply here unchanged. Verified directly:
`tests/test_cli.py::test_cli_fit_hierarchical_gl_actionable_error_on_missing_data`.

### 14.4 Grid plots: genotype-likelihood data has no single "observed frequency" to show

Components A/B's grid plots (section 13.3) scatter each individual's
*observed* frequency against distance/hybrid index -- straightforward
when the input is a hard count. Genotype-likelihood data has no single
observed value; each individual is a probability distribution over three
genotypes, not one number. Components D/E's grid plots use each
individual's *most likely* genotype (argmax of the GL triple) purely as a
display convenience for marker placement -- explicitly noted in the CLI's
own output message ("marker positions use each individual's most-likely
genotype, for display only") so this isn't mistaken for what the fit
itself used, which is the full marginalization over all three genotypes,
not the single most likely one.

### 14.5 Real-data validation

Both `fit-hierarchical-geographic-gl` and `fit-hierarchical-genomic-gl` were run
end-to-end on freshly generated synthetic multi-locus genotype-likelihood
data (not just unit-test fixtures) with a genuine sparse/rich depth
contrast among loci, matching the same style of validation Components D/E
themselves used (sections 11 and 12). Both produced sensible, correctly-
shaped grid plots (clear sigmoid transitions for geographic loci;
recognizable barrier/neutral genomic cline shapes, with `barrierX`'s
fitted `v` -- 5.3 -- landing close to its true generating value of 4.0
despite a fast, low-iteration test configuration). `coupling-geographic`/
`coupling-genomic` were run directly on the real output files from
section 13.4's Manakin and Phase-3-synthetic CLI runs, not synthetic
locus-summary fixtures -- confirming the whole pipeline (fit
hierarchically -> analyze coupling) works end to end on real data, not
just on data shaped to fit each command's test in isolation. One of these
real runs also surfaced `coupling-genomic`'s own negative-theta warning
(section 9.4) organically, on real fitted data rather than a
constructed edge case -- the synthetic dataset's genuine spread of
barrier/permeable/biased loci happened to fall outside the theta
calibration's supported range, and the CLI correctly reported that rather
than a misleadingly precise number.

### 14.6 Known rough edges

- **CLI for Phase 5's own standalone single-locus/independent-multi-locus
  GL fitting is now done** -- `hycline fit-gl`/`fit-genomic-gl`/
  `fit-geographic-gl-multi`/`fit-genomic-gl-multi`. See `BACKEND_PHASE5.md` section 6
  for the full design, including a real design decision the independent
  path needed that the hierarchical path (Components D/E) didn't --
  independent multi-locus fitting has no dense-data requirement, so its
  grid plots needed genuinely new plotting functions rather than reusing
  Components D/E's own.
- `coupling-geographic`/`coupling-genomic` don't fit anything themselves,
  so there's no option to run a fit and immediately analyze coupling in
  one command -- it's a deliberate two-step pipeline (fit, then analyze
  the output), not a missing convenience feature, but worth knowing if a
  single combined command would be more useful for a particular workflow.
- The auto-detected column-naming fallback (section 14.1) only recognizes
  the specific conventions HyCline's own commands produce
  (`width`/`width_mean`, `u`/`u_mean`, `v`/`v_mean`) -- a summary CSV from
  a different source with different column names still needs
  `--width-col`/`--u-col`/`--v-col` specified explicitly.

## 15. Candidate-locus flagging: adapting Fitzpatrick (2013)'s multivariate outlier method to both cline types

README.md's Tier 1 backlog specified this feature in full before any code
was written -- the design discussion below covers what that spec-writing
process settled during implementation, not the original design (see
README.md's "What's implemented" section for the shipped feature itself,
and the git history for the spec discussion this was built from).

### 15.1 The statistic: Fitzpatrick's own D^2/chi-squared approach, read directly

Fitzpatrick (2013)'s "Outlier detection" section, read directly from the
PDF rather than paraphrased, describes exactly the approach implemented
here: "Assuming the joint distribution of parameter estimates among loci
is multivariate normal, the squared Mahalanobis distance D^2 of each
locus is expected to be distributed as a chi-squared random variable with
degrees of freedom equal to the number of parameters... For automated
outlier detection, I used the Bonferroni-adjusted critical P-value."

This project's own hierarchical models (Components A/B) don't jointly
estimate a center-rate covariance -- `population_summary()` reports
`mu_center`/`sigma_center` and `mu_log_width`/`sigma_log_width` (or
`mu_u`/`sigma_u`, `mu_log_v`/`sigma_log_v`) as independent priors, no
covariance term -- so the honest statistic available is a
**diagonal-covariance** D^2 = z_center^2 + z_rate^2, not a full bivariate
Mahalanobis distance with an estimated off-diagonal term the way
Fitzpatrick's own (built on raw per-locus MLE point estimates, where the
covariance can be estimated empirically across loci) does. This is a
known, stated simplification -- a true correlation between center and
rate deviations (plausible for some mechanisms, e.g. a Dobzhansky-Muller
case) is invisible to this statistic. Extending Components A/B to
estimate a joint covariance is out of scope for this pass.

`_flag_from_z_scores` in `hycline/analysis/coupling.py` is the shared
statistical core both `flag_geographic_candidates` and
`flag_genomic_candidates` call into -- one implementation of the D^2/
chi-squared logic, not two parallel copies.

### 15.2 Two Bonferroni-correction decisions, made explicit rather than left implicit

Two design choices weren't fully pinned down by Fitzpatrick's own text
and needed a decision during implementation:

1. **The joint D^2 test's correction factor is `alpha/n_loci`** --
   directly matching Fitzpatrick's own stated convention ("Bonferroni-
   adjusted critical P <= 0.05/100 = 0.0005" for his 100-locus
   simulations, i.e. one correction factor for n tests, one per locus).
2. **The marginal per-parameter z-tests use the same `alpha/n_loci`
   correction, not a further split across the two parameters**
   (`alpha/(2*n_loci)`). Fitzpatrick's own text doesn't specify a
   marginal single-parameter test at all -- his method flags a locus
   outlier via D^2 alone, with no described procedure for attributing
   *which* dimension drove it. The per-parameter marginal test (used
   here to decide `center_outlier`/`rate_outlier`, and hence category)
   is this project's own addition, needed because Fitzpatrick's four-way
   category split (steep-only / displaced-only / shallow-only / both)
   requires knowing which dimension(s) triggered, not just whether the
   locus is an outlier at all. Using the same per-locus correction factor
   for both the joint and marginal tests (rather than inventing a
   further split) was chosen as the simpler, more defensible reading --
   stated here explicitly as a real design decision, not a hidden
   assumption, so a future reader can revisit it if it turns out to
   matter empirically.

A consequence of this choice, verified directly
(`tests/test_coupling.py::test_flag_candidates_ambiguous_via_joint_but_
not_marginal_deviation`): a locus can be a joint D^2 outlier without
either marginal z individually crossing its own (numerically identical,
but squared-vs-unsquared) threshold -- e.g. z_center=z_rate=2.0 gives
D^2=8.0, above the 2-locus chi-squared(df=2) critical value of ~7.38, but
neither |z|=2.0 exceeds the 2-locus per-parameter critical value of
~2.24 individually. This is handled explicitly, not silently: such a
locus lands in `ambiguous` rather than falling through uncategorized,
since it's a real joint deviation the two single-parameter tests aren't
positioned to explain on their own.

### 15.3 The sign-convention problem: "steep" means opposite things for the two cline types

Geographic and genomic clines define steepness in *opposite* directions
relative to the parameter each type's hierarchical model actually pools:
narrower **width** is steeper (an inverse relationship) for geographic
clines, while larger **v** is steeper (a direct relationship) for genomic
clines. `_flag_from_z_scores` is written direction-agnostic -- it expects
`z_rate_steep_positive` already signed so positive always means
steeper -- and each caller owns its own sign convention:
`flag_geographic_candidates` computes `z_log_width` and then negates it
(`z_rate_steep_positive = -z_log_width`); `flag_genomic_candidates` uses
`z_log_v` directly with no sign flip. Getting this backwards for either
caller would silently swap `restricted_gene_flow`/`elevated_gene_flow`
for that cline type specifically -- caught during implementation by
building explicit sign-convention tests for both cline types
(`test_flag_geographic_candidates_categorizes_each_pattern`,
`test_flag_genomic_candidates_categorizes_each_pattern`) before trusting
either direction, rather than after a bug report.

### 15.4 Category naming: a real correction made during design discussion, not after the fact

The original spec (README.md's Tier 1 backlog, written before any code)
used three category names of inconsistent register: `barrier` (an
interpretive/functional label -- "resists gene flow") and
`adaptive_introgression` (a named evolutionary process) alongside
`elevated_gene_flow` (a bare pattern description) for the fourth. This
inconsistency was caught and fixed during a design discussion *before*
implementation began, not discovered as a bug afterward: all three
outlier categories now name the observed statistical pattern only
(`restricted_gene_flow`, `elevated_gene_flow`, `asymmetric_gene_flow`),
with which evolutionary mechanism each pattern is *most consistent with*
moved entirely into documentation (this module's own section-header
comment above `LocusCandidateFlag`, and `CANDIDATE_FLAGGING_CAVEAT`) --
never in the machine-readable `category` value itself.

The motivating concern, stated directly rather than just asserted: a
column of confident-sounding labels like `barrier`/`adaptive_introgression`
in a CSV is exactly the kind of thing that gets copied into a results
table or a paper's methods section without the caveat text that should
accompany it. `restricted_gene_flow`/`asymmetric_gene_flow` read as data describing a
pattern, not as a claim about mechanism -- a small but real difference in
how a downstream reader is likely to treat the column.

`asymmetric_gene_flow` in particular turned out to be the *more* standard
term in the hybrid-zone literature than the name it replaced
(`adaptive_introgression`) -- "asymmetric introgression"/"asymmetric gene
flow" is already established phrasing that doesn't assert *why* the flow
is asymmetric (selection-driven, differential hybrid fitness, or
incomplete lineage sorting mimicking it -- exactly the ambiguity the
caveat text has to spell out regardless of the name chosen), so this
rename is not a precision-for-safety tradeoff -- it is, if anything,
closer to correct on both counts.

### 15.5 Output: a CSV comment header, not just a printed caveat

The original spec called for "a printed caveat block... the same way the
near-invariant-locus warning is surfaced directly rather than left to a
docs page." Implemented as specified, and extended one step further: the
same caveat text (`CANDIDATE_FLAGGING_CAVEAT`) is also written as `#`-
prefixed comment lines at the top of `<prefix>_candidates.csv` itself,
before the header row -- not just printed to stdout. This directly
addresses the same copy-paste risk discussed in 15.4: stdout is easy to
lose track of once a CSV is passed downstream (into R, Excel, a
notebook), while a comment header travels with the file. Verified this
doesn't break normal CSV consumption:
`tests/test_cli.py::test_cli_flag_candidates_geographic_end_to_end` reads
the file back with `pandas.read_csv(..., comment="#")` and confirms the
caveat lines are skipped cleanly and the data rows parse correctly.

### 15.6 Verification

- Statistical-core unit tests confirm all four categories trigger
  correctly on constructed synthetic cases, the ambiguous-via-joint-only
  case (15.2), rejection of degenerate inputs (fewer than 2 loci,
  non-positive width/v/sigma, mismatched array lengths), and that the
  Bonferroni correction actually gets stricter as `n_loci` grows (a fixed
  marginal deviation that's flagged at n=2 loci is not flagged at
  n=500) -- `tests/test_coupling.py`.
- `flag_genomic_candidates`'s `pool_center=True` and `pool_center=False`
  paths are confirmed to compute the same `z_center` quantity (and hence
  the same categories) in the special case `v=1` everywhere, where
  `u/v == u` exactly -- a direct check that the two code paths reduce to
  the same computation where they should.
- Two full CLI commands (`flag-candidates-geographic`,
  `flag-candidates-genomic`), including `--input-dir`/`--input-prefix`
  convenience-path derivation, `--pool-center` wiring (and its
  actionable error when a population summary written *without*
  `--pool-center` is read back *with* it -- the required population-
  summary rows are simply absent, and the existing
  `_load_population_summary_csv` validation catches this the same way it
  catches any other missing-row case, not a special-cased check), and
  the CSV comment-header caveat mechanism -- `tests/test_cli.py`.
- **Real CLI-to-CLI composition, not just hand-built fixture CSVs**:
  `test_cli_flag_candidates_geographic_composes_with_real_fit_hierarchical_
  output` runs the actual `fit-hierarchical-geographic` command against
  the real, published Manakin multi-locus dataset, then feeds its own
  output files straight into `flag-candidates-geographic` via
  `--input-dir`/`--input-prefix` -- confirming the two commands' output/
  input CSV conventions actually agree in practice, not just on paper.
- Full test suite reconciled against `--collect-only`'s total after this
  feature landed: 455 tests collected, all passing in batched runs.

## 16. Two small, independent items surfaced by section 15's work but not part of it

README.md's Tier 1 backlog named these as "two genuinely separate items
surfaced during that same design process" while section 15 was being
specified -- both from Bailey (2024), read directly, not the paper
section 15 itself is built on (Fitzpatrick 2013).

### 16.1 A real documentation inaccuracy found and corrected before implementing either item

Before starting, README.md's own spec text for the first item described
it as "distinct from the existing near-invariant-locus warning" in
`hycline/inference/mle.py`. Checked directly rather than trusted (per
this project's own standing practice, and the recent precedent of a
fabricated `arviz==1.2.0` version claim in section 8): `grep -rn
"invariant" hycline/` across the entire package, including tests and
examples, returns nothing except two unrelated statistical usages of the
word "invariant" in `examples/validate_gghybrid_italy.py`. No such
warning function exists, and none ever did in this codebase's git
history searched for the string. An earlier session's spec-writing
apparently assumed this warning existed (plausibly because a *pattern*
like it was being planned elsewhere, or by analogy with the T-cline
work's own boundary warning in `hycline/inference/mle.py`) without
verifying against the actual code. Corrected directly in README.md
rather than left standing, matching this project's stated principle that
a wrong claim gets corrected in place, not quietly dropped.

### 16.2 `_warn_if_hybrid_index_distribution_uneven`: read Bailey's own numbers directly, not the secondhand summary already in this project's docs

Bailey (2024)'s section 2.3.4, read directly from the PDF rather than
trusted from `references/README.md`'s own prior summary of it (which
turned out accurate, but was checked rather than assumed): simulated
1000 diploid loci twice, once at `v=1` (null) and once at `v=30` (very
steep), both with evenly spaced true cline centres across [0.0001,
0.9999] but an uneven distribution of 1000 simulated individuals'
hybrid indices. Their finding, precisely: "For null cline steepness
(v=1) clustering of hybrid indices had little impact on prediction
accuracy [...] but for a steep cline (v=30) both parameters were
affected. Estimated cline centres were highly clustered and v estimates
were less accurate and with higher uncertainty for loci whose steep
clines were centred in gaps in hybrid index." Their own Figure S3
identifies the affected loci as those passing `p<=.05` and `v>20`.

**Implementation** (`hycline.inference.mcmc_genomic_hierarchical.
_warn_if_hybrid_index_distribution_uneven`, called automatically at the
end of `fit_hierarchical_genomic_batch`): for each locus whose fitted
posterior-mean `v > 20` (Bailey's own reported threshold, not invented),
compute the distance from its fitted center (`sigmoid(u/v)`, on the
natural [0,1] hybrid-index scale) to the nearest actually-sampled
hybrid-index value. If that distance exceeds `0.05` (5% of the full
[0,1] range), warn, naming the specific locus/loci, their fitted `v`/
center, and how far the nearest sample actually is.

**Two design decisions Bailey's own paper doesn't specify, made
explicit rather than left implicit:**
- The `v>20` steepness threshold is Bailey's own number, directly
  matched. The `0.05` gap-size threshold is **not** from the paper --
  Bailey illustrates the effect visually (Figure 7, Figure S3) rather
  than stating a specific gap-size criterion, so there is no "correct"
  published number to match. Stated as this project's own choice,
  clearly labeled as such in the function's own docstring, not presented
  as if it came from the paper.
- The check runs **post-fit**, not as a pre-fit data-quality check on
  `hybrid_index` alone. This is deliberate: Bailey's own finding is
  specifically about steep clines centered in a hybrid-index gap, not
  about uneven sampling in the abstract (their own null-cline simulation
  showed the same uneven sampling was largely harmless) -- so the check
  genuinely needs each locus's own fitted `v` and center, which don't
  exist until after the hierarchical fit completes. A pre-fit check could
  only ever flag "your sampling has gaps," which isn't the actual,
  narrower finding worth surfacing.

**Verification, not just a mocked unit test:** confirmed the helper
function fires correctly in isolation (a locus with `v>20` centered
exactly in a constructed gap fires; the same gap with a well-sampled
center, or with `v<=20`, does not; evenly-sampled hybrid index never
fires) *and* end-to-end through the real `fit_hierarchical_genomic_batch`
pipeline on genuinely simulated data with a real gap. The end-to-end
case needed real iteration, not just writing the test once: a
`true_v=30` locus's posterior-mean `v` shrank to ~16-18 under the
default hierarchical prior (`sigma_log_v_scale=1.0`) with only 2 loci to
estimate population-level spread from -- below the warning's own
threshold, a real demonstration that hierarchical partial pooling can
itself mask exactly the kind of extreme, poorly-identified locus this
warning exists to catch. Fixed for the test by loosening
`sigma_log_v_scale` to 3.0 and raising the true `v` to 80, at which point
the fitted locus's own posterior for `v` became extremely large and
poorly constrained (posterior mean in the millions) -- not a test
artifact, but a genuine demonstration of Bailey's underlying point: a
steep cline with no individuals sampled near its transition has a
genuinely unidentifiable rate parameter.

### 16.3 `null_model_aicc_delta`: the simpler comparison Bailey's own richer one was scoped down from

Bailey (2024) section 3.2, also read directly: "Running the full model
and both one-parameter models (with each of v or centre fixed at the
null value) and ranking loci by delta-AIC or delta-waic for each
parameter comparison is preferred [to raw p-values], with waic
performing best in simulations." This describes three models per locus
(full; centre-fixed-at-null; rate-fixed-at-null) and two separate
delta-AIC/delta-WAIC comparisons, telling you not just *whether* a locus
deviates from null but *which* parameter drives it.

**What's implemented is the simpler comparison README.md's Tier 1 backlog
actually scoped**: one full-vs-fully-null (`u=0`, `v=1`, zero free
parameters) delta-AICc, not Bailey's three-model decomposition, and not
delta-WAIC (which needs Bayesian posterior samples this classic-MLE-only
path doesn't produce). This is a stated scoping choice specific to
`fit-genomic`/`fit-genomic`'s GL variant being classic-MLE, not a
misreading of the paper -- the richer three-model version, and a
delta-WAIC equivalent for `fit-hierarchical-genomic`'s Bayesian path, are
real, distinct further extensions, not implemented here.

`hycline.inference.mle_genomic.null_model_aicc_delta(data, fit,
log_likelihood_fn=...)` evaluates the null model's log-likelihood at
fixed `(u=0, v=1)` -- no optimization, just one likelihood evaluation --
computes its AICc with `k=0` free parameters (which correctly reduces to
plain AIC, since the small-sample correction term `2k(k+1)/(n-k-1)` is
zero when `k=0`), and returns `delta_aicc_from_null = aicc_null -
aicc_full`. Positive means the fitted model is favored; the higher, the
stronger the evidence against a neutral origin for that locus. A generic
`log_likelihood_fn` parameter (default: the hard-count binomial
likelihood) lets the same function serve both `fit-genomic`'s hard-count
path and its genotype-likelihood path (passing
`hycline.likelihoods.genotype_likelihood.log_likelihood_genomic`)
without `mle_genomic.py` needing to import GL-specific code it otherwise
has no dependency on.

Wired into both `_fit_genomic_hard_count` and `_fit_genomic_gl` in
`cli.py`: three new columns (`log_likelihood_null`, `aicc_null`,
`delta_aicc_from_null`) added to `*_summary.csv`, and a ranked table
(descending `delta_aicc_from_null`) printed to stdout whenever more than
one locus was fit -- skipped for single-locus runs, where a ranking has
nothing to rank.

**Verification, on real synthetic data with known ground truth**
(`examples/synthetic_genomic_clines.csv`, the same 5-locus dataset
section 15's own coupling/candidate-flagging examples and the genomic
user guide's hierarchical-fitting example already use): the three
genuinely non-null loci (`biased1`, `barrier1`, `permeable1`) rank at
the top with `delta_aicc_from_null` of 242.16, 101.64, and 65.06
respectively; the two genuinely neutral loci (`neutral1`, `neutral2`)
correctly land near zero (-0.70, -1.52) -- the two extra free parameters
weren't worth their AICc penalty for either. Also verified directly on
synthetic data spanning the range from clearly-null through weakly- to
strongly-non-null that `delta_aicc_from_null` orders them correctly
(`tests/test_mle_genomic.py::
test_null_model_aicc_delta_ranks_multiple_loci_sensibly`), and that the
formula matches a from-scratch manual AICc computation exactly, not just
approximately.

### 16.4 Verification

Full test suite re-run in batches after both items landed: 24 new tests
across `tests/test_mle_genomic.py` (delta-AICc formula correctness,
ranking direction, real-data recovery), `tests/test_cli.py` (both
`fit-genomic` code paths -- hard-count and genotype-likelihood -- writing
the new summary columns and printing the ranked table; single-locus runs
correctly skip the ranking table), and
`tests/test_mcmc_genomic_hierarchical.py` (the unevenness-warning helper
in isolation across four constructed cases, plus one real end-to-end
fit that triggers it). Every collected test reconciled against
`--collect-only`'s total, per this project's own standing practice.

## 17. Candidate-flagging overlay plot: fitted curves colored by category

Section 15 shipped `flag-candidates-geographic`/`flag-candidates-genomic`
with CSV and printed-summary output only -- no plot. Requested directly:
"different classes of candidates are a different color and all the
other loci are a light grey color," and, after a design discussion on
what "the different classes" should actually show (a z-score scatter in
the space the flagging decision is made in, vs. the loci's actual fitted
cline curves), the fitted curves -- more biologically intuitive, and
needed for both cline types.

### 17.1 What's plotted, and why curves rather than a z-score/parameter scatter

`hycline.viz.plotting.plot_geographic_candidates_overlay`/
`plot_genomic_candidates_overlay` draw one fitted-curve line per locus on
a single set of shared axes -- allele frequency vs. geographic distance,
or phi(h) vs. hybrid index -- colored by the locus's own candidate
category, with non-candidate loci drawn first, in light grey and
partially transparent, so flagged loci visually sit on top rather than
getting lost among the rest of the locus set. This was chosen over a
z-score scatter (`z_center` vs. `z_rate`, the space the flagging decision
is actually made in) specifically because it shows *how* a flagged
locus's shape differs -- steeper, shallower, shifted -- not just that it
crossed a statistical threshold, which is what a domain reader actually
wants to see first. `_plot_candidates_overlay` is the shared core both
wrappers call into, mirroring `_flag_from_z_scores`'s own role as the
one-statistical-core-two-cline-types pattern this module already uses.

Genomic needs nothing beyond `u`/`v` (hybrid index is always in [0, 1]),
so `plot_genomic_candidates_overlay` also draws the same neutral-null
reference diagonal `plot_hierarchical_genomic_grid` already does per
panel. Geographic needs each locus's fixed `pmin`/`pmax` to evaluate
`cline_function` (Components A/D still have no tail-model support -- see
README.md's Tier 1 backlog -- so this is always plotted with
`TailType.NONE`), and has no observed distance range to fall back to the
way per-locus fitting commands do (`flag-candidates-geographic` reads
only the population/locus summary CSVs, not the original per-site data),
so it defaults to each locus's own `center +/- 3*width`, wide enough to
show every curve's full transition.

### 17.2 A fixed, CVD-validated category-color assignment, not four arbitrary colors

Read the `dataviz` skill's color-formula guidance before writing any of
this: categorical hues must be assigned in a fixed order and validated
for color-vision-deficiency separation, not picked by eye or left to
matplotlib's default cycle. Ran the skill's own validator
(`validate_palette.js`) against several 4-hue orderings drawn from the
project's categorical ramp, under `--pairs all` rather than only
adjacent-pair mode -- the right mode here, since every category can
appear simultaneously on one overlay (unlike, say, a stacked bar, where
only neighboring segments are ever adjacent). blue/orange/aqua/violet
(`#2a78d6`/`#eb6834`/`#1baf7a`/`#4a3aa7`) is the ordering that actually
passed: the first 4-hue combination tried (blue/orange/aqua/red) failed
on the orange-red pair (CVD Delta E 5.6, normal-vision Delta E 7.1, both
below the required floors). Non-candidate loci use a separate light grey
(`#c7c6c2`), deliberately outside the categorical ramp so it reads as
"background," not a fifth category. Assignment is fixed by category
identity (`_CANDIDATE_CATEGORY_COLORS`, a module-level dict), not by
whichever order categories happen to appear in a given fit's data, and
the legend lists categories in that same fixed order -- restricted,
elevated, asymmetric, ambiguous -- only omitting entries for categories
absent from the current result, never reordering the ones present.

### 17.3 CLI wiring: on by default, degrading gracefully rather than failing

Both `flag-candidates-geographic`/`flag-candidates-genomic` write
`<prefix>_candidates_overlay.png` by default alongside the existing CSV;
`--no-plot` skips it. Geographic's plot needs `pmin_fixed`/`pmax_fixed`
columns the locus summary CSV may not have (a hand-assembled summary, or
one from before this feature) -- rather than making the whole command
fail on a column the CSV output itself never needed,
`_resolve_optional_summary_col` (a `_resolve_summary_col` variant that
returns `None` instead of exiting on failed auto-detection, while still
treating an explicitly-given-but-wrong `--pmin-col`/`--pmax-col` as a
hard error) lets the command print a clear NOTE and skip just the plot.
Verified directly: `tests/test_cli.py::
test_cli_flag_candidates_geographic_missing_pmin_pmax_skips_plot_gracefully`
confirms this against the exact fixture (`_write_geographic_hierarchical_
summaries`) every other hand-crafted test in that file already uses,
which predates this feature and has no `pmin_fixed`/`pmax_fixed` columns
-- so this also confirms the feature doesn't break any pre-existing test
input format. Genomic has no such gap (`u`/`v` are already required for
the flagging statistics themselves), so its plot is unconditional
whenever `--no-plot` isn't given.

### 17.4 Verification

`tests/test_plotting_candidates.py` (new file, 9 tests) checks the
plotting functions' actual contract directly -- fixed category-to-color
assignment, legend order and omission of absent categories, non-flagged
loci drawn grey and first, the default/explicit distance-range behavior,
the genomic neutral-reference line -- not just "it runs without an
exception." `tests/test_cli.py` gained CLI-level coverage: the existing
`test_cli_flag_candidates_geographic_composes_with_real_fit_hierarchical_
output` test (real `fit-hierarchical-geographic` output on real Manakin
data, piped into `flag-candidates-geographic`) extended to assert the
PNG exists and is non-trivially sized; `test_cli_flag_candidates_genomic_
end_to_end` extended the same way; plus dedicated `--no-plot` tests for
both commands and the graceful-degradation test above. Both real plots
(the Manakin composition run, and a hand-built 5-locus scenario
constructed to actually produce restricted/elevated/asymmetric outliers)
were rendered and visually inspected, not just asserted to exist -- the
flagged loci's curves visibly stand out in their assigned colors against
the grey background, and the legend lists only the categories present,
in the fixed order.
