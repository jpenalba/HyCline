# HyCline Backend Notes: Phase 2 (Bayesian / MCMC)

This document explains the *why*, not just the *what*, of the Phase 2
inference backend: the modeling choices, the sampler mechanics, the
vectorization strategy, and the tradeoffs that came up while building it.
It's meant to be read alongside the code, not instead of it.

---

## 1. Why NumPyro is the primary backend

The architecture plan flagged NumPyro (JAX-based) over PyMC as the primary
backend for two concrete reasons, both of which held up during implementation:

1. **NUTS on JAX is fast for this problem shape.** A single cline fit has
   only 2-6 parameters (center, width, optionally pmin/pmax, optionally
   tail parameters) and a handful to a few dozen data points (sampling
   sites). This is a *low-dimensional, cheap-likelihood* problem, which is
   exactly the regime where JAX's JIT compilation overhead pays for itself
   quickly and NUTS mixes well.
2. **Native vectorization across loci.** NumPyro's `plate` construct lets a
   *single* model definition and a *single* MCMC run fit many loci at once,
   with the sampler operating on a batched (loci x parameters) array under
   the hood. This is the practical mechanism behind the "vectorized
   multi-locus batch fitting" goal from the architecture plan -- see
   section 4 below for why this uses `numpyro.plate` rather than an
   explicit `jax.vmap` call around a Python fitting loop, which was the
   original phrasing in the architecture plan but turned out to be the
   less idiomatic (and, for MCMC specifically, less correct) way to get
   the same result.

PyMC remains available as a secondary backend (`mcmc_pymc.py`) for people
who prefer its more declarative context-manager API or already have PyMC
installed for other work. It is *not* vectorized across loci in this
scaffold -- see section 6.

---

## 2. The probabilistic model

For a single locus, the generative model is:

```
center ~ Uniform(d_min - span, d_max + span)      # cline center, in transect distance units
width  ~ Uniform(eps, span * 10)                   # cline width; see note on priors below
pmin   ~ Uniform(0, 1)     [or fixed to a constant]
pmax   ~ Uniform(0, 1)     [or fixed to a constant]

for each site i:
    p_i = pmin + (pmax - pmin) / (1 + exp(-4 * (distance_i - center) / width))
    n_focal_i ~ Binomial(n_alleles_i, p_i)
```

This is deliberately the *same* model as the Phase 1 MLE fit
(`hycline.models.cline.cline_function` + the binomial likelihood) -- Phase
2 doesn't change what's being modeled, only how it's fit. That's an
intentional design choice: it means Phase 1's MLE point estimates and
Phase 2's posterior means/modes should agree (up to Monte Carlo and
prior-influence noise), which is itself a useful cross-check and is
exercised in `tests/test_mcmc_numpyro.py`.

### Note on the width prior

A flat `Uniform(eps, span * 10)` prior on width is simple and matches the
MLE fit's bounds (so the two backends are as comparable as possible), but
it is not the only reasonable choice. Two alternatives worth knowing about
if you extend this:

- **`LogNormal` prior on width** would put more prior mass on small-to-moderate
  widths and less on very large ones, which is often a better match to
  biological expectations (most clines are not thousands of km wide) and
  can improve NUTS geometry by avoiding a hard boundary. Not used here so
  that Phase 2 posteriors are a clean, comparable extension of Phase 1's
  MLE fit rather than a different model with different assumptions baked in.
- **A weakly-informative prior centered on a rough field estimate of the
  organism's dispersal distance** is what real hybrid-zone studies often
  use in practice (tension-zone theory relates width to dispersal and
  selection strength; see Barton 1979). This is a natural target for a
  future "informative priors" convenience API, not implemented here.

### Note on `pmin`/`pmax`

Mirroring HZAR's own three "scaling" modes (fixed to 0/1, fixed to
observed extremes, estimated), `fit_mcmc` accepts optional `pmin_fixed`/
`pmax_fixed` arguments. When given, those parameters are treated as
constants in the model (no `numpyro.sample` call, no prior, no posterior
uncertainty) rather than as `Uniform(0,1)`-distributed unknowns -- this
mirrors the Phase 1 MLE's `fit_mle_fixed_extremes` and is what's needed to
reproduce HZAR's own "Model I" analysis in a Bayesian setting.

---

## 3. Why NUTS, and what its diagnostics mean here

NUTS (the No-U-Turn Sampler, Hoffman & Gelman 2014) is Hamiltonian Monte
Carlo with automatic step-count tuning. It's the default and near-universal
choice for continuous-parameter models in both NumPyro and PyMC, and there
was no reason to consider anything else for a model this well-behaved
(smooth, low-dimensional, no discrete latent variables).

Three diagnostics matter in practice for cline fits, all surfaced by
`hycline.diagnostics.convergence`:

- **R-hat** (potential scale reduction factor): compares between-chain and
  within-chain variance. Values should be very close to 1.0 (HyCline flags
  anything above 1.01). For a cline fit, R-hat problems usually mean either
  the data doesn't actually constrain the cline well (e.g. very few sites,
  or all sites on one side of the transition) or the width prior's upper
  bound is so loose relative to the data that the sampler struggles to
  explore it -- tightening the width prior is often the fix, not more samples.
- **Effective sample size (ESS)**: how many independent-equivalent draws
  the correlated MCMC chain represents. Low ESS relative to the total draw
  count usually shows up on the `width` parameter specifically, since cline
  width is often the least identifiable parameter when sampling is sparse
  near the transition zone.
- **Divergences**: NUTS-specific warnings that the sampler hit a region of
  pathological curvature. A handful of divergences on a model this simple
  is usually a sign the width's uniform prior boundary is being pushed
  against by the data (e.g. very narrow apparent cline with a loose prior
  upper bound) rather than a fundamental modeling problem.

---

## 4. Vectorized multi-locus fitting: plates, not a manual vmap loop

The architecture plan's original phrasing was "JAX's vmap lets you fit the
same cline model across thousands of loci in a single vectorized call."
That's directionally right but the actual mechanism in `fit_mcmc_batch` is
`numpyro.plate`, not a hand-written `jax.vmap` wrapped around repeated MCMC
calls. Worth being precise about why:

- **`jax.vmap` vectorizes a pure function call.** It's the right tool if
  you're doing, say, batched MLE optimization (running `scipy`-style
  gradient descent independently per locus) or batched *prediction* from
  already-fit parameters.
- **MCMC is not a pure function you can `vmap` over naively** -- each
  chain's sampler state (position, momentum, step size, tree depth) is
  stateful across iterations, and NUTS's own control flow (building the
  trajectory tree) doesn't vmap cleanly through `numpyro.infer.MCMC`'s
  ordinary interface.
- **`numpyro.plate`** is NumPyro's actual mechanism for this: parameters
  and observations declared inside a plate get an extra batch dimension,
  and NumPyro's own machinery (which *does* use `vmap` and `jit` internally,
  just not exposed as something you write yourself) runs one NUTS chain
  whose state is a *batch* of per-locus states, advancing every locus's
  chain together at each iteration. The net effect -- one MCMC run,
  wall-clock cost close to a single-locus fit rather than thousands of
  independent ones -- is exactly what the architecture plan was going for.
  It just happens through `plate` rather than a user-facing `vmap` call.

This matters for one concrete modeling consequence: **`fit_mcmc_batch` in
this scaffold fits each locus with independent priors, not shared
(hierarchical) ones.** There is no pooling of information across loci --
locus A's posterior doesn't influence locus B's. That's intentional and
matches the Phase 2 scope in the architecture plan; the *hierarchical*
version (loci sharing a population-level prior, à la `bgchm`) is Phase 4
work, and would change `fit_mcmc_batch`'s model definition (loci sampled
from a shared `Normal(mu_center, sigma_center)` etc. rather than each from
its own flat `Uniform`) while reusing the same plate-based vectorization
mechanism described here.

Implementation note: only the per-locus parameters (`center`, `width`,
`pmin`, `pmax`) are declared inside `numpyro.plate("locus", n_loci)`. The
observation step (`numpyro.sample("obs", ...)`) happens outside any plate,
directly on the full `(n_loci, n_sites)` broadcast array. An earlier
version tried nesting a second `site` plate inside `locus` to mark the
site dimension explicitly too, and hit a plate dimension-allocation
conflict (`numpyro.plate` picks negative axis positions automatically and
two plates both defaulting to `dim=-1` collide). Explicit plates matter for
SVI-style guides (which need to know conditional-independence structure to
subsample correctly); for plain MCMC evaluating a fixed joint density, the
observation likelihood doesn't need its own plate to be summed correctly,
so it was simplest to drop it rather than fight the dimension bookkeeping.

`fit_mcmc_batch` requires all loci to share the same `distance` vector
(i.e. the same sampling sites), which is the common case for real datasets
like the Manakin data (multiple loci scored at the same set of localities).
Loci with different sampling sites would need to be fit individually (or
the batch API extended to handle ragged data via masking, not implemented
here).

---

## 5. What `BayesianFitResult` gives you vs. `FitResult` (Phase 1)

`BayesianFitResult` (in `mcmc_numpyro.py`) intentionally mirrors Phase 1's
`FitResult` API where possible (`.summary()`, access to point-estimate
`ClineParams` for reuse with `cline_function`/plotting) but exposes
genuinely different information:

- **Credible intervals, not confidence intervals.** Phase 1's Hessian-based
  CIs are a Gaussian approximation around the MLE; Phase 2's highest-density
  intervals (HDIs) come directly from the posterior distribution and can be
  asymmetric or multimodal if the data supports that shape (rare for this
  model, but not assumed away).
- **The full posterior (`idata`, an ArviZ `InferenceData`)** is retained,
  not just summary statistics -- needed for the credible-band plotting in
  `viz/plotting.py::plot_bayesian_fit`, and for any downstream use (e.g.
  posterior predictive checks, or feeding into the Phase 3/4 hierarchical
  extensions).

---

## 6. PyMC secondary backend: what's different

`mcmc_pymc.py` implements the same single-locus model in PyMC's API. Two
deliberate scope limits, both documented in that module's docstring:

- **No batch/plate-vectorized multi-locus fitting.** PyMC supports this in
  principle (via `pm.Data` with a batch dimension and appropriately shaped
  `pm.Binomial`), but implementing and testing it in parallel with the
  NumPyro version wasn't worth the duplication for a secondary backend --
  if you need to fit thousands of loci, use the NumPyro backend.
- **No fixed-pmin/pmax convenience path yet** -- straightforward to add
  following the same pattern as `mcmc_numpyro.py`, just not done in this
  pass.

**Environment note, not a modeling one:** `fit_mcmc_pymc` hardcodes
`cores=1` in its call to `pm.sample`. PyMC's default core-count
auto-detection divides by the detected CPU count to decide how to split
multiprocessing workers, and in sandboxed/containerized environments where
CPU visibility is restricted or misreported, that detection can return 0
and crash with a `ZeroDivisionError` before sampling even starts. Forcing
`cores=1` sidesteps it -- chains still run (sequentially) and R-hat still
works, it just gives up PyMC's multiprocessing parallelism. On a normal
development machine this is worth revisiting (drop the override and let
PyMC auto-detect) for faster multi-chain sampling.

Both backends return the same `BayesianFitResult` type (constructed from an
ArviZ `InferenceData` either way, since both NumPyro and PyMC interoperate
with ArviZ), so downstream code (diagnostics, plotting) doesn't need to
know which backend produced a given fit.

---

## 7. Performance notes (measured on this scaffold's dev environment)

- Single-locus fit (11 sites, 4 chains x 1000 warmup x 1000 samples): a
  few seconds on CPU, dominated by JIT compilation on the first call within
  a process -- subsequent fits in the same process are faster.
- Batch fit (multiple loci, same transect, via plate): the *marginal* cost
  per additional locus is small relative to the fixed compilation cost, so
  batch-fitting many loci at once is meaningfully cheaper per-locus than
  looping `fit_mcmc` once per locus.
- No GPU-specific tuning was done here; for a model this small, GPU
  dispatch overhead likely outweighs any benefit. GPU/TPU would start to
  matter for the hierarchical multi-locus model at genome-wide scale
  (thousands of loci with shared hyperparameters), which is Phase 4 scope.
- **Multi-chain CPU note:** by default JAX sees a single CPU "device", so
  `num_chains > 1` runs chains *sequentially* rather than in parallel (NumPyro
  warns about this at runtime). Multiple chains are still worth running for
  the R-hat diagnostic (section 3) even sequentially -- it's a correctness
  tool, not just a speed one -- but if wall-clock time matters, call
  `numpyro.set_host_device_count(n)` once at the start of your program
  (before any JAX arrays are created, same constraint as the x64 flag in
  section 9) to get genuine parallel chains on CPU.

---

## 8. Import structure: why Phase 2 isn't in the top-level `hycline` namespace

`hycline/__init__.py` (the Phase 1 top-level API) deliberately does *not*
import anything from `mcmc_numpyro.py` or `mcmc_pymc.py`. JAX, NumPyro,
ArviZ, and PyMC are substantial dependencies (JAX alone pulls in a fair
amount), and someone who only wants Phase 1's MLE fitting shouldn't be
forced to install or import them. Phase 2 functionality is reached via
explicit submodule imports:

```python
from hycline.inference.mcmc_numpyro import fit_mcmc, fit_mcmc_batch
from hycline.inference.mcmc_pymc import fit_mcmc_pymc
from hycline.diagnostics.convergence import check
from hycline.viz.plotting import plot_bayesian_fit
```

This is why `pyproject.toml` splits `bayes` out as an optional dependency
group (`pip install -e ".[bayes]"`) rather than a base requirement.

**Versions this was built and tested against:** `numpyro==0.21.0`,
`arviz==1.2.0`, `jax==0.11.0`, `pymc==6.2.0`. Called out specifically
because ArviZ's `hdi()` API changed between major versions in a way that
fails loudly (`TypeError`) rather than silently -- pre-1.0 ArviZ used
`hdi_prob=` and returned an `hdi` coordinate with `'lower'`/`'higher'`
labels; the version here uses `prob=` and a `ci_bound` coordinate with
`'lower'`/`'upper'` labels. `BayesianFitResult.hdi()` is written against
the newer API; if you pin an older ArviZ, update that method to match.

---

## 9. Case study: the default width prior on real, sparse data

`examples/demo_bayesian_manakin.py` fits the same real Manakin Ada locus
data used for the Phase 1 validation (see the project README) with the
NumPyro backend, and it's a genuinely useful worked example of section 3's
diagnostics advice, not just a smoke test:

**With the default width prior** (`Uniform(eps, 10 * transect_span)` =
`Uniform(eps, ~5695)` km for this dataset), NUTS's posterior mean for width
lands around 990 km -- nowhere near HZAR's published 263 km -- and
`hycline.diagnostics.convergence.check()` correctly flags it: R-hat 1.02,
5 divergences, low ESS on `width` specifically. The sampler isn't
malfunctioning; the prior is simply so loose relative to what 11 sparse
sampling sites can constrain that the posterior has a long, poorly-explored
tail, and NUTS's diagnostics surface that honestly rather than returning a
falsely-confident number.

**Bounding the width prior at the transect's actual geographic span**
(569.5 km -- still generous, just not absurd) resolves this cleanly:
R-hat 1.00, zero divergences, ESS in the thousands. The posterior mean for
width (364 km) still sits above the MLE/HZAR point estimate (262-263 km),
but the 95% credible interval (`[163, 569]`) comfortably contains it. This
is expected, not a bug: a flat prior on a weakly-identified parameter with
this little data will show some prior influence, and the honest comparison
is "does the interval contain the reference value," not "does the
posterior mean exactly match a point estimate from a different fitting
procedure." `tests/test_mcmc_manakin_bayesian.py` locks in both halves of
this story -- the default prior is asserted to trip the diagnostics, and
the bounded prior is asserted to pass cleanly and bracket HZAR's published
values.

The practical takeaway for anyone using `fit_mcmc` on their own data: don't
treat the default `width_upper_bound` as safe to ignore. Run
`hycline.diagnostics.convergence.check()` on every real fit, and if it
flags R-hat/ESS/divergence problems, tightening the width prior toward a
biologically plausible range for your system is the first thing to try,
not more warmup samples.

---

## 10. A real bug caught while building the CLI: `fit_mcmc`'s `pmin_fixed`/`pmax_fixed` don't auto-derive

`fit_mle_fixed_extremes` (Phase 1) has a convenience default: if you don't
pass explicit `pmin`/`pmax`, it derives them from the observed frequencies
at the transect's geographic extremes. `fit_mcmc` (this module) does
*not* do the equivalent -- `pmin_fixed=None`/`pmax_fixed=None` means
"sample these freely," not "derive them from the data." This is a real
asymmetry between the two functions, not a documentation gap: passing
`method="classic", pmin_fixed=None, pmax_fixed=None` and
`method="bayesian", pmin_fixed=None, pmax_fixed=None"` to
`hycline.api.fit_cline` with the *intent* of matching HZAR's
`scaling="fixed"` mode on both does **not** do the same thing on both
paths, and this went unnoticed until end-to-end CLI testing surfaced it:
a `--scaling fixed` CLI run without explicit `--pmin`/`--pmax` silently
fit pmin/pmax as free parameters on the Bayesian path, producing a
posterior with real (and wrong) uncertainty on parameters that were
supposed to be constants.

The CLI (`hycline/cli.py::cmd_fit`) now derives the observed extremes
itself before calling `fit_mcmc`, matching `fit_mle_fixed_extremes`'s
behavior, and `tests/test_cli.py::
test_cli_fit_bayesian_scaling_fixed_actually_fixes_pmin_pmax` locks this
in. But this fix lives in the CLI layer, not in `fit_mcmc` itself -- if
you call `hycline.inference.mcmc_numpyro.fit_mcmc` or
`hycline.api.fit_cline(..., method="bayesian")` directly from Python with
`pmin_fixed=None`, you will still get freely-estimated pmin/pmax, not an
auto-derived fixed value. The more consistent long-term fix would be
giving `fit_mcmc` the same auto-derivation default `fit_mle_fixed_extremes`
has, so the two functions actually agree on what "no explicit value given"
means -- flagged here as a followup rather than done silently, since
changing `fit_mcmc`'s default behavior is a behavior change for any
existing caller relying on the current "None means free" semantics.

---

## 11. Known rough edges / follow-ups

- **Tail models are now wired into the NumPyro backend** (section 12) --
  this bullet previously flagged them as unimplemented. PyMC
  (`mcmc_pymc.py`) still only supports `TailType.NONE`; it wasn't
  extended given NumPyro's status as the primary backend (section 1) and
  the real, documented convergence difficulty tail models introduce for
  gradient-based samplers in general (section 12.2) -- extending the
  secondary backend to a capability that needs this much caveat wasn't a
  good use of the additional maintenance surface.
- `mcmc_numpyro.py` has its own small JAX-native reimplementation of the
  cline sigmoid (`_cline_core_jax`) rather than reusing
  `hycline.models.cline.cline_function` directly, because that function is
  written with plain NumPy (`np.clip`, `np.exp`), which isn't traceable by
  JAX inside a NumPyro model. This is a real duplication (two places that
  need to agree on the cline formula) rather than a design feature --
  flagged here rather than hidden, with a regression test
  (`test_jax_core_matches_numpy_core` in `tests/test_mcmc_numpyro.py`)
  that checks the two implementations agree numerically. A cleaner
  long-term fix would be writing `models/cline.py` against `numpy.array_api`
  or a small backend-dispatch shim so one implementation serves both the
  MLE and MCMC paths -- worth doing before Phase 3 adds more model variants
  that would otherwise need to be written twice.
- `mcmc_numpyro.py` enables `jax_enable_x64` at import time (must happen
  before any JAX arrays are created). JAX defaults to 32-bit floats, which
  is standard for most ML workloads but introduces ~1e-7 relative error --
  large enough that, without this, the JAX-native cline core would visibly
  disagree with the NumPy version in the parity test mentioned above, and
  posterior means would drift slightly from the float64 MLE fit they're
  meant to be checked against. This is a real, easy-to-miss JAX default
  worth knowing about if you add more JAX code to this project.

---

## 12. Tail-model support: what was built, what genuinely can't be fixed, and why

Tail models (`TailType.LEFT`/`RIGHT`/`BOTH`/`MIRROR`) are now supported by
the NumPyro Bayesian backend and by Phase 5's genotype-likelihood-aware
classic MLE (`fit_mle_geographic_gl`). This section covers the math
extension, a genuine, carefully-investigated limitation that turned out
*not* to be fixable within this pass, a related but more tractable issue
that *was* fixable, and four real bugs the investigation surfaced
elsewhere in the codebase along the way.

### 12.1 The JAX math extension itself was the easy, low-risk part

`_cline_tail_jax` in `mcmc_numpyro.py` reimplements
`hycline.models.cline.cline_function`'s tail extension (Barton & Gale
1993 exponential-decay tails grafted onto the sigmoid core) using
`jnp.where` in place of NumPy's boolean-mask assignment
(`out[mask] = ...`), the same substitution `_cline_core_jax` already
makes for the sigmoid core itself -- JAX arrays are immutable, so
`jnp.where` is the differentiable, traceable equivalent. Verified to
match the NumPy reference exactly (`atol=1e-10`) across all five tail
types (`tests/test_mcmc_numpyro.py::test_jax_tail_matches_numpy_tail`).
`_single_locus_model`/`_batch_model` were extended to conditionally
sample `delta_l`/`tau_l`/`delta_r`/`tau_r` with flat `Uniform` priors
matching Phase 1 classic MLE's own bounds exactly (delta in `[0, 1]` as a
height fraction, tau on the same scale as width) -- the same "match the
existing bounds convention" approach already used throughout this
project (e.g. Phase 5's `_bounds_for_geographic_gl` mirroring Phase 1's
`_bounds_for`).

None of this was where the real difficulty turned out to be.

### 12.2 The real finding: Bayesian (NUTS) tail-model fitting has severe, genuine convergence difficulty -- not a bug, and not fixable by more compute

The original expectation was that, once the math was correct, tail-model
Bayesian fitting would behave like every other addition in this
project's history: work reasonably, with the usual caveats about sparse
data needing tighter priors (section 9). That expectation was wrong, and
the investigation that found this is worth documenting in full, not just
the conclusion.

**What was observed**: fitting a synthetic `TailType.LEFT` cline (known
true parameters, 80-200 sites, realistic sample sizes) via `fit_mcmc`
produced R-hat values from 1.87 up to 3.55 and effective sample sizes as
low as 4, even at the CLI's default 1000 warmup/1000 samples. This is
far outside "needs more tuning" territory -- R-hat above roughly 1.05 is
already a serious warning sign (section 3); R-hat above 3 with ESS in
the single digits means the chains are not exploring a shared
distribution at all.

**What was ruled out, in order, before concluding this is fundamental**:

1. **Insufficient data or too-loose priors** -- ruled out by increasing
   the sample size (80 -> 200 sites) and tightening `width_upper_bound`
   to a realistic value (from the default ~1500 down to 50). Convergence
   remained just as poor.
2. **Just needs more warmup** -- ruled out by increasing warmup from 1000
   to 4000 iterations. The run did not even complete within a 280-second
   budget that the exact same model with `TailType.NONE` finishes in
   under 40 seconds with *better* convergence (R-hat=1.00) than the
   tail-model run achieved at 1/4 the warmup. This rules out "just slow,
   given enough patience it converges" -- the *rate* of progress toward a
   good posterior, not just the total time, is the problem.
3. **A more complex tail parameterization is the culprit** -- ruled out
   by testing `TailType.MIRROR` (4 fewer effective degrees of freedom
   than `BOTH`, since delta/tau are shared between both sides). It timed
   out even at 1500 warmup, the same as or worse than `LEFT`.
4. **The JAX translation itself is wrong** -- ruled out directly: section
   12.1's exact point-wise numerical match against the NumPy reference
   rules out a math error as the cause.

**The isolating test that pins down the mechanism**: fitting the *exact
same simulated data* (200 sites, generated from a true `LEFT`-tail
cline) with `tail_type=TailType.NONE` instead of `LEFT` -- everything
else identical -- converges perfectly: R-hat=1.00 across all four
parameters, ESS in the thousands. The only variable that changed between
a perfectly-converging fit and a badly-failing one is whether tail
parameters are present at all. This is about as close to a controlled
experiment as this kind of investigation gets.

**The likely mechanism**: the tail model is mathematically continuous at
the join between the sigmoid core and the exponential tail (the tail
value equals the core value exactly at `x = center +/- width/2`, by
construction), but not smooth there -- the *derivative* generally differs
on either side of the join. Gradient-based samplers like NUTS rely on a
leapfrog integrator that assumes locally smooth log-density; a kink
introduces a genuine discontinuity in the gradient the integrator has to
navigate. Critically, the join's *location* (`center +/- width/2`) is
itself a function of sampled parameters, so the kink moves around in
data-space as the sampler explores -- meaning there's no fixed "bad
region" to avoid, the pathology follows the sampler's own current
parameter values. This is a known, general class of difficulty for
Hamiltonian Monte Carlo applied to piecewise-smooth models; it is not
specific to this codebase's implementation.

**Why this isn't fixed in this pass, and likely isn't a "fix the code"
problem at all**: making a piecewise model's log-density fully smooth
(e.g. by replacing the hard join with some kind of soft blend between
the sigmoid and the tail) is a real *model redesign*, not a bug fix --
it would change what the tail model actually represents, needs its own
derivation and validation against the existing Barton & Gale (1993)
formulation, and might not even be the right statistical choice (a
smoothed join is a different model, with different fitted-parameter
meanings, not a strictly-better version of the same one). That is
explicitly out of scope here. This is presented as a genuine, current
limitation of this project's Bayesian tail-model support, not a
temporary rough edge expected to disappear with a future patch.

**What ships despite this**: the capability is not removed or hidden --
`fit_mcmc`/`fit_mcmc_batch` accept any tail type, and the math is
correct. But:

- Both functions emit a `UserWarning` whenever `tail_type != TailType.NONE`,
  explaining the issue and recommending classic MLE instead (verified:
  `tests/test_mcmc_numpyro.py` -- the warning fires for every tail type
  and stays silent for `TailType.NONE`).
- The CLI (`hycline fit-geographic`/`fit-geographic-multi --method bayesian --tail mirror/both`)
  prints the same guidance proactively, before fitting starts, in
  addition to the existing convergence diagnostics that will already
  flag the resulting bad R-hat/ESS after the fact (section 3). Two layers
  of warning -- one before the run in case the user wants to reconsider,
  one after confirming what actually happened -- rather than relying on
  either alone.
- `--tail`'s CLI help text states the limitation plainly rather than
  claiming unqualified support.

**Practical recommendation, stated directly rather than left implicit**:
for tail-model fitting, use classic MLE (`--method classic`, or
`fit_mle`/`fit_mle_geographic_gl` in Python) — it does not have this
issue (section 12.3). Reach for Bayesian tail-model fitting only if full
posterior uncertainty is specifically needed and you're prepared to
scrutinize the diagnostics carefully; do not trust a tail-model Bayesian
result without checking `hycline.diagnostics.convergence.check()` first.

### 12.3 The related, more tractable issue in classic MLE -- and why multi-start actually helps here, unlike NUTS

Extending Phase 5's `fit_mle_geographic_gl` to tail models (mechanically
straightforward -- `log_likelihood_geographic` already dispatches to
`cline_function` generically for any tail type; the only real work was
extending `_initial_guess_geographic`/`_bounds_for_geographic_gl` to
match Phase 1 classic MLE's own tail-parameter bounds/initial-guess
convention exactly) surfaced a related-but-more-tractable version of the
same underlying issue: on a single-start optimization, L-BFGS-B could
terminate with `success=False` and a corner solution (`delta_l=0`,
hitting its lower bound) even when a substantially better solution
existed nearby -- confirmed directly by comparing against 20 randomized
restarts, which found a solution with *higher* likelihood than the
single-start result (and, separately, higher likelihood than the true
generating parameters themselves, consistent with the same finite-sample
overfitting-to-noise theme documented throughout this project, e.g.
section 9's width-prior case study and `BACKEND_PHASE4.md` section 11).

Unlike NUTS, this **was** fixable, because the mechanism is different:
`fit_mle_geographic_gl` now runs `n_restarts` (default 8) additional
L-BFGS-B optimizations from randomized starting points whenever
`tail_type != TailType.NONE`, keeping the best (lowest negative
log-likelihood) result. This works for classical optimization in a way
it fundamentally can't for MCMC: L-BFGS-B restarts are independent
attempts, each cheap, and the kink only needs to be avoided by *one* of
them landing in the right basin -- NUTS instead has to successfully
integrate *through* the kink continuously as part of a single connected
exploration of the posterior, so more attempts (more warmup, effectively)
doesn't have the same kind of parallel-independent-tries structure to
exploit. Verified directly:
`tests/test_genotype_likelihood_fitting.py::
test_multistart_finds_better_optimum_than_single_start` confirms the
multi-start result's likelihood is never worse than a matched
single-start fit on the same data.

Even with multi-start, `tau_l`/`tau_r` (decay length) specifically can
still show large residual uncertainty on realistic sample sizes -- this
looks like genuine weak practical identifiability (few data points
typically fall within the tail zone itself, geometrically) rather than
an optimization failure, and is not something multi-start is expected to
resolve; it's a property of the data, not the algorithm.

### 12.4 Four real bugs this investigation surfaced elsewhere, unprompted

Building and stress-testing tail-model support end to end (not just unit
math checks) surfaced four pre-existing correctness issues in code that
had never actually been exercised with a working non-`NONE` Bayesian
tail fit before, or (for the third and fourth) had shipped with Phase 1
CLI tail support without ever being checked this thoroughly:

1. **`plot_gl_multi_locus_grid` hardcoded `TailType.NONE`** when
   constructing the fitted curve for display, regardless of what tail
   type the fit actually used -- would have silently plotted the wrong
   (tail-less) curve shape for any GL-aware tail-model fit. Fixed to use
   `fit.tail_type` (already available on the `FitResult` object, just
   not read). Regression test: `tests/test_plotting_bayesian.py` doesn't
   cover this one directly (it's the GL grid plot, not the Bayesian
   plot), but the CLI end-to-end tests (section 12.4 item 3 below) now
   exercise the corrected path.
2. **`plot_bayesian_fit` had the identical hardcoding**, plus it had
   never extracted tail-parameter posterior draws at all -- meaning
   before this fix, plotting a tail-model Bayesian fit would have passed
   `delta_l=None` etc. into `cline_function` with a non-`NONE` tail type,
   which would raise rather than silently misbehave (since the tail
   branch's arithmetic needs an actual float). Fixed to extract
   `delta_l`/`tau_l`/`delta_r`/`tau_r` draws when present and construct a
   complete `ClineParams` per posterior draw. Regression test:
   `tests/test_plotting_bayesian.py::
   test_plot_bayesian_fit_with_mirror_tail_runs_without_error`.
3. **`cmd_fit`'s Bayesian branch (fitted-curve CSV, posterior-samples
   CSV, and JSON summary) all independently hardcoded the same
   `("center", "width", "pmin", "pmax")` tuple**, three separate times in
   one function -- the fitted-curve construction specifically would have
   *raised* for any real tail-model Bayesian fit (same `ClineParams`
   `None`-field issue as item 2), not merely dropped data silently. All
   three now iterate over the actual set of sampled parameters (checking
   `"delta_l" in post`, etc.) rather than a fixed list. Regression test:
   `tests/test_cli.py::
   test_cli_fit_bayesian_tail_model_does_not_crash_and_includes_tail_params`.
4. **`cmd_fit_multi`'s Bayesian branch had the identical three-fold bug**,
   and its **classic-method branch had a separate, longer-standing issue**:
   the per-locus summary CSV `row` dict never included tail parameter
   *values* at all (only, inconsistently, their standard errors) --
   meaning `hycline fit-geographic-multi --tail mirror` has silently dropped
   `delta_l`/`tau_l` from its output CSV since Phase 1's CLI first shipped
   tail-model support, not something introduced by this pass. All fixed
   the same way as item 3. Regression tests: `tests/test_cli.py::
   test_cli_fit_multi_bayesian_tail_model_does_not_crash`, `::
   test_cli_fit_multi_classic_tail_model_includes_tail_values_in_csv`.

None of these four were caught by the existing test suite before this
investigation, because no existing test had ever actually run a
multi-locus or Bayesian fit with a non-`NONE` tail type all the way
through to its plotting/CSV/JSON output -- the original Phase 1/2 tests
checked the *fitting* math for tail models, but not the full reporting
pipeline downstream of it. This is itself worth noting as a gap in test
coverage philosophy (unit-testing a capability's core math is necessary
but not sufficient; the surrounding plumbing needs its own end-to-end
check) rather than just a list of bugs -- reflected going forward in how
the regression tests above are structured (full CLI runs producing real
output files, not just checking the underlying function's return value).

### 12.5 Known rough edges

- **Hierarchical tail-model pooling (Components A/D) is explicitly out of
  scope.** `fit_hierarchical_batch`/`fit_hierarchical_batch_gl` still
  only support `TailType.NONE`. Whether tail parameters should be pooled
  across loci the same way center/width are (`BACKEND_PHASE4.md`
  sections 2/11) is a real, separate design question -- and given
  section 12.2's finding that even independent-locus Bayesian tail
  fitting struggles with NUTS, a hierarchical version would need its own
  investigation into whether the difficulty compounds or is somehow
  mitigated by partial pooling, not an assumption either way.
- **PyMC (`mcmc_pymc.py`) was not extended to tail models** -- the
  secondary backend remains `TailType.NONE`-only (section 11). Given
  section 12.2's finding is about gradient-based sampling in general, not
  a NumPyro-specific implementation detail, there's no reason to expect
  PyMC's own NUTS implementation would fare any better; extending it
  wasn't judged worth the added maintenance surface without evidence it
  would behave differently.
- **The multi-start fix's `n_restarts=8` default is a reasonable but not
  rigorously tuned choice.** It was checked to meaningfully improve
  reliability over a single start, not swept to find an optimal value.

## 13. Wiring `fit_mcmc_batch` into `fit-geographic-multi`'s CLI: an automatic, opt-out-free speedup

The Python API's `fit_mcmc_batch` (section 4) -- a single plate-
vectorized NUTS run fitting every locus at once, instead of one
independent MCMC run per locus -- existed and was validated well before
this pass, but `fit-geographic-multi --method bayesian` never used it: the CLI
looped one `fit_mcmc` call per locus regardless. Tier 2 of the README's
"Not yet implemented" list flagged this as the real cost-saver at
"thousands of loci" scale specifically for Bayesian fitting, since each
locus's NUTS run is seconds-to-tens-of-seconds (BACKEND_SCALE.md didn't
measure this directly, but flagged it as the known cost center classic
MLE's numbers don't change the assessment of).

### 13.1 A real constraint that shapes the whole design: `fit-geographic-multi` doesn't require dense data, but `fit_mcmc_batch` does

`fit_mcmc_batch` requires every locus to share the exact same `distance`
vector (no per-locus missing-data masking -- the same constraint
Components A/B's hierarchical batch fitting has, for the same reason).
`fit-geographic-multi`, by contrast, exists specifically to fit loci
*independently*, and its own per-locus readers drop missing rows
independently for each locus -- meaning different loci commonly end up
with different site subsets if the input CSV has any missing values
anywhere. Silently requiring dense data would have been a real, breaking
behavior change for exactly the use case `fit-geographic-multi` is for (as opposed
to `fit-hierarchical-geographic`, which already requires dense data and exists
partly for that reason).

**The fix: detect, don't require.** `_try_stack_dense` (a new, non-fatal
refactor of the dense-data check `_stack_for_hierarchical` already used)
checks whether the loaded data happens to already be dense. If it does
-- genuinely common for real datasets where every locus was genotyped in
the same individuals/sites -- `cmd_fit_multi` uses the fast batched path
automatically, no flag needed. If it doesn't, `cmd_fit_multi` falls back
to the original independent per-locus loop, completely unchanged.
Verified directly, not just designed to work in principle: ragged data
(one locus with a missing site) correctly falls back and still fits all
loci (`tests/test_cli.py::
test_cli_fit_multi_bayesian_falls_back_on_ragged_data`).

**A second real constraint**: `fit_mcmc_batch` has no equivalent of
`pmin_fixed`/`pmax_fixed` -- it always samples pmin/pmax freely. So the
batched path is also skipped (falling back to the per-locus loop, which
does support fixed extremes via `fit_mcmc`) whenever `--scaling fixed`
or an explicit `--pmin`/`--pmax` is given, even on otherwise-dense data
(`tests/test_cli.py::
test_cli_fit_multi_bayesian_scaling_fixed_falls_back_even_on_dense_data`).

### 13.2 Reusing the existing per-locus reporting logic, not duplicating it

The batched path produces one `az.InferenceData` with a `locus`
dimension covering every fitted locus at once. `idata.isel(locus=i)`
slices this down to a per-locus `InferenceData` indistinguishable in
shape from what a single-locus `fit_mcmc` call would have produced
(confirmed directly: `check()`'s diagnostics on a sliced idata give the
same shape of report either way). This let the *existing* per-locus
Bayesian reporting code (summary rows, fitted-curve percentile bands,
credible intervals, grid-plot data) be extracted into one shared
function, `_report_bayesian_multilocus_fit`, called identically whether
the underlying fit came from the batched path or the original per-locus
loop -- rather than writing separate, drifting-apart reporting logic for
each path.

**A real bug caught by the test suite, not missed silently**: the
extracted helper initially relied on `cline_function` being imported in
the *caller's* local scope (true for the original per-locus loop, since
`cmd_fit_multi` imports it at the top), which broke the moment the same
code ran as a standalone function reached via the batched path --
`NameError: name 'cline_function' is not defined`, caught immediately by
`tests/test_cli.py::test_cli_fit_multi_bayesian_tail_model_does_not_crash`
on the very first test run after the refactor. Fixed by giving the
extracted function its own explicit import rather than depending on
whatever the caller happened to have imported.

### 13.3 Validated against known ground truth through the real CLI, not just "it runs"

A synthetic 5-locus dense dataset (15 sites, 5 distinct true
center/width combinations) was fit via `hycline fit-geographic-multi --method
bayesian` with no special flags. The batched path triggered
automatically ("using the plate-vectorized batch fitting path" printed
to the console), completed in 15.1s for all 5 loci (3.0s/locus
amortized), with clean convergence (R-hat 1.00-1.01) and per-locus
center/width estimates consistent with the known true values given the
finite-sample noise expected at n=15 sites. `tests/test_cli.py::
test_cli_fit_multi_bayesian_uses_batched_path_on_dense_data_and_recovers_truth`
locks this in as a regression test (center recovery within 15 units of
true, R-hat < 1.05).

**What wasn't directly measured**: a rigorous, controlled speedup
comparison (same data, same warmup/samples, batched vs. looped, wall-
clock ratio) wasn't run as part of this pass -- BACKEND_SCALE.md's own
stress test was run on classic MLE, not Bayesian fitting, and this
section's validation focused on correctness (does the batched path
produce the right answer, does the fallback correctly trigger) rather
than a formal benchmark. The amortized 3.0s/locus figure above is a real
number from one real run, not a controlled A/B comparison -- worth doing
properly if a precise speedup claim is ever needed, but the qualitative
case (avoiding N separate JIT-compilation/MCMC-startup overheads plus
genuine within-core vectorization, versus N full independent runs) was
the reason this was prioritized in the first place, not something this
pass needed to re-derive.

### 13.4 Known rough edges

- **No partial-failure granularity in the batched path.** The original
  per-locus loop catches exceptions per locus and continues (a
  problematic locus gets skipped, recorded in
  `*_multi_locus_failures.json`, and the rest still fit) -- a single
  `fit_mcmc_batch` call either succeeds for every locus or raises for the
  whole batch. In practice this matters less for Bayesian fitting than
  it might sound: NUTS with a Binomial likelihood is considerably more
  robust to a single degenerate locus (e.g. all-zero counts) than classic
  MLE's optimizer is, so outright batch failure caused by one bad locus
  is expected to be rare -- but it hasn't been stress-tested against a
  deliberately pathological locus mixed into an otherwise-healthy batch,
  so this is a real, acknowledged gap rather than a claim it's been ruled
  out.
- **Tail-model support inherits section 12's known convergence
  difficulty** unchanged -- the batched path uses the same underlying
  tail-model log-density, so the same non-smooth-join issue applies per
  locus within the batch. `fit-geographic-multi` still prints the existing tail-model
  warning regardless of which path (batched or per-locus) ends up being
  used.
- **`compare-tails` and the other multi-locus Bayesian commands
  (`fit-genomic-multi --method bayesian`, `fit-geographic-gl-multi`,
  `fit-genomic-gl-multi`) were not extended with an equivalent batched
  path in this pass** -- this was scoped specifically to `fit-geographic-multi`
  (geographic, hard-count Bayesian fitting), the highest-value single
  target given BACKEND_SCALE.md's own findings. Extending the same
  detect-and-batch pattern to the other commands is a natural, mostly
  mechanical follow-up (the underlying batch-fitting functions already
  exist for genomic clines via Component B's infrastructure -- see
  BACKEND_PHASE4.md -- though not for the GL-aware commands specifically),
  not attempted here.

## 14. `hzar_tail` extended to the Bayesian path, and six real curve-export bugs found doing it

The verified HZAR tail construction (`hycline.models.cline.
hzar_cline_function`, first built for quantitative trait clines and then
ported to classic MLE allele-frequency fitting as opt-in `--hzar-tail`
-- see BACKEND_TRAIT.md sections 1-2 and 5) was extended to the Bayesian
path in this pass, prompted by a direct question about which parts of
this project rest on a less-established foundation than the rest --
`cline_function`'s own tail model (the project's original, default one)
was the clearest answer, since it had never been checked against HZAR's
real source the way the rest of the project increasingly has been.

### 14.1 Scope turned out smaller than expected: Components A and D don't support tail models at all

Before touching anything, the actual scope was checked directly rather
than assumed. Searching for `tail_type`/`TailType` in
`mcmc_hierarchical.py` (Component A) and `mcmc_hierarchical_gl.py`
(Component D) returns zero matches -- neither hierarchical geographic
model supports *any* tail model, HZAR-parameterized or otherwise; both
are always implicitly `TailType.NONE`. PyMC (`fit_mcmc_pymc`) explicitly
raises `NotImplementedError` for any non-`NONE` tail type. This means
the *only* Bayesian/hierarchical functions that could meaningfully
receive an `hzar_tail` flag were `fit_mcmc` and `fit_mcmc_batch` in
`mcmc_numpyro.py` -- a real, precisely-scoped-down task, not the
four-component undertaking it might have looked like from the outside.
Extending Components A/D to support tail models at all (a separate,
larger capability gap, not a parameterization question) is noted as a
real follow-up in the README's "Not yet implemented" list, not silently
left undocumented.

### 14.2 What was built

`_hzar_cline_tail_jax`/`_hzar_tail_alpha_jax` (JAX-traced ports of
`hycline.models.cline.hzar_cline_function`'s tail formula, verified
against the NumPy version to machine precision -- `atol=1e-10` -- for
every tail type: `tests/test_mcmc_numpyro.py::
test_jax_hzar_tail_matches_numpy_hzar_tail`) and
`_sample_tail_params_hzar` (HZAR-semantics priors: `delta` as a raw
distance, not a `[0,1]` fraction) were added alongside the existing
`_cline_tail_jax`/`_sample_tail_params`, with `hzar_tail: bool = False`
threaded through `_single_locus_model`/`_batch_model`/`fit_mcmc`/
`fit_mcmc_batch` and the CLI's `--hzar-tail` flag (already added to
`add_model_args`, shared by `fit-geographic`/`fit-geographic-multi`, in the classic-MLE pass).
Additive by design, the same pattern used throughout this project:
`hzar_tail` defaults to `False`, and the entire existing
`test_mcmc_numpyro.py`/`test_plotting_bayesian.py` suites were re-run
unmodified and confirmed still passing after every change in this
section.

### 14.3 A real prior-bound bug found during validation, not assumed correct

The first version of `_sample_tail_params_hzar` bounded `delta` at
`width_bound * 5`. Since `width_bound` here is *already* ~10x the
transect's own geographic span (`fit_mcmc`'s own default), this gave a
`delta` prior roughly 50x the data's span -- dramatically wider than the
`d_span * 5` bound `hycline.inference.mle_trait.
_bounds_for_trait_hzar` actually uses for the same parameter in the
verified classic-MLE case. This wasn't caught by inspection; it was
caught by validating against known ground truth: a real fit (true
`delta_l=15`) returned a posterior mean around 3000, with the sampler
drifting into an essentially unconstrained region of an
already-known-to-be-weakly-identified parameter (section 12.3) rather
than failing outright. Tightened to `width_bound / 2` (matching the
classic-MLE convention's effective scale) and re-validated -- the
posterior for `delta_l` is now appropriately imprecise (still wide, but
consistent with the documented weak-identifiability pattern, not off by
two orders of magnitude).

### 14.4 Six real bugs found in curve export/plotting -- present since `--hzar-tail` first shipped for classic MLE

The single most substantial finding from this pass. Every place in the
codebase that turns a fitted-parameter posterior/point-estimate into a
smooth curve for a CSV or a plot was checked systematically (a full
`grep` sweep for `cline_function(` calls, not just the paths touched in
this pass), and six of them were unconditionally using
`cline_function` (the original, non-HZAR tail model) regardless of
whether the fit itself used `hzar_tail=True`:

1. `cmd_fit`'s classic-MLE fitted-curve CSV export
2. `cmd_fit`'s Bayesian fitted-curve CSV export
3. `cmd_fit_multi`'s classic-MLE per-locus curve export
4. `_report_bayesian_multilocus_fit` (shared by `cmd_fit_multi`'s
   per-locus and plate-vectorized batched Bayesian paths -- section 13)
5. `hycline.viz.plotting.plot_fit`
6. `hycline.viz.plotting.plot_bayesian_fit`

**Bug #1 (classic-MLE single-locus curve export) had been present since
`--hzar-tail` was first built for classic MLE**, in an earlier pass of
this project -- undetected because that work's own tests checked the
*parameter estimates* (which were correct, since the fitting itself was
never affected -- only the curve re-evaluation downstream of it) but
never checked the *exported fitted-curve CSV's actual values* against
the right formula. This is a meaningful methodological lesson, not just
a bug: verifying that fitted parameters are correct is necessary but not
sufficient when a separate code path re-derives a curve from those
parameters afterward for display/export purposes.

All six were fixed the same way: branch on `getattr(fit, "hzar_tail",
False)` (the `getattr` default handles `FitResult`/`BayesianFitResult`
objects from paths that never touch `hzar_tail` at all, e.g. Phase 5's
GL-aware fitting, without needing every caller to know which result
types have the attribute). Both fixes (classic-MLE curve export and the
Bayesian equivalent) were independently confirmed to actually catch
their respective bugs, not just assumed to: each fix was temporarily
reverted, the corresponding regression test was confirmed to fail
cleanly on the reverted code, and then the fix was restored and
reconfirmed passing --
`tests/test_cli.py::test_cli_fit_hzar_tail_recovers_known_cline_simulated_from_hzar_formula`
(now also checks the curve CSV directly, not just the parameter
estimates), `::
test_cli_fit_bayesian_hzar_tail_recovers_known_cline_and_exports_correct_curve`,
`tests/test_plotting_bayesian.py::
test_plot_bayesian_fit_uses_hzar_cline_function_when_hzar_tail_true`.

**Everywhere else `cline_function(` appears was checked and confirmed
correctly untouched** -- Phase 5's GL-aware fitting/plotting
(`cmd_fit_gl`, `cmd_fit_gl_multi`, `hycline.likelihoods.
genotype_likelihood.log_likelihood_geographic`,
`plot_hierarchical_geographic_grid`'s GL-aware variant) and Component
A's hierarchical grid plot all call `cline_function` with either a
hardcoded `TailType.NONE` or a result type that was never extended with
`hzar_tail` in the first place -- these are correct as-is, not additional
instances of the same bug.

### 14.5 A separate, real, pre-existing inconsistency found but not fixed

While writing the MIRROR-tail regression test for this section,
`--tail mirror`'s parameter naming was found to differ between the
classic MLE and Bayesian JSON summaries: classic MLE labels the shared
tail parameter `delta_m`/`tau_m` in its JSON output (via
`hycline.models.cline.param_names`), while the Bayesian path's JSON
always labels it `delta_l`/`tau_l` (since NumPyro's model always samples
under those literal site names, even for MIRROR, and the CLI's Bayesian
JSON-writing code iterates over the literal
`("center", "width", "pmin", "pmax", "delta_l", "tau_l", "delta_r",
"tau_r")` tuple rather than going through `param_names()`). This is a
real, user-visible inconsistency -- not introduced by this pass, not
fixed by it either, since it's an unrelated naming-consistency issue
orthogonal to the `hzar_tail` extension this section is about. Noted
here, and in the README's "Not yet implemented" list, so it isn't lost.

## 15. The T-cline (Baird & Daley 2025): a structural fix for section 12's finding, not a workaround for it

Section 12.2 found that Bayesian tail-model fitting fails because the
tripartite model's sigmoid-core/exponential-tail join is a genuine
mathematical discontinuity -- gradient-based sampling struggles with the
kink itself, not with anything tunable. Baird & Daley (2025), read
directly (`references/`), give the same diagnosis independently and
propose a fix at the root: the T-cline family, built from the (possibly
noncentral) Student's-t distribution, continuous everywhere with no join
at all. This section covers what got built on top of that paper, checked
directly rather than assumed at each step: a symmetric implementation
(classic MLE only), a direct feasibility check of the two natural
follow-ups (asymmetric shape, Bayesian support) before writing any more
code for either, and the asymmetric implementation once one of those two
turned out feasible.

### 15.1 The symmetric T-cline: `t_cline_function`, classic MLE only

`hycline.models.cline.t_cline_function` implements the paper's `T(v)`
(Table 1): a Student's-t CDF with `t_shape` degrees of freedom, rescaled
so this project's own width convention (`width` = inverse of the maximum
gradient) holds exactly -- the same convention `_sigmoid_core`'s factor of
4 already encodes for the logistic case, generalized via
`_t_cline_scale(v) = scipy.stats.t.pdf(0, df=v)` (the standard Student-t's
own peak density; delegating to scipy's own PDF evaluation rather than a
hand-evaluated Gamma-function ratio, which can overflow for large `v`
before the ratio is taken).

Verified directly, not just implemented:
- Exact match to the Cauchy CDF at `t_shape=1` (Table 1's own stated
  exact case) to floating-point precision.
- Matches the paper's own MAD-0.001 calibration points: `t_shape=6.67`
  against the ordinary logistic/Bazykin sigmoid, `t_shape=116` against
  the Normal CDF.
- The width convention itself (gradient at `x=center` equals `1/width`)
  holds to 8 decimal places by finite-difference check across
  `t_shape` in [0.5, 500] and width in [1, 37.5].
- Cross-checked the scale-factor derivation against the reference
  implementation this is ported from (Baird & Daley 2023, `Clinalyse`,
  github.com/Studenecivb/clinalyse): its own
  `PreparationGosset._studentcdfgrad` computes the identical quantity in
  closed form.

Wired into `--tail t-cline` on `fit-geographic` and `compare-tails`
(`hycline.selection.aicc.compare_models` needed no changes at all --
`TailType.T_CLINE` is just another candidate to `fit_mle`). Classic MLE
only from the start; whether Bayesian support was even feasible was an
open question deliberately left open at this point rather than guessed
at, addressed directly in 15.3 below.

### 15.2 Real result on genuinely stepped synthetic data

150 sites, `n_alleles=200`, true `t_shape=1.5` (thick-tailed, well into
"stepped" territory): the T-cline recovered `t_shape=1.375 +/- 0.267`
(true value inside the 95% CI) alongside accurate `center`/`width`/
`pmin`/`pmax`, and AICc decisively preferred it over the plain logistic
core (881.85 vs. 917.76, delta-AICc ~36) -- confirming both that the
model fits correctly and that AICc-based model selection (as
`compare-tails` already does for the tripartite models) correctly
discriminates it when the signal is genuinely there.

### 15.3 Checking the two natural follow-ups directly, before writing more code

Two extensions were obvious candidates: the paper's fully general
asymmetric `T(v, alpha)` family (needed to exactly reproduce the paper's
own headline result on Fisher's 1937 wave-of-advance data, `T(8.365,
1.367)`, an 18-fold MAD improvement over the best tripartite fit), and
Bayesian/NUTS support (the case section 12.2's finding was originally
about). Rather than implement either speculatively, both were checked for
feasibility directly first.

**Asymmetric, classic MLE: confirmed feasible, but not via the route
originally assumed.** The obvious plan -- port Clinalyse's own closed-form
peak-location formula for the noncentral case
(`PreparationGosset.preparation_gosset_asy`, built from Hermite/confluent-
hypergeometric functions) -- turned out to be unnecessary. `fit_mle` calls
`scipy.optimize.minimize` without a `jac=` argument, so L-BFGS-B already
uses finite-difference gradients on the whole objective; nothing inside
the model needs to be analytically differentiable at all for classic MLE.
A plain numerical peak-finder (`scipy.optimize.minimize_scalar` on
`scipy.stats.nct.pdf`, bracketed around `t_skew` rather than a fixed wide
window) was checked directly against 48 `(t_shape, t_skew)` combinations
spanning this project's own existing `t_shape` bounds (0.05-500) and a
representative asymmetry range: clean, no NaN/overflow, ~0.8ms/call. (A
naive fixed-width bracket does hit a real `scipy.stats.nct.pdf` overflow
deep in the extreme tail at `t_shape=500` -- traced to the bracket
evaluating the PDF somewhere numerically meaningless, not to the
distribution being unusable; the alpha-centered bracket avoids it
entirely.)

**Bayesian/NUTS, for either shape: confirmed NOT feasible with the
JAX/NumPyro versions available (jax 0.10.2), for a more fundamental reason
than expected, and it turned out to block the *already-shipped symmetric*
case too, not just the asymmetric extension.** `jax.scipy.special.betainc`
-- which both a hand-built JAX Student's-t CDF and NumPyro's own official
`numpyro.distributions.StudentT.cdf()` are built on -- explicitly does not
support gradients with respect to its own shape parameters. Confirmed
directly, not just against a homemade implementation: calling `jax.grad`
on NumPyro's *own* `StudentT(df=...).cdf()` with respect to `df` raises
`ValueError: Betainc gradient with respect to a and b not supported`.
Since `t_shape` enters exactly there, and NUTS needs the gradient of the
log-density with respect to every parameter it samples (`t_shape`
included, for any Bayesian T-cline fit at all), this is a real capability
gap in the JAX/NumPyro ecosystem itself, not something more porting from
Clinalyse would close. (The `hyp1f1`-based closed-form asymmetric
peak-location formula has its own, separate numerical problem, found along
the way and now moot given the above: `jax.scipy.special.hyp1f1` diverges
from `scipy.special.hyp1f1` by a factor of ~150,000 at `t_shape=500`, well
within this project's own existing bounds.) `--tail t-cline --method
bayesian` gives an immediate, clear CLI error rather than attempting this
and failing deep inside NumPyro; `t_cline_function`'s own module-level
docstring and `HyCline_UserGuide_Geographic.md`'s tail-model section both
now explain *why*, citing this finding directly, the same way section 12.2
itself is cited rather than left as "doesn't work yet." A narrower
alternative if Bayesian T-cline credible intervals are ever wanted badly
enough: fix `t_shape`/`t_skew` at their classic-MLE-estimated values (not
sampled parameters) and let NUTS explore only `center`/`width`/`pmin`/
`pmax` -- sidesteps the broken gradient entirely, at the cost of not
propagating shape-parameter uncertainty into the credible intervals. Not
implemented.

### 15.4 Implementing the asymmetric case: two real problems found and fixed during verification, not assumed away

**A real bug: `scipy.stats.nct.cdf` itself returns NaN deep in its own
extreme tail.** Found while stress-testing the implementation across a
wide `(t_shape, t_skew, x)` grid, not in ordinary use: at `t_shape=100`,
`t_skew=10`, `nct.cdf` returns NaN over a real, non-astronomical range of
its argument (roughly `raw_t` in `[-92, -2.15]` in one traced case) --
not just asymptotically at +/-infinity. Left unhandled, a single NaN
would silently propagate into an entire fit's log-likelihood via
`binom.logpmf`, turning one far-tail site into a completely
uninformative optimizer target rather than the correctly-negligible one
it should be. The fix (`t_cline_function`, asymmetric branch): substitute
the correct limit (0 or 1, depending which side of the distribution's
mode the point falls on) for any NaN. Justified directly, not just
conveniently: the nearby, successfully-computed values in that same
region are already ~1e-16 to 1e-29, far below this project's own
likelihood floor (`_EPS=1e-10` in `hycline.likelihoods.binomial`), so the
true value being replaced is already negligible by construction, not
merely assumed to be.

**A real optimizer-robustness problem, confirmed on a known-truth
case.** A single L-BFGS-B start from the data-driven initial guess
converged to a badly wrong local optimum: on one synthetic case (true
`center=50, width=20, t_shape=3.0, t_skew=2.5`), the single-start fit
recovered `width=36.67` (center/width off by more than 80%) with a
negative log-likelihood 264 units worse than the true parameters' own.
Starting the identical optimizer from the true parameters converged
almost exactly, confirming the model and likelihood were correct and the
problem was purely the optimizer landscape.

The first fix tried -- randomized restarts drawn uniformly across the
*entire* parameter range, the exact strategy already working for the
tripartite tail model in `hycline.inference.mle_genotype_likelihood.
fit_mle_geographic_gl` (section 12 covers a related but more tractable
version of this same general problem) -- was checked directly and found
badly insufficient here: 100 such restarts found the true optimum's
basin of attraction **zero times out of 100** on the same representative
case. The root cause: `t_shape`'s bounds span multiple orders of
magnitude (0.05-500), and a linear-uniform draw over that range puts more
than 99% of restarts above `t_shape=5`, so the search essentially never
explores the genuinely "stepped" (small-`t_shape`) region a uniform-over-
bounds strategy nominally covers.

`hycline.inference.mle._fit_t_cline_multistart` uses two things checked
directly to actually work, not assumed to:
1. `t_shape` restarts drawn **log-uniformly**, not linearly, so orders of
   magnitude get sampled roughly evenly.
2. The asymmetric search is **anchored on a separately, robustly fit
   symmetric optimum** -- `t_skew=0` is an exact point in the asymmetric
   parameter space, not an approximation, so this is a legitimate
   starting region -- explored with both a `t_skew` grid and further
   log-uniform-shape/uniform-skew random restarts from the data-driven
   guess for diversity.

Checked across 6 representative `(t_shape, t_skew)` combinations spanning
a wide range: 4 recovered known truth cleanly (fitted negative
log-likelihood within ~3 units of the truth's own). The 2 that didn't
were both *very* thick-tailed (`t_shape` below 1) *and* strongly
asymmetric (`|t_skew| >= 5`) simultaneously -- a genuinely harder region
of parameter space, not a case this function silently mishandles: even
the symmetric anchor fit struggled there (e.g. one case's symmetric-only
negative log-likelihood was 710.29 against a true asymmetric value of
539.27), so the asymmetric search inherited an already-poor starting
point in those specific cases.

**A guarantee added by construction, not left to chance**: since
`t_skew=0` is a valid point in the asymmetric family, the symmetric
optimum (extended with `t_skew=0.0`) is always included as a literal
restart candidate, so the asymmetric fit's log-likelihood can never be
worse than the symmetric-only fit's on the same data -- verified by test
across all 6 cases above, including the 2 that didn't fully converge to
the true optimum. This closes off the confusing, mathematically-backwards
symptom that motivated the whole investigation (an asymmetric AICc worse
than the symmetric-only AICc for the same data, despite asymmetric being
a strict superset model).

### 15.5 A genuinely scientific finding, not an optimizer bug: `t_skew` can be weakly identified from real data

While diagnosing the optimizer problem above, one specific case revealed
something distinct from a convergence failure: a fit whose negative
log-likelihood (448.30) was *better* than the true generating
parameters' own (448.70) landed at `t_skew` between 13 and 20 across
repeated runs (the exact value varying run to run due to the multi-start
scheme's own randomization), against a true value of 2.5. This is not
the optimizer failing to find a good solution -- 448.30 genuinely is a
better fit to the observed (noisy) data than the truth itself achieves.
It means different `t_skew` values, over the sampled geographic range,
produce curves that are barely distinguishable in likelihood terms -- the
asymmetry parameter itself is weakly identified by this dataset, a
different phenomenon from (though it can compound with) the local-optima
problem section 15.4 fixes.

This mirrors the tripartite tail model's own already-documented weak
identifiability of `delta`/`tau` (`hycline.inference.mle_genotype_
likelihood.fit_mle_geographic_gl`'s own test suite explicitly declines to
check those parameters against ground truth for exactly this reason).
`fit_mle` now warns (`UserWarning`) when the fitted `t_skew` lands at its
search bound -- the practical, checkable symptom of this weak
identifiability, verified to fire correctly with a forced boundary result
in `tests/test_mle.py::
test_fit_mle_t_cline_asymmetric_warns_when_t_skew_hits_bound` rather than
left to stochastic luck to exercise. `center`/`width`/`pmin`/`pmax` and
the AICc comparison against the symmetric-only fit remain meaningful even
when this fires; the `t_skew` point estimate itself should be treated
with real caution.

### 15.6 Known rough edges

- **Asymmetric fits are meaningfully slower**: ~8s vs. ~1s for a
  symmetric T-cline fit on a representative dataset in this
  investigation, due to the extra restart phases `_fit_t_cline_
  multistart` runs. A real cost of the robustness fix in 15.4, not
  hidden -- documented in the CLI help text and user guide directly.
- **The `n_restarts` defaults (8 for both the log-uniform shape restarts
  and the `t_skew` grid) are reasonable but not rigorously tuned** --
  checked to meaningfully improve reliability over both a single start
  and the naive uniform-restart strategy, not swept for an optimal value.
  Mirrors the identical caveat already noted for `fit_mle_geographic_
  gl`'s own `n_restarts=8` default in section 12.5.
- **Not wired into any other fitting path**: genotype-likelihood input
  (`fit_mle_geographic_gl`), trait clines (`fit_mle_trait`), hierarchical
  pooling (Components A/D), and `compare-tails`'s asymmetric variant are
  all explicitly out of scope for this pass. `fit-geographic` with
  genotype-likelihood input gives a clear, immediate CLI error for `--tail
  t-cline` rather than a confusing crash deep inside a code path that
  never learned about `TailType.T_CLINE` at all.
