# HyCline Backend Notes: Phase 1 (Classic / Maximum-Likelihood)

Companion to `BACKEND_PHASE2.md` (the Bayesian/MCMC backend). Where that
document explains the Bayesian machinery, this one explains the "classic"
fitting path -- the one designed to reproduce numbers that match what HZAR
itself, and papers that used HZAR, would report.

---

## 1. What "classic" means here, precisely

HZAR's own internal fitting procedure is actually Metropolis-Hastings MCMC
(see `hzar.doFit` in the R source) -- it is *not* a pure optimizer. But the
numbers that end up in papers and in HZAR's own published comparison table
(Derryberry et al. 2014, Table 4) are point estimates (cline center, cline
width) with **2-log-likelihood-unit support intervals**, which is a
frequentist/likelihood-profile concept, not a Bayesian credible interval.
HZAR arrives at that point estimate via MCMC exploration of the likelihood
surface, but what's *reported* is the maximum-likelihood point and a
likelihood-ratio-based interval around it -- functionally indistinguishable
from what a direct optimizer would find, and this is confirmed by the
validation in the project README (`fit_mle_fixed_extremes` matches HZAR's
published center/width almost exactly using plain `scipy.optimize`, no
MCMC involved).

So "classic" in HyCline means: **find the MLE directly by numerical
optimization, and report uncertainty via a Gaussian (Hessian-based)
approximation** -- arriving at the same kind of answer HZAR's own MCMC
search converges to, by a more direct and much faster route. This is not
an approximation of HZAR's *method*; it's a different (faster, more
standard) route to the *same statistical target* HZAR's own numbers
represent. See section 5 for the one place this equivalence is inexact
(Hessian CIs vs. HZAR's 2-LL support intervals).

---

## 2. The model (identical to Phase 2)

Same generative model as `BACKEND_PHASE2.md` section 2:

```
p(x) = pmin + (pmax - pmin) / (1 + exp(-4 * (x - c) / w))
n_focal_i ~ Binomial(n_alleles_i, p(distance_i))
```

This is deliberate, not incidental -- Phase 1 and Phase 2 are two fitting
*procedures* for the same model, not two different models. That's what
makes it meaningful to compare their outputs directly (as the README's
three-way Manakin comparison does) rather than treating them as unrelated
analyses that happen to produce similar-looking numbers.

`hycline.likelihoods.binomial.log_likelihood` implements the raw binomial
log-likelihood used internally for optimization. A second function,
`hzar_style_log_likelihood`, exists purely for *reporting* results in units
comparable to HZAR's own published log-likelihoods -- see section 6, this
is one of the more interesting things this scaffold turned up.

---

## 3. Why `scipy.optimize.minimize` with L-BFGS-B

The cline log-likelihood surface (as a function of center, width, pmin,
pmax) is smooth and low-dimensional (2-6 parameters), with box constraints
that have an obvious, physically meaningful form (pmin/pmax in [0,1], width
positive, center within some sane range of the transect). L-BFGS-B is the
standard choice for exactly this shape of problem: gradient-based
(quasi-Newton, so no analytic gradient needed -- SciPy computes it
numerically), respects box bounds natively, and converges in a handful of
iterations for a model this size.

**Initial guesses matter more than the optimizer choice here.** A cline
log-likelihood surface, while smooth, is not globally convex -- there can
be shallow local regions (e.g. a "flat" solution with center outside the
sampled range and enormous width, mimicking a a weak monotonic trend
rather than a true sigmoid). `_initial_guess()` in `mle.py` builds a
data-driven starting point (pmin/pmax from the extreme quintiles of
observed frequency, center from where observed frequency crosses the
midpoint, width as a quarter of the transect span) specifically to land the
optimizer in the right basin on the first try, rather than relying on
L-BFGS-B's local convergence to escape a bad start. This is a real
practical concern, not a theoretical nicety -- an earlier version of this
scaffold without data-driven starting points occasionally converged to
degenerate wide-flat fits on sparse test data.

---

## 4. Two fitting modes: `fit_mle` vs. `fit_mle_fixed_extremes`

### 4.1 The two modes

These correspond to two of HZAR's own named model variants (Brumfield et
al. 2001's "Model 2" and "Model I" respectively, both without tails):

- **`fit_mle`**: pmin and pmax are free parameters, estimated jointly with
  center and width (4 free parameters total). This is the more flexible
  fit, and is what `hycline.selection.aicc.compare_models` uses by default
  alongside the tail variants.
- **`fit_mle_fixed_extremes`**: pmin and pmax are fixed to the *observed*
  frequencies at the transect's geographic extremes (not fixed to the
  theoretical extremes 0/1 -- that's a different, third HZAR mode not yet
  implemented here), leaving only center and width free (2 parameters).
  This is the mode that reproduces HZAR's published Ada-locus result in
  the project's validation tests, and is generally the better comparison
  point for older HZAR-based papers, many of which report this as their
  preferred model by AICc (as HZAR's own paper does for this exact locus).

Both return the same `FitResult` type; `n_params`/`k` differs correctly
between them (2 vs 4), which matters for AICc-based comparison between
them -- see section 7.

### 4.2 `n_edge_sites`: the single-site fragility fix, made available here too

`fit_mle_fixed_extremes`'s pmin/pmax derivation originally anchored on a
*single* site at each geographic extreme -- HZAR's own convention, and
what the Manakin cross-validation above was checked against. Building
Phase 4 Component A's hierarchical model (`BACKEND_PHASE4.md` section 3)
found and fixed a real problem with that convention: a single site's
observed frequency is fragile to exactly the kind of sampling noise a
binomial likelihood is expected to produce. A true pmin near 0.05 can
easily draw an exact 0/n at one site -- a few-percent event, not a rare
one, at realistic sample sizes -- and when that happens, the fixed
asymptote gets locked at 0.0 instead of 0.05, forcing the optimizer to
substantially widen the fitted cline to reconcile the rest of the data
with the wrong anchor.

That fix (averaging the `n_edge_sites` sites nearest each end, pooling
counts -- `sum(n_focal)/sum(n_alleles)` -- rather than reading one site's
frequency) was deliberately scoped to the hierarchical model only in
that pass, specifically because changing shared Phase 1/2 code risked
silently altering the already-validated Manakin numbers, and doing that
without direct evidence it was needed there too would have been the
wrong kind of change to make quietly.

It's now available here as well, as an explicit, opt-in parameter --
`_derive_pmin_pmax_single_locus` in `hycline/inference/mle.py`, exposed
as `fit_mle_fixed_extremes(..., n_edge_sites=N)` and, on the CLI, as
`--n-edge-sites` (default 1) on `fit-geographic`/`fit-geographic-multi`. **The default (1)
reproduces the original single-site convention exactly, not just
approximately** -- verified directly (`tests/test_mle.py::
test_fit_mle_fixed_extremes_default_n_edge_sites_matches_original_behavior`),
and the Manakin cross-validation tests were re-run unchanged after this
addition to confirm nothing shifted. A user who suspects the fragility
(e.g. small per-site sample sizes, or a pmin/pmax fit that looks
suspiciously wide compared to the raw data) can opt into the more robust
version by passing a higher value -- 3, matching the hierarchical
model's own default, is a reasonable starting point.

**Demonstrated to actually help, not just theoretically motivated**: a
synthetic dataset (true center=50, width=15, pmin=0.05, pmax=0.95, n=20
alleles/site) with an unlucky zero-count draw forced at the true extreme
site was fit both ways. `n_edge_sites=1` (the default) reproduced the
pathology exactly as described above: `pmin` locked at 0.0, fitted width
42.9 (roughly 3x the true value). `n_edge_sites=3` recovered `pmin=0.033`
(close to the true 0.05) and a fitted width of 11.8 (much closer to the
true 15) -- see `tests/test_mle.py::
test_fit_mle_fixed_extremes_n_edge_sites_mitigates_unlucky_extreme_draw`
for the regression test, though note it checks the *direction and
relative size* of the improvement rather than these exact numbers, since
the specific values depend on the random draw.

The same `n_edge_sites` parameter was added to the CLI's
`_resolve_fixed_extremes` (used to pre-derive pmin/pmax for the Bayesian
path too, since `fit_mcmc` -- unlike `fit_mle_fixed_extremes` -- doesn't
derive them itself), reusing the exact same
`_derive_pmin_pmax_single_locus` helper rather than a separate inline
copy, so the classic and Bayesian CLI paths can never silently disagree
about how pmin/pmax get derived.

---

## 5. Uncertainty: Hessian-based confidence intervals

`fit_mle`/`fit_mle_fixed_extremes` compute standard errors from the
numerical Hessian of the negative log-likelihood at the optimum
(`_numerical_hessian`, a small central-difference implementation -- fine
for the handful of parameters here, not something that needs a dedicated
autodiff dependency the way the Bayesian backend's gradients do). Standard
errors are then `sqrt(diag(inverse Hessian))`, and 95% CIs are the usual
`estimate +/- 1.96 * SE` Gaussian approximation.

**This is not the same interval HZAR itself reports.** HZAR's published
"2-log-likelihood-unit support" intervals come from directly profiling the
likelihood surface (finding where log-likelihood drops by 2 units from the
maximum) rather than assuming local Gaussian curvature. The two coincide
well when the likelihood surface really is close to quadratic near the
optimum (true for `center` in the Manakin validation, less true for
`width`, whose Hessian-based CI in that validation includes a physically
impossible negative lower bound -- see the printed output of
`examples/validate_manakin.py`). This is flagged rather than hidden: the
Hessian CI is a fast, useful sanity check, but the Bayesian backend's
credible intervals (Phase 2) are the more trustworthy uncertainty
quantification when the likelihood surface is asymmetric or when a
parameter is only weakly identified by sparse data -- which is exactly
the situation the Phase 2 Manakin case study documents in
`BACKEND_PHASE2.md` section 9. A likelihood-profile-based interval (true
HZAR-style, not Gaussian-approximated) is a reasonable future addition to
this module if exact interval comparability with HZAR becomes important,
not just point-estimate comparability.

---

## 6. The `hzar_style_log_likelihood` finding

Worth restating here because it's squarely a Phase 1 backend detail, not
just a validation footnote: HZAR does not report the raw binomial
log-likelihood. Reverse-engineering HZAR's own R source
(`sampleLikelihoodMolecularPop` in `R/20-hzarPrep.R` of the CRAN `hzar`
package) showed it reports log-likelihood *relative to the saturated
(per-site perfect-fit) model* -- algebraically, the standard binomial
log-likelihood kernel minus a constant offset (with respect to model
parameters) equal to that same kernel evaluated at each site's own MLE
frequency.

Two consequences, both important if you're extending this module:

- **The offset doesn't affect optimization.** Since it's a constant given
  the data (not the model parameters), `fit_mle`/`fit_mle_fixed_extremes`
  correctly find the same MLE whether they use the raw binomial
  log-likelihood or HZAR's saturated-model-relative version -- this is why
  fitted center/width matched HZAR's published values almost exactly
  *before* this discrepancy was even understood, while the raw
  log-likelihood numbers didn't match at all.
- **It does affect AICc if you're trying to reproduce HZAR's exact
  published AICc numbers**, since AICc is `2k - 2*logL` plus a
  small-sample correction. `hycline.likelihoods.binomial.
  hzar_style_log_likelihood` exists specifically for this: computing AICc
  from it (rather than from `log_likelihood`) reproduces HZAR's published
  AICc-column numbers far more closely (see `tests/
  test_manakin_validation.py` and the discussion in the project README).
  `hycline.selection.aicc.compare_models`, by contrast, uses the raw
  binomial log-likelihood internally, which is completely valid for
  HyCline's own internal model ranking (the offset is identical across
  models fit to the same data, so it cancels in AICc *differences* and
  therefore in model *ranking*) but produces different absolute AICc
  numbers than HZAR's own table. If you need HyCline's absolute AICc
  numbers to match a specific HZAR-based paper's table entry for entry,
  use `hzar_style_log_likelihood` to recompute them; if you only need
  correct model *ranking* within HyCline's own comparisons, the default
  is already correct and this doesn't matter.

---

## 7. Model selection: `hycline.selection.aicc`

`compare_models` fits a set of nested tail-model variants and ranks them by
AICc -- HZAR's own `autofit` does the same thing across up to 15 model
variants (3 scaling modes x 5 tail configurations). This scaffold's
`compare_models` currently covers the tail-configuration axis
(`TailType.NONE`, `MIRROR`, `BOTH`) but not yet the scaling-mode axis
(fixed-to-0/1 vs. fixed-to-observed-extremes vs. estimated) as a single
unified sweep -- `fit_mle` and `fit_mle_fixed_extremes` cover two of
HZAR's three scaling modes as separate function calls rather than options
within one `compare_models` sweep. Unifying this into a single
"all-HZAR-model-variants" comparison function, closer to HZAR's own
`autofit`, would be a reasonable next step if broad model-selection parity
with HZAR (not just point-estimate parity on a chosen model) becomes a
priority.

---

## 8. When to use "classic" vs. Phase 2's "bayesian"

Both are now reachable through one entry point,
`hycline.api.fit_cline(transect, method="classic" | "bayesian", ...)` --
see that module's docstring for the exact dispatch logic. Some concrete
guidance on which to reach for:

- **Use `"classic"`** when you want numbers directly comparable to a
  published HZAR-based study (same statistical target, much faster to
  compute, no MCMC convergence diagnostics to check), when you're doing
  quick exploratory fits, or when you're batch-fitting very large numbers
  of loci where the Bayesian backend's per-fit overhead would add up
  (though see `BACKEND_PHASE2.md` section 4 for the plate-vectorized batch
  path that mitigates this for the Bayesian backend too).
- **Use `"bayesian"`** when the likelihood surface might be poorly
  approximated by a Gaussian (see section 5's caveat on Hessian CIs -- this
  is especially likely for `width` with sparse data, exactly the situation
  documented in `BACKEND_PHASE2.md` section 9), when you want to propagate
  full posterior uncertainty into a downstream calculation rather than
  just a point estimate, or when you're building the Phase 4 hierarchical
  multi-locus model, which is a natural Bayesian extension and has no
  obvious classic/frequentist equivalent.

Neither is "the modern one deprecating the other" -- they answer the same
question with different assumptions about how to quantify uncertainty, and
`fit_cline` is written so that choice is a single keyword argument, not an
architecture decision.

---

## 9. Data-layer additions: coordinates, individual counts, multi-locus

Three gaps identified after Phase 1/2 were already built and validated,
addressed together since they all live in `hycline/data/` and were driven
by the same underlying goal: accepting data closer to the form field
biologists actually have it in, rather than requiring pre-processing
outside the tool first. None of these change the model or the fitting
math (sections 1-8 above still describe exactly what happens once a
`AlleleFreqTransect` exists) -- they're entirely about getting to that
`AlleleFreqTransect` from more different starting shapes of input.

### 9.1 Coordinates -> 1D transect distance (`hycline.data.geo`)

HZAR itself expects a pre-computed `distance` column -- its own bundled
`manakinLocations` dataset is documented as "distance from locality A,"
i.e. computed by hand before HZAR ever sees it. Most users start from GPS
coordinates. `hycline.data.geo` adds two ways to collapse 2D coordinates
to the 1D distance the cline model needs:

- **`distance_from_reference`** (CLI default, `--transect-method reference`):
  great-circle (haversine) distance from every site to one reference
  point, defaulting to the first row -- this is exactly HZAR's own
  convention (their "distance from locality A" is literally this).
- **`project_onto_transect_axis`** (`--transect-method pca`): projects all
  sites onto their best-fit line (via PCA on a local equirectangular
  approximation), which is more robust when sampling doesn't radiate
  cleanly from one natural endpoint (e.g. a zigzagging transect).

**This is explicitly not the 2D spatial cline surface extension** from the
original architecture plan (fitting a cline as a smooth function over 2D
space -- a genuinely harder, still-unaddressed research problem, deferred
on purpose). This is a preprocessing convenience: reduce 2D input down to
the same 1D distance a hand-measured transect tape would give, so the
existing validated 1D model can be used unchanged. Anyone reading fitted
results should know they're looking at a 1D cline fit to a collapsed
coordinate, same as if they'd measured the transect by hand -- the
collapse method matters for how well that 1D approximation reflects the
real sampling geometry, and is worth stating explicitly in any resulting
write-up, the same way a hand-measured transect's assumptions would be.

### 9.2 Individual counts instead of allele counts (`--n-ind-col`/`--ploidy`)

HZAR's own bundled examples are already in allele-count terms, so this
isn't a HZAR-parity gap -- but plenty of real datasets record individual
sample sizes, and silently forgetting to double for a diploid autosomal
marker is an easy, easy-to-miss error. `n_alleles = n_ind * ploidy` is
computed by `hycline.data.io._resolve_n_alleles`, shared by both the
single-locus and multi-locus readers, with `--n-alleles-col` and
`--n-ind-col` treated as mutually exclusive (an explicit error if both are
given, rather than silently preferring one).

### 9.3 Multi-locus wide-format CSVs (`hycline.data.io.read_multi_locus_wide_csv`)

This was the highest-priority gap: a single-locus-only data layer doesn't
match the actual shape of most real HZAR-style datasets (HZAR's own
Manakin example has 7 loci in one file), and would have needed
retrofitting anyway once Phase 3 (genomic clines, inherently multi-locus)
arrived -- so it was worth fixing before Phase 3 rather than after.

The reader auto-detects loci from paired column names (`<locus>.nSamples`
+ `<locus>.A` by default) -- chosen to match HZAR's own bundled dataset
naming exactly, so an HZAR CSV should be readable with zero configuration
(validated directly: `examples/manakin_multilocus.csv` is HZAR's own real
`manakinMolecular.csv`, converted to a standard comma-delimited CSV with
no other changes, and `read_multi_locus_wide_csv`'s defaults read all 7
loci from it correctly -- see `tests/test_data_io.py::
test_read_multi_locus_wide_csv_matches_real_hzar_manakin_dataset`).

One deliberate correctness decision: missing data is handled **per locus,
not per file**. If locus X has a blank cell at site 5 but locus Y has data
there, only locus X's transect drops that site -- requiring every locus to
have complete data at every site would throw away usable data for no
reason, and real multi-marker datasets commonly have this pattern (a
marker that failed to amplify at one locality doesn't mean every other
marker's data from that locality is unusable). The CLI's `fit-geographic-multi`
subcommand also treats one locus failing to fit as a per-locus event, not
a batch-aborting one -- see `hycline/cli.py::cmd_fit_multi` and its
`UnsupportedFitConfiguration` exception handling.

### 9.4 The VCF + metadata converter

Originally deferred (see the note this replaces, still worth reading for
why it wasn't built immediately: a real, valuable gap that wasn't
blocking anything at the time). Now built:
`hycline.data.vcf.read_vcf` (a parser) plus four writer functions, and
two CLI commands, `hycline generate-geographic-input`/`convert-vcf-genomic`.

**No `pysam`/`cyvcf2` dependency, deliberately.** Both are C-extension
packages needing compilation against `htslib` -- a real installation
risk in a sandboxed or otherwise constrained environment, with no
guarantee they build cleanly. VCF is a well-specified, line-oriented
tab-delimited text format; the subset this project actually needs
(biallelic SNP genotype calls, optionally Phred-scaled genotype
likelihoods) is fully parseable with the standard library alone (`gzip`
+ string splitting). This mirrors the same reasoning that led every other
reader in this project to be a plain-text CSV parser rather than reaching
for a heavier dependency.

**Scope, stated plainly rather than discovered by trial and error**:
supports biallelic SNPs (single-base REF, single single-base ALT) with a
`GT` FORMAT field, and optionally `PL` (the modern GATK/bcftools/ANGSD
genotype-likelihood convention, converted via the same
`phred_to_log_likelihood` every other GL-consuming part of this project
already uses). Does not support indels, multiallelic sites, or the older
log10-scaled `GL` FORMAT field -- these are common in real VCFs and
skipped (with a count reported to the user), not guessed at, since
handling them correctly would need a real modeling decision (which ALT
is "focal" at a multiallelic site? how does a 3+-genotype-class site fit
this project's inherently biallelic HWE marginalization?) rather than an
arbitrary default.

**Output matches each existing reader's own default column convention
exactly**, so converted files work with zero extra CLI flags on the
fitting side -- `generate-geographic-input --genotype-likelihoods` writes
`.gl0`/`.gl1`/`.gl2` suffixes (matching `read_gl_multi_locus_wide_csv`'s
defaults), the hard-count mode writes `.nSamples`/`.A` (matching this
project's own `read_multi_locus_wide_csv` defaults, which in turn match
HZAR's own bundled dataset convention -- section 9.3), and the genomic
writers match `read_genomic_cline_multi_locus_wide_csv`/
`read_genomic_gl_multi_locus_wide_csv`'s `.n`/`.freq` and `.gl0`/`.gl1`/
`.gl2` defaults respectively. Getting the `.nSamples` semantics right
required checking, not assuming: HZAR's own naming suggests an
individual/sample count, but this project's existing reader (already
validated against real published HZAR results) treats `.nSamples` as a
direct *allele* count, not one needing a separate ploidy multiplication
-- confirmed by reading the reader's own code rather than guessing, before
writing the aggregation math that needed to match it exactly.

**A site-aggregation correctness requirement, unique to the geographic
hard-count writer**: the other three writers are one-row-per-individual
(no aggregation), but geographic hard counts need individuals pooled by
site. Every individual sharing a site must have the *same* distance value
in the metadata -- checked explicitly, raising a clear error naming the
offending site rather than silently picking one distance value if they
disagree.

**Two real bugs found and fixed during this component's own build, not
shipped and discovered later:**
1. A locus with genotype-likelihood data missing for *every* individual
   (no `PL` field anywhere) originally produced an all-`NaN` GL column
   triple in GL-format output -- which `read_gl_multi_locus_wide_csv`
   would then reject outright (it requires at least one complete row per
   locus), turning one uninformative locus into a hard failure for the
   whole file. Fixed by excluding such loci from GL output up front, with
   a clear report of which loci were dropped and why -- confirmed
   directly: `tests/test_vcf.py::
   test_write_geographic_gl_csv_excludes_loci_with_no_pl_data`.
2. If *every* locus in a file lacked PL data (the degenerate case of the
   above), the command still reported success and left a locus-free,
   effectively useless CSV on disk. Fixed to detect this and fail loudly
   (exit code 2, explanatory message, no file written) rather than a
   silently hollow "success" -- confirmed directly: `tests/test_cli.py::
   test_cli_convert_vcf_genomic_gl_no_pl_data_gives_clear_error_and_no_file`.

**Real-data validation, not just unit tests on hand-crafted fixtures**:
full VCF -> CSV -> fit pipelines were run end to end on freshly simulated
synthetic data with known true parameters, through the actual CLI (not
just the Python API) -- a geographic cline (true center=70, width=25)
recovered as center=75.0 through `generate-geographic-input` piped into
`fit-geographic-multi`; a genomic barrier locus (true v=4.0) recovered as v=3.99
through `convert-vcf-genomic` piped into `fit-genomic-multi`. Both are
locked in as regression tests (`tests/test_cli.py::
test_cli_convert_vcf_geographic_recovers_known_cline`, `::
test_cli_convert_vcf_genomic_recovers_known_barrier_locus`), not just
manual one-off checks.

### 9.5 Known rough edges

- **No multiallelic or indel support**, by design (section 9.4) -- a
  real, common feature of production VCFs this converter doesn't handle,
  not an oversight. Filtering upstream with `bcftools view -m2 -M2
  -v snps` before conversion is the practical workaround -- and, since
  this was originally the only place that guidance lived, the CLI now
  also prints it directly whenever a VCF actually has skipped
  multiallelic/indel sites (not just for a clean file), so the guidance
  reaches the user at the moment it's relevant rather than only in this
  document. Verified: `tests/test_cli.py::
  test_cli_convert_vcf_geographic_gives_actionable_guidance_on_skipped_sites`
  and its counterpart confirming the note stays silent when nothing was
  skipped.
- **No `GL` (log10-scaled) FORMAT field support**, only `PL`
  (Phred-scaled) -- `PL` is the dominant modern convention (GATK,
  bcftools, ANGSD all use it), but older tools or pipelines using the
  original VCF-spec `GL` field would need a conversion step first.
- **No streaming/chunked processing** -- `read_vcf` loads the full parsed
  result into memory as NumPy arrays. `--max-variants` offers a safety
  valve for a quick look at a large file, but there's no way to process a
  genome-scale VCF in bounded memory without pre-filtering it (e.g. with
  `bcftools` to a manageable locus set) first.
- **No direct hybrid-index estimation from a VCF** -- `convert-vcf-genomic`
  expects hybrid index already present in the metadata (or use `hycline
  estimate-hybrid-index` first, on a separately-prepared genotype-dosage
  CSV, then merge its output into the metadata by individual name); there's
  no single command that goes straight from a VCF's own reference/parental
  samples to hybrid index without that intermediate step.

### 9.6 Filtering to ancestry-informative markers before conversion (`hycline.data.filter_aims`, `hycline filter-aims`)

Most genome-wide SNP data is uninformative for cline analysis -- the
overwhelming majority of loci don't differ meaningfully between the two
parental populations flanking a hybrid zone, and including them dilutes
real signal rather than adding power. Pre-filtering to strongly
differentiated loci (ancestry-informative markers, AIMs) before any
downstream conversion or fitting is standard practice in this field, and
was previously something a user had to do entirely outside this project.
`hycline filter-aims` does that filtering directly, on the two real
upstream data formats the rest of this project's own pipeline is built
around.

**Two input formats, one filtering core, VCF and beagle only.**
`hycline.data.filter_aims` accepts a VCF or ANGSD's own beagle-format
genotype-likelihood output (`angsd -doGlf 2`, the standard input format
for tools like NGSadmix/PCAngsd), auto-detected from the `--input` file
extension. A genotype-dosage CSV path (`filter_aims_csv`) existed
originally and is kept for any direct Python caller, but is no longer
part of the CLI or the documented pipeline -- VCF/beagle became the only
supported starting formats once `filter_aims_vcf` was built (see the
note on this below), matching the rest of this project's own converters.
Both remaining paths share one filtering function, `compute_aim_mask`: a
locus is retained if one parental population's allele frequency is at or
above `--high-threshold` (default 0.9) and the other's is at or below
`--low-threshold` (default 0.1), checked in *either* direction (it
doesn't matter which parental population happens to carry the
higher-frequency allele).

**Allele frequency from genotype likelihoods is a simplified, stated
estimator, not full MLE.** For VCF (hard-called) input, a locus's
population frequency is the ordinary mean dosage over that population's
labeled individuals, divided by ploidy -- exact, no approximation. For
beagle (genotype-likelihood) input, there's no hard call to average
directly, so each individual's *expected* focal-allele dosage is used
instead: treating the three genotype-likelihood values as a posterior
under a flat prior (normalized per individual, in case the beagle
values aren't already normalized to sum to 1), `E[dosage] = 0*p0 + 1*p1
+ 2*p2`. This is a standard, defensible estimator for a filtering step,
but it is *not* the full per-site maximum-likelihood allele-frequency
estimator ANGSD's own `-doMaf` implements (which properly accounts for
the joint likelihood across all individuals via EM, rather than
averaging independent per-individual posterior means) -- stated plainly
here rather than implied to be equivalent. For the purpose this function
serves (deciding whether a locus clears a coarse informativeness
threshold, not producing a publication-grade frequency estimate), the
simplified estimator is adequate; a user wanting ANGSD's own frequency
estimates for other purposes should still use `-doMaf` directly.

**A separate metadata file, not embedded population labels.** Unlike
`estimate-hybrid-index`'s default mode (which reads population labels
from a column within the same genotype file), this always takes a
*separate* metadata CSV mapping sample names to population labels --
matching the shape of a real analysis where the same genome-wide dataset
commonly contains parental references *and* hybrid-zone individuals
together, and only the two reference groups are needed to judge a
locus's informativeness. Individuals with no metadata entry, or a label
other than the two parental population labels (e.g. hybrid-zone
individuals), are simply excluded from the frequency calculation, not
treated as an error -- `match_samples_to_populations` only raises if a
whole population ends up with zero matched individuals, or a required
column is missing.

**Output preserves the input format and modifies nothing except which
rows survive.** The beagle writer doesn't recompute or renormalize
anything -- it writes out the exact original genotype-likelihood values
for every retained marker, so a downstream tool reading the filtered
file sees the same numbers ANGSD originally produced. The VCF writer
(`filter_aims_vcf`, built once VCF became a directly-supported input
format -- see the note below) is stricter still: every retained line is
copied byte-for-byte from the original file, not re-derived from parsed
fields, so INFO annotations, QUAL/FILTER values, and any FORMAT field
beyond `GT` survive filtering exactly as written -- verified directly
with a test VCF carrying an extra `DP` (read-depth) FORMAT field, which
came through filtering completely untouched
(`tests/test_filter_aims.py::test_filter_aims_vcf_output_preserves_lines_verbatim`).

**Verified end to end, not just at the filtering step in isolation.** A
synthetic dataset with a known true cline (informative locus simulated
from real cline-predicted frequencies; several genuinely uninformative
loci simulated at ~0.5 frequency in both parental populations) was
written as a VCF, filtered, then piped through
`generate-geographic-input` and fit via `fit-geographic-multi` --
recovering the true generating center to within the expected
finite-sample margin
(`tests/test_cli.py::test_cli_filter_aims_vcf_end_to_end_recovers_known_cline`).
A real, honest finite-sample finding surfaced during this validation,
not smoothed over: with a modest reference panel (8 individuals per
parental population), a genuinely informative locus (true frequencies
0.02/0.98) came back at an empirical frequency of 0.875 in one
population -- narrowly missing the default 0.9 threshold by chance, not
because the locus wasn't real. This is an expected, honest property of
any threshold-based filter applied to finite reference samples, not a
defect in the filtering logic -- confirmed directly by inspecting the
actual computed frequencies rather than just observing the retained
count.

**Note: this section originally described a CSV+beagle design.** The
first version of `filter-aims` took a genotype-dosage CSV or beagle file
and fed a CSV-based conversion pipeline (`genotype-to-gl`/
`genotype-to-hard-counts`, since removed from the CLI). Once the
project's whole geographic-cline pipeline moved to VCF/beagle-only
input, `filter_aims_vcf` was built with the byte-exact preservation
guarantee described above, and the CSV path (and the two commands that
consumed its output) was dropped from the CLI -- kept in the Python API
only, for any caller who still has genotype-dosage data from some other
source.

### 9.7 Known rough edges

- **The beagle-to-HyCline converter is now built** (section 9.8) --
  this bullet previously flagged it as missing. Since further
  consolidated into `generate-geographic-input` (one command for both
  VCF and beagle geographic input -- see the note at the end of section
  9.6); `convert-beagle-genomic` remains a separate command for genomic
  clines.
- **The simplified per-individual-posterior-mean allele frequency
  estimator for beagle input** (section 9.6) is adequate for a coarse
  filtering threshold but is not ANGSD's own MLE-via-EM estimator --
  stated as a real, deliberate scope limit above, not revisited here.
- **No support for beagle files ANGSD wrote without real sample names**
  in a way that would let `--sample-col`-based metadata matching fail
  silently -- if ANGSD was run without per-sample identifiers, the
  beagle header's individual labels may be generic/uninformative (e.g.
  positional placeholders), and the metadata file must be constructed to
  match whatever labels are actually in the beagle header, which this
  tool has no way to verify against the user's own intent beyond a
  "no samples matched" error if the labels don't overlap at all.

### 9.8 The beagle-to-HyCline converter (`hycline.data.beagle`, underlying `generate-geographic-input`/`convert-beagle-genomic`)

`hycline.data.beagle`'s four writer functions (originally exposed via
their own `convert-beagle-geographic`/`convert-beagle-genomic` commands,
since consolidated -- geographic beagle input now goes through
`generate-geographic-input`, sharing one command with VCF input; see the
note at the end of section 9.6) close the gap section 9.7 flagged:
beagle output previously had
nowhere to go within this project beyond `filter-aims` itself.

**Mirrors the VCF converter's structure closely, as planned, with one
real simplification made necessary by the format itself.** All four
writers (geographic/genomic x GL/hard-count) follow `hycline.data.vcf`'s
exact join-then-write shape, output column conventions, and per-site
distance-consistency validation. The one real difference: beagle data is
*always* genotype-likelihood data (there's no VCF-style "was PL actually
present at this site" question, since a beagle file's whole reason for
existing is carrying that information), so the VCF converter's
"exclude/error on loci with no GL data" logic (`BACKEND_PHASE1.md`
section 9.4) has no beagle equivalent to build -- every marker in a
valid beagle file already has usable genotype likelihoods by
construction.

**Hard-count output needs a real, stated approximation beagle's GL-only
nature makes unavoidable.** Unlike the VCF converter (which can use the
VCF's own `GT` field directly for hard counts, independent of whether
`PL` is present), a beagle file has no hard genotype call at all --
`write_beagle_geographic_hard_count_csv`/`write_beagle_genomic_hard_count_csv`
derive one via the same argmax ("most likely genotype") convention this
project's plotting code already uses for displaying GL-based data
(`_beagle_most_likely_dosage`), discarding the genotype uncertainty
beagle data was built to carry. Stated plainly rather than hidden: this
is exactly the point-estimate-vs-uncertainty tradeoff Phase 5 exists to
avoid (`BACKEND_PHASE5.md` section 1) -- the GL-format writers are the
better fit for beagle input specifically, and the CLI's own help text
says so directly rather than treating the two output modes as
interchangeable defaults.

**Verified end to end against known ground truth, through the actual
CLI commands, not just the Python API in isolation.** (Command names
below -- `fit-geographic-gl-multi`, `fit-geographic-multi`,
`fit-genomic-gl-multi` -- are this feature's original ones and no longer
exist in the CLI; see BACKEND_PHASE5.md section 6's correction note for
the full account of the later merge into `fit-geographic`/`fit-genomic`.
The recovery numbers below are unaffected.) A synthetic
geographic cline (true center=70, width=25) was written as a beagle file
and converted both ways: `--genotype-likelihoods` output, fit via
`fit-geographic-gl-multi`, recovered center=70.4; site-aggregated hard-count output
(8 sites), fit via `fit-geographic-multi`, recovered center=70.5, width=24.2 -- both
essentially exact. A synthetic genomic barrier locus (true `v=4.0`)
converted with `--genotype-likelihoods` and fit via `fit-genomic-gl-multi`
recovered `v>1.5`. The complete originally-requested workflow --
`filter-aims` on genome-wide beagle data, straight into
`generate-geographic-input`, with no manual reshaping in between -- was
also confirmed working end to end
(`tests/test_cli.py::test_cli_filter_aims_then_generate_geographic_input_beagle_full_chain`).

**One real bug caught before it shipped, the same class of mistake made
once already in `hycline/data/vcf.py`**: a missing `import pandas as pd`
in `hycline/data/beagle.py`, caught immediately by the first test run
(a `NameError`, not a silent wrong answer) and fixed before any of the
validation above was actually meaningful.
