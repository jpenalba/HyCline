# HyCline Backend Notes: Phase 5 (Genotype-Likelihood-Aware Fitting)

Companion to `BACKEND_PHASE1.md`-`BACKEND_PHASE4.md`. Phase 5 adds the
ability to fit both geographic and genomic clines directly from genotype
likelihoods (as reported by variant callers -- GATK, bcftools, ANGSD, and
similar) rather than requiring a hard genotype call first. This was
flagged as future work as far back as `BACKEND_PHASE3.md` (motivating why
per-individual sample size is allowed to vary in genomic cline data) and
in the original architecture plan's list of advanced extensions.

---

## 1. Why this matters

Real hybrid-zone sequencing data is commonly low-coverage: RAD-seq,
ddRAD, and reduced-representation approaches routinely produce depth of a
handful of reads per individual per locus, and even whole-genome
resequencing studies are often run at modest depth to fit more
individuals within a sequencing budget. At low depth, the true genotype
at a given locus for a given individual is genuinely uncertain -- a
heterozygote can easily show zero reads of one allele purely by chance,
and a homozygote can show a spurious read of the "wrong" allele from
sequencing error. Modern variant callers report this uncertainty
explicitly as a genotype likelihood (GL) -- P(observed reads | genotype)
for each of the three possible diploid genotypes -- rather than
collapsing straight to a single hard call.

Everywhere else in this project (Phases 1-4), cline data is expected as
already-resolved allele or genotype counts, implicitly treating those
counts as known with certainty. That's the right assumption for
confidently-called genotypes, but it's the wrong one for low-coverage
data: taking the single most likely genotype at each locus and treating
it as ground truth throws away real information about how uncertain that
call actually was, information a proper cline fit should be using.

---

## 2. The model: marginalizing over genotype, not just accepting fuzzy counts

The correct way to incorporate genotype uncertainty is not to feed GL
values into the existing binomial likelihood as if they were fuzzy
counts -- it's to marginalize the likelihood over the unobserved true
genotype:

```
P(reads_i | p) = sum_{g=0}^{2} P(reads_i | genotype=g) * P(genotype=g | p)
```

where `P(reads_i | genotype=g)` is the sequencing-derived genotype
likelihood (the GL value itself) and `P(genotype=g | p)` is the
Hardy-Weinberg-expected probability of that genotype given the cline
model's *predicted* focal-allele probability `p` at that individual's
position:

```
P(g=0|p) = (1-p)^2      P(g=1|p) = 2p(1-p)      P(g=2|p) = p^2
```

For geographic clines, `p` is the geographic cline's predicted frequency
at that individual's distance (`hycline.models.cline.cline_function`);
for genomic clines, `p` is the genomic cline's predicted `phi(h)` at that
individual's hybrid index (`hycline.models.genomic_cline.
genomic_cline_function`). Everything else about the two models --
parameters, bounds, fitting approach -- is unchanged; only the
observation model connecting predicted allele frequency to the raw data
is different.

**This is a strict generalization of the project's previously-deferred
multinomial genotype likelihood**, not a separate feature that happens to
address a similar gap. If a genotype is called with certainty (one GL is
1, the other two are 0 -- a "hard call"), the marginalization above
collapses exactly to the standard multinomial-under-HWE genotype
likelihood that was listed in the README's "Not yet implemented" section
since Phase 1. Confirmed directly, not just argued:
`tests/test_genotype_likelihood_fitting.py::
test_hard_call_marginal_likelihood_matches_plain_hwe` checks that a
deterministic genotype likelihood reduces the marginal likelihood exactly
to `log(2p(1-p))` for a hard-called heterozygote. So implementing
genotype-likelihood support also closes that older gap as a side effect,
rather than leaving it open alongside a new, separate capability.

**A numerical note carried over from Phase 1's own history (the
saturated-model log-likelihood finding)**: genotype likelihoods do not
need to be normalized to sum to 1 across the three genotypes -- they are
likelihoods, `P(data|genotype)`, not genotype probabilities, so an
arbitrary multiplicative constant (equivalently, an additive constant in
log-space) per individual doesn't need to be removed. That constant
doesn't depend on the cline model's parameters, so it doesn't affect
where the likelihood is maximized -- the same reasoning that explained
why HZAR's saturated-model-relative log-likelihood convention didn't
change fitted center/width values in Phase 1's validation (BACKEND_PHASE1.md
section 6), now showing up again in a new context. It does mean the raw
log-likelihood magnitude reported by a GL-aware fit depends on whatever
normalization convention your GL data happens to use (e.g. GATK's PL
values are conventionally rescaled so the best genotype has PL=0, but
this isn't universal) -- fine for optimization and AICc-based comparisons
between GL-aware fits on the same data, not necessarily comparable across
differently-normalized datasets.

---

## 3. What was validated, and a real recalibration of expectations along the way

The original hypothesis going into this component was that naive
hard-genotype-calling at low depth would show measurable **point-estimate
bias** in fitted cline parameters (a "reference bias"-style effect,
analogous to what's discussed in the population-genomics literature for
low-coverage genotype calling). Building the validation surfaced that
this hypothesis, at least in the idealized simulation used here, was not
correct -- and the honest process of finding that out, rather than the
originally-expected result, is worth documenting.

### 3.1 What didn't hold up: point-estimate bias

Comparing GL-aware fits against naive-hard-call fits (taking the
maximum-likelihood genotype from the GL triple and feeding it to the
standard binomial likelihood as if certain) across sequencing depths from
1x to 30x showed **nearly identical point estimates** at every depth, not
a bias that shrinks with depth. Investigating why: a single random read
from a diploid individual samples one of their two chromosome copies
uniformly at random, so `P(read shows alt allele) = g_true/2` exactly
(ignoring sequencing error). That means the maximum-likelihood hard call
at very low depth (1 read) is, in expectation, an *unbiased* proxy for
the individual's true fractional genotype, even though it's a noisy
binary (0 or 1, never intermediate) rather than the true continuous-ish
value -- averaged across many individuals, that noise washes out without
leaving a systematic bias in the fitted cline's center or width.

This is a real, useful negative result, not a failure: the "reference
bias" phenomenon commonly discussed for low-coverage genotype calling in
the population-genomics literature is understood there to be
predominantly a *mapping/alignment* artifact (reads carrying the
non-reference allele mapping or being retained slightly less often than
reference-matching reads), which this project's simulation model
deliberately doesn't include (it models only symmetric sequencing error,
not asymmetric mapping success) -- so this simulation wasn't actually set
up to reproduce that specific literature phenomenon, and it's honest to
say so rather than claim a bias-reduction result that this validation
doesn't actually support.

### 3.2 What did hold up, cleanly: uncertainty calibration

The correct and more standard motivation for genotype-likelihood-based
methods in the population genomics literature (this is explicitly how
tools like ANGSD frame their own rationale) is proper uncertainty
propagation, not point-estimate correction -- and this is exactly what
the validation confirmed, checked via 95% confidence interval coverage
across 40 independent sequencing simulations at each of several depths
(`examples/demo_genotype_likelihood.py`, `tests/
test_genotype_likelihood_fitting.py::
test_gl_aware_ci_coverage_better_calibrated_than_hard_call_at_low_depth`):

| Depth | GL-aware coverage | GL-aware mean SE | Hard-call coverage | Hard-call mean SE |
|---|---|---|---|---|
| 1 | 82.4% | 3.91 | 66.7% | 2.79 |
| 2 | 94.3% | 3.32 | 86.1% | 2.87 |
| 3 | 94.6% | 3.16 | 89.5% | 2.93 |
| 5 | 97.4% | 2.84 | 85.0% | 3.11 |
| 8 | 97.5% | 2.69 | 92.3% | 2.65 |
| 15 | 100.0% | 2.63 | 97.5% | 2.63 |
| 30 | 100.0% | 2.61 | 100.0% | 2.61 |

The pattern is exactly what proper uncertainty propagation predicts:
**at low depth, hard-calling is measurably overconfident** -- its
reported standard errors are smaller than GL-aware's despite there being
more real uncertainty in the data, and its confidence intervals cover the
true value less often than their nominal rate (67% actual vs. 95%
nominal at depth 1). GL-aware fitting's coverage stays much closer to
nominal throughout, and **both approaches converge as depth increases**
(by depth 15-30, genotype calls are effectively certain either way, so
there's nothing left for the marginalization to correctly account for).
This is the validation result actually built into the test suite and
demo, in place of the originally-hypothesized bias story.

---

## 4. API surface

```python
from hycline.data.genotype_likelihood import (
    phred_to_log_likelihood, simulate_genotype_likelihoods,
)
from hycline.data.schema import GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData
from hycline.inference.mle_genotype_likelihood import (
    fit_mle_geographic_gl, fit_mle_genomic_gl,
)

# Geographic: one row per individual, sharing distance values within a site
transect = GenotypeLikelihoodTransect(distance=..., log_gl0=..., log_gl1=..., log_gl2=...)
fit = fit_mle_geographic_gl(transect)   # returns the same FitResult type as fit_mle

# Genomic: one row per individual
data = GenotypeLikelihoodLocusData(hybrid_index=..., log_gl0=..., log_gl1=..., log_gl2=...)
fit = fit_mle_genomic_gl(data)          # returns the same GenomicFitResult type as fit_genomic_cline_mle
```

Both fitting functions return the *same result types* used elsewhere in
the project (`FitResult`, `GenomicFitResult`) precisely so GL-aware and
hard-count fits are directly comparable with the same reporting code, the
same way the demo/tests do throughout section 3.

`phred_to_log_likelihood` converts from the Phred-scaled `PL` format most
variant callers actually output (`PL = -10*log10(likelihood)`) to the
natural-log values these functions expect. `simulate_genotype_likelihoods`
generates realistic synthetic GL data from known true genotypes and
sequencing depth -- used for this phase's own validation, and offered as
a general-purpose tool for exploring what a given depth/error-rate
combination's uncertainty actually looks like before committing to a real
sequencing plan.

### Per-individual data shape, not per-site aggregates

`GenotypeLikelihoodTransect` (geographic) holds one row per *individual*,
not one row per site the way `AlleleFreqTransect` does -- individuals
sampled at the same locality simply share the same `distance` value. This
is a real, necessary difference in data shape, not an inconsistency:
genotype likelihoods cannot be losslessly aggregated to a site-level
count the way hard genotype calls can (summing hard counts across
individuals at a site discards no information; summing genotype
*likelihoods* across individuals would be mathematically wrong, since
each individual's uncertainty needs to be marginalized separately before
any combination happens).

---

## 5. What's not included in this pass

- **Tail models are now supported in `fit_mle_geographic_gl`** (added
  after this section was first written, in a later pass -- see
  BACKEND_PHASE2.md section 12.3 for the design, including a real
  optimization-robustness fix (multi-start restarts) the tail model's
  non-smooth likelihood surface required). This bullet originally listed
  tail models as out of scope; kept here, corrected, rather than
  silently removed, so the history is honest about what changed and why.
- **No Bayesian GL-aware backend.** Both fitting functions here are
  classic MLE only, following the same phased pattern every other
  component in this project has used (MLE first, Bayesian as a tracked
  follow-up) -- see BACKEND_PHASE3.md section 4 for the same pattern
  applied to the non-GL genomic cline model.
- **Hierarchical (Phase 4) integration is done for both geographic and
  genomic clines.** `hycline.inference.mcmc_hierarchical_gl.
  fit_hierarchical_batch_gl` (Phase 4 Component D) combines Component A's
  partial pooling with this module's genotype-likelihood marginalization
  for geographic clines -- see BACKEND_PHASE4.md section 11 for the full
  build and a real finding along the way (a hard-called sparse locus's
  distorted estimate contaminates the shared population-level
  hyperparameters, not just its own posterior). `hycline.inference.
  mcmc_genomic_hierarchical_gl.fit_hierarchical_genomic_batch_gl` (Phase 4
  Component E) is the genomic-cline analog -- see BACKEND_PHASE4.md
  section 12 for a genuinely different finding: at a moderate depth
  contrast, genotype-likelihood-awareness makes no detectable difference
  for genomic clines under pooling, unlike the dramatic effect found for
  geographic clines.
- **No CLI wiring for Phase 5's own standalone single-locus fitting**
  (`fit_mle_geographic_gl`/`fit_mle_genomic_gl`). The multi-locus,
  hierarchical path built on top of this module's math (Phase 4
  Components D/E) does have CLI wiring -- `hycline fit-hierarchical-geographic-gl`/
  `fit-hierarchical-genomic-gl` -- see BACKEND_PHASE4.md section 14.
- **Ploidy is fixed at diploid** (3 genotype classes: 0, 1, 2 copies).
  Generalizing to arbitrary ploidy would need the HWE genotype
  probabilities generalized from the simple binomial-coefficient form
  used here to the general multinomial/hypergeometric form for higher
  ploidy -- not done in this pass.
- **The GL simulator doesn't model mapping bias** (section 3.1) -- only
  symmetric sequencing error. A more complete simulation including
  asymmetric mapping success by allele would be needed to actually test
  for the classic "reference bias" phenomenon, which this validation
  correctly found no evidence for under its own (simpler) assumptions.

---

## 6. CLI wiring for Phase 5's own standalone fitting

`hycline fit-gl`/`fit-genomic-gl` (single-locus) and `hycline
fit-gl-multi`/`fit-genomic-gl-multi` (independent multi-locus) expose
`fit_mle_geographic_gl`/`fit_mle_genomic_gl` from the command line --
closing the last CLI gap flagged in section 5 and in
`BACKEND_PHASE4.md` section 14.6 (Components D/E's own CLI wiring covered
the hierarchical, pooled path; this covers the independent path Phase 5
itself was originally built around).

### 6.1 New readers, reusing already-validated conversion logic

Single-locus genotype-likelihood CSV readers (`read_gl_csv`,
`read_genomic_gl_csv`) didn't exist before this -- only the multi-locus
wide-format readers built for Components D/E did (`BACKEND_PHASE4.md`
section 14.2). Both single-locus readers share the same `--phred`
conversion path (`hycline.data.genotype_likelihood.
phred_to_log_likelihood`) already verified for the multi-locus readers,
applied to single flat columns instead of per-locus suffixed ones.
Re-verified at the CLI level specifically, not just the reader level:
`tests/test_cli.py::test_cli_fit_gl_phred_matches_non_phred` runs the
full `fit-geographic-gl` command twice (once on natural-log columns, once on the
same data re-encoded as Phred-scaled PL and read with `--phred`) and
confirms the fitted parameters match to `1e-4` -- exercising the whole
CLI path (argument parsing, file reading, conversion, fitting), not just
the conversion function in isolation.

### 6.2 The multi-locus commands needed a real design decision Components D/E didn't

Independent multi-locus fitting (`fit-geographic-gl-multi`/`fit-genomic-gl-multi`)
does *not* require loci to share a common covariate vector the way
Components D/E's hierarchical batch models do (`BACKEND_PHASE4.md`
section 13.1) -- each locus is fit on its own, so different loci can
freely have different numbers of individuals or different distance
values. This meant `plot_hierarchical_geographic_grid`/
`plot_hierarchical_genomic_grid` (built for Components D/E, which assume
pre-stacked, shared arrays) genuinely could not be reused here, unlike
how Components D/E themselves reused Component A/B's dense-data
enforcement helper without modification.

This was caught mid-implementation, not designed correctly from the
start: an initial version of `cmd_fit_gl_multi` left unused variables and
a no-op conditional block where grid-plotting logic should have been,
rather than either building the right plotting function or leaving the
feature out cleanly. Corrected by building two new, correctly-scoped
functions -- `plot_gl_multi_locus_grid`/`plot_genomic_gl_multi_locus_grid`
in `hycline/viz/plotting.py` -- that take a dict of
`{locus_name: (data, fit)}` pairs and plot each locus's own array
directly, the same shape `plot_multi_locus_grid`/
`plot_genomic_multi_locus_grid` (the original, non-GL small-multiples
plotters from Phase 1/3) already use, rather than the newer
hierarchical-specific plotters' shared-array assumption.

### 6.3 Real-data validation

All four commands were run end-to-end on freshly generated synthetic
data (not just test fixtures): `fit-geographic-gl` and `fit-genomic-gl` each
correctly recovered known true parameters from a single locus (a
`center=50` geographic cline recovered within a generous but correct
range; a `v=4.0` genomic barrier locus recovered as `v=4.56`).
`fit-geographic-gl-multi`/`fit-genomic-gl-multi` each correctly fit three
independent loci, with `fit-geographic-gl-multi` also surfacing a real, legitimate
edge case rather than hiding it: one locus's Hessian-based standard
errors came back `None` (the existing, correct fallback path in
`fit_mle_geographic_gl` for a non-invertible Hessian, not a new bug),
and the CLI output reflects that missing information plainly rather than
fabricating a number.

### 6.4 Known rough edges

- **The hard-genotype convenience path is now built** (section 7) --
  this bullet previously flagged it as missing.
- **No Bayesian backend, tail-model support, or arbitrary ploidy** --
  unchanged from section 5's existing scope limits; this CLI wiring adds
  no new capability beyond what the underlying Python functions already
  do, only a way to reach them without writing Python.
- **`fit-geographic-gl-multi`/`fit-genomic-gl-multi` have no dense-data requirement
  and therefore no equivalent grid-plot limitation** Components D/E have
  -- worth knowing as a genuine point of difference, not an oversight,
  when choosing between the independent and hierarchical multi-locus
  paths for a given dataset.

---

## 7. The hard-genotype convenience path

`hycline.data.genotype_likelihood.hard_genotype_to_log_likelihood`,
`hycline.data.converter.convert_genotype_csv_to_gl_csv`, and the
`hycline genotype-to-gl` CLI command close the gap flagged in section 5
and the project README: using this project's GL-aware fitting on
*ordinary* hard-called genotypes (no real sequencing uncertainty)
previously meant constructing one-hot genotype likelihoods by hand.

### 7.1 The conversion is exact, not an approximation

`hard_genotype_to_log_likelihood` converts a genotype call to natural-log
likelihood 0 for the true genotype and a large negative value (`-1e10`,
not literal `-inf`, to keep CSV/JSON round-tripping well-behaved --
numerically indistinguishable from `-inf` in double precision, since
`exp(-1e10)` underflows to exactly `0.0` either way) for the other two.
This isn't a heuristic stand-in for certainty -- it's the literal
genotype likelihood a certain observation implies, and marginalizing it
(section 2) reduces *exactly* to the plain HWE likelihood a non-GL fit
would compute directly. Verified as an exact numerical match, not just
argued: `tests/test_genotype_likelihood_data.py::
test_hard_genotype_to_log_likelihood_marginalizes_to_exact_hwe_likelihood`
checks all three genotypes against the closed-form HWE probabilities to
`1e-9`. Missing genotypes (`NaN`) convert to a flat, uninformative
likelihood (log-likelihood `0` for all three candidates) rather than
raising an error, so a genotype matrix with some missing calls converts
in one pass without a separate masking step.

### 7.2 The CSV converter reuses the same reserved-column convention as hybrid-index estimation

`convert_genotype_csv_to_gl_csv` and `hycline estimate-hybrid-index`
(`BACKEND_PHASE3.md` section 9) both take the same shape of input -- a
wide genotype-dosage CSV with some reserved covariate columns
(`individual`, `distance`, `hybrid_index`, `population`, etc.) and the
rest treated as locus columns -- so they share
`hycline.data.converter.DEFAULT_RESERVED_COLUMNS`/`detect_genotype_loci`
rather than each having their own detection logic. This is a real,
useful side effect of building the two together: any wide genotype
CSV that works with one command's auto-detection behaves the same way
with the other, rather than each command having its own subtly different
notion of "which columns are loci."

### 7.3 Real-data validation, including a genuine full-pipeline check

Beyond the direct conversion-correctness tests (section 7.1), the
converter was checked for what actually matters in practice: does its
output work as real input to the rest of the project, not just look
correctly shaped. `tests/test_data_io.py::
test_converted_gl_csv_fits_correctly_with_fit_gl_multi` and `tests/
test_cli.py::test_cli_genotype_to_gl_output_chains_into_fit_gl_multi`
each simulate a real geographic cline's hard genotype calls, convert
them via the converter, and confirm the converted file --  read back in
through the *actual* multi-locus GL reader and fit through the *actual*
GL-aware MLE function -- recovers the true generating cline parameters.
This is a stronger check than validating the converter in isolation -- it
confirms the whole intended workflow (hard genotypes in, GL-aware fit
out) actually works end to end.

### 7.4 The reverse direction: `genotype-to-hard-counts`

`hycline.data.converter.convert_genotype_csv_to_hard_count_csv` and the
`hycline genotype-to-hard-counts` CLI command close the gap this
section's rough edges originally flagged: aggregating per-individual
genotype dosage into HZAR-style site-level allele counts
(`<locus>.nSamples`/`<locus>.A`, matching `read_multi_locus_wide_csv`'s
own default convention exactly -- BACKEND_PHASE1.md section 9.3), ready
for the original, non-GL `fit-geographic`/`fit-geographic-multi` commands with zero extra
flags.

**This direction genuinely can aggregate, unlike `genotype-to-gl`.**
Section 7.2 explains why `convert_genotype_csv_to_gl_csv` deliberately
stays per-individual: genotype likelihoods can't be losslessly summed
across individuals the way hard counts can, since marginalizing a sum of
likelihoods isn't the same as summing marginalized likelihoods. Going
from certain (hard-called) genotypes to a site-level count has no such
problem -- summing dosages and counting non-missing individuals loses
no information a hard call carries, so aggregation is the mathematically
correct operation here, not a lossy convenience.

**Every individual at a site must share the same distance value**, the
same requirement and error-reporting convention already established for
`hycline.data.vcf.write_geographic_hard_count_csv` (BACKEND_PHASE1.md
section 9.4) -- this function mirrors that one for plain genotype-CSV
input instead of a VCF, and was checked directly for numeric correctness
against hand-computed site aggregations (`tests/test_data_io.py::
test_convert_genotype_csv_to_hard_count_csv_matches_hand_computation`)
and for full-pipeline correctness by converting a known synthetic cline,
fitting the result through the real `fit-geographic-multi` CLI command, and
confirming the true generating center is recovered
(`tests/test_cli.py::test_cli_genotype_to_hard_counts_recovers_known_cline`).

### 7.5 Known rough edges

- **No validation that locus columns actually contain valid dosage
  values beyond the 0/1/2/NaN check** -- e.g. a column that's
  accidentally a site ID or other non-genotype numeric data would be
  silently "converted" if it happens to only contain values in {0, 1, 2}.
  `--loci` lets a user sidestep auto-detection entirely if this is a
  concern for a particular file. Applies to both `genotype-to-gl` and
  `genotype-to-hard-counts`.
