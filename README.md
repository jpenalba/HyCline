# HyCline (Phase 1 + Phase 2 + Phase 3 + Phase 4a/4b/4c/4d/4e + Phase 5 scaffold)

A Python package for hybrid-zone cline analysis -- a modern, from-scratch
successor to the R package HZAR (Derryberry et al. 2014), extended to
cover genomic clines, which HZAR doesn't do at all.

This covers **Phase 1** (MLE fitting, functional parity with HZAR's core
geographic-cline workflow), **Phase 2** (Bayesian/MCMC backend for
geographic clines), **Phase 3** (genomic clines -- locus ancestry vs.
genome-wide hybrid index, a Fitzpatrick 2013 logit-logistic model with no
HZAR equivalent), **Phase 4** (hierarchical Bayesian partial pooling
across loci for both geographic and genomic clines -- Components A and B
-- plus coupling/concordance analysis across loci -- Component C -- plus
hierarchical fitting combined with genotype-likelihood marginalization
for geographic clines -- Component D -- and its genomic-cline analog --
Component E -- see "Why hierarchical clines?" below for the biological
motivation), **Phase 5** (genotype-likelihood-aware fitting --
marginalizing over genotype uncertainty from low-coverage sequencing data
rather than requiring a hard genotype call first -- see "Why genotype
likelihoods?" below), and a **command-line interface** covering
everything in this project -- every Phase 4 component (A-E), Phase 5's
own standalone fitting, and hybrid-index estimation all have CLI
subcommands (see below).

## Command-line usage (recommended if you don't work in Python)

Everything below runs as one command with flags -- no Python scripting
required, and it works the same way on a laptop or as a cluster job.

### Setup (one time)

```bash
git clone <this-repo-url>
cd hycline
pip install -r requirements.txt              # core (classic/MLE fitting)
pip install -r requirements-bayes.txt         # optional, for --method bayesian
```

Then either install it as a command (`pip install -e .`, which gives you
the `hycline` command below) or just call the script directly with
`python run_hycline.py` instead of `hycline` everywhere -- both work
identically; the standalone script is there specifically so you can clone
the repo onto a cluster and run it immediately without an install step.

### Full documentation lives in five dedicated guides, not here

Rather than duplicate command-by-command documentation in two places
(this file and the guides below -- exactly the kind of duplication that
goes stale when commands change), the full, worked-example-driven
documentation for every pipeline lives in:

- **`HyCline_UserGuide_Geographic.md`** -- the geographic-cline pipeline
  end to end: `filter-aims` -> `generate-geographic-input` ->
  `fit-geographic`/`fit-hierarchical-geographic`, hard counts vs.
  genotype likelihoods, classic vs. Bayesian fitting, tail models and
  HZAR parity, fixed vs. estimated scaling.
- **`HyCline_UserGuide_Genomic.md`** -- the genomic-cline pipeline:
  `filter-aims` -> `estimate-hybrid-index` -> `generate-genomic-input` ->
  `fit-genomic`/`fit-hierarchical-genomic`.
- **`HyCline_UserGuide_Coupling.md`** -- across-locus coupling/
  concordance analysis (`coupling-geographic`/`coupling-genomic`),
  including the Firneno et al. (2023) theta estimator's real, honest
  scope and limitations.
- **`HyCline_UserGuide_Trait.md`** -- quantitative trait cline fitting
  (`fit-trait`/`fit-trait-multi`), including fitting genome-wide hybrid
  index itself as a geographic trait cline.
- **`HyCline_UserGuide_StudyDesign.md`** -- study design recommendations
  (reference panel size, sequencing depth, individuals-per-locus,
  loci-per-study) grounded in specific, verified findings from building
  and testing this project, cross-referenced back to the four guides
  above.

Every command and every block of output shown in those guides was run
for real against test data while writing them, not reconstructed from
memory -- they're the authoritative, up-to-date reference. `hycline
<subcommand> --help` is always accurate for any individual command's full
flag list.

## Choosing a fitting method: classic (HZAR-equivalent) or Bayesian

Reachable both from the command line (`--method classic`/`bayesian`,
above) and directly in Python. This is a first-class choice, not an
implementation detail you have to dig for. `hycline.fit_cline()` fits the
same model either way and returns whichever result type matches your
choice:

```python
import hycline

transect = hycline.AlleleFreqTransect(distance=..., n_alleles=..., n_focal=...)

# "classic": direct MLE, matches numbers from published HZAR-based studies
classic_fit = hycline.fit_cline(transect, method="classic",
                                 pmin_fixed=0.1, pmax_fixed=0.9)
print(classic_fit.summary())

# "bayesian": full posterior via NUTS, credible intervals instead of a
# Gaussian approximation -- useful when the likelihood surface is
# asymmetric or a parameter is only weakly identified by sparse data
bayes_fit = hycline.fit_cline(transect, method="bayesian",
                               pmin_fixed=0.1, pmax_fixed=0.9,
                               num_warmup=1000, num_samples=1000, num_chains=4)
print(bayes_fit.summary())
```

Both are the *same underlying cline model* -- see `BACKEND_PHASE1.md`
section 8 for concrete guidance on which to reach for and when. The
"classic" path exists specifically so results are comparable to studies
that used HZAR itself; the "bayesian" path is where this project extends
past what HZAR could do (proper posterior uncertainty, credible intervals
that don't assume local Gaussian curvature, and a foundation for the
hierarchical multi-locus model in Phase 4, below).

## A real limitation: Bayesian tail-model fitting

Worth reading before reaching for `--method bayesian --tail mirror` (or
`both`) -- this is a genuine, investigated limitation, not a temporary
rough edge expected to disappear with a patch.

**The finding**: fitting a tail model (`TailType.LEFT`/`RIGHT`/`BOTH`/
`MIRROR`) via NUTS (`fit_mcmc`/`fit_mcmc_batch`, i.e. `--method bayesian`)
has severe, genuine convergence difficulty -- R-hat values from 1.9 up to
3.5+ and effective sample sizes in the single digits are typical, even
at the CLI's default sample counts, and even large increases in warmup
(4x the default) sometimes don't help at all. This was investigated
carefully, not assumed: the exact same data, with the exact same priors,
converges perfectly (R-hat=1.00, thousands of effective samples) when
fit *without* a tail -- the only thing that changed between a clean fit
and a badly-failing one is the presence of tail parameters. More data,
tighter priors, and a simpler tail parameterization were all tried and
none fixed it.

**Why**: the tail model is continuous where the sigmoid core meets the
exponential tail, but not smooth there -- the slope generally differs on
either side of the join. Gradient-based samplers like NUTS assume a
locally smooth log-density; a kink is a real obstacle for the underlying
integrator, made worse here because the join's location depends on the
sampled `center`/`width` values themselves, so it moves as the sampler
explores rather than sitting in one fixed place to route around.

**What this means practically**:
- **Use `--method classic` for tail models.** Classic MLE does not have
  this issue. A related, more tractable version of the same difficulty
  does show up there too (occasional corner-solution convergence
  failures) -- fixed with an automatic multi-start optimization scheme,
  not something you need to configure.
- If you do use `--method bayesian` with a tail model, both the CLI and
  the underlying Python functions warn you explicitly, before and after
  fitting -- read the convergence diagnostics carefully rather than
  trusting the reported tail parameters at face value.
- The genotype-likelihood-aware path (`fit-geographic`/`fit-geographic`) supports
  tail models too, via the same classic-MLE approach -- no equivalent
  Bayesian GL-aware backend exists at all yet, so this limitation doesn't
  apply there.

See BACKEND_PHASE2.md section 12 for the complete investigation,
including the elimination process that ruled out "just needs more
compute" as an explanation, and four real, previously-shipped bugs this
work surfaced and fixed elsewhere in the codebase along the way.

## Why hierarchical clines? The biological motivation, not just the statistics

Phase 4 (`hycline.inference.mcmc_hierarchical` for geographic clines,
`hycline.inference.mcmc_genomic_hierarchical` for genomic clines) fits
many loci at once with *partial pooling*: each locus's cline parameters
are drawn from a shared population-level distribution, rather than fit
completely independently the way `fit_mcmc_batch` (Phase 2) and
`fit-genomic`/`fit-genomic-hierarchical` inputs otherwise would.
This isn't just a statistical refinement -- it's motivated by a specific,
long-standing prediction from hybrid zone theory, and it exists to answer
a question independent per-locus fitting is bad at answering.

**The theoretical basis: concordance is the null expectation.** Barton &
Hewitt's foundational "tension zone" model (Barton & Hewitt 1985, *Annual
Review of Ecology and Systematics* 16:113-148; see also Barton 1979,
*Heredity* 43:333-339 for the underlying dispersal/selection dynamics)
predicts that a hybrid zone maintained by a balance between dispersal and
selection against hybrids should show **concordant clines** across
neutral loci -- similar centers, similar widths -- because all of them are
being shaped by the same underlying population dynamics, not by
locus-specific selection. Barton & Hewitt (1981, in *Evolution and
Speciation*, eds. Atchley & Woodruff) explicitly used this as a diagnostic:
concordant clines across independent, neutral markers are evidence for a
tension zone maintained by secondary contact, rather than, say, independent
adaptation to a shared environmental gradient at each locus. Departures
from concordance -- a locus with a markedly narrower cline than the rest
of the genome -- are in turn the classic signature of a locus under
selection: linked to a barrier gene, an incompatibility, or otherwise
resistant to introgression relative to the neutral background (Barton &
Hewitt 1985; Szymura & Barton 1986, the same fire-bellied toad system
this project's own Phase 1 validation data comes from).

**What this means for fitting many loci.** If concordance is the expected
pattern for most of the genome, with genuine outliers being the
interesting exception, then treating every locus as an independent,
unrelated fitting problem throws away exactly the structure the biology
predicts should be there -- and, more practically, it makes sparse or
noisy loci behave badly on their own (see BACKEND_PHASE4.md section 3 for
a real, worked example: a single unlucky low read count at one site
pushed an independently-fit cline's width to roughly double its true
value). A hierarchical model encodes the concordance expectation directly
as a prior: loci pool statistical strength from each other, so a
sparse/noisy locus's estimate gets pulled toward the (data-driven, not
assumed) population-wide pattern rather than free to swing wildly on its
own thin evidence -- while a well-sampled locus's estimate is barely
affected, since its own data already dominates. This is the shrinkage
behavior demonstrated in BACKEND_PHASE4.md sections 4 and 7, and it's
the reason "recovers the right numbers" isn't the interesting question for
these components -- the interesting question is whether sparse loci get
appropriately stabilized without distorting well-sampled ones.

**Why this matters more directly for genomic clines specifically.** The
same logic extends to the genomic cline's `u`/`v` parameters against
hybrid index, but with an added, more targeted payoff: genome-wide scans
for differential introgression are, definitionally, a search for outlier
loci among many. Doing that search well requires distinguishing a locus
that's *genuinely* behaving differently from the rest of the genome from
a locus that just happens to be *poorly estimated* (few informative
reads, low-coverage sequencing, a marker with more missing data than
others) -- and independently-fit point estimates conflate the two: a
noisy locus can look like an extreme outlier purely because its own
estimate is unreliable, not because anything biologically interesting is
happening at it. This is explicitly the motivation behind `bgchm`
(Gompert, DeRaad & Buerkle 2024, *Ecology and Evolution*, "A next
generation of hierarchical Bayesian analyses of hybrid zones enables
model-based quantification of variation in introgression in R") and the
genomic cline framework it builds on (Gompert & Buerkle 2011, *Molecular
Ecology*; Fitzpatrick 2013, *Ecology and Evolution*, the logit-logistic
form this project implements) -- a hierarchical model's population-level
`sigma_u`/`sigma_log_v` (or their geographic analogs) is a direct,
principled answer to "how much does the genome actually vary in
introgression," properly accounting for estimation uncertainty rather
than treating every locus's point estimate as equally trustworthy.
`bgchm`'s own framing -- quantifying *model-based* variation in
introgression, not just tabulating per-locus point estimates -- is the
single sentence that best explains why Phase 4 exists at all.

**When you wouldn't want this.** Pooling assumes the loci being pooled
are drawn from a shared distribution in the first place -- reasonable for,
say, a set of anonymous SNPs sampled genome-wide, but worth pausing on if
you're deliberately combining loci that shouldn't be expected to behave
similarly. Mitochondrial markers are the clearest example: mtDNA is
haploid and maternally inherited, giving it roughly a quarter the
effective population size of an autosomal nuclear marker, and real hybrid
zone studies do find mtDNA clines behaving differently from nuclear ones
-- McKenzie et al. (2016, *Ecology and Evolution* 6:5771-5787) found the
two mtDNA markers in their killifish hybrid zone showed the steepest
clines of all 32 loci surveyed, narrower than nuclear-marker expectations
under neutrality (attributed there to selection/cytonuclear
incompatibility rather than pure drift, but the practical point holds
either way: mtDNA and autosomal markers aren't safe to assume interchangeable).
Neither hierarchical model currently supports separate pooling groups for
markers expected to behave differently (all loci passed in share one
population-level prior) -- mixing fundamentally different marker classes
into one hierarchical fit without accounting for that is a real way to
get a misleading population-level estimate, not something the tool
currently guards against. With very few loci (Phase 4's own validation
used 8), the hyperparameters themselves -- particularly the across-locus
variance terms -- are only loosely identified; the benefit of pooling
grows with more loci, and is weakest to demonstrate cleanly with just a
handful.

**From concordance to a number: Component C.** Everything above frames
concordance qualitatively -- loci *should* look similar under a shared
tension zone, and departures are the interesting signal. Component C
(`hycline.analysis.coupling`) turns that into an actual quantity: Barton
(1983) showed that when multiple barrier loci are linked, linkage
disequilibrium among them causes selection to spill over from each locus
onto the others, and how strong that spillover is -- whether barrier loci
act as a set of independent, locus-specific barriers or fuse into a
single genome-wide barrier -- is governed by his coupling coefficient,
theta = s/r (selection over recombination). Theta itself isn't directly
observable, but Firneno et al. (2023) showed via simulation that the
*variance* in fitted cline parameters across loci is a good proxy for it
-- concordant clines (low variance) indicate strong coupling; discordant
clines (high variance) indicate weak coupling, loci behaving more
independently. This is exactly the quantity Components A and B's
population-level `sigma` hyperparameters already estimate, which is why
Component C is built on top of them rather than needing its own separate
model: `hycline.analysis.coupling.geographic_cline_variance()` and
`genomic_cline_variance()` take per-locus estimates (ideally from A/B's
hierarchical fits, for the same reason those exist -- see above) and
report the across-locus variance directly, with the genomic version also
offering an estimated theta calibrated against Firneno et al.'s own
published results on 25 real hybrid zones. See BACKEND_PHASE4.md section
9 for the full methodology, including an honest accounting of what's
directly implemented versus approximated, and a real edge case (a
theoretically-impossible negative theta estimate for certain inputs) that
turned up during validation and is surfaced explicitly rather than hidden.

## Why genotype likelihoods? The biological motivation

Everything above assumes each individual's genotype is known with
certainty. Real hybrid-zone sequencing data often isn't: RAD-seq, ddRAD,
and other reduced-representation approaches routinely produce a handful
of reads per individual per locus, and even whole-genome resequencing is
frequently run at modest depth to fit more individuals into a sequencing
budget. At low depth, the true genotype is genuinely uncertain -- a
heterozygote can show zero reads of one allele purely by chance, and a
homozygote can show a spurious read of the "wrong" allele from sequencing
error. Modern variant callers (GATK, bcftools, ANGSD) report this
uncertainty explicitly as a genotype likelihood -- P(reads | genotype) for
each of the three possible diploid genotypes -- rather than collapsing
straight to one hard call. Discarding that and just taking the
single most likely genotype throws away real information a cline fit
should be using.

Phase 5 (`hycline.inference.mle_genotype_likelihood`) fits both geographic
and genomic clines directly from genotype likelihoods, by marginalizing
the likelihood over the unobserved true genotype (weighted by the
sequencing-derived genotype likelihood and the Hardy-Weinberg-expected
genotype probability under the cline model's predicted allele frequency)
rather than requiring a hard call first. This is, as it happens, a strict
generalization of the multinomial genotype likelihood this project's own
architecture plan flagged as future work from the very beginning: a
certain (hard-called) genotype likelihood collapses the marginalization
exactly to that same multinomial-under-HWE likelihood.

**What this actually fixes, and what it honestly doesn't**: the original
expectation going in was that naive hard-genotype-calling at low depth
would show measurable point-estimate bias in fitted cline parameters --
building the validation found this wasn't true, at least in the
idealized simulation used here (no asymmetric mapping bias modeled, only
symmetric sequencing error), and that negative result is documented
rather than replaced with a more flattering claim. What *did* hold up
cleanly is **uncertainty calibration**: at low sequencing depth,
genotype-likelihood-aware fitting's confidence intervals achieve close to
their nominal coverage, while naively hard-calling genotypes and fitting
with the standard likelihood is measurably overconfident (its CIs are too
narrow, because it discards real uncertainty and then treats the
resulting hard call as if it were certain data). Both approaches converge
as depth increases, exactly as expected once genotype calls become
effectively certain either way. See BACKEND_PHASE5.md for the full
investigation, including why the originally-hypothesized bias didn't
materialize and what that implies about what "reference bias" in
low-coverage genotyping actually is (predominantly a mapping/alignment
artifact this simulation doesn't include, not a pure statistical
consequence of genotype uncertainty on its own).

**Does this still matter once you're already pooling across loci?**
Phase 4's hierarchical models already stabilize sparse loci by borrowing
strength from data-rich ones -- so it's a fair question whether that
alone already fixes what genotype-likelihood-awareness fixes for
independent per-locus fits, making the combination redundant. Phase 4
Component D (`hycline.inference.mcmc_hierarchical_gl`) tests this
directly rather than assuming an answer: fit many loci hierarchically,
one of them sparse/low-depth, either with full genotype-likelihood
marginalization or via "hard-call then pool." **Pooling alone is not
enough** -- across 12 independent replicate simulations, the sparse
locus's 95% credible interval covered the true cline width 100% of the
time under GL-aware fitting versus 8.3% of the time under hard-call.
More importantly, the *mechanism* turned out to be more consequential
than expected: a hard-called sparse locus's distorted estimate doesn't
just stay wrong on its own -- it visibly pulls the *shared*
population-level hyperparameter (which every other locus's prior is
centered on) away from its true value, since the hierarchical model has
no way to know that estimate came from unreliable low-coverage data
rather than genuine biological signal. Skipping genotype-likelihood-
awareness when pooling across loci risks contaminating the whole batch
from a single poorly-sequenced locus, not just that locus's own result.
See BACKEND_PHASE4.md section 11 for the full investigation.

**Does the same finding hold for genomic clines?** Tested directly rather
than assumed to transfer -- Component E
(`hycline.inference.mcmc_genomic_hierarchical_gl`) is the genomic-cline
analog of Component D, and its result is genuinely different, not a
repeat. At the same moderate depth contrast that produced Component D's
dramatic gap, GL-aware and hard-call hierarchical fitting perform
*identically* for genomic clines -- no detectable advantage from
genotype-likelihood-awareness. Only at a more extreme depth contrast does
a modest difference appear, and even then the mechanism looks different:
hard-call's point estimates can actually be *closer* to the truth on
average while still failing to cover it, because its intervals are too
narrow relative to their own error (closer to Phase 5's own standalone
"overconfidence" finding than to Component D's "population contamination"
one). The likely explanation is structural: genomic clines have no
free asymptote parameter for sparse data to corrupt (unlike geographic
clines' `pmin`/`pmax`), and hybrid index always spans the full [0,1]
range by construction, giving the rate parameter inherently
better-conditioned identifiability even from uncertain genotypes. The
practical upshot: prioritize genotype-likelihood-awareness for
geographic clines whenever depth varies meaningfully across loci in a
hierarchical fit; for genomic clines, it matters less unless the depth
contrast is genuinely extreme. See BACKEND_PHASE4.md section 12 for the
full investigation.

## What's implemented

**Command-line interface:**
- `hycline/cli.py` (installed as the `hycline` command, or run directly as
  `python run_hycline.py`) -- 19 subcommands spanning every phase of this
  project (fitting, hierarchical pooling, coupling analysis,
  genotype-likelihood-aware fitting, data conversion), full flag
  reference via `--help` on each. See the "Command-line usage" section
  above for the full walkthrough.
- **Validated at "thousands of loci" scale, not just assumed to work
  there.** VCF parsing (~12,000 loci/sec, confirmed linear from 3,000-
  10,000 loci), wide-format CSV I/O, and sequential classic-MLE fitting
  (geographic hard-count, geographic genotype-likelihood-aware, and
  genomic) were all stress-tested with real, timed numbers and checked
  against known true signal recovered from among realistic background
  noise, not just confirmed to complete -- see BACKEND_SCALE.md. Two
  real bugs surfaced and fixed as a direct result: a genuine pandas
  DataFrame-fragmentation performance issue affecting seven different
  CSV-writing functions across the VCF/beagle converters and the
  genotype-to-GL converter (fixed consistently, verified byte-identical
  output before and after), and every multi-locus CLI command dumping
  the full locus-name list to the console at scale (fixed with a shared
  truncating helper, applied across all nine occurrences found).
- **`fit-geographic --method bayesian` now uses plate-vectorized batch
  fitting automatically when the data supports it** -- one NUTS run for
  every locus at once (`hycline.inference.mcmc_numpyro.fit_mcmc_batch`,
  already implemented) instead of one independent run per locus,
  addressing the real cost center Bayesian fitting is at "thousands of
  loci" scale. Detected and applied automatically, with no flag needed --
  and, just as importantly, falls back to the original per-locus loop
  automatically (not an error) whenever the data doesn't qualify (loci
  with different site sets, or `--scaling fixed`, which the batched path
  doesn't support), so nothing that worked before stops working. Verified
  through the real CLI on a synthetic 5-locus dataset with known distinct
  true parameters per locus: correct recovery, clean convergence
  (R-hat 1.00-1.01), and confirmed the fallback genuinely triggers on
  ragged data and fixed-extremes requests rather than silently batching
  something it shouldn't. See BACKEND_PHASE2.md section 13.
- `hycline.data.io` -- flexible CSV reading (single-locus and wide-format
  multi-locus, for both geographic and genomic clines, hard counts and
  genotype likelihoods) with actionable error messages (missing columns,
  non-numeric values, etc. all say exactly what's wrong rather than
  raising a bare Python traceback).
- `hycline.data.vcf` -- a pure-Python VCF parser (no `pysam`/`cyvcf2`
  dependency) underlying the `generate-geographic-input`/`generate-genomic-input` CLI
  commands, converting a VCF + metadata CSV directly into any of this
  project's CSV formats. Validated end to end against known ground truth,
  not just unit tests on hand-crafted fixtures: a synthetic geographic
  cline (true center=70, width=25) converted through the full VCF ->
  CSV -> `fit-geographic` pipeline recovered center=75.0; a synthetic genomic
  barrier locus (true v=4.0) recovered as v=3.99. Two real bugs caught
  and fixed during this component's own build (a locus with no usable
  genotype-likelihood data breaking the whole file's conversion; an
  all-loci-excluded edge case reporting false success) -- see
  BACKEND_PHASE1.md section 9.4 for the full design and both fixes.
- `hycline.data.geo` -- collapse lat/lon coordinates to a 1D transect
  distance (great-circle-from-reference or PCA-projection). See
  BACKEND_PHASE1.md section 9.1 for why this is a preprocessing
  convenience, not a 2D spatial model.

**Unified entry point (Python API):**
- `hycline.api.fit_cline` (also importable as `hycline.fit_cline`) --
  dispatches to either fitting procedure by a `method=` keyword. See
  above and `BACKEND_PHASE1.md` section 8.

**Phase 1 (MLE):**
- `hycline.models.cline` -- the Szymura & Barton (1986, 1991) sigmoid cline
  model, plus an exponential-tail extension (Barton & Gale 1993). See the
  module docstring for a caveat on tail-model validation.
- `hycline.data.schema` -- typed container for allele-frequency transect data.
- `hycline.likelihoods.binomial` -- binomial likelihood (HZAR's "freq" data type).
- `hycline.inference.mle` -- maximum-likelihood fitting via `scipy.optimize`,
  with Hessian-based standard errors and confidence intervals.
  `fit_mle_fixed_extremes` now accepts `--n-edge-sites` (CLI: `fit-geographic`/
  `fit-geographic --n-edge-sites`), an opt-in fix for a real fragility in
  the original single-site pmin/pmax convention -- an unlucky all-zero
  (or all-focal) draw at the extreme site can lock the fixed asymptote
  at the wrong value and inflate the fitted width substantially. First
  found and fixed for the hierarchical model (`BACKEND_PHASE4.md`
  section 3); the default here (`n_edge_sites=1`) reproduces the
  original single-site behavior *exactly*, verified directly and by
  re-running the Manakin cross-validation unchanged, so nothing already
  validated shifted when this was added. See BACKEND_PHASE1.md
  section 4.2 for the full story and a worked before/after example.
- `hycline.selection.aicc` -- AICc-based comparison across nested tail models
  (HZAR's `autofit` equivalent).
- `hycline.viz.plotting` -- observed-vs-fitted cline plots (`plot_fit`).

**See `BACKEND_PHASE1.md`** for the full design rationale -- why this
reproduces HZAR's own numbers despite using a direct optimizer rather than
HZAR's internal MCMC search, the Hessian-CI caveats, and the
`hzar_style_log_likelihood` finding (also summarized below).

**Quantitative trait clines (HZAR's other core data type, alongside allele
frequencies) -- classic MLE, full HZAR parity including tail models:**
- `hycline.data.schema.TraitTransect` -- site-aggregated continuous
  phenotype data (mean, n, required per-site SD -- HZAR's own model
  always uses per-site sample variance as data, so there's no
  optional-SD fallback). Built either directly (`read_trait_csv`/
  `read_multi_trait_wide_csv`, pre-aggregated site summaries) or from
  raw per-individual measurements (`read_raw_trait_csv`, `--raw` on the
  CLI) -- HZAR's own two supported input styles, both producing the
  same shape internally.
- `hycline.likelihoods.gaussian` -- full HZAR-parity Gaussian likelihood,
  verified directly against HZAR's own R source
  (github.com/cran/hzar), not a principled reimplementation: the
  *variance itself* follows a cline shape sharing the mean's center/
  width/tail parameters (`var_l`/`var_r`, plus `var_h` for real,
  documented excess variance in the hybrid-zone center), and the
  log-likelihood uses each site's observed sample variance as data, not
  just its mean -- confirmed to exactly match summing raw individual
  observations' own log-densities. Tail models (`left`/`right`/`both`/
  `mirror`) are also full HZAR parity, including a real, named
  structural difference from the allele-frequency cline's own tail
  model (HZAR's tail edges are `center +/- delta` directly, not
  `center +/- width/2`) -- verified continuous at the tail/center
  boundary directly, not assumed.
- `hycline.inference.mle_trait.fit_mle_trait` / `hycline.selection.aicc.
  compare_trait_models` -- fitting and tail-model comparison. Verified
  against known synthetic ground truth simulated from HZAR's actual
  model: all parameters recover correctly (including asymmetric
  parental variances, `var_r != var_l`, which the project's earlier,
  simpler design -- since removed -- couldn't represent at all), for
  both no-tail and tail-model cases, with the same tail-parameter weak
  identifiability already documented for the allele-frequency case, not
  a new issue.
- `hycline fit-trait` / `hycline fit-trait-multi` -- the CLI commands
  (single trait / many traits from one wide-format CSV), including a
  dedicated trait plot (per-site error bars, unbounded y-axis, visibly
  widening toward the cline center -- the `var_h` term made visible) and
  correct skip-with-message handling for a trait missing its required
  `.sd` column. Verified end to end through the real CLI, including a
  real bug (the CLI's shared tail-type parser not recognizing the new
  `left`/`right` choices) found and fixed during that verification.
- **Full HZAR-parity tail construction is also available for allele-
  frequency fitting, for both classic MLE and Bayesian (NUTS) fitting**
  (`fit-geographic`/`fit-multi --hzar-tail`, opt-in, default behavior unchanged) --
  `hycline.models.cline.hzar_cline_function`, ported directly from the
  verified trait-cline tail formula rather than re-derived, closing a
  real, precisely-named divergence this pass found between this
  project's original allele-frequency tail model and HZAR's actual one.
  Extending this to the Bayesian path surfaced six real bugs where
  exported/plotted curves ignored `hzar_tail` entirely (one present,
  undetected, since the classic-MLE version first shipped) -- all found
  and fixed, each confirmed to actually catch its bug by reverting and
  re-testing. See BACKEND_TRAIT.md section 5 and BACKEND_PHASE2.md
  section 14.
- **Deliberately out of scope for this pass, stated plainly**: Bayesian/
  hierarchical trait fitting (which also covers extending `--hzar-tail`
  to the Bayesian/hierarchical allele-frequency paths) and a genomic-
  cline (hybrid-index) analog of trait clines. See BACKEND_TRAIT.md
  section 6.

**Phase 2 (Bayesian/MCMC), now including tail-model support with a real, documented caveat:**
- `hycline.inference.mcmc_numpyro` -- primary backend: NUTS via NumPyro,
  single-locus (`fit_mcmc`) and plate-vectorized multi-locus batch fitting
  (`fit_mcmc_batch`). Supports every `TailType`, not just `NONE` -- but
  Bayesian tail-model fitting has a real, investigated convergence
  limitation (a non-smooth log-density NUTS struggles with severely) --
  see "A real limitation: Bayesian tail-model fitting" above and
  BACKEND_PHASE2.md section 12 before relying on it. Both functions emit
  a `UserWarning` when a non-`NONE` tail type is requested.
- `hycline.inference.mcmc_pymc` -- secondary backend: NUTS via PyMC
  (`fit_mcmc_pymc`), single-locus only, cross-validated to agree with the
  NumPyro backend on the same data. Remains `TailType.NONE`-only,
  deliberately not extended to tail models given the convergence finding
  above applies to gradient-based sampling generally, not a NumPyro-
  specific implementation detail (BACKEND_PHASE2.md section 12.5).
- `hycline.diagnostics.convergence` -- ArviZ-based R-hat/ESS/divergence
  checks (`check()`), with cline-fit-specific interpretation of what each
  diagnostic usually means when it fires.
- `hycline.viz.plotting.plot_bayesian_fit` -- observed data plus posterior
  median cline and shaded credible band, correctly rendering tail-model
  posteriors now (a real bug -- hardcoded `TailType.NONE` regardless of
  the actual fit -- was caught and fixed while validating this;
  BACKEND_PHASE2.md section 12.4).

**See `BACKEND_PHASE2.md`** for the full design rationale behind every one of
these choices -- why NumPyro over PyMC, the exact probabilistic model,
why `numpyro.plate` rather than `jax.vmap` for batching, what the
diagnostics mean, and a worked case study (below) of the diagnostics
catching a real problem on real data.

**Phase 3 (genomic clines) -- classic MLE only for now:**
- `hycline.models.genomic_cline` -- the logit-logistic genomic cline model
  (Fitzpatrick 2013): locus ancestry as a function of genome-wide hybrid
  index, `logit(phi(h)) = u + v*logit(h)`. A genuinely different model
  from the geographic cline above, not a relabeled version of it -- see
  BACKEND_PHASE3.md section 2.
- `hycline.data.schema.HybridIndexLocusData` -- per-individual data
  container (the genomic-cline analog of `AlleleFreqTransect`).
- `hycline.data.hybrid_index.estimate_hybrid_index` -- a simple
  mean-proportion hybrid-index estimator from multi-locus genotype data.
  Not the full uncertainty-propagating Bayesian estimator `gghybrid`/
  `bgchm` use -- see BACKEND_PHASE3.md section 5 for the honest scope note.
- `hycline estimate-hybrid-index` -- the CLI command, in two mutually
  exclusive modes: compute parental allele frequencies directly from
  labeled reference individuals in your own genotype file (no separate
  frequency file needed), or supply them precomputed via
  `--parental-freq-csv` for when you already have a reference panel's
  frequencies summarized. Writes an `(individual, hybrid_index)` CSV
  either way. Validated against known ground truth, not just confirmed
  to run: correlation 0.977 with true hybrid index on synthetic data,
  confirmed to chain correctly into a real genomic-cline fit end to end,
  and the two modes verified to produce numerically identical results
  when given the same underlying frequencies -- see BACKEND_PHASE3.md
  section 9.
- `hycline.inference.mle_genomic` -- MLE fitting via `scipy.optimize`,
  same approach as Phase 1's geographic MLE. `GenomicFitResult.summary()`
  includes a plain-language interpretation of the fitted rate parameter
  (barrier locus / elevated introgression / ~neutral).
- `hycline.viz.plotting.plot_genomic_cline_fit` /
  `plot_genomic_multi_locus_grid` -- observed genotypes vs. hybrid index,
  with the neutral (v=1, u=0) expectation shown as a reference line.

**See `BACKEND_PHASE3.md`** for the full design rationale -- most
importantly, why there's no HZAR baseline to validate against here (HZAR
doesn't do genomic clines at all), and how that gap is instead filled with
both synthetic-data validation (known true parameters -- see below) and
real-data cross-validation against BGC (Gompert & Buerkle) on 77 real loci
from a published Italian sparrow hybrid zone dataset (see below).

**Phase 4, Component A (hierarchical geographic clines) -- now with a CLI
command:**
- `hycline.inference.mcmc_hierarchical.fit_hierarchical_batch` -- partial
  pooling of cline center and (log) width across many loci sharing a
  transect, so sparse loci borrow statistical strength from data-rich
  ones. Builds on Phase 2's `fit_mcmc_batch` (which gives every locus
  fully independent priors) by adding shared population-level priors.
- Returns a `HierarchicalFitResult` with `.population_summary()`
  (population-level hyperparameters) and `.locus_summary()` (per-locus
  estimates), plus the full ArviZ `InferenceData` for anything else.
- `hycline fit-hierarchical-geographic` -- the CLI command, using the same
  wide-format multi-locus CSV as `fit-geographic`. Unlike `fit-geographic`, every
  locus must have data at exactly the same sites (no per-locus missing
  data) -- the CLI checks this and fails with a clear, actionable error
  naming the offending locus rather than a cryptic crash, if violated.
  See BACKEND_PHASE4.md section 13 for the full design, including
  real-data validation on the bundled Manakin multi-locus dataset.

**Phase 4, Component B (hierarchical genomic clines) -- now with a CLI
command:**
- `hycline.inference.mcmc_genomic_hierarchical.fit_hierarchical_genomic_batch`
  -- the direct genomic-cline analog of Component A: partial pooling of
  the logit-logistic `u` (bias) and `v` (rate) parameters across many loci
  sharing the same set of individuals/hybrid indices. This is also the
  first Bayesian backend for genomic clines at all -- Phase 3 shipped
  classic MLE only.
- Returns a `HierarchicalGenomicFitResult` with the same
  `.population_summary()`/`.locus_summary()` shape as Component A's result.
- Hyperpriors are centered at the neutral null (`mu_u=0`, `mu_v=1`,
  reproducing `phi(h)=h`) rather than an arbitrary default -- see "Why
  hierarchical clines?" above and BACKEND_PHASE4.md section 8 for why
  that's the biologically meaningful reference point.
- `hycline fit-hierarchical-genomic` -- the CLI command, same wide-format
  multi-locus CSV as `fit-genomic`, with the same dense-data
  requirement and error handling as `fit-hierarchical-geographic`. Verified on the
  Phase 3 synthetic multi-locus dataset to correctly recover known true
  parameters (barrier/permeable/biased loci all correctly identified)
  with clean convergence.
- `pool_center=True` (`--pool-center` on the CLI) -- an additive,
  opt-in alternative parameterization: hierarchically pools
  `logit(center) = u/v` directly (Firneno et al. 2023's own quantity)
  instead of the raw bias parameter `u`, reporting a genuine
  `sigma_center` hyperparameter usable directly as Component C's sigma_c
  input, closing a real gap found while verifying Component C's own
  theta calibration. The default (`pool_center=False`) is unchanged --
  confirmed bit-for-bit by re-running the original Component B test
  suite without modification -- and the new mode was validated against
  known ground truth (15 simulated loci, true sigma_center=0.8, clean
  convergence, per-locus center/v recovery correlation 0.965/0.940). See
  BACKEND_PHASE4.md section 7.4.

**Phase 4, Component C (coupling/concordance analysis) -- now with a CLI
command:**
- `hycline.analysis.coupling.geographic_cline_variance()` /
  `genomic_cline_variance()` -- compute the across-locus variance in
  fitted cline parameters (cline slope for geographic clines; `logit(u/v)`
  and `log(v)` for genomic clines, following Firneno et al. 2023's exact
  definitions), the empirical proxy for Barton's (1983) theoretical
  coupling coefficient theta. If you fit with Component B's
  `pool_center=True` (above), prefer its own jointly-estimated
  `sigma_center` over this function's SD-of-point-estimates approximation
  -- feed it directly into `estimate_coupling_coefficient` as sigma_c
  instead.
- `genomic_cline_variance()` also returns an estimated theta, calibrated
  by reverse-engineering Firneno et al.'s own applied model from their
  published results on 25 real hybrid zones (`FIRNENO_2023_TABLE1` in the
  module) -- verified to reproduce their reported values to within 0.005
  (`tests/test_coupling.py`), confirmed digit-for-digit against the real
  published Table 1 (not the informal GitHub summary, which has a
  transcription error this project's copy was never affected by), and
  further validated against 16 genuinely held-out published data points
  from the paper's own Supplemental Material (mean absolute error 0.008,
  never used to fit the calibration) -- see BACKEND_PHASE4.md section 9.3
  for the full verification. No theta estimate is offered for geographic
  clines, since Firneno et al. calibrated theta against real data using
  genomic clines only.
- A real, documented edge case: the calibration can return a
  theoretically-impossible negative theta for some inputs, even within
  the reference data's range -- surfaced as an explicit warning in
  `GenomicClineVariance.summary()`, not hidden or silently clipped.
- `hycline coupling-geographic`/`coupling-genomic` -- the CLI commands.
  Unlike every other subcommand, these don't fit anything -- they take
  the per-locus summary CSV that `fit-geographic`/`fit-hierarchical-geographic` (or their
  genomic counterparts) already produce, auto-detecting the right column
  regardless of which upstream command (and its own, inconsistent naming
  convention -- `width` vs. `width_mean`, etc.) produced it. See
  BACKEND_PHASE4.md section 14 for the full design.

**See `BACKEND_PHASE4.md`** for the full design rationale -- why Phase 4
is split into components, the exact hierarchical models and prior choices
for each, a real bug found and fixed during Component A's own testing (a
single noisy extreme site could badly distort a fitted cline's width --
see below for the details and the fix), and Component C's
reverse-engineered theta calibration methodology with its confidence
level stated explicitly. Component B's own validation involved a
different kind of honest iteration: two initial synthetic sparsity
contrasts (a 2x sample-size difference, then the same contrast with more
individuals) showed essentially no shrinkage signal at all before a
larger, more realistic contrast (a single low-confidence SNP vs. a
resolved multi-SNP ancestry block) produced a clean,
correctly-directioned result -- documented rather than silently swapped
in without explanation.

**Phase 5 (genotype-likelihood-aware fitting) -- now with a full CLI:**
- `hycline.inference.mle_genotype_likelihood.fit_mle_geographic_gl` /
  `fit_mle_genomic_gl` -- fit geographic/genomic clines directly from
  per-individual genotype likelihoods, marginalizing over genotype
  uncertainty rather than requiring a hard call first. Return the same
  `FitResult`/`GenomicFitResult` types as the hard-count fitting
  functions, for direct comparability.
- `fit_mle_geographic_gl` now supports every `TailType`, not just `NONE`
  -- and, unlike the Bayesian path (above), doesn't have the same
  convergence limitation. A related issue does show up (occasional
  corner-solution failures from the tail model's kinked likelihood
  surface) -- fixed automatically via randomized-restart multi-start
  optimization, verified to find genuinely better optima than a single
  start. See BACKEND_PHASE2.md section 12.3.
- `hycline.data.genotype_likelihood.phred_to_log_likelihood` -- converts
  from the Phred-scaled `PL` format most variant callers actually output.
- `hycline.data.genotype_likelihood.simulate_genotype_likelihoods` --
  generates realistic synthetic genotype likelihoods from known true
  genotypes and sequencing depth; used for this phase's own validation,
  and useful on its own for exploring what a given depth/error-rate
  combination's uncertainty looks like.
- `GenotypeLikelihoodTransect`/`GenotypeLikelihoodLocusData` -- per-*
  individual* data containers (not per-site aggregates, unlike
  `AlleleFreqTransect` -- genotype likelihoods can't be losslessly
  aggregated the way hard counts can).
- A real, validated finding, not a claimed one: this doesn't fix
  point-estimate bias (none was found in an idealized simulation without
  mapping bias) -- it fixes *uncertainty calibration*. See "Why genotype
  likelihoods?" above and BACKEND_PHASE5.md section 3 for the full
  investigation and the actual coverage numbers.
- `hycline fit-geographic`/`fit-genomic` (single-locus) and `fit-geographic`/
  `fit-genomic` (independent multi-locus) -- the CLI commands.
  The multi-locus commands needed genuinely new plotting code, not a
  reuse of Components D/E's: independent fitting has no dense-data
  requirement (different loci can have different individuals), so the
  hierarchical grid plotters' shared-array assumption doesn't apply --
  see BACKEND_PHASE5.md section 6 for the full design, including a real
  mid-implementation mistake (dead code left where the multi-locus grid
  plot should have been) caught and fixed before this shipped.
- `hycline.data.genotype_likelihood.hard_genotype_to_log_likelihood` --
  the hard-genotype convenience path. Converts ordinary hard-called
  genotypes to the genotype-likelihood format directly, exactly (not an
  approximation -- marginalizing a hard call's one-hot likelihood
  reproduces the same likelihood a non-GL fit would use, verified to
  `1e-9`), so hard-called data can be fed into any GL-aware command
  without constructing one-hot likelihoods by hand. See
  BACKEND_PHASE5.md section 7 for the full design, including a real
  end-to-end check (convert, then fit, then confirm the true generating
  parameters are recovered) rather than validating the conversion in
  isolation. No CLI command for this anymore (the CLI's own conversion
  path moved to VCF/beagle-native `generate-geographic-input`/
  `generate-genomic-input`/`generate-genomic-input` -- this function
  remains available for direct Python callers with genotype-dosage data
  from some other source).
- `hycline.data.converter.convert_genotype_csv_to_hard_count_csv` -- the
  reverse direction: aggregates per-individual genotype dosage into
  HZAR-style site-level allele counts. Genuinely lossless (unlike
  genotype likelihoods, hard counts *can* be losslessly summed across
  individuals at a site), verified against hand-computed aggregations
  and a full convert-then-fit pipeline recovering known true cline
  parameters. See BACKEND_PHASE5.md section 7.4. Same status as above --
  Python-only now, no CLI command.
- `hycline.data.filter_aims` / `hycline filter-aims` -- filter genome-
  wide SNP data (a wide genotype-dosage CSV, or ANGSD's own beagle-
  format genotype-likelihood output, via a new `hycline.data.beagle`
  reader/writer) down to ancestry-informative markers before any of the
  conversion steps above. A separate metadata file (not embedded
  population labels) maps sample names to the two parental reference
  populations, so the same file can hold parental references and hybrid-
  zone individuals together. Verified end to end -- filter, convert,
  fit, recover known true cline parameters -- not just at the filtering
  step in isolation; also surfaced a real, honest finite-sample finding
  along the way (a genuinely informative locus can narrowly miss the
  default threshold with a small reference panel, by chance, not because
  the filter is wrong). See BACKEND_PHASE1.md section 9.6.
- `hycline.data.beagle`'s four writer functions, underlying
  `generate-geographic-input` (beagle input) / `generate-genomic-input` --
  the beagle-format analog of the VCF converter, mirroring its structure
  closely. GL-format output preserves beagle's genotype uncertainty
  directly; the default site-aggregated output uses each individual's
  *expected* dosage (posterior mean under their own genotype
  likelihoods), not an argmax point estimate, so the resulting frequency
  is genuinely GL-informed rather than a lossy most-likely-genotype
  summary -- verified against a real case where the two approaches
  diverge (genuinely ambiguous GLs 0.49/0.02/0.49 gave frequency 0.0
  under the old argmax approach and correctly give 0.5 under expected
  dosage). Verified end to end against known ground truth through the
  actual CLI commands for all four writer/output combinations, including
  the full `filter-aims` -> `generate-geographic-input` -> fitting chain
  with no manual reshaping in between. See BACKEND_PHASE1.md section 9.8.

**Phase 4, Component D (hierarchical + genotype-likelihood-aware
fitting) -- now with a CLI command:**
- `hycline.inference.mcmc_hierarchical_gl.fit_hierarchical_batch_gl` --
  combines Component A's partial pooling across loci with Phase 5's
  genotype-likelihood marginalization: fit many loci hierarchically
  directly from per-individual genotype likelihoods, rather than choosing
  between pooling (Component A, hard counts only) and proper uncertainty
  propagation (Phase 5, single-locus only). Returns the same
  `HierarchicalFitResult` type as Component A's `fit_hierarchical_batch`.
- Answers a real open question rather than assuming it: does genotype-
  likelihood-awareness still matter once loci are already pooled? Yes --
  12/12 replicates covered the true value under GL-aware fitting vs.
  1/12 under hard-call, for a sparse locus's width. See "Why genotype
  likelihoods?" above and BACKEND_PHASE4.md section 11 for the full
  investigation, including the more consequential-than-expected mechanism
  (population-level hyperparameter contamination from one bad locus, not
  just that locus's own estimate).
- `hycline fit-hierarchical-geographic` -- the CLI command. Needed genuinely new
  infrastructure that didn't exist anywhere in the project yet: Phase 5
  itself shipped with no CSV reader for genotype-likelihood data at all
  (single-locus or multi-locus), so wiring this up meant building
  `read_gl_multi_locus_wide_csv` first. Also supports `--phred` for
  reading Phred-scaled PL values directly (the format GATK/bcftools/ANGSD
  actually output), verified as an exact round-trip conversion. See
  BACKEND_PHASE4.md section 14 for the full design.

**Phase 4, Component E (genomic-cline analog of Component D) -- now with
a CLI command:**
- `hycline.inference.mcmc_genomic_hierarchical_gl.
  fit_hierarchical_genomic_batch_gl` -- combines Component B's partial
  pooling with Phase 5's genotype-likelihood marginalization for genomic
  clines. Returns the same `HierarchicalGenomicFitResult` type as
  Component B's `fit_hierarchical_genomic_batch`. No `pmin`/`pmax` to
  derive at all (genomic clines have no free asymptotes), so this
  component needed no analog of Component D's `pmin`/`pmax`-derivation
  work.
- **A genuinely different finding from Component D, not a repeat**: at
  the same moderate depth contrast that gave Component D its dramatic
  result, GL-aware and hard-call hierarchical fitting perform
  *identically* for genomic clines. Only at a more extreme depth contrast
  does a modest difference appear, and it looks like an overconfidence
  pattern (Phase 5's own standalone finding) rather than Component D's
  population-contamination mechanism -- plausibly because genomic clines
  lack the free-asymptote fragility source and because hybrid index
  always spans the full [0,1] range by construction. See "Why genotype
  likelihoods?" above and BACKEND_PHASE4.md section 12 for the full
  investigation.
- `hycline fit-hierarchical-genomic` -- the CLI command, sharing the
  same new GL CSV-reading infrastructure and `--phred` support as
  `fit-hierarchical-geographic`.

## Quick start (Python API / development)

```bash
# Phase 1 only
pip install -e ".[dev]"
python examples/demo_synthetic_fit.py
pytest tests/test_cline.py tests/test_mle.py tests/test_manakin_validation.py tests/test_data_io.py tests/test_geo.py tests/test_genomic_cline.py tests/test_mle_genomic.py tests/test_hybrid_index.py tests/test_coupling.py tests/test_genotype_likelihood_data.py tests/test_genotype_likelihood_fitting.py

# Phase 2 (adds NumPyro/JAX/ArviZ/PyMC)
pip install -e ".[dev,bayes]"
python examples/demo_bayesian_manakin.py
pytest   # full suite, including tests/test_api.py and tests/test_cli.py, needs the bayes extra
```

(If you just want to run fits from the command line rather than write
Python, see "Command-line usage" above instead -- `pip install -r
requirements.txt` is the lighter-weight setup for that.)

The Phase 1 demo simulates a noisy hybrid-zone transect with known true
parameters, fits it, compares tail-model variants by AICc, and saves a plot
to `demo_fit.png`. The Phase 2 demo fits the real Manakin data with NUTS
and compares against both the Phase 1 MLE fit and HZAR's own published
result (see below).

## Not yet implemented (see the architecture plan)

Reprioritized into tiers -- roughly, small/contained/high-confidence
items first, larger or more open-ended ones later. Cross-references
point to where each gap was actually found and characterized, not just
asserted.

**Necessary vs. recommended, stated explicitly, not left implicit.**
Every item below was checked against one question: does its absence mean
something in this package is currently *wrong, misleading, or resting on
an unverified foundation presented as verified*? As of this pass, the
honest answer is **no** -- there is no outstanding "necessary" item.
That's not a default assumption; it's the result of directly checking:
the allele-frequency tail model's real divergence from HZAR is now
found, verified, and available as an explicit opt-in
(`--hzar-tail`) rather than a silent gap; the quantitative-trait model
is full HZAR parity, verified against the actual R source rather than a
principled guess; Component C's theta calibration is verified against
the real paper and supplementary data; three small correctness-adjacent
issues found during this pass (listed first in Tier 1) either already
fail loudly with a clear message rather than silently, or are labeling
inconsistencies with no effect on the underlying numbers. Everything
below is **recommended** -- real value, but the package doesn't
misrepresent its own reliability without it. Where an item is closer to
"probably not worth building at all," that's said plainly too, not
left for the reader to infer from tier placement alone.

### Tier 1 -- small, contained, ready to pick up

**Two items added from Baird & Daley (2025), *The Shapes of Clines and
Wavefronts* (Mol Ecol, doi:10.1111/mec.70109) -- read in full, not just
abstracted, and placed here at high priority since the first item
directly explains (rather than just works around) a limitation this
project had already found and documented empirically:**

- **A geographic tail-model alternative that doesn't have the
  convergence problem HyCline's own `--hzar-tail` tripartite model has.**
  This project already documented, empirically, that "Bayesian
  (NUTS) tail-model fitting has a known, real convergence difficulty"
  caused by "the tail model's non-smooth join between the sigmoid core
  and the exponential tail" (see the geographic user guide's tail-model
  section, and the `fit_mcmc`/`fit_mle` docstrings), and worked around
  it with randomized restarts on the classic-MLE path. Baird & Daley
  give the underlying reason, independently and rigorously: the
  tripartite model (which HZAR's own tail parameterization, and
  therefore `--hzar-tail`, is built from) has genuine mathematical
  discontinuities -- "corners" -- at the two points where the logistic
  core joins each exponential tail. That's not a numerical
  inconvenience specific to this project's optimizer or sampler; it's a
  structural property of the model family itself. The randomized
  restarts address the symptom, not the cause. Baird & Daley propose
  and validate a fix: the **T-cline family**, built from the
  non-central Student's-t distribution, continuous everywhere (no
  corners), needing only 2 parameters for symmetric stepped-ness (4
  with asymmetry, vs. the tripartite model's 4-6) -- and demonstrate a
  real, quantified result on Fisher's own 1937 wave-of-advance data: an
  18-fold accuracy improvement over the best tripartite fit, using
  *fewer* free parameters, not more. Working Python code already
  exists: `Clinalyse` (Baird & Daley 2023,
  github.com/Studenecivb/clinalyse) -- this would be a genuine port/
  adaptation, not research from scratch. Recommended scope: implement
  as an *additional* `--tail` option alongside the existing
  HZAR-compatible tripartite one (which needs to stay, for direct
  HZAR-parity comparisons), not a replacement.
- **Update the existing tail-model convergence warning to cite the
  actual explanation.** Small, no new model work: the warning text in
  `fit_mcmc` currently describes the convergence difficulty as an
  empirical observation ("a known, real convergence difficulty");
  once the T-cline work above lands (or even before, as an interim
  documentation fix) this should cite Baird & Daley's structural
  explanation directly, and point at the T-cline alternative once
  available.
- **Add an explicit shape-mismatch caveat to `coupling-geographic`/
  `coupling-genomic`'s own documentation.** Baird & Daley's outlier-
  analysis discussion maps directly onto what these two modules do:
  both rely on across-locus variance in cline center/width as the
  coupling signal, and the paper gives a concrete mechanism by which
  that signal can be contaminated by unmodeled cline shape -- a model
  that doesn't allow for asymmetry will misattribute incomplete-
  lineage-sorting-driven asymmetry as a center/width outlier, and a
  model that doesn't allow for steppedness will systematically
  overestimate width and thereby miss real barrier-effect outliers.
  HyCline's coupling modules currently have no diagnostic that would
  catch either failure mode. This is a documentation-only item (no
  model change implied) unless/until the geographic and genomic model
  extensions above exist to actually fit shape-aware clines feeding
  into coupling analysis.

**Adaptive-introgression candidate flagging, added at the user's
request after a joint review of four primary sources** (Gompert,
Parchman & Buerkle 2012, *Phil Trans R Soc B*, doi:10.1098/rstb.2011.0196;
Bailey 2024, the gghybrid paper, doi:10.1111/1755-0998.13910;
Jofre & Rosenthal 2021, *Mol Ecol Resour*, doi:10.1111/1755-0998.13428;
Gompert & Buerkle 2011, *Mol Ecol*, doi:10.1111/j.1365-294X.2011.05074.x)
-- specified in enough detail here to build directly, not just a
pointer to "go read the outlier-detection literature":

- **Core mechanism: rank by delta-AIC/delta-waic against the neutral null
  (u=0, v=1), not by raw Bayesian p-value or credible-interval-excludes-
  zero alone.** Bailey (2024)'s own simulations found the reported
  Bayesian p-value's relationship to the true deviation is
  non-monotonic -- peaked at intermediate cline steepness rather than
  scaling with the actual effect size -- and states directly in the
  abstract that "use of Bayesian posterior distributions and p values
  to select candidate non-null loci is problematic and should be
  avoided." waic performed best in their comparison. This is
  computable from quantities `fit-genomic`/`fit-hierarchical-genomic`
  already produce (log-likelihood and parameter count for classic MLE;
  posterior samples for Bayesian) -- the AICc HyCline already reports
  per locus is most of the way there for the classic-MLE path; the
  fixed-null-model comparison (u=0, v=1 forced) is the piece that
  doesn't exist yet.
- **Report center-outlier and rate-outlier status separately, not
  merged into one flag.** Gompert et al. (2012) found BDM epistatic
  incompatibilities showed up mainly as *center* (`u`) shifts rather
  than *rate* (`v`) shifts -- but so does ordinary directional
  selection, and so can drift. Jofre & Rosenthal (2021) documented a
  third, specifically deceptive pattern for geographic clines directly
  analogous to this: drift under underdominance or BDM incompatibility
  can narrow a cline's width *and* shift its center in a way that
  specifically mimics directional/adaptive selection ("spurious centre
  outliers") -- not just generic noise, a false signal shaped like the
  thing being searched for.
- **A mandatory caveat in the output itself, not just documentation**:
  cline-shape signatures alone cannot distinguish *which* mechanism
  (directional selection, underdominance, BDM epistasis, or drift)
  produced an outlier. This isn't a hedge -- it's Gompert et al.
  (2012)'s own stated conclusion about their own model: "previous
  attempts (including our own) to associate a specific form of
  selection... with a locus based on the sign or magnitude of cline
  parameters were perhaps overly simplistic and naive." Any HyCline
  output that flags candidate loci needs to say this plainly next to
  the flags, the same way the near-invariant-locus warning
  (`hycline/inference/mle.py::_warn_if_locus_near_invariant`) is
  surfaced directly rather than left to a docs page.
- **A second, explicit documented limitation: this method is weak at
  detecting epistatic (BDM-style) incompatibilities specifically.**
  Gompert & Buerkle (2011)'s own simulations found a false discovery
  rate of 0.81-0.95 for pairwise epistasis, dramatically worse than for
  underdominance or directional selection (both <0.1 under most
  conditions they tested). A user relying on this feature to search
  for epistatic barriers specifically should be told this directly,
  not left to infer it.
- **A separate, genuinely new warning candidate, distinct from the
  existing near-invariant-locus warning**: Bailey (2024) found that an
  *uneven distribution of hybrid index values* specifically degrades
  accuracy for steep clines (`v` estimates less accurate, cline centers
  artificially clustered, higher uncertainty for loci whose true steep
  transition falls in a sparsely-sampled region of hybrid-index space)
  -- a different failure mode than a near-invariant focal allele, and
  one HyCline currently has no check for at all. Worth its own
  `_warn_if_hybrid_index_distribution_uneven`-style check alongside the
  existing one, likely checking for large gaps in the empirical hybrid
  index distribution relative to the fitted cline's steepness.
- **Document, not automate, the strongest available validation**:
  Jofre & Rosenthal (2021) found outlier analysis on a single hybrid
  zone transect is only reliable within a fairly narrow demographic
  window (their own false-positive rates ranged from 1.35% in the best
  case -- young zone, high migration, large deme -- to 98.9% in the
  worst), and recommend cross-validating candidate loci against
  independent replicate hybrid zones of the same species pair wherever
  available, citing concrete precedent (e.g. 59/366 outlier loci
  replicating across independent house mouse transects). This can't be
  automated in general (it requires a second dataset), but the study
  design guide should say so plainly rather than let a single-transect
  result read as more confident than it is.
- **Now read directly, closing the gap noted above**: Fitzpatrick (2013)
  and Firneno et al. (2023) have both been read in full (PDFs in
  `references/`, notes in `references/README.md`). Neither changes the
  design above -- if anything both reinforce it directly: Fitzpatrick's
  own simulations found the logit-logistic model has the best precision/
  sensitivity balance for outlier detection of the models he compared,
  and that naive null-hypothesis testing (ignoring drift) is badly
  miscalibrated, independently supporting the delta-AIC/delta-WAIC
  ranking approach above over raw p-values. Firneno et al.'s own
  robustness checks add one caveat not yet reflected in
  `coupling-geographic`/`coupling-genomic`'s docs: their sigma-to-theta
  calibration is sensitive to the migration rate and genetic map size
  assumed in the underlying simulations, and can lose essentially all
  predictive power (~0% out-of-bag variance explained in one of their own
  checks) when applied outside those conditions -- worth folding into the
  shape-mismatch caveat item below as a second, distinct reason the
  calibration can mislead.
- **A third primary source, read directly in the same pass, not
  previously on this project's radar**: Payseur (2010), *Mol Ecol
  Resour* 10:806-820, doi:10.1111/j.1755-0998.2010.02883.x -- a review of
  differential-introgression methods that names confounds for outlier-
  based candidate detection not yet listed above. Worth folding into the
  adaptive-introgression flagging feature's caveat text alongside the
  drift/epistasis caveats already specified:
  - **Incomplete lineage sorting (ILS)** produces the same locus-level
    signature as recent gene flow, and is a larger confound than drift
    alone for recently diverged species or large effective population
    sizes -- worth naming explicitly, not left implicit under "drift."
  - **Marker ascertainment bias**: fixed-difference markers -- exactly
    what `filter-aims` selects for -- systematically bias toward
    *underestimating* introgression, and regions that actually carry
    reproductive-isolation genes are, paradoxically, more likely to have
    fixed differences in the first place, making reduced introgression
    there specifically harder to detect by comparison to other loci.
  - **A false-negative risk, not just false-positive**: citing the wider
    genome-scan literature (Kelley et al. 2006; Teshima et al. 2006),
    truly selected loci need not lie in the most extreme tail of the
    empirical outlier distribution -- worth stating alongside the
    existing false-positive-focused caveats so the mandatory caveat text
    doesn't read as one-sided.

**Speciation-candidate locus flagging for geographic clines, extending
the item above to cover both cline types with one shared statistical
approach.** The item above is scoped to genomic clines only (its core
mechanism, delta-AIC/delta-WAIC against the fixed neutral null u=0,v=1,
is genomic-cline-specific -- geographic clines have no equivalent
theory-fixed null). Checked directly against the existing literature
(Payseur 2010's review, cross-checked with Gompert & Buerkle 2011,
Bailey 2024, Fitzpatrick 2013 already read for the item above) whether an
equivalent, dedicated tool already exists for geographic clines the way
`bgc`/`bgc-hm`/`HIest`/`gghybrid` exist for genomic clines: **it doesn't**.
The classic geographic-cline toolchain (`Analyse`/ClineFit, Barton & Baird
1995 -- the direct ancestor of the Barton cline `bgc` splices) offers only
pairwise likelihood-ratio tests between two loci's cline parameters, a
composite-profile test for whether *any* cross-locus heterogeneity exists
at all (Phillips et al. 2004; Macholan et al. 2007), and Long's (1991)
test for a single admixed population -- none of which produce a ranked,
per-locus candidate list. HZAR itself has no cross-locus outlier flagging
whatsoever. This is a real gap in the wider ecosystem, not just in
HyCline, so there is no single paper or tool to port -- the design below
adapts Fitzpatrick (2013)'s genomic-cline multivariate outlier method to
geographic-cline parameters instead.

- **Scope: hierarchical fits only, deliberately narrower than the
  genomic-cline item above.** Requires `fit-hierarchical-geographic`/
  `fit-hierarchical-genomic` (Components A/B) rather than also supporting
  independent per-locus fits. This is a deliberate choice, not an
  oversight: an independent per-locus point estimate's own noise can look
  exactly like a real outlier (the "single unlucky low read count...
  pushed an independently-fit cline's width to roughly double its true
  value" case already documented under "Why hierarchical clines?" above),
  and the hierarchical population-level hyperparameters
  (`mu_center`/`sigma_center`, `mu_log_width`/`sigma_log_width` for
  geographic; the analogous `u`/`v` hyperparameters for genomic) already
  estimate the correct empirical null distribution while properly
  accounting for that per-locus uncertainty -- exactly the problem
  Component A/B's partial pooling exists to solve, so building the flag on
  top of it rather than around it is a natural fit, not new machinery.
- **Statistic: Fitzpatrick (2013)'s multivariate Mahalanobis-distance
  outlier test, generalized by parameter pair rather than reimplemented
  per cline type.** Assume each locus's hierarchical-model parameter pair
  is jointly distributed as multivariate normal (per the model's own
  population-level prior); compute each locus's squared Mahalanobis
  distance from the population mean using the population-level covariance;
  compare to a chi-squared distribution with degrees of freedom equal to
  the number of parameters (2, in both cases below); flag via a
  Bonferroni-adjusted critical value, with the per-locus D^2 and its
  chi-squared quantile reported either way (not just a binary flag) so a
  user can inspect a Q-Q plot the way Fitzpatrick recommends for
  exploratory analysis. Parameter pairs: `(center, log(width))` for
  geographic clines; `(u, log(v))` for genomic clines fit without
  `--pool-center`, or `(logit(center), log(v))` -- Firneno et al.'s own
  quantities -- for genomic clines fit with it.
- **Where it lives in code**: `hycline.analysis.coupling` already computes
  the cross-locus variance/covariance this statistic needs
  (`geographic_cline_variance`/`genomic_cline_variance`) -- a new
  `flag_outlier_loci()`-style function belongs alongside them, reusing
  that computation rather than duplicating it in a separate module.
- **CLI: two new dedicated commands, not an extension of
  `coupling-geographic`/`coupling-genomic`.** `flag-candidates-geographic`/
  `flag-candidates-genomic`, matching the existing `verb-domain` naming
  convention (`fit-geographic`, `coupling-genomic`, etc.) and taking the
  same per-locus summary CSV inputs those commands already consume.
  Keeping this separate from `coupling-geographic`/`coupling-genomic`
  keeps those commands focused on the aggregate theta/variance question
  they already answer well, and gives the per-locus candidate list (a
  materially different kind of output -- a ranked table, not a single
  summary statistic) its own clearly-named entry point.
- **Caveats: one shared block, not duplicated per cline type.** All of
  Payseur (2010)'s confounds named above (incomplete lineage sorting,
  `filter-aims`'s own marker-ascertainment bias, recombination-rate
  heterogeneity, the false-negative risk) are geography-agnostic and apply
  identically to both commands' output -- write the caveat text once and
  share it, along with the already-planned mandatory
  mechanism-cannot-be-identified caveat and the epistatic-incompatibility
  blind-spot warning (both from the genomic-cline item above, and both
  equally applicable here: Gavrilets (1997), read via Payseur's review,
  shows epistatic incompatibilities distort geographic clines too, and
  only when linkage is tight and selection is strong).

Three real, small items surfaced while extending `--hzar-tail` to the
Bayesian path (BACKEND_PHASE2.md section 14, BACKEND_TRAIT.md
section 6) -- none are silent gaps (each fails clearly or is cosmetic,
not wrong), but all are genuinely quick to close:

- **A parameter-naming inconsistency for `TailType.MIRROR`**: classic
  MLE's JSON output labels the shared tail parameter `delta_m`/`tau_m`;
  the Bayesian path's JSON always labels the same quantity `delta_l`/
  `tau_l`. The underlying values are correct in both cases -- this is a
  labeling inconsistency, not a numerical one -- but it's a real,
  user-visible rough edge for anyone comparing classic and Bayesian
  MIRROR fits side by side.
- **`fit_mle_fixed_extremes` has no `hzar_tail` parameter at all.**
  Combining `--hzar-tail` with `--scaling fixed`/`--pmin`/`--pmax` now
  gives a clear, actionable error (fixed this pass) rather than
  silently ignoring the flag -- but actually adding HZAR-tail support to
  the fixed-extremes fitting path itself remains undone. The Bayesian
  path doesn't have this gap (`fit_mcmc` handles fixed extremes and
  `hzar_tail` in the same function).
- **Components A and D (Phase 4's hierarchical geographic models) don't
  support *any* tail model**, HZAR-parameterized or otherwise -- not a
  parameterization question but a missing capability, discovered while
  scoping the Bayesian `hzar_tail` extension. Smaller in one sense
  (nothing to choose between, just needs building) but larger in another
  (real new hierarchical-model work, not a port) -- kept in Tier 1
  because the *investigation* is done and the *scope* is now precisely
  known, even though the implementation itself is a real, non-trivial
  hierarchical hard-error-path/tail-math build, not a quick add.

Four more, found while checking real-world input compatibility ahead of
a benchmarking pass against published datasets -- verified against an
actual file (HZAR's own bundled `manakinMolecular.csv`), not just
inferred from reading the reader code:

- **`_read_csv` assumes comma-separated input, with no delimiter
  auto-detection or `--sep` override.** HZAR's own R-exported CSVs (its
  bundled example datasets included) use `;` as the field separator, a
  common R/European-locale convention -- reading one as-is currently
  fails outright (with a clear, actionable error, not a silent
  misparse) rather than being read directly. Confirmed the underlying
  `.nSamples`/`.A` column-naming convention genuinely does match once
  the delimiter is fixed, so this is a small, contained gap, not a
  deeper compatibility problem.
- **The wide-format CSV reader silently collapses multiallelic loci to
  biallelic.** For a locus with columns like `<locus>.A`/`<locus>.B`/
  `<locus>.C`, `detect_loci`/`read_multi_locus_wide_csv` only ever reads
  `.A`, treating everything else as an implicit "not-A" category, with
  no warning that the other allele columns exist and are being ignored.
  Confirmed directly against two real multiallelic loci in HZAR's own
  bundled Manakin dataset. Not necessarily the wrong biological choice
  (A-vs-pooled-other is a defensible simplification when A is a known
  diagnostic allele) -- the gap is that it's currently an implicit
  default, not a deliberate, visible choice the way the VCF converter's
  multiallelic handling already is.
- **No documented recipe for using ADMIXTURE/STRUCTURE Q-matrix output
  directly as genomic-cline hybrid index.** Almost free to close --
  `read_genomic_cline_csv` already just needs a `hybrid_index` column,
  so a Q-matrix column needs only a rename, not real conversion logic --
  but currently undocumented as a supported path.
- **No reader for STRUCTURE/PLINK ped-bed/genepop format.** Unlike VCF
  and ANGSD beagle (both directly supported), these classic population-
  genetics formats have no HyCline converter at all. The highest-value
  of the four items here specifically for benchmarking against published
  datasets, since several real, well-known hybrid zone studies
  (including some referenced in Component C's own Firneno et al. 2023
  calibration table) distribute genotype data in exactly these formats.
- **No output streams in real time when piped or redirected, for any
  multi-locus or Bayesian command.** Measured directly, not assumed: a
  7-locus Bayesian `fit-geographic` run piped through a subprocess pipe
  (the same situation as redirecting to a log file) produced 18 lines of
  output over 33.9 seconds, but all except the first 5 header lines --
  every actual per-locus fit result, the entire substance of the run --
  arrived in a single burst at 33.1s, right at the end. The per-locus
  progress messages genuinely exist in the code (`fit-geographic`'s classic
  path, checked separately via direct terminal output, does show them
  as each locus finishes), but no `flush=True` calls exist anywhere in
  the codebase, so Python's default block-buffering on non-TTY output
  swallows them until the process exits. Hierarchical fitting
  (`fit-hierarchical-geographic`) has no per-locus checkpoint to report against at
  all -- one message, then silence for the entire joint MCMC run,
  regardless of buffering. Small and low-risk to fix (explicit
  `flush=True` on the relevant `print()` calls, or forcing
  unbuffered/line-buffered output globally), and directly relevant to
  monitoring long benchmarking runs -- kept here rather than fixed
  immediately only because it surfaced while scoping other work, not
  because it's expected to be difficult.

### Tier 2 -- real, moderate-effort improvements, mostly about scale and speed

- Extending `fit-geographic`'s new plate-vectorized batched Bayesian path
  (below, under "What's implemented") to the other multi-locus Bayesian
  commands -- `fit-genomic --method bayesian`, `fit-geographic`,
  `fit-genomic`, and `compare-tails`. The detect-dense-data-and-
  batch pattern itself is proven and mostly mechanical to repeat; the
  genomic-cline case also already has a batch-fitting function to call
  into (Component B's hierarchical infrastructure, though that pools
  across loci rather than fitting independently -- an independent-fit
  genomic batch function doesn't exist yet either). See
  BACKEND_PHASE2.md section 13.4.
- Multithreading/parallelization across independent per-locus classic-MLE
  fits (`fit-geographic`/`fit-genomic`/`fit-geographic`/
  `fit-genomic`/`compare-tails`'s per-locus loops, and the 8
  randomized restarts `fit_mle_geographic_gl` now runs for tail models --
  BACKEND_PHASE2.md section 12.3 -- are all embarrassingly parallel and
  currently sequential). `ProcessPoolExecutor` (not threads, since the
  surrounding Python doesn't reliably release the GIL) with per-worker
  deterministic seeding (derived from a master seed + locus index, not
  shared RNG state) and explicit BLAS/JAX thread-count pinning per worker
  (to avoid oversubscription against libraries that are already
  internally multi-threaded) is the intended approach. A genuine speedup
  at "thousands of loci" scale (measured per-locus costs: ~0.054s
  geographic hard-count, ~0.101s geographic genotype-likelihood-aware,
  ~0.015s genomic -- BACKEND_SCALE.md section 1), not a correctness fix.
- Scale-aware default outputs (grid plots, per-locus curve CSVs) for
  multi-locus commands -- reviewable and useful at tens of loci, not at
  thousands. Now backed by a real, striking number rather than a
  reasonable guess: at just 300 loci, the default grid plot is already
  2,025 x 52,500 pixels, large enough that Pillow's own decompression-
  bomb heuristic flags it as looking like an attack pattern
  (BACKEND_SCALE.md section 3) -- the threshold where this becomes a
  real problem is far lower than "thousands." Currently every multi-locus
  command produces one plot panel and one curve-CSV block per locus
  regardless of count; a locus-count-based default (auto-suppressing or
  restructuring above some threshold, with an override either way)
  hasn't been built.
- Non-hierarchical Bayesian backend for genomic clines (Phase 3 is
  classic MLE only, and Phase 4 Component B went straight to the
  hierarchical Bayesian version -- BACKEND_PHASE4.md section 6 for why).
  Lower priority on reflection: the hierarchical version already covers
  the single-locus case (n_loci=1), so this mostly duplicates existing
  capability under a different name. **Leaning toward not building this
  at all** rather than just deprioritizing it, absent a concrete reason
  the hierarchical n_loci=1 case doesn't already serve.

### Tier 3 -- larger, more open-ended, or narrower-audience

- **A shape-generalized genomic-cline model, addressing a real,
  substantive critique from Baird & Daley (2025) (see the Tier 1 entry
  above for the geographic-side finding from the same paper).** The
  logit-logistic model HyCline's `fit-genomic` uses (Fitzpatrick 2013's
  `u`/`v` parameterization) is, verified directly against the paper's
  Box 3, mathematically identical to what the paper critiques: it
  implicitly assumes *both* the genome-wide hybrid-index cline and each
  individual locus's cline are logistic in shape. When that's false --
  the paper's own tested case uses a Cauchy-shaped true zone cline --
  fitting the logit-logistic model gives biased per-locus `u`/`v`
  estimates that aren't independently correctable after the fact
  (center and width errors interact). This is not a HyCline-specific
  defect -- the same equation is used verbatim by `gghybrid` and
  `bgchm`, the two other actively-maintained tools in this space -- but
  it's a real, published limitation of the whole model family HyCline
  belongs to, not a hypothetical one. **Unlike the geographic T-cline
  item above, there is no existing software implementation to port
  from**: the paper's Box 3 sketches how the logit-logistic
  special case generalizes to any pair of invertible cline shapes (via
  the same `{U_Z, U_i}` framework used for the geographic side, Table 1
  candidates including the T-cline family), but doesn't provide working
  code for the genomic case the way it does (`Clinalyse`) for the
  geographic one. Real design work needed before implementation:
  working out which invertible-shape pairs are worth exposing, how
  shape parameters interact with the existing hierarchical `sigma_v`/
  `sigma_c` machinery, and how to keep the common (well-supported,
  fixed-shape) case as simple as it is today. The paper itself notes a
  real complication specific to typical genomic-cline sampling designs
  (few genomes, many SNPs): there's often not enough power to estimate
  an extra shape parameter *per locus*, so cline shape may need to be
  treated as a shared or hierarchically-pooled nuisance parameter
  rather than something free per locus -- worth resolving as part of
  the design work, not assumed away.
- Quantitative trait cline extensions (classic MLE, geographic, full
  HZAR parity for both the variance model and every tail type, now
  including the same verified tail construction ported back to
  allele-frequency fitting for both classic MLE and Bayesian NUTS
  fitting -- see "What's implemented" above, BACKEND_TRAIT.md, and
  BACKEND_PHASE2.md section 14). Two real items remain:
  - **Bayesian/hierarchical trait fitting** -- classic MLE only so far,
    for the same reason Phase 1 (classic) preceded Phase 2 (Bayesian)
    and Phase 4 (hierarchical) for allele frequencies.
  - **A genomic-cline analog** (trait vs. hybrid index rather than
    geography). Investigated directly, not just flagged as
    unestablished: Fitzpatrick (2013)'s own genomic-cline derivation
    depends on both axes being bounded proportions related by
    `S = E(p_i)` across loci, an identity with no natural analog for a
    continuous trait; a literature search found no established
    precedent for the direct extension; and there's a real argument
    that a sigmoid isn't even the right null expectation for a purely
    additive polygenic trait (which should trend closer to linear in
    hybrid index). **On balance, recommended against building this
    speculatively** -- worth real design work only for a specific study
    with a specific hypothesis that needs it (e.g. suspected large-
    effect, epistatic trait architecture), not as a default capability
    extension. See BACKEND_TRAIT.md section 6.
- Separate pooling groups within a hierarchical fit for marker classes
  expected to behave differently (e.g. mitochondrial vs. autosomal) --
  see "Why hierarchical clines?" above for why this matters. Real value
  for datasets mixing marker classes, but a genuine hierarchical-model
  extension, not a quick add.
- Bayesian backend for genotype-likelihood-aware fitting (Phase 5 is
  classic MLE only -- BACKEND_PHASE5.md section 5). Tail-model support
  for Phase 5's classic MLE path is now done
  (`fit_mle_geographic_gl`/`hycline fit-geographic`/`fit-geographic` all accept
  `--tail`, with a multi-start optimization fix for a real corner-
  solution issue the tail model's kinked likelihood surface can cause --
  BACKEND_PHASE2.md section 12.3). Hierarchical (Phase 4) integration is
  done for both geographic (Component D) and genomic (Component E)
  clines.
- Tail models in the NumPyro Bayesian backend (both this project's
  original parameterization and the newer `--hzar-tail` option) have a
  real, documented convergence limitation, not a full clean capability
  -- see "A real limitation: Bayesian tail-model fitting" above and
  BACKEND_PHASE2.md section 12.2. PyMC (the secondary backend) remains
  `TailType.NONE`-only, deliberately not extended given that finding
  (BACKEND_PHASE2.md section 12.5). Not something more engineering effort
  is expected to fix (the limitation is a property of gradient-based
  sampling applied to a non-smooth model, not an implementation gap) --
  listed here as an open problem to revisit if a genuinely different
  approach (e.g. a smoothed tail model) is ever pursued, not a task with
  a clear next step, and not a "recommended" item in the same sense as
  the rest of this list.
- Arbitrary ploidy for genotype-likelihood-aware fitting (currently fixed
  at diploid -- BACKEND_PHASE5.md section 5). Narrower audience
  (polyploid systems); needs the HWE math generalized past the diploid
  biallelic case.
- A genotype-likelihood simulation model including mapping/alignment
  bias, not just symmetric sequencing error -- the honest gap identified
  by Phase 5's own validation (BACKEND_PHASE5.md section 3.1). Mostly
  serves this project's own internal validation/science story rather
  than unlocking new user-facing capability.
- Multiallelic/indel support for the VCF converter (biallelic SNPs only
  -- a deliberate scope limit, not an oversight; see BACKEND_PHASE1.md
  section 9.4/9.5 for why, and the `bcftools` pre-filter workaround the
  CLI now surfaces automatically when it's relevant). Full support would
  need a real modeling decision (which allele is "focal" at a
  multiallelic site?), not just a parsing extension.
- Streaming/chunked VCF processing, and an optional `cyvcf2` fast path --
  deprioritized specifically because "thousands of loci" (not
  hundreds of thousands) is line-by-line-Python-parser territory, not
  somewhere this is likely to matter in practice. Now confirmed rather
  than just assumed: the stress test found linear scaling and
  ~12,000 loci/sec parsing 3,000-10,000-locus VCFs (BACKEND_SCALE.md
  section 1) -- this item is deprioritized with real evidence behind it
  now, not just a reasonable guess. **Not recommended** absent a
  concrete genome-scale (hundreds of thousands of loci) use case, which
  is explicitly outside this project's stated target scale.
- Systematic exploration of why Component D and E's findings differ
  (BACKEND_PHASE4.md section 12.4 gives a plausible structural
  explanation -- no free-asymptote fragility source and full hybrid-index
  range coverage for genomic clines -- but hasn't independently verified
  it, e.g. by testing a geographic scenario with denser, evenly
  distributed sampling to see if that alone closes the gap). Placed in
  Tier 3 by size, but worth flagging as higher-priority *within* this
  tier than most neighboring items -- an unverified explanation for a
  cross-component discrepancy is closer to the "necessary" end of the
  spectrum than a pure capability extension, even though nothing
  currently indicates either component is actually wrong.
- A geographic-cline theta estimate (Component C only offers one for
  genomic clines, since that's what Firneno et al. calibrated against
  real data -- BACKEND_PHASE4.md section 9.3). Blocked less by effort
  than by the lack of real calibration data to fit one against.
- Blocked/grouped hierarchical pooling across multiple taxa sharing a
  suture zone, as an extension beyond this project's current within-one-
  hybrid-zone scope -- discussed but not committed to; would need its
  own design and validation, and depends on whether a given dataset's
  sampling design actually supports it (shared localities across taxa).

### Tier 4 -- large, speculative

- Interactive dashboard.

## Validation against HZAR: Phase 1 (MLE)

`examples/validate_manakin.py` fits HyCline to the real Ada locus data from
the Manacus (manakin) hybrid zone (Brumfield et al. 2001), pulled directly
from HZAR's own bundled `manakinMolecular` dataset, and compares against
HZAR's own published fit (Derryberry et al. 2014, Tables 3-4).

**Result: center and width match closely.**

| Parameter | HyCline | HZAR published | HZAR 2-LL support |
|---|---|---|---|
| center | 171.04 km | 170.9 km | [109.9, 201.2] |
| width | 261.97 km | 263.1 km | [96.2, 630.0] |

Both fitted values fall inside HZAR's own published support interval, and
AICc-based model selection agrees with HZAR's (both prefer the model with
pmin/pmax fixed to the observed extremes, no tails, over the model with
pmin/pmax freely estimated).

**One real finding worth knowing about:** the raw log-likelihood values did
*not* match at first (-30.3 vs HZAR's published -9.7). Tracing this into
HZAR's R source (`sampleLikelihoodMolecularPop` in `R/20-hzarPrep.R`)
showed that HZAR reports log-likelihood *relative to the saturated
(per-site perfect-fit) model*, not the raw binomial log-likelihood. That
offset is a constant with respect to model parameters, so it doesn't shift
where the likelihood is maximized -- which is exactly why the fitted
center/width matched almost exactly despite the raw likelihood not
matching. `hycline.likelihoods.binomial.hzar_style_log_likelihood`
reproduces HZAR's exact formula for reporting purposes; once used, the
log-likelihoods match HZAR's published values almost exactly (-9.66 vs
-9.70, -8.26 vs -8.20). This is documented in that function's docstring
and covered by `tests/test_manakin_validation.py`.

A smaller, still-open discrepancy: HZAR's published "AICc" column for this
example matches plain, uncorrected AIC far more closely than a proper
small-sample-corrected AICc would at n=11 -- flagged in
`validate_manakin.py`'s output rather than silently resolved, since it's
not yet clear whether this is a labeling quirk in the paper or a difference
in HZAR's own AICc implementation. It doesn't affect model *ranking* in
this example either way.

Run it yourself with:

```bash
python examples/validate_manakin.py
```

## Validation against HZAR: Phase 2 (Bayesian/NUTS)

`examples/demo_bayesian_manakin.py` fits the same real Ada locus data with
the NumPyro/NUTS backend and compares three ways: HZAR's published result,
HyCline's own Phase 1 MLE fit, and HyCline's Phase 2 posterior.

| | center | width |
|---|---|---|
| HZAR published | 170.9 km | 263.1 km |
| HyCline MLE (Phase 1) | 171.0 km | 262.0 km |
| HyCline NUTS (Phase 2) | 165.3 km | 364.0 km |

Center agrees closely across all three. Width is the more interesting
case: **this dataset has only 11 sampling sites**, and with the Bayesian
backend's *default* (very loose) width prior, NUTS's posterior mean for
width drifts to ~990 km with convergence diagnostics correctly flagging
the fit as unreliable (R-hat 1.02, divergences, low ESS) -- a real
demonstration of `hycline.diagnostics.convergence.check()` catching a
genuine identifiability problem rather than silently returning a bad
number. Bounding the width prior at the transect's actual geographic span
(still generous, just not absurd) resolves this cleanly -- R-hat 1.00,
zero divergences -- and while the posterior mean (364 km) still sits above
the MLE/HZAR point estimate, the 95% credible interval (`[163, 569]`)
comfortably contains it. That's the honest way to read this: agreement via
"the interval contains the reference value," not "the posterior mean
exactly matches a point estimate from a different fitting procedure with
different assumptions." See BACKEND_PHASE2.md section 9 for the full writeup, and
`tests/test_mcmc_manakin_bayesian.py` for the regression tests locking in
both the failure mode (default prior) and the fix (bounded prior).

Run it yourself with:

```bash
python examples/demo_bayesian_manakin.py
```

## Validation: multi-locus wide-format reading against HZAR's real dataset

`examples/manakin_multilocus.csv` is HZAR's own bundled `manakinMolecular.csv`
(all 7 loci: ada, ak2, gsr, pgm2, l5, pscn3, mtDNA), converted to a
standard comma-delimited CSV with no other changes -- same source as the
single-locus Ada validation above, just the full multi-locus file this
time. `read_multi_locus_wide_csv`'s *default* column-suffix settings
(`.nSamples`, `.A`) read all 7 loci from it correctly with zero
configuration, since those defaults were chosen to match HZAR's own naming
convention exactly.

```bash
hycline fit-geographic --input examples/manakin_multilocus.csv \
    --site-col locationID --method classic --scaling fixed \
    --output-dir results/ --prefix manakin
```

The `ada` locus extracted this way gives the identical fit to the
single-locus validation above (center=171.04, width=261.97 -- exact
match, since it's literally the same underlying numbers reached through
the wide-format reader instead of the single-locus one). The other 6 loci
fit cleanly too and show a genuinely informative real result: most cluster
tightly around center ~210-220 km with narrow widths (28-78 km), while
`ada` is a real outlier with a much wider, shallower cline -- consistent
with what a biologist would want to see from a multi-locus concordance
plot, produced automatically by the `_overlay.png` output.
Covered by `tests/test_data_io.py` and `tests/test_cli.py`.

## Validation: Phase 3 (genomic clines) against synthetic data with known parameters

As explained in BACKEND_PHASE3.md section 1, HZAR has no genomic cline
functionality, so there's no published number to reproduce the way the
sections above validate against HZAR's own output. What exists instead is
validation against synthetic data with known true parameters --
`examples/synthetic_genomic_clines.csv` simulates 150 individuals across 5
loci with deliberately different true `(u, v)` values, and fitting
recovers all of them correctly:

```bash
hycline fit-genomic --input examples/synthetic_genomic_clines.csv \
    --hybrid-index-col hybrid_index --individual-col individual \
    --output-dir results/ --prefix synthetic
```

| Locus | True (u, v) | Fitted (u, v) | Correctly identified as |
|---|---|---|---|
| neutral1 | (0.0, 1.0) | (0.21, 0.90) | ~neutral |
| neutral2 | (0.1, 0.95) | (-0.14, 0.85) | ~neutral |
| barrier1 | (0.0, 5.0) | (-0.09, 5.13) | barrier locus (elevated resistance) |
| permeable1 | (0.0, 0.3) | (-0.08, 0.22) | elevated introgression |
| biased1 | (2.5, 1.0) | (2.77, 1.26) | biased/asymmetric introgression |

The barrier and permeable loci in particular are unambiguous even with
this modest sample size (v clearly separated from 1 in the correct
direction in both cases), and the overlaid multi-locus plot
(`synthetic_overlay.png`) makes the qualitative pattern
immediately visible -- neutral loci hug the diagonal, `barrier1` shows a
sharp step, `permeable1` a shallow S-curve, and `biased1` a clear leftward
shift. Covered by `tests/test_genomic_cline.py`, `tests/test_mle_genomic.py`,
and `tests/test_cli.py`.

**What this synthetic validation does *not* establish on its own**:
agreement with an independent real-world implementation on real data --
see the next section for that (now done, on a real 77-locus dataset).

Run it yourself with:

```bash
hycline fit-genomic --input examples/synthetic_genomic_clines.csv \
    --hybrid-index-col hybrid_index --individual-col individual \
    --output-dir results/
```

## Validation: Phase 3 cross-validation against BGC on real data (77 loci)

The synthetic-data validation above establishes that HyCline's math is
correct; it doesn't establish agreement with an independent, real-world
implementation on real data. This section closes that gap.

`gghybrid` (Bailey 2024) bundles both a real dataset and precomputed
output from BGC (Gompert & Buerkle 2012) -- a well-established genomic
cline tool from the same model lineage -- used by the gghybrid author for
her own validation work. That means real ground truth was already sitting
in a public repository, put there for exactly this kind of comparison:

- **Dataset**: Hermansen et al. (2014)'s Italian sparrow (*Passer
  italiae*) hybrid zone -- 77 real loci (mostly SNPs, plus the mtDNA
  marker ND2), 433 individuals.
- **Reference values**: BGC's own published per-locus `alpha`/`beta`
  estimates and per-individual hybrid index, on this exact dataset.

`examples/hermansen_italy_sparrows.csv` is this data reshaped into
HyCline's format (all 433 individuals matched cleanly in the conversion,
none dropped); `examples/hermansen_italy_bgc_reference.json` holds BGC's
own values for comparison. Fitting all 77 loci with HyCline and comparing
against BGC:

| Comparison | Pearson r | Spearman rho | Sign agreement |
|---|---|---|---|
| HyCline `u` vs. BGC `alpha` | 0.85 (p=1.3e-22) | 0.68 (p=1.3e-11) | 75.3% |
| HyCline `v` vs. BGC `beta` (raw scale) | 0.85 (p=1.2e-22) | 0.66 (p=6.1e-11) | -- |
| HyCline `ln(v)` vs. BGC `beta` (log scale) | 0.69 (p=2.5e-12) | 0.66 (p=6.1e-11) | 81.8% |

Strong, highly significant, same-direction agreement across every one of
the 77 loci, under either plausible interpretation of how BGC's `beta`
maps onto `v` (see below for why there are two).

**What this does and doesn't establish, stated plainly.** BGC's exact link
function isn't confirmed identical to the Fitzpatrick (2013) logit-logistic
model HyCline implements -- Fitzpatrick's own paper frames the
logit-logistic form as an *alternative* to Gompert & Buerkle's original
parameterization, not a reimplementation of it. So this validates via
correlation and sign agreement (both robust to getting the exact
alpha/beta-to-u/v transform slightly wrong), not exact numerical
reproduction of BGC's specific reported numbers. What it does establish:
two independent implementations of closely related genomic cline models,
run on the same real 77-locus dataset, recover the same qualitative
per-locus signal with high statistical consistency -- a real, meaningful
cross-check that the model and MLE fitting are doing the right thing on
real data, not just on synthetic data with known answers baked in.
Confirming BGC's exact link function for bit-for-bit numerical parity is a
reasonable follow-up if that becomes a priority; see BACKEND_PHASE3.md
section 7.3 for the full discussion, and section 7.1-7.2 for exactly how
the data was obtained and reshaped, with full attribution.

Run it yourself with:

```bash
python examples/validate_gghybrid_italy.py
```

This regenerates `examples/hermansen_italy_gghybrid_cross_validation.png`
(scatter plots of both comparisons) and prints the full statistics.
Covered by `tests/test_gghybrid_cross_validation.py`.

## Validation: Phase 4 Component A (hierarchical geographic clines) via shrinkage on synthetic data

There is no HZAR-equivalent gold standard for hierarchical pooling either
(HZAR doesn't do it at all -- see BACKEND_PHASE4.md section 1), so
validation here is against synthetic data with a known population-level
structure. But "recovers the right numbers" isn't actually the most
important thing to check for a hierarchical model -- **shrinkage** is: a
sparse, noisy locus should get pulled toward the population mean more than
a data-rich locus does, since it has less of its own information to
resist that pull.

`examples/demo_hierarchical_geographic.py` simulates 8 loci sharing one
40-site transect, with known population-level structure
(`mu_center=75, sigma_center=10, mu_width=25`) and deliberately varied
sample sizes -- one sparse locus (4 alleles/site), one data-rich locus (60
alleles/site), six moderate (20 alleles/site):

```bash
python examples/demo_hierarchical_geographic.py
```

**Population-level hyperparameter recovery:**

| Parameter | True | Recovered |
|---|---|---|
| mu_center | 75.0 | 76.0 |
| sigma_center | 10.0 | 9.9 |
| mu_width | 25.0 | 27.5 |

**Shrinkage** (comparing each locus's hierarchical estimate against its
own independent, unpooled fit):

| | independent | hierarchical | shrunk toward population mean by |
|---|---|---|---|
| locus_0 center (sparse, n=4) | 80.82 | 80.38 | 9.3% |
| locus_1 center (rich, n=60) | 74.44 | 74.52 | 5.3% |
| locus_0 width (sparse, n=4) | 36.05 | 34.35 | 19.8% |
| locus_1 width (rich, n=60) | 32.87 | 31.95 | 17.1% |

Correctly directioned in both cases -- the sparse locus shrinks more than
the data-rich one -- and the data-rich locus's hierarchical estimate stays
close to its own independent estimate (center moved by only 0.08 units),
showing the model correctly "gets out of the way" when a locus has enough
of its own data rather than imposing unwanted pooling bias. Reported
honestly rather than oversold: the magnitude here is modest, not dramatic
-- see BACKEND_PHASE4.md section 4.3 for why, and what would make the
contrast sharper.

**A genuinely useful bug got caught building this validation**, worth
knowing about if you use this module: a locus's fitted width came back
roughly 2x too high across most of the batch, traced (via cross-checking
against the already-validated Phase 1 MLE on the same data, not assumed
away) to a single noisy extreme site locking the model's `pmin`/`pmax`
asymptote at a badly wrong value. Fixed by averaging several near-extreme
sites instead of anchoring on one -- see BACKEND_PHASE4.md section 3 for
the full investigation and fix, including why it's currently scoped to
just the hierarchical model rather than applied everywhere. Both the
hyperparameter recovery and shrinkage numbers above are from *after* this
fix.

Covered by `tests/test_mcmc_hierarchical.py`, including a dedicated
regression test for the bug fix itself
(`test_derive_pmin_pmax_robust_to_single_noisy_extreme_site`).

## Validation: Phase 4 Component B (hierarchical genomic clines) via shrinkage on synthetic data

The genomic-cline analog of Component A's validation above -- same idea
(no HZAR-equivalent gold standard, so synthetic data with known
population-level structure, with shrinkage as the signature that
actually matters), applied to the `u`/`v` genomic cline parameters
instead of geographic center/width.

`examples/demo_hierarchical_genomic.py` simulates 8 loci sharing 120
individuals, with known population-level structure centered at the
neutral null (`mu_u=0, sigma_u=1.0, mu_v=1.0, sigma_log_v=0.5` -- see
"Why hierarchical clines?" above for why the neutral point is the
biologically meaningful center, not an arbitrary default) and two
realistic sparsity extremes: a single low-confidence SNP (`n_alleles=1`
at every individual) and a resolved multi-SNP ancestry block
(`n_alleles=20`, representing pooled evidence from several tightly
linked markers):

```bash
python examples/demo_hierarchical_genomic.py
```

**Shrinkage** (comparing each locus's hierarchical estimate against its
own independent, unpooled fit):

| | independent | hierarchical | shrunk toward population mean by |
|---|---|---|---|
| locus_0 u (sparse, n=1) | -0.201 | -0.189 | 2.6% |
| locus_1 u (rich, n=20) | -0.854 | -0.854 | 0.0% |
| locus_0 v (sparse, n=1) | 0.458 | 0.529 | 38.9% |
| locus_1 v (rich, n=20) | 0.488 | 0.495 | 4.0% |

A cleaner, more decisive result than Component A's geographic validation
-- especially for `v` (38.9% vs. 4.0%). **Getting there took two failed
attempts, reported rather than discarded**: an initial `n=1` vs. `n=2`
contrast (only a 2x information difference) showed almost no shrinkage
signal at all, and increasing to 400 individuals to reduce noise made it
worse, not better -- even `n=1` per individual across 400 individuals
turned out to already be enough data that the population-level prior had
little left to contribute. The eventual scenario above uses a much larger,
more realistic contrast (`n=1` vs. `n=20`, representing a real difference
in marker resolution) with fewer individuals (120), and that's what
produced a clean, correctly-directioned signal. See BACKEND_PHASE4.md
section 7.3 for the full account, including why the two failed attempts
are informative in their own right about how much sparsity contrast is
actually needed for shrinkage to be visible, not just that it exists in
principle.

**Population-level recovery** also behaved correctly, though worth a note
on how to read it: with only 8 loci, the hierarchical fit's population
mean estimate tracks the *sample* mean of the 8 true per-locus values used
in that particular simulation, not necessarily the *theoretical*
population value (8 random draws from a distribution rarely average out
to exactly the population mean) -- this is expected behavior, not bias,
and `tests/test_mcmc_genomic_hierarchical.py` checks against the sample
mean accordingly.

Covered by `tests/test_mcmc_genomic_hierarchical.py`, including a
regression test confirming the JAX-native genomic cline core matches the
NumPy version used everywhere else in the package
(`test_jax_genomic_core_matches_numpy_core`), the same parity check
pattern used for the geographic model.

## Validation: Phase 4 Component C (coupling/concordance analysis)

Component C's validation is different in character from A and B's: the
raw variance metrics (`geographic_cline_variance`, `genomic_cline_variance`)
are a direct implementation of Firneno et al. (2023)'s own definitions, so
there's nothing to calibrate there -- the validation that matters is
whether the *theta estimate* (reverse-engineered from their published
Table 1, see "Why hierarchical clines?" above and BACKEND_PHASE4.md
section 9.3 for the full methodology) actually reproduces their results,
and whether the analysis behaves sensibly end-to-end on hierarchically-fit
synthetic data.

**Reproducing Firneno et al.'s own published values**: fitting HyCline's
calibration to their Table 1 and checking it against the same table's
theta column gives a maximum absolute error of 0.0049 across all 31 data
points (`tests/test_coupling.py::
test_calibration_reproduces_firneno_2023_table1_closely`) -- consistent
with simply reading values off a table rounded to 2 decimals, which is
strong evidence this recovers their actual applied model rather than
fitting fresh noise (their own regression on the real, much larger
simulated calibration dataset explained only 52.3% of theta's variance;
a 31-point regression hitting 99.99% would be implausible unless it's
recovering a near-deterministic input-output mapping, not fitting noise).

**End-to-end behavior**: `examples/demo_coupling.py` fits concordant and
discordant synthetic locus sets hierarchically (via Components A/B) and
runs them through Component C:

| Scenario | sigma_beta (geographic) | CV | sigma_c (genomic) | sigma_v | theta |
|---|---|---|---|---|---|
| Concordant | 0.0152 | 0.089 | 0.105 | 0.064 | 1.845 |
| Discordant | 0.0668 | 0.406 | 0.797 | 0.575 | -0.490 (flagged) |

The geographic metrics behave exactly as expected in both directions --
concordant loci show roughly 4x lower slope SD and CV than discordant
ones. The genomic case shows the same qualitative pattern in the raw
sigma_c/sigma_v values, and the "concordant" theta estimate (1.845, above
1, consistent with strong coupling) is sensible -- but the "discordant"
theta estimate comes back negative, a real and important limitation
documented rather than hidden.

**Why the negative estimate happened, and why it's informative rather
than just a bug**: theta = s/r is a ratio of two positive quantities and
is theoretically always positive. A negative estimate here doesn't mean
"very weak coupling" -- it means this particular (sigma_c, sigma_v)
combination falls somewhere the fitted linear+interaction surface doesn't
behave well, even though both values individually sit within the
reference data's observed range ([0.29, 1.73] for sigma_c, [0.08, 0.56]
for sigma_v -- the discordant scenario's sigma_c=0.797 and sigma_v=0.575
are both inside those bounds). This is expected once you consider that 31
scattered 2D points support a smaller, irregular convex-hull region far
better than they support the full rectangular bounding box implied by
each coordinate's min/max -- and it's exactly why
`GenomicClineVariance.summary()` prints an explicit `WARNING` whenever
theta comes back non-positive, and why `within_calibration_range` reports
whether a query point falls inside the reference data's bounding box at
all (necessary but, as this example shows, not sufficient on its own).
The raw sigma_c/sigma_v values remain valid and useful either way --
they're what's actually shown to be different between the concordant and
discordant scenarios above, with or without a trustworthy theta.

Covered by `tests/test_coupling.py` (17 tests, including a dedicated
regression test locking in this exact negative-theta finding) and
`examples/demo_coupling.py`.

Run it yourself with:

```bash
python examples/demo_coupling.py
```

## Validation: Phase 5 (genotype-likelihood-aware fitting)

This validation is worth reading in full, not skimming for the headline
number -- the story here is as much about a hypothesis that *didn't* pan
out as one that did, and both halves are documented rather than only the
flattering one.

**What was expected going in**: that naively hard-calling genotypes at
low sequencing depth (taking the single most likely genotype and treating
it as certain) would show measurable *bias* in fitted cline center/width,
compared to properly marginalizing over genotype uncertainty.

**What actually happened**: comparing GL-aware fits against hard-call fits
across sequencing depths from 1x to 30x showed nearly identical point
estimates at every depth -- no bias that shrinks with depth, the pattern
that would have confirmed the original hypothesis. Investigating why
(rather than just reporting a null result and moving on): a single random
read from a diploid individual samples one of their two chromosome copies
uniformly at random, so a maximum-likelihood hard call at very low depth
is, in expectation, an *unbiased* (if noisy) proxy for that individual's
true fractional genotype. The "reference bias" phenomenon often discussed
for low-coverage genotyping in the population-genomics literature is
understood there to be predominantly a mapping/alignment artifact, which
this project's simulation deliberately doesn't model (only symmetric
sequencing error) -- so this validation wasn't actually built to detect
that specific effect, and it's honest to say so.

**What did hold up, cleanly**: the more standard and defensible motivation
for genotype-likelihood methods in the literature (this is explicitly how
tools like ANGSD frame their own rationale) -- proper uncertainty
propagation, not point-estimate correction. Checked via 95% CI coverage
across 40 independent sequencing simulations at each depth:

| Depth | GL-aware coverage | GL-aware mean SE | Hard-call coverage | Hard-call mean SE |
|---|---|---|---|---|
| 1 | 82.4% | 3.91 | 66.7% | 2.79 |
| 2 | 94.3% | 3.32 | 86.1% | 2.87 |
| 3 | 94.6% | 3.16 | 89.5% | 2.93 |
| 5 | 97.4% | 2.84 | 85.0% | 3.11 |
| 8 | 97.5% | 2.69 | 92.3% | 2.65 |
| 15 | 100.0% | 2.63 | 97.5% | 2.63 |
| 30 | 100.0% | 2.61 | 100.0% | 2.61 |

At low depth, hard-calling is measurably overconfident: its reported
standard errors are *smaller* than GL-aware's despite there being more
real uncertainty in the data (e.g. depth 1: SE=2.79 vs. SE=3.91), and its
confidence intervals cover the true value far less often than their
nominal rate (67% actual vs. 95% nominal). GL-aware fitting's coverage
stays close to nominal throughout. Both converge as depth increases --
exactly as expected once genotype calls become effectively certain either
way, with nothing left for the marginalization to correctly account for.

Covered by `tests/test_genotype_likelihood_data.py` (10 tests: Phred
conversion, the GL simulator's depth-dependent behavior) and
`tests/test_genotype_likelihood_fitting.py` (6 tests: marginal-likelihood
math verified against hand computation, a regression test confirming hard
calls reduce exactly to the plain HWE likelihood, MLE recovery at high
depth, and the CI-coverage calibration finding itself).

Run it yourself with:

```bash
python examples/demo_genotype_likelihood.py
```

## Validation: Phase 4 Component D (hierarchical geographic clines + genotype likelihoods)

The question this component exists to answer: once loci are already
pooled hierarchically (Component A), does genotype-likelihood-awareness
(Phase 5) still add value, or does pooling alone already fix what Phase 5
fixes for independent per-locus fits? This wasn't assumed going in -- it
was tested directly, and the answer turned out to have a more consequential
mechanism than the original hypothesis predicted.

**Setup**: 6 loci sharing 100 individuals, one locus at sequencing depth 2
(sparse), the rest at depth 8. Two approaches compared on the same
simulated data each time: the full GL-aware hierarchical model
(`fit_hierarchical_batch_gl`) versus "hard-call then hierarchically pool"
(take each individual's maximum-likelihood genotype and feed the result
to Component A's existing `fit_hierarchical_batch`). 12 independent
replicate simulations, each refit both ways.

**Result: pooling alone is not enough.**

| | GL-aware | Hard-call |
|---|---|---|
| Sparse-locus center coverage, 95% CI | 100% (12/12) | 75% (9/12) |
| Sparse-locus width coverage, 95% CI | 100% (12/12) | 8.3% (1/12) |
| Mean center CI width | 13.91 | 13.77 |
| Mean width CI width | 32.53 | 31.18 |

**The mechanism is more consequential than expected.** The CI *widths*
above are nearly identical between the two methods -- so this isn't the
same "hard-call is overconfident with too-narrow intervals" story from
Phase 5's standalone (non-hierarchical) finding. It's a *location*
problem: hard-call's intervals are about the same size, but centered in
the wrong place. Checking the actual posterior means (not just coverage
booleans) on a representative replicate: the sparse locus's true width is
31.1. The GL-aware fit's estimate is 29.3 (close). The hard-call fit's
estimate is 70.3 -- badly inflated, the same kind of width
non-identifiability documented throughout this project when data is
sparse. **Critically, the population-level `mu_width` hyperparameter is
also distorted**: 33.4 under GL-aware (close to the true population mean
of 25) versus 60.7 under hard-call -- more than double. The hard-called
sparse locus's bad estimate doesn't stay contained to itself; the
hierarchical model pools it into the *shared* hyperparameter that every
other locus's prior is centered on.

This reframes the practical stakes: it's not just "does genotype-
likelihood-awareness help one sparse locus's own estimate" (Phase 5's
finding, still true) -- it's "does a poorly-sequenced locus contaminate
the shared population-level estimate that *every* locus in a hierarchical
batch gets pulled toward." That's a risk specific to combining
hierarchical pooling with imperfect low-coverage data, and arguably the
stronger reason to use genotype-likelihood-aware fitting whenever both
are in play. See BACKEND_PHASE4.md section 11 for the full investigation,
including why a JAX-native reimplementation of the marginal-likelihood
math was needed (the existing NumPy/SciPy version isn't traceable inside
a NumPyro model, the same reason Phase 2/3's hierarchical models each
carry their own JAX-native cline core) and how `pmin`/`pmax` derivation
was generalized from hard counts to genotype likelihoods.

Covered by `tests/test_mcmc_hierarchical_gl.py` (6 tests: JAX/NumPy
parity, `pmin`/`pmax` derivation correctness, population-hyperparameter
recovery, and a regression test locking in the coverage-comparison finding
on one representative seed).

Run it yourself with:

```bash
python examples/demo_hierarchical_gl.py
```

## Validation: Phase 4 Component E (hierarchical genomic clines + genotype likelihoods)

The genomic-cline analog of Component D's validation above -- and worth
reading precisely because it finds something genuinely different, not a
repeat of Component D's result.

**Moderate depth contrast** (the same setup as Component D's own study: 6
loci, one at depth 2, the rest at depth 8, shared individuals), 8
independent replicates:

| | GL-aware | Hard-call |
|---|---|---|
| Sparse-locus `v` coverage, 95% CI | 100% (8/8) | 100% (8/8) |
| Sparse-locus `u` coverage, 95% CI | 87.5% (7/8) | 87.5% (7/8) |

**Identical performance.** At the exact depth contrast that produced
Component D's 100% vs. 8.3% gap for geographic clines, genotype-
likelihood-awareness makes no detectable difference for genomic clines
under hierarchical pooling. This was checked directly rather than assumed
to transfer, and it's reported as the real finding rather than adjusted
to manufacture a difference that isn't there.

**Extreme depth contrast** (depth 1 vs. depth 20, fewer individuals),
8 further replicates:

| | GL-aware | Hard-call |
|---|---|---|
| Sparse-locus `v` coverage, 95% CI | 100% (8/8) | 87.5% (7/8) |
| Mean absolute error, point estimate | 0.410 | 0.176 |

A real difference finally appears, but via a **different mechanism** than
Component D's: hard-call's mean point-estimate error (0.176) is actually
*smaller* than GL-aware's (0.410) -- hard-call is, on average, closer to
the true value in raw terms. GL-aware still achieves better coverage
despite this, meaning the issue here is closer to Phase 5's own
standalone finding (hard-calling is overconfident, with intervals too
narrow relative to their real error) than to Component D's "population
contamination" story.

**Why the two components differ**: two plausible structural reasons, not
independently verified beyond the empirical pattern itself. First,
genomic clines have no `pmin`/`pmax`-equivalent free asymptote parameter
for sparse data to corrupt -- Component D's dramatic result traced
directly to that specific failure mode, and there's no analog here.
Second, hybrid index always spans the full [0,1] range by construction,
unlike geographic distance, where sparse or clustered sampling can leave
a cline's actual transition zone poorly covered -- this likely gives the
genomic rate parameter inherently better-conditioned identifiability even
from uncertain genotypes.

**Practical takeaway**: the recommendation genuinely differs by cline
type. For geographic clines under hierarchical pooling, use genotype-
likelihood-aware fitting whenever depth varies meaningfully across loci
-- the contamination risk showed up even at a moderate contrast. For
genomic clines under hierarchical pooling, it matters less unless the
depth contrast is genuinely extreme (closer to 1x than 2x).

Covered by `tests/test_mcmc_genomic_hierarchical_gl.py` (3 tests:
population-hyperparameter recovery, a regression test locking in the
"no difference at moderate contrast" finding, and a regression test
locking in the "GL-aware wins at extreme contrast" finding on one
representative seed).

Run it yourself with:

```bash
python examples/demo_genomic_hierarchical_gl.py
```

## Validation: CLI wiring for Phase 4 Components C, D, and E

Unlike the components' own statistical validations above, this is about
confirming the CLI *plumbing* works correctly end to end -- real CSV in,
correct fitting function called, correct files out -- not re-deriving the
underlying findings. All three new command groups were run on real data
generated fresh for this purpose (not pre-built test fixtures), and on
the actual output files earlier commands produced.

**`coupling-geographic`/`coupling-genomic`** were run directly on the real
output of `fit-hierarchical-geographic`/`fit-hierarchical-genomic` from earlier in
this README (the Manakin multi-locus dataset and the Phase 3 synthetic
genomic dataset) -- not synthetic locus-summary fixtures built just for
this test. The genomic run organically triggered Component C's own
documented negative-theta warning: this particular synthetic dataset's
real spread of barrier/permeable/biased loci happened to fall outside the
theta calibration's supported range, and the CLI reported that explicitly
rather than a misleadingly precise number -- the same honest behavior
built into the Python API, now confirmed to survive the CLI's own summary
formatting and JSON output.

**`fit-hierarchical-geographic-gl`/`fit-hierarchical-genomic-gl`** needed real new
plumbing to test at all: no CSV reader for genotype-likelihood data
existed anywhere in the project before this -- Phase 5 itself only ever
took Python-constructed data containers. Both commands were run on freshly
generated synthetic multi-locus genotype-likelihood data with genuine
sparse/rich depth contrasts among loci (mirroring how Components D/E
themselves were validated), producing correctly-shaped grid plots (clear
sigmoid transitions for geographic loci; recognizable barrier/neutral
genomic cline shapes) and sensible parameter recovery despite fast,
low-iteration test settings (a `barrierX` locus with true `v=4.0`
recovered as `v_mean=5.34`). The `--phred` flag (for the Phred-scaled `PL`
format GATK/bcftools/ANGSD actually output) was verified as an exact
round-trip against the natural-log-encoded equivalent of the same data,
not just "runs without erroring."

**The dense-data requirement's error handling was tested for all four new
fitting commands** (`fit-hierarchical-geographic-gl`/`fit-hierarchical-genomic-gl`,
reusing the exact same `_stack_for_hierarchical` enforcement built for
Components A/B) by constructing real CSVs with a deliberately missing
value in one locus's data and confirming a clean, actionable error rather
than a crash.

Covered by `tests/test_data_io.py` (7 new tests for the GL readers) and
`tests/test_cli.py` (11 new tests spanning all four new command groups).
See BACKEND_PHASE4.md section 14 for the full design writeup.

## Validation: CLI wiring for Phase 5's own standalone fitting

Same spirit as the Components C/D/E CLI validation above -- confirming
the plumbing, not re-deriving the underlying Phase 5 findings. All four
commands (`fit-geographic-gl`, `fit-genomic-gl`, `fit-geographic-gl-multi`, `fit-genomic-gl-multi`)
were run end-to-end on freshly generated synthetic data.

**Single-locus commands correctly recovered known true parameters**:
`fit-geographic-gl` on a `center=50` geographic cline recovered a sensible fit;
`fit-genomic-gl` on a `v=4.0` genomic barrier locus recovered `v=4.56`.

**Multi-locus commands needed real new plotting code, not a shortcut
reuse of Components D/E's.** Independent multi-locus fitting has no
dense-data requirement (unlike Components D/E's hierarchical batch
models) -- different loci can freely have different numbers of
individuals, since each is fit on its own. This was caught mid-build: an
initial version of `fit-geographic-gl-multi` left dead code (unused variables, a
no-op conditional) where the grid-plot logic should have been, rather
than either building it correctly or leaving the feature out cleanly.
Fixed by building two new, correctly-scoped plotting functions that
handle each locus's own independent array, mirroring the *original*
Phase 1/3 small-multiples plotters' shape rather than the newer
hierarchical-specific ones. Both multi-locus commands were then run
end-to-end and produced correctly-shaped grid plots for three
independent loci each, including `fit-geographic-gl-multi` surfacing a real,
legitimate edge case (one locus's Hessian-based standard errors coming
back unavailable -- the existing correct fallback for a non-invertible
Hessian, reported plainly rather than papered over).

**The `--phred` flag was verified at the full CLI level, not just the
reader level**: `fit-geographic-gl` was run twice on the same underlying data --
once with natural-log GL columns, once with the same values re-encoded
as Phred-scaled PL and read back in with `--phred` -- confirming the
fitted parameters matched to `1e-4`, exercising the complete path
(argument parsing, file reading, conversion, fitting) rather than only
the conversion function in isolation.

Covered by `tests/test_data_io.py` (5 new tests for the single-locus GL
readers) and `tests/test_cli.py` (10 new tests spanning all four
commands). See BACKEND_PHASE5.md section 6 for the full design writeup.

## Validation: the VCF + metadata converter

Real end-to-end validation, not just parser unit tests: freshly simulated
synthetic VCFs with known true generating parameters were run through the
*actual CLI commands* (not just the Python API) and checked against
ground truth.

**Geographic**: a synthetic dataset (8 sites, 6 individuals each, true
cline center=70, width=25) was written as a VCF, converted via `hycline
generate-geographic-input` (site-aggregated hard counts), and fit via
`hycline fit-geographic-multi` -- recovered center=75.0, in line with the same kind
of finite-sample width variability already documented throughout this
project (BACKEND_PHASE1.md/BACKEND_PHASE2.md's own case studies on this
exact theme).

**Genomic**: a synthetic barrier locus (true `u=0, v=4.0`) was written as
a VCF with per-individual hybrid index, converted via `hycline
convert-vcf-genomic`, and fit via `hycline fit-genomic-multi` --
recovered `v=3.99`, essentially exact.

**Two real bugs were found and fixed while building this component, not
discovered later**: (1) a locus with genotype-likelihood data missing for
every individual originally produced a broken all-missing column triple
that made the *entire file* fail to read downstream -- fixed by excluding
such loci up front with a clear report of what was dropped and why; (2) if
every locus in a file had this problem, the command still reported success
and left a useless, locus-free CSV on disk -- fixed to fail loudly (exit
code 2, explanatory message, no misleading file left behind) instead.
Both are locked in as regression tests, not just fixed and forgotten.

**Scope is stated plainly, not discovered by users hitting a wall**:
multiallelic sites and indels are detected and skipped (with counts
reported), not guessed at -- correctly handling them would need a real
modeling decision this project isn't in a position to make silently
(which allele is "focal" at a 3-allele site?). `bcftools view -m2 -M2
-v snps` before conversion is the practical workaround for a VCF with
either.

Covered by `tests/test_vcf.py` (20 tests: parser correctness against
hand-computed VCF fixtures, gzip handling, all four writers, the two bug
fixes above) and `tests/test_cli.py` (8 tests: both CLI commands,
including the two known-ground-truth recovery tests). See
BACKEND_PHASE1.md section 9.4 for the full design writeup.

## Validation: tail-model support for Bayesian and genotype-likelihood-aware fitting

This section documents a real, negative-and-positive finding pair, not
just a feature announcement -- worth reading in full if you're deciding
whether to use `--tail` with `--method bayesian`.

**What was expected**: extending tail-model support (`TailType.LEFT`/
`RIGHT`/`BOTH`/`MIRROR`) to the Bayesian backend would work like every
other addition to this project -- correct math, the usual caveats about
sparse data needing tighter priors, done. **What was found instead**:
Bayesian (NUTS) tail-model fitting has severe, genuine convergence
difficulty. Synthetic tail-model clines with realistic sample sizes
(80-200 sites) produced R-hat values from 1.9 up to 3.5+ and effective
sample sizes as low as 4 -- far outside acceptable territory -- even at
default sample counts.

**The investigation, not just the conclusion**: four explanations were
tested and ruled out in order before concluding this is fundamental --
insufficient data (ruled out: more data didn't help), too-loose priors
(ruled out: tightening `width_upper_bound` didn't help), needing more
warmup (ruled out: 4x the warmup didn't even *complete* within a time
budget the no-tail model finishes comfortably inside, with *better*
convergence, at 1/4 the warmup), and a wrong JAX translation (ruled out
directly -- the tail math matches the NumPy reference exactly, `atol=
1e-10`, for every tail type). The isolating experiment: the *exact same
simulated data* converges perfectly (R-hat=1.00, thousands of effective
samples) when fit with `TailType.NONE` instead of a real tail -- the only
variable that changed is whether tail parameters are present. The likely
mechanism: the tail model is continuous but not smooth where the sigmoid
core meets the exponential tail, and that join's location itself depends
on sampled `center`/`width` values, so the resulting kink in the
log-density moves as NUTS explores rather than sitting in one place to
route around -- a known, general difficulty for gradient-based samplers
applied to piecewise-smooth models, not specific to this codebase.

**This is presented as a genuine current limitation, not a bug to be
patched.** Smoothing the model to remove the kink would be a real
redesign (a different tail model, needing its own derivation and
validation against Barton & Gale 1993), explicitly out of scope. Instead:
both `fit_mcmc`/`fit_mcmc_batch` emit a `UserWarning` whenever a tail
type is requested, and the CLI prints the same guidance before fitting
starts, in addition to the existing convergence diagnostics that flag the
resulting bad R-hat/ESS afterward. **Use classic MLE for tail models** --
it doesn't have this issue.

**The positive half of the finding**: extending Phase 5's classic-MLE
`fit_mle_geographic_gl` to tail models surfaced a related-but-tractable
version of the same difficulty -- L-BFGS-B could report `success=False`
with a corner solution even when a much better solution existed nearby.
Unlike NUTS, this **was** fixable: independent randomized restarts (8 by
default) are cheap, and unlike MCMC's need to integrate *through* a kink
continuously, only one restart needs to land in the right basin. Verified
directly: multi-start never finds a worse solution than single-start on
matched data, and typically finds a substantially better one.

**Four real, previously-shipped bugs were found and fixed along the
way**, none caught by the existing test suite because no prior test had
run a tail-model fit all the way through to its plotting/CSV/JSON output
-- two plotting functions and two CLI summary-construction code paths all
independently hardcoded `TailType.NONE`/a fixed 4-parameter list. One had
genuine crash risk (constructing a `ClineParams` with tail fields left as
`None`, then calling `cline_function` with a real tail type); another
meant `fit-geographic-multi --tail mirror`'s classic-method summary CSV has silently
dropped tail parameter values since Phase 1 first shipped tail-model
support, not something introduced by this work. All four are fixed and
covered by regression tests.

Covered by `tests/test_mcmc_numpyro.py` (2 math-parity/warning tests, 3
tail-model smoke tests that deliberately don't assert convergence
quality -- only that the math is right and the sampler completes),
`tests/test_genotype_likelihood_fitting.py` (4 tests including the
multi-start regression test), `tests/test_plotting_bayesian.py` (1
regression test for the plotting bug), and `tests/test_cli.py` (3
regression tests for the CLI bugs, including two full end-to-end runs
that would have crashed before the fix). See BACKEND_PHASE2.md section
12 for the complete investigation.
