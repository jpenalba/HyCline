# HyCline

A Python successor to the R package HZAR (Derryberry et al. 2014) for hybrid
zone cline analysis, extended to genomic clines, quantitative trait clines,
and coupling/concordance analysis. All tests pass except one known,
pre-existing failure, `test_mcmc_numpyro.py::
test_diagnostics_check_runs_and_flags_nothing_on_a_clean_fit` (R-hat 1.0136
against a 1.01 cutoff). CLI entry point: `hycline` (or `python run_hycline.py`).

**Before doing anything else, read `README.md` in full** -- particularly the
"What's implemented" and "Not yet implemented (see the architecture plan)"
sections. That second section is a live, tiered priority list (Tier 1 =
small/contained/ready to pick up, through Tier 4 = large/speculative), kept
up to date with citations and reasoning, not just a bullet list. Treat it as
the actual backlog, not background reading.

## Current priorities (Tier 1, top of the list)

Two items were added most recently, both from a direct reading of Baird &
Daley (2025), *The Shapes of Clines and Wavefronts* (Mol Ecol,
doi:10.1111/mec.70109) -- see `references/` for the PDF, and README.md's
Tier 1 section for full detail:

1. **A geographic T-cline tail-model alternative.** HyCline's existing
   `--hzar-tail` tripartite model has a real, previously-only-empirically-
   documented convergence difficulty; Baird & Daley's paper gives the
   underlying reason (genuine mathematical discontinuities where the
   tripartite model's pieces join) and a validated, already-coded
   alternative (`Clinalyse`, github.com/Studenecivb/clinalyse) to port from.
2. **Adaptive-introgression candidate flagging**, specified in detail in
   README.md's Tier 1 section after reading four primary sources directly
   (Gompert et al. 2012; Bailey 2024; Jofre & Rosenthal 2021; Gompert &
   Buerkle 2011 -- all in `references/`). Key constraints already worked
   out: rank by delta-AIC/delta-waic, not raw p-values (Bailey 2024 found
   p-values non-monotonic with true effect size); report center-outliers
   and rate-outliers separately; the output must carry a mandatory caveat
   that cline shape alone cannot identify the selective mechanism (this is
   the original bgc authors' own stated conclusion about their model, not
   a hedge); and the method has a documented blind spot for epistatic
   incompatibilities specifically (0.81-0.95 false discovery rate in
   Gompert & Buerkle 2011's own simulations).

Also worth reading before building on top of them further: Fitzpatrick
(2013), the original logit-logistic genomic cline paper HyCline's own model
is built from, and Firneno et al. 2023, which already shapes the
`--pool-center` reparameterization and `coupling-genomic`'s theta
calibration -- both are cited repeatedly by papers already read for this
project, but neither has been read directly yet. Worth closing that gap.

## References folder

`references/` holds primary-source PDFs read directly during this project's
development, cited by DOI in `README.md`'s roadmap and in relevant
docstrings/BACKEND_PHASE*.md sections. When a design decision or a
documented limitation cites a paper, prefer reading the actual PDF here over
trusting a secondhand summary -- that's the standard this project has tried
to hold itself to throughout (see "Key principles" below).

## Where things live

- `hycline/cli.py` -- the CLI, 13 top-level commands.
- `hycline/models/` -- the cline functions themselves (`cline.py`,
  `genomic_cline.py`).
- `hycline/likelihoods/` -- binomial likelihood, genotype-likelihood-aware
  variants.
- `hycline/inference/` -- classic MLE (`mle.py`, `mle_genotype_likelihood.py`)
  and Bayesian/NUTS (`mcmc_numpyro.py`, `mcmc_hierarchical.py`,
  `mcmc_genomic_hierarchical.py`, `mcmc_pymc.py`).
- `hycline/data/` -- readers/writers, VCF/beagle handling, AIM filtering.
- `hycline/viz/` -- plotting.
- `tests/` -- 477 tests. `test_cli.py` (every test runs the CLI as a
  subprocess, paying ~1.2 s of start-up) and the MCMC test files are the
  slowest. Run them in parallel with pytest-xdist (`pytest tests/ -n 4`);
  if the environment has tight per-command timeouts, split large runs into
  batches rather than assuming a single call will complete.
- `scripts/select_tests.py` -- selects the tests affected by the current
  changes (`--run` runs them, in parallel; `--base origin/main` includes
  local commits). See "Testing" below.
- `HyCline_UserGuide_*.md` -- user-facing docs per pipeline (Geographic,
  Genomic, Coupling, Trait, StudyDesign), verified command-by-command
  against real output, not just written from the code.
- `BACKEND_PHASE*.md`, `BACKEND_SCALE.md`, `BACKEND_TRAIT.md` -- the
  detailed development history, organized by phase, with the actual
  verification runs and real bugs found along the way.

## Key principles this project has held to -- keep holding to them

- **Verify before reporting.** Run real commands and show real output;
  don't reconstruct results from memory or assume correctness. This
  extends to citations -- when a paper is invoked to justify a design
  decision, read it, don't paraphrase a paraphrase.
- **A wrong first diagnosis is not a failure -- reporting it as right
  would be.** More than one real bug in this project's history was
  initially misdiagnosed before the actual root cause was found (e.g. the
  Sceloporus pmin/pmax "literal 0/1 saturation" theory, which direct
  likelihood comparison disproved before the real orientation-mismatch
  cause was found). State the correction plainly when it happens, including
  in your own reasoning, rather than quietly revising history.
- **Every real bug gets a regression test built from the actual data that
  surfaced it**, not a synthetic case that conveniently passes.
- **Documentation describes only the current state** -- no stale command
  names, no "previously/used to" language, no historical framing creeping
  into user-facing docs.
- **Targeted tests while working, the full suite before pushing to main**
  (see "Testing" below). Reconcile the full suite's batch counts against
  the total collected count explicitly, and never report "tests passing"
  from a partial run -- say which tests ran.

## Testing

- **While working:** `python scripts/select_tests.py --run` runs only the
  tests affected by uncommitted changes (`--base <ref>` to compare against
  something other than HEAD; pytest arguments go after `--`). It traces
  imports between `hycline` modules, maps `cli.py` changes and library
  changes to the CLI commands they reach and selects `test_cli.py` tests by
  the commands they run, runs changed test files in full, and selects
  tests that mention a changed data or script file (`examples/`,
  `benchmarking/`). Docs select nothing; anything it can't trace
  (`run_hycline.py`, `pyproject.toml`, requirements files, shared parts of
  `cli.py`) selects the full suite. Run it without `--run` to see what it
  picked and why.
- **Before pushing to main:** the full suite, `pytest tests/ -n 4` (or in
  batches, e.g. `test_cli.py` / the `test_mcmc*.py` files / everything else,
  if a single call would hit a timeout), with the counts reconciled
  against `pytest tests/ --co -q`. The selector traces names statically,
  so it can miss a dependency it can't see (a file read through a path
  built at run time, behaviour reached only through a subprocess other
  than the CLI); the full suite is what catches those.
