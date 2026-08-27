# HyCline

A Python successor to the R package HZAR (Derryberry et al. 2014) for hybrid
zone cline analysis, extended to genomic clines, quantitative trait clines,
and coupling/concordance analysis. 455 tests, all passing as of this file's
last update. CLI entry point: `hycline` (or `python run_hycline.py`).

**Before doing anything else, read `README.md` in full** -- particularly the
"What's implemented" and "Not yet implemented (see the architecture plan)"
sections. That second section is a live, tiered priority list (Tier 1 =
small/contained/ready to pick up, through Tier 4 = large/speculative), kept
up to date with citations and reasoning, not just a bullet list. Treat it as
the actual backlog, not background reading.

## Current priorities (Tier 1, top of the list)

The two items previously headlined here -- a geographic T-cline tail-model
alternative (Baird & Daley 2025), and candidate-locus flagging (Fitzpatrick
2013, adapted to both cline types) -- are both done, not backlog. See
README.md's "What's implemented" section (the `t_cline_function` and
`flag_geographic_candidates`/`flag_genomic_candidates` bullets) for what
shipped, and BACKEND_PHASE2.md section 15 / BACKEND_PHASE4.md section 15
respectively for the full design writeups. Fitzpatrick (2013) and Firneno
et al. (2023) -- both cited by the papers above but not read directly at
the time this section was first written -- have since been read directly
too (commit `b83afc0`), with every specific claim checked against the PDF
text rather than trusted secondhand.

What's actually next, in order, per README.md's Tier 1 section:

1. **Two small, genuinely separate items surfaced by the candidate-flagging
   work but not part of it**: a `fit-hierarchical-genomic` warning for
   uneven hybrid-index sampling (Bailey 2024's own finding that this
   degrades accuracy specifically for steep genomic clines), and a
   delta-AIC/WAIC ranking addition for the classic-MLE genomic path
   (`fit-genomic`/`fit-hierarchical-genomic`), independent of the
   hierarchical-only flagging feature already built.
2. Several older Tier 1 items already investigated and scoped, not
   freshly discovered: a `TailType.MIRROR` parameter-naming inconsistency
   between classic and Bayesian output; missing HZAR-tail support in
   `fit_mle_fixed_extremes` and in Components A/D (Phase 4's hierarchical
   geographic models); real-world input-compatibility gaps found against
   HZAR's own bundled datasets (semicolon-delimited CSVs, multiallelic-
   locus handling, no STRUCTURE/PLINK reader); and unbuffered stdout on
   long multi-locus/Bayesian runs. See README.md's Tier 1 section for the
   full detail on each -- all are scoped, none require further
   investigation before picking up.

## References folder

`references/` holds primary-source PDFs read directly during this project's
development, cited by DOI in `README.md`'s roadmap and in relevant
docstrings/BACKEND_PHASE*.md sections. When a design decision or a
documented limitation cites a paper, prefer reading the actual PDF here over
trusting a secondhand summary -- that's the standard this project has tried
to hold itself to throughout (see "Key principles" below).

## Where things live

- `hycline/cli.py` -- the CLI, 15 top-level commands.
- `hycline/models/` -- the cline functions themselves (`cline.py`,
  `genomic_cline.py`).
- `hycline/likelihoods/` -- binomial likelihood, genotype-likelihood-aware
  variants.
- `hycline/inference/` -- classic MLE (`mle.py`, `mle_genotype_likelihood.py`)
  and Bayesian/NUTS (`mcmc_numpyro.py`, `mcmc_hierarchical.py`,
  `mcmc_genomic_hierarchical.py`, `mcmc_pymc.py`).
- `hycline/data/` -- readers/writers, VCF/beagle handling, AIM filtering.
- `hycline/viz/` -- plotting.
- `tests/` -- 455 tests. `test_cli.py` and the MCMC test files are the
  slowest; if the environment has tight per-command timeouts, split large
  test runs into batches rather than assuming a single `pytest tests/` call
  will complete.
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
- **Run the full test suite before considering anything done**, and
  reconcile the batch counts against the total collected count explicitly
  -- don't report "tests passing" from a partial run.
