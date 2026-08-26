"""HyCline command-line interface.

One script, run with flags -- no Python programming required. Designed to
work identically on a laptop or submitted as a cluster job (matplotlib is
forced to a non-interactive backend before anything else imports it, so
this never tries to open a window on a headless node).

Core subcommands (see `hycline <subcommand> --help` for the full flag
list for each):

    hycline filter-aims ...        filter genome-wide VCF/beagle data
                                     down to ancestry-informative markers
    hycline generate-geographic-   convert VCF/beagle + metadata into a
        input ...                   geographic-cline CSV (format auto-
                                     detects hard counts vs. GL)
    hycline generate-genomic-      the genomic-cline analog
        input ...
    hycline estimate-hybrid-       estimate genome-wide hybrid index from
        index ...                   VCF/beagle for genomic-cline fitting
    hycline fit-geographic ...     fit geographic clines, any locus count,
                                     hard counts or genotype likelihoods
                                     (auto-detected)
    hycline compare-tails ...      fit several tail-model variants and
                                     rank them by AICc (HZAR's `autofit`
                                     equivalent)
    hycline fit-genomic ...        fit genomic clines, any locus count,
                                     hard counts or genotype likelihoods
                                     (auto-detected)
    hycline fit-hierarchical-      partially pooled Bayesian fitting
        geographic ...              across geographic-cline loci
    hycline fit-hierarchical-      the genomic-cline analog
        genomic ...

Run `hycline <subcommand> --help` for the full flag list for each. See
README.md's "Command-line usage" section for worked examples, and
BACKEND_PHASE1.md / BACKEND_PHASE2.md / BACKEND_PHASE3.md for what
"classic"/"bayesian" and "geographic"/"genomic" cline actually mean and do
differently under the hood.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must happen before any other module imports pyplot --
                        # cluster nodes typically have no display, and even
                        # on a laptop this script should never try to pop a window

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hycline",
        description="HyCline: geographic cline fitting for hybrid zones "
                     "(a Python/CLI successor to HZAR). Run as "
                     "'hycline <command> --help' for details on a specific command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- shared arguments, added to multiple subcommands ----
    def add_distance_args(g):
        g.add_argument("--distance-col", default="distance",
                        help="Column name for pre-computed geographic distance "
                             "(default: distance). Ignored if --lat-col/--lon-col given.")
        g.add_argument("--lat-col", default=None,
                        help="Column name for latitude (decimal degrees). If given "
                             "together with --lon-col, distance is computed "
                             "automatically from coordinates instead of --distance-col "
                             "-- see --transect-method. This collapses 2D coordinates "
                             "to a 1D transect distance; it does not fit a 2D spatial "
                             "model (see BACKEND_PHASE1.md section 9.1).")
        g.add_argument("--lon-col", default=None,
                        help="Column name for longitude (decimal degrees). See --lat-col.")
        g.add_argument("--transect-method", choices=["reference", "pca"], default="reference",
                        help="How to collapse lat/lon coordinates to a 1D distance. "
                             "'reference' (default): great-circle distance from a single "
                             "reference site (matches HZAR's own convention -- see "
                             "hycline.data.geo.distance_from_reference). 'pca': project "
                             "onto the best-fit line through all sites, better for "
                             "zigzagging sampling with no natural reference endpoint.")
        g.add_argument("--reference-lat", type=float, default=None,
                        help="Latitude of the reference site for --transect-method "
                             "reference (default: the first row's coordinates).")
        g.add_argument("--reference-lon", type=float, default=None,
                        help="Longitude of the reference site for --transect-method "
                             "reference (default: the first row's coordinates).")

    def add_sample_size_args(g):
        g.add_argument("--n-alleles-col", default="n_alleles",
                        help="Column name for total sampled alleles per site "
                             "(default: n_alleles). Ignored if --n-ind-col given.")
        g.add_argument("--n-ind-col", default=None,
                        help="Column name for individual (not allele) sample counts, "
                             "if that's what you have. Combined with --ploidy to get "
                             "total alleles. If given, --n-alleles-col is ignored.")
        g.add_argument("--ploidy", type=int, default=2,
                        help="Ploidy to use with --n-ind-col (default: 2, i.e. diploid "
                             "autosomal). alleles = individuals x ploidy.")

    def add_input_args(p):
        g = p.add_argument_group("input data")
        g.add_argument("--input", required=True, metavar="FILE.csv",
                        help="CSV file with one row per sampling site.")
        add_distance_args(g)
        add_sample_size_args(g)
        g.add_argument("--n-focal-col", default="n_focal",
                        help="Column name for focal-allele counts per site "
                             "(default: n_focal). Mutually exclusive with --freq-col.")
        g.add_argument("--freq-col", default=None,
                        help="Column name for observed frequency (0-1), if you "
                             "have frequencies rather than raw counts. If given, "
                             "--n-focal-col is ignored.")
        g.add_argument("--site-col", default=None,
                        help="Optional column of site labels (for reporting only).")

    def add_model_args(p):
        g = p.add_argument_group("model")
        g.add_argument("--scaling", choices=["estimated", "fixed"], default="estimated",
                        help="'estimated': pmin/pmax are fit as free parameters. "
                             "'fixed': pmin/pmax are held at the observed frequencies "
                             "at the transect's geographic extremes (HZAR's "
                             "scaling='fixed' mode) unless overridden by --pmin/--pmax. "
                             "(default: estimated)")
        g.add_argument("--pmin", type=float, default=None,
                        help="Override the fixed pmin value (implies --scaling fixed).")
        g.add_argument("--pmax", type=float, default=None,
                        help="Override the fixed pmax value (implies --scaling fixed).")
        g.add_argument("--n-edge-sites", type=int, default=1,
                        help="[--scaling fixed only] Number of sites nearest each end "
                             "of the transect to pool (sum(n_focal)/sum(n_alleles)) "
                             "when deriving pmin/pmax, instead of a single extreme "
                             "site (default: 1, matching HZAR's own convention exactly "
                             "-- unchanged from before this flag existed). A single "
                             "site's observed frequency can be fragile with realistic "
                             "sample sizes: an unlucky all-zero (or all-focal) draw at "
                             "the extreme site locks pmin (or pmax) at the wrong value "
                             "and can force the fitted cline substantially wider than "
                             "the data actually supports to compensate. Increase this "
                             "(e.g. to 3, matching the hierarchical model's own "
                             "default) if you suspect this is happening -- see "
                             "BACKEND_PHASE1.md section 4.2 / BACKEND_PHASE4.md "
                             "section 3 for the full story and a worked example.")
        g.add_argument("--hzar-tail", action="store_true",
                        help="[--method classic, tail models only] Use HZAR's own tail "
                             "parameterization (verified directly against HZAR's R "
                             "source) instead of this project's own, differently-"
                             "parameterized tail model. Default: off, preserving "
                             "original behavior exactly. A real, named difference: "
                             "HZAR's tail edge is center +/- delta directly (delta is a "
                             "raw distance), not center +/- width/2 the way the default "
                             "tail model defines it -- delta_l/tau_l/delta_r/tau_r mean "
                             "something different with this flag than without it. See "
                             "BACKEND_TRAIT.md sections 1-2.")

    def add_output_args(p):
        g = p.add_argument_group("output")
        g.add_argument("--output-dir", default="./hycline_output",
                        help="Directory to write results into (created if needed). "
                             "(default: ./hycline_output)")
        g.add_argument("--prefix", default="hycline",
                        help="Filename prefix for all output files (default: hycline)")
        g.add_argument("--no-plot", action="store_true",
                        help="Skip generating the PNG plot (still writes CSV/JSON output).")

    def add_bayes_args(p):
        g = p.add_argument_group("bayesian fitting (only used with --method bayesian)")
        g.add_argument("--backend", choices=["numpyro", "pymc"], default="numpyro",
                        help="MCMC backend (default: numpyro). See BACKEND_PHASE2.md "
                             "section 6 for the difference -- numpyro is faster and "
                             "supports batch multi-locus fitting; pymc doesn't yet "
                             "support --scaling fixed.")
        g.add_argument("--num-warmup", type=int, default=1000,
                        help="MCMC warmup/tuning iterations per chain (default: 1000)")
        g.add_argument("--num-samples", type=int, default=1000,
                        help="MCMC post-warmup samples per chain (default: 1000)")
        g.add_argument("--num-chains", type=int, default=4,
                        help="Number of MCMC chains (default: 4). More chains give a "
                             "more reliable R-hat convergence check; see BACKEND_PHASE2.md "
                             "section 3.")
        g.add_argument("--width-upper-bound", type=float, default=None,
                        help="Upper bound of the (flat) prior on cline width, in the "
                             "same distance units as --distance-col. Default: 10x the "
                             "transect's geographic span, which can be far looser than "
                             "sparse data can constrain -- see BACKEND_PHASE2.md section "
                             "9. If the run reports convergence problems, try setting "
                             "this to roughly the transect's total span or tighter.")
        g.add_argument("--credible-prob", type=float, default=0.95,
                        help="Width of the credible interval / band to report and plot "
                             "(default: 0.95)")
        g.add_argument("--seed", type=int, default=0,
                        help="Random seed, for reproducibility (default: 0)")

    def add_hierarchical_bayes_args(p):
        # Deliberately smaller than add_bayes_args: hierarchical batch
        # fitting (Phase 4 Components A/B) is NumPyro-only (no --backend
        # choice) and uses population-level hyperpriors for width/rate
        # rather than a single flat prior with an upper bound, so
        # --width-upper-bound doesn't apply -- see the "hierarchical
        # model" argument group instead for the actual prior controls.
        g = p.add_argument_group("MCMC (NumPyro/NUTS)")
        g.add_argument("--num-warmup", type=int, default=1500,
                        help="MCMC warmup/tuning iterations per chain (default: 1500)")
        g.add_argument("--num-samples", type=int, default=1500,
                        help="MCMC post-warmup samples per chain (default: 1500)")
        g.add_argument("--num-chains", type=int, default=4,
                        help="Number of MCMC chains (default: 4). More chains give a "
                             "more reliable R-hat convergence check.")
        g.add_argument("--seed", type=int, default=0,
                        help="Random seed, for reproducibility (default: 0)")

    # ---- `fit-geographic` subcommand (merged: hard-count/GL, single/multi locus) ----
    fit_p = subparsers.add_parser(
        "fit-geographic",
        help="Fit geographic clines from a generate-geographic-input CSV (any locus count).",
        description="Fit a geographic cline to every locus in a wide-format CSV "
                     "produced by 'generate-geographic-input' (or matching its column "
                     "convention) -- one locus or thousands, the same command either "
                     "way. Input type is auto-detected from the column suffixes "
                     "present: '.nSamples'/'.A' means site-aggregated hard counts, "
                     "'.gl0'/'.gl1'/'.gl2' means per-individual genotype likelihoods -- "
                     "no flag to choose, matching the same design already used by "
                     "'generate-geographic-input' itself. Writes a per-locus summary "
                     "table (one row even when there's only one locus), a combined "
                     "long-format fitted-curves CSV, and a single overlaid plot (one "
                     "line per locus, not a small-multiples grid -- see "
                     "BACKEND_SCALE.md section 3 for why the grid didn't scale). One "
                     "locus failing to fit does not stop the others. Genotype-"
                     "likelihood input is classic MLE only, with independent per-locus "
                     "fitting -- see 'fit-hierarchical-geographic' for pooled fitting "
                     "across loci, either input type.",
    )
    fit_g = fit_p.add_argument_group("input data")
    fit_p.add_argument("--input", required=True, metavar="FILE.csv",
                        help="CSV file from 'generate-geographic-input' (or matching "
                             "its column convention): one row per site (hard counts) "
                             "or one row per individual (genotype likelihoods).")
    add_distance_args(fit_g)
    fit_g.add_argument("--n-alleles-suffix", default=".nSamples",
                        help="[hard-count input] Column suffix for each locus's total "
                             "sampled alleles, e.g. '.nSamples' for columns like "
                             "'ada.nSamples' (default: .nSamples, matching HZAR's own "
                             "convention and generate-geographic-input's VCF output). "
                             "Ignored if --n-ind-suffix given.")
    fit_g.add_argument("--n-ind-suffix", default=None,
                        help="[hard-count input] Column suffix for each locus's "
                             "individual (not allele) sample counts, if that's what "
                             "you have. Combined with --ploidy. If given, "
                             "--n-alleles-suffix is ignored.")
    fit_g.add_argument("--ploidy", type=int, default=2,
                        help="[hard-count input] Ploidy to use with --n-ind-suffix "
                             "(default: 2).")
    fit_g.add_argument("--freq-suffix", default=".A",
                        help="[hard-count input] Column suffix for each locus's "
                             "observed focal-allele frequency, e.g. '.A' for columns "
                             "like 'ada.A' (default: .A, matching HZAR's own convention "
                             "and generate-geographic-input's VCF output). Mutually "
                             "exclusive with --n-focal-suffix.")
    fit_g.add_argument("--n-focal-suffix", default=None,
                        help="[hard-count input] Column suffix for each locus's "
                             "focal-allele counts, if you have counts rather than "
                             "frequency. If given, --freq-suffix is ignored.")
    fit_g.add_argument("--site-col", default=None,
                        help="[hard-count input] Optional column of site labels (for "
                             "reporting only).")
    fit_g.add_argument("--gl0-suffix", default=".gl0",
                        help="[genotype-likelihood input] Column suffix for each "
                             "locus's natural-log genotype likelihood of 0 focal-"
                             "allele copies (default: .gl0, matching generate-"
                             "geographic-input's beagle output).")
    fit_g.add_argument("--gl1-suffix", default=".gl1",
                        help="[genotype-likelihood input] Same for 1 focal-allele "
                             "copy (default: .gl1).")
    fit_g.add_argument("--gl2-suffix", default=".gl2",
                        help="[genotype-likelihood input] Same for 2 focal-allele "
                             "copies (default: .gl2).")
    fit_g.add_argument("--phred", action="store_true",
                        help="[genotype-likelihood input] Treat the gl0/gl1/gl2 "
                             "columns as Phred-scaled PL values (as GATK/bcftools/"
                             "ANGSD commonly output), converting to natural-log "
                             "likelihoods before fitting. Default: columns already "
                             "hold natural-log values.")
    fit_g.add_argument("--individual-col", default=None,
                        help="[genotype-likelihood input] Optional column of "
                             "individual/sample labels (for reporting only).")
    fit_g.add_argument("--loci", default=None,
                        help="Comma-separated list of locus names to fit (default: "
                             "all loci auto-detected from column names) -- also the "
                             "natural way to fit just one locus out of a larger file.")
    add_model_args(fit_p)
    fit_p.add_argument("--method", choices=["classic", "bayesian"], default="classic",
                        help="[hard-count input only -- genotype-likelihood input is "
                             "classic MLE only, gives a clear error if --method "
                             "bayesian is requested] 'classic': direct maximum-"
                             "likelihood fit (fast, matches numbers comparable to "
                             "published HZAR-based studies). 'bayesian': full "
                             "posterior via NUTS MCMC (slower, gives proper credible "
                             "intervals); automatically uses the faster plate-"
                             "vectorized batch path (one NUTS run for every locus at "
                             "once) whenever the data is dense (every locus shares "
                             "the same sites and --scaling is not fixed) -- see "
                             "BACKEND_PHASE2.md section 13. Falls back automatically "
                             "to one independent MCMC run per locus for ragged data "
                             "or --scaling fixed. (default: classic)")
    fit_p.add_argument("--tail", choices=["none", "mirror", "both"], default="none",
                        help="Cline tail configuration (default: none). Supported for "
                             "both --method classic and --method bayesian on hard-"
                             "count input, but Bayesian tail-model fitting has a "
                             "known, real convergence difficulty (non-smooth log-"
                             "density) -- classic MLE is recommended for tail models. "
                             "See BACKEND_PHASE2.md section 12. Genotype-likelihood "
                             "input uses several randomized restarts internally to "
                             "guard against the same non-smooth surface -- see "
                             "BACKEND_PHASE2.md section 12.3.")
    add_bayes_args(fit_p)
    add_output_args(fit_p)
    fit_p.add_argument("--plot-grid", action="store_true", default=True,
                        help="Generate a single overlaid plot of all loci (default: "
                             "on) -- one fitted-curve line per locus, sharing one set "
                             "of axes, rather than a grid of separate panels.")
    fit_p.add_argument("--no-plot-grid", dest="plot_grid", action="store_false",
                        help="Skip the overlay plot.")

    # ---- `compare-tails` subcommand ----
    cmp_p = subparsers.add_parser(
        "compare-tails", help="Fit several tail-model variants and rank by AICc.",
        description="Fit multiple cline tail-model variants (HZAR's model-selection "
                     "workflow) and rank them by AICc. Currently uses the classic "
                     "MLE fitting procedure only (see BACKEND_PHASE1.md section 7).",
    )
    add_input_args(cmp_p)
    cmp_p.add_argument("--candidates", default="none,mirror,both",
                        help="Comma-separated list of tail models to compare, from "
                             "{none, mirror, both} (default: none,mirror,both)")
    add_output_args(cmp_p)
    cmp_p.add_argument("--plot-best", action="store_true",
                        help="Also generate a plot of the best (lowest-AICc) model.")

    # ---- `fit-trait` subcommand ----
    trait_p = subparsers.add_parser(
        "fit-trait", help="Fit a cline model to a single quantitative trait.",
        description="Fit a sigmoid cline model to quantitative trait data (a "
                     "continuous, measured phenotype -- body size, a color/plumage "
                     "score, a song frequency -- rather than a molecular marker) -- "
                     "HZAR's other core data type, alongside allele frequencies. Full "
                     "HZAR parity: the mean follows the usual sigmoid cline, and the "
                     "variance follows its OWN cline shape (var_l/var_r/var_h) rather "
                     "than a single constant, matching HZAR's own quantitative-trait "
                     "model exactly (verified directly against HZAR's R source, "
                     "including tail models -- see BACKEND_TRAIT.md). Two input "
                     "styles, matching HZAR's own: site-aggregated (default -- one row "
                     "per site, with mean/n/SD columns) or --raw (one row per measured "
                     "individual, aggregated internally to the same shape). Classic "
                     "MLE fitting only in this pass -- no Bayesian backend yet.",
    )
    trait_g = trait_p.add_argument_group("input data")
    trait_p.add_argument("--input", required=True, metavar="FILE.csv",
                          help="CSV file -- one row per sampling site (default), or "
                               "one row per individual (--raw).")
    add_distance_args(trait_g)
    trait_p.add_argument("--raw", action="store_true",
                          help="Treat --input as raw, individual-level data (one row "
                               "per measured individual) instead of pre-aggregated "
                               "site summaries -- aggregated internally via "
                               "hycline.data.io.read_raw_trait_csv. Uses --trait-col/"
                               "--site-col/--ddof instead of --mean-col/--n-col/"
                               "--sd-col. See BACKEND_TRAIT.md section 4.")
    trait_g.add_argument("--mean-col", default="trait_mean",
                          help="[aggregated input] Column name for each site's mean "
                               "trait value (default: trait_mean).")
    trait_g.add_argument("--n-col", default="n",
                          help="[aggregated input] Column name for the number of "
                               "individuals measured at each site (default: n).")
    trait_g.add_argument("--sd-col", default=None,
                          help="[aggregated input] Column name for the SD of "
                               "individual trait measurements at each site. Required "
                               "unless --raw -- HZAR's own model always uses each "
                               "site's observed sample variance as data, not just its "
                               "mean (see BACKEND_TRAIT.md section 2). If you only "
                               "have site means, this model cannot be fit; there is no "
                               "single-shared-variance fallback in this project.")
    trait_g.add_argument("--trait-col", default="trait_value",
                          help="[--raw input only] Column holding each individual's "
                               "own raw trait measurement (default: trait_value).")
    trait_g.add_argument("--ddof", type=int, default=0,
                          help="[--raw input only] Delta degrees of freedom for the "
                               "per-site SD computed during aggregation (default: 0, "
                               "numpy's own population/biased-SD convention). Whether "
                               "HZAR's own raw-data aggregation uses ddof=0 or ddof=1 "
                               "(R's var() default) wasn't independently confirmed -- "
                               "see BACKEND_TRAIT.md section 4. Pass --ddof 1 if you "
                               "specifically need to match a var()-based computation "
                               "elsewhere.")
    trait_g.add_argument("--site-col", default=None,
                          help="Optional column of site labels. For aggregated input, "
                               "purely for reporting; for --raw input, this is how "
                               "individuals are grouped into sites (default: group by "
                               "exact matching distance value instead).")
    trait_p.add_argument("--tail", choices=["none", "left", "right", "both", "mirror"],
                          default="none",
                          help="Cline tail configuration (default: none). Full HZAR "
                               "parity for every option, including the tail formula "
                               "itself -- unlike `fit-geographic`'s own tail model, "
                               "which uses a different, not-yet-HZAR-verified "
                               "parameterization (see BACKEND_TRAIT.md section 1 for "
                               "this real, named difference). Note delta_l/delta_r here "
                               "are raw distances (the tail edge is exactly "
                               "center +/- delta), not the width-relative fraction "
                               "`fit-geographic` uses.")
    add_output_args(trait_p)

    # ---- `fit-trait-multi` subcommand ----
    traitm_p = subparsers.add_parser(
        "fit-trait-multi", help="Fit many traits at once from a wide-format multi-trait CSV.",
        description="Fit a cline model separately to each trait in a wide-format "
                     "multi-trait CSV (one row per site, one <trait>.mean/<trait>.n/"
                     "<trait>.sd column group per trait -- all three required per trait, "
                     "see `fit-trait --help` for why). The multi-trait analog of "
                     "`fit-trait`, mirroring `fit-geographic`'s own wide-format structure. One trait "
                     "failing to fit does not stop the others.",
    )
    traitm_g = traitm_p.add_argument_group("input data")
    traitm_p.add_argument("--input", required=True, metavar="FILE.csv",
                           help="CSV file with one row per sampling site.")
    add_distance_args(traitm_g)
    traitm_g.add_argument("--mean-suffix", default=".mean",
                           help="Column suffix for each trait's site mean, e.g. '.mean' "
                                "for columns like 'mass.mean' (default: .mean).")
    traitm_g.add_argument("--n-suffix", default=".n",
                           help="Column suffix for each trait's per-site sample size "
                                "(default: .n).")
    traitm_g.add_argument("--sd-suffix", default=".sd",
                           help="Column suffix for each trait's per-site SD (default: "
                                ".sd). Required per trait -- a trait missing its .sd "
                                "column is skipped with a clear message (see "
                                "`fit-trait --help` for why this is required, not "
                                "optional).")
    traitm_g.add_argument("--site-col", default=None,
                           help="Optional column of site labels (for reporting only).")
    traitm_g.add_argument("--traits", default=None,
                           help="Comma-separated list of trait names to fit (default: "
                                "all traits auto-detected from column names).")
    traitm_p.add_argument("--tail", choices=["none", "left", "right", "both", "mirror"],
                           default="none",
                           help="Same meaning as in `fit-trait` (default: none).")
    add_output_args(traitm_p)
    traitm_p.add_argument("--plot-grid", action="store_true", default=True,
                           help="Generate a small-multiples plot of all traits (default: on).")
    traitm_p.add_argument("--no-plot-grid", dest="plot_grid", action="store_false",
                           help="Skip the small-multiples plot.")

    # ---- `fit-genomic` subcommand (merged: hard-count/GL, single/multi locus) ----
    gen_p = subparsers.add_parser(
        "fit-genomic",
        help="Fit genomic clines from a generate-genomic-input CSV (any locus count).",
        description="Fit the logit-logistic genomic cline model (Fitzpatrick 2013) to "
                     "every locus in a wide-format CSV produced by "
                     "'generate-genomic-input' (or matching its column convention) -- "
                     "one locus or thousands, the same command either way. Input type "
                     "is auto-detected from the column suffixes present: '.n'/'.freq' "
                     "means hard counts, '.gl0'/'.gl1'/'.gl2' means per-individual "
                     "genotype likelihoods -- no flag to choose, matching "
                     "'generate-genomic-input' itself. Classic MLE only -- no Bayesian "
                     "backend exists yet for independent per-locus genomic-cline "
                     "fitting (see 'fit-hierarchical-genomic' for the one genomic-cline "
                     "path that is Bayesian, via partial pooling). Writes a per-locus "
                     "summary table (one row even when there's only one locus), a "
                     "combined long-format fitted-curves CSV, and a single overlaid "
                     "plot (one line per locus, not a small-multiples grid). One locus "
                     "failing to fit does not stop the others.",
    )
    gen_g = gen_p.add_argument_group("input data")
    gen_p.add_argument("--input", required=True, metavar="FILE.csv",
                        help="CSV file from 'generate-genomic-input' (or matching its "
                             "column convention): one row per individual.")
    gen_g.add_argument("--hybrid-index-col", default="hybrid_index",
                        help="Column name for each individual's genome-wide hybrid "
                             "index, 0-1 (default: hybrid_index). Estimate this first "
                             "if you don't already have it -- see "
                             "'hycline estimate-hybrid-index'.")
    gen_g.add_argument("--n-alleles-suffix", default=".n",
                        help="[hard-count input] Column suffix for each locus's "
                             "per-individual sample size, e.g. '.n' for columns like "
                             "'locus1.n' (default: .n). Set to empty ('') to use "
                             "--ploidy instead of per-locus sample-size columns.")
    gen_g.add_argument("--ploidy", type=int, default=2,
                        help="[hard-count input] Ploidy, used when --n-alleles-suffix "
                             "is empty (default: 2).")
    gen_g.add_argument("--freq-suffix", default=".freq",
                        help="[hard-count input] Column suffix for each locus's "
                             "observed genotype frequency, e.g. '.freq' for columns "
                             "like 'locus1.freq' (default: .freq). Mutually exclusive "
                             "with --n-focal-suffix.")
    gen_g.add_argument("--n-focal-suffix", default=None,
                        help="[hard-count input] Column suffix for each locus's "
                             "focal-allele counts, if you have counts rather than "
                             "frequency. If given, --freq-suffix is ignored.")
    gen_g.add_argument("--gl0-suffix", default=".gl0",
                        help="[genotype-likelihood input] Column suffix for each "
                             "locus's natural-log genotype likelihood of 0 focal-"
                             "allele copies (default: .gl0).")
    gen_g.add_argument("--gl1-suffix", default=".gl1",
                        help="[genotype-likelihood input] Same for 1 focal-allele "
                             "copy (default: .gl1).")
    gen_g.add_argument("--gl2-suffix", default=".gl2",
                        help="[genotype-likelihood input] Same for 2 focal-allele "
                             "copies (default: .gl2).")
    gen_g.add_argument("--phred", action="store_true",
                        help="[genotype-likelihood input] Treat the gl0/gl1/gl2 "
                             "columns as Phred-scaled PL values, converting to "
                             "natural-log likelihoods before fitting.")
    gen_g.add_argument("--individual-col", default=None,
                        help="Optional column of individual/sample labels (for "
                             "reporting only).")
    gen_g.add_argument("--loci", default=None,
                        help="Comma-separated list of locus names to fit (default: "
                             "all loci auto-detected from column names) -- also the "
                             "natural way to fit just one locus out of a larger file.")
    add_output_args(gen_p)
    gen_p.add_argument("--plot-grid", action="store_true", default=True,
                        help="Generate a single overlaid plot of all loci (default: "
                             "on) -- one fitted-curve line per locus, sharing one set "
                             "of axes, rather than a grid of separate panels.")
    gen_p.add_argument("--no-plot-grid", dest="plot_grid", action="store_false",
                        help="Skip the overlay plot.")

    # ---- `fit-hierarchical-geographic` subcommand (merged: hard-count/GL, Phase 4 Components A/D) ----
    hier_p = subparsers.add_parser(
        "fit-hierarchical-geographic",
        help="Fit many geographic-cline loci at once with partial pooling across loci.",
        description="Fit a hierarchical Bayesian geographic cline model: center and "
                     "width are partially pooled across loci via shared population-"
                     "level priors, so sparse/noisy loci borrow statistical strength "
                     "from data-rich ones, rather than each locus being fit fully "
                     "independently (as 'fit-geographic' does). Same wide-format "
                     "multi-locus CSV input as 'fit-geographic', and the same auto-"
                     "detection: '.nSamples'/'.A' column suffixes mean site-aggregated "
                     "hard counts (Phase 4 Component A), '.gl0'/'.gl1'/'.gl2' mean "
                     "per-individual genotype likelihoods (Phase 4 Component D) -- no "
                     "flag to choose. Requires every locus to have data at exactly the "
                     "same sites/individuals (no per-locus missing data) -- see "
                     "BACKEND_PHASE4.md sections 2, 6, and 11.",
    )
    hier_g = hier_p.add_argument_group("input data")
    hier_p.add_argument("--input", required=True, metavar="FILE.csv",
                         help="CSV file from 'generate-geographic-input' (or matching "
                              "its column convention): one row per site (hard counts) "
                              "or one row per individual (genotype likelihoods).")
    add_distance_args(hier_g)
    hier_g.add_argument("--n-alleles-suffix", default=".nSamples",
                         help="[hard-count input] Column suffix for each locus's total "
                              "sampled alleles (default: .nSamples, matching HZAR's own "
                              "convention). Ignored if --n-ind-suffix given.")
    hier_g.add_argument("--n-ind-suffix", default=None,
                         help="[hard-count input] Column suffix for each locus's "
                              "individual (not allele) sample counts. Combined with "
                              "--ploidy. If given, --n-alleles-suffix is ignored.")
    hier_g.add_argument("--ploidy", type=int, default=2,
                         help="[hard-count input] Ploidy to use with --n-ind-suffix "
                              "(default: 2)")
    hier_g.add_argument("--freq-suffix", default=".A",
                         help="[hard-count input] Column suffix for each locus's "
                              "observed focal-allele frequency (default: .A, matching "
                              "HZAR's own convention). Mutually exclusive with "
                              "--n-focal-suffix.")
    hier_g.add_argument("--n-focal-suffix", default=None,
                         help="[hard-count input] Column suffix for each locus's "
                              "focal-allele counts, if you have counts rather than "
                              "frequency.")
    hier_g.add_argument("--site-col", default=None,
                         help="[hard-count input] Optional column of site labels "
                              "(for reporting only).")
    hier_g.add_argument("--gl0-suffix", default=".gl0",
                         help="[genotype-likelihood input] Column suffix for each "
                              "locus's natural-log genotype likelihood of 0 focal-"
                              "allele copies (default: .gl0).")
    hier_g.add_argument("--gl1-suffix", default=".gl1",
                         help="[genotype-likelihood input] Same for 1 focal-allele "
                              "copy (default: .gl1).")
    hier_g.add_argument("--gl2-suffix", default=".gl2",
                         help="[genotype-likelihood input] Same for 2 focal-allele "
                              "copies (default: .gl2).")
    hier_g.add_argument("--phred", action="store_true",
                         help="[genotype-likelihood input] Treat the gl0/gl1/gl2 "
                              "columns as Phred-scaled PL values, converting to "
                              "natural-log likelihoods before fitting.")
    hier_g.add_argument("--individual-col", default=None,
                         help="[genotype-likelihood input] Optional column of "
                              "individual/sample labels (for reporting only).")
    hier_g.add_argument("--loci", default=None,
                         help="Comma-separated list of locus names to fit (default: "
                              "all loci auto-detected from column names).")
    hier_model_g = hier_p.add_argument_group("hierarchical model")
    hier_model_g.add_argument("--n-edge-sites", type=int, default=3,
                               help="[hard-count input] How many sites near each end "
                                    "of the transect to average when deriving each "
                                    "locus's fixed pmin/pmax (default: 3) -- see "
                                    "BACKEND_PHASE4.md section 3.")
    hier_model_g.add_argument("--n-edge-individuals", type=int, default=3,
                               help="[genotype-likelihood input] Same idea as "
                                    "--n-edge-sites, but counting individuals rather "
                                    "than sites (default: 3).")
    hier_model_g.add_argument("--sigma-center-scale", type=float, default=None,
                               help="Scale of the HalfNormal hyperprior on the across-"
                                    "locus SD of cline center. Default: half the "
                                    "transect's geographic span.")
    hier_model_g.add_argument("--mu-log-width-loc", type=float, default=None,
                               help="Mean of the Normal prior on the population-level "
                                    "mean log-width. Default: log(span/4).")
    hier_model_g.add_argument("--mu-log-width-scale", type=float, default=1.5,
                               help="SD of the Normal prior on the population-level "
                                    "mean log-width (default: 1.5).")
    hier_model_g.add_argument("--sigma-log-width-scale", type=float, default=1.0,
                               help="Scale of the HalfNormal hyperprior on the across-"
                                    "locus SD of log-width (default: 1.0).")
    add_hierarchical_bayes_args(hier_p)
    add_output_args(hier_p)
    hier_p.add_argument("--plot-grid", action="store_true", default=True,
                         help="Generate a single overlaid plot of all loci (default: "
                              "on) -- one fitted-curve line per locus, sharing one set "
                              "of axes, rather than a grid of separate panels.")
    hier_p.add_argument("--no-plot-grid", dest="plot_grid", action="store_false",
                         help="Skip the overlay plot.")

    # ---- `fit-hierarchical-genomic` subcommand (merged: hard-count/GL, Phase 4 Components B/E) ----
    hierg_p = subparsers.add_parser(
        "fit-hierarchical-genomic",
        help="Fit many genomic-cline loci at once with partial pooling across loci.",
        description="Fit a hierarchical Bayesian genomic cline model: u (bias) and v "
                     "(rate) are partially pooled across loci via shared population-"
                     "level priors, so sparse/noisy loci borrow statistical strength "
                     "from data-rich ones, rather than each locus being fit fully "
                     "independently (as 'fit-genomic' does). Same wide-format "
                     "multi-locus CSV input as 'fit-genomic', and the same auto-"
                     "detection: '.n'/'.freq' column suffixes mean hard counts (Phase "
                     "4 Component B), '.gl0'/'.gl1'/'.gl2' mean per-individual genotype "
                     "likelihoods (Phase 4 Component E) -- no flag to choose. Requires "
                     "every locus to have data for exactly the same individuals (no "
                     "per-locus missing data) -- see BACKEND_PHASE4.md sections 6 and "
                     "12.",
    )
    hierg_g = hierg_p.add_argument_group("input data")
    hierg_p.add_argument("--input", required=True, metavar="FILE.csv",
                          help="CSV file from 'generate-genomic-input' (or matching "
                               "its column convention): one row per individual.")
    hierg_g.add_argument("--hybrid-index-col", default="hybrid_index",
                          help="Column name for each individual's genome-wide hybrid "
                               "index (default: hybrid_index).")
    hierg_g.add_argument("--n-alleles-suffix", default=".n",
                          help="[hard-count input] Column suffix for each locus's "
                               "per-individual sample size (default: .n). Pass '' to "
                               "use --ploidy instead.")
    hierg_g.add_argument("--ploidy", type=int, default=2,
                          help="[hard-count input] Ploidy, used when "
                               "--n-alleles-suffix is empty (default: 2).")
    hierg_g.add_argument("--freq-suffix", default=".freq",
                          help="[hard-count input] Column suffix for each locus's "
                               "observed genotype frequency (default: .freq). Mutually "
                               "exclusive with --n-focal-suffix.")
    hierg_g.add_argument("--n-focal-suffix", default=None,
                          help="[hard-count input] Column suffix for each locus's "
                               "focal-allele counts, if you have counts rather than "
                               "frequency.")
    hierg_g.add_argument("--gl0-suffix", default=".gl0",
                          help="[genotype-likelihood input] Column suffix for each "
                               "locus's natural-log genotype likelihood of 0 focal-"
                               "allele copies (default: .gl0).")
    hierg_g.add_argument("--gl1-suffix", default=".gl1",
                          help="[genotype-likelihood input] Same for 1 focal-allele "
                               "copy (default: .gl1).")
    hierg_g.add_argument("--gl2-suffix", default=".gl2",
                          help="[genotype-likelihood input] Same for 2 focal-allele "
                               "copies (default: .gl2).")
    hierg_g.add_argument("--phred", action="store_true",
                          help="[genotype-likelihood input] Treat the gl0/gl1/gl2 "
                               "columns as Phred-scaled PL values, converting to "
                               "natural-log likelihoods before fitting.")
    hierg_g.add_argument("--individual-col", default=None,
                          help="Optional column of individual/sample labels (for "
                               "reporting only).")
    hierg_g.add_argument("--loci", default=None,
                          help="Comma-separated list of locus names to fit (default: "
                               "all loci auto-detected from column names).")
    hierg_model_g = hierg_p.add_argument_group("hierarchical model")
    hierg_model_g.add_argument("--mu-u-scale", type=float, default=2.0,
                                help="SD of the Normal(0, mu_u_scale) prior on the "
                                     "population-level mean bias -- centered at the "
                                     "neutral u=0 by construction (default: 2.0).")
    hierg_model_g.add_argument("--sigma-u-scale", type=float, default=2.0,
                                help="Scale of the HalfNormal hyperprior on the "
                                     "across-locus SD of u (default: 2.0).")
    hierg_model_g.add_argument("--mu-log-v-loc", type=float, default=0.0,
                                help="Mean of the Normal prior on the population-level "
                                     "mean log-rate -- centered at the neutral "
                                     "log_v=0 (v=1) by default.")
    hierg_model_g.add_argument("--mu-log-v-scale", type=float, default=1.0,
                                help="SD of the Normal prior on the population-level "
                                     "mean log-rate (default: 1.0).")
    hierg_model_g.add_argument("--sigma-log-v-scale", type=float, default=1.0,
                                help="Scale of the HalfNormal hyperprior on the "
                                     "across-locus SD of log-rate (default: 1.0) -- the "
                                     "parameter most directly answering 'how much does "
                                     "introgression rate vary across the genome.'")
    hierg_model_g.add_argument("--pool-center", action="store_true",
                                help="Hierarchically pool logit(center) = u/v instead of "
                                     "the raw bias parameter u (default: pool u). "
                                     "logit(center) is the quantity Firneno et al. "
                                     "(2023)'s own coupling-coefficient methodology uses "
                                     "-- with this flag, 'sigma_center' in the output "
                                     "population summary can be fed directly into "
                                     "'hycline coupling-genomic' as sigma_c, instead of "
                                     "that command falling back to computing it from "
                                     "per-locus point estimates. u/v themselves are still "
                                     "reported either way -- see BACKEND_PHASE4.md "
                                     "section 9.3.")
    hierg_model_g.add_argument("--mu-center-scale", type=float, default=2.0,
                                help="[--pool-center only] SD of the Normal(0, "
                                     "mu_center_scale) prior on the population-level mean "
                                     "logit(center) -- centered at the neutral center=0.5 "
                                     "by construction (default: 2.0).")
    hierg_model_g.add_argument("--sigma-center-scale", type=float, default=2.0,
                                help="[--pool-center only] Scale of the HalfNormal "
                                     "hyperprior on the across-locus SD of logit(center) "
                                     "(default: 2.0).")
    add_hierarchical_bayes_args(hierg_p)
    add_output_args(hierg_p)
    hierg_p.add_argument("--plot-grid", action="store_true", default=True,
                          help="Generate a single overlaid plot of all loci (default: "
                               "on) -- one fitted-curve line per locus, sharing one "
                               "set of axes, rather than a grid of separate panels.")
    hierg_p.add_argument("--no-plot-grid", dest="plot_grid", action="store_false",
                          help="Skip the overlay plot.")

    # ---- `coupling-geographic` subcommand (Phase 4 Component C) ----
    covg_p = subparsers.add_parser(
        "coupling-geographic",
        help="Coupling/concordance analysis across loci from a geographic-cline "
             "locus summary.",
        description="Compute the across-locus variance in geographic cline slope "
                     "(Firneno et al. 2023) from a per-locus summary CSV -- the exact "
                     "kind of file `fit-geographic`/`fit-hierarchical-geographic` already produce. "
                     "Lower variance across loci means more concordant clines, the "
                     "empirical proxy for Barton's (1983) coupling coefficient. No "
                     "theta estimate for geographic clines (Firneno et al. calibrated "
                     "theta against real data using genomic clines only) -- use "
                     "`coupling-genomic` for that. See BACKEND_PHASE4.md section 9.",
    )
    covg_p.add_argument("--input", required=True, metavar="FILE.csv",
                         help="Per-locus summary CSV (e.g. "
                              "'<prefix>_summary.csv' from `fit-geographic`, or "
                              "'<prefix>_hierarchical_locus_summary.csv' from "
                              "`fit-hierarchical-geographic` -- hierarchical estimates are "
                              "recommended, see BACKEND_PHASE4.md section 9's module "
                              "docstring for why.")
    covg_p.add_argument("--locus-col", default="locus",
                         help="Column with locus names (default: locus).")
    covg_p.add_argument("--width-col", default=None,
                         help="Column with each locus's fitted cline width. Default: "
                              "auto-detect 'width_mean' (fit-hierarchical-geographic's convention) "
                              "then 'width' (fit-geographic's convention).")
    covg_out_g = covg_p.add_argument_group("output")
    covg_out_g.add_argument("--output-dir", default="./hycline_output",
                             help="Directory to write results into (created if needed). "
                                  "(default: ./hycline_output)")
    covg_out_g.add_argument("--prefix", default="hycline",
                             help="Filename prefix for all output files (default: hycline)")

    # ---- `coupling-genomic` subcommand (Phase 4 Component C) ----
    covgn_p = subparsers.add_parser(
        "coupling-genomic",
        help="Coupling/concordance analysis across loci from a genomic-cline "
             "locus summary.",
        description="Compute the across-locus variance in genomic cline parameters "
                     "(logit(u/v), log(v) -- Firneno et al. 2023's exact definitions) "
                     "from a per-locus summary CSV -- the exact kind of file "
                     "`fit-genomic`/`fit-hierarchical-genomic` already produce, "
                     "and an estimated coupling coefficient (theta), calibrated by "
                     "reverse-engineering Firneno et al.'s own published results. See "
                     "BACKEND_PHASE4.md section 9 for the full methodology and its "
                     "confidence level, including a real limitation: theta can come "
                     "back negative (theoretically impossible) for some inputs, which "
                     "this command flags explicitly rather than hiding.",
    )
    covgn_p.add_argument("--input", required=True, metavar="FILE.csv",
                          help="Per-locus summary CSV (e.g. "
                               "'<prefix>_summary.csv' from "
                               "`fit-genomic`, or "
                               "'<prefix>_hierarchical_locus_summary.csv' from "
                               "`fit-hierarchical-genomic` -- hierarchical estimates "
                               "are recommended.")
    covgn_p.add_argument("--locus-col", default="locus",
                          help="Column with locus names (default: locus).")
    covgn_p.add_argument("--u-col", default=None,
                          help="Column with each locus's fitted u. Default: "
                               "auto-detect 'u_mean' (fit-hierarchical-genomic's "
                               "convention) then 'u' (fit-genomic's convention).")
    covgn_p.add_argument("--v-col", default=None,
                          help="Column with each locus's fitted v. Default: "
                               "auto-detect 'v_mean' then 'v', same as --u-col.")
    covgn_p.add_argument("--no-theta", action="store_true",
                          help="Skip the theta estimate, reporting only the raw "
                               "sigma_c/sigma_v variance metrics.")
    covgn_out_g = covgn_p.add_argument_group("output")
    covgn_out_g.add_argument("--output-dir", default="./hycline_output",
                              help="Directory to write results into (created if needed). "
                                   "(default: ./hycline_output)")
    covgn_out_g.add_argument("--prefix", default="hycline",
                              help="Filename prefix for all output files (default: hycline)")

    def add_gl_columns_args(g, single_locus=True):
        g.add_argument("--gl0-col" if single_locus else "--gl0-suffix",
                        default="log_gl0" if single_locus else ".gl0",
                        help="Column" + ("" if single_locus else " suffix") +
                             " for natural-log genotype likelihood of 0 focal-allele "
                             f"copies (default: {'log_gl0' if single_locus else '.gl0'}).")
        g.add_argument("--gl1-col" if single_locus else "--gl1-suffix",
                        default="log_gl1" if single_locus else ".gl1",
                        help="Same for 1 focal-allele copy "
                             f"(default: {'log_gl1' if single_locus else '.gl1'}).")
        g.add_argument("--gl2-col" if single_locus else "--gl2-suffix",
                        default="log_gl2" if single_locus else ".gl2",
                        help="Same for 2 focal-allele copies "
                             f"(default: {'log_gl2' if single_locus else '.gl2'}).")
        g.add_argument("--phred", action="store_true",
                        help="Treat the GL columns as Phred-scaled PL values (as "
                             "GATK/bcftools/ANGSD commonly output), converting to "
                             "natural-log likelihoods before fitting. Default: "
                             "columns already hold natural-log values.")

    # ---- `estimate-hybrid-index` subcommand ----
    ehi_p = subparsers.add_parser(
        "estimate-hybrid-index",
        help="Estimate genome-wide hybrid index from a VCF or ANGSD beagle file.",
        description="Estimate each individual's genome-wide hybrid index (the x-axis "
                     "variable genomic cline fitting needs) as the mean, across loci, of "
                     "that individual's parental-population-1 allele frequency -- a "
                     "standard, defensible baseline estimator (not the more sophisticated "
                     "Bayesian estimators in gghybrid/bgchm, which also propagate "
                     "uncertainty in h itself -- see BACKEND_PHASE3.md section 3 for the "
                     "honest scope note). VCF input uses each individual's hard GT dosage "
                     "directly; beagle input uses each individual's expected dosage under "
                     "their own genotype likelihoods (posterior mean, not an argmax point "
                     "estimate) -- no proper marginalized hybrid-index estimator exists "
                     "yet (that would need its own model, unlike allele-frequency "
                     "estimation, which cline fitting itself marginalizes over), so this "
                     "is the best available option for genotype-likelihood data, not a "
                     "claim of full rigor. Format is auto-detected from the --input file "
                     "extension unless --format is given. Two ways to supply parental "
                     "allele frequencies: (1) --metadata/--population-col/--pop0-label/"
                     "--pop1-label, computing them from reference individuals labeled in "
                     "a metadata file (the default, and the more common case), or (2) "
                     "--parental-freq-csv, a separate CSV of precomputed per-locus "
                     "frequencies, for when you already have a reference panel's "
                     "frequencies summarized rather than raw reference genotypes. Exactly "
                     "one of the two must be given. Output is a simple (individual, "
                     "hybrid_index) CSV that merges directly into the metadata any "
                     "genomic-cline command needs.",
    )
    ehi_g = ehi_p.add_argument_group("input data")
    ehi_p.add_argument("--input", required=True, metavar="FILE",
                        help="VCF file (.vcf or .vcf.gz) or ANGSD beagle-format file "
                             "(.beagle or .beagle.gz). Should include the parental "
                             "reference individuals too if using --population-col (mode "
                             "1) -- hybrid index is computed for every individual "
                             "present, including references, as a sanity check (they "
                             "should land near 0 or 1). Format is auto-detected from the "
                             "file extension unless --format is given.")
    ehi_p.add_argument("--format", choices=["vcf", "beagle"], default=None,
                        help="Force the input format instead of auto-detecting it from "
                             "the --input file extension.")
    ehi_g.add_argument("--metadata", default=None, metavar="FILE.csv",
                        help="[Mode 1] Metadata CSV with one row per individual: a "
                             "sample-name column and a population-label column. Required "
                             "for Mode 1 (--population-col); not used for Mode 2.")
    ehi_g.add_argument("--sample-col", default="individual",
                        help="Column in --metadata holding sample/individual names "
                             "(default: individual). Must match the VCF's own #CHROM "
                             "header row, or ANGSD's own beagle header row.")
    ehi_g.add_argument("--population-col", default=None,
                        help="[Mode 1] Column in --metadata identifying which "
                             "population each individual belongs to -- used to find the "
                             "parental reference individuals for each of the two "
                             "parental populations. Mutually exclusive with "
                             "--parental-freq-csv.")
    ehi_g.add_argument("--pop0-label", default=None,
                        help="[Mode 1] Value in --population-col identifying parental "
                             "population 0's reference individuals. Required if "
                             "--population-col is given.")
    ehi_g.add_argument("--pop1-label", default=None,
                        help="[Mode 1] Value in --population-col identifying parental "
                             "population 1's reference individuals. Required if "
                             "--population-col is given.")
    ehi_g.add_argument("--parental-freq-csv", default=None, metavar="FILE.csv",
                        help="[Mode 2] A separate CSV of precomputed per-locus parental "
                             "allele frequencies, as an alternative to --population-col "
                             "-- one row per locus, with columns identifying the locus "
                             "name and each parental population's allele frequency (see "
                             "--freq-locus-col/--freq-pop0-col/--freq-pop1-col to match "
                             "your own column names). Every locus in --input must have a "
                             "matching row here. Mutually exclusive with --population-col.")
    ehi_g.add_argument("--freq-locus-col", default="locus",
                        help="[Mode 2] Column in --parental-freq-csv holding locus names "
                             "(default: locus) -- must match your --input file's locus "
                             "names exactly.")
    ehi_g.add_argument("--freq-pop0-col", default="freq_pop0",
                        help="[Mode 2] Column in --parental-freq-csv holding parental "
                             "population 0's allele frequency at each locus (default: "
                             "freq_pop0).")
    ehi_g.add_argument("--freq-pop1-col", default="freq_pop1",
                        help="[Mode 2] Column in --parental-freq-csv holding parental "
                             "population 1's allele frequency at each locus (default: "
                             "freq_pop1).")
    ehi_g.add_argument("--ploidy", type=int, default=2,
                        help="Ploidy of the genotype calls (default: 2, diploid). Used "
                             "to normalize dosage to a frequency for both input formats.")
    ehi_g.add_argument("--loci", default=None,
                        help="Comma-separated list of locus names to use (default: "
                             "every locus in the file).")
    ehi_g.add_argument("--max-variants", type=int, default=None,
                        help="[VCF input only] Optional cap on the number of biallelic "
                             "SNPs to parse. Ignored for beagle input.")
    ehi_g.add_argument("--weight-by-differentiation", action="store_true",
                        help="Weight each locus's contribution by how differentiated "
                             "it is between the two parental populations (a locus with "
                             "similar frequencies in both parental populations carries "
                             "little ancestry information and is down-weighted, rather "
                             "than treated as equally informative as a fixed-difference "
                             "locus). Default: every locus weighted equally -- "
                             "appropriate if you've already pre-filtered to strongly "
                             "differentiated (ancestry-informative) markers.")
    ehi_p.add_argument("--output", required=True, metavar="FILE.csv",
                        help="Path to write the (individual, hybrid_index) CSV to.")

    # ---- `generate-geographic-input` subcommand ----
    ggi_p = subparsers.add_parser(
        "generate-geographic-input",
        help="Convert a VCF or ANGSD beagle file + metadata into a HyCline geographic-cline CSV.",
        description="Convert a VCF (biallelic SNPs; a pure-Python parser, no pysam/"
                     "cyvcf2 dependency -- see BACKEND_PHASE1.md section 9.4 for exactly "
                     "what VCF features are and aren't supported) or an ANGSD beagle-"
                     "format file (`angsd -doGlf 2`) plus a metadata CSV mapping sample "
                     "names to geographic distance (or coordinates -- see --lat-col "
                     "below) into a HyCline-ready CSV. Input format determines the "
                     "output path, with no flag to choose otherwise: VCF input always "
                     "produces site-aggregated hard-count output from GT calls, ready "
                     "for 'fit-geographic'/'fit-hierarchical-geographic'; beagle "
                     "input always produces per-individual genotype-likelihood output, "
                     "also ready for 'fit-geographic'/'fit-hierarchical-geographic' -- "
                     "both auto-detect which kind of input they're given. "
                     "Want genotype-likelihood-aware fitting? Generate genotype "
                     "likelihoods with ANGSD (`-doGlf 2`) and provide that beagle file "
                     "here -- a VCF's own GT field is a hard call, not a genotype "
                     "likelihood, regardless of what variant caller produced it. Format "
                     "is auto-detected from the file extension unless --format is "
                     "given. Output uses each downstream reader's own default "
                     "column-suffix convention, so it works with zero extra flags. Have "
                     "genome-wide data rather than an already-curated locus set? Run "
                     "'hycline filter-aims' on it first.",
    )
    ggi_p.add_argument("--input", required=True, metavar="FILE",
                        help="VCF file (.vcf or .vcf.gz) or ANGSD beagle-format file "
                             "(.beagle or .beagle.gz), optionally gzip-compressed. "
                             "Format is auto-detected from the file extension unless "
                             "--format is given.")
    ggi_p.add_argument("--format", choices=["vcf", "beagle"], default=None,
                        help="Force the input format instead of auto-detecting it "
                             "from the --input file extension.")
    ggi_p.add_argument("--metadata", required=True, metavar="FILE.csv",
                        help="Metadata CSV with one row per individual. --sample-col "
                             "must match the sample labels the VCF's own #CHROM header "
                             "row, or ANGSD's own beagle header row, actually uses.")
    ggi_p.add_argument("--sample-col", default="individual",
                        help="Column in the metadata CSV holding sample names "
                             "(default: individual).")
    ggi_p.add_argument("--distance-col", default="distance",
                        help="Column in the metadata CSV holding geographic distance "
                             "(default: distance). Ignored if --lat-col/--lon-col given.")
    ggi_p.add_argument("--lat-col", default=None,
                        help="Column in the metadata CSV holding latitude (decimal "
                             "degrees). If given together with --lon-col, distance is "
                             "computed automatically from coordinates instead of "
                             "--distance-col -- see --transect-method. For VCF input, "
                             "every individual at a given site must have the same "
                             "(lat, lon) -- checked directly, before any distance is "
                             "computed.")
    ggi_p.add_argument("--lon-col", default=None,
                        help="Column in the metadata CSV holding longitude (decimal "
                             "degrees). See --lat-col.")
    ggi_p.add_argument("--transect-method", choices=["reference", "pca"], default="reference",
                        help="How to collapse lat/lon coordinates to a 1D distance "
                             "(km). 'reference' (default): great-circle distance from a "
                             "single reference site, matching HZAR's own convention. "
                             "'pca': project onto the best-fit line through all sites, "
                             "better for zigzagging sampling with no natural reference "
                             "endpoint.")
    ggi_p.add_argument("--reference-lat", type=float, default=None,
                        help="Latitude of the reference site for --transect-method "
                             "reference (default: the first row's coordinates).")
    ggi_p.add_argument("--reference-lon", type=float, default=None,
                        help="Longitude of the reference site for --transect-method "
                             "reference (default: the first row's coordinates).")
    ggi_p.add_argument("--site-col", default="site",
                        help="[VCF input only] Column in the metadata CSV identifying "
                             "each individual's sampling site (default: site) -- "
                             "individuals sharing a site are aggregated into one row. "
                             "Not used for beagle input, which always stays one row "
                             "per individual.")
    ggi_p.add_argument("--ploidy", type=int, default=2,
                        help="[VCF input only] Ploidy, used when aggregating "
                             "individuals into site-level allele counts (default: 2).")
    ggi_p.add_argument("--max-variants", type=int, default=None,
                        help="[VCF input only] Optional cap on the number of biallelic "
                             "SNPs to parse (a safety valve for a quick look at a large "
                             "VCF). Ignored for beagle input.")
    ggi_p.add_argument("--output", required=True, metavar="FILE.csv",
                        help="Path to write the converted CSV to.")

    # ---- `generate-genomic-input` subcommand ----
    ggni_p = subparsers.add_parser(
        "generate-genomic-input",
        help="Convert a VCF or ANGSD beagle file + metadata into a HyCline genomic-cline CSV.",
        description="Convert a VCF or an ANGSD beagle-format file plus a metadata CSV "
                     "mapping sample names to hybrid index into a HyCline-ready CSV. "
                     "Input format determines the output path, with no flag to choose "
                     "otherwise: VCF input always produces hard-count output from GT "
                     "calls, ready for 'fit-genomic'/'fit-hierarchical-genomic'; "
                     "beagle input always produces per-individual genotype-likelihood "
                     "output, also ready for 'fit-genomic'/'fit-hierarchical-genomic' "
                     "-- both auto-detect which kind of input they're given. "
                     "Want genotype-likelihood-aware fitting? Generate genotype "
                     "likelihoods with ANGSD (-doGlf 2) and provide that beagle file "
                     "here -- a VCF's own GT field is a hard call, not a genotype "
                     "likelihood, regardless of what variant caller produced it. Format "
                     "is auto-detected from the file extension unless --format is given. "
                     "Don't have hybrid index yet? Run 'hycline estimate-hybrid-index' "
                     "first and merge its output into your metadata by individual name.",
    )
    ggni_p.add_argument("--input", required=True, metavar="FILE",
                         help="VCF file (.vcf or .vcf.gz) or ANGSD beagle-format file "
                              "(.beagle or .beagle.gz), optionally gzip-compressed. "
                              "Format is auto-detected from the file extension unless "
                              "--format is given.")
    ggni_p.add_argument("--format", choices=["vcf", "beagle"], default=None,
                         help="Force the input format instead of auto-detecting it "
                              "from the --input file extension.")
    ggni_p.add_argument("--metadata", required=True, metavar="FILE.csv",
                         help="Metadata CSV with one row per individual. --sample-col "
                              "must match the sample labels the VCF's own #CHROM header "
                              "row, or ANGSD's own beagle header row, actually uses.")
    ggni_p.add_argument("--sample-col", default="individual",
                         help="Column in the metadata CSV holding sample names "
                              "(default: individual).")
    ggni_p.add_argument("--hybrid-index-col", default="hybrid_index",
                         help="Column in the metadata CSV holding each individual's "
                              "genome-wide hybrid index (default: hybrid_index).")
    ggni_p.add_argument("--ploidy", type=int, default=2,
                         help="[VCF input only] Ploidy, used when writing hard-count "
                              "output (default: 2).")
    ggni_p.add_argument("--max-variants", type=int, default=None,
                         help="[VCF input only] Optional cap on the number of biallelic "
                              "SNPs to parse. Ignored for beagle input.")
    ggni_p.add_argument("--output", required=True, metavar="FILE.csv",
                         help="Path to write the converted CSV to.")

    # ---- `filter-aims` subcommand ----
    faims_p = subparsers.add_parser(
        "filter-aims",
        help="Filter whole-genome SNPs to ancestry-informative markers (AIMs).",
        description="Filter a VCF or ANGSD beagle-format file (genome-wide SNP data, "
                     "before any HyCline-specific conversion) down to ancestry-"
                     "informative markers: loci where one parental population's allele "
                     "frequency is at or above --high-threshold (default 0.9) and the "
                     "other's is at or below --low-threshold (default 0.1), in either "
                     "direction. Most genome-wide SNPs aren't informative for cline "
                     "analysis -- this is a standard pre-filtering step before "
                     "'generate-geographic-input'/'generate-genomic-input'. Output is "
                     "the same format "
                     "as the input (a filtered VCF from VCF input, a filtered beagle "
                     "file from beagle input) -- for VCF, every retained line is copied "
                     "byte-for-byte from the input (not re-derived from parsed fields), "
                     "so INFO annotations, QUAL/FILTER values, and any FORMAT field "
                     "beyond GT survive filtering exactly as written; only whole lines "
                     "(variants) are removed.",
    )
    faims_p.add_argument("--input", required=True, metavar="FILE",
                          help="VCF file (.vcf or .vcf.gz) or ANGSD beagle-format file "
                               "(.beagle or .beagle.gz). Format is auto-detected from "
                               "the file extension unless --format is given.")
    faims_p.add_argument("--format", choices=["vcf", "beagle"], default=None,
                          help="Force the input format instead of auto-detecting it "
                               "from the --input file extension.")
    faims_p.add_argument("--metadata", required=True, metavar="FILE.csv",
                          help="A separate CSV with one row per individual: a sample-"
                               "name column and a population-label column (values "
                               "--pop1-label/--pop2-label for the two parental "
                               "reference populations -- individuals with any other "
                               "label, or no entry at all, are simply not used for the "
                               "frequency calculation, not an error, so hybrid-zone "
                               "individuals can safely be present in the same input "
                               "file as the parental references).")
    faims_p.add_argument("--sample-col", default="individual",
                          help="Column in --metadata holding sample/individual names "
                               "(default: individual). Must match the sample labels "
                               "the VCF's own #CHROM header row, or ANGSD's own beagle "
                               "header row, actually uses.")
    faims_p.add_argument("--population-col", default="population",
                          help="Column in --metadata holding each individual's "
                               "population label (default: population).")
    faims_p.add_argument("--pop1-label", default="parent1",
                          help="Value in --population-col identifying parental "
                               "population 1's reference individuals (default: parent1).")
    faims_p.add_argument("--pop2-label", default="parent2",
                          help="Value in --population-col identifying parental "
                               "population 2's reference individuals (default: parent2).")
    faims_p.add_argument("--low-threshold", type=float, default=0.1,
                          help="A locus is retained if one population's frequency is "
                               "at or below this value and the other's is at or above "
                               "--high-threshold (default: 0.1).")
    faims_p.add_argument("--high-threshold", type=float, default=0.9,
                          help="See --low-threshold (default: 0.9).")
    faims_p.add_argument("--output", required=True, metavar="FILE",
                          help="Path to write the filtered file to, in the same format "
                               "as --input.")

    return parser


def _load_transect(args):
    from hycline.data.io import read_allele_freq_csv, TransectCSVError

    try:
        return read_allele_freq_csv(
            args.input,
            distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
            lat_col=args.lat_col, lon_col=args.lon_col,
            transect_method=args.transect_method,
            reference_lat=args.reference_lat, reference_lon=args.reference_lon,
            n_alleles_col=(None if args.n_ind_col else args.n_alleles_col),
            n_ind_col=args.n_ind_col, ploidy=args.ploidy,
            n_focal_col=(None if args.freq_col else args.n_focal_col),
            freq_col=args.freq_col, site_col=args.site_col,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _detect_geographic_input_type(args):
    """Sniff a generate-geographic-input CSV's header to determine
    whether it holds site-aggregated hard counts (columns ending in
    --n-alleles-suffix, default '.nSamples') or per-individual genotype
    likelihoods (columns ending in --gl0-suffix, default '.gl0') -- the
    same auto-detection philosophy already used by
    generate-geographic-input/filter-aims for VCF vs. beagle input.
    Respects any custom --n-alleles-suffix/--gl0-suffix the caller gave
    (not hardcoded literals), so this still works for a hand-renamed
    wide-format file. Returns 'hard_count' or 'gl'; exits with a clear
    error if the file matches both conventions or neither."""
    import pandas as pd
    try:
        header_cols = list(pd.read_csv(args.input, nrows=0).columns)
    except Exception as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)

    n_alleles_suffix = args.n_ind_suffix if args.n_ind_suffix else args.n_alleles_suffix
    has_hard_count = any(c.endswith(n_alleles_suffix) for c in header_cols)
    has_gl = any(c.endswith(args.gl0_suffix) for c in header_cols)

    if has_hard_count and has_gl:
        print(f"\nERROR: '{args.input}' has columns matching both the hard-count "
              f"convention (ending in '{n_alleles_suffix}') and the genotype-"
              f"likelihood convention (ending in '{args.gl0_suffix}') -- can't "
              f"auto-detect which this file is. This shouldn't happen with a file "
              f"from 'generate-geographic-input' -- check whether it was hand-"
              f"edited or combined from two different sources.\n", file=sys.stderr)
        sys.exit(2)
    if has_gl:
        return "gl"
    if has_hard_count:
        return "hard_count"
    print(f"\nERROR: could not detect the input type of '{args.input}' -- no "
          f"columns ending in '{n_alleles_suffix}' (hard counts) or "
          f"'{args.gl0_suffix}' (genotype likelihoods) found. Was this file "
          f"produced by 'generate-geographic-input'? If your columns use "
          f"different suffixes, pass them explicitly (--n-alleles-suffix/"
          f"--gl0-suffix).\n", file=sys.stderr)
    sys.exit(2)


def _load_multi_locus_transects(args):
    from hycline.data.io import read_multi_locus_wide_csv, TransectCSVError

    loci = [l.strip() for l in args.loci.split(",")] if args.loci else None
    try:
        return read_multi_locus_wide_csv(
            args.input,
            distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
            lat_col=args.lat_col, lon_col=args.lon_col,
            transect_method=args.transect_method,
            reference_lat=args.reference_lat, reference_lon=args.reference_lon,
            n_alleles_suffix=args.n_alleles_suffix,
            n_ind_suffix=args.n_ind_suffix, ploidy=args.ploidy,
            freq_suffix=(None if args.n_focal_suffix else args.freq_suffix),
            n_focal_suffix=args.n_focal_suffix,
            site_col=args.site_col, loci=loci,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _load_genomic_data(args):
    from hycline.data.io import read_genomic_cline_csv, TransectCSVError

    try:
        return read_genomic_cline_csv(
            args.input, hybrid_index_col=args.hybrid_index_col,
            n_alleles_col=args.n_alleles_col, ploidy=args.ploidy,
            n_focal_col=(None if args.freq_col else args.n_focal_col),
            freq_col=args.freq_col, individual_col=args.individual_col,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _detect_genomic_input_type(args):
    """Sniff a generate-genomic-input CSV's header to determine whether
    it holds hard counts (columns ending in --n-alleles-suffix, default
    '.n', or --freq-suffix, default '.freq') or per-individual genotype
    likelihoods (columns ending in --gl0-suffix, default '.gl0') -- same
    auto-detection philosophy as _detect_geographic_input_type. Returns
    'hard_count' or 'gl'; exits with a clear error if the file matches
    both conventions or neither."""
    import pandas as pd
    try:
        header_cols = list(pd.read_csv(args.input, nrows=0).columns)
    except Exception as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)

    focal_suffix = args.n_focal_suffix if args.n_focal_suffix else args.freq_suffix
    has_hard_count = any(c.endswith(focal_suffix) for c in header_cols)
    has_gl = any(c.endswith(args.gl0_suffix) for c in header_cols)

    if has_hard_count and has_gl:
        print(f"\nERROR: '{args.input}' has columns matching both the hard-count "
              f"convention (ending in '{focal_suffix}') and the genotype-likelihood "
              f"convention (ending in '{args.gl0_suffix}') -- can't auto-detect which "
              f"this file is. This shouldn't happen with a file from "
              f"'generate-genomic-input' -- check whether it was hand-edited or "
              f"combined from two different sources.\n", file=sys.stderr)
        sys.exit(2)
    if has_gl:
        return "gl"
    if has_hard_count:
        return "hard_count"
    print(f"\nERROR: could not detect the input type of '{args.input}' -- no columns "
          f"ending in '{focal_suffix}' (hard counts) or '{args.gl0_suffix}' "
          f"(genotype likelihoods) found. Was this file produced by "
          f"'generate-genomic-input'? If your columns use different suffixes, pass "
          f"them explicitly (--freq-suffix/--n-focal-suffix/--gl0-suffix).\n",
          file=sys.stderr)
    sys.exit(2)


def _load_genomic_multi_locus_data(args):
    from hycline.data.io import read_genomic_cline_multi_locus_wide_csv, TransectCSVError

    loci = [l.strip() for l in args.loci.split(",")] if args.loci else None
    n_alleles_suffix = args.n_alleles_suffix if args.n_alleles_suffix else None
    try:
        return read_genomic_cline_multi_locus_wide_csv(
            args.input, hybrid_index_col=args.hybrid_index_col,
            n_alleles_suffix=n_alleles_suffix, ploidy=args.ploidy,
            freq_suffix=(None if args.n_focal_suffix else args.freq_suffix),
            n_focal_suffix=args.n_focal_suffix,
            individual_col=args.individual_col, loci=loci,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _load_gl_multi_locus_transects(args):
    from hycline.data.io import read_gl_multi_locus_wide_csv, TransectCSVError

    loci = [l.strip() for l in args.loci.split(",")] if args.loci else None
    try:
        return read_gl_multi_locus_wide_csv(
            args.input,
            distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
            lat_col=args.lat_col, lon_col=args.lon_col,
            transect_method=args.transect_method,
            reference_lat=args.reference_lat, reference_lon=args.reference_lon,
            gl0_suffix=args.gl0_suffix, gl1_suffix=args.gl1_suffix, gl2_suffix=args.gl2_suffix,
            phred=args.phred, individual_col=args.individual_col, loci=loci,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _load_genomic_gl_multi_locus_data(args):
    from hycline.data.io import read_genomic_gl_multi_locus_wide_csv, TransectCSVError

    loci = [l.strip() for l in args.loci.split(",")] if args.loci else None
    try:
        return read_genomic_gl_multi_locus_wide_csv(
            args.input, hybrid_index_col=args.hybrid_index_col,
            gl0_suffix=args.gl0_suffix, gl1_suffix=args.gl1_suffix, gl2_suffix=args.gl2_suffix,
            phred=args.phred, individual_col=args.individual_col, loci=loci,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _load_gl_transect(args):
    from hycline.data.io import read_gl_csv, TransectCSVError

    try:
        return read_gl_csv(
            args.input,
            distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
            lat_col=args.lat_col, lon_col=args.lon_col,
            transect_method=args.transect_method,
            reference_lat=args.reference_lat, reference_lon=args.reference_lon,
            gl0_col=args.gl0_col, gl1_col=args.gl1_col, gl2_col=args.gl2_col,
            phred=args.phred, individual_col=args.individual_col,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _load_genomic_gl_data(args):
    from hycline.data.io import read_genomic_gl_csv, TransectCSVError

    try:
        return read_genomic_gl_csv(
            args.input, hybrid_index_col=args.hybrid_index_col,
            gl0_col=args.gl0_col, gl1_col=args.gl1_col, gl2_col=args.gl2_col,
            phred=args.phred, individual_col=args.individual_col,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def _load_locus_summary_csv(path, locus_col):
    import pandas as pd
    if not Path(path).exists():
        print(f"\nERROR: input file not found: '{path}'\n", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(path)
    if locus_col not in df.columns:
        print(f"\nERROR: --locus-col '{locus_col}' not found in '{path}'. "
              f"Columns present: {', '.join(df.columns)}\n", file=sys.stderr)
        sys.exit(2)
    return df


def _resolve_summary_col(df, explicit, candidates, path, flag_name):
    if explicit is not None:
        if explicit not in df.columns:
            print(f"\nERROR: {flag_name} '{explicit}' not found in '{path}'. "
                  f"Columns present: {', '.join(df.columns)}\n", file=sys.stderr)
            sys.exit(2)
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    print(f"\nERROR: could not auto-detect a column for {flag_name} in '{path}' "
          f"(tried: {', '.join(candidates)}). Specify {flag_name} explicitly. "
          f"Columns present: {', '.join(df.columns)}\n", file=sys.stderr)
    sys.exit(2)


def _tail_type(name: str):
    from hycline.models.cline import TailType
    return {"none": TailType.NONE, "mirror": TailType.MIRROR, "both": TailType.BOTH,
            "left": TailType.LEFT, "right": TailType.RIGHT}[name]


def _format_loci_list(loci, max_shown: int = 10) -> str:
    """Format a locus-name list for console output, truncating when long
    so a "thousands of loci" run doesn't dump an unreadable wall of text
    to the terminal -- a real issue found during a stress test at that
    scale (BACKEND_SCALE.md), not something visible at the handful-of-loci
    scale every command was originally tested at."""
    loci = list(loci)
    if len(loci) <= max_shown:
        return str(loci)
    shown = loci[:max_shown]
    return f"{shown} ... and {len(loci) - max_shown} more"


def _try_stack_dense(data_dict, covariate_attr, count_attrs):
    """Non-fatal core of the dense-data check: returns
    (is_dense: bool, locus_names, reference_covariate, stacked_dict).
    The last three are None when `is_dense` is False. Shared by
    `_stack_for_hierarchical` (which turns a False result into a hard
    error, since hierarchical batch fitting has no fallback) and
    `cmd_fit_multi`'s Bayesian path (which falls back to independent
    per-locus fitting instead of failing -- see BACKEND_PHASE2.md
    section 13 for why plate-vectorized batching is only used when data
    happens to already be dense, not required).

    Order, not just the set of covariate values, matters for stacking
    correctly -- but every locus here originates from the same
    wide-format CSV's row order (`read_multi_locus_wide_csv`/
    `read_genomic_cline_multi_locus_wide_csv` drop missing rows
    independently per locus, but never reorder the rows that remain), so
    two loci with the exact same *set* of covariate values are
    guaranteed to have them in the same relative order too -- checking
    the sorted values match is sufficient here, not a shortcut that
    risks silent misalignment.
    """
    locus_names = list(data_dict.keys())
    reference = getattr(data_dict[locus_names[0]], covariate_attr)

    for name in locus_names[1:]:
        cov = getattr(data_dict[name], covariate_attr)
        if len(cov) != len(reference) or not np.allclose(np.sort(cov), np.sort(reference)):
            return False, None, None, None

    stacked = {}
    for attr in count_attrs:
        stacked[attr] = np.stack([getattr(data_dict[name], attr) for name in locus_names])

    return True, locus_names, reference, stacked


def _stack_for_hierarchical(data_dict, covariate_attr, count_attrs, kind_label):
    """Convert a dict of per-locus data containers (as returned by
    `read_multi_locus_wide_csv`/`read_genomic_cline_multi_locus_wide_csv`)
    into the dense, stacked (n_loci, n_sites_or_individuals) arrays the
    hierarchical batch-fitting functions need.

    Hierarchical batch fitting has no per-locus missing-data masking (see
    BACKEND_PHASE4.md sections 2 and 6) -- every locus must share the
    exact same covariate vector (`distance` or `hybrid_index`). The
    per-locus readers used by `fit-geographic`/`fit-genomic`, by
    contrast, drop missing rows independently for each locus -- fine for
    independent per-locus fitting, but it means different loci can end up
    with different site/individual subsets if the input CSV has any
    missing values anywhere. This function checks for that and fails with
    an actionable error rather than silently misaligning data, since a
    dimension mismatch here would otherwise fail deep inside NumPyro with
    a much less useful error, or (worse) silently broadcast incorrectly.
    """
    is_dense, locus_names, reference, stacked = _try_stack_dense(
        data_dict, covariate_attr, count_attrs)
    if not is_dense:
        all_names = list(data_dict.keys())
        # find one mismatching locus to name in the error, matching the
        # previous (pre-refactor) error message's specificity
        ref = getattr(data_dict[all_names[0]], covariate_attr)
        bad_name = all_names[0]
        for name in all_names[1:]:
            cov = getattr(data_dict[name], covariate_attr)
            if len(cov) != len(ref) or not np.allclose(np.sort(cov), np.sort(ref)):
                bad_name = name
                break
        print(
            f"\nERROR: hierarchical batch fitting requires every locus to have data "
            f"at exactly the same {'sites' if kind_label == 'geographic' else 'individuals'} "
            f"(no per-locus missing-data masking) -- locus '{bad_name}' has a different "
            f"{covariate_attr} pattern than locus '{all_names[0]}'. This usually "
            f"means your input CSV has missing values for some loci but not others.\n"
            f"Either provide complete data for every locus, or use "
            f"'hycline fit-{'geographic' if kind_label == 'geographic' else 'genomic'}' "
            f"instead, which fits each locus independently and handles per-locus "
            f"missing data by dropping only the affected rows for that locus.\n",
            file=sys.stderr)
        sys.exit(2)

    return locus_names, reference, stacked


def _write_observed_data_csv(transect, path):
    import pandas as pd
    df = pd.DataFrame({
        "distance": transect.distance,
        "n_alleles": transect.n_alleles,
        "n_focal": transect.n_focal,
        "observed_freq": transect.observed_freq,
    })
    if transect.site_names is not None:
        df.insert(0, "site", transect.site_names)
    df.to_csv(path, index=False)


class UnsupportedFitConfiguration(Exception):
    """Raised when a requested method/backend/scaling combination isn't
    supported (e.g. PyMC backend + fixed scaling). Caught separately from
    other errors so a multi-locus fit-geographic run can skip just the
    offending locus rather than aborting the whole batch, while a
    genuinely single-locus input still exits with a clear message for
    one bad configuration."""


def _resolve_fixed_extremes(transect, pmin_fixed, pmax_fixed, use_fixed, n_edge_sites=1):
    """Match fit_mle_fixed_extremes's own default: derive from the observed
    frequencies at the transect's geographic extremes. fit_mcmc (unlike
    fit_mle_fixed_extremes) does NOT do this derivation itself -- it treats
    pmin_fixed=None as "sample freely" -- so the CLI has to do it explicitly
    here for the bayesian path to actually match --scaling fixed's stated
    meaning. See BACKEND_PHASE2.md section 10 for the bug this was written
    to fix. Uses the same `_derive_pmin_pmax_single_locus` helper
    `fit_mle_fixed_extremes` itself uses, rather than a separate inline
    copy, so the classic and bayesian CLI paths always agree on how
    pmin/pmax get derived -- including the `n_edge_sites` fragility fix
    (BACKEND_PHASE1.md section 4.2, BACKEND_PHASE4.md section 3): the
    default (1) reproduces the exact original single-site convention;
    pass a higher value for the more sample-size-robust averaged version."""
    if use_fixed and (pmin_fixed is None or pmax_fixed is None):
        from hycline.inference.mle import _derive_pmin_pmax_single_locus
        derived_pmin, derived_pmax = _derive_pmin_pmax_single_locus(
            transect.distance, transect.n_alleles, transect.n_focal, n_edge_sites)
        if pmin_fixed is None:
            pmin_fixed = derived_pmin
        if pmax_fixed is None:
            pmax_fixed = derived_pmax
    return pmin_fixed, pmax_fixed


def _fit_single_locus_core(transect, args, tail_type, use_fixed, pmin_fixed, pmax_fixed):
    """Fit one locus with whichever method/backend `args` specifies.
    Returns the fit object (FitResult or BayesianFitResult). Raises
    UnsupportedFitConfiguration for combinations that aren't implemented
    (e.g. PyMC + fixed scaling) -- the caller (fit-geographic's per-locus
    loop) skips just the offending locus rather than aborting the whole
    run, whether there's one locus or many."""
    if args.method == "classic":
        from hycline.inference.mle import fit_mle, fit_mle_fixed_extremes
        from hycline.models.cline import TailType
        if use_fixed:
            if getattr(args, "hzar_tail", False) and tail_type != TailType.NONE:
                raise UnsupportedFitConfiguration(
                    "--hzar-tail is not supported together with --scaling fixed / --pmin / "
                    "--pmax -- fit_mle_fixed_extremes has no HZAR-parameterized tail option "
                    "(only fit_mle, the estimated-extremes path, does). This is a real, "
                    "currently-unclosed gap, not a silently-ignored flag -- see "
                    "BACKEND_TRAIT.md section 5. Use estimated scaling (the default) with "
                    "--hzar-tail, or drop --hzar-tail to use fixed extremes."
                )
            return fit_mle_fixed_extremes(transect, tail_type=tail_type,
                                           pmin=pmin_fixed, pmax=pmax_fixed,
                                           n_edge_sites=args.n_edge_sites)
        return fit_mle(transect, tail_type=tail_type,
                        hzar_tail=getattr(args, "hzar_tail", False))

    if args.backend == "numpyro":
        from hycline.inference.mcmc_numpyro import fit_mcmc
        return fit_mcmc(
            transect, tail_type=tail_type,
            pmin_fixed=pmin_fixed if use_fixed else None,
            pmax_fixed=pmax_fixed if use_fixed else None,
            width_upper_bound=args.width_upper_bound,
            num_warmup=args.num_warmup, num_samples=args.num_samples,
            num_chains=args.num_chains, seed=args.seed,
            hzar_tail=getattr(args, "hzar_tail", False),
        )
    else:
        if use_fixed:
            raise UnsupportedFitConfiguration(
                "--backend pymc does not yet support --scaling fixed / "
                "--pmin / --pmax. Use --backend numpyro instead."
            )
        from hycline.inference.mcmc_pymc import fit_mcmc_pymc
        return fit_mcmc_pymc(
            transect, tail_type=tail_type,
            num_warmup=args.num_warmup, num_samples=args.num_samples,
            num_chains=args.num_chains, seed=args.seed,
        )


def _load_trait_transect(args):
    from hycline.data.io import read_trait_csv, read_raw_trait_csv, TransectCSVError

    if getattr(args, "raw", False):
        try:
            return read_raw_trait_csv(
                args.input, trait_col=args.trait_col,
                distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
                site_col=args.site_col, lat_col=args.lat_col, lon_col=args.lon_col,
                transect_method=args.transect_method,
                reference_lat=args.reference_lat, reference_lon=args.reference_lon,
                ddof=args.ddof,
            )
        except TransectCSVError as e:
            print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
            sys.exit(2)

    if args.sd_col is None:
        print("\nERROR: --sd-col is required for site-aggregated input (or use --raw "
              "for individual-level input instead).\n", file=sys.stderr)
        sys.exit(2)
    try:
        return read_trait_csv(
            args.input,
            distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
            lat_col=args.lat_col, lon_col=args.lon_col,
            transect_method=args.transect_method,
            reference_lat=args.reference_lat, reference_lon=args.reference_lon,
            mean_col=args.mean_col, n_col=args.n_col, sd_col=args.sd_col,
            site_col=args.site_col,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        sys.exit(2)


def cmd_fit_trait(args) -> int:
    from hycline.models.cline import param_names as _cline_param_names
    from hycline.likelihoods.gaussian import mean_cline
    from hycline.inference.mle_trait import fit_mle_trait
    from hycline.viz.plotting import plot_trait_fit

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    print(f"Reading input data from '{args.input}' ...")
    transect = _load_trait_transect(args)
    print(f"  {len(transect)} sampling sites loaded.")

    tail_type = _tail_type(args.tail)

    t0 = time.time()
    print(f"Fitting (tail={args.tail}, HZAR-parity variance model) ...")
    try:
        fit = fit_mle_trait(transect, tail_type=tail_type)
    except ValueError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. {fit.summary()}")

    x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 300)
    mp = fit.mean_params
    y_smooth = mean_cline(x_smooth, mp.center, mp.width, mp.pmin, mp.pmax, tail_type,
                           mp.delta_l, mp.tau_l, mp.delta_r, mp.tau_r)
    import pandas as pd
    pd.DataFrame({"distance": x_smooth, "fitted_trait": y_smooth}).to_csv(
        out_dir / f"{prefix}_fitted_curve.csv", index=False)

    summary_text = fit.summary()
    (out_dir / f"{prefix}_summary.txt").write_text(summary_text + "\n")

    cline_param_count = fit.n_params - 3
    param_names_for_json = _cline_param_names(tail_type)[:cline_param_count] + [
        "var_l", "var_r", "var_h"]
    summary_json = {
        "method": "classic", "tail_type": args.tail,
        "n_obs": fit.n_obs, "n_params": fit.n_params,
        "log_likelihood": fit.log_likelihood, "aicc": fit.aicc,
        "converged": fit.success,
        "parameters": {
            name: {
                "estimate": float(fit.param_vec[i]),
                "std_error": (float(fit.std_errors[i])
                              if fit.std_errors is not None
                              and np.isfinite(fit.std_errors[i]) else None),
            }
            for i, name in enumerate(param_names_for_json)
        },
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary_json, indent=2))

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_trait_fit(transect, fit, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_trait_cline_plot.png", dpi=150)
        plt.close(fig)

    print(f"\nAll output written to: {out_dir}")
    return 0


def cmd_fit_trait_multi(args) -> int:
    from hycline.data.io import read_multi_trait_wide_csv, TransectCSVError
    from hycline.inference.mle_trait import fit_mle_trait
    from hycline.viz.plotting import plot_trait_multi_grid
    from hycline.models.cline import TailType

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    print(f"Reading input data from '{args.input}' ...")
    traits_requested = [t.strip() for t in args.traits.split(",")] if args.traits else None
    try:
        all_data = read_multi_trait_wide_csv(
            args.input,
            distance_col=(None if (args.lat_col and args.lon_col) else args.distance_col),
            lat_col=args.lat_col, lon_col=args.lon_col,
            transect_method=args.transect_method,
            reference_lat=args.reference_lat, reference_lon=args.reference_lon,
            mean_suffix=args.mean_suffix, n_suffix=args.n_suffix, sd_suffix=args.sd_suffix,
            site_col=args.site_col, traits=traits_requested,
        )
    except TransectCSVError as e:
        print(f"\nERROR reading input data: {e}\n", file=sys.stderr)
        return 2
    print(f"  {len(all_data)} traits detected: {_format_loci_list(all_data.keys())}")

    tail_type = _tail_type(args.tail)

    import pandas as pd
    summary_rows = []
    curve_rows = []
    plot_results = {}
    failures = {}

    for trait_name, transect in all_data.items():
        print(f"\n--- Trait '{trait_name}' ({len(transect)} sites) ---")
        if transect.trait_sd is None:
            msg = (f"trait '{trait_name}' has no '{trait_name}{args.sd_suffix}' column -- "
                   f"skipping. HZAR's own model always requires each site's observed "
                   f"sample variance as data (see 'hycline fit-trait --help').")
            print(f"  SKIPPED: {msg}")
            failures[trait_name] = msg
            continue
        t0 = time.time()
        try:
            fit = fit_mle_trait(transect, tail_type=tail_type)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            failures[trait_name] = str(e)
            continue
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. {fit.summary()}")

        from hycline.likelihoods.gaussian import mean_cline
        x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 200)
        mp = fit.mean_params
        y_smooth = mean_cline(x_smooth, mp.center, mp.width, mp.pmin, mp.pmax, tail_type,
                               mp.delta_l, mp.tau_l, mp.delta_r, mp.tau_r)
        for x, y in zip(x_smooth, y_smooth):
            curve_rows.append({"trait": trait_name, "distance": x, "fitted_trait": y})

        row = {
            "trait": trait_name, "n_sites": len(transect), "converged": fit.success,
            "log_likelihood": fit.log_likelihood, "aicc": fit.aicc,
            "center": mp.center, "width": mp.width, "pmin": mp.pmin, "pmax": mp.pmax,
        }
        if tail_type != TailType.NONE:
            for name in ("delta_l", "tau_l", "delta_r", "tau_r"):
                val = getattr(mp, name, None)
                if val is not None:
                    row[name] = val
        row["var_l"] = fit.var_l
        row["var_r"] = fit.var_r
        row["var_h"] = fit.var_h
        for i, name in enumerate(["center", "width", "pmin", "pmax"]):
            if fit.std_errors is not None and i < len(fit.std_errors) and np.isfinite(fit.std_errors[i]):
                row[f"{name}_se"] = fit.std_errors[i]
        summary_rows.append(row)
        plot_results[trait_name] = (transect, fit)

    if not summary_rows:
        print("\nERROR: no traits fit successfully.", file=sys.stderr)
        return 1

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{prefix}_trait_multi_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / f"{prefix}_trait_multi_fitted_curves.csv", index=False)

    if failures:
        (out_dir / f"{prefix}_trait_multi_failures.json").write_text(json.dumps(failures, indent=2))

    if args.plot_grid and plot_results:
        import matplotlib.pyplot as plt
        fig = plot_trait_multi_grid(plot_results)
        fig.savefig(out_dir / f"{prefix}_trait_multi_grid.png", dpi=150)
        plt.close(fig)

    print(f"\n{len(summary_rows)}/{len(all_data)} traits fit successfully.")
    print(f"\nAll output written to: {out_dir}")
    return 0


def cmd_compare_tails(args) -> int:
    from hycline.selection.aicc import compare_models
    from hycline.viz.plotting import plot_fit

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    print(f"Reading input data from '{args.input}' ...")
    transect = _load_transect(args)
    print(f"  {len(transect)} sampling sites loaded.")

    candidate_names = [c.strip() for c in args.candidates.split(",") if c.strip()]
    candidates = [_tail_type(c) for c in candidate_names]

    print(f"Fitting {len(candidates)} tail-model variants: {candidate_names} ...")
    t0 = time.time()
    comparison = compare_models(transect, candidates=candidates)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s.\n{comparison.summary()}")

    comparison.table.to_csv(out_dir / f"{prefix}_model_comparison.csv", index=False)
    (out_dir / f"{prefix}_model_comparison.txt").write_text(comparison.summary() + "\n")

    _write_observed_data_csv(transect, out_dir / f"{prefix}_observed_data.csv")
    (out_dir / f"{prefix}_run_config.json").write_text(
        json.dumps({k: v for k, v in vars(args).items()}, indent=2, default=str))

    if args.plot_best and not args.no_plot:
        import matplotlib.pyplot as plt
        best_fit = comparison.results[comparison.best_model]
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_fit(transect, best_fit, ax=ax,
                 title=f"Best model by AICc: {comparison.best_model.value}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_best_model_plot.png", dpi=150)
        plt.close(fig)

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_model_comparison.csv/.txt -- AICc ranking table")
    print(f"  {prefix}_observed_data.csv         -- your input data, parsed and echoed back")
    if args.plot_best and not args.no_plot:
        print(f"  {prefix}_best_model_plot.png       -- plot of the best-fitting model")
    print(f"  {prefix}_run_config.json           -- the exact settings used (for reproducibility)")
    return 0


def _report_bayesian_multilocus_fit(locus_name, fit, transect, x_smooth, tail_type, args,
                                     curve_rows, summary_rows, plot_results):
    """Shared reporting logic for one locus's Bayesian fit within
    `cmd_fit_multi` -- appends to `curve_rows`/`summary_rows`/
    `plot_results` in place, matching the mutation pattern the calling
    loop already uses. Shared between the original per-locus MCMC loop
    (one `fit_mcmc` call per locus) and the plate-vectorized batched path
    (`_cmd_fit_multi_bayesian_batched`, one `fit_mcmc_batch` call sliced
    per locus via `idata.isel(locus=i)`) -- both produce the same
    `BayesianFitResult` shape, so this function doesn't need to know
    which path a given `fit-geographic` came from. See BACKEND_PHASE2.md section 13
    for why the batched path exists and when it's used.
    """
    from hycline.diagnostics.convergence import check
    from hycline.models.cline import ClineParams, cline_function, hzar_cline_function

    report = check(fit.idata)
    curve_fn = hzar_cline_function if fit.hzar_tail else cline_function

    post = fit.idata.posterior
    center_draws = post["center"].values.flatten()
    width_draws = post["width"].values.flatten()
    pmin_draws = (post["pmin"].values.flatten() if "pmin" in post
                  else np.full_like(center_draws, fit._fixed_pmin))
    pmax_draws = (post["pmax"].values.flatten() if "pmax" in post
                  else np.full_like(center_draws, fit._fixed_pmax))
    delta_l_draws = post["delta_l"].values.flatten() if "delta_l" in post else None
    tau_l_draws = post["tau_l"].values.flatten() if "tau_l" in post else None
    delta_r_draws = post["delta_r"].values.flatten() if "delta_r" in post else None
    tau_r_draws = post["tau_r"].values.flatten() if "tau_r" in post else None
    n_curves = min(200, len(center_draws))
    idx = np.random.default_rng(args.seed).choice(len(center_draws), n_curves, replace=False)
    curves = np.empty((n_curves, len(x_smooth)))
    for row_i, i in enumerate(idx):
        p = ClineParams(
            center=float(center_draws[i]), width=float(width_draws[i]),
            pmin=float(pmin_draws[i]), pmax=float(pmax_draws[i]),
            delta_l=float(delta_l_draws[i]) if delta_l_draws is not None else None,
            tau_l=float(tau_l_draws[i]) if tau_l_draws is not None else None,
            delta_r=float(delta_r_draws[i]) if delta_r_draws is not None else None,
            tau_r=float(tau_r_draws[i]) if tau_r_draws is not None else None,
        )
        curves[row_i] = curve_fn(x_smooth, p, tail_type)
    lo_pct = (100 - args.credible_prob * 100) / 2
    hi_pct = 100 - lo_pct
    for j, x in enumerate(x_smooth):
        curve_rows.append({
            "locus": locus_name, "distance": x,
            "fitted_freq_median": np.percentile(curves[:, j], 50),
            "fitted_freq_lower": np.percentile(curves[:, j], lo_pct),
            "fitted_freq_upper": np.percentile(curves[:, j], hi_pct),
        })

    row = {
        "locus": locus_name, "n_sites": len(transect),
        "converged_diagnostics_ok": report.ok,
        "max_r_hat": report.rhat_max, "n_divergences": report.n_divergences,
        "center_mean": float(post["center"].mean()),
        "width_mean": float(post["width"].mean()),
    }
    for name in ("center", "width", "pmin", "pmax", "delta_l", "tau_l", "delta_r", "tau_r"):
        if name in post:
            lo, hi = fit.hdi(name, prob=args.credible_prob)
            row[f"{name}_credible_lower"] = lo
            row[f"{name}_credible_upper"] = hi
    summary_rows.append(row)
    plot_results[locus_name] = (transect, fit)
    return report


def _cmd_fit_multi_bayesian_batched(args, transects, tail_type, out_dir, prefix):
    """The plate-vectorized fast path for `cmd_fit_multi`'s Bayesian
    method: one `fit_mcmc_batch` call for every locus at once, instead of
    one `fit_mcmc` call per locus. Only reachable when the data happens
    to already be dense (every locus shares the exact same distance
    vector) and fixed-extremes scaling isn't requested (`fit_mcmc_batch`
    has no equivalent of `pmin_fixed`/`pmax_fixed` -- BACKEND_PHASE2.md
    section 13). Returns None if either precondition fails, signaling
    the caller to fall back to the original per-locus loop -- this
    function never partially commits to the batched path and then bails
    out midway.
    """
    from hycline.inference.mcmc_numpyro import fit_mcmc_batch
    from hycline.models.cline import cline_function

    is_dense, locus_names, reference_distance, stacked = _try_stack_dense(
        transects, "distance", ["n_alleles", "n_focal"])
    if not is_dense:
        return None

    print(f"\nAll {len(locus_names)} loci share the same {len(reference_distance)} sites -- "
          f"using the plate-vectorized batch fitting path (one NUTS run for all loci at "
          f"once, instead of {len(locus_names)} separate runs). See BACKEND_PHASE2.md "
          f"section 13.")
    t0 = time.time()
    batch_result = fit_mcmc_batch(
        reference_distance, stacked["n_alleles"], stacked["n_focal"], locus_names=locus_names,
        tail_type=tail_type, width_upper_bound=args.width_upper_bound,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, seed=args.seed,
        hzar_tail=getattr(args, "hzar_tail", False))
    elapsed = time.time() - t0
    print(f"Batch fit done in {elapsed:.1f}s ({elapsed / len(locus_names):.2f}s/locus "
          f"amortized).")

    from hycline.inference.mcmc_numpyro import BayesianFitResult
    summary_rows, curve_rows, plot_results = [], [], {}
    n_not_converged = 0
    for i, locus_name in enumerate(locus_names):
        transect = transects[locus_name]
        x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 200)
        sliced_idata = batch_result.idata.isel(locus=i)
        fit = BayesianFitResult(idata=sliced_idata, tail_type=tail_type, n_obs=len(transect),
                                 backend="numpyro", param_names=batch_result.param_names,
                                 hzar_tail=batch_result.hzar_tail)
        fit._fixed_pmin = None
        fit._fixed_pmax = None
        report = _report_bayesian_multilocus_fit(
            locus_name, fit, transect, x_smooth, tail_type, args,
            curve_rows, summary_rows, plot_results)
        if not report.ok:
            n_not_converged += 1
    if n_not_converged:
        print(f"\nNOTE: {n_not_converged}/{len(locus_names)} loci showed convergence "
              f"diagnostics issues (see the 'converged_diagnostics_ok'/'max_r_hat' columns "
              f"in the summary CSV) -- the batched fit uses the same NUTS sampler and "
              f"priors as the per-locus path, so the same tuning advice applies "
              f"(--width-upper-bound, --num-samples).", file=sys.stderr)

    return summary_rows, curve_rows, plot_results


def cmd_fit_geographic(args) -> int:
    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2

    print(f"Reading input data from '{args.input}' ...")
    data_type = _detect_geographic_input_type(args)

    if data_type == "gl":
        if args.method == "bayesian":
            print("\nERROR: genotype-likelihood input has no Bayesian backend yet -- "
                  "classic MLE only (drop --method bayesian, or use hard-count input "
                  "for Bayesian fitting).\n", file=sys.stderr)
            return 2
        use_fixed = (args.scaling == "fixed") or (args.pmin is not None) or (args.pmax is not None)
        if use_fixed:
            print("\nERROR: genotype-likelihood input does not support --scaling "
                  "fixed/--pmin/--pmax -- pmin/pmax are always estimated for this "
                  "input type.\n", file=sys.stderr)
            return 2
        if args.hzar_tail:
            print("\nERROR: genotype-likelihood input does not support --hzar-tail -- "
                  "only the default tail parameterization is available for this "
                  "input type.\n", file=sys.stderr)
            return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    if data_type == "gl":
        return _fit_geographic_gl(args, out_dir, prefix)
    return _fit_geographic_hard_count(args, out_dir, prefix)


def _fit_geographic_hard_count(args, out_dir, prefix) -> int:
    from hycline.models.cline import (TailType, cline_function, hzar_cline_function,
                                       param_names as _cline_param_names)
    from hycline.viz.plotting import plot_multi_locus_overlay

    transects = _load_multi_locus_transects(args)
    print(f"  {len(transects)} loci detected: {_format_loci_list(transects.keys())}")

    tail_type = _tail_type(args.tail)
    if tail_type != TailType.NONE and args.method == "bayesian" and args.backend == "numpyro":
        print("NOTE: Bayesian tail-model fitting is known to have real convergence "
              "difficulty (a non-smooth log-density where the sigmoid core meets the "
              "exponential tail) -- classic MLE (--method classic) is recommended for "
              "tail models instead. See BACKEND_PHASE2.md section 12.2. Proceeding, but "
              "check the per-locus convergence diagnostics carefully.", file=sys.stderr)

    import pandas as pd
    failures = {}

    use_fixed_globally = (args.scaling == "fixed") or (args.pmin is not None) or (args.pmax is not None)
    batched = None
    if args.method == "bayesian" and args.backend == "numpyro" and not use_fixed_globally:
        batched = _cmd_fit_multi_bayesian_batched(args, transects, tail_type, out_dir, prefix)

    if batched is not None:
        summary_rows, curve_rows, plot_results = batched
    else:
        summary_rows = []
        curve_rows = []
        plot_results = {}

        for locus_name, transect in transects.items():
            print(f"\n--- Locus '{locus_name}' ({len(transect)} sites) ---")

            pmin_fixed = args.pmin
            pmax_fixed = args.pmax
            use_fixed = (args.scaling == "fixed") or (pmin_fixed is not None) or (pmax_fixed is not None)
            pmin_fixed, pmax_fixed = _resolve_fixed_extremes(
                transect, pmin_fixed, pmax_fixed, use_fixed, n_edge_sites=args.n_edge_sites)

            t0 = time.time()
            try:
                fit = _fit_single_locus_core(transect, args, tail_type, use_fixed,
                                              pmin_fixed, pmax_fixed)
            except UnsupportedFitConfiguration as e:
                print(f"  SKIPPED: {e}")
                failures[locus_name] = str(e)
                continue
            except Exception as e:
                print(f"  FAILED to fit: {e}")
                failures[locus_name] = str(e)
                continue
            elapsed = time.time() - t0

            x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 200)

            if args.method == "classic":
                print(f"  Done in {elapsed:.1f}s. {fit.summary()}")
                curve_fn = hzar_cline_function if getattr(fit, "hzar_tail", False) else cline_function
                y_smooth = curve_fn(x_smooth, fit.params, tail_type)
                for x, y in zip(x_smooth, y_smooth):
                    curve_rows.append({"locus": locus_name, "distance": x, "fitted_freq": y})

                row = {
                    "locus": locus_name, "n_sites": len(transect),
                    "converged": fit.success, "log_likelihood": fit.log_likelihood,
                    "aicc": fit.aicc, "center": fit.params.center, "width": fit.params.width,
                    "pmin": fit.params.pmin, "pmax": fit.params.pmax,
                }
                if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
                    row["delta_l"], row["tau_l"] = fit.params.delta_l, fit.params.tau_l
                if tail_type in (TailType.RIGHT, TailType.BOTH):
                    row["delta_r"], row["tau_r"] = fit.params.delta_r, fit.params.tau_r
                if fit.std_errors is not None:
                    names = (["center", "width"] if use_fixed
                             else _cline_param_names(tail_type)[:len(fit.std_errors)])
                    for i, name in enumerate(names):
                        if np.isfinite(fit.std_errors[i]):
                            row[f"{name}_se"] = float(fit.std_errors[i])
                summary_rows.append(row)
                plot_results[locus_name] = (transect, fit)

            else:  # bayesian
                report = _report_bayesian_multilocus_fit(
                    locus_name, fit, transect, x_smooth, tail_type, args,
                    curve_rows, summary_rows, plot_results)
                print(f"  Done in {elapsed:.1f}s. {fit.summary()}\n  {report.summary()}")

    if not summary_rows:
        print("\nERROR: no loci fit successfully.", file=sys.stderr)
        return 1

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_fitted_curves.csv", index=False)

    observed_rows = []
    for locus_name, transect in transects.items():
        for i in range(len(transect)):
            row = {
                "locus": locus_name, "distance": transect.distance[i],
                "n_alleles": transect.n_alleles[i], "n_focal": transect.n_focal[i],
                "observed_freq": transect.observed_freq[i],
            }
            if transect.site_names is not None:
                row["site"] = transect.site_names[i]
            observed_rows.append(row)
    pd.DataFrame(observed_rows).to_csv(out_dir / f"{prefix}_observed_data.csv", index=False)

    if failures:
        (out_dir / f"{prefix}_failures.json").write_text(json.dumps(failures, indent=2))

    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        # classic fits store a single 'fitted_freq' column per point; bayesian fits
        # (per-locus loop or the plate-vectorized batch path) store a credible band
        # instead ('fitted_freq_median'/'_lower'/'_upper') -- use the median as this
        # locus's line in the overlay, since showing every locus's full credible
        # band at once would be unreadable regardless of how few loci there are.
        freq_col = "fitted_freq" if "fitted_freq" in curve_df.columns else "fitted_freq_median"
        curves = {locus: (g["distance"].to_numpy(), g[freq_col].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves)
        fig.savefig(out_dir / f"{prefix}_overlay.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-geographic"
    (out_dir / f"{prefix}_run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print(f"\n{len(summary_rows)}/{len(transects)} loci fit successfully.")
    if failures:
        print(f"Failed/skipped loci: {list(failures.keys())} "
              f"(see {prefix}_failures.json for details)")
    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_summary.csv        -- one row per locus: parameters + statistics")
    print(f"  {prefix}_fitted_curves.csv  -- fitted curve points for every locus (long format)")
    print(f"  {prefix}_observed_data.csv  -- your input data, parsed and echoed back (long format)")
    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        print(f"  {prefix}_overlay.png        -- one fitted-curve line per locus, shared axes")
    print(f"  {prefix}_run_config.json    -- the exact settings used (for reproducibility)")
    return 0


def _fit_geographic_gl(args, out_dir, prefix) -> int:
    from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl
    from hycline.models.cline import TailType, cline_function
    from hycline.viz.plotting import plot_multi_locus_overlay

    tail_type = _tail_type(args.tail)

    transects = _load_gl_multi_locus_transects(args)
    print(f"  {len(transects)} loci detected: {_format_loci_list(transects.keys())}")

    import pandas as pd
    from hycline.models.cline import param_names as _cline_param_names
    fitted_param_names = _cline_param_names(tail_type)
    summary_rows = []
    curve_rows = []
    failures = {}
    n_not_converged = 0

    for locus_name, transect in transects.items():
        print(f"\n--- Locus '{locus_name}' ({len(transect)} individuals) ---")
        t0 = time.time()
        try:
            fit = fit_mle_geographic_gl(transect, tail_type=tail_type)
        except Exception as e:
            print(f"  FAILED to fit: {e}")
            failures[locus_name] = str(e)
            continue
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. {fit.summary()}")
        if not fit.success:
            n_not_converged += 1

        x_smooth = np.linspace(transect.distance.min(), transect.distance.max(), 200)
        y_smooth = cline_function(x_smooth, fit.params, tail_type)
        for x, y in zip(x_smooth, y_smooth):
            curve_rows.append({"locus": locus_name, "distance": x, "fitted_freq": y})

        row = {
            "locus": locus_name, "n_obs": fit.n_obs, "converged": fit.success,
            "log_likelihood": fit.log_likelihood, "aicc": fit.aicc,
            "center": fit.params.center, "width": fit.params.width,
            "pmin": fit.params.pmin, "pmax": fit.params.pmax,
        }
        if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
            row["delta_l"], row["tau_l"] = fit.params.delta_l, fit.params.tau_l
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            row["delta_r"], row["tau_r"] = fit.params.delta_r, fit.params.tau_r
        if fit.std_errors is not None:
            for i, name in enumerate(fitted_param_names):
                if np.isfinite(fit.std_errors[i]):
                    row[f"{name}_se"] = float(fit.std_errors[i])
        summary_rows.append(row)

    if not summary_rows:
        print("\nERROR: no loci fit successfully.", file=sys.stderr)
        return 1
    if tail_type != TailType.NONE and n_not_converged > 0:
        print(f"\nNOTE: {n_not_converged}/{len(summary_rows)} loci did not report clean "
              f"convergence even after randomized restarts (see BACKEND_PHASE2.md "
              f"section 12.3 for why tail models can trigger this). Check the "
              f"'converged' column in the summary CSV and treat those loci's tail "
              f"parameters with extra caution.", file=sys.stderr)

    pd.DataFrame(summary_rows).to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_fitted_curves.csv", index=False)

    if failures:
        (out_dir / f"{prefix}_failures.json").write_text(json.dumps(failures, indent=2))

    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        curves = {locus: (g["distance"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        import matplotlib.pyplot as plt
        fig = plot_multi_locus_overlay(curves)
        fig.savefig(out_dir / f"{prefix}_overlay.png", dpi=150)
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-geographic"
    (out_dir / f"{prefix}_run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print(f"\n{len(summary_rows)}/{len(transects)} loci fit successfully.")
    if failures:
        print(f"Failed loci: {list(failures.keys())} (see {prefix}_failures.json)")
    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_summary.csv        -- one row per locus: parameters + statistics")
    print(f"  {prefix}_fitted_curves.csv  -- fitted curve points, every locus (long format)")
    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        print(f"  {prefix}_overlay.png        -- one fitted-curve line per locus, shared axes")
    print(f"  {prefix}_run_config.json    -- the exact settings used (for reproducibility)")
    return 0


def cmd_fit_genomic(args) -> int:
    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2

    print(f"Reading input data from '{args.input}' ...")
    data_type = _detect_genomic_input_type(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    if data_type == "gl":
        return _fit_genomic_gl(args, out_dir, prefix)
    return _fit_genomic_hard_count(args, out_dir, prefix)


def _fit_genomic_hard_count(args, out_dir, prefix) -> int:
    from hycline.inference.mle_genomic import fit_genomic_cline_mle
    from hycline.models.genomic_cline import genomic_cline_function
    from hycline.viz.plotting import plot_multi_locus_overlay

    all_data = _load_genomic_multi_locus_data(args)
    print(f"  {len(all_data)} loci detected: {_format_loci_list(all_data.keys())}")

    import pandas as pd
    summary_rows = []
    curve_rows = []
    failures = {}

    h_smooth = np.linspace(0.001, 0.999, 200)

    for locus_name, data in all_data.items():
        print(f"\n--- Locus '{locus_name}' ({len(data)} individuals) ---")
        t0 = time.time()
        try:
            fit = fit_genomic_cline_mle(data)
        except Exception as e:
            print(f"  FAILED to fit: {e}")
            failures[locus_name] = str(e)
            continue
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. {fit.summary()}")

        phi_smooth = genomic_cline_function(h_smooth, fit.params)
        for h, phi in zip(h_smooth, phi_smooth):
            curve_rows.append({"locus": locus_name, "hybrid_index": h, "fitted_freq": phi})

        row = {
            "locus": locus_name, "n_obs": fit.n_obs, "converged": fit.success,
            "log_likelihood": fit.log_likelihood, "aicc": fit.aicc,
            "u": fit.params.u, "v": fit.params.v,
        }
        if fit.std_errors is not None:
            for i, name in enumerate(["u", "v"]):
                if np.isfinite(fit.std_errors[i]):
                    row[f"{name}_se"] = float(fit.std_errors[i])
        summary_rows.append(row)

    if not summary_rows:
        print("\nERROR: no loci fit successfully.", file=sys.stderr)
        return 1

    pd.DataFrame(summary_rows).to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_fitted_curves.csv", index=False)

    if failures:
        (out_dir / f"{prefix}_failures.json").write_text(json.dumps(failures, indent=2))

    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        curves = {locus: (g["hybrid_index"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves, xlabel="Hybrid index",
                                        ylabel="Locus ancestry (focal-allele frequency)")
        fig.savefig(out_dir / f"{prefix}_overlay.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-genomic"
    (out_dir / f"{prefix}_run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print(f"\n{len(summary_rows)}/{len(all_data)} loci fit successfully.")
    if failures:
        print(f"Failed loci: {list(failures.keys())} "
              f"(see {prefix}_failures.json for details)")
    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_summary.csv        -- one row per locus: u, v, statistics")
    print(f"  {prefix}_fitted_curves.csv  -- fitted curve points, every locus (long format)")
    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        print(f"  {prefix}_overlay.png        -- one fitted-curve line per locus, shared axes")
    print(f"  {prefix}_run_config.json    -- the exact settings used (for reproducibility)")
    return 0


def cmd_fit_hierarchical_geographic(args) -> int:
    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2

    print(f"Reading input data from '{args.input}' ...")
    data_type = _detect_geographic_input_type(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    if data_type == "gl":
        return _fit_hierarchical_geographic_gl(args, out_dir, prefix)
    return _fit_hierarchical_geographic_hard_count(args, out_dir, prefix)


def _fit_hierarchical_geographic_hard_count(args, out_dir, prefix) -> int:
    from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch, _derive_pmin_pmax
    from hycline.diagnostics.convergence import check
    from hycline.viz.plotting import plot_multi_locus_overlay

    transects = _load_multi_locus_transects(args)
    print(f"  {len(transects)} loci detected: {_format_loci_list(transects.keys())}")

    locus_names, distance, stacked = _stack_for_hierarchical(
        transects, covariate_attr="distance", count_attrs=["n_alleles", "n_focal"],
        kind_label="geographic")
    n_alleles, n_focal = stacked["n_alleles"], stacked["n_focal"]
    print(f"  All {len(locus_names)} loci share {len(distance)} sites -- ready for "
          f"hierarchical fitting.")

    print(f"\nFitting hierarchically ({args.num_chains} chains, {args.num_warmup} warmup, "
          f"{args.num_samples} samples) -- this can take a while for many loci ...")
    t0 = time.time()
    result = fit_hierarchical_batch(
        distance, n_alleles, n_focal, locus_names=locus_names,
        n_edge_sites=args.n_edge_sites, sigma_center_scale=args.sigma_center_scale,
        mu_log_width_loc=args.mu_log_width_loc, mu_log_width_scale=args.mu_log_width_scale,
        sigma_log_width_scale=args.sigma_log_width_scale,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, seed=args.seed,
    )
    elapsed = time.time() - t0
    report = check(result.idata)
    print(f"Done in {elapsed:.1f}s.\n\n{result.summary()}\n\n{report.summary()}")
    if not report.ok:
        print("\nNOTE: convergence diagnostics flagged possible issues (see above). "
              "Try tightening --sigma-center-scale/--sigma-log-width-scale, or more "
              "--num-samples.", file=sys.stderr)

    pmin_all, pmax_all = _derive_pmin_pmax(distance, n_alleles, n_focal,
                                            n_edge_sites=args.n_edge_sites)

    import pandas as pd
    pop_df = result.population_summary()
    pop_df.to_csv(out_dir / f"{prefix}_hierarchical_population_summary.csv")
    locus_df = result.locus_summary()
    locus_df["pmin_fixed"] = pmin_all
    locus_df["pmax_fixed"] = pmax_all
    locus_df.to_csv(out_dir / f"{prefix}_hierarchical_locus_summary.csv", index=False)

    (out_dir / f"{prefix}_hierarchical_summary.txt").write_text(
        result.summary() + "\n\n" + report.summary() + "\n")

    summary_json = {
        "model": "hierarchical_geographic_cline",
        "n_loci": len(locus_names), "n_sites": len(distance),
        "convergence": {
            "ok": report.ok, "max_r_hat": report.rhat_max,
            "min_ess_bulk": report.ess_bulk_min, "min_ess_tail": report.ess_tail_min,
            "n_divergences": report.n_divergences, "warnings": report.warnings,
        },
        "population_level": pop_df["mean"].to_dict(),
    }
    (out_dir / f"{prefix}_hierarchical_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str))

    # long-format fitted curves, one per locus, using posterior-mean center/width
    from hycline.models.cline import ClineParams, TailType, cline_function
    x_smooth = np.linspace(distance.min(), distance.max(), 200)
    curve_rows = []
    for i, name in enumerate(locus_names):
        row = locus_df[locus_df["locus"] == name].iloc[0]
        params = ClineParams(center=row["center_mean"], width=row["width_mean"],
                              pmin=pmin_all[i], pmax=pmax_all[i])
        y_smooth = cline_function(x_smooth, params, TailType.NONE)
        for x, y in zip(x_smooth, y_smooth):
            curve_rows.append({"locus": name, "distance": x, "fitted_freq": y})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_hierarchical_fitted_curves.csv", index=False)

    if args.plot_grid and not args.no_plot:
        import matplotlib.pyplot as plt
        curves = {locus: (g["distance"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves)
        fig.savefig(out_dir / f"{prefix}_hierarchical_overlay.png", dpi=150)
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-hierarchical-geographic"
    (out_dir / f"{prefix}_run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str))

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_hierarchical_population_summary.csv  -- population-level hyperparameters")
    print(f"  {prefix}_hierarchical_locus_summary.csv        -- one row per locus: center, width, statistics")
    print(f"  {prefix}_hierarchical_fitted_curves.csv        -- fitted curve points, every locus (long format)")
    if args.plot_grid and not args.no_plot:
        print(f"  {prefix}_hierarchical_overlay.png              -- one fitted-curve line per locus, shared axes")
    print(f"  {prefix}_run_config.json                       -- the exact settings used (for reproducibility)")
    return 0


def cmd_fit_hierarchical_genomic(args) -> int:
    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2

    print(f"Reading input data from '{args.input}' ...")
    data_type = _detect_genomic_input_type(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    if data_type == "gl":
        return _fit_hierarchical_genomic_gl(args, out_dir, prefix)
    return _fit_hierarchical_genomic_hard_count(args, out_dir, prefix)


def _fit_hierarchical_genomic_hard_count(args, out_dir, prefix) -> int:
    from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch
    from hycline.diagnostics.convergence import check
    from hycline.viz.plotting import plot_multi_locus_overlay

    all_data = _load_genomic_multi_locus_data(args)
    print(f"  {len(all_data)} loci detected: {_format_loci_list(all_data.keys())}")

    locus_names, hybrid_index, stacked = _stack_for_hierarchical(
        all_data, covariate_attr="hybrid_index", count_attrs=["n_alleles", "n_focal"],
        kind_label="genomic")
    n_alleles, n_focal = stacked["n_alleles"], stacked["n_focal"]
    print(f"  All {len(locus_names)} loci share {len(hybrid_index)} individuals -- ready "
          f"for hierarchical fitting.")

    print(f"\nFitting hierarchically ({args.num_chains} chains, {args.num_warmup} warmup, "
          f"{args.num_samples} samples) -- this can take a while for many loci ...")
    t0 = time.time()
    result = fit_hierarchical_genomic_batch(
        hybrid_index, n_alleles, n_focal, locus_names=locus_names,
        mu_u_scale=args.mu_u_scale, sigma_u_scale=args.sigma_u_scale,
        mu_log_v_loc=args.mu_log_v_loc, mu_log_v_scale=args.mu_log_v_scale,
        sigma_log_v_scale=args.sigma_log_v_scale,
        pool_center=args.pool_center, mu_center_scale=args.mu_center_scale,
        sigma_center_scale=args.sigma_center_scale,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, seed=args.seed,
    )
    elapsed = time.time() - t0
    report = check(result.idata)
    print(f"Done in {elapsed:.1f}s.\n\n{result.summary()}\n\n{report.summary()}")
    if not report.ok:
        tuning_hint = "--sigma-center-scale" if args.pool_center else "--sigma-u-scale"
        print(f"\nNOTE: convergence diagnostics flagged possible issues (see above). "
              f"Try tightening {tuning_hint}/--sigma-log-v-scale, or more "
              f"--num-samples.", file=sys.stderr)

    import pandas as pd
    pop_df = result.population_summary()
    pop_df.to_csv(out_dir / f"{prefix}_hierarchical_population_summary.csv")
    locus_df = result.locus_summary()
    locus_df.to_csv(out_dir / f"{prefix}_hierarchical_locus_summary.csv", index=False)

    (out_dir / f"{prefix}_hierarchical_summary.txt").write_text(
        result.summary() + "\n\n" + report.summary() + "\n")

    summary_json = {
        "model": "hierarchical_genomic_cline",
        "n_loci": len(locus_names), "n_individuals": len(hybrid_index),
        "convergence": {
            "ok": report.ok, "max_r_hat": report.rhat_max,
            "min_ess_bulk": report.ess_bulk_min, "min_ess_tail": report.ess_tail_min,
            "n_divergences": report.n_divergences, "warnings": report.warnings,
        },
        "population_level": pop_df["mean"].to_dict(),
    }
    (out_dir / f"{prefix}_hierarchical_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str))

    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
    h_smooth = np.linspace(0.001, 0.999, 200)
    curve_rows = []
    for name in locus_names:
        row = locus_df[locus_df["locus"] == name].iloc[0]
        params = GenomicClineParams(u=row["u_mean"], v=row["v_mean"])
        phi_smooth = genomic_cline_function(h_smooth, params)
        for h, phi in zip(h_smooth, phi_smooth):
            curve_rows.append({"locus": name, "hybrid_index": h, "fitted_freq": phi})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_hierarchical_fitted_curves.csv", index=False)

    if args.plot_grid and not args.no_plot:
        import matplotlib.pyplot as plt
        curves = {locus: (g["hybrid_index"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves, xlabel="Hybrid index",
                                        ylabel="Locus ancestry (focal-allele frequency)")
        fig.savefig(out_dir / f"{prefix}_hierarchical_overlay.png", dpi=150)
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-hierarchical-genomic"
    (out_dir / f"{prefix}_run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str))

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_hierarchical_population_summary.csv  -- population-level hyperparameters")
    print(f"  {prefix}_hierarchical_locus_summary.csv        -- one row per locus: u, v, statistics")
    print(f"  {prefix}_hierarchical_fitted_curves.csv        -- fitted curve points, every locus")
    if args.plot_grid and not args.no_plot:
        print(f"  {prefix}_hierarchical_overlay.png              -- one fitted-curve line per locus, shared axes")
    print(f"  {prefix}_run_config.json                       -- the exact settings used")
    return 0


def _fit_hierarchical_geographic_gl(args, out_dir, prefix) -> int:
    from hycline.inference.mcmc_hierarchical_gl import (
        fit_hierarchical_batch_gl, _derive_pmin_pmax_gl,
    )
    from hycline.diagnostics.convergence import check
    from hycline.viz.plotting import plot_multi_locus_overlay

    transects = _load_gl_multi_locus_transects(args)
    print(f"  {len(transects)} loci detected: {_format_loci_list(transects.keys())}")

    locus_names, distance, stacked = _stack_for_hierarchical(
        transects, covariate_attr="distance",
        count_attrs=["log_gl0", "log_gl1", "log_gl2"], kind_label="geographic")
    log_gl0, log_gl1, log_gl2 = stacked["log_gl0"], stacked["log_gl1"], stacked["log_gl2"]
    print(f"  All {len(locus_names)} loci share {len(distance)} individuals -- ready for "
          f"hierarchical fitting.")

    print(f"\nFitting hierarchically ({args.num_chains} chains, {args.num_warmup} warmup, "
          f"{args.num_samples} samples) -- this can take a while for many loci ...")
    t0 = time.time()
    result = fit_hierarchical_batch_gl(
        distance, log_gl0, log_gl1, log_gl2, locus_names=locus_names,
        n_edge_individuals=args.n_edge_individuals, sigma_center_scale=args.sigma_center_scale,
        mu_log_width_loc=args.mu_log_width_loc, mu_log_width_scale=args.mu_log_width_scale,
        sigma_log_width_scale=args.sigma_log_width_scale,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, seed=args.seed,
    )
    elapsed = time.time() - t0
    report = check(result.idata)
    print(f"Done in {elapsed:.1f}s.\n\n{result.summary()}\n\n{report.summary()}")
    if not report.ok:
        print("\nNOTE: convergence diagnostics flagged possible issues (see above). "
              "Try tightening --sigma-center-scale/--sigma-log-width-scale, or more "
              "--num-samples.", file=sys.stderr)

    pmin_all, pmax_all = _derive_pmin_pmax_gl(distance, log_gl0, log_gl1, log_gl2,
                                               n_edge_individuals=args.n_edge_individuals)

    import pandas as pd
    pop_df = result.population_summary()
    pop_df.to_csv(out_dir / f"{prefix}_hierarchical_population_summary.csv")
    locus_df = result.locus_summary()
    locus_df["pmin_fixed"] = pmin_all
    locus_df["pmax_fixed"] = pmax_all
    locus_df.to_csv(out_dir / f"{prefix}_hierarchical_locus_summary.csv", index=False)

    (out_dir / f"{prefix}_hierarchical_summary.txt").write_text(
        result.summary() + "\n\n" + report.summary() + "\n")

    summary_json = {
        "model": "hierarchical_geographic_cline_genotype_likelihood",
        "n_loci": len(locus_names), "n_individuals": len(distance),
        "convergence": {
            "ok": report.ok, "max_r_hat": report.rhat_max,
            "min_ess_bulk": report.ess_bulk_min, "min_ess_tail": report.ess_tail_min,
            "n_divergences": report.n_divergences, "warnings": report.warnings,
        },
        "population_level": pop_df["mean"].to_dict(),
    }
    (out_dir / f"{prefix}_hierarchical_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str))

    from hycline.models.cline import ClineParams, TailType, cline_function
    x_smooth = np.linspace(distance.min(), distance.max(), 200)
    curve_rows = []
    for i, name in enumerate(locus_names):
        row = locus_df[locus_df["locus"] == name].iloc[0]
        params = ClineParams(center=row["center_mean"], width=row["width_mean"],
                              pmin=pmin_all[i], pmax=pmax_all[i])
        y_smooth = cline_function(x_smooth, params, TailType.NONE)
        for x, y in zip(x_smooth, y_smooth):
            curve_rows.append({"locus": name, "distance": x, "fitted_freq": y})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_hierarchical_fitted_curves.csv", index=False)

    if args.plot_grid and not args.no_plot:
        import matplotlib.pyplot as plt
        curves = {locus: (g["distance"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves)
        fig.savefig(out_dir / f"{prefix}_hierarchical_overlay.png", dpi=150)
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-hierarchical-geographic"
    (out_dir / f"{prefix}_run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str))

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_hierarchical_population_summary.csv  -- population-level hyperparameters")
    print(f"  {prefix}_hierarchical_locus_summary.csv        -- one row per locus: center, width, statistics")
    print(f"  {prefix}_hierarchical_fitted_curves.csv        -- fitted curve points, every locus")
    if args.plot_grid and not args.no_plot:
        print(f"  {prefix}_hierarchical_overlay.png              -- one fitted-curve line per locus, "
              f"shared axes (the fit itself uses the full genotype-likelihood marginalization, "
              f"not a hard-called proxy)")
    print(f"  {prefix}_run_config.json                       -- the exact settings used")
    return 0


def _fit_hierarchical_genomic_gl(args, out_dir, prefix) -> int:
    from hycline.inference.mcmc_genomic_hierarchical_gl import fit_hierarchical_genomic_batch_gl
    from hycline.diagnostics.convergence import check
    from hycline.viz.plotting import plot_multi_locus_overlay

    all_data = _load_genomic_gl_multi_locus_data(args)
    print(f"  {len(all_data)} loci detected: {_format_loci_list(all_data.keys())}")

    locus_names, hybrid_index, stacked = _stack_for_hierarchical(
        all_data, covariate_attr="hybrid_index",
        count_attrs=["log_gl0", "log_gl1", "log_gl2"], kind_label="genomic")
    log_gl0, log_gl1, log_gl2 = stacked["log_gl0"], stacked["log_gl1"], stacked["log_gl2"]
    print(f"  All {len(locus_names)} loci share {len(hybrid_index)} individuals -- ready "
          f"for hierarchical fitting.")

    print(f"\nFitting hierarchically ({args.num_chains} chains, {args.num_warmup} warmup, "
          f"{args.num_samples} samples) -- this can take a while for many loci ...")
    t0 = time.time()
    result = fit_hierarchical_genomic_batch_gl(
        hybrid_index, log_gl0, log_gl1, log_gl2, locus_names=locus_names,
        mu_u_scale=args.mu_u_scale, sigma_u_scale=args.sigma_u_scale,
        mu_log_v_loc=args.mu_log_v_loc, mu_log_v_scale=args.mu_log_v_scale,
        sigma_log_v_scale=args.sigma_log_v_scale,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, seed=args.seed,
    )
    elapsed = time.time() - t0
    report = check(result.idata)
    print(f"Done in {elapsed:.1f}s.\n\n{result.summary()}\n\n{report.summary()}")
    if not report.ok:
        print("\nNOTE: convergence diagnostics flagged possible issues (see above). "
              "Try tightening --sigma-u-scale/--sigma-log-v-scale, or more "
              "--num-samples.", file=sys.stderr)

    import pandas as pd
    pop_df = result.population_summary()
    pop_df.to_csv(out_dir / f"{prefix}_hierarchical_population_summary.csv")
    locus_df = result.locus_summary()
    locus_df.to_csv(out_dir / f"{prefix}_hierarchical_locus_summary.csv", index=False)

    (out_dir / f"{prefix}_hierarchical_summary.txt").write_text(
        result.summary() + "\n\n" + report.summary() + "\n")

    summary_json = {
        "model": "hierarchical_genomic_cline_genotype_likelihood",
        "n_loci": len(locus_names), "n_individuals": len(hybrid_index),
        "convergence": {
            "ok": report.ok, "max_r_hat": report.rhat_max,
            "min_ess_bulk": report.ess_bulk_min, "min_ess_tail": report.ess_tail_min,
            "n_divergences": report.n_divergences, "warnings": report.warnings,
        },
        "population_level": pop_df["mean"].to_dict(),
    }
    (out_dir / f"{prefix}_hierarchical_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str))

    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
    h_smooth = np.linspace(0.001, 0.999, 200)
    curve_rows = []
    for name in locus_names:
        row = locus_df[locus_df["locus"] == name].iloc[0]
        params = GenomicClineParams(u=row["u_mean"], v=row["v_mean"])
        phi_smooth = genomic_cline_function(h_smooth, params)
        for h, phi in zip(h_smooth, phi_smooth):
            curve_rows.append({"locus": name, "hybrid_index": h, "fitted_freq": phi})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_hierarchical_fitted_curves.csv", index=False)

    if args.plot_grid and not args.no_plot:
        import matplotlib.pyplot as plt
        curves = {locus: (g["hybrid_index"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves, xlabel="Hybrid index",
                                        ylabel="Locus ancestry (focal-allele frequency)")
        fig.savefig(out_dir / f"{prefix}_hierarchical_overlay.png", dpi=150)
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-hierarchical-genomic"
    (out_dir / f"{prefix}_run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str))

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_hierarchical_population_summary.csv  -- population-level hyperparameters")
    print(f"  {prefix}_hierarchical_locus_summary.csv        -- one row per locus: u, v, statistics")
    print(f"  {prefix}_hierarchical_fitted_curves.csv        -- fitted curve points, every locus")
    if args.plot_grid and not args.no_plot:
        print(f"  {prefix}_hierarchical_overlay.png              -- one fitted-curve line per locus, "
              f"shared axes (the fit itself uses the full genotype-likelihood marginalization, "
              f"not a hard-called proxy)")
    print(f"  {prefix}_run_config.json                       -- the exact settings used")
    return 0


def cmd_coupling_geographic(args) -> int:
    from hycline.analysis.coupling import geographic_cline_variance

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    print(f"Reading locus summary from '{args.input}' ...")
    df = _load_locus_summary_csv(args.input, args.locus_col)
    width_col = _resolve_summary_col(df, args.width_col, ["width_mean", "width"],
                                      args.input, "--width-col")
    print(f"  {len(df)} loci, using column '{width_col}' for cline width.")

    result = geographic_cline_variance(df[width_col].to_numpy())
    print(f"\n{result.summary()}")

    summary_json = {
        "model": "geographic_coupling_analysis", "n_loci": result.n_loci,
        "mean_slope": result.mean_slope, "sd_slope": result.sd_slope,
        "cv_slope": result.cv_slope, "width_col_used": width_col,
    }
    (out_dir / f"{prefix}_coupling_geographic_summary.json").write_text(
        json.dumps(summary_json, indent=2))
    (out_dir / f"{prefix}_coupling_geographic_summary.txt").write_text(result.summary() + "\n")

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_coupling_geographic_summary.txt/.json")
    return 0


def cmd_coupling_genomic(args) -> int:
    from hycline.analysis.coupling import genomic_cline_variance

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    print(f"Reading locus summary from '{args.input}' ...")
    df = _load_locus_summary_csv(args.input, args.locus_col)
    u_col = _resolve_summary_col(df, args.u_col, ["u_mean", "u"], args.input, "--u-col")
    v_col = _resolve_summary_col(df, args.v_col, ["v_mean", "v"], args.input, "--v-col")
    print(f"  {len(df)} loci, using columns '{u_col}'/'{v_col}' for u/v.")

    result = genomic_cline_variance(df[u_col].to_numpy(), df[v_col].to_numpy(),
                                     estimate_theta=not args.no_theta)
    print(f"\n{result.summary()}")

    summary_json = {
        "model": "genomic_coupling_analysis", "n_loci": result.n_loci,
        "sd_logit_center": result.sd_logit_center, "sd_log_v": result.sd_log_v,
        "theta_estimate": result.theta_estimate,
        "within_calibration_range": result.within_calibration_range,
        "u_col_used": u_col, "v_col_used": v_col,
    }
    (out_dir / f"{prefix}_coupling_genomic_summary.json").write_text(
        json.dumps(summary_json, indent=2))
    (out_dir / f"{prefix}_coupling_genomic_summary.txt").write_text(result.summary() + "\n")

    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_coupling_genomic_summary.txt/.json")
    return 0


def _fit_genomic_gl(args, out_dir, prefix) -> int:
    from hycline.inference.mle_genotype_likelihood import fit_mle_genomic_gl
    from hycline.models.genomic_cline import genomic_cline_function
    from hycline.viz.plotting import plot_multi_locus_overlay

    all_data = _load_genomic_gl_multi_locus_data(args)
    print(f"  {len(all_data)} loci detected: {_format_loci_list(all_data.keys())}")

    import pandas as pd
    summary_rows = []
    curve_rows = []
    failures = {}
    h_smooth = np.linspace(0.001, 0.999, 200)

    for locus_name, data in all_data.items():
        print(f"\n--- Locus '{locus_name}' ({len(data)} individuals) ---")
        t0 = time.time()
        try:
            fit = fit_mle_genomic_gl(data)
        except Exception as e:
            print(f"  FAILED to fit: {e}")
            failures[locus_name] = str(e)
            continue
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. {fit.summary()}")

        phi_smooth = genomic_cline_function(h_smooth, fit.params)
        for h, phi in zip(h_smooth, phi_smooth):
            curve_rows.append({"locus": locus_name, "hybrid_index": h, "fitted_freq": phi})

        row = {
            "locus": locus_name, "n_obs": fit.n_obs, "converged": fit.success,
            "log_likelihood": fit.log_likelihood, "aicc": fit.aicc,
            "u": fit.params.u, "v": fit.params.v,
        }
        if fit.std_errors is not None:
            for i, name in enumerate(["u", "v"]):
                if np.isfinite(fit.std_errors[i]):
                    row[f"{name}_se"] = float(fit.std_errors[i])
        summary_rows.append(row)

    if not summary_rows:
        print("\nERROR: no loci fit successfully.", file=sys.stderr)
        return 1

    pd.DataFrame(summary_rows).to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_fitted_curves.csv", index=False)

    if failures:
        (out_dir / f"{prefix}_failures.json").write_text(json.dumps(failures, indent=2))

    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        import matplotlib.pyplot as plt
        curves = {locus: (g["hybrid_index"].to_numpy(), g["fitted_freq"].to_numpy())
                  for locus, g in curve_df.groupby("locus", sort=False)}
        fig = plot_multi_locus_overlay(curves, xlabel="Hybrid index",
                                        ylabel="Locus ancestry (focal-allele frequency)")
        fig.savefig(out_dir / f"{prefix}_overlay.png", dpi=150)
        plt.close(fig)

    run_config = {k: v for k, v in vars(args).items()}
    run_config["hycline_command"] = "fit-genomic"
    (out_dir / f"{prefix}_run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print(f"\n{len(summary_rows)}/{len(all_data)} loci fit successfully.")
    if failures:
        print(f"Failed loci: {list(failures.keys())} (see {prefix}_failures.json)")
    print(f"\nAll output written to: {out_dir.resolve()}")
    print(f"  {prefix}_summary.csv        -- one row per locus: u, v, statistics")
    print(f"  {prefix}_fitted_curves.csv  -- fitted curve points, every locus")
    if args.plot_grid and not args.no_plot and len(curve_df) > 0:
        print(f"  {prefix}_overlay.png        -- one fitted-curve line per locus, shared axes")
    print(f"  {prefix}_run_config.json    -- the exact settings used")
    return 0


def cmd_estimate_hybrid_index(args) -> int:
    import pandas as pd
    from hycline.data.hybrid_index import estimate_hybrid_index

    mode1 = args.population_col is not None
    mode2 = args.parental_freq_csv is not None
    if mode1 and mode2:
        print("\nERROR: --population-col and --parental-freq-csv are mutually exclusive "
              "-- give exactly one (see --help for the two modes).\n", file=sys.stderr)
        return 2
    if not mode1 and not mode2:
        print("\nERROR: give either --population-col (with --pop0-label/--pop1-label) "
              "or --parental-freq-csv to supply parental allele frequencies (see --help "
              "for the two modes).\n", file=sys.stderr)
        return 2
    if mode1 and (args.pop0_label is None or args.pop1_label is None):
        print("\nERROR: --population-col requires --pop0-label and --pop1-label too.\n",
              file=sys.stderr)
        return 2
    if mode1 and args.metadata is None:
        print("\nERROR: --population-col (Mode 1) requires --metadata to supply "
              "population labels -- a VCF/beagle file has no room for a population "
              "column of its own.\n", file=sys.stderr)
        return 2

    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt is None:
        fmt = "beagle" if args.input.endswith((".beagle", ".beagle.gz")) else "vcf"
        print(f"Detected input format: {fmt} (from file extension; use --format to override)")

    print(f"Parsing '{args.input}' ...")
    if fmt == "vcf":
        from hycline.data.vcf import read_vcf
        try:
            data = read_vcf(args.input, max_variants=args.max_variants)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        _report_vcf_parse_summary(data, extract_pl=False)
        all_locus_names = list(data.locus_names)
        sample_names = list(data.sample_names)
        dosage_by_locus = data.dosage  # (n_loci, n_samples)
    else:
        from hycline.data.beagle import read_beagle, _beagle_expected_dosage
        try:
            data = read_beagle(args.input)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        print(f"  {len(data)} markers parsed, {len(data.sample_names)} samples.")
        all_locus_names = list(data.marker_names)
        sample_names = list(data.sample_names)
        dosage_by_locus = _beagle_expected_dosage(data.gl0, data.gl1, data.gl2)

    if args.loci:
        loci = [l.strip() for l in args.loci.split(",")]
        missing = [l for l in loci if l not in all_locus_names]
        if missing:
            print(f"\nERROR: requested loci {missing} not found in '{args.input}'.\n",
                  file=sys.stderr)
            return 2
    else:
        loci = all_locus_names
    print(f"  {len(loci)} loci detected: {_format_loci_list(loci)}")

    locus_idx = [all_locus_names.index(l) for l in loci]
    genotypes = dosage_by_locus[locus_idx, :].T  # (n_samples, n_loci)

    if mode1:
        metadata_df = pd.read_csv(args.metadata)
        from hycline.data.filter_aims import match_samples_to_populations
        try:
            pop0_mask, pop1_mask = match_samples_to_populations(
                sample_names, metadata_df, args.sample_col, args.population_col,
                args.pop0_label, args.pop1_label)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        print(f"  Parental population 0 ('{args.pop0_label}'): {pop0_mask.sum()} reference individuals")
        print(f"  Parental population 1 ('{args.pop1_label}'): {pop1_mask.sum()} reference individuals")

        parental_freq_pop0 = np.nanmean(genotypes[pop0_mask], axis=0) / args.ploidy
        parental_freq_pop1 = np.nanmean(genotypes[pop1_mask], axis=0) / args.ploidy
    else:
        print(f"Reading precomputed parental frequencies from '{args.parental_freq_csv}' ...")
        if not Path(args.parental_freq_csv).exists():
            print(f"\nERROR: --parental-freq-csv file not found: "
                  f"'{args.parental_freq_csv}'\n", file=sys.stderr)
            return 2
        freq_df = pd.read_csv(args.parental_freq_csv)
        for col in (args.freq_locus_col, args.freq_pop0_col, args.freq_pop1_col):
            if col not in freq_df.columns:
                print(f"\nERROR: column '{col}' not found in '{args.parental_freq_csv}'. "
                      f"Columns present: {', '.join(freq_df.columns)}\n", file=sys.stderr)
                return 2

        freq_by_locus = freq_df.set_index(args.freq_locus_col)
        missing_loci = [l for l in loci if l not in freq_by_locus.index]
        if missing_loci:
            print(f"\nERROR: {args.parental_freq_csv} has no row for locus(-i) "
                  f"{missing_loci} -- every locus in --input needs a matching "
                  f"{args.freq_locus_col} entry in --parental-freq-csv.\n", file=sys.stderr)
            return 2

        parental_freq_pop0 = freq_by_locus.loc[loci, args.freq_pop0_col].to_numpy(dtype=float)
        parental_freq_pop1 = freq_by_locus.loc[loci, args.freq_pop1_col].to_numpy(dtype=float)
        print(f"  Matched precomputed frequencies for all {len(loci)} loci.")

    try:
        h = estimate_hybrid_index(
            genotypes, ploidy=args.ploidy,
            parental_freq_pop0=parental_freq_pop0, parental_freq_pop1=parental_freq_pop1,
            weight_by_differentiation=args.weight_by_differentiation,
        )
    except ValueError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2

    out_df = pd.DataFrame({"individual": sample_names, "hybrid_index": h})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nHybrid index estimated for {len(out_df)} individuals.")
    print(f"  Range: [{h.min():.3f}, {h.max():.3f}]  (parental reference individuals should "
          f"land near 0 or 1 -- a useful sanity check on the result)")
    print(f"\nWrote {out_path.resolve()}")
    print("Merge this into your metadata (e.g. via the sample-name column) to get a "
          "hybrid_index column any genomic-cline fitting command can use.")
    return 0


def _report_vcf_parse_summary(vcf_data, extract_pl):
    print(f"  {len(vcf_data)} biallelic SNPs parsed, {len(vcf_data.sample_names)} samples.")
    n_skipped = vcf_data.n_skipped_multiallelic + vcf_data.n_skipped_indel
    if n_skipped or vcf_data.n_skipped_no_gt:
        print(f"  Skipped: {vcf_data.n_skipped_multiallelic} multiallelic, "
              f"{vcf_data.n_skipped_indel} indel, {vcf_data.n_skipped_no_gt} without a "
              f"GT field.")
    if n_skipped:
        total = len(vcf_data) + n_skipped
        pct = 100 * n_skipped / total if total > 0 else 0
        print(f"  NOTE: {n_skipped} sites ({pct:.0f}% of all variant records) were "
              f"multiallelic or indels, which this converter deliberately doesn't handle "
              f"(see BACKEND_PHASE1.md section 9.4 for why -- it's a real modeling "
              f"decision, not just a parsing gap). To include biallelic SNPs hiding "
              f"inside multiallelic records, or to otherwise pre-filter, try:\n"
              f"    bcftools view -m2 -M2 -v snps your.vcf.gz -Oz -o filtered.vcf.gz\n"
              f"  then re-run this command on filtered.vcf.gz.")
    if extract_pl and vcf_data.n_loci_without_pl:
        print(f"  NOTE: {vcf_data.n_loci_without_pl} of {len(vcf_data)} loci have no PL "
              f"data for any sample -- these will be excluded from genotype-likelihood "
              f"output (they carry no usable GL information).")


def _resolve_geographic_metadata_distance(args, metadata_df, fmt):
    """Resolve the metadata DataFrame's distance column for
    `generate-geographic-input`: either the
    pre-computed `--distance-col` as-is, or a `distance` column computed
    from `--lat-col`/`--lon-col` (km, via `hycline.data.geo`) inserted
    into a copy of `metadata_df`. Returns `(metadata_df, distance_col)`.

    For VCF input (always site-aggregated), coordinates are checked for
    per-site consistency directly, before any distance is computed from
    them, the coordinate analog of the hard-count writer's own existing
    per-site distance-consistency check. Not meaningful for beagle input,
    which always stays one row per individual with no site aggregation
    at all.
    """
    lat_col = getattr(args, "lat_col", None)
    lon_col = getattr(args, "lon_col", None)
    if lat_col is None and lon_col is None:
        return metadata_df, args.distance_col
    if lat_col is None or lon_col is None:
        raise ValueError("Specify both --lat-col and --lon-col together, or neither.")

    from hycline.data.geo import (
        distance_from_reference, project_onto_transect_axis,
        check_site_coordinate_consistency,
    )

    if lat_col not in metadata_df.columns:
        raise ValueError(f"--lat-col '{lat_col}' not found in the metadata file.")
    if lon_col not in metadata_df.columns:
        raise ValueError(f"--lon-col '{lon_col}' not found in the metadata file.")

    lat = metadata_df[lat_col].to_numpy(dtype=float)
    lon = metadata_df[lon_col].to_numpy(dtype=float)

    if fmt == "vcf":
        site_col = args.site_col
        if site_col not in metadata_df.columns:
            raise ValueError(f"--site-col '{site_col}' not found in the metadata file.")
        check_site_coordinate_consistency(metadata_df[site_col].tolist(), lat, lon)

    transect_method = getattr(args, "transect_method", "reference")
    if transect_method == "reference":
        distance = distance_from_reference(
            lat, lon, getattr(args, "reference_lat", None), getattr(args, "reference_lon", None))
    elif transect_method == "pca":
        distance = project_onto_transect_axis(lat, lon)
    else:
        raise ValueError(f"Unknown --transect-method '{transect_method}'.")

    metadata_df = metadata_df.copy()
    metadata_df["distance"] = distance
    return metadata_df, "distance"


def cmd_generate_geographic_input(args) -> int:
    import pandas as pd

    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2
    if not Path(args.metadata).exists():
        print(f"\nERROR: metadata file not found: '{args.metadata}'\n", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt is None:
        fmt = "beagle" if args.input.endswith((".beagle", ".beagle.gz")) else "vcf"
        print(f"Detected input format: {fmt} (from file extension; use --format to override)")

    metadata_df = pd.read_csv(args.metadata)
    try:
        metadata_df, distance_col = _resolve_geographic_metadata_distance(args, metadata_df, fmt)
    except ValueError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "vcf":
        from hycline.data.vcf import read_vcf, write_geographic_hard_count_csv

        print(f"Parsing '{args.input}' ...")
        try:
            vcf_data = read_vcf(args.input, extract_pl=False, max_variants=args.max_variants)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        _report_vcf_parse_summary(vcf_data, extract_pl=False)

        try:
            info = write_geographic_hard_count_csv(vcf_data, metadata_df, args.sample_col,
                                                     args.site_col, distance_col,
                                                     str(out_path), ploidy=args.ploidy)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2

        n_dropped_label = "VCF"
    else:
        from hycline.data.beagle import read_beagle, write_beagle_geographic_gl_csv

        print(f"Parsing '{args.input}' ...")
        try:
            beagle_data = read_beagle(args.input)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        print(f"  {len(beagle_data)} markers parsed, {len(beagle_data.sample_names)} samples.")

        try:
            info = write_beagle_geographic_gl_csv(beagle_data, metadata_df, args.sample_col,
                                                    distance_col, str(out_path))
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2

        n_dropped_label = "beagle"

    if info["n_samples_dropped"] > 0:
        print(f"  NOTE: {info['n_samples_dropped']} {n_dropped_label} sample(s) had no "
              f"matching row in the metadata and were dropped.")

    print(f"\nWrote {out_path.resolve()}")
    if fmt == "beagle":
        print(f"  {info['n_individuals']} individuals, {info['n_loci']} loci")
        print(f"Ready for: hycline fit-geographic --input {out_path} --individual-col "
              f"{args.sample_col}")
        print(f"       or: hycline fit-hierarchical-geographic --input {out_path} "
              f"--individual-col {args.sample_col}")
    else:
        print(f"  {info['n_sites']} sites, {info['n_loci']} loci")
        print(f"Ready for: hycline fit-geographic --input {out_path}")
        print(f"       or: hycline fit-hierarchical-geographic --input {out_path} "
              f"--site-col {args.site_col}")
    return 0


def cmd_generate_genomic_input(args) -> int:
    import pandas as pd

    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2
    if not Path(args.metadata).exists():
        print(f"\nERROR: metadata file not found: '{args.metadata}'\n", file=sys.stderr)
        return 2

    fmt = args.format
    if fmt is None:
        fmt = "beagle" if args.input.endswith((".beagle", ".beagle.gz")) else "vcf"
        print(f"Detected input format: {fmt} (from file extension; use --format to override)")

    metadata_df = pd.read_csv(args.metadata)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "vcf":
        from hycline.data.vcf import read_vcf, write_genomic_hard_count_csv

        print(f"Parsing '{args.input}' ...")
        try:
            vcf_data = read_vcf(args.input, extract_pl=False, max_variants=args.max_variants)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        _report_vcf_parse_summary(vcf_data, extract_pl=False)

        try:
            info = write_genomic_hard_count_csv(vcf_data, metadata_df, args.sample_col,
                                                 args.hybrid_index_col, str(out_path),
                                                 ploidy=args.ploidy)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2

        n_dropped_label = "VCF"
    else:
        from hycline.data.beagle import read_beagle, write_beagle_genomic_gl_csv

        print(f"Parsing '{args.input}' ...")
        try:
            beagle_data = read_beagle(args.input)
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2
        print(f"  {len(beagle_data)} markers parsed, {len(beagle_data.sample_names)} samples.")

        try:
            info = write_beagle_genomic_gl_csv(beagle_data, metadata_df, args.sample_col,
                                                args.hybrid_index_col, str(out_path))
        except ValueError as e:
            print(f"\nERROR: {e}\n", file=sys.stderr)
            return 2

        n_dropped_label = "beagle"

    if info["n_samples_dropped"] > 0:
        print(f"  NOTE: {info['n_samples_dropped']} {n_dropped_label} sample(s) had no "
              f"matching row in the metadata and were dropped.")

    print(f"\nWrote {out_path.resolve()}")
    print(f"  {info['n_individuals']} individuals, {info['n_loci']} loci")
    print(f"Ready for: hycline fit-genomic --input {out_path} "
          f"--individual-col {args.sample_col}")
    print(f"       or: hycline fit-hierarchical-genomic --input {out_path} "
          f"--individual-col {args.sample_col}")
    return 0


def cmd_filter_aims(args) -> int:
    fmt = args.format
    if fmt is None:
        fmt = "beagle" if args.input.endswith((".beagle", ".beagle.gz")) else "vcf"
        print(f"Detected input format: {fmt} (from file extension; use --format to override)")

    if not Path(args.input).exists():
        print(f"\nERROR: input file not found: '{args.input}'\n", file=sys.stderr)
        return 2
    if not Path(args.metadata).exists():
        print(f"\nERROR: metadata file not found: '{args.metadata}'\n", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {fmt} data from '{args.input}' ...")
    try:
        if fmt == "beagle":
            from hycline.data.filter_aims import filter_aims_beagle
            result = filter_aims_beagle(
                args.input, args.metadata, str(out_path),
                sample_col=args.sample_col, population_col=args.population_col,
                pop1_label=args.pop1_label, pop2_label=args.pop2_label,
                low_threshold=args.low_threshold, high_threshold=args.high_threshold,
            )
        else:
            from hycline.data.filter_aims import filter_aims_vcf
            result = filter_aims_vcf(
                args.input, args.metadata, str(out_path),
                sample_col=args.sample_col, population_col=args.population_col,
                pop1_label=args.pop1_label, pop2_label=args.pop2_label,
                low_threshold=args.low_threshold, high_threshold=args.high_threshold,
            )
    except ValueError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2

    print(f"  {args.pop1_label}: {result['n_pop1_samples']} individuals; "
          f"{args.pop2_label}: {result['n_pop2_samples']} individuals")
    print(f"  {result['n_total']} loci checked, {result['n_retained']} retained "
          f"(>= {args.high_threshold} in one population and <= {args.low_threshold} in "
          f"the other, either direction)")
    if fmt == "vcf":
        skipped = (result["n_skipped_multiallelic"] + result["n_skipped_indel"]
                   + result["n_skipped_no_gt"])
        if skipped:
            print(f"  ({skipped} additional sites skipped: "
                  f"{result['n_skipped_multiallelic']} multiallelic, "
                  f"{result['n_skipped_indel']} indel, "
                  f"{result['n_skipped_no_gt']} with no GT field -- not counted "
                  f"toward loci checked above)")
    if result['n_retained'] == 0:
        print(f"\nWARNING: no loci passed the filter -- the output file has a header "
              f"but no data. Check --pop1-label/--pop2-label match your metadata "
              f"file's actual values, or consider loosening --low-threshold/"
              f"--high-threshold.\n", file=sys.stderr)

    print(f"\nWrote {out_path.resolve()}")
    if fmt == "vcf":
        print("Ready for: hycline generate-geographic-input / generate-genomic-input, "
              "using this file as --input -- or use it with any other tool that reads "
              "VCF directly.")
    else:
        print("Ready for: hycline generate-geographic-input / generate-genomic-input, "
              "using this file as --input -- or use it with any other tool that reads "
              "ANGSD's beagle format directly.")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "fit-geographic":
            return cmd_fit_geographic(args)
        elif args.command == "compare-tails":
            return cmd_compare_tails(args)
        elif args.command == "fit-trait":
            return cmd_fit_trait(args)
        elif args.command == "fit-trait-multi":
            return cmd_fit_trait_multi(args)
        elif args.command == "fit-genomic":
            return cmd_fit_genomic(args)
        elif args.command == "fit-hierarchical-geographic":
            return cmd_fit_hierarchical_geographic(args)
        elif args.command == "fit-hierarchical-genomic":
            return cmd_fit_hierarchical_genomic(args)
        elif args.command == "coupling-geographic":
            return cmd_coupling_geographic(args)
        elif args.command == "coupling-genomic":
            return cmd_coupling_genomic(args)
        elif args.command == "estimate-hybrid-index":
            return cmd_estimate_hybrid_index(args)
        elif args.command == "generate-geographic-input":
            return cmd_generate_geographic_input(args)
        elif args.command == "generate-genomic-input":
            return cmd_generate_genomic_input(args)
        elif args.command == "filter-aims":
            return cmd_filter_aims(args)
        else:
            parser.print_help()
            return 1
    except Exception:
        print("\nHyCline hit an unexpected error:\n", file=sys.stderr)
        traceback.print_exc()
        print("\nIf this looks like a bug rather than a data/input problem, "
              "please report it with the command you ran and the traceback above.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
