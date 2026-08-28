"""Coupling and concordance analysis across loci.

Phase 4, Component C. Quantifies how concordant cline parameters are
across a set of loci, connecting Barton's (1983) theoretical coupling
coefficient (theta = s/r, the ratio of total selection to recombination
among barrier loci) to the empirically observable across-locus variance
in fitted cline parameters -- following Firneno, Semenov, Dopman, Taylor,
Larson & Gompert (2023, "Quantitative Analyses of Coupling in Hybrid
Zones," Cold Spring Harb Perspect Biol 15:a041434).

See BACKEND_PHASE4.md section 9 for the full derivation, including
exactly what's a direct implementation of Firneno et al.'s published
approach versus what's approximated because their full simulation
output isn't reproducible from the published paper alone (their
simulation code lives at https://github.com/zgompert/ClineCoupling, not
re-run here).

WHY THIS COMES AFTER COMPONENTS A/B, NOT BEFORE: Firneno et al.'s own
approach is explicit that this analysis is built on hierarchical
per-locus estimates, not independent ones -- their paper states "we take
a hierarchical Bayesian approach to fitting geographic and genomic
clines. This is important because it allows us to explicitly estimate
the cline variances as parameters in our models." The functions here
accept per-locus point estimates from *any* source (hierarchical or
independent fits), but are intended to be fed Components A/B's output,
for the same reason Components A/B themselves exist: independent,
unpooled per-locus estimates conflate genuine biological variation with
estimation noise, and this analysis is directly about quantifying that
variation.

FOR SIGMA_V, AND NOW SIGMA_C TOO: prefer Component B's own jointly-
estimated hyperparameters over this module's point-estimate-based SD
when they're available. `hycline.inference.mcmc_genomic_hierarchical.
fit_hierarchical_genomic_batch(..., pool_center=True)` hierarchically
pools `logit(center) = u/v` directly (Firneno et al.'s own quantity),
reporting a genuine `sigma_center` hyperparameter in
`population_summary()` -- feed that straight into
`estimate_coupling_coefficient` as sigma_c, rather than computing
`genomic_cline_variance`'s SD-of-posterior-means approximation from
`locus_summary()`'s per-locus point estimates. `sigma_log_v` was already
available this way (`pool_center=False`, the default, already reports
it); this closes the same gap for sigma_c. See BACKEND_PHASE4.md section
9.3 for the full story of why this was a real, precisely-identified gap
rather than a hypothetical one, and section 7.4 for the model extension
itself.

A REAL SHAPE-MISMATCH CAVEAT, not a hypothetical one (Baird & Daley 2025,
read directly -- `references/`): both functions here use across-locus
variance in cline *center*/*width* (equivalently, slope) as the coupling
signal, which implicitly assumes every locus's underlying cline is
symmetric and non-stepped. The paper gives a concrete mechanism by which
that assumption, when wrong, contaminates the signal in two specific,
opposite-direction ways: a model that doesn't allow for asymmetry will
misattribute incomplete-lineage-sorting-driven asymmetry as a spurious
center/width outlier (a false positive for coupling variation that isn't
really there), while a model that doesn't allow for steppedness will
systematically overestimate width for genuinely stepped loci and thereby
miss real barrier-effect outliers (a false negative for the loci coupling
analysis exists to find). This is a real limitation of `geographic_cline_
variance`/`genomic_cline_variance` as they stand today, not fixed by
`hycline.models.cline.t_cline_function` (the T-cline, BACKEND_PHASE2.md
section 15) existing elsewhere in this project -- feeding T-cline-fitted
loci into these functions would still only extract width/slope from the
summary CSV, discarding the T-cline's own shape (`t_shape`) and asymmetry
(`t_skew`) parameters entirely, so it doesn't by itself close this gap.
Extending these functions to account for shape (e.g. treating markedly
different `t_shape`/`t_skew` values as their own signal, separate from
center/width variance) is a real, open, unimplemented extension -- see
README.md's Tier 1 backlog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Geographic clines: cline slope (beta = 4/width) variance across loci
# ---------------------------------------------------------------------------

@dataclass
class GeographicClineVariance:
    mean_slope: float
    sd_slope: float
    cv_slope: float
    n_loci: int

    def summary(self) -> str:
        return (f"Geographic cline variance across {self.n_loci} loci:\n"
                f"  mean slope (beta) = {self.mean_slope:.4f}\n"
                f"  SD slope (sigma_beta) = {self.sd_slope:.4f}\n"
                f"  CV slope = {self.cv_slope:.4f}\n"
                f"  (lower SD/CV = more concordant clines across loci = stronger coupling; "
                f"see this module's docstring -- no theta estimate is provided for geographic "
                f"clines, only genomic, since Firneno et al. 2023 calibrated theta against real "
                f"data using genomic clines only. See BACKEND_PHASE4.md section 9.)")


def geographic_cline_variance(locus_widths) -> GeographicClineVariance:
    """Compute the across-locus mean and SD of cline slope (beta = 4/width),
    Firneno et al.'s metric of cline variance for geographic clines.

    `beta = 4/width` follows directly from this project's own cline
    parameterization (`hycline.models.cline`): the logistic cline
    `p(x) = pmin + (pmax-pmin)/(1+exp(-4(x-c)/w))` has slope exactly
    `4/w` (in units of logit(p) per unit distance) at its center when
    pmin=0, pmax=1 -- the same slope definition Firneno et al. (2023) fit
    a linear model to logit allele frequency to estimate directly.

    Parameters
    ----------
    locus_widths : per-locus cline width point estimates (e.g. from
        `hycline.inference.mcmc_hierarchical.HierarchicalFitResult.
        locus_summary()`'s `width_mean` column, or from independent
        per-locus `fit_mle`/`fit_mcmc` results -- see this module's
        docstring for why hierarchical estimates are recommended).

    Returns
    -------
    GeographicClineVariance
    """
    widths = np.asarray(locus_widths, dtype=float)
    if not np.all(np.isfinite(widths)):
        raise ValueError("Cline widths must be finite (no NaN/Inf).")
    if np.any(widths <= 0):
        raise ValueError("Cline widths must be positive.")
    if len(widths) < 2:
        raise ValueError("Need at least 2 loci to compute across-locus variance.")

    slopes = 4.0 / widths
    mean_slope = float(np.mean(slopes))
    sd_slope = float(np.std(slopes, ddof=1))
    # mean_slope is always > 0 here (widths are already validated positive
    # and finite above, so 4/widths is too) -- no divide-by-zero guard needed.
    cv_slope = float(sd_slope / mean_slope)

    return GeographicClineVariance(mean_slope=mean_slope, sd_slope=sd_slope,
                                    cv_slope=cv_slope, n_loci=len(widths))


# ---------------------------------------------------------------------------
# Genomic clines: logit(center) and log(v) variance across loci, plus a
# theta estimate calibrated against Firneno et al.'s own published data
# ---------------------------------------------------------------------------

@dataclass
class GenomicClineVariance:
    sd_logit_center: float
    sd_log_v: float
    n_loci: int
    theta_estimate: Optional[float] = None
    within_calibration_range: Optional[bool] = None

    def summary(self) -> str:
        lines = [
            f"Genomic cline variance across {self.n_loci} loci:",
            f"  SD logit(center) (sigma_c) = {self.sd_logit_center:.4f}",
            f"  SD log(v) (sigma_v) = {self.sd_log_v:.4f}",
        ]
        if self.theta_estimate is not None:
            lines.append(f"  estimated coupling coefficient (theta) = {self.theta_estimate:.3f}")
            if self.within_calibration_range is False:
                lines.append("    NOTE: sigma_c and/or sigma_v fall outside the bounding box "
                              "of the 31 reference data points the calibration was fit on "
                              "(see calibration_data_range()) -- theta here is an extrapolation, "
                              "treat it with extra caution.")
            if not np.isfinite(self.theta_estimate):
                lines.append("    WARNING: theta is not a finite number (NaN/Inf) -- this "
                              "means sigma_c and/or sigma_v themselves were not finite, so "
                              "no coupling interpretation is possible. Check the raw "
                              "sigma_c/sigma_v inputs.")
            elif self.theta_estimate <= 0:
                lines.append("    WARNING: theta is a ratio of selection to recombination "
                              "(s/r) and is theoretically always positive -- a non-positive "
                              "estimate here means this (sigma_c, sigma_v) combination falls "
                              "somewhere the linear calibration doesn't behave well, not a "
                              "real negative coupling coefficient. Treat the raw sigma_c/sigma_v "
                              "values as the reliable output for this locus set, not theta. "
                              "See BACKEND_PHASE4.md section 9.")
            elif self.theta_estimate > 1:
                lines.append("    theta > 1: consistent with strong, genome-wide coupling "
                              "(a near-complete barrier to gene flow)")
            else:
                lines.append("    theta < 1: consistent with weaker/partial coupling "
                              "(loci behaving more independently)")
        lines.append("  (lower sigma_c/sigma_v = more concordant genomic clines = stronger "
                      "coupling; see this module's docstring for the theta calibration's scope "
                      "and caveats -- BACKEND_PHASE4.md section 9)")
        return "\n".join(lines)


def genomic_cline_variance(locus_u, locus_v,
                            estimate_theta: bool = True) -> GenomicClineVariance:
    """Compute the across-locus SD of logit(center) = u/v and log(v) --
    Firneno et al.'s metrics of cline variance for genomic clines -- and,
    optionally, an estimated coupling coefficient (theta).

    NOTE ON `logit(center) = u/v`: this is NOT the same quantity as this
    project's own `u` parameter (`hycline.models.genomic_cline`). Firneno
    et al. (following Bailey 2022/gghybrid's reparameterization) define
    cline center `c` such that `logit(c) = u/v` -- the hybrid-index value
    at which phi=0.5 -- which is a nonlinear function of both `u` and `v`
    together, not `u` alone. Computing `SD(u/v)` directly from per-locus
    `u`, `v` estimates (as this function does) is the correct way to
    reproduce their metric; using `SD(u)` instead would be a different,
    incorrect quantity. See BACKEND_PHASE4.md section 9 for the full
    explanation of why this distinction matters.

    Parameters
    ----------
    locus_u, locus_v : per-locus u, v point estimates (e.g. from
        `hycline.inference.mcmc_genomic_hierarchical.
        HierarchicalGenomicFitResult.locus_summary()`'s `u_mean`/`v_mean`
        columns, or independent per-locus `fit_genomic_cline_mle` results)
    estimate_theta : if True (default), also compute an estimated coupling
        coefficient using the calibration in `estimate_coupling_coefficient`
        below. Set False to get only the raw variance metrics, with no
        attempt at converting them to theta.

    Returns
    -------
    GenomicClineVariance
    """
    u = np.asarray(locus_u, dtype=float)
    v = np.asarray(locus_v, dtype=float)
    if len(u) != len(v):
        raise ValueError("locus_u and locus_v must have the same length.")
    if len(u) < 2:
        raise ValueError("Need at least 2 loci to compute across-locus variance.")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("locus_u and locus_v must be finite (no NaN/Inf) for every locus.")
    if np.any(v <= 0):
        raise ValueError("v must be positive for every locus.")

    logit_center = u / v
    log_v = np.log(v)
    sd_logit_center = float(np.std(logit_center, ddof=1))
    sd_log_v = float(np.std(log_v, ddof=1))

    theta = None
    in_range = None
    if estimate_theta:
        theta = estimate_coupling_coefficient(sd_logit_center, sd_log_v)
        r = calibration_data_range()
        in_range = (r["sigma_c_min"] <= sd_logit_center <= r["sigma_c_max"]
                    and r["sigma_v_min"] <= sd_log_v <= r["sigma_v_max"])

    return GenomicClineVariance(sd_logit_center=sd_logit_center, sd_log_v=sd_log_v,
                                 n_loci=len(u), theta_estimate=theta,
                                 within_calibration_range=in_range)


# Empirical (sigma_c, sigma_v, theta) triples from 25 published hybrid zone
# studies (some with multiple transects, denoted _1/_2/_3), reanalyzed with
# a hierarchical Bayesian genomic cline model in Firneno et al. (2023),
# Table 1. Reproduced here (with attribution) specifically to re-derive a
# usable raw-scale calibration equation -- see
# `_fit_theta_calibration_from_firneno2023_table1` below for why this
# works and what it does and doesn't establish.
FIRNENO_2023_TABLE1 = [
    # (organism, sigma_c, sigma_v, theta)
    ("Agalychnis", 1.42, 0.56, 0.12),
    ("Alouatta", 1.02, 0.23, 0.65),
    ("Coenonympha", 1.11, 0.31, 0.44),
    ("Corvus", 0.86, 0.40, 0.14),
    ("Crotalus", 0.91, 0.33, 0.37),
    ("Encelia", 0.38, 0.31, 0.41),
    ("Fundulus_1", 0.45, 0.12, 1.30),
    ("Fundulus_2", 0.33, 0.08, 1.58),
    ("Gryllus_1", 0.81, 0.24, 0.67),
    ("Gryllus_2", 0.48, 0.15, 1.15),
    ("Hirundo", 0.53, 0.31, 0.42),
    ("Iris", 1.16, 0.19, 0.71),
    ("Lissotriton_1", 0.35, 0.14, 1.26),
    ("Lissotriton_2", 0.97, 0.14, 0.92),
    ("Lycaeides", 0.94, 0.35, 0.31),
    ("Mus_1", 0.69, 0.16, 1.00),
    ("Mus_2", 0.70, 0.14, 1.07),
    ("Mus_3", 1.01, 0.23, 0.66),
    ("Mytilus", 0.29, 0.27, 0.62),
    ("Nematocharax", 0.80, 0.41, 0.08),
    ("Neotoma", 0.52, 0.26, 0.64),
    ("Oleria", 1.73, 0.50, 0.42),
    ("Papilio", 0.29, 0.21, 0.93),
    ("Papio", 0.84, 0.41, 0.10),
    ("Picea", 0.67, 0.22, 0.77),
    ("Pinus", 1.10, 0.37, 0.29),
    ("Poecile_1", 0.94, 0.40, 0.17),
    ("Poecile_2", 0.82, 0.38, 0.19),
    ("Sceloporus", 1.09, 0.21, 0.69),
    ("Sternotherus", 0.57, 0.28, 0.55),
    ("Yucca", 0.64, 0.15, 1.06),
]

_theta_calibration_coefs = None  # lazily computed, cached


def _fit_theta_calibration_from_firneno2023_table1() -> np.ndarray:
    """Re-derive raw-scale linear-model coefficients for predicting theta
    from (sigma_c, sigma_v), by fitting `theta ~ sigma_c + sigma_v +
    sigma_c:sigma_v` directly to Firneno et al.'s own published empirical
    results (`FIRNENO_2023_TABLE1`).

    WHY THIS WORKS DESPITE NOT HAVING THEIR ORIGINAL SIMULATION DATA: the
    paper states their empirical theta estimates were computed by
    "estimat[ing] coupling coefficients based on the values of sigma_c
    and sigma_v and the parameterized linear model ... from the simulated
    data sets" -- i.e. Table 1's theta column is itself the *output* of
    their fitted linear model, applied to their own sigma_c/sigma_v
    estimates. Fitting the same linear model form to these
    (sigma_c, sigma_v, theta) triples therefore isn't really a fresh
    regression on 31 noisy data points -- it's solving for what their
    original coefficients must have been, since we're given the model's
    own outputs. This is confirmed empirically: the fit achieves
    R^2 = 0.9999 on this dataset (not exactly 1.0, consistent with
    reading values off a table rounded to 2 decimals), which would be a
    wildly implausible fit for a genuinely noisy 31-point regression --
    Firneno et al.'s own regression on their much larger *simulated*
    dataset (thousands of points) explained only 52.3% of the variance in
    theta. That gap is exactly what you'd expect if this function is
    recovering their applied model rather than fitting a new one.

    GENUINE OUT-OF-SAMPLE VALIDATION, not just the fit above: Table S4 of
    the paper's own Supplemental Material reports (sigma_c, sigma_v,
    theta) separately for autosomes and sex chromosomes for 8 of the 25
    hybrid zones -- 16 real, published data points never used to fit this
    calibration. Checked against all 16 with no refitting: mean absolute
    error 0.008, max 0.017 (`tests/test_coupling.py::
    test_calibration_holds_out_of_sample_on_table_s4`). One of those 16
    falls outside this calibration's bounding box entirely and still
    predicted almost exactly -- real evidence this recovers the true
    underlying relationship rather than overfitting the 31-point table.

    CAVEATS, stated plainly:
    - The model form and scale conventions are now confirmed exactly, not
      inferred: the paper's own text states the theta-predicting model
      used for Table 1 is linear with an interaction term (not the
      polynomial regression used elsewhere in the paper for exploratory
      figures), and confirms the SD is computed on natural-log(v) and
      logit(center) scale -- both checked directly against the paper's
      full text and its Supplemental Material (`FIRNENO_2023_TABLE1`
      itself was also confirmed digit-for-digit against the published
      Table 1, not the informal GitHub README summary, which has at
      least one transcription error the reproduced table here was never
      affected by).
    - What's still not available is Firneno et al.'s *original*
      unstandardized regression coefficients (fit on their ~1140
      simulated datasets, not published in raw form) -- what's
      implemented here is confirmed to reproduce their *applied model's
      outputs* faithfully, on both the original 31 points and 16 new
      held-out ones, which is the practically relevant guarantee, but is
      a distinct claim from "these are their exact original coefficients."
    - The calibration reflects one specific simulation study's assumptions
      (a stepping-stone model with specific migration rates, deme sizes,
      and genetic architecture -- see BACKEND_PHASE4.md section 9) and
      Firneno et al. themselves caution that their theta estimates "should
      be taken as estimates given the specific simulation conditions and
      linear model used."
    - **This isn't a hedge -- the paper's own robustness check quantifies
      it, and the number is large.** Firneno et al. re-ran their
      simulations at a lower migration rate (m=0.05, vs. m=0.1/0.2 for
      the calibration this module implements) and found the sigma_c/
      sigma_v-to-theta relationship measurably weaker there on its own
      (linear r^2 dropped from 0.523 to 0.415), and, more strikingly,
      that a model *fit* at m=0.1/0.2 essentially failed to predict theta
      for the m=0.05 simulations at all when cross-applied (random-forest
      out-of-bag variance explained ~0%; the linear-model cross-
      application did little better, explaining only 22.5% of the
      variance in the true m=0.05 theta values) -- despite the same
      m=0.1-vs-m=0.2 cross-application working well in both directions.
      In practice: this calibration should be treated as most reliable
      for hybrid zones with migration/dispersal on a scale broadly
      similar to the reference data set's own simulated hybrid zones
      (see `calibration_data_range()`), not assumed to generalize across
      migration regimes generally -- a genomic cline shaped by unusually
      low gene flow between demes is exactly the case this robustness
      check found the calibration degrades on. Genetic map size has a
      smaller, still-quantified version of the same sensitivity: across
      0.5/1/2-Morgan simulated chromosomes, sigma_c/sigma_v alone
      explained only 37% of theta's variance (vs. 52.3% in the main
      calibration), rising to 53.4% only once an explicit map-size
      interaction term was added -- a term this module's calibration does
      not include, since it is fit purely from Table 1's published
      (sigma_c, sigma_v, theta) triples with no map-size column.
    """
    global _theta_calibration_coefs
    if _theta_calibration_coefs is None:
        data = np.array([(row[1], row[2], row[3]) for row in FIRNENO_2023_TABLE1])
        sc, sv, theta = data[:, 0], data[:, 1], data[:, 2]
        X = np.column_stack([np.ones(len(sc)), sc, sv, sc * sv])
        coefs, _, _, _ = np.linalg.lstsq(X, theta, rcond=None)
        _theta_calibration_coefs = coefs
    return _theta_calibration_coefs


def estimate_coupling_coefficient(sd_logit_center: float, sd_log_v: float) -> float:
    """Estimate Barton's coupling coefficient (theta) from the across-locus
    SD of genomic cline center (logit scale) and slope (log scale), using
    a linear calibration re-derived from Firneno et al. (2023)'s own
    published empirical results.

    See `_fit_theta_calibration_from_firneno2023_table1`'s docstring for
    the full methodology and caveats -- this is a practical approximation
    of a published simulation-calibrated relationship, not a fresh
    theoretical derivation, and should be reported alongside the raw
    sigma_c/sigma_v values (which are on firmer ground) rather than
    treated as a precise estimate on its own.

    Can return a negative value for some (sigma_c, sigma_v) combinations,
    even within the observed range of the calibration data -- theta is
    theoretically always positive (a ratio of two positive quantities),
    so a non-positive result signals the linear model doesn't behave well
    for that particular combination, not a genuine negative coupling
    coefficient. `calibration_data_range()` reports the bounding box of
    the 31 reference data points this was fit on, as a rough guide to
    where the calibration is best supported.
    """
    intercept, b_sc, b_sv, b_interaction = _fit_theta_calibration_from_firneno2023_table1()
    return float(intercept + b_sc * sd_logit_center + b_sv * sd_log_v
                 + b_interaction * sd_logit_center * sd_log_v)


def calibration_data_range() -> dict:
    """Bounding box of (sigma_c, sigma_v) values in the 31-point reference
    dataset (`FIRNENO_2023_TABLE1`) the theta calibration was fit on --
    a rough guide to where `estimate_coupling_coefficient` is best
    supported by real data, not a rigorous validity region (the true
    supported region is the 2D convex hull of the actual points, not this
    simpler bounding box)."""
    data = np.array([(row[1], row[2]) for row in FIRNENO_2023_TABLE1])
    return {
        "sigma_c_min": float(data[:, 0].min()), "sigma_c_max": float(data[:, 0].max()),
        "sigma_v_min": float(data[:, 1].min()), "sigma_v_max": float(data[:, 1].max()),
    }


# ---------------------------------------------------------------------------
# Candidate-locus flagging: adapts Fitzpatrick (2013)'s multivariate
# outlier-detection method (his own D^2 vs. chi-squared(df=2) approach,
# read directly from his "Outlier detection" section) to both geographic
# and genomic hierarchical fits with one shared statistical core.
#
# CATEGORY NAMES NAME THE STATISTICAL PATTERN, NOT A CONFIRMED MECHANISM.
# An earlier design used mechanism-flavored names ("barrier"/"speciation
# candidate", "adaptive_introgression") for two of the four categories
# while the third ("elevated_gene_flow") was already pattern-based --an
# inconsistency, and a real risk: a column of confident-sounding category
# labels in a CSV is exactly the kind of thing that gets copy-pasted into
# a results table without the caveat text attached to it. All three
# outlier categories are now named after the observed pattern alone
# (restricted / elevated / asymmetric gene flow); which evolutionary
# mechanism a given pattern is *most consistent with*, per Fitzpatrick's
# own single-mechanism simulations, lives in this module's docstring and
# in CANDIDATE_FLAGGING_CAVEAT below, not in the machine-readable value:
#   - restricted_gene_flow (steep rate, not displaced): Fitzpatrick's
#     heterozygote-disadvantage simulation produced exactly this signature
#     ("cleanest, least ambiguous" of his single-mechanism cases) -- the
#     pattern a genuine reproductive-isolation barrier locus is expected
#     to show, resisting introgression from both directions symmetrically.
#   - elevated_gene_flow (shallow rate, not displaced): Fitzpatrick's
#     heterozygote-advantage simulation produced the mirror-image "inside-
#     out" signature -- symmetric excess porosity, consistent with hybrid
#     vigor at this locus specifically, but NOT reported as adaptive
#     introgression, since there is no directional bias signal
#     establishing which parent's allele is favored. A second, unrelated,
#     equally real cause of this same pattern: recombination-rate
#     heterogeneity (Payseur 2010) -- a neutral locus sitting in a region
#     of elevated local recombination will show reduced linkage
#     disequilibrium with nearby real barrier loci, mimicking this
#     signature with zero heterozygote advantage at the locus itself.
#   - asymmetric_gene_flow (center/bias displaced, rate not steep):
#     Fitzpatrick's directional/environmental-selection simulation
#     produced displacement without steepening -- the closest single-
#     mechanism match to one-directional, selection-driven introgression.
#     But displacement alone is also exactly what incomplete lineage
#     sorting produces in recently diverged taxa (Payseur 2010) -- this
#     category is a *candidate* for adaptive introgression, not a finding
#     of it.
#   - ambiguous (both dimensions triggered, or a joint-but-not-marginal
#     deviation): not one of Fitzpatrick's simulated single-mechanism
#     cases. Real, named scenarios that land here, not just "unclear":
#     multiple physically linked selected loci with different individual
#     effects (a true barrier locus in LD with a separately-favored one);
#     a single locus under two selection pressures simultaneously
#     (reduced heterozygote fitness AND a habitat-driven advantage); and,
#     importantly, drift acting on top of an underdominant/DMI locus,
#     which Jofre & Rosenthal (2021) show can narrow a cline's width AND
#     shift its center as a statistical artifact ("spurious centre
#     outliers") with no combined-selection scenario behind it at all --
#     the single strongest reason this category is a real "reported, not
#     force-categorized" bucket rather than always resolved one way.
#
# See README.md's Tier 1 backlog for the full design discussion this was
# built from, and BACKEND_PHASE4.md section 15 for the implementation
# notes (statistic derivation, Bonferroni-correction convention, and the
# geographic/genomic sign-convention handling for "steep").
# ---------------------------------------------------------------------------

CANDIDATE_FLAGGING_CAVEAT = """\
CAVEATS (read before interpreting these categories as findings):
1. Cline-shape signatures alone cannot identify *which* evolutionary
   mechanism produced an outlier. Quoting Gompert, Parchman & Buerkle
   (2012) on their own genomic-cline model: "previous attempts (including
   our own) to associate a specific form of selection... with a locus
   based on the sign or magnitude of cline parameters were perhaps overly
   simplistic and naive." These categories narrow *what pattern* was
   observed, not *why* it occurred.
2. This method is weak at detecting epistatic (Dobzhansky-Muller-style)
   incompatibilities specifically -- Gompert & Buerkle (2011)'s own
   simulations found a 0.81-0.95 false discovery rate for pairwise
   epistasis (vs. <0.1 for underdominance/directional selection), and
   Payseur (2010) confirms this is not genomic-cline-specific.
3. Genetic drift alone generates real cross-locus variance, and
   incomplete lineage sorting is a larger confound than drift for
   recently diverged species or large effective population sizes
   (Payseur 2010) -- named explicitly, not left implicit under "drift."
4. Marker ascertainment bias: fixed-difference markers (exactly what
   `filter-aims` selects for) systematically bias toward underestimating
   introgression, and loci that actually carry reproductive-isolation
   genes are, paradoxically, more likely to have fixed differences in the
   first place -- making reduced introgression there specifically harder
   to detect by comparison to other loci (Payseur 2010).
5. A false-negative risk, not just false-positive: truly selected loci
   need not lie in the most extreme tail of the empirical outlier
   distribution (Kelley et al. 2006; Teshima et al. 2006, via Payseur
   2010).
6. Drift acting on an underdominant or Dobzhansky-Muller-incompatible
   locus can narrow a cline's width *and* shift its center in a way that
   specifically mimics a combined restricted-plus-asymmetric signal
   ("spurious centre outliers," Jofre & Rosenthal 2021) -- not generic
   noise, a false signal shaped like the thing being searched for, and
   the specific reason the 'ambiguous' category exists rather than
   forcing such loci into one of the other three.
7. Outlier analysis on a single hybrid zone transect is only reliable
   within a fairly narrow demographic window (Jofre & Rosenthal 2021:
   their own false-positive rates ranged from 1.35% in the best case to
   98.9% in the worst). Cross-validating candidates against independent
   replicate transects of the same species pair, where available, is the
   strongest available check, and cannot be automated in general.
"""


@dataclass
class LocusCandidateFlag:
    locus: str
    d2: float
    chi2_quantile: float
    is_outlier: bool
    center_outlier: bool
    rate_outlier: bool
    rate_direction: Optional[str]  # "steep" / "shallow" / None (not a rate outlier)
    category: Optional[str]  # "restricted_gene_flow" / "elevated_gene_flow" /
                              # "asymmetric_gene_flow" / "ambiguous" / None (not flagged)


@dataclass
class CandidateFlagResult:
    flags: List[LocusCandidateFlag]
    alpha: float
    n_loci: int

    def summary(self) -> str:
        n_flagged = sum(1 for f in self.flags if f.is_outlier)
        counts: dict = {}
        for f in self.flags:
            if f.category is not None:
                counts[f.category] = counts.get(f.category, 0) + 1
        lines = [f"Candidate-locus flagging across {self.n_loci} loci "
                 f"(Bonferroni-adjusted alpha={self.alpha}/{self.n_loci} per test):",
                 f"  {n_flagged}/{self.n_loci} loci flagged as statistical outliers"]
        for cat in ("restricted_gene_flow", "elevated_gene_flow",
                    "asymmetric_gene_flow", "ambiguous"):
            if cat in counts:
                lines.append(f"    {cat}: {counts[cat]}")
        lines.append("")
        lines.append(CANDIDATE_FLAGGING_CAVEAT)
        return "\n".join(lines)


def _flag_from_z_scores(locus_names, z_center, z_rate_steep_positive,
                         alpha: float) -> CandidateFlagResult:
    """Shared statistical core for both cline types: a diagonal-covariance
    squared Mahalanobis distance D^2 = z_center^2 + z_rate^2, treated as a
    chi-squared(df=2) random variable under the null (Fitzpatrick 2013's
    own approach, read directly from his "Outlier detection" section --
    the sum of two independent standardized-normal squares), with a
    Bonferroni-adjusted critical value (alpha/n_loci, matching Fitzpatrick's
    own convention of one correction factor for n tests, not a further
    split across the two parameters).

    Which single dimension individually exceeds its own per-parameter
    threshold (same alpha/n_loci correction, applied as a two-sided
    Normal test) determines `center_outlier`/`rate_outlier` -- D^2 alone
    determines `is_outlier`. A locus can be a joint (D^2) outlier without
    either marginal z exceeding its own stricter threshold; this lands in
    'ambiguous' rather than silently going uncategorized, since it's a
    real joint deviation the two single-parameter tests aren't positioned
    to explain individually.

    `z_rate_steep_positive` must already be signed so that positive means
    steeper-than-population-average and negative means shallower -- this
    helper is direction-agnostic; callers own the sign convention (see
    `flag_geographic_candidates`/`flag_genomic_candidates`, where the two
    cline types define "steep" in opposite directions relative to their
    own raw parameter).
    """
    locus_names = list(locus_names)
    z_center = np.asarray(z_center, dtype=float)
    z_rate_steep_positive = np.asarray(z_rate_steep_positive, dtype=float)
    n = len(locus_names)
    if n < 2:
        raise ValueError("Need at least 2 loci to flag candidates.")
    if len(z_center) != n or len(z_rate_steep_positive) != n:
        raise ValueError("locus_names, z_center, and z_rate_steep_positive must have "
                          "the same length.")

    corrected_alpha = alpha / n
    chi2_crit = stats.chi2.ppf(1.0 - corrected_alpha, df=2)
    z_crit = stats.norm.ppf(1.0 - corrected_alpha / 2.0)

    flags = []
    for name, zc, zr in zip(locus_names, z_center, z_rate_steep_positive):
        d2 = float(zc ** 2 + zr ** 2)
        is_outlier = d2 > chi2_crit
        center_outlier = abs(zc) > z_crit
        rate_outlier = abs(zr) > z_crit

        if rate_outlier:
            rate_direction = "steep" if zr > 0 else "shallow"
        else:
            rate_direction = None

        if not is_outlier:
            category = None
        elif center_outlier and rate_outlier:
            category = "ambiguous"
        elif center_outlier:
            category = "asymmetric_gene_flow"
        elif rate_outlier:
            category = "restricted_gene_flow" if rate_direction == "steep" else "elevated_gene_flow"
        else:
            # a joint D^2 outlier where neither marginal test individually
            # crosses its own (stricter) threshold -- a real, subtler
            # combined deviation, not a fifth "no reason" bucket.
            category = "ambiguous"

        flags.append(LocusCandidateFlag(
            locus=name, d2=d2, chi2_quantile=float(stats.chi2.cdf(d2, df=2)),
            is_outlier=bool(is_outlier), center_outlier=bool(center_outlier),
            rate_outlier=bool(rate_outlier), rate_direction=rate_direction,
            category=category))

    return CandidateFlagResult(flags=flags, alpha=alpha, n_loci=n)


def flag_geographic_candidates(locus_names, locus_center, locus_width,
                                mu_center: float, sigma_center: float,
                                mu_log_width: float, sigma_log_width: float,
                                alpha: float = 0.05) -> CandidateFlagResult:
    """Flag geographic-cline loci as candidates for restricted, elevated,
    or asymmetric gene flow relative to the rest of the locus set, using
    Fitzpatrick (2013)'s multivariate outlier method against a
    hierarchical fit's own population-level hyperparameters.

    Scope: intended for hierarchical fits only (`fit-hierarchical-
    geographic`'s `population_summary()`/`locus_summary()`), not
    independent per-locus fits -- an independent point estimate's own
    sampling noise can look exactly like a real outlier, while the
    hierarchical population-level hyperparameters already estimate the
    correct empirical null while properly accounting for that per-locus
    uncertainty. See this module's own "WHY THIS COMES AFTER COMPONENTS
    A/B" docstring note above for the same reasoning applied to
    `geographic_cline_variance`.

    "Steep" for geographic clines means *narrower* width than the
    population average (width and steepness are inversely related) --
    this function flips the sign of z(log width) internally so a
    positive `rate` z-score always means steeper, matching the shared
    convention `_flag_from_z_scores` expects.

    Parameters
    ----------
    locus_names : per-locus names, same order as locus_center/locus_width
    locus_center, locus_width : per-locus point estimates (e.g.
        `HierarchicalFitResult.locus_summary()`'s `center_mean`/
        `width_mean` columns)
    mu_center, sigma_center, mu_log_width, sigma_log_width : population-
        level hyperparameter posterior means (e.g.
        `HierarchicalFitResult.population_summary()`'s `mean` column for
        each of these four rows)
    alpha : family-wise significance level before Bonferroni correction
        across loci (default 0.05, matching Fitzpatrick's own choice)

    Returns
    -------
    CandidateFlagResult

    See `CANDIDATE_FLAGGING_CAVEAT` (and this module's section header
    comment above `LocusCandidateFlag`) for the mandatory caveats on
    interpreting these categories -- they name statistical patterns, not
    confirmed evolutionary mechanisms.
    """
    locus_center = np.asarray(locus_center, dtype=float)
    locus_width = np.asarray(locus_width, dtype=float)
    if np.any(locus_width <= 0):
        raise ValueError("locus_width must be positive.")
    if sigma_center <= 0 or sigma_log_width <= 0:
        raise ValueError("sigma_center and sigma_log_width must be positive.")

    z_center = (locus_center - mu_center) / sigma_center
    z_log_width = (np.log(locus_width) - mu_log_width) / sigma_log_width
    z_rate_steep_positive = -z_log_width  # narrower width = steeper -> flip sign

    return _flag_from_z_scores(locus_names, z_center, z_rate_steep_positive, alpha)


def flag_genomic_candidates(locus_names, locus_u, locus_v,
                             mu_center_or_u: float, sigma_center_or_u: float,
                             mu_log_v: float, sigma_log_v: float,
                             pool_center: bool = False,
                             alpha: float = 0.05) -> CandidateFlagResult:
    """Flag genomic-cline loci as candidates for restricted, elevated, or
    asymmetric gene flow relative to the rest of the locus set, using
    Fitzpatrick (2013)'s multivariate outlier method against a
    hierarchical fit's own population-level hyperparameters.

    Scope, and why hierarchical-only, exactly mirrors
    `flag_geographic_candidates` -- see that function's docstring.

    "Steep" for genomic clines means `v > 1` (steeper than the neutral/
    population-average rate), so `z(log v)` is used directly with no sign
    flip -- unlike the geographic case, where narrower width (the
    opposite-signed relationship) is steeper.

    Parameters
    ----------
    locus_names : per-locus names, same order as locus_u/locus_v
    locus_u, locus_v : per-locus point estimates (e.g.
        `HierarchicalGenomicFitResult.locus_summary()`'s `u_mean`/`v_mean`
        columns -- present regardless of whether the fit used
        `pool_center`, per that method's own docstring)
    mu_center_or_u, sigma_center_or_u : population-level hyperparameter
        posterior means for the center/bias dimension -- pass
        `population_summary()`'s `mu_center`/`sigma_center` row means if
        the fit used `pool_center=True`, or `mu_u`/`sigma_u` otherwise
        (set `pool_center` to match)
    mu_log_v, sigma_log_v : population-level hyperparameter posterior
        means for `log_v` (present in `population_summary()` either way)
    pool_center : must match whatever `pool_center` state the hierarchical
        fit that produced `mu_center_or_u`/`sigma_center_or_u` used --
        determines whether the center dimension is computed as
        `logit(center) = u/v` (True, Firneno et al. 2023's own quantity)
        or used as raw `u` directly (False, the default)
    alpha : family-wise significance level before Bonferroni correction
        across loci (default 0.05, matching Fitzpatrick's own choice)

    Returns
    -------
    CandidateFlagResult

    See `CANDIDATE_FLAGGING_CAVEAT` for the mandatory caveats on
    interpreting these categories.
    """
    locus_u = np.asarray(locus_u, dtype=float)
    locus_v = np.asarray(locus_v, dtype=float)
    if len(locus_u) != len(locus_v):
        raise ValueError("locus_u and locus_v must have the same length.")
    if np.any(locus_v <= 0):
        raise ValueError("locus_v must be positive.")
    if sigma_center_or_u <= 0 or sigma_log_v <= 0:
        raise ValueError("sigma_center_or_u and sigma_log_v must be positive.")

    center_dim = (locus_u / locus_v) if pool_center else locus_u
    z_center = (center_dim - mu_center_or_u) / sigma_center_or_u
    z_log_v = (np.log(locus_v) - mu_log_v) / sigma_log_v
    z_rate_steep_positive = z_log_v  # v>1 = steep -> already the right sign

    return _flag_from_z_scores(locus_names, z_center, z_rate_steep_positive, alpha)
