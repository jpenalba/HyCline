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
from typing import Optional

import numpy as np


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
    if np.any(widths <= 0):
        raise ValueError("Cline widths must be positive.")
    if len(widths) < 2:
        raise ValueError("Need at least 2 loci to compute across-locus variance.")

    slopes = 4.0 / widths
    mean_slope = float(np.mean(slopes))
    sd_slope = float(np.std(slopes, ddof=1))
    cv_slope = float(sd_slope / abs(mean_slope)) if mean_slope != 0 else float("inf")

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
            if self.theta_estimate <= 0:
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
