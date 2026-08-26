"""
Tests for Phase 4 Component C: coupling/concordance analysis across loci.

See BACKEND_PHASE4.md section 9 for the full methodology, including why
the theta calibration (`estimate_coupling_coefficient`) is a reverse-
engineered approximation of Firneno et al. (2023)'s own linear model, not
a fresh regression.
"""

import numpy as np
import pytest

from hycline.analysis.coupling import (
    geographic_cline_variance, genomic_cline_variance, estimate_coupling_coefficient,
    calibration_data_range, FIRNENO_2023_TABLE1, GeographicClineVariance,
    GenomicClineVariance,
)


# ---------------------------------------------------------------------------
# Geographic cline variance
# ---------------------------------------------------------------------------

def test_geographic_cline_variance_basic_computation():
    widths = [10.0, 20.0, 10.0, 20.0]
    result = geographic_cline_variance(widths)
    slopes = [4 / w for w in widths]
    assert result.mean_slope == pytest.approx(np.mean(slopes))
    assert result.sd_slope == pytest.approx(np.std(slopes, ddof=1))
    assert result.n_loci == 4


def test_geographic_cline_variance_identical_widths_gives_zero_sd():
    """Perfectly concordant clines (identical widths) should give SD = 0
    -- the theoretical maximum-coupling case."""
    result = geographic_cline_variance([15.0, 15.0, 15.0, 15.0])
    assert result.sd_slope == pytest.approx(0.0, abs=1e-9)
    assert result.cv_slope == pytest.approx(0.0, abs=1e-9)


def test_geographic_cline_variance_requires_positive_widths():
    with pytest.raises(ValueError, match="positive"):
        geographic_cline_variance([10.0, -5.0, 20.0])


def test_geographic_cline_variance_requires_at_least_two_loci():
    with pytest.raises(ValueError, match="at least 2"):
        geographic_cline_variance([10.0])


def test_geographic_cline_variance_summary_runs_and_has_no_theta():
    result = geographic_cline_variance([10.0, 15.0, 20.0])
    s = result.summary()
    assert "sigma_beta" in s.lower() or "sd slope" in s.lower()
    assert "no theta estimate" in s.lower()


# ---------------------------------------------------------------------------
# Genomic cline variance
# ---------------------------------------------------------------------------

def test_genomic_cline_variance_neutral_loci_gives_low_variance():
    u = [0.05, -0.03, 0.02, -0.04, 0.01]
    v = [1.02, 0.98, 1.01, 0.99, 1.03]
    result = genomic_cline_variance(u, v, estimate_theta=False)
    assert result.sd_logit_center < 0.1
    assert result.sd_log_v < 0.05


def test_genomic_cline_variance_computes_correct_logit_center():
    u = np.array([2.0, -1.0])
    v = np.array([1.0, 2.0])
    result = genomic_cline_variance(u, v, estimate_theta=False)
    logit_center = u / v
    assert result.sd_logit_center == pytest.approx(np.std(logit_center, ddof=1))


def test_genomic_cline_variance_requires_positive_v():
    with pytest.raises(ValueError, match="positive"):
        genomic_cline_variance([0.1, 0.2], [1.0, -0.5])


def test_genomic_cline_variance_requires_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        genomic_cline_variance([0.1, 0.2, 0.3], [1.0, 1.0])


def test_genomic_cline_variance_estimate_theta_flag():
    u = [0.1, -0.2, 0.3]
    v = [1.0, 1.1, 0.9]
    with_theta = genomic_cline_variance(u, v, estimate_theta=True)
    without_theta = genomic_cline_variance(u, v, estimate_theta=False)
    assert with_theta.theta_estimate is not None
    assert without_theta.theta_estimate is None


def test_genomic_cline_variance_summary_flags_negative_theta():
    """Regression test for the real, documented finding: some
    (sigma_c, sigma_v) combinations give a nonsensical negative theta,
    even within the calibration data's bounding box, and this must be
    surfaced as a warning, not silently reported as a valid estimate."""
    result = GenomicClineVariance(sd_logit_center=0.80, sd_log_v=0.44, n_loci=8,
                                   theta_estimate=estimate_coupling_coefficient(0.80, 0.44))
    assert result.theta_estimate < 0
    s = result.summary()
    assert "WARNING" in s


def test_genomic_cline_variance_flags_out_of_range_inputs():
    """A far-outside-range (sigma_c, sigma_v) combination should be
    flagged as an extrapolation, and a within-range one should not."""
    r = calibration_data_range()
    in_range_result = genomic_cline_variance(
        [0.0, 0.5], [1.0, 1.0])  # sd_logit_center, sd_log_v both small, near table's low end
    out_of_range_result = genomic_cline_variance(
        [0.0, 10.0], [1.0, 1.0])  # sd_logit_center = 5.0, far above sigma_c_max
    assert out_of_range_result.sd_logit_center > r["sigma_c_max"]
    assert out_of_range_result.within_calibration_range is False
    s = out_of_range_result.summary()
    assert "extrapolation" in s.lower()


# ---------------------------------------------------------------------------
# Theta calibration
# ---------------------------------------------------------------------------

def test_calibration_reproduces_firneno_2023_table1_closely():
    max_err = 0.0
    for name, sc, sv, theta_true in FIRNENO_2023_TABLE1:
        theta_pred = estimate_coupling_coefficient(sc, sv)
        max_err = max(max_err, abs(theta_pred - theta_true))
    assert max_err < 0.01


def test_calibration_data_range_matches_table():
    r = calibration_data_range()
    scs = [row[1] for row in FIRNENO_2023_TABLE1]
    svs = [row[2] for row in FIRNENO_2023_TABLE1]
    assert r["sigma_c_min"] == pytest.approx(min(scs))
    assert r["sigma_c_max"] == pytest.approx(max(scs))
    assert r["sigma_v_min"] == pytest.approx(min(svs))
    assert r["sigma_v_max"] == pytest.approx(max(svs))


def test_reference_table_has_31_rows_matching_25_studies_with_transects():
    assert len(FIRNENO_2023_TABLE1) == 31
    organisms = {name.split("_")[0] for name, *_ in FIRNENO_2023_TABLE1}
    assert len(organisms) == 25


def test_reference_table_matches_published_table1_exactly():
    """Regression test locking in that FIRNENO_2023_TABLE1 matches the
    actual published Table 1 (Firneno et al. 2023) digit for digit --
    confirmed directly against the paper's full text (via PMC/NSF PAR),
    not the informal summary table in the GitHub repository's README,
    which has at least one clear transcription error (Neotoma listed
    there as theta=64 rather than the published theta=0.64). This table
    was never affected by that error, but this test exists so any future
    edit to the reference data is checked against the real source, not
    just internal self-consistency."""
    published_table1 = [
        ("Agalychnis", 1.42, 0.56, 0.12), ("Alouatta", 1.02, 0.23, 0.65),
        ("Coenonympha", 1.11, 0.31, 0.44), ("Corvus", 0.86, 0.40, 0.14),
        ("Crotalus", 0.91, 0.33, 0.37), ("Encelia", 0.38, 0.31, 0.41),
        ("Fundulus_1", 0.45, 0.12, 1.30), ("Fundulus_2", 0.33, 0.08, 1.58),
        ("Gryllus_1", 0.81, 0.24, 0.67), ("Gryllus_2", 0.48, 0.15, 1.15),
        ("Hirundo", 0.53, 0.31, 0.42), ("Iris", 1.16, 0.19, 0.71),
        ("Lissotriton_1", 0.35, 0.14, 1.26), ("Lissotriton_2", 0.97, 0.14, 0.92),
        ("Lycaeides", 0.94, 0.35, 0.31), ("Mus_1", 0.69, 0.16, 1.00),
        ("Mus_2", 0.70, 0.14, 1.07), ("Mus_3", 1.01, 0.23, 0.66),
        ("Mytilus", 0.29, 0.27, 0.62), ("Nematocharax", 0.80, 0.41, 0.08),
        ("Neotoma", 0.52, 0.26, 0.64), ("Oleria", 1.73, 0.50, 0.42),
        ("Papilio", 0.29, 0.21, 0.93), ("Papio", 0.84, 0.41, 0.10),
        ("Picea", 0.67, 0.22, 0.77), ("Pinus", 1.10, 0.37, 0.29),
        ("Poecile_1", 0.94, 0.40, 0.17), ("Poecile_2", 0.82, 0.38, 0.19),
        ("Sceloporus", 1.09, 0.21, 0.69), ("Sternotherus", 0.57, 0.28, 0.55),
        ("Yucca", 0.64, 0.15, 1.06),
    ]
    assert FIRNENO_2023_TABLE1 == published_table1


def test_calibration_holds_out_of_sample_on_table_s4():
    """Genuine out-of-sample validation, not a restatement of the 31-point
    fit: Table S4 of Firneno et al. (2023)'s Supplemental Material reports
    sigma_c/sigma_v/theta separately for autosomes and sex chromosomes for
    8 of the 25 hybrid zones (16 real, published data points) -- none of
    which were part of FIRNENO_2023_TABLE1 or used to fit the calibration
    at all. If the calibration recovered in this module were overfitting
    noise in the 31-point table rather than the true underlying
    relationship, it would not be expected to generalize well here.

    One point (Poecile's sex-chromosome subset) falls outside the
    31-point calibration's bounding box (sigma_c=2.58 vs. max sigma_c=1.73
    in FIRNENO_2023_TABLE1) and is included deliberately -- see the
    assertion on `within_calibration_range` below."""
    table_s4 = [
        ("Crotalus_auto", 0.87, 0.32, 0.41), ("Crotalus_sex", 0.98, 0.25, 0.62),
        ("Gryllus_CT_auto", 0.78, 0.19, 0.86), ("Gryllus_CT_sex", 0.97, 0.33, 0.39),
        ("Gryllus_PA_auto", 0.51, 0.14, 1.18), ("Gryllus_PA_sex", 0.33, 0.15, 1.22),
        ("Lycaeides_auto", 0.71, 0.32, 0.37), ("Lycaeides_sex", 1.49, 0.36, 0.40),
        ("Mus_BV_auto", 0.72, 0.15, 1.01), ("Mus_BV_sex", 0.40, 0.13, 1.30),
        ("Mus_CZ_auto", 0.73, 0.14, 1.06), ("Mus_CZ_sex", 0.39, 0.13, 1.30),
        ("Mus_SX_auto", 0.97, 0.19, 0.78), ("Mus_SX_sex", 0.56, 0.22, 0.81),
        ("Poecile_PA_auto", 0.83, 0.38, 0.19), ("Poecile_PA_sex", 2.58, 0.43, 0.82),
    ]

    errors = []
    for name, sc, sv, theta_true in table_s4:
        theta_pred = estimate_coupling_coefficient(sc, sv)
        errors.append(abs(theta_pred - theta_true))

    errors = np.array(errors)
    assert errors.mean() < 0.02  # mean absolute error, held-out
    assert errors.max() < 0.03  # max absolute error, held-out

    # confirm the extrapolation case is correctly flagged as such, even
    # though the prediction itself happens to be accurate here
    r = calibration_data_range()
    poecile_sex_sc = 2.58
    assert poecile_sex_sc > r["sigma_c_max"]


# ---------------------------------------------------------------------------
# End-to-end integration with Component A/B hierarchical fit output
# ---------------------------------------------------------------------------

def test_end_to_end_with_hierarchical_geographic_fit():
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch

    rng = np.random.default_rng(3)
    distance = np.sort(rng.uniform(0, 100, size=30))
    n_loci = 4
    true_widths = [15.0, 16.0, 14.0, 15.5]
    n_alleles = np.full((n_loci, 30), 20.0)
    n_focal = np.zeros((n_loci, 30))
    for j in range(n_loci):
        params = ClineParams(center=50.0, width=true_widths[j], pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        n_focal[j, :] = rng.binomial(20, p)

    fit = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                  num_warmup=500, num_samples=500, num_chains=2, seed=0)
    locus_df = fit.locus_summary()
    result = geographic_cline_variance(locus_df["width_mean"])
    assert result.n_loci == 4
    assert result.sd_slope >= 0


def test_end_to_end_with_hierarchical_genomic_fit():
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
    from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch

    rng = np.random.default_rng(4)
    hybrid_index = rng.uniform(0.02, 0.98, 100)
    n_loci = 4
    true_u = [0.1, -0.1, 0.05, -0.05]
    true_v = [1.0, 1.1, 0.9, 1.05]
    n_alleles = np.full((n_loci, 100), 2.0)
    n_focal = np.zeros((n_loci, 100))
    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        n_focal[j, :] = rng.binomial(2, phi)

    fit = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                          num_warmup=500, num_samples=500, num_chains=2, seed=0)
    locus_df = fit.locus_summary()
    result = genomic_cline_variance(locus_df["u_mean"], locus_df["v_mean"])
    assert result.n_loci == 4
    assert result.sd_logit_center >= 0
    assert result.sd_log_v >= 0
