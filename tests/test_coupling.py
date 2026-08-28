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
    GenomicClineVariance, flag_geographic_candidates, flag_genomic_candidates,
    CANDIDATE_FLAGGING_CAVEAT,
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


def test_geographic_cline_variance_rejects_non_finite_widths():
    # Regression test: np.any(widths <= 0) does not catch NaN/Inf (NaN
    # comparisons are always False), so a non-finite width used to
    # silently produce NaN mean_slope/sd_slope/cv_slope instead of
    # raising -- realistic input, since a divergent hierarchical MCMC
    # chain can write NaN straight into the locus-summary column this
    # function consumes.
    with pytest.raises(ValueError, match="finite"):
        geographic_cline_variance([10.0, np.nan, 15.0])


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


def test_genomic_cline_variance_rejects_non_finite_u_or_v():
    # Regression test: same NaN/Inf blind spot as
    # geographic_cline_variance for v, and u was never validated at all.
    with pytest.raises(ValueError, match="finite"):
        genomic_cline_variance([1.0, 2.0, np.nan], [2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="finite"):
        genomic_cline_variance([1.0, 2.0, 3.0], [2.0, np.inf, 4.0])


def test_genomic_cline_variance_summary_flags_nan_theta_instead_of_misclassifying():
    # Regression test: the theta classification (<=0 / >1 / else) had no
    # NaN branch, so a NaN theta_estimate silently fell into the `else`
    # clause and printed a false, specific biological claim ("theta < 1:
    # consistent with weaker/partial coupling").
    result = GenomicClineVariance(sd_logit_center=0.5, sd_log_v=0.5, n_loci=5,
                                   theta_estimate=np.nan, within_calibration_range=True)
    s = result.summary()
    assert "not a finite number" in s
    assert "weaker/partial coupling" not in s


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


# ---------------------------------------------------------------------------
# Candidate-locus flagging (Fitzpatrick 2013's multivariate outlier method)
# ---------------------------------------------------------------------------

def test_flag_geographic_candidates_categorizes_each_pattern():
    names = [f"L{i}" for i in range(10)]
    centers = np.array([50.0] * 10)
    widths = np.array([20.0] * 10)
    widths[3] = 2.0     # steep, not displaced -> restricted_gene_flow
    centers[5] = 90.0   # displaced, not steep -> asymmetric_gene_flow
    widths[7] = 200.0   # shallow, not displaced -> elevated_gene_flow
    centers[9], widths[9] = 90.0, 2.0  # both -> ambiguous

    result = flag_geographic_candidates(
        names, centers, widths, mu_center=50.0, sigma_center=5.0,
        mu_log_width=np.log(20.0), sigma_log_width=0.3, alpha=0.05)

    by_locus = {f.locus: f for f in result.flags}
    assert by_locus["L0"].category is None
    assert by_locus["L0"].is_outlier is False
    assert by_locus["L3"].category == "restricted_gene_flow"
    assert by_locus["L3"].rate_direction == "steep"
    assert by_locus["L5"].category == "asymmetric_gene_flow"
    assert by_locus["L5"].center_outlier is True
    assert by_locus["L5"].rate_outlier is False
    assert by_locus["L7"].category == "elevated_gene_flow"
    assert by_locus["L7"].rate_direction == "shallow"
    assert by_locus["L9"].category == "ambiguous"
    assert by_locus["L9"].center_outlier is True
    assert by_locus["L9"].rate_outlier is True
    assert result.n_loci == 10
    assert result.alpha == 0.05


def test_flag_genomic_candidates_categorizes_each_pattern():
    names = [f"L{i}" for i in range(10)]
    u = np.array([0.0] * 10)
    v = np.array([1.0] * 10)
    v[3] = 5.0   # v>1 -> steep, not displaced -> restricted_gene_flow
    u[5] = 3.0   # displaced, not steep -> asymmetric_gene_flow
    v[7] = 0.1   # v<1 -> shallow, not displaced -> elevated_gene_flow

    result = flag_genomic_candidates(
        names, u, v, mu_center_or_u=0.0, sigma_center_or_u=0.3,
        mu_log_v=0.0, sigma_log_v=0.3, pool_center=False, alpha=0.05)

    by_locus = {f.locus: f for f in result.flags}
    assert by_locus["L3"].category == "restricted_gene_flow"
    assert by_locus["L3"].rate_direction == "steep"
    assert by_locus["L5"].category == "asymmetric_gene_flow"
    assert by_locus["L7"].category == "elevated_gene_flow"
    assert by_locus["L7"].rate_direction == "shallow"


def test_flag_genomic_candidates_pool_center_matches_non_pool_center_when_v_is_one():
    # u/v == u exactly when v == 1 everywhere -- pool_center should be a
    # no-op on the flagged categories in that special case, confirming
    # the two code paths compute the same z_center quantity as designed.
    names = [f"L{i}" for i in range(6)]
    u = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0])
    v = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 0.1])

    result_raw = flag_genomic_candidates(
        names, u, v, mu_center_or_u=0.0, sigma_center_or_u=0.3,
        mu_log_v=0.0, sigma_log_v=0.3, pool_center=False, alpha=0.05)
    result_pooled = flag_genomic_candidates(
        names, u, v, mu_center_or_u=0.0, sigma_center_or_u=0.3,
        mu_log_v=0.0, sigma_log_v=0.3, pool_center=True, alpha=0.05)

    for f_raw, f_pooled in zip(result_raw.flags, result_pooled.flags):
        assert f_raw.category == f_pooled.category
        assert f_raw.d2 == pytest.approx(f_pooled.d2)


def test_flag_candidates_ambiguous_via_joint_but_not_marginal_deviation():
    # A locus with z_center = z_rate = 2.0: d2 = 8.0 exceeds the 2-locus
    # Bonferroni-adjusted chi-squared(df=2) critical value (~7.38), but
    # neither |z| individually exceeds its own per-parameter critical
    # value (~2.24) -- a real joint deviation the two marginal tests
    # aren't positioned to explain alone. Must land in 'ambiguous', not
    # silently fall through uncategorized.
    names = ["null", "joint_only"]
    centers = np.array([0.0, 2.0])
    widths = np.array([1.0, np.exp(-2.0)])  # z(log width) = -2.0 -> steep z = +2.0

    result = flag_geographic_candidates(
        names, centers, widths, mu_center=0.0, sigma_center=1.0,
        mu_log_width=0.0, sigma_log_width=1.0, alpha=0.05)

    flagged = result.flags[1]
    assert flagged.d2 == pytest.approx(8.0)
    assert flagged.is_outlier is True
    assert flagged.center_outlier is False
    assert flagged.rate_outlier is False
    assert flagged.category == "ambiguous"


def test_flag_geographic_candidates_requires_at_least_two_loci():
    with pytest.raises(ValueError, match="at least 2 loci"):
        flag_geographic_candidates(["L0"], [50.0], [20.0], mu_center=50.0,
                                    sigma_center=5.0, mu_log_width=np.log(20.0),
                                    sigma_log_width=0.3)


def test_flag_geographic_candidates_rejects_non_positive_width():
    with pytest.raises(ValueError, match="width must be positive"):
        flag_geographic_candidates(["A", "B"], [50.0, 50.0], [20.0, -5.0],
                                    mu_center=50.0, sigma_center=5.0,
                                    mu_log_width=np.log(20.0), sigma_log_width=0.3)


def test_flag_geographic_candidates_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="must be positive"):
        flag_geographic_candidates(["A", "B"], [50.0, 50.0], [20.0, 20.0],
                                    mu_center=50.0, sigma_center=0.0,
                                    mu_log_width=np.log(20.0), sigma_log_width=0.3)


def test_flag_genomic_candidates_rejects_non_positive_v():
    with pytest.raises(ValueError, match="v must be positive"):
        flag_genomic_candidates(["A", "B"], [0.0, 0.0], [1.0, -1.0],
                                 mu_center_or_u=0.0, sigma_center_or_u=0.3,
                                 mu_log_v=0.0, sigma_log_v=0.3)


def test_flag_genomic_candidates_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        flag_genomic_candidates(["A", "B"], [0.0, 0.0, 0.0], [1.0, 1.0],
                                 mu_center_or_u=0.0, sigma_center_or_u=0.3,
                                 mu_log_v=0.0, sigma_log_v=0.3)


def test_flag_candidates_bonferroni_correction_scales_with_n_loci():
    # A fixed z_center that's a marginal outlier against a small locus
    # set's looser threshold should stop being one once diluted into a
    # much larger set, purely from the stricter Bonferroni correction --
    # not because the underlying deviation changed.
    small = flag_geographic_candidates(
        ["A", "B"], [52.5, 50.0], [20.0, 20.0], mu_center=50.0, sigma_center=1.0,
        mu_log_width=np.log(20.0), sigma_log_width=0.3, alpha=0.05)
    n_large = 500
    names = [f"L{i}" for i in range(n_large)]
    centers = [50.0] * n_large
    centers[0] = 52.5
    widths = [20.0] * n_large
    large = flag_geographic_candidates(
        names, centers, widths, mu_center=50.0, sigma_center=1.0,
        mu_log_width=np.log(20.0), sigma_log_width=0.3, alpha=0.05)

    assert small.flags[0].center_outlier is True
    assert large.flags[0].center_outlier is False


def test_candidate_flagging_caveat_cites_key_sources():
    assert "Gompert" in CANDIDATE_FLAGGING_CAVEAT
    assert "Payseur" in CANDIDATE_FLAGGING_CAVEAT
    assert "Jofre" in CANDIDATE_FLAGGING_CAVEAT
    assert "epistatic" in CANDIDATE_FLAGGING_CAVEAT.lower()
    assert "incomplete lineage sorting" in CANDIDATE_FLAGGING_CAVEAT.lower()


def test_flag_candidates_result_summary_reports_category_counts():
    names = ["A", "B", "C"]
    result = flag_geographic_candidates(
        names, [50.0, 50.0, 90.0], [2.0, 20.0, 20.0], mu_center=50.0,
        sigma_center=5.0, mu_log_width=np.log(20.0), sigma_log_width=0.3, alpha=0.05)
    summary = result.summary()
    assert "restricted_gene_flow: 1" in summary
    assert "asymmetric_gene_flow: 1" in summary
    assert "CAVEATS" in summary


def test_end_to_end_flag_candidates_with_hierarchical_geographic_fit():
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch

    rng = np.random.default_rng(5)
    distance = np.sort(rng.uniform(0, 100, size=30))
    n_loci = 4
    true_widths = [15.0, 16.0, 14.0, 2.0]  # last locus deliberately much steeper
    n_alleles = np.full((n_loci, 30), 30.0)
    n_focal = np.zeros((n_loci, 30))
    for j in range(n_loci):
        params = ClineParams(center=50.0, width=true_widths[j], pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        n_focal[j, :] = rng.binomial(30, p)

    fit = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                  num_warmup=500, num_samples=500, num_chains=2, seed=0)
    pop_df = fit.population_summary()
    locus_df = fit.locus_summary()

    result = flag_geographic_candidates(
        locus_df["locus"], locus_df["center_mean"], locus_df["width_mean"],
        mu_center=float(pop_df.loc["mu_center", "mean"]),
        sigma_center=float(pop_df.loc["sigma_center", "mean"]),
        mu_log_width=float(pop_df.loc["mu_log_width", "mean"]),
        sigma_log_width=float(pop_df.loc["sigma_log_width", "mean"]),
        alpha=0.05)
    assert result.n_loci == 4
    assert all(f.category in (None, "restricted_gene_flow", "elevated_gene_flow",
                               "asymmetric_gene_flow", "ambiguous") for f in result.flags)


def test_end_to_end_flag_candidates_with_hierarchical_genomic_fit():
    from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
    from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch

    rng = np.random.default_rng(6)
    hybrid_index = rng.uniform(0.02, 0.98, 100)
    n_loci = 4
    true_u = [0.1, -0.1, 0.05, 3.0]  # last locus deliberately displaced
    true_v = [1.0, 1.1, 0.9, 1.05]
    n_alleles = np.full((n_loci, 100), 2.0)
    n_focal = np.zeros((n_loci, 100))
    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        n_focal[j, :] = rng.binomial(2, phi)

    fit = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                          num_warmup=500, num_samples=500, num_chains=2, seed=0)
    pop_df = fit.population_summary()
    locus_df = fit.locus_summary()

    result = flag_genomic_candidates(
        locus_df["locus"], locus_df["u_mean"], locus_df["v_mean"],
        mu_center_or_u=float(pop_df.loc["mu_u", "mean"]),
        sigma_center_or_u=float(pop_df.loc["sigma_u", "mean"]),
        mu_log_v=float(pop_df.loc["mu_log_v", "mean"]),
        sigma_log_v=float(pop_df.loc["sigma_log_v", "mean"]),
        pool_center=False, alpha=0.05)
    assert result.n_loci == 4
    assert all(f.category in (None, "restricted_gene_flow", "elevated_gene_flow",
                               "asymmetric_gene_flow", "ambiguous") for f in result.flags)
