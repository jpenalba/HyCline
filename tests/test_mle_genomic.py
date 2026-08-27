import numpy as np
import pytest

from hycline.data.schema import HybridIndexLocusData
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
from hycline.inference.mle_genomic import fit_genomic_cline_mle, null_model_aicc_delta


def _simulate(true_params, n_ind=200, ploidy=2, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.02, 0.98, n_ind)
    phi = genomic_cline_function(h, true_params)
    n_alleles = np.full(n_ind, float(ploidy))
    n_focal = rng.binomial(ploidy, phi).astype(float)
    return HybridIndexLocusData(hybrid_index=h, n_alleles=n_alleles, n_focal=n_focal)


def test_mle_recovers_neutral_locus():
    true_params = GenomicClineParams(u=0.0, v=1.0)
    data = _simulate(true_params, n_ind=300, seed=1)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.u == pytest.approx(0.0, abs=0.4)
    assert fit.params.v == pytest.approx(1.0, abs=0.3)


def test_mle_recovers_barrier_locus():
    true_params = GenomicClineParams(u=0.0, v=5.0)
    data = _simulate(true_params, n_ind=300, seed=2)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.v == pytest.approx(5.0, abs=1.5)
    assert fit.params.v > 2.0  # unambiguously steeper than neutral


def test_mle_recovers_permeable_locus():
    true_params = GenomicClineParams(u=0.0, v=0.3)
    data = _simulate(true_params, n_ind=300, seed=3)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.v < 0.7  # unambiguously shallower than neutral


def test_mle_recovers_biased_locus():
    true_params = GenomicClineParams(u=2.5, v=1.0)
    data = _simulate(true_params, n_ind=300, seed=4)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.u == pytest.approx(2.5, abs=0.8)
    assert fit.params.u > 1.0  # unambiguously biased toward population 1


def test_summary_runs_and_labels_barrier_locus_correctly():
    true_params = GenomicClineParams(u=0.0, v=6.0)
    data = _simulate(true_params, n_ind=300, seed=5)
    fit = fit_genomic_cline_mle(data)
    s = fit.summary()
    assert "barrier" in s.lower()


def test_summary_runs_and_labels_permeable_locus_correctly():
    true_params = GenomicClineParams(u=0.0, v=0.2)
    data = _simulate(true_params, n_ind=300, seed=6)
    fit = fit_genomic_cline_mle(data)
    s = fit.summary()
    assert "introgression" in s.lower()


# ---------------------------------------------------------------------------
# null_model_aicc_delta -- delta-AICc ranking against the fixed neutral null
# (Bailey 2024's recommended alternative to raw p-values)
# ---------------------------------------------------------------------------

def test_null_model_aicc_delta_strongly_positive_for_clearly_non_null_locus():
    true_params = GenomicClineParams(u=1.5, v=5.0)
    data = _simulate(true_params, n_ind=300, seed=10)
    fit = fit_genomic_cline_mle(data)
    result = null_model_aicc_delta(data, fit)
    assert result["delta_aicc_from_null"] > 20.0
    assert result["aicc_null"] > fit.aicc


def test_null_model_aicc_delta_near_zero_or_negative_for_genuinely_null_locus():
    true_params = GenomicClineParams(u=0.0, v=1.0)
    data = _simulate(true_params, n_ind=300, seed=11)
    fit = fit_genomic_cline_mle(data)
    result = null_model_aicc_delta(data, fit)
    # the null model has 0 free parameters vs. the fitted model's 2, so a
    # genuinely neutral locus should show little to no AICc improvement
    # from fitting -- the two extra parameters aren't worth their penalty
    assert result["delta_aicc_from_null"] < 5.0


def test_null_model_aicc_delta_log_likelihood_null_matches_direct_computation():
    from hycline.likelihoods.genomic_binomial import log_likelihood as raw_log_likelihood

    true_params = GenomicClineParams(u=0.8, v=2.0)
    data = _simulate(true_params, n_ind=200, seed=12)
    fit = fit_genomic_cline_mle(data)
    result = null_model_aicc_delta(data, fit)
    expected_ll_null = raw_log_likelihood(data, GenomicClineParams(u=0.0, v=1.0))
    assert result["log_likelihood_null"] == pytest.approx(expected_ll_null)


def test_null_model_aicc_delta_formula_matches_manual_aicc_computation():
    true_params = GenomicClineParams(u=1.0, v=3.0)
    data = _simulate(true_params, n_ind=250, seed=13)
    fit = fit_genomic_cline_mle(data)
    result = null_model_aicc_delta(data, fit)

    # AICc with k=0 free parameters reduces to plain AIC (the small-
    # sample correction term is zero when k=0)
    expected_aicc_null = -2 * result["log_likelihood_null"]
    assert result["aicc_null"] == pytest.approx(expected_aicc_null)
    assert result["delta_aicc_from_null"] == pytest.approx(
        result["aicc_null"] - fit.aicc)


def test_null_model_aicc_delta_ranks_multiple_loci_sensibly():
    # a null, a weakly non-null, and a strongly non-null locus should
    # rank in that order by delta_aicc_from_null
    null_data = _simulate(GenomicClineParams(u=0.0, v=1.0), n_ind=300, seed=20)
    weak_data = _simulate(GenomicClineParams(u=0.3, v=1.3), n_ind=300, seed=21)
    strong_data = _simulate(GenomicClineParams(u=2.0, v=6.0), n_ind=300, seed=22)

    deltas = {}
    for name, data in [("null", null_data), ("weak", weak_data), ("strong", strong_data)]:
        fit = fit_genomic_cline_mle(data)
        deltas[name] = null_model_aicc_delta(data, fit)["delta_aicc_from_null"]

    assert deltas["null"] < deltas["weak"] < deltas["strong"]
