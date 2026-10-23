import numpy as np
import pytest

from hycline.data.schema import HybridIndexLocusData
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
from hycline.inference.mle_genomic import fit_genomic_cline_mle


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
