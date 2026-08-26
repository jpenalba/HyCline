import numpy as np
import pytest

from hycline.data.genotype_likelihood import (
    phred_to_log_likelihood, log_likelihood_to_phred, simulate_genotype_likelihoods,
    hard_genotype_to_log_likelihood,
)
from hycline.data.schema import GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData


def test_phred_to_log_likelihood_known_values():
    pl = np.array([0.0, 10.0, 20.0])
    ll = phred_to_log_likelihood(pl)
    np.testing.assert_allclose(ll, [0.0, -np.log(10), -2 * np.log(10)])


def test_phred_roundtrip():
    pl = np.array([0.0, 5.5, 30.0, 99.0])
    ll = phred_to_log_likelihood(pl)
    pl_back = log_likelihood_to_phred(ll)
    np.testing.assert_allclose(pl_back, pl, atol=1e-9)


def test_simulate_genotype_likelihoods_high_depth_favors_true_genotype():
    rng = np.random.default_rng(0)
    true_genotypes = np.array([0, 1, 2, 0, 1, 2] * 20)
    result = simulate_genotype_likelihoods(true_genotypes, depth=50, error_rate=0.01, rng=rng)
    gl_stack = np.stack([result["log_gl0"], result["log_gl1"], result["log_gl2"]])
    called = np.argmax(gl_stack, axis=0)
    accuracy = np.mean(called == true_genotypes)
    assert accuracy > 0.85


def test_simulate_genotype_likelihoods_low_depth_is_less_certain():
    rng = np.random.default_rng(1)
    true_genotypes = np.full(200, 1)

    high = simulate_genotype_likelihoods(true_genotypes, depth=40, error_rate=0.01,
                                          rng=np.random.default_rng(2))
    low = simulate_genotype_likelihoods(true_genotypes, depth=2, error_rate=0.01,
                                         rng=np.random.default_rng(2))

    def separation(result):
        gls = np.stack([result["log_gl0"], result["log_gl1"], result["log_gl2"]])
        sorted_gls = np.sort(gls, axis=0)
        return np.mean(sorted_gls[-1] - sorted_gls[-2])

    assert separation(high) > separation(low)


def test_simulate_genotype_likelihoods_rejects_invalid_genotype():
    with pytest.raises(ValueError, match="0, 1, or 2"):
        simulate_genotype_likelihoods([0, 1, 3], depth=10)


def test_simulate_genotype_likelihoods_zero_depth_gives_flat_likelihoods():
    result = simulate_genotype_likelihoods([1], depth=0)
    assert result["log_gl0"][0] == 0.0
    assert result["log_gl1"][0] == 0.0
    assert result["log_gl2"][0] == 0.0


def test_genotype_likelihood_transect_basic():
    t = GenotypeLikelihoodTransect(
        distance=np.array([0.0, 10.0, 20.0]),
        log_gl0=np.array([-1.0, -2.0, -0.5]),
        log_gl1=np.array([-0.5, -0.5, -1.0]),
        log_gl2=np.array([-2.0, -1.0, -1.5]),
    )
    assert len(t) == 3


def test_genotype_likelihood_transect_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        GenotypeLikelihoodTransect(
            distance=np.array([0.0, 10.0]),
            log_gl0=np.array([-1.0, -2.0, -0.5]),
            log_gl1=np.array([-0.5, -0.5]),
            log_gl2=np.array([-2.0, -1.0]),
        )


def test_genotype_likelihood_locus_data_hybrid_index_validation():
    with pytest.raises(ValueError, match="0, 1"):
        GenotypeLikelihoodLocusData(
            hybrid_index=np.array([0.5, 1.5]),
            log_gl0=np.array([-1.0, -2.0]),
            log_gl1=np.array([-0.5, -0.5]),
            log_gl2=np.array([-2.0, -1.0]),
        )


def test_genotype_likelihood_locus_data_basic():
    d = GenotypeLikelihoodLocusData(
        hybrid_index=np.array([0.1, 0.5, 0.9]),
        log_gl0=np.array([-1.0, -2.0, -0.5]),
        log_gl1=np.array([-0.5, -0.5, -1.0]),
        log_gl2=np.array([-2.0, -1.0, -1.5]),
    )
    assert len(d) == 3


# ---------------------------------------------------------------------------
# hard_genotype_to_log_likelihood (the hard-genotype convenience path)
# ---------------------------------------------------------------------------

def test_hard_genotype_to_log_likelihood_one_hot():
    result = hard_genotype_to_log_likelihood(np.array([0, 1, 2]))
    np.testing.assert_array_equal(result["log_gl0"], [0.0, -1e10, -1e10])
    np.testing.assert_array_equal(result["log_gl1"], [-1e10, 0.0, -1e10])
    np.testing.assert_array_equal(result["log_gl2"], [-1e10, -1e10, 0.0])


def test_hard_genotype_to_log_likelihood_missing_is_flat():
    result = hard_genotype_to_log_likelihood(np.array([np.nan]))
    assert result["log_gl0"][0] == 0.0
    assert result["log_gl1"][0] == 0.0
    assert result["log_gl2"][0] == 0.0


def test_hard_genotype_to_log_likelihood_rejects_invalid_values():
    with pytest.raises(ValueError, match="0, 1, or 2"):
        hard_genotype_to_log_likelihood(np.array([0, 1, 3]))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        hard_genotype_to_log_likelihood(np.array([1.5]))


def test_hard_genotype_to_log_likelihood_marginalizes_to_exact_hwe_likelihood():
    """The core correctness property: a hard-called genotype's one-hot GL,
    marginalized, must reproduce the plain HWE likelihood exactly -- the
    same property `test_hard_call_marginal_likelihood_matches_plain_hwe`
    in test_genotype_likelihood_fitting.py checks starting from a
    hand-constructed one-hot GL; this test checks it starting from this
    function's actual output instead."""
    from hycline.likelihoods.genotype_likelihood import _marginal_log_likelihood

    p = 0.3
    for g, expected_fn in [(0, lambda p: np.log((1 - p) ** 2)),
                            (1, lambda p: np.log(2 * p * (1 - p))),
                            (2, lambda p: np.log(p ** 2))]:
        gls = hard_genotype_to_log_likelihood(np.array([float(g)]))
        ll = _marginal_log_likelihood(gls["log_gl0"], gls["log_gl1"], gls["log_gl2"],
                                       np.array([p]))
        assert ll[0] == pytest.approx(expected_fn(p), abs=1e-9)


def test_hard_genotype_to_log_likelihood_2d_array():
    genotype = np.array([[0, 1], [2, np.nan]])
    result = hard_genotype_to_log_likelihood(genotype)
    assert result["log_gl0"].shape == (2, 2)
    assert result["log_gl0"][0, 0] == 0.0
    assert result["log_gl2"][1, 0] == 0.0
    assert result["log_gl0"][1, 1] == 0.0  # missing -> flat
