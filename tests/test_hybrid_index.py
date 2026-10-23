import numpy as np
import pytest

from hycline.data.hybrid_index import estimate_hybrid_index_ml


# ---------------------------------------------------------------------------
# estimate_hybrid_index_ml
# ---------------------------------------------------------------------------

def _simulate_genotypes(seed=3, n_ind=60, n_loci=200):
    rng = np.random.default_rng(seed)
    p0, p1 = rng.uniform(0, 1, n_loci), rng.uniform(0, 1, n_loci)
    h = rng.uniform(0, 1, n_ind)
    h[:5], h[5:10] = 0.0, 1.0
    g = rng.binomial(2, (1 - h)[:, None] * p0 + h[:, None] * p1)
    return h, g, p0, p1


def _one_hot(g):
    return [(g == k).astype(float) for k in range(3)]


def test_ml_matches_brute_force_likelihood_maximum_and_interval():
    h, g, p0, p1 = _simulate_genotypes()
    r = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)
    grid = np.linspace(0, 1, 20001)
    q0, q1 = np.clip(p0, 1e-3, 1 - 1e-3), np.clip(p1, 1e-3, 1 - 1e-3)
    pp = (1 - grid)[:, None] * q0 + grid[:, None] * q1
    P = np.stack([(1 - pp) ** 2, 2 * pp * (1 - pp), pp ** 2], axis=-1)
    for i in range(20):  # includes the h = 0 and h = 1 individuals
        L = np.stack([x[i] for x in _one_hot(g)], axis=-1)
        ll = np.log((P * L[None]).sum(axis=-1)).sum(axis=1)
        inside = grid[ll >= ll.max() - 1.920729]
        assert r["hybrid_index"][i] == pytest.approx(grid[np.argmax(ll)], abs=1e-4)
        assert r["lower"][i] == pytest.approx(inside.min(), abs=1e-4)
        assert r["upper"][i] == pytest.approx(inside.max(), abs=1e-4)


def test_ml_is_invariant_to_allele_polarity():
    _, g, p0, p1 = _simulate_genotypes()
    base = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)["hybrid_index"]
    flip = np.zeros(g.shape[1], bool)
    flip[::2] = True
    g_f = np.where(flip, 2 - g, g)
    p0_f, p1_f = np.where(flip, 1 - p0, p0), np.where(flip, 1 - p1, p1)
    np.testing.assert_allclose(estimate_hybrid_index_ml(*_one_hot(g_f), p0_f, p1_f)["hybrid_index"],
                               base, atol=1e-8)


def test_ml_missing_genotypes_are_the_same_as_dropping_them():
    _, g, p0, p1 = _simulate_genotypes()
    include = np.ones(g.shape, bool)
    include[:, :50] = False
    with_missing = estimate_hybrid_index_ml(*_one_hot(g), p0, p1, include=include)
    dropped = estimate_hybrid_index_ml(*[x[:, 50:] for x in _one_hot(g)], p0[50:], p1[50:])
    np.testing.assert_allclose(with_missing["hybrid_index"], dropped["hybrid_index"], atol=1e-8)
    assert (with_missing["n_loci_used"] == 150).all()


def test_ml_tolerates_alleles_unseen_in_the_reference_panels():
    # A hybrid carries the alternate allele at a locus fixed for the
    # reference allele in both panels: without the frequency floor this
    # locus would make every h impossible (log 0).
    g = np.array([[2, 0, 0, 2, 2]])
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    r = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)
    assert np.isfinite(r["hybrid_index"]).all()
    assert r["hybrid_index"][0] == pytest.approx(0.5, abs=0.02)


def test_ml_individual_without_usable_loci_raises():
    _, g, p0, p1 = _simulate_genotypes(n_ind=3, n_loci=10)
    include = np.ones(g.shape, bool)
    include[1] = False
    with pytest.raises(ValueError, match="no usable genotype"):
        estimate_hybrid_index_ml(*_one_hot(g), p0, p1, include=include)
