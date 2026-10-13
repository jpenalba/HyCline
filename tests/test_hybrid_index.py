import numpy as np
import pytest

from hycline.data.hybrid_index import estimate_hybrid_index, estimate_hybrid_index_ml


def test_pure_parental_individuals_give_extreme_hybrid_index():
    # 3 individuals, 5 loci: individual 0 is pure pop0 (all 0s), individual
    # 2 is pure pop1 (all 2s, diploid), individual 1 is F1 (all 1s)
    genotypes = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ], dtype=float)
    h = estimate_hybrid_index(genotypes, ploidy=2)
    np.testing.assert_allclose(h, [0.0, 0.5, 1.0])


def test_missing_genotypes_excluded_from_mean():
    genotypes = np.array([
        [2.0, np.nan, 2.0],  # should average only the two non-missing loci -> 1.0 (freq)
    ])
    h = estimate_hybrid_index(genotypes, ploidy=2)
    assert h[0] == pytest.approx(1.0)


def test_weighted_by_differentiation():
    # locus 0 is highly differentiated (freq 0 vs 1), locus 1 is not (0.5 vs 0.5)
    genotypes = np.array([[2.0, 1.0]])  # individual is homozygous pop1 at locus0, het at locus1
    pop0_freq = np.array([0.0, 0.5])
    pop1_freq = np.array([1.0, 0.5])
    h_weighted = estimate_hybrid_index(
        genotypes, ploidy=2, parental_freq_pop0=pop0_freq,
        parental_freq_pop1=pop1_freq, weight_by_differentiation=True,
    )
    # locus 1 carries no information (weight 0), so h should be entirely
    # determined by locus 0 -> individual freq at locus 0 is 1.0
    assert h_weighted[0] == pytest.approx(1.0)


def test_weight_by_differentiation_requires_both_parental_freqs():
    genotypes = np.array([[2.0, 1.0]])
    with pytest.raises(ValueError, match="requires both"):
        estimate_hybrid_index(genotypes, weight_by_differentiation=True)


def test_all_zero_differentiation_raises_error():
    genotypes = np.array([[1.0, 1.0]])
    pop0_freq = np.array([0.5, 0.5])
    pop1_freq = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="zero differentiation"):
        estimate_hybrid_index(genotypes, parental_freq_pop0=pop0_freq,
                               parental_freq_pop1=pop1_freq, weight_by_differentiation=True)


def test_individual_with_all_missing_raises_error():
    genotypes = np.array([[np.nan, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError, match="no non-missing genotypes"):
        estimate_hybrid_index(genotypes)


def test_hybrid_index_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    genotypes = rng.integers(0, 3, size=(20, 10)).astype(float)
    h = estimate_hybrid_index(genotypes, ploidy=2)
    assert np.all(h >= 0) and np.all(h <= 1)


def test_loci_are_oriented_by_parental_frequencies():
    # Locus 0 counts population 1's allele, locus 1 counts population 0's.
    genotypes = np.array([[0, 2],    # pure population 0
                          [2, 0],    # pure population 1
                          [1, 1]])   # F1-like
    h = estimate_hybrid_index(genotypes, parental_freq_pop0=np.array([0.0, 1.0]),
                              parental_freq_pop1=np.array([1.0, 0.0]))
    np.testing.assert_allclose(h, [0.0, 1.0, 0.5])
    # Without parental frequencies the counts are taken as given.
    np.testing.assert_allclose(estimate_hybrid_index(genotypes), [0.5, 0.5, 0.5])


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
