import numpy as np
import pytest

from hycline.data.hybrid_index import estimate_hybrid_index


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
