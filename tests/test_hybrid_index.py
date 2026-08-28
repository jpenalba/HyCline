import numpy as np
import pytest

from hycline.data.hybrid_index import estimate_hybrid_index, polarize_genotypes


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


def test_polarize_genotypes_flips_reversed_orientation_locus():
    # Regression test for a real bug found during a code-review pass and
    # verified live end-to-end through the CLI: without polarization,
    # mixing a normally-oriented locus (pop1 more-ALT) with a
    # reversed-oriented one (pop0 more-ALT -- a completely normal
    # consequence of arbitrary VCF REF/ALT assignment, not a data
    # problem) pulled every individual's estimated hybrid index toward
    # 0.5 regardless of true ancestry, including the four parental
    # reference individuals themselves who should read near 0 or 1.
    #
    # locusA: normal orientation (pop1 is more-ALT: freq_pop0=0, freq_pop1=1)
    # locusB: reversed orientation (pop0 is more-ALT: freq_pop0=1, freq_pop1=0)
    genotypes = np.array([
        [0.0, 2.0],  # pure pop0 individual: 0/0 at locusA, 1/1 (ALT) at locusB
        [0.0, 2.0],
        [2.0, 0.0],  # pure pop1 individual: 1/1 (ALT) at locusA, 0/0 at locusB
        [2.0, 0.0],
        [2.0, 0.0],  # test individual, matches pop1's pattern
    ])
    parental_freq_pop0 = np.array([0.0, 1.0])
    parental_freq_pop1 = np.array([1.0, 0.0])

    polarized_genotypes, polarized_freq_pop0, polarized_freq_pop1 = polarize_genotypes(
        genotypes, parental_freq_pop0, parental_freq_pop1, ploidy=2)

    # locusB should be flipped (pop0 > pop1 there); locusA untouched.
    np.testing.assert_allclose(polarized_freq_pop0, [0.0, 0.0])
    np.testing.assert_allclose(polarized_freq_pop1, [1.0, 1.0])
    np.testing.assert_allclose(polarized_genotypes[:, 0], genotypes[:, 0])
    np.testing.assert_allclose(polarized_genotypes[:, 1], 2.0 - genotypes[:, 1])

    h = estimate_hybrid_index(polarized_genotypes, ploidy=2,
                               parental_freq_pop0=polarized_freq_pop0,
                               parental_freq_pop1=polarized_freq_pop1)
    # Parental reference individuals now correctly land at the extremes,
    # not 0.5 -- this is the concrete, verified fix.
    np.testing.assert_allclose(h, [0.0, 0.0, 1.0, 1.0, 1.0])


def test_polarize_genotypes_leaves_equal_or_nan_frequencies_unflipped():
    # A locus with equal parental frequencies (no orientation information
    # either way) or a NaN parental frequency (e.g. no reference
    # individuals had a non-missing genotype there) must not be flipped --
    # the flip condition (pop0 > pop1) is False for both NaN comparisons
    # and exact equality, which is the correct, conservative choice.
    genotypes = np.array([[1.0, 1.0]])
    parental_freq_pop0 = np.array([0.5, np.nan])
    parental_freq_pop1 = np.array([0.5, 0.3])

    polarized_genotypes, polarized_freq_pop0, polarized_freq_pop1 = polarize_genotypes(
        genotypes, parental_freq_pop0, parental_freq_pop1, ploidy=2)

    np.testing.assert_allclose(polarized_genotypes, genotypes)
    np.testing.assert_allclose(polarized_freq_pop0[0], 0.5)
    assert np.isnan(polarized_freq_pop0[1])
    np.testing.assert_allclose(polarized_freq_pop1, parental_freq_pop1)


def test_polarize_genotypes_rejects_mismatched_locus_count():
    genotypes = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="n_loci"):
        polarize_genotypes(genotypes, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
