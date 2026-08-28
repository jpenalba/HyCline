"""Direct construction-validation tests for hycline.data.schema's typed
containers -- regression tests for real gaps found during a code-review
pass, not general coverage of the containers themselves (which are
exercised indirectly throughout the rest of the test suite)."""

import numpy as np
import pytest

from hycline.data.schema import (
    AlleleFreqTransect, GenotypeLikelihoodTransect, HybridIndexLocusData, TraitTransect,
    values_consistent, polarize_allele_freq_transect, polarize_genotype_likelihood_transect,
)


def test_allele_freq_transect_rejects_empty_arrays():
    # Regression test: AlleleFreqTransect used to construct silently with
    # zero-length arrays (every np.any(...) validation check is vacuously
    # False on an empty array), unlike GenotypeLikelihoodTransect in the
    # same file, which explicitly rejects an empty transect.
    with pytest.raises(ValueError, match="at least one site"):
        AlleleFreqTransect(distance=np.array([]), n_alleles=np.array([]),
                            n_focal=np.array([]))


def test_hybrid_index_locus_data_rejects_empty_arrays():
    with pytest.raises(ValueError, match="at least one individual"):
        HybridIndexLocusData(hybrid_index=np.array([]), n_alleles=np.array([]),
                              n_focal=np.array([]))


def test_trait_transect_rejects_empty_arrays():
    with pytest.raises(ValueError, match="at least one site"):
        TraitTransect(distance=np.array([]), trait_mean=np.array([]), n=np.array([]))


def test_allele_freq_transect_rejects_negative_n_focal():
    # Regression test: n_focal was checked against the upper bound
    # (n_alleles) but never checked for being negative -- a negative
    # n_focal constructed silently and then made log_likelihood silently
    # return -inf instead of the constructor raising its own clear,
    # data-identifying error (as it already does for n_focal>n_alleles).
    with pytest.raises(ValueError, match="negative"):
        AlleleFreqTransect(distance=[0.], n_alleles=[10.], n_focal=[-2.])


def test_hybrid_index_locus_data_rejects_negative_n_focal():
    with pytest.raises(ValueError, match="negative"):
        HybridIndexLocusData(hybrid_index=[0.5], n_alleles=[10.], n_focal=[-2.])


def test_polarize_allele_freq_transect_flips_when_reversed():
    # Frequency is high at distance 0, low at the far end -- the
    # opposite of the desired "low near, high far" convention -- so
    # should be flipped (n_focal -> n_alleles - n_focal).
    transect = AlleleFreqTransect(
        distance=[0., 10., 20.], n_alleles=[10., 10., 10.], n_focal=[9., 5., 1.])
    new_transect, flipped = polarize_allele_freq_transect(transect)
    assert flipped is True
    np.testing.assert_array_equal(new_transect.n_focal, [1., 5., 9.])
    np.testing.assert_array_equal(new_transect.n_alleles, transect.n_alleles)
    np.testing.assert_array_equal(new_transect.distance, transect.distance)


def test_polarize_allele_freq_transect_leaves_correctly_oriented_alone():
    transect = AlleleFreqTransect(
        distance=[0., 10., 20.], n_alleles=[10., 10., 10.], n_focal=[1., 5., 9.])
    new_transect, flipped = polarize_allele_freq_transect(transect)
    assert flipped is False
    np.testing.assert_array_equal(new_transect.n_focal, transect.n_focal)


def test_polarize_allele_freq_transect_ignores_site_order_in_input():
    # Same data as the "flip" case above but rows given out of distance
    # order -- the function must sort by distance itself, not assume
    # the input is already sorted.
    transect = AlleleFreqTransect(
        distance=[20., 0., 10.], n_alleles=[10., 10., 10.], n_focal=[1., 9., 5.])
    new_transect, flipped = polarize_allele_freq_transect(transect)
    assert flipped is True


def test_polarize_allele_freq_transect_pools_n_edge_sites():
    # Single nearest site at each end would call this "already correctly
    # oriented" (site0's own frequency, 0.1, is lower than site3's, 0.2)
    # but pooling n_edge_sites=2 at each end reveals the true reversed
    # trend (the two sites nearest distance=0 pool to freq 0.5, higher
    # than the two nearest the far end, which pool to 0.15) -- exactly
    # the sample-size-robustness case _derive_pmin_pmax's own
    # n_edge_sites was built for.
    transect = AlleleFreqTransect(
        distance=[0., 1., 2., 3.],
        n_alleles=[10., 10., 10., 10.],
        n_focal=[1., 9., 1., 2.])
    # site0 (near end) freq=0.1, site3 (far end) freq=0.2: n_edge_sites=1
    # alone would see this as already "low near end" and NOT flip --
    # confirm that baseline first.
    _, flipped_default = polarize_allele_freq_transect(transect, n_edge_sites=1)
    assert flipped_default is False

    new_transect, flipped_pooled = polarize_allele_freq_transect(transect, n_edge_sites=2)
    assert flipped_pooled is True


def test_polarize_allele_freq_transect_ties_do_not_flip():
    # freq_low == freq_high exactly: not a reversed trend, so no flip
    # (the <= in the comparison, not <, matters here).
    transect = AlleleFreqTransect(
        distance=[0., 10., 20.], n_alleles=[10., 10., 10.], n_focal=[5., 5., 5.])
    new_transect, flipped = polarize_allele_freq_transect(transect)
    assert flipped is False
    np.testing.assert_array_equal(new_transect.n_focal, transect.n_focal)


def test_polarize_genotype_likelihood_transect_flips_when_reversed():
    # Individuals near distance=0 mostly homozygous focal (log_gl2 high),
    # individuals near the far end mostly homozygous non-focal (log_gl0
    # high) -- reversed orientation, should flip (log_gl0 <-> log_gl2).
    transect = GenotypeLikelihoodTransect(
        distance=[0., 0., 20., 20.],
        log_gl0=[-10., -10., 0., 0.],
        log_gl1=[-10., -10., -10., -10.],
        log_gl2=[0., 0., -10., -10.],
    )
    new_transect, flipped = polarize_genotype_likelihood_transect(transect)
    assert flipped is True
    np.testing.assert_array_equal(new_transect.log_gl0, transect.log_gl2)
    np.testing.assert_array_equal(new_transect.log_gl2, transect.log_gl0)
    np.testing.assert_array_equal(new_transect.log_gl1, transect.log_gl1)


def test_polarize_genotype_likelihood_transect_leaves_correctly_oriented_alone():
    transect = GenotypeLikelihoodTransect(
        distance=[0., 0., 20., 20.],
        log_gl0=[0., 0., -10., -10.],
        log_gl1=[-10., -10., -10., -10.],
        log_gl2=[-10., -10., 0., 0.],
    )
    new_transect, flipped = polarize_genotype_likelihood_transect(transect)
    assert flipped is False
    np.testing.assert_array_equal(new_transect.log_gl0, transect.log_gl0)
    np.testing.assert_array_equal(new_transect.log_gl2, transect.log_gl2)


def test_values_consistent_treats_nan_as_consistent_with_nan():
    # Regression test for the shared helper behind the NaN-comparison
    # bug found in 5 different site/individual-consistency checks across
    # the codebase (vcf.py, beagle.py, converter.py, io.py, geo.py):
    # ordinary `!=` comparison treats NaN != NaN as True, so a site where
    # every individual genuinely has the same missing value used to raise
    # a spurious "inconsistent" error.
    assert values_consistent(np.nan, np.nan) is True
    assert values_consistent(1.0, 1.0) is True
    assert values_consistent(1.0, 2.0) is False
    assert values_consistent(np.nan, 1.0) is False
    assert values_consistent(1.0, np.nan) is False
