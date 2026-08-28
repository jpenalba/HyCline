"""Direct construction-validation tests for hycline.data.schema's typed
containers -- regression tests for real gaps found during a code-review
pass, not general coverage of the containers themselves (which are
exercised indirectly throughout the rest of the test suite)."""

import numpy as np
import pytest

from hycline.data.schema import (
    AlleleFreqTransect, HybridIndexLocusData, TraitTransect, values_consistent,
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
