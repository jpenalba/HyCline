"""Binomial likelihood for genomic cline data.

Structurally identical to `hycline.likelihoods.binomial` (same binomial
form: observed focal-allele count out of total alleles, at a predicted
probability), just evaluated against `genomic_cline_function(h, params)`
instead of the geographic `cline_function(x, params, tail_type)`. Kept as
a separate module rather than a generic shared function because the two
domains (geographic distance vs. hybrid index) have different validity
ranges and different downstream reporting needs -- see BACKEND_PHASE3.md
section 2 for why genomic and geographic clines are siblings, not the same
code path with a relabeled x-axis.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from hycline.data.schema import HybridIndexLocusData
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function

_EPS = 1e-10


def predicted_freq(data: HybridIndexLocusData, params: GenomicClineParams) -> np.ndarray:
    phi = genomic_cline_function(data.hybrid_index, params)
    return np.clip(phi, _EPS, 1.0 - _EPS)


def log_likelihood(data: HybridIndexLocusData, params: GenomicClineParams) -> float:
    """Total binomial log-likelihood of the locus data under the genomic
    cline model, summed across individuals."""
    phi = predicted_freq(data, params)
    ll = binom.logpmf(data.n_focal, data.n_alleles, phi)
    return float(np.sum(ll))


def negative_log_likelihood(param_vec: np.ndarray, data: HybridIndexLocusData) -> float:
    """Objective function for the optimizer (minimizes -log-likelihood)."""
    params = GenomicClineParams.from_array(param_vec)
    if params.v <= 0:
        return np.inf
    return -log_likelihood(data, params)
