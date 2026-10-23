"""Genotype-likelihood-aware likelihood functions: marginalizing over
unobserved (uncertain) genotypes rather than assuming genotype counts are
known with certainty.

Standard `hycline.likelihoods.binomial`/`genomic_binomial` treat the
data as a known count of focal-allele copies out of a known total --
appropriate for confidently-called genotypes, but not for genotypes
called from low-coverage sequencing, where the true genotype is
genuinely uncertain and that uncertainty should propagate into the cline
fit rather than being discarded by a hard call. This module implements
the correct way to do that: marginalizing the likelihood over the three
possible diploid genotypes, weighted by (a) the sequencing-derived
genotype likelihood for each candidate genotype and (b) the
Hardy-Weinberg-expected probability of each genotype given the cline
model's predicted allele frequency.

See BACKEND_PHASE5.md for the full derivation, its relationship to the
project's own previously-deferred multinomial genotype likelihood (this
turns out to be a strict generalization of it -- see section 2), and the
validation showing why this matters in practice (a real, demonstrated
bias in naive hard-genotype-calling at low sequencing depth).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from hycline.data.schema import GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_allele_freq

_EPS = 1e-10


def _hwe_log_genotype_probs(p):
    """log P(genotype=0|p), log P(genotype=1|p), log P(genotype=2|p)
    under Hardy-Weinberg equilibrium, for a diploid biallelic locus with
    focal-allele frequency p."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log((1 - p) ** 2), np.log(2 * p * (1 - p)), np.log(p ** 2)


def _marginal_log_likelihood(log_gl0, log_gl1, log_gl2, p):
    """Per-individual marginal log-likelihood: log sum_g GL_g * P(g|p),
    computed via logsumexp for numerical stability."""
    log_p0, log_p1, log_p2 = _hwe_log_genotype_probs(p)
    terms = np.stack([log_gl0 + log_p0, log_gl1 + log_p1, log_gl2 + log_p2], axis=0)
    return logsumexp(terms, axis=0)


# ---------------------------------------------------------------------------
# Geographic clines
# ---------------------------------------------------------------------------

def log_likelihood_geographic(transect: GenotypeLikelihoodTransect, params: ClineParams,
                               tail_type: TailType = TailType.NONE) -> float:
    """Total genotype-likelihood-marginalized log-likelihood of a
    geographic cline transect, summed across individuals."""
    p = cline_function(transect.distance, params, tail_type)
    ll = _marginal_log_likelihood(transect.log_gl0, transect.log_gl1, transect.log_gl2, p)
    return float(np.sum(ll))


def negative_log_likelihood_geographic(param_vec: np.ndarray,
                                        transect: GenotypeLikelihoodTransect,
                                        tail_type: TailType) -> float:
    params = ClineParams.from_array(param_vec, tail_type)
    if params.width <= 0:
        return np.inf
    if not (0.0 <= params.pmin <= 1.0 and 0.0 <= params.pmax <= 1.0):
        return np.inf
    return -log_likelihood_geographic(transect, params, tail_type)


# ---------------------------------------------------------------------------
# Genomic clines
# ---------------------------------------------------------------------------

def log_likelihood_genomic(data: GenotypeLikelihoodLocusData,
                            params: GenomicClineParams) -> float:
    """Total genotype-likelihood-marginalized log-likelihood of a genomic
    cline locus, summed across individuals."""
    p = genomic_cline_allele_freq(data.hybrid_index, params,
                                  data.parental_freq_pop0, data.parental_freq_pop1)
    ll = _marginal_log_likelihood(data.log_gl0, data.log_gl1, data.log_gl2, p)
    return float(np.sum(ll))


def negative_log_likelihood_genomic(param_vec: np.ndarray,
                                     data: GenotypeLikelihoodLocusData) -> float:
    params = GenomicClineParams.from_array(param_vec)
    if params.v <= 0:
        return np.inf
    return -log_likelihood_genomic(data, params)
