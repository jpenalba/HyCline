"""Estimate genome-wide hybrid index from multi-locus genotype data.

Hybrid index (h) is the proportion of an individual's alleles, averaged
across many ancestry-informative loci, that derive from one of two
parental reference populations. It's the x-axis variable the genomic
cline model (`hycline.models.genomic_cline`) is fit against, and this
module is what produces it in the first place from raw genotype data.

WHAT'S IMPLEMENTED: a simple mean-proportion estimator -- for each
individual, the mean (across loci) of that individual's observed
"population-1" allele frequency, optionally weighted by each locus's
allele-frequency differentiation between the parental populations. This
is a standard, defensible baseline estimator, and is exactly what you'd
compute by hand as "fraction of alleles matching parental population 1
across N markers."

WHAT'S NOT IMPLEMENTED: the more sophisticated Bayesian hybrid-index
estimators in `gghybrid` and `bgchm` model each locus's contribution with
an explicit likelihood and place a prior over h, propagating uncertainty
in h itself into the downstream genomic cline fit. The mean-proportion
estimator here treats h as a fixed, known covariate once estimated --
consistent with how Fitzpatrick (2013)'s original logit-logistic model
description treats hybrid index, but a real simplification relative to
the current state of the art. Flagged here rather than silently matched
to a more sophisticated method it isn't. See BACKEND_PHASE3.md section 3.
"""

from __future__ import annotations

import numpy as np


def estimate_hybrid_index(genotypes: np.ndarray, ploidy: int = 2,
                           parental_freq_pop0: np.ndarray | None = None,
                           parental_freq_pop1: np.ndarray | None = None,
                           weight_by_differentiation: bool = False) -> np.ndarray:
    """Estimate each individual's hybrid index as the mean, across loci, of
    that individual's population-1 allele frequency.

    Parameters
    ----------
    genotypes : (n_individuals, n_loci) array of population-1 allele
        counts per individual per locus (0..ploidy). Use NaN for
        missing genotypes -- they're excluded from that individual's
        mean rather than treated as 0.
    ploidy : ploidy of the genotype calls (default 2, diploid).
    parental_freq_pop0, parental_freq_pop1 : optional (n_loci,) arrays of
        each locus's frequency, in parental reference population 0 and
        population 1, of the allele `genotypes` counts. When both are
        given, every locus is first oriented so the counted allele is the
        one more common in population 1: a locus with
        parental_freq_pop1 < parental_freq_pop0 has its genotypes flipped
        (ploidy - count). Without this, a VCF's ALT allele -- which is
        population 0's allele at roughly half of real AIMs -- cancels
        out across loci and every individual lands near 0.5. Already-
        oriented input is unchanged. They also feed
        `weight_by_differentiation`.
    weight_by_differentiation : if True (requires both parental frequency
        arrays), weight each locus's contribution by
        abs(parental_freq_pop1 - parental_freq_pop0) -- a locus fixed for
        different alleles in the two parental populations contributes
        fully; a locus with similar frequencies in both contributes little.
        If False (default), every locus contributes equally, which is
        appropriate if you've already pre-filtered to strongly
        differentiated (ancestry-informative) markers.

    Returns
    -------
    (n_individuals,) array of hybrid index estimates in [0, 1].
    """
    genotypes = np.asarray(genotypes, dtype=float)
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be a 2D array (n_individuals, n_loci)")

    if parental_freq_pop0 is not None and parental_freq_pop1 is not None:
        flip = (np.asarray(parental_freq_pop1, dtype=float)
                < np.asarray(parental_freq_pop0, dtype=float))
        genotypes = np.where(flip[None, :], ploidy - genotypes, genotypes)

    freq = genotypes / ploidy  # per-individual, per-locus population-1 allele frequency

    if weight_by_differentiation:
        if parental_freq_pop0 is None or parental_freq_pop1 is None:
            raise ValueError(
                "weight_by_differentiation=True requires both parental_freq_pop0 "
                "and parental_freq_pop1."
            )
        weights = np.abs(np.asarray(parental_freq_pop1, dtype=float)
                          - np.asarray(parental_freq_pop0, dtype=float))
        if np.all(weights == 0):
            raise ValueError(
                "All loci have zero differentiation between the parental "
                "populations (parental_freq_pop1 == parental_freq_pop0 everywhere) "
                "-- these loci carry no ancestry information."
            )
    else:
        weights = np.ones(genotypes.shape[1])

    valid = ~np.isnan(freq)
    weights_matrix = np.broadcast_to(weights, freq.shape) * valid

    weight_sums = weights_matrix.sum(axis=1)
    if np.any(weight_sums == 0):
        raise ValueError(
            "At least one individual has no non-missing genotypes at any "
            "weighted locus -- cannot estimate hybrid index for that individual."
        )

    weighted_freq = np.nan_to_num(freq, nan=0.0) * weights_matrix
    return weighted_freq.sum(axis=1) / weight_sums
