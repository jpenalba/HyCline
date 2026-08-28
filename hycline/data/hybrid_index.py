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


def polarize_genotypes(genotypes: np.ndarray, parental_freq_pop0: np.ndarray,
                        parental_freq_pop1: np.ndarray, ploidy: int = 2) -> tuple:
    """Re-orient per-locus allele dosage to consistently count the
    "population-1" allele, as `estimate_hybrid_index` requires (see its
    own docstring: "genotypes ... population-1 allele counts").

    A REAL BUG THIS FIXES, not a defensive nicety: which allele a VCF
    calls REF vs. ALT is arbitrary with respect to which parental
    population carries more of it -- completely normal, expected variant
    -calling behavior, not an edge case. Raw ALT-allele dosage (what
    `hycline.data.vcf`/`hycline.data.beagle` produce) is therefore only
    "population-1 allele dosage" at loci where population 1 happens to
    be the more-ALT population; at a locus where population 0 is more
    -ALT instead, the raw dosage points the opposite direction. Averaging
    unpolarized dosage across loci with mixed orientations pulls every
    individual's estimated hybrid index toward 0.5 regardless of true
    ancestry -- confirmed directly: a synthetic 2-locus VCF with one
    normally-oriented and one reversed-oriented locus made every
    individual, including the four known-pure parental reference
    individuals, come back at hybrid_index=0.500.

    For each locus where `parental_freq_pop0 > parental_freq_pop1`
    (population 0 is the more-ALT population there), this flips that
    locus's dosage (`ploidy - dosage`) and complements its parental
    frequencies (`1 - freq`) so that, after polarization, population 1
    is always the more-"focal-allele" population at every locus,
    matching what `estimate_hybrid_index` assumes. Loci where the two
    parental frequencies are equal (locus provides no orientation
    information either way) or where either is NaN (e.g. no reference
    individuals had a non-missing genotype there) are left unflipped --
    the comparison is False either way, which is the conservative,
    correct choice since there's nothing to orient by.

    Parameters
    ----------
    genotypes : (n_individuals, n_loci) array of raw allele dosage
        (0..ploidy per individual per locus, NaN for missing).
    parental_freq_pop0, parental_freq_pop1 : (n_loci,) arrays of each
        locus's allele frequency in parental reference population 0/1,
        in the SAME raw orientation `genotypes` uses (i.e. computed
        directly from the same raw dosage, not already polarized).
    ploidy : ploidy of the genotype calls (default 2, diploid).

    Returns
    -------
    (polarized_genotypes, polarized_freq_pop0, polarized_freq_pop1) --
    same shapes as the inputs, ready to pass to `estimate_hybrid_index`.
    """
    genotypes = np.asarray(genotypes, dtype=float)
    parental_freq_pop0 = np.asarray(parental_freq_pop0, dtype=float)
    parental_freq_pop1 = np.asarray(parental_freq_pop1, dtype=float)
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be a 2D array (n_individuals, n_loci)")
    if not (genotypes.shape[1] == parental_freq_pop0.shape[0] == parental_freq_pop1.shape[0]):
        raise ValueError(
            "genotypes.shape[1] (n_loci) must match the length of both "
            "parental_freq_pop0 and parental_freq_pop1."
        )

    flip = parental_freq_pop0 > parental_freq_pop1  # NaN comparisons are False -- left unflipped
    polarized_genotypes = np.where(flip, ploidy - genotypes, genotypes)
    polarized_freq_pop0 = np.where(flip, 1.0 - parental_freq_pop0, parental_freq_pop0)
    polarized_freq_pop1 = np.where(flip, 1.0 - parental_freq_pop1, parental_freq_pop1)
    return polarized_genotypes, polarized_freq_pop0, polarized_freq_pop1


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
        each locus's allele frequency in parental reference population 0
        and population 1. If given, only loci with a real frequency
        difference between the two parental populations carry ancestry
        information; `weight_by_differentiation` uses this to down-weight
        loci with little differentiation rather than treating every locus
        as equally informative.
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
