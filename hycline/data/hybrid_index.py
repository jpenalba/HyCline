"""Estimate genome-wide hybrid index from multi-locus genotype data.

Hybrid index (h) is the proportion of an individual's alleles, averaged
across many ancestry-informative loci, that derive from one of two
parental reference populations. It's the x-axis variable the genomic
cline model (`hycline.models.genomic_cline`) is fit against, and this
module is what produces it in the first place from raw genotype data.

WHAT'S IMPLEMENTED: `estimate_hybrid_index_ml`, the maximum-likelihood
hybrid index. Each locus's allele frequency in an individual with hybrid
index h is (1 - h) p0 + h p1 from the parental frequencies, genotypes are
Hardy-Weinberg given that frequency, and h maximizes the likelihood of the
individual's genotypes (hard calls as one-hot likelihoods, or beagle
genotype likelihoods), with a 95% likelihood-profile interval. The same
likelihood gghybrid uses (Bailey 2023), without its Beta prior. Given the
reference panels' sizes, the parental frequencies' sampling error is
averaged over in closed form (each panel frequency's Beta posterior;
genotype probabilities are quadratic in the frequency), so the intervals
include it. The Genomic user guide (Step 2, "How the estimate is made")
walks through it step by step with a worked example.

WHAT'S NOT IMPLEMENTED: propagating uncertainty in h into the downstream
genomic cline fit, as the Bayesian models in `gghybrid` and `bgchm` do.
h is handed on as a point estimate (the interval is reported, not
used), consistent with how Fitzpatrick (2013)'s
logit-logistic model treats hybrid index as a known covariate. See
BACKEND_PHASE3.md section 5.
"""

from __future__ import annotations

import numpy as np


def estimate_hybrid_index_ml(gl0: np.ndarray, gl1: np.ndarray, gl2: np.ndarray,
                              parental_freq_pop0: np.ndarray,
                              parental_freq_pop1: np.ndarray,
                              include: np.ndarray | None = None,
                              parental_beta_pop0: tuple | None = None,
                              parental_beta_pop1: tuple | None = None,
                              freq_floor: float = 1e-3,
                              interval_drop: float | None = 1.920729) -> dict:
    """Maximum-likelihood hybrid index from per-genotype likelihoods.

    For individual i with hybrid index h, the counted allele at locus l has
    frequency p_l(h) = (1 - h) * parental_freq_pop0[l] + h * parental_freq_pop1[l],
    its genotype follows Hardy-Weinberg proportions given p_l(h), and

        log L_i(h) = sum_l log( GL0 (1-p)^2 + GL1 2p(1-p) + GL2 p^2 ),

    maximized over h in [0, 1]. Compared with averaging allele dosages:
      - no orientation step: p0/p1 are frequencies of whichever allele the
        likelihoods count, so either polarity gives the same answer;
      - partially differentiated loci are modelled, not treated as
        diagnostic (a 0.3/0.6 locus doesn't cap anyone at 0.3-0.6), and
        loci with p0 ~ p1 carry ~no weight on their own;
      - genotype uncertainty enters through the likelihoods, so low-depth
        data isn't pulled toward 0.5 the way per-individual expected
        dosages are. Hard calls (VCF) are one-hot likelihoods.

    Parameters
    ----------
    gl0, gl1, gl2 : (n_individuals, n_loci) likelihoods of 0/1/2 copies of
        the counted allele (any positive scale per individual and locus).
    parental_freq_pop0, parental_freq_pop1 : (n_loci,) frequencies of the
        counted allele in parental populations 0 and 1. Clipped to
        [freq_floor, 1 - freq_floor] so that an allele unseen in a finite
        reference panel doesn't make a locus impossible (log 0).
    include : optional (n_individuals, n_loci) bool; False marks missing
        genotypes, which are left out. Default: every non-NaN entry.
    parental_beta_pop0, parental_beta_pop1 : optional (a, b) pairs of
        (n_loci,) arrays: Beta posteriors for the parental frequencies from
        the reference panels (`hycline.data.filter_aims.panel_beta_posteriors`),
        which then replace `parental_freq_pop0/1`. The likelihood averages
        over them. Because each locus's term is quadratic in the allele
        frequency this is exact in closed form: the frequencies become the
        Beta means and the term gains gamma Var[p], with
        Var[p] = (1-h)^2 V0 + h^2 V1 (gamma below). The interval then
        reflects the panels' sampling error as well as the hybrid's own
        genotypes.
    interval_drop : log-likelihood drop defining the profile interval
        (1.92 = chi-square(1) 95% / 2); None skips the interval.

    Returns
    -------
    dict with "hybrid_index", "lower", "upper" (n_individuals,) arrays
    (lower/upper NaN when interval_drop is None) and "n_loci_used".
    """
    gl0, gl1, gl2 = (np.asarray(g, dtype=float) for g in (gl0, gl1, gl2))
    if not gl0.shape == gl1.shape == gl2.shape or gl0.ndim != 2:
        raise ValueError("gl0, gl1, gl2 must be 2D arrays of the same shape "
                         "(n_individuals, n_loci)")
    n_ind, n_loci = gl0.shape
    p0 = np.asarray(parental_freq_pop0, dtype=float)
    p1 = np.asarray(parental_freq_pop1, dtype=float)
    if p0.shape != (n_loci,) or p1.shape != (n_loci,):
        raise ValueError("parental frequency arrays must have one entry per locus")
    if (parental_beta_pop0 is None) != (parental_beta_pop1 is None):
        raise ValueError("parental_beta_pop0 and parental_beta_pop1 must be given together")
    var0 = var1 = None
    if parental_beta_pop0 is not None:
        from hycline.models.genomic_cline import beta_moments
        if any(np.shape(x) != (n_loci,) for x in (*parental_beta_pop0, *parental_beta_pop1)):
            raise ValueError("parental Beta posterior arrays must have one entry per locus")
        p0, var0 = beta_moments(*parental_beta_pop0)
        p1, var1 = beta_moments(*parental_beta_pop1)
    p0 = np.clip(p0, freq_floor, 1 - freq_floor)
    p1 = np.clip(p1, freq_floor, 1 - freq_floor)

    total = gl0 + gl1 + gl2
    usable = np.isfinite(total) & (total > 0)
    if include is not None:
        usable &= np.asarray(include, dtype=bool)
    # Loci with no parental frequency estimate can't inform h.
    usable &= (np.isfinite(p0) & np.isfinite(p1))[None, :]
    n_used = usable.sum(axis=1)
    if np.any(n_used == 0):
        raise ValueError(
            "At least one individual has no usable genotype at any locus with "
            "parental frequencies -- cannot estimate hybrid index for that individual.")

    # Each locus's term is a quadratic in h: with p = p0 + h * dp,
    #   GL0 (1-p)^2 + GL1 2p(1-p) + GL2 p^2 = A + B h + C h^2.
    # Precomputing A, B, C makes every evaluation of log L(h) cheap.
    # Unusable entries get equal likelihoods, which add the constant
    # log(1/3) to log L(h) and so don't affect the estimate.
    with np.errstate(invalid="ignore", divide="ignore"):
        L0, L1, L2 = (np.where(usable, g / total, 1.0 / 3.0) for g in (gl0, gl1, gl2))
    del total, usable
    alpha, beta, gamma = L0, 2 * (L1 - L0), L0 - 2 * L1 + L2
    del L0, L1, L2
    dp = (p1 - p0)[None, :]
    A = alpha + beta * p0[None, :] + gamma * p0[None, :] ** 2
    B = (beta + 2 * gamma * p0[None, :]) * dp
    C = gamma * dp ** 2
    if var0 is not None:
        # E[p^2] = E[p]^2 + (1-h)^2 V0 + h^2 V1 over the panel posteriors.
        v0, v1 = np.nan_to_num(var0)[None, :], np.nan_to_num(var1)[None, :]
        A = A + gamma * v0
        B = B - 2 * gamma * v0
        C = C + gamma * (v0 + v1)
    del alpha, beta, gamma
    chunk = max(1, 4_000_000 // max(n_ind, 1))

    def loglik(h: np.ndarray) -> np.ndarray:
        """log L_i(h_i) for a per-individual vector h."""
        out = np.zeros(n_ind)
        hh = h[:, None]
        for a in range(0, n_loci, chunk):
            b = min(a + chunk, n_loci)
            out += np.log(A[:, a:b] + hh * (B[:, a:b] + hh * C[:, a:b])).sum(axis=1)
        return out

    def dloglik(h: np.ndarray) -> np.ndarray:
        """d log L_i / dh at h_i."""
        out = np.zeros(n_ind)
        hh = h[:, None]
        for a in range(0, n_loci, chunk):
            b = min(a + chunk, n_loci)
            num = B[:, a:b] + 2 * hh * C[:, a:b]
            den = A[:, a:b] + hh * (B[:, a:b] + hh * C[:, a:b])
            out += (num / den).sum(axis=1)
        return out

    # Coarse grid for the global maximum (log L(h) need not be unimodal),
    # then bisection on the sign of the derivative within the neighbouring
    # grid cells; a maximum on the 0/1 boundary is reached as a limit.
    grid = np.linspace(0.0, 1.0, 21)
    step = grid[1] - grid[0]
    ll_grid = np.stack([loglik(np.full(n_ind, g)) for g in grid], axis=1)
    k = np.argmax(ll_grid, axis=1)
    lo = grid[np.maximum(k - 1, 0)]
    hi = grid[np.minimum(k + 1, grid.size - 1)]
    for _ in range(30):  # bracket width 0.1 / 2^30 ~ 1e-10
        mid = (lo + hi) / 2
        rising = dloglik(mid) > 0
        lo = np.where(rising, mid, lo)
        hi = np.where(rising, hi, mid)
    h_hat = (lo + hi) / 2
    ll_hat = loglik(h_hat)
    rows = np.arange(n_ind)
    better = ll_grid[rows, k] > ll_hat
    h_hat = np.where(better, grid[k], h_hat)
    ll_hat = np.where(better, ll_grid[rows, k], ll_hat)

    lower = np.full(n_ind, np.nan)
    upper = np.full(n_ind, np.nan)
    if interval_drop is not None:
        target = ll_hat - interval_drop

        def bound(direction: int) -> np.ndarray:
            # Walk the grid outward from h_hat to the first point below the
            # target, then bisect within that cell. If no grid point is
            # below it, the interval reaches the edge (0 or 1).
            below = ll_grid < target[:, None]
            if direction < 0:
                cand = np.where(below & (grid[None, :] < h_hat[:, None]), grid[None, :], -np.inf)
                out_pt = cand.max(axis=1)
                edge = 0.0
            else:
                cand = np.where(below & (grid[None, :] > h_hat[:, None]), grid[None, :], np.inf)
                out_pt = cand.min(axis=1)
                edge = 1.0
            reaches_edge = ~np.isfinite(out_pt)
            outside = np.where(reaches_edge, edge, out_pt)
            inside = np.where(reaches_edge, edge,
                              np.clip(outside - direction * step,
                                      np.minimum(h_hat, outside), np.maximum(h_hat, outside)))
            # the inside end must be at or above the target
            inside = np.where(loglik(inside) >= target, inside, h_hat)
            for _ in range(30):
                mid = (inside + outside) / 2
                ok = loglik(mid) >= target
                inside = np.where(ok, mid, inside)
                outside = np.where(ok, outside, mid)
            return np.where(reaches_edge, edge, (inside + outside) / 2)

        lower, upper = bound(-1), bound(1)

    return {"hybrid_index": h_hat, "lower": lower, "upper": upper,
            "n_loci_used": n_used}
