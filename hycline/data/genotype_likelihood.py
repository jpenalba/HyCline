"""Utilities for working with genotype likelihoods: converting from the
Phred-scaled ("PL") format most variant callers output, and simulating
realistic genotype likelihoods from known true genotypes and sequencing
depth (used for this project's own validation, and useful for anyone
wanting to understand what their own low-coverage data's uncertainty
looks like before fitting anything).

See BACKEND_PHASE5.md for the full design rationale behind genotype-
likelihood-aware fitting generally.
"""

from __future__ import annotations

import numpy as np


def phred_to_log_likelihood(pl: np.ndarray) -> np.ndarray:
    """Convert Phred-scaled genotype likelihoods (PL, as commonly output
    by GATK/bcftools/ANGSD) to natural-log genotype likelihoods.

    PL is defined as PL = -10 * log10(likelihood) (up to an arbitrary
    per-site/per-individual additive constant, since PL is conventionally
    but not universally rescaled so the most likely genotype has PL=0 --
    that rescaling is exactly the kind of constant this project's
    marginal-likelihood math is unaffected by, see
    `hycline.likelihoods.genotype_likelihood`'s module docstring).

    Parameters
    ----------
    pl : array of Phred-scaled values (any shape)

    Returns
    -------
    array of the same shape, natural-log genotype likelihoods
    """
    pl = np.asarray(pl, dtype=float)
    return -pl * np.log(10.0) / 10.0


def log_likelihood_to_phred(log_gl: np.ndarray) -> np.ndarray:
    """Inverse of `phred_to_log_likelihood` -- natural-log genotype
    likelihoods back to Phred-scaled (PL) values."""
    log_gl = np.asarray(log_gl, dtype=float)
    return -log_gl * 10.0 / np.log(10.0)


def hard_genotype_to_log_likelihood(genotype: np.ndarray) -> dict:
    """Convert hard (certain) diploid genotype calls -- 0, 1, or 2 copies
    of the focal allele, NaN for missing -- to one-hot natural-log
    genotype likelihoods, for feeding into any of this project's
    genotype-likelihood-aware fitting functions/CLI commands.

    Exists specifically because constructing one-hot genotype likelihoods
    by hand (log-likelihood 0 for the true genotype, a very negative
    value for the other two) is an awkward, easy-to-get-wrong way to
    express "I have an ordinary hard-called genotype, not real sequencing
    uncertainty" -- flagged as a real gap in BACKEND_PHASE5.md section 5,
    and closed here rather than left as a one-off workaround.

    This is the direct inverse of what BACKEND_PHASE5.md section 2
    describes: marginalizing a one-hot genotype likelihood reduces
    exactly to the plain HWE likelihood for that genotype (verified in
    `tests/test_genotype_likelihood_fitting.py::
    test_hard_call_marginal_likelihood_matches_plain_hwe`) -- this
    function produces that one-hot input from an ordinary genotype call,
    so a GL-aware fit on hard-called data reproduces the exact same
    likelihood surface a non-GL fit would use directly.

    Missing genotypes (NaN) convert to a flat (uninformative) likelihood
    -- log-likelihood 0 for all three candidate genotypes -- rather than
    raising an error, so a genotype matrix with some missing calls can
    still be converted in one pass; downstream fitting then correctly
    treats those entries as carrying no information about that locus,
    without needing a separate masking step.

    A large finite negative value (-1e10) is used for the "impossible"
    genotypes rather than literal `-inf`, to keep CSV/JSON round-tripping
    well-behaved -- numerically indistinguishable from `-inf` for any
    likelihood computation in double precision (`exp(-1e10)` underflows
    to exactly `0.0` either way), but avoids edge cases some tools have
    with literal infinities in text files.

    Parameters
    ----------
    genotype : array of hard genotype calls (0, 1, or 2 copies of the
        focal allele), NaN for missing, any shape

    Returns
    -------
    dict with keys 'log_gl0', 'log_gl1', 'log_gl2', each the same shape
    as `genotype`
    """
    genotype = np.asarray(genotype, dtype=float)
    valid = ~np.isnan(genotype)
    if np.any(valid & ((genotype < 0) | (genotype > 2) | (genotype % 1 != 0))):
        raise ValueError("genotype must be 0, 1, or 2 (diploid) or NaN for missing.")

    very_negative = -1e10
    result = {}
    for g, key in [(0, "log_gl0"), (1, "log_gl1"), (2, "log_gl2")]:
        val = np.where(~valid, 0.0, np.where(genotype == g, 0.0, very_negative))
        result[key] = val
    return result


def simulate_genotype_likelihoods(true_genotypes, depth, error_rate: float = 0.01,
                                   rng: np.random.Generator | None = None) -> dict:
    """Simulate realistic genotype likelihoods from known true genotypes,
    sequencing depth, and a per-read error rate.

    Model: for a diploid individual with true genotype g in {0, 1, 2}
    (copies of the focal/alt allele), each of `depth` independent reads
    has probability `p_read_alt(g) = (g/2)*(1-error_rate) +
    (1-g/2)*error_rate` of showing the alt allele (the `(g/2)` term is the
    true fraction of alt-allele chromosome copies; the error terms account
    for reads misread as the other allele). The observed alt-read count is
    drawn `Binomial(depth, p_read_alt(g_true))`, and the genotype
    likelihood for each candidate genotype g' is then the binomial
    probability of that same observed count under `p_read_alt(g')`.

    This is a standard, simplified genotype-likelihood model (the same
    basic form underlying GATK's and ANGSD's simpler likelihood
    calculations) -- it doesn't model mapping quality, base quality
    variation within a read, or other real-world complications, but
    reproduces the essential and most important behavior: likelihoods
    sharply favor the true genotype at high depth, and become
    increasingly flat/uncertain at low depth, exactly the phenomenon
    genotype-likelihood-aware fitting exists to properly account for.

    Parameters
    ----------
    true_genotypes : array of true genotypes (0, 1, or 2), one per
        individual
    depth : sequencing depth per individual -- either a scalar (same depth
        for everyone) or an array matching `true_genotypes`' shape (e.g.
        to simulate realistically variable coverage)
    error_rate : per-read error probability (default 0.01, a fairly
        typical modern short-read sequencing error rate)
    rng : optional `numpy.random.Generator` for reproducibility

    Returns
    -------
    dict with keys 'log_gl0', 'log_gl1', 'log_gl2' (natural-log genotype
    likelihoods, one array per candidate genotype) and 'depth', 'n_alt'
    (the realized depth and alt-read count per individual, useful for
    inspection/debugging)
    """
    if rng is None:
        rng = np.random.default_rng()

    true_genotypes = np.asarray(true_genotypes, dtype=float)
    n = len(true_genotypes)
    depth = np.broadcast_to(np.asarray(depth, dtype=int), (n,)).copy()

    if np.any((true_genotypes < 0) | (true_genotypes > 2)):
        raise ValueError("true_genotypes must be 0, 1, or 2 (diploid).")
    if np.any(depth < 0):
        raise ValueError("depth must be non-negative.")

    def p_read_alt(g):
        return (g / 2.0) * (1 - error_rate) + (1 - g / 2.0) * error_rate

    p_true = p_read_alt(true_genotypes)
    n_alt = rng.binomial(depth, p_true)

    from scipy.stats import binom
    log_gls = {}
    for g, key in [(0, "log_gl0"), (1, "log_gl1"), (2, "log_gl2")]:
        p_g = p_read_alt(g)
        # depth=0 individuals carry no information -- all genotypes
        # equally (un)likely, represented here as log-likelihood 0 for
        # every genotype (an arbitrary but genotype-independent constant,
        # consistent with genotype likelihoods not needing to be
        # normalized -- see this module's docstring).
        with np.errstate(invalid="ignore"):
            ll = binom.logpmf(n_alt, depth, p_g)
        ll = np.where(depth == 0, 0.0, ll)
        log_gls[key] = ll

    return {**log_gls, "depth": depth, "n_alt": n_alt}
