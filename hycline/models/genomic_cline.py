"""Genomic cline model: locus ancestry as a function of genome-wide hybrid
index, rather than geographic distance.

Implements the logit-logistic genomic cline (Fitzpatrick 2013; used as the
core model in both `gghybrid` (Bailey 2024) and the Bayesian hierarchical
models in `bgc-hm`/`bgchm` (Gompert & Buerkle 2011; Gompert, DeRaad &
Buerkle 2024)):

    logit(phi(h)) = u + v * logit(h)

where `h` is an individual's genome-wide hybrid index (0 = pure parental
population 0, 1 = pure parental population 1) and `phi(h)` is the expected
probability that a given locus carries the "parental population 1" allele,
given that individual's hybrid index.

NOTE ON SCOPE: this is a genuinely different model from the geographic
cline in `hycline.models.cline` -- it describes a locus's relationship to
*genome-wide admixture proportion*, not to *geographic position*. See
BACKEND_PHASE3.md for the full rationale, including why HZAR (which this
project otherwise reproduces closely) has no equivalent to cross-validate
against -- genomic clines are outside HZAR's scope entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-10


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _inv_logit(x):
    x = np.clip(x, -700, 700)  # avoid overflow in exp()
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GenomicClineParams:
    """Parameters of the logit-logistic genomic cline.

    Attributes
    ----------
    u : center/bias parameter (literature symbol `u`, sometimes written
        `c` or `alpha`). Controls phi at h=0.5: phi(0.5) = inv_logit(u).
        u=0 means the locus is unbiased at the 50% hybrid-index point;
        u != 0 indicates asymmetric introgression -- the locus's ancestry
        favors one parental genome relative to the genome-wide average.
    v : rate parameter (literature symbol `v`, sometimes `beta`). Controls
        the steepness of the transition relative to hybrid index. v=1 (with
        u=0) reproduces the neutral null expectation phi(h) = h -- the
        locus tracks genome-wide admixture exactly. v>1 means the locus
        transitions *more steeply* than the genome-wide average (more
        resistant to introgression, a signature of a barrier locus / linked
        selection against the minority ancestry near intermediate hybrid
        index). v<1 means a *shallower* transition than average (elevated
        introgression / weaker barrier at that locus relative to the rest
        of the genome). v must be positive for the cline to be monotonic
        in the conventional direction; see `fit_genomic_cline_mle`'s bounds.
    """

    u: float = 0.0
    v: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([self.u, self.v], dtype=float)

    @staticmethod
    def from_array(vec: np.ndarray) -> "GenomicClineParams":
        return GenomicClineParams(u=float(vec[0]), v=float(vec[1]))


def genomic_cline_function(h, params: GenomicClineParams) -> np.ndarray:
    """Evaluate phi(h), the expected focal-allele probability at a locus,
    for hybrid index value(s) `h` (each in [0, 1])."""
    h = np.asarray(h, dtype=float)
    return _inv_logit(params.u + params.v * _logit(h))


def param_names() -> list[str]:
    return ["u", "v"]
