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


def genomic_cline_allele_freq(h, params: GenomicClineParams,
                              parental_freq_pop0: float = 0.0,
                              parental_freq_pop1: float = 1.0) -> np.ndarray:
    """Expected frequency of the counted allele at hybrid index `h`:
    p0 + (p1 - p0) * phi(h), with p0/p1 the allele's frequency in parental
    populations 0 and 1 (Bailey 2023, gghybrid, section 2.1.1; the
    allele-frequency form of the logit-logistic cline). phi(h) is the
    probability that an allele copy derives from parental population 1,
    so the cline describes ancestry even when the parental populations
    aren't fixed for different alleles, and either allele can be the one
    counted (p1 < p0 gives a decreasing frequency for increasing phi).
    The defaults p0 = 0, p1 = 1 give phi(h) itself: the counted allele is
    treated as fully diagnostic of parental population 1."""
    phi = genomic_cline_function(h, params)
    return parental_freq_pop0 + (parental_freq_pop1 - parental_freq_pop0) * phi


def param_names() -> list[str]:
    return ["u", "v"]


def beta_quadrature(a: float, b: float, k: int):
    """Nodes, weights (summing to 1) and the density's log at the nodes,
    for averaging a smooth function over Beta(a, b). Gauss-Jacobi, exact
    for polynomials of degree up to 2k - 1 and unaffected by the density's
    singularity at 0 or 1 when a or b < 1; for a + b > 1000, where
    Gauss-Jacobi overflows (scipy's rule computes 2^(a + b - 1)) and the
    Beta is close to normal, Gauss-Hermite on that normal (whose log
    density is then the one returned)."""
    from scipy.special import roots_hermitenorm, roots_jacobi
    from scipy.stats import beta as beta_dist, norm
    a, b = float(a), float(b)
    if a + b <= 1000:
        x, w = roots_jacobi(k, b - 1.0, a - 1.0)  # weight (1-x)^(b-1) (1+x)^(a-1)
        nodes = np.clip((x + 1.0) / 2.0, 1e-12, 1.0 - 1e-12)
        log_density = beta_dist.logpdf(nodes, a, b)
    else:
        x, w = roots_hermitenorm(k)
        mean = a / (a + b)
        sd = np.sqrt(mean * (1.0 - mean) / (a + b + 1.0))
        nodes = np.clip(mean + sd * x, 1e-12, 1.0 - 1e-12)
        log_density = norm.logpdf(nodes, mean, sd)
    return nodes, w / w.sum(), log_density


def parental_nodes(panel_betas, node_betas, k: int):
    """Quadrature over two independent parental frequencies with Beta
    posteriors `panel_betas` = [(a0, b0), (a1, b1)]: a k x k grid of
    (p0, p1) nodes and log weights, so that sum(exp(log_w) * f(p0, p1))
    approximates E[f]. Nodes come from Gauss-Jacobi rules for `node_betas`
    (the panel posteriors themselves, or Betas nearer where f is large),
    each weighted by panel density / node density. Returns flat
    (k * k,) arrays p0_nodes, p1_nodes, log_w."""
    from scipy.stats import beta as beta_dist
    nodes, log_w = [], []
    for (a, b), (pa, pb) in zip(node_betas, panel_betas):
        q, w, log_density = beta_quadrature(a, b, k)
        nodes.append(q)
        log_w.append(np.log(w) + beta_dist.logpdf(q, pa, pb) - log_density)
    p0_nodes, p1_nodes = (g.ravel() for g in np.meshgrid(*nodes, indexing="ij"))
    return p0_nodes, p1_nodes, np.add.outer(*log_w).ravel()


def widened_beta(mean, var, factor: float):
    """Beta with the given mean and `factor` times the given variance, for
    placing quadrature nodes around a posterior; a and b are kept >= 1/2,
    since Gauss-Jacobi rules degrade as either approaches 0 (a posterior
    pinned near 0 or 1 then gets nodes from a slightly wider Beta)."""
    mean, var = np.asarray(mean, dtype=float), np.asarray(var, dtype=float)
    tot = np.maximum(mean * (1 - mean) / (factor * np.maximum(var, 1e-14)) - 1.0, 1.0)
    return np.maximum(mean * tot, 0.5), np.maximum((1 - mean) * tot, 0.5)


def beta_moments(a, b):
    """Mean and variance of Beta(a, b) (scalars or arrays)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mean = a / (a + b)
    return mean, mean * (1.0 - mean) / (a + b + 1.0)
