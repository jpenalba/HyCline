"""Hierarchical Bayesian geographic cline model with genotype-likelihood-
aware marginalization: partial pooling of cline center/width across many
loci (Phase 4 Component A), combined with proper marginalization over
genotype uncertainty from low-coverage sequencing data (Phase 5), rather
than requiring hard genotype calls first.

This is Phase 4, Component D. See BACKEND_PHASE4.md section 11 for the
full design rationale, why this needed real new work rather than just
wiring two existing pieces together, and the validation answering the
actual open empirical question this component exists to answer: does
genotype-likelihood-awareness still add value on top of hierarchical
pooling's own stabilizing effect, or does pooling alone already fix most
of what Phase 5 fixed for independent per-locus fits?

Builds on `hycline.inference.mcmc_hierarchical` (reuses its
`HierarchicalFitResult` class directly, since the output shape --
population-level hyperparameters plus per-locus center/width -- is
identical) and `hycline.likelihoods.genotype_likelihood` (reimplements its
marginal-likelihood math in JAX, the same pattern `mcmc_numpyro`/
`mcmc_genomic_hierarchical` use for their own NumPy-likelihood siblings).
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import jax

jax.config.update("jax_enable_x64", True)  # see BACKEND_PHASE2.md section 11 for why

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp as jax_logsumexp
from numpyro.infer import MCMC, NUTS

from hycline.inference.mcmc_hierarchical import HierarchicalFitResult

_EPS = 1e-6


def _hwe_log_genotype_probs_jax(p):
    p = jnp.clip(p, _EPS, 1.0 - _EPS)
    return jnp.log((1 - p) ** 2), jnp.log(2 * p * (1 - p)), jnp.log(p ** 2)


def _marginal_log_likelihood_jax(log_gl0, log_gl1, log_gl2, p):
    """JAX-native reimplementation of
    `hycline.likelihoods.genotype_likelihood._marginal_log_likelihood`,
    so it's traceable inside a NumPyro model -- the same reason
    `mcmc_numpyro`/`mcmc_genomic_hierarchical` duplicate their respective
    NumPy cline cores. Verified to match the NumPy/scipy version exactly
    in `tests/test_mcmc_hierarchical_gl.py::
    test_jax_marginal_likelihood_matches_numpy_version`."""
    log_p0, log_p1, log_p2 = _hwe_log_genotype_probs_jax(p)
    terms = jnp.stack([log_gl0 + log_p0, log_gl1 + log_p1, log_gl2 + log_p2], axis=0)
    return jax_logsumexp(terms, axis=0)


def _cline_core_jax(x, center, width, pmin, pmax):
    """Same sigmoid core as `hycline.inference.mcmc_numpyro._cline_core_jax`
    -- reimplemented locally (a small, contained duplication) rather than
    imported, to keep this module's dependency on the non-GL hierarchical
    machinery limited to the result class it reuses."""
    z = -4.0 * (x - center) / width
    z = jnp.clip(z, -700, 700)
    return pmin + (pmax - pmin) / (1.0 + jnp.exp(z))


def _derive_pmin_pmax_gl(distance: np.ndarray, log_gl0: np.ndarray, log_gl1: np.ndarray,
                          log_gl2: np.ndarray, n_edge_individuals: int = 3) -> tuple:
    """Derive per-locus pmin/pmax from genotype likelihoods near the
    transect's geographic extremes, generalizing
    `hycline.inference.mcmc_hierarchical._derive_pmin_pmax` (which
    averages hard allele counts over several edge sites, rather than
    anchoring on one, to avoid the single-site fragility documented in
    BACKEND_PHASE4.md section 3) to genotype-likelihood input.

    Since there's no hard count to average here, each individual's
    contribution is its *expected* focal-allele dosage under a flat
    (uniform) prior over genotype -- i.e. treating the genotype
    likelihood as if it were a genotype posterior under a flat prior,
    which is a reasonable, standard approximation for this purpose (a
    rough point estimate to fix pmin/pmax at, not the model's actual
    inference target). Concretely: normalize (log_gl0, log_gl1, log_gl2)
    to sum to 1 via softmax, then take E[genotype]/2 = (0*p0 + 1*p1 +
    2*p2)/2 as that individual's expected allele frequency, and average
    this over the `n_edge_individuals` nearest each end of the transect.
    """
    order = np.argsort(distance)
    log_gl0_sorted = log_gl0[:, order]
    log_gl1_sorted = log_gl1[:, order]
    log_gl2_sorted = log_gl2[:, order]

    def expected_freq(g0, g1, g2):
        m = np.maximum(np.maximum(g0, g1), g2)
        e0, e1, e2 = np.exp(g0 - m), np.exp(g1 - m), np.exp(g2 - m)
        total = e0 + e1 + e2
        return (0 * e0 + 1 * e1 + 2 * e2) / total / 2.0

    k = min(n_edge_individuals, log_gl0_sorted.shape[1])
    freq_left = expected_freq(log_gl0_sorted[:, :k], log_gl1_sorted[:, :k],
                               log_gl2_sorted[:, :k])
    freq_right = expected_freq(log_gl0_sorted[:, -k:], log_gl1_sorted[:, -k:],
                                log_gl2_sorted[:, -k:])
    return freq_left.mean(axis=1), freq_right.mean(axis=1)


def _hierarchical_gl_batch_model(distance, log_gl0, log_gl1, log_gl2, n_loci, pmin, pmax,
                                  mu_center_bounds, sigma_center_scale,
                                  mu_log_width_loc, mu_log_width_scale,
                                  sigma_log_width_scale):
    mu_center = numpyro.sample("mu_center", dist.Uniform(*mu_center_bounds))
    sigma_center = numpyro.sample("sigma_center", dist.HalfNormal(sigma_center_scale))
    mu_log_width = numpyro.sample("mu_log_width",
                                   dist.Normal(mu_log_width_loc, mu_log_width_scale))
    sigma_log_width = numpyro.sample("sigma_log_width", dist.HalfNormal(sigma_log_width_scale))

    with numpyro.plate("locus", n_loci):
        center = numpyro.sample("center", dist.Normal(mu_center, sigma_center))
        log_width = numpyro.sample("log_width", dist.Normal(mu_log_width, sigma_log_width))

    width = jnp.exp(log_width)
    p = _cline_core_jax(distance[None, :], center[:, None], width[:, None],
                         pmin[:, None], pmax[:, None])
    p = jnp.clip(p, _EPS, 1.0 - _EPS)

    log_lik = _marginal_log_likelihood_jax(log_gl0, log_gl1, log_gl2, p)
    numpyro.factor("gl_marginal_likelihood", jnp.sum(log_lik))


def fit_hierarchical_batch_gl(distance: np.ndarray, log_gl0: np.ndarray, log_gl1: np.ndarray,
                               log_gl2: np.ndarray, locus_names: Optional[list] = None,
                               pmin: Optional[np.ndarray] = None,
                               pmax: Optional[np.ndarray] = None,
                               n_edge_individuals: int = 3,
                               sigma_center_scale: Optional[float] = None,
                               mu_log_width_loc: Optional[float] = None,
                               mu_log_width_scale: float = 1.5,
                               sigma_log_width_scale: float = 1.0,
                               num_warmup: int = 1500, num_samples: int = 1500,
                               num_chains: int = 4, seed: int = 0) -> HierarchicalFitResult:
    """Fit a hierarchical geographic cline model directly from per-
    individual genotype likelihoods, combining Component A's partial
    pooling across loci with Phase 5's marginalization over genotype
    uncertainty -- rather than requiring either independent per-locus
    fitting (Phase 5 on its own) or hard genotype calls first (Component
    A on its own).

    Returns the same `HierarchicalFitResult` type as
    `hycline.inference.mcmc_hierarchical.fit_hierarchical_batch`, with the
    same `.population_summary()`/`.locus_summary()` API -- the two
    functions differ only in what likelihood they use underneath (plain
    binomial on hard counts vs. genotype-likelihood marginalization), not
    in what they return.

    LIMITATIONS INHERITED FROM Component A: every locus must have data
    (however uncertain) for every individual in `distance` -- no
    per-locus missing-individual masking (BACKEND_PHASE4.md section 2).
    `pmin`/`pmax` are derived per-locus, not pooled across loci, using the
    same multi-individual-averaging approach that fixed Component A's own
    single-site fragility bug (BACKEND_PHASE4.md section 3), adapted for
    genotype-likelihood input (see `_derive_pmin_pmax_gl`).

    Parameters
    ----------
    distance : (n_individuals,) array of geographic distances, shared
        across loci (individuals at the same site share a distance value)
    log_gl0, log_gl1, log_gl2 : (n_loci, n_individuals) arrays of natural-
        log genotype likelihoods, P(reads | genotype=0/1/2 focal-allele
        copies), per locus per individual
    locus_names : optional labels; defaults to locus_0, locus_1, ...
    pmin, pmax : optional (n_loci,) arrays of fixed asymptotes per locus;
        derived automatically from the data if not given (see
        `_derive_pmin_pmax_gl`)
    n_edge_individuals : how many individuals near each end of the
        transect to average when deriving pmin/pmax (default 3)
    sigma_center_scale, mu_log_width_loc, mu_log_width_scale,
        sigma_log_width_scale : hyperprior settings, same meaning and
        same defaults as `fit_hierarchical_batch`
    num_warmup, num_samples, num_chains, seed : standard NUTS/MCMC controls

    Returns
    -------
    HierarchicalFitResult
    """
    n_loci = log_gl0.shape[0]
    if locus_names is None:
        locus_names = [f"locus_{i}" for i in range(n_loci)]

    d_span = float(distance.max() - distance.min())
    mu_center_bounds = (float(distance.min()) - d_span, float(distance.max()) + d_span)
    if sigma_center_scale is None:
        sigma_center_scale = d_span / 2.0
    if mu_log_width_loc is None:
        mu_log_width_loc = float(np.log(max(d_span / 4.0, 1e-3)))

    if pmin is None or pmax is None:
        derived_pmin, derived_pmax = _derive_pmin_pmax_gl(
            distance, log_gl0, log_gl1, log_gl2, n_edge_individuals=n_edge_individuals)
        pmin = derived_pmin if pmin is None else pmin
        pmax = derived_pmax if pmax is None else pmax

    kernel = NUTS(_hierarchical_gl_batch_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key,
              distance=jnp.asarray(distance), log_gl0=jnp.asarray(log_gl0),
              log_gl1=jnp.asarray(log_gl1), log_gl2=jnp.asarray(log_gl2), n_loci=n_loci,
              pmin=jnp.asarray(pmin), pmax=jnp.asarray(pmax),
              mu_center_bounds=mu_center_bounds, sigma_center_scale=sigma_center_scale,
              mu_log_width_loc=mu_log_width_loc, mu_log_width_scale=mu_log_width_scale,
              sigma_log_width_scale=sigma_log_width_scale)

    idata = az.from_numpyro(mcmc, coords={"locus": locus_names},
                             dims={"center": ["locus"], "log_width": ["locus"]})
    return HierarchicalFitResult(idata=idata, locus_names=locus_names)
