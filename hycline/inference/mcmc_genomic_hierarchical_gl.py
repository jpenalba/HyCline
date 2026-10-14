"""Hierarchical Bayesian genomic cline model with genotype-likelihood-
aware marginalization: partial pooling of u/v across many loci (Phase 4
Component B), combined with proper marginalization over genotype
uncertainty from low-coverage sequencing data (Phase 5), rather than
requiring hard genotype calls first.

This is Phase 4, Component E -- the genomic-cline analog of Component D
(`hycline.inference.mcmc_hierarchical_gl`, geographic clines). See
BACKEND_PHASE4.md section 12 for the full design rationale and
validation, including whether the same finding from Component D (a
hard-called sparse locus contaminates the *shared* population-level
hyperparameters, not just its own estimate) reproduces for genomic
clines.

Reuses `_marginal_log_likelihood_jax` from `mcmc_hierarchical_gl` (fully
generic -- takes genotype likelihoods and a predicted probability, with
no geographic-specific assumptions) and `_genomic_cline_core_jax` from
`mcmc_genomic_hierarchical` (the same JAX-native genomic cline core
Component B already validated), rather than duplicating either. Reuses
`HierarchicalGenomicFitResult` from `mcmc_genomic_hierarchical` directly,
since the output shape (population-level hyperparameters plus per-locus
u/v) is identical to Component B's own.
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
from numpyro.infer import MCMC, NUTS

from hycline.inference.mcmc_genomic_hierarchical import (
    HierarchicalGenomicFitResult, _genomic_cline_core_jax, _parental_allele_freq_jax,
    _parental_freq_arrays,
)
from hycline.inference.mcmc_hierarchical_gl import _marginal_log_likelihood_jax

_EPS = 1e-6


def _hierarchical_genomic_gl_batch_model(hybrid_index, log_gl0, log_gl1, log_gl2, n_loci,
                                          p0, p1, mu_u_scale, sigma_u_scale,
                                          mu_log_v_loc, mu_log_v_scale, sigma_log_v_scale):
    # population-level (hyper) parameters, centered at the neutral null
    # (mu_u=0, mu_log_v=0 i.e. mu_v=1) -- see BACKEND_PHASE4.md section 8
    # for why that's the biologically meaningful reference point, same
    # reasoning as Component B's own non-GL model.
    mu_u = numpyro.sample("mu_u", dist.Normal(0.0, mu_u_scale))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(sigma_u_scale))
    mu_log_v = numpyro.sample("mu_log_v", dist.Normal(mu_log_v_loc, mu_log_v_scale))
    sigma_log_v = numpyro.sample("sigma_log_v", dist.HalfNormal(sigma_log_v_scale))

    with numpyro.plate("locus", n_loci):
        u = numpyro.sample("u", dist.Normal(mu_u, sigma_u))
        log_v = numpyro.sample("log_v", dist.Normal(mu_log_v, sigma_log_v))

    v = jnp.exp(log_v)
    phi = _genomic_cline_core_jax(hybrid_index[None, :], u[:, None], v[:, None])
    p = jnp.clip(_parental_allele_freq_jax(phi, p0, p1), _EPS, 1.0 - _EPS)

    log_lik = _marginal_log_likelihood_jax(log_gl0, log_gl1, log_gl2, p)
    numpyro.factor("gl_marginal_likelihood", jnp.sum(log_lik))


def fit_hierarchical_genomic_batch_gl(hybrid_index, log_gl0, log_gl1, log_gl2,
                                       locus_names: Optional[list] = None,
                                       mu_u_scale: float = 2.0, sigma_u_scale: float = 2.0,
                                       mu_log_v_loc: float = 0.0, mu_log_v_scale: float = 1.0,
                                       sigma_log_v_scale: float = 1.0,
                                       parental_freq_pop0: Optional[np.ndarray] = None,
                                       parental_freq_pop1: Optional[np.ndarray] = None,
                                       num_warmup: int = 1500, num_samples: int = 1500,
                                       num_chains: int = 4, seed: int = 0
                                       ) -> HierarchicalGenomicFitResult:
    """Fit a hierarchical genomic cline model directly from per-individual
    genotype likelihoods, combining Component B's partial pooling across
    loci with Phase 5's marginalization over genotype uncertainty --
    rather than requiring either independent per-locus fitting (Phase 5's
    `fit_mle_genomic_gl` on its own) or hard genotype calls first
    (Component B's `fit_hierarchical_genomic_batch` on its own).

    Returns the same `HierarchicalGenomicFitResult` type as Component B's
    `fit_hierarchical_genomic_batch`, with the same `.population_summary()`/
    `.locus_summary()` API -- the two functions differ only in what
    likelihood they use underneath (plain binomial on hard counts vs.
    genotype-likelihood marginalization), not in what they return.

    Unlike the geographic analog (Component D), there is no `pmin`/`pmax`
    to derive -- the genomic cline has no free asymptotes (phi(0)=0,
    phi(1)=1 always), so this function's signature is simpler than
    `fit_hierarchical_batch_gl`'s, mirroring the same simplicity
    Component B already has relative to Component A.

    LIMITATION INHERITED FROM Component B: every locus must have data
    (however uncertain) for every individual in `hybrid_index` -- no
    per-locus missing-individual masking. As with Component B's own
    non-GL model, this maps reasonably well onto real genomic data: a
    "low-quality" locus is typically one with reduced per-individual
    confidence everywhere (low coverage), not one missing from most
    individuals outright.

    Parameters
    ----------
    hybrid_index : (n_individuals,) array of genome-wide hybrid index
        (0-1), shared across loci
    log_gl0, log_gl1, log_gl2 : (n_loci, n_individuals) arrays of natural-
        log genotype likelihoods, P(reads | genotype=0/1/2 focal-allele
        copies), per locus per individual
    locus_names : optional labels; defaults to locus_0, locus_1, ...
    mu_u_scale, sigma_u_scale, mu_log_v_loc, mu_log_v_scale,
        sigma_log_v_scale : hyperprior settings, same meaning and same
        defaults as `fit_hierarchical_genomic_batch`
    parental_freq_pop0, parental_freq_pop1 : optional (n_loci,) parental
        frequencies of the counted allele, as in
        `fit_hierarchical_genomic_batch`. Default 0 and 1 for every locus.
    num_warmup, num_samples, num_chains, seed : standard NUTS/MCMC controls

    Returns
    -------
    HierarchicalGenomicFitResult
    """
    n_loci = log_gl0.shape[0]
    p0, p1 = _parental_freq_arrays(n_loci, parental_freq_pop0, parental_freq_pop1)
    if locus_names is None:
        locus_names = [f"locus_{i}" for i in range(n_loci)]

    kernel = NUTS(_hierarchical_genomic_gl_batch_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key,
              hybrid_index=jnp.asarray(hybrid_index), log_gl0=jnp.asarray(log_gl0),
              log_gl1=jnp.asarray(log_gl1), log_gl2=jnp.asarray(log_gl2), n_loci=n_loci,
              p0=p0, p1=p1, mu_u_scale=mu_u_scale, sigma_u_scale=sigma_u_scale,
              mu_log_v_loc=mu_log_v_loc, mu_log_v_scale=mu_log_v_scale,
              sigma_log_v_scale=sigma_log_v_scale)

    idata = az.from_numpyro(mcmc, coords={"locus": locus_names},
                             dims={"u": ["locus"], "log_v": ["locus"]})
    return HierarchicalGenomicFitResult(idata=idata, locus_names=locus_names)
