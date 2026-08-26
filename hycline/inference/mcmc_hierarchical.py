"""Hierarchical Bayesian geographic cline model: partial pooling of cline
center and width across many loci sharing the same transect.

This is Phase 4, Component A. See BACKEND_PHASE4.md for the full design
rationale (why hierarchical pooling, the exact model, prior choices, and
the shrinkage-based validation strategy since there is no HZAR-equivalent
gold standard for this). This module docstring covers only the API surface.

Builds directly on `hycline.inference.mcmc_numpyro`: reuses its JAX-native
cline core (`_cline_core_jax`) rather than duplicating it, and follows the
same NUTS/plate-based approach as `fit_mcmc_batch` -- the difference is
that `fit_mcmc_batch` gives every locus fully independent priors (no
pooling at all), while this module's `fit_hierarchical_batch` puts a
shared population-level prior over center and (log) width, so loci with
sparse data borrow statistical strength from the rest of the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import arviz as az
import jax

jax.config.update("jax_enable_x64", True)  # see BACKEND_PHASE2.md section 11 for why

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from hycline.inference.mcmc_numpyro import _cline_core_jax

_EPS = 1e-6


def _derive_pmin_pmax(distance: np.ndarray, n_alleles: np.ndarray,
                       n_focal: np.ndarray, n_edge_sites: int = 3) -> tuple:
    """Derive per-locus pmin/pmax from the observed frequencies near the
    transect's geographic extremes.

    Averages the `n_edge_sites` sites closest to each end of the transect
    (pooling their counts, i.e. sum(n_focal)/sum(n_alleles) across those
    sites) rather than anchoring on a single extreme site's frequency.
    This matters more than it might look: anchoring on one site is
    fragile to exactly the kind of sampling noise a binomial likelihood
    is expected to produce -- with realistic sample sizes, a true
    pmin around 0.05 can easily draw 0/n at a single site (a few-percent
    event, not a rare one), which then locks the model's asymptote at
    0.0 instead of 0.05 and can force the optimizer to substantially
    widen the fitted cline to reconcile the rest of the data with that
    wrong anchor. This was found and fixed during this module's own
    synthetic-data validation, not hypothetically -- see BACKEND_PHASE4.md
    section 3 for the full story, including a worked example showing the
    single-site version giving a ~2x too-wide fit while this version
    recovers the true width. The single-site convention is still what
    Phase 1/2's `fit_mle_fixed_extremes`/CLI use (inherited from matching
    HZAR's own convention, validated fine against the real, larger-sample
    Manakin dataset) -- not changed there in this pass; see the same
    BACKEND_PHASE4.md section for why this fix is scoped to the
    hierarchical model for now rather than applied everywhere.
    """
    order = np.argsort(distance)
    n_alleles_sorted = n_alleles[:, order]
    n_focal_sorted = n_focal[:, order]
    k = min(n_edge_sites, n_alleles_sorted.shape[1])

    pmin = n_focal_sorted[:, :k].sum(axis=1) / n_alleles_sorted[:, :k].sum(axis=1)
    pmax = n_focal_sorted[:, -k:].sum(axis=1) / n_alleles_sorted[:, -k:].sum(axis=1)
    return pmin, pmax


def _hierarchical_batch_model(distance, n_alleles, n_focal, n_loci, pmin, pmax,
                               mu_center_bounds, sigma_center_scale,
                               mu_log_width_loc, mu_log_width_scale,
                               sigma_log_width_scale):
    # population-level (hyper) parameters, shared across all loci
    mu_center = numpyro.sample("mu_center", dist.Uniform(*mu_center_bounds))
    sigma_center = numpyro.sample("sigma_center", dist.HalfNormal(sigma_center_scale))
    mu_log_width = numpyro.sample("mu_log_width",
                                   dist.Normal(mu_log_width_loc, mu_log_width_scale))
    sigma_log_width = numpyro.sample("sigma_log_width", dist.HalfNormal(sigma_log_width_scale))

    # per-locus parameters, drawn from the population-level distributions
    # (this is the pooling step -- compare to fit_mcmc_batch's _batch_model,
    # where center/width are drawn from flat, locus-independent priors)
    with numpyro.plate("locus", n_loci):
        center = numpyro.sample("center", dist.Normal(mu_center, sigma_center))
        log_width = numpyro.sample("log_width", dist.Normal(mu_log_width, sigma_log_width))

    width = jnp.exp(log_width)
    p = _cline_core_jax(distance[None, :], center[:, None], width[:, None],
                         pmin[:, None], pmax[:, None])
    p = jnp.clip(p, _EPS, 1.0 - _EPS)
    numpyro.sample("obs", dist.Binomial(total_count=n_alleles, probs=p), obs=n_focal)


@dataclass
class HierarchicalFitResult:
    idata: az.InferenceData
    locus_names: list

    def population_summary(self) -> pd.DataFrame:
        """Posterior summary for the 4 population-level (hyper) parameters:
        mu_center, sigma_center (across-locus SD of cline center),
        mu_log_width, sigma_log_width (across-locus SD of log cline width)."""
        return az.summary(self.idata, var_names=["mu_center", "sigma_center",
                                                   "mu_log_width", "sigma_log_width"])

    def locus_summary(self) -> pd.DataFrame:
        """Posterior summary per locus, one row each, with width converted
        back to the natural (non-log) scale for readability."""
        post = self.idata.posterior
        rows = []
        for i, name in enumerate(self.locus_names):
            center_i = post["center"].isel(locus=i)
            width_i = np.exp(post["log_width"].isel(locus=i).values)
            rows.append({
                "locus": name,
                "center_mean": float(center_i.mean()),
                "center_sd": float(center_i.std()),
                "width_mean": float(np.mean(width_i)),
                "width_sd": float(np.std(width_i)),
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = ["Hierarchical geographic cline fit -- population-level parameters:",
                 self.population_summary().to_string(),
                 "\nPer-locus posterior means:",
                 self.locus_summary().to_string(index=False)]
        return "\n".join(lines)


def fit_hierarchical_batch(distance: np.ndarray, n_alleles: np.ndarray,
                            n_focal: np.ndarray,
                            locus_names: Optional[list] = None,
                            pmin: Optional[np.ndarray] = None,
                            pmax: Optional[np.ndarray] = None,
                            n_edge_sites: int = 3,
                            sigma_center_scale: Optional[float] = None,
                            mu_log_width_loc: Optional[float] = None,
                            mu_log_width_scale: float = 1.5,
                            sigma_log_width_scale: float = 1.0,
                            num_warmup: int = 1500, num_samples: int = 1500,
                            num_chains: int = 4, seed: int = 0) -> HierarchicalFitResult:
    """Fit a hierarchical geographic cline model: many loci sharing one
    transect, with cline center and (log) width partially pooled across
    loci via shared population-level priors.

    LIMITATION INHERITED FROM `fit_mcmc_batch` (Phase 2): every locus must
    have data at every site in `distance` -- there is no per-locus missing-
    site masking. A "sparse" locus should have small (but nonzero) sample
    sizes at every site, not zero sample size at most sites and larger
    sample sizes at a few -- the latter isn't representable by this
    function's (n_loci, n_sites) dense-matrix input. See BACKEND_PHASE4.md
    section 2 for why this wasn't lifted in this pass.

    Parameters
    ----------
    distance : (n_sites,) array of geographic distances, shared across loci
    n_alleles, n_focal : (n_loci, n_sites) arrays of sampled/focal allele
        counts per locus per site
    locus_names : optional labels; defaults to locus_0, locus_1, ...
    pmin, pmax : optional (n_loci,) arrays of fixed asymptotes per locus.
        If not given, derived from each locus's own observed extremes
        (see `_derive_pmin_pmax`, averaged over `n_edge_sites` sites near
        each end rather than a single site) -- matching Phase 1/2's "fixed
        scaling" convention. pmin/pmax are NOT pooled across loci in this
        model; see BACKEND_PHASE4.md section 2 for why.
    n_edge_sites : how many sites near each end of the transect to pool
        when deriving pmin/pmax (default 3). See `_derive_pmin_pmax`'s
        docstring for why this matters more than it might look.
    sigma_center_scale : scale of the HalfNormal hyperprior on the
        across-locus SD of cline center. Defaults to half the transect's
        geographic span if not given -- a generous, weakly-informative
        choice; tighten it if you have prior knowledge that loci should be
        tightly concordant, or the fit's own posterior for sigma_center
        will simply reflect how much loci actually disagree in the data.
    mu_log_width_loc, mu_log_width_scale : mean and SD of the Normal prior
        on the population-level mean log-width. Defaults to log(span/4)
        (matching Phase 1's own initial-guess heuristic for a "typical"
        cline width) with a generous SD of 1.5 natural-log units.
    sigma_log_width_scale : scale of the HalfNormal hyperprior on the
        across-locus SD of log-width.
    num_warmup, num_samples, num_chains, seed : standard NUTS/MCMC controls

    Returns
    -------
    HierarchicalFitResult
    """
    n_loci = n_alleles.shape[0]
    if locus_names is None:
        locus_names = [f"locus_{i}" for i in range(n_loci)]

    d_span = float(distance.max() - distance.min())
    mu_center_bounds = (float(distance.min()) - d_span, float(distance.max()) + d_span)
    if sigma_center_scale is None:
        sigma_center_scale = d_span / 2.0
    if mu_log_width_loc is None:
        mu_log_width_loc = float(np.log(max(d_span / 4.0, 1e-3)))

    if pmin is None or pmax is None:
        derived_pmin, derived_pmax = _derive_pmin_pmax(distance, n_alleles, n_focal,
                                                         n_edge_sites=n_edge_sites)
        pmin = derived_pmin if pmin is None else pmin
        pmax = derived_pmax if pmax is None else pmax

    kernel = NUTS(_hierarchical_batch_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key,
              distance=jnp.asarray(distance), n_alleles=jnp.asarray(n_alleles),
              n_focal=jnp.asarray(n_focal), n_loci=n_loci,
              pmin=jnp.asarray(pmin), pmax=jnp.asarray(pmax),
              mu_center_bounds=mu_center_bounds, sigma_center_scale=sigma_center_scale,
              mu_log_width_loc=mu_log_width_loc, mu_log_width_scale=mu_log_width_scale,
              sigma_log_width_scale=sigma_log_width_scale)

    idata = az.from_numpyro(mcmc, coords={"locus": locus_names},
                             dims={"center": ["locus"], "log_width": ["locus"]})
    return HierarchicalFitResult(idata=idata, locus_names=locus_names)
