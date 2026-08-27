"""Hierarchical Bayesian genomic cline model: partial pooling of the
logit-logistic cline parameters (u, v) across many loci sharing the same
set of individuals/hybrid indices.

This is Phase 4, Component B -- the genomic-cline analog of Component A
(`hycline.inference.mcmc_hierarchical`, geographic clines). See
BACKEND_PHASE4.md for the full design rationale (why hierarchical
pooling, the exact model, prior choices, and the shrinkage-based
validation strategy since there is no HZAR-equivalent gold standard for
this -- HZAR doesn't do genomic clines at all, let alone hierarchical
ones). This module docstring covers only the API surface.

Also, per BACKEND_PHASE3.md section 4: this is the first Bayesian
backend for genomic clines at all (Phase 3 shipped classic MLE only) --
built directly as the hierarchical version rather than a non-hierarchical
Bayesian batch model first, since the hierarchical version was the
requested and more broadly useful next increment.
"""

from __future__ import annotations

import warnings
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

_EPS = 1e-6


def _genomic_cline_core_jax(h, u, v):
    """JAX-native reimplementation of the logit-logistic genomic cline
    core. Mirrors `hycline.models.genomic_cline.genomic_cline_function`
    exactly, but written with `jax.numpy` so it's traceable inside a
    NumPyro model -- the same reason `hycline.inference.mcmc_numpyro`
    duplicates the geographic cline core rather than importing the NumPy
    version (see BACKEND_PHASE2.md section 9, now section 11 in that
    doc's numbering, for that precedent and the regression-test pattern
    used to keep the two implementations in agreement --
    `tests/test_mcmc_genomic_hierarchical.py::
    test_jax_genomic_core_matches_numpy_core` here does the same for
    this module).

    Unlike the geographic cline core, there is no pmin/pmax to worry
    about -- the genomic cline is anchored so phi(0)=0 and phi(1)=1
    always, by construction of what hybrid index means (BACKEND_PHASE3.md
    section 2), so there's also no equivalent of Component A's
    single-site pmin/pmax fragility (BACKEND_PHASE4.md section 3) to
    guard against here.
    """
    h_clipped = jnp.clip(h, _EPS, 1.0 - _EPS)
    logit_h = jnp.log(h_clipped / (1.0 - h_clipped))
    z = u + v * logit_h
    z = jnp.clip(z, -700, 700)
    return 1.0 / (1.0 + jnp.exp(-z))


def _hierarchical_genomic_batch_model(hybrid_index, n_alleles, n_focal, n_loci,
                                       mu_u_scale, sigma_u_scale,
                                       mu_log_v_loc, mu_log_v_scale, sigma_log_v_scale,
                                       pool_center: bool = False,
                                       mu_center_scale: float = 2.0,
                                       sigma_center_scale: float = 2.0):
    # log_v's hierarchical prior is shared by both parameterizations below
    # -- pooling choice only affects how the *other* cline parameter is
    # modeled (raw u vs. logit(center) = u/v).
    mu_log_v = numpyro.sample("mu_log_v", dist.Normal(mu_log_v_loc, mu_log_v_scale))
    sigma_log_v = numpyro.sample("sigma_log_v", dist.HalfNormal(sigma_log_v_scale))

    if pool_center:
        # Firneno et al. (2023)'s own parameterization (BACKEND_PHASE4.md
        # section 9.3 for the full story of why this was added): pool
        # logit(center) = u/v directly, not u. Centered at 0 (center=0.5
        # on the hybrid-index scale, the neutral null) for the same
        # reason mu_u=0 is below -- see BACKEND_PHASE4.md section 8.
        mu_center = numpyro.sample("mu_center", dist.Normal(0.0, mu_center_scale))
        sigma_center = numpyro.sample("sigma_center", dist.HalfNormal(sigma_center_scale))
        with numpyro.plate("locus", n_loci):
            logit_center = numpyro.sample("logit_center", dist.Normal(mu_center, sigma_center))
            log_v = numpyro.sample("log_v", dist.Normal(mu_log_v, sigma_log_v))
        v = jnp.exp(log_v)
        u = v * logit_center
    else:
        # population-level (hyper) parameters, shared across all loci.
        # Centered at the neutral null (mu_u=0, mu_log_v=0 i.e. mu_v=1) --
        # see BACKEND_PHASE4.md section 8 for why that's the biologically
        # meaningful reference point here, not an arbitrary choice.
        mu_u = numpyro.sample("mu_u", dist.Normal(0.0, mu_u_scale))
        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(sigma_u_scale))
        with numpyro.plate("locus", n_loci):
            u = numpyro.sample("u", dist.Normal(mu_u, sigma_u))
            log_v = numpyro.sample("log_v", dist.Normal(mu_log_v, sigma_log_v))
        v = jnp.exp(log_v)

    phi = _genomic_cline_core_jax(hybrid_index[None, :], u[:, None], v[:, None])
    phi = jnp.clip(phi, _EPS, 1.0 - _EPS)
    numpyro.sample("obs", dist.Binomial(total_count=n_alleles, probs=phi), obs=n_focal)


@dataclass
class HierarchicalGenomicFitResult:
    idata: az.InferenceData
    locus_names: list

    @property
    def pool_center(self) -> bool:
        """Whether this fit used the center-pooled parameterization
        (`logit_center` in the posterior) rather than the default
        u-pooled one (`u` in the posterior) -- detected from the
        posterior itself, not stored separately, so a result object is
        always self-describing."""
        return "logit_center" in self.idata.posterior

    def population_summary(self) -> pd.DataFrame:
        """Posterior summary for the 4 population-level (hyper) parameters.

        Default (u-pooled) parameterization: mu_u, sigma_u (across-locus
        SD of bias), mu_log_v, sigma_log_v (across-locus SD of log rate).
        mu_u near 0 and mu_log_v near 0 (mu_v near 1) indicate the
        genome-wide average locus is close to the neutral expectation;
        sigma_u/sigma_log_v quantify how much loci vary around that
        average -- the genome-wide "variation in introgression" `bgchm`
        (Gompert, DeRaad & Buerkle 2024) is built to quantify.

        Center-pooled parameterization (`pool_center=True` at fit time):
        mu_center, sigma_center (across-locus SD of logit(center) = u/v)
        in place of mu_u/sigma_u -- Firneno et al. (2023)'s own quantity
        (BACKEND_PHASE4.md section 9.3), directly usable as sigma_c in
        `hycline.analysis.coupling.estimate_coupling_coefficient` without
        the SD-of-point-estimates approximation `genomic_cline_variance`
        otherwise has to fall back on.
        """
        if self.pool_center:
            return az.summary(self.idata, var_names=["mu_center", "sigma_center",
                                                       "mu_log_v", "sigma_log_v"])
        return az.summary(self.idata, var_names=["mu_u", "sigma_u",
                                                   "mu_log_v", "sigma_log_v"])

    def locus_summary(self) -> pd.DataFrame:
        """Posterior summary per locus, one row each, with v converted
        back to the natural (non-log) scale for readability. Includes
        `u_mean`/`u_sd` either way -- derived from `logit_center * v` per
        draw when the fit used the center-pooled parameterization, so
        `locus_summary()`'s shape doesn't depend on which parameterization
        was used, only `population_summary()`'s does."""
        post = self.idata.posterior
        rows = []
        for i, name in enumerate(self.locus_names):
            v_i = np.exp(post["log_v"].isel(locus=i).values)
            if self.pool_center:
                logit_center_i = post["logit_center"].isel(locus=i).values
                u_i = logit_center_i * v_i
                center_i = 1.0 / (1.0 + np.exp(-logit_center_i))
                row = {
                    "locus": name,
                    "center_mean": float(np.mean(center_i)), "center_sd": float(np.std(center_i)),
                    "u_mean": float(np.mean(u_i)), "u_sd": float(np.std(u_i)),
                    "v_mean": float(np.mean(v_i)), "v_sd": float(np.std(v_i)),
                }
            else:
                u_i = post["u"].isel(locus=i).values
                row = {
                    "locus": name,
                    "u_mean": float(np.mean(u_i)), "u_sd": float(np.std(u_i)),
                    "v_mean": float(np.mean(v_i)), "v_sd": float(np.std(v_i)),
                }
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = ["Hierarchical genomic cline fit -- population-level parameters:",
                 self.population_summary().to_string(),
                 "\nPer-locus posterior means:",
                 self.locus_summary().to_string(index=False)]
        return "\n".join(lines)


def _warn_if_hybrid_index_distribution_uneven(hybrid_index, locus_names, u_mean, v_mean,
                                                v_steep_threshold: float = 20.0,
                                                gap_threshold: float = 0.05) -> None:
    """Bailey (2024, the gghybrid paper, *Mol Ecol Resour* 24:e13910) found,
    via direct simulation (their section 2.3.4, Figure 7, Figure S3), that
    an uneven distribution of hybrid index values specifically degrades
    genomic-cline estimation accuracy for STEEP clines -- fitted cline
    centers become highly clustered and v estimates less accurate, with
    higher posterior uncertainty, specifically for loci whose true steep
    transition falls in a gap of hybrid-index sampling. Their own null/
    shallow-cline simulation (v=1) showed little impact from the same
    uneven sampling (their Figure 7a,b) -- this is not a generic "check
    your sampling" warning, it targets exactly the condition their
    simulations showed matters, not sparse sampling in general.

    `v_steep_threshold=20.0` matches Bailey's own reported criterion
    directly (their Figure S3 identifies the clustering effect among loci
    passing p<=.05 and v>20), not an arbitrary round number.

    `gap_threshold=0.05` (5% of the full [0,1] hybrid-index range) is
    this project's own choice, stated explicitly rather than left
    implicit: Bailey's paper illustrates the effect visually (Figure 7/
    S3) rather than stating a specific gap-size criterion, so there is no
    "correct" published number to match here. See BACKEND_PHASE4.md
    section 16 for the full account, including why this check runs
    post-fit (needs each locus's own fitted v/center) rather than as a
    pre-fit data check.
    """
    hybrid_index = np.asarray(hybrid_index, dtype=float)
    u_mean = np.asarray(u_mean, dtype=float)
    v_mean = np.asarray(v_mean, dtype=float)
    logit_center = u_mean / v_mean
    center = 1.0 / (1.0 + np.exp(-logit_center))
    sorted_hi = np.sort(hybrid_index)

    flagged = []
    for name, v_i, c_i in zip(locus_names, v_mean, center):
        if v_i <= v_steep_threshold:
            continue
        nearest_dist = float(np.min(np.abs(sorted_hi - c_i)))
        if nearest_dist > gap_threshold:
            flagged.append((name, v_i, c_i, nearest_dist))

    if flagged:
        detail = "; ".join(
            f"{name} (v={v_i:.1f}, center={c_i:.3f}, nearest sampled hybrid "
            f"index {nearest_dist:.3f} away)"
            for name, v_i, c_i, nearest_dist in flagged)
        warnings.warn(
            f"{len(flagged)} locus/loci have a steep fitted cline (v > "
            f"{v_steep_threshold:g}) centered in a sparsely-sampled region of "
            f"hybrid-index space (nearest sampled individual more than "
            f"{gap_threshold:g} away): {detail}. Bailey (2024, gghybrid paper) "
            "found this specific combination -- a steep cline whose transition "
            "falls in a gap of hybrid-index sampling -- degrades cline-parameter "
            "accuracy (v estimates less accurate, cline centers artificially "
            "clustered, higher posterior uncertainty), while null/shallow clines "
            "were largely unaffected by the same uneven sampling. Treat these "
            "loci's fitted parameters with extra caution, and consider whether "
            "denser hybrid-index sampling near their fitted center is feasible "
            "before drawing conclusions from them. See BACKEND_PHASE4.md "
            "section 16 for the full investigation.",
            UserWarning, stacklevel=3,
        )


def fit_hierarchical_genomic_batch(hybrid_index: np.ndarray, n_alleles: np.ndarray,
                                    n_focal: np.ndarray,
                                    locus_names: Optional[list] = None,
                                    mu_u_scale: float = 2.0, sigma_u_scale: float = 2.0,
                                    mu_log_v_loc: float = 0.0, mu_log_v_scale: float = 1.0,
                                    sigma_log_v_scale: float = 1.0,
                                    pool_center: bool = False,
                                    mu_center_scale: float = 2.0, sigma_center_scale: float = 2.0,
                                    num_warmup: int = 1500, num_samples: int = 1500,
                                    num_chains: int = 4, seed: int = 0
                                    ) -> HierarchicalGenomicFitResult:
    """Fit a hierarchical genomic cline model: many loci sharing one set
    of individuals/hybrid indices, with the bias (u) and (log) rate (v)
    parameters partially pooled across loci via shared population-level
    priors.

    LIMITATION INHERITED FROM the geographic `fit_mcmc_batch`/
    `fit_hierarchical_batch` (Phase 2 / Component A): every locus must
    have data for every individual in `hybrid_index` -- there is no
    per-locus missing-individual masking. Unlike the geographic case,
    this maps naturally onto a common real scenario for genomic data: a
    "low-quality" locus is one with reduced allele-count confidence at
    every individual (e.g. `n_alleles=1` from low-coverage sequencing
    instead of the full diploid `n_alleles=2`), not one missing from most
    individuals entirely -- see BACKEND_PHASE4.md section 6 for why this
    inherited constraint is actually a reasonably good fit for genomic
    data specifically, more so than it was for the geographic case.

    Parameters
    ----------
    hybrid_index : (n_individuals,) array of genome-wide hybrid index
        (0-1), shared across loci
    n_alleles, n_focal : (n_loci, n_individuals) arrays of sampled/focal
        allele counts per locus per individual
    locus_names : optional labels; defaults to locus_0, locus_1, ...
    mu_u_scale : SD of the Normal(0, mu_u_scale) prior on the
        population-level mean bias. Centered at 0 (the neutral
        expectation) by construction, not a free location parameter --
        see BACKEND_PHASE4.md section 8. Ignored if `pool_center=True`.
    sigma_u_scale : scale of the HalfNormal hyperprior on the across-locus
        SD of u. Ignored if `pool_center=True`.
    mu_log_v_loc, mu_log_v_scale : mean and SD of the Normal prior on the
        population-level mean log-rate. Defaults to a Normal(0, 1) --
        centered at the neutral log_v=0 (v=1) expectation. Used by both
        parameterizations.
    sigma_log_v_scale : scale of the HalfNormal hyperprior on the
        across-locus SD of log-rate -- this is the parameter most directly
        answering "how much does introgression rate vary across the
        genome," the central question `bgchm` was built around. Used by
        both parameterizations.
    pool_center : if False (default), hierarchically pool the raw bias
        parameter `u` -- the original parameterization this function
        shipped with, unchanged, so existing results (including the
        real-data BGC/gghybrid cross-validation in BACKEND_PHASE3.md
        section 7) are reproduced exactly. If True, hierarchically pool
        `logit(center) = u/v` instead -- Firneno et al. (2023)'s own
        parameterization (BACKEND_PHASE4.md section 9.3), and the one
        `hycline.analysis.coupling.genomic_cline_variance`'s sigma_c is
        actually trying to approximate from point estimates when this
        isn't used. `u` is still derived and reported (as
        `logit_center * v`) either way, so downstream code that expects
        `u`/`v` doesn't need to change based on this flag.
    mu_center_scale : SD of the Normal(0, mu_center_scale) prior on the
        population-level mean logit(center). Centered at 0 (center=0.5 on
        the hybrid-index scale, the neutral expectation), the same
        convention as `mu_u_scale`. Only used if `pool_center=True`.
    sigma_center_scale : scale of the HalfNormal hyperprior on the
        across-locus SD of logit(center) -- this is what directly answers
        the same question Firneno et al.'s own sigma_c does. Only used if
        `pool_center=True`.
    num_warmup, num_samples, num_chains, seed : standard NUTS/MCMC controls

    Returns
    -------
    HierarchicalGenomicFitResult
    """
    n_loci = n_alleles.shape[0]
    if locus_names is None:
        locus_names = [f"locus_{i}" for i in range(n_loci)]

    kernel = NUTS(_hierarchical_genomic_batch_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key,
              hybrid_index=jnp.asarray(hybrid_index), n_alleles=jnp.asarray(n_alleles),
              n_focal=jnp.asarray(n_focal), n_loci=n_loci,
              mu_u_scale=mu_u_scale, sigma_u_scale=sigma_u_scale,
              mu_log_v_loc=mu_log_v_loc, mu_log_v_scale=mu_log_v_scale,
              sigma_log_v_scale=sigma_log_v_scale, pool_center=pool_center,
              mu_center_scale=mu_center_scale, sigma_center_scale=sigma_center_scale)

    if pool_center:
        dims = {"logit_center": ["locus"], "log_v": ["locus"]}
    else:
        dims = {"u": ["locus"], "log_v": ["locus"]}
    idata = az.from_numpyro(mcmc, coords={"locus": locus_names}, dims=dims)
    result = HierarchicalGenomicFitResult(idata=idata, locus_names=locus_names)

    locus_df = result.locus_summary()
    _warn_if_hybrid_index_distribution_uneven(
        hybrid_index, locus_df["locus"], locus_df["u_mean"], locus_df["v_mean"])

    return result
