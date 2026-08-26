"""Secondary Bayesian inference backend: NUTS via PyMC.

See BACKEND_PHASE2.md section 6 for the scope limits relative to the primary
NumPyro backend (no batch/plate-vectorized multi-locus fitting, no
fixed-pmin/pmax convenience path yet). This exists for people who prefer
PyMC's API or already have it installed for other work -- for fitting many
loci at once, use `hycline.inference.mcmc_numpyro.fit_mcmc_batch` instead.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mcmc_numpyro import BayesianFitResult
from hycline.models.cline import TailType

_EPS = 1e-6


def fit_mcmc_pymc(transect: AlleleFreqTransect, tail_type: TailType = TailType.NONE,
                   num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 4,
                   seed: int = 0) -> BayesianFitResult:
    """Fit a single-locus cline model by NUTS MCMC via PyMC.

    Same model as `hycline.inference.mcmc_numpyro.fit_mcmc` (see BACKEND_PHASE2.md
    section 2) -- this is an alternative backend, not an alternative model.
    """
    if tail_type != TailType.NONE:
        raise NotImplementedError(
            "fit_mcmc_pymc currently only supports TailType.NONE -- see BACKEND_PHASE2.md "
            "section 12.5 for why (PyMC wasn't extended alongside NumPyro's tail-model "
            "support, given the underlying convergence difficulty is a gradient-based-"
            "sampling issue in general, not specific to NumPyro). Try --method classic "
            "or --backend numpyro instead (numpyro's tail-model Bayesian fitting has the "
            "same convergence caveat -- see section 12.2 -- but is at least implemented)."
        )

    distance = transect.distance
    n_alleles = transect.n_alleles
    n_focal = transect.n_focal

    d_span = distance.max() - distance.min()
    center_lo, center_hi = distance.min() - d_span, distance.max() + d_span
    width_hi = d_span * 10 + 1e-3

    with pm.Model() as model:
        center = pm.Uniform("center", lower=center_lo, upper=center_hi)
        width = pm.Uniform("width", lower=_EPS, upper=width_hi)
        pmin = pm.Uniform("pmin", lower=0.0, upper=1.0)
        pmax = pm.Uniform("pmax", lower=0.0, upper=1.0)

        z = pm.math.clip(-4.0 * (distance - center) / width, -700, 700)
        p = pmin + (pmax - pmin) / (1.0 + pm.math.exp(z))
        p = pm.math.clip(p, _EPS, 1.0 - _EPS)

        pm.Binomial("obs", n=n_alleles, p=p, observed=n_focal)

        # cores=1: PyMC's multiprocessing core-count auto-detection can
        # divide-by-zero in sandboxed/containerized environments where CPU
        # count detection misbehaves (see BACKEND_PHASE2.md section 6). Chains
        # still run sequentially and still support the R-hat diagnostic;
        # this only affects wall-clock parallelism, not correctness. Worth
        # revisiting on a normal (non-sandboxed) machine where cores=None
        # (PyMC's default auto-detection) should work fine.
        idata = pm.sample(draws=num_samples, tune=num_warmup, chains=num_chains,
                           random_seed=seed, progressbar=False, cores=1)

    result = BayesianFitResult(idata=idata, tail_type=tail_type, n_obs=len(transect),
                                backend="pymc", param_names=["center", "width", "pmin", "pmax"])
    return result
