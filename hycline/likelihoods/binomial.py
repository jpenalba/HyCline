"""Binomial likelihood for allele-frequency cline data.

At each sampled site, the observed count of the focal allele out of the
total sampled alleles is treated as Binomial(n_alleles, p(x)), where p(x)
is the cline model's predicted frequency at that site's geographic position.
This is the likelihood HZAR uses for its "freq" data type.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import (ClineParams, TailType, cline_function,
                                   hzar_cline_function, t_cline_function)

# Numerical floor to keep log(0) from appearing when a fitted p(x) lands
# exactly at 0 or 1 but the data disagrees.
_EPS = 1e-10


def predicted_freq(transect: AlleleFreqTransect, params: ClineParams,
                    tail_type: TailType, hzar_tail: bool = False) -> np.ndarray:
    """`hzar_tail=True` uses `hzar_cline_function` (HZAR's own, verified
    tail parameterization -- BACKEND_TRAIT.md sections 1-2) instead of
    `cline_function`'s own tail model. Default `False` preserves this
    project's original, unchanged behavior. `tail_type=TailType.T_CLINE`
    dispatches to `t_cline_function` instead of either -- the T-cline is a
    standalone continuous shape model, not a choice of tail
    parameterization, so `hzar_tail` has no meaning for it and must be
    False."""
    if tail_type == TailType.T_CLINE:
        if hzar_tail:
            raise ValueError(
                "hzar_tail is not meaningful with TailType.T_CLINE -- the T-cline "
                "has no HZAR-parameterized variant (see t_cline_function's docstring)."
            )
        p = t_cline_function(transect.distance, params)
    else:
        fn = hzar_cline_function if hzar_tail else cline_function
        p = fn(transect.distance, params, tail_type)
    return np.clip(p, _EPS, 1.0 - _EPS)


def log_likelihood(transect: AlleleFreqTransect, params: ClineParams,
                    tail_type: TailType = TailType.NONE, hzar_tail: bool = False) -> float:
    """Total binomial log-likelihood of the transect data under the model."""
    p = predicted_freq(transect, params, tail_type, hzar_tail=hzar_tail)
    ll = binom.logpmf(transect.n_focal, transect.n_alleles, p)
    return float(np.sum(ll))


def negative_log_likelihood(param_vec: np.ndarray, transect: AlleleFreqTransect,
                             tail_type: TailType, hzar_tail: bool = False) -> float:
    """Objective function for the optimizer (minimizes -log-likelihood)."""
    params = ClineParams.from_array(param_vec, tail_type)
    if params.width <= 0:
        return np.inf
    if not (0.0 <= params.pmin <= 1.0 and 0.0 <= params.pmax <= 1.0):
        return np.inf
    if tail_type == TailType.T_CLINE and (params.t_shape is None or params.t_shape <= 0):
        return np.inf
    if tail_type == TailType.T_CLINE and hzar_tail:
        return np.inf
    if hzar_tail:
        for name in ("delta_l", "delta_r", "tau_l", "tau_r"):
            val = getattr(params, name, None)
            if val is not None and val <= 0:
                return np.inf
    return -log_likelihood(transect, params, tail_type, hzar_tail=hzar_tail)


def hzar_style_log_likelihood(transect: AlleleFreqTransect, params: ClineParams,
                               tail_type: TailType = TailType.NONE,
                               hzar_tail: bool = False) -> float:
    """Log-likelihood reported the way HZAR itself reports it -- NOT the raw
    binomial log-likelihood used internally by `log_likelihood()` above.

    Reverse-engineered from HZAR's R source (`sampleLikelihoodMolecularPop`
    in R/20-hzarPrep.R of the CRAN `hzar` package). At each site, HZAR
    reports the log of the likelihood *ratio* between the fitted model and
    the saturated model (i.e. the model where p is set to the site's own
    observed frequency, which maximizes the binomial likelihood for that
    site). Algebraically this is:

        raw_binomial_kernel(p_est) - raw_binomial_kernel(p_obs)

    i.e. the standard binomial log-likelihood kernel, minus a constant (with
    respect to model parameters) offset equal to the same kernel evaluated
    at the saturated fit. That offset is why HZAR's reported log-likelihoods
    sit close to 0 for a good fit, while the raw binomial log-likelihood from
    `log_likelihood()` above does not -- but because the offset doesn't
    depend on the model parameters, it does not change *which* parameter
    values maximize the likelihood, nor which nested model AICc prefers.
    This function exists purely so fitted results can be reported in
    HZAR-comparable units; `log_likelihood()` remains the function actually
    used for optimization and internal AICc computation.
    """
    p_est = predicted_freq(transect, params, tail_type, hzar_tail=hzar_tail)
    p_obs = np.clip(transect.n_focal / transect.n_alleles, _EPS, 1.0 - _EPS)
    n = transect.n_alleles

    with np.errstate(divide="ignore", invalid="ignore"):
        general = n * ((1 - p_obs) * np.log((1 - p_est) / (1 - p_obs))
                        + p_obs * np.log(p_est / p_obs))
    at_pobs_1 = n * np.log(p_est)
    at_pobs_0 = n * np.log(1 - p_est)

    res = np.where(p_obs >= 1.0 - _EPS, at_pobs_1, general)
    res = np.where(p_obs <= _EPS, at_pobs_0, res)
    return float(np.sum(res))
