"""Maximum-likelihood fitting of geographic and genomic clines directly
from genotype likelihoods, marginalizing over genotype uncertainty rather
than requiring a hard genotype call first.

Mirrors `hycline.inference.mle`/`mle_genomic`'s structure and approach
(L-BFGS-B via scipy.optimize, data-driven initial guess, numerical-Hessian
standard errors) -- see those modules and BACKEND_PHASE1.md/BACKEND_PHASE3.md
for the general rationale, not repeated here. This module reuses the same
`ClineParams`/`GenomicClineParams`/`FitResult`/`GenomicFitResult` classes
as the hard-count fitting paths -- only the likelihood function underneath
differs, so results from GL-aware and hard-count fits are directly
comparable using the exact same reporting machinery.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from hycline.data.schema import GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData
from hycline.inference.mle import FitResult, _numerical_hessian
from hycline.inference.mle_genomic import GenomicFitResult
from hycline.likelihoods.genotype_likelihood import (
    log_likelihood_geographic, negative_log_likelihood_geographic,
    log_likelihood_genomic, negative_log_likelihood_genomic,
)
from hycline.models.cline import ClineParams, TailType
from hycline.models.genomic_cline import GenomicClineParams


def _aicc(log_lik: float, k: int, n: int) -> float:
    aic = 2 * k - 2 * log_lik
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _initial_guess_geographic(transect: GenotypeLikelihoodTransect,
                               tail_type: TailType = TailType.NONE) -> ClineParams:
    """Data-driven starting values, using each individual's most-likely
    genotype (argmax of the three GLs) as a rough stand-in for the true
    genotype -- fine as an optimizer starting point even though the fit
    itself properly marginalizes over the uncertainty this discards."""
    gl_stack = np.stack([transect.log_gl0, transect.log_gl1, transect.log_gl2], axis=0)
    hard_genotype = np.argmax(gl_stack, axis=0)
    freq_proxy = hard_genotype / 2.0

    order = np.argsort(transect.distance)
    d_sorted, f_sorted = transect.distance[order], freq_proxy[order]

    pmin_guess = float(np.clip(np.mean(f_sorted[:max(1, len(f_sorted) // 5)]), 0.01, 0.99))
    pmax_guess = float(np.clip(np.mean(f_sorted[-max(1, len(f_sorted) // 5):]), 0.01, 0.99))
    mid = (pmin_guess + pmax_guess) / 2.0
    idx = np.argmin(np.abs(f_sorted - mid))
    center_guess = float(d_sorted[idx])
    width_guess = float(max((transect.distance.max() - transect.distance.min()) / 4.0, 1e-3))

    params = ClineParams(center=center_guess, width=width_guess, pmin=pmin_guess, pmax=pmax_guess)
    # tail starting values match Phase 1 classic MLE's own defaults exactly
    # (`hycline.inference.mle._initial_guess`) -- a modest height fraction,
    # decay length on the same scale as the width guess
    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        params.delta_l, params.tau_l = 0.1, width_guess
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        params.delta_r, params.tau_r = 0.1, width_guess
    return params


def _bounds_for_geographic_gl(transect: GenotypeLikelihoodTransect,
                               tail_type: TailType) -> list:
    """Same bounds convention as Phase 1 classic MLE's `_bounds_for`
    (`hycline.inference.mle`), applied to genotype-likelihood data."""
    d_span = transect.distance.max() - transect.distance.min()
    bounds = [
        (transect.distance.min() - d_span, transect.distance.max() + d_span),
        (1e-3, d_span * 10 + 1e-3),
        (0.0, 1.0), (0.0, 1.0),
    ]
    if tail_type in (TailType.LEFT, TailType.BOTH):
        bounds += [(0.0, 1.0), (1e-3, d_span * 10 + 1e-3)]
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        bounds += [(0.0, 1.0), (1e-3, d_span * 10 + 1e-3)]
    if tail_type == TailType.MIRROR:
        bounds += [(0.0, 1.0), (1e-3, d_span * 10 + 1e-3)]
    return bounds


def fit_mle_geographic_gl(transect: GenotypeLikelihoodTransect,
                           tail_type: TailType = TailType.NONE,
                           compute_std_errors: bool = True,
                           n_restarts: int = 8, seed: int = 0) -> FitResult:
    """Fit a geographic cline directly from per-individual genotype
    likelihoods, marginalizing over genotype uncertainty (see this
    module's docstring). Returns the same `FitResult` type as
    `hycline.inference.mle.fit_mle` for direct comparability.

    Supports any TailType (LEFT/RIGHT/BOTH/MIRROR), not just NONE --
    `log_likelihood_geographic` already dispatches to `cline_function`
    generically for any tail type, so extending this to tail models was a
    matter of extending the initial-guess/bounds construction to match
    Phase 1's classic-MLE convention exactly (`_initial_guess_geographic`/
    `_bounds_for_geographic_gl` above), not a new likelihood.

    A REAL finding surfaced while validating this extension
    (BACKEND_PHASE2.md section 12.3): the tail model's join between the
    sigmoid core and the exponential tail creates a non-smooth (kinked)
    likelihood surface that even L-BFGS-B (a quasi-Newton method, not
    gradient-free) can get stuck on -- from a single starting point,
    optimization can terminate with scipy's `success=False` even when a
    genuinely good solution exists nearby, because the line search fails
    strict convergence criteria near the kink, not because no good
    solution was found. Unlike the analogous (and more severe) issue for
    Bayesian tail-model fitting via NUTS (BACKEND_PHASE2.md section
    12.2, where more restarts don't help since MCMC explores one
    connected trajectory rather than independent attempts), classical
    optimization is naturally suited to a multi-start fix: for
    `tail_type != TailType.NONE`, this function runs `n_restarts`
    independent L-BFGS-B optimizations from randomized starting points
    (in addition to the data-driven initial guess) and keeps the one with
    the best (lowest) negative log-likelihood, verified directly to find
    solutions with *higher* likelihood than a single-start fit
    (`tests/test_mle_genotype_likelihood_tail.py::
    test_multistart_finds_better_optimum_than_single_start`).

    Parameters
    ----------
    transect, tail_type, compute_std_errors : as before
    n_restarts : number of additional randomized restarts to try when
        `tail_type != TailType.NONE` (ignored for TailType.NONE, which
        doesn't show this issue and doesn't need the extra cost).
        Default 8, a modest but meaningfully more robust number given
        each restart is a fast bounded optimization.
    seed : RNG seed for the randomized restart starting points
    """
    bounds = _bounds_for_geographic_gl(transect, tail_type)

    x0 = _initial_guess_geographic(transect, tail_type).as_array(tail_type)
    best_result = minimize(negative_log_likelihood_geographic, x0, args=(transect, tail_type),
                            method="L-BFGS-B", bounds=bounds)

    if tail_type != TailType.NONE:
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            x0_random = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            candidate = minimize(negative_log_likelihood_geographic, x0_random,
                                  args=(transect, tail_type), method="L-BFGS-B", bounds=bounds)
            if candidate.fun < best_result.fun:
                best_result = candidate

    result = best_result
    params = ClineParams.from_array(result.x, tail_type)
    ll = log_likelihood_geographic(transect, params, tail_type)
    k = len(result.x)
    n = len(transect)
    aicc = _aicc(ll, k, n)

    std_errors = None
    if compute_std_errors:
        try:
            hess = _numerical_hessian(
                lambda v: negative_log_likelihood_geographic(v, transect, tail_type), result.x)
            cov = np.linalg.inv(hess)
            variances = np.diag(cov)
            std_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
        except np.linalg.LinAlgError:
            std_errors = None

    return FitResult(
        params=params, tail_type=tail_type, log_likelihood=ll, aicc=aicc,
        n_params=k, n_obs=n, success=bool(result.success),
        std_errors=std_errors, param_vec=result.x,
    )


def _initial_guess_genomic(data: GenotypeLikelihoodLocusData) -> GenomicClineParams:
    gl_stack = np.stack([data.log_gl0, data.log_gl1, data.log_gl2], axis=0)
    hard_genotype = np.argmax(gl_stack, axis=0)
    p0, p1 = data.parental_freq_pop0, data.parental_freq_pop1
    freq_proxy = (hard_genotype / 2.0 - p0) / (p1 - p0)  # onto the phi scale

    mid_mask = np.abs(data.hybrid_index - 0.5) < 0.25
    if np.any(mid_mask):
        mid_freq = np.clip(np.mean(freq_proxy[mid_mask]), 0.02, 0.98)
    else:
        mid_freq = np.clip(np.mean(freq_proxy), 0.02, 0.98)
    u0 = float(np.log(mid_freq / (1 - mid_freq)))
    return GenomicClineParams(u=u0, v=1.0)


def fit_mle_genomic_gl(data: GenotypeLikelihoodLocusData,
                        compute_std_errors: bool = True) -> GenomicFitResult:
    """Fit a genomic cline directly from per-individual genotype
    likelihoods, marginalizing over genotype uncertainty. Returns the same
    `GenomicFitResult` type as
    `hycline.inference.mle_genomic.fit_genomic_cline_mle` for direct
    comparability.
    """
    x0 = _initial_guess_genomic(data).as_array()
    bounds = [(-15.0, 15.0), (1e-3, 20.0)]

    result = minimize(negative_log_likelihood_genomic, x0, args=(data,),
                       method="L-BFGS-B", bounds=bounds)

    params = GenomicClineParams.from_array(result.x)
    ll = log_likelihood_genomic(data, params)
    k = len(result.x)
    n = len(data)
    aicc = _aicc(ll, k, n)

    std_errors = None
    if compute_std_errors:
        try:
            hess = _numerical_hessian(
                lambda v: negative_log_likelihood_genomic(v, data), result.x)
            cov = np.linalg.inv(hess)
            variances = np.diag(cov)
            std_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
        except np.linalg.LinAlgError:
            std_errors = None

    return GenomicFitResult(
        params=params, log_likelihood=ll, aicc=aicc, n_params=k, n_obs=n,
        success=bool(result.success), std_errors=std_errors, param_vec=result.x,
    )
