"""Maximum-likelihood fitting of a cline model to allele-frequency data.

Uses scipy.optimize.minimize (L-BFGS-B, with box constraints keeping the
fit in a sane region) to find the MLE, then estimates parameter standard
errors from the inverse Hessian of the negative log-likelihood at the
optimum (a standard asymptotic approximation; see e.g. Barton & Gale 1993
and the HZAR manual, which uses MCMC credible intervals instead -- the
Bayesian backend in Phase 2 will supersede this for final reporting, but
the Hessian-based CI is a fast, dependency-light sanity check on its own).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from hycline.data.schema import AlleleFreqTransect
from hycline.likelihoods.binomial import log_likelihood, negative_log_likelihood
from hycline.models.cline import ClineParams, TailType, param_names


@dataclass
class FitResult:
    params: ClineParams
    tail_type: TailType
    log_likelihood: float
    aicc: float
    n_params: int
    n_obs: int
    success: bool
    hzar_tail: bool = False
    std_errors: np.ndarray | None = None
    param_vec: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        names = param_names(self.tail_type)[:len(self.param_vec)]
        if len(names) < len(self.param_vec):  # fixed-extremes fits: only center/width are free
            names = ["center", "width"]
        tail_label = self.tail_type.value
        if self.tail_type != TailType.NONE and self.hzar_tail:
            tail_label += ", HZAR-parameterized"
        lines = [
            f"Cline fit (tail={tail_label}): "
            f"success={self.success}, logL={self.log_likelihood:.3f}, "
            f"AICc={self.aicc:.3f}, n={self.n_obs}, k={self.n_params}",
        ]
        for i, name in enumerate(names):
            val = self.param_vec[i]
            if self.std_errors is not None and np.isfinite(self.std_errors[i]):
                se = self.std_errors[i]
                lines.append(f"  {name:>10s} = {val:10.4f}  (SE {se:.4f}, "
                             f"95% CI [{val - 1.96*se:.4f}, {val + 1.96*se:.4f}])")
            else:
                lines.append(f"  {name:>10s} = {val:10.4f}")
        return "\n".join(lines)


def _initial_guess(transect: AlleleFreqTransect, tail_type: TailType,
                    hzar_tail: bool = False) -> ClineParams:
    """Rough, data-driven starting values -- helps the optimizer converge
    reliably without the user needing to hand-tune a starting point.

    `hzar_tail=True` gives delta/tau starting values appropriate for
    HZAR's own tail parameterization (delta as a raw distance from
    center to the tail edge, not a [0,1] height fraction the way
    `cline_function`'s own tail model uses it) -- see
    `hycline.models.cline.hzar_cline_function`'s docstring for the full
    story of this real, named difference."""
    d = transect.distance
    f = transect.observed_freq
    order = np.argsort(d)
    d_sorted, f_sorted = d[order], f[order]

    pmin_guess = float(np.clip(np.mean(f_sorted[:max(1, len(f_sorted) // 5)]), 0.01, 0.99))
    pmax_guess = float(np.clip(np.mean(f_sorted[-max(1, len(f_sorted) // 5):]), 0.01, 0.99))

    # center guess: distance at which observed freq crosses the midpoint
    mid = (pmin_guess + pmax_guess) / 2.0
    idx = np.argmin(np.abs(f_sorted - mid))
    center_guess = float(d_sorted[idx])

    width_guess = float(max((d.max() - d.min()) / 4.0, 1e-3))

    params = ClineParams(center=center_guess, width=width_guess,
                          pmin=pmin_guess, pmax=pmax_guess)
    if hzar_tail:
        if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
            params.delta_l, params.tau_l = width_guess / 2.0, 0.5
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            params.delta_r, params.tau_r = width_guess / 2.0, 0.5
    else:
        if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
            params.delta_l, params.tau_l = 0.1, width_guess
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            params.delta_r, params.tau_r = 0.1, width_guess
    return params


def _bounds_for(transect: AlleleFreqTransect, tail_type: TailType,
                 hzar_tail: bool = False) -> list[tuple]:
    d_span = transect.distance.max() - transect.distance.min()
    bounds = [
        (transect.distance.min() - d_span, transect.distance.max() + d_span),  # center
        (1e-3, d_span * 10 + 1e-3),                                            # width
        (0.0, 1.0),                                                            # pmin
        (0.0, 1.0),                                                            # pmax
    ]
    if hzar_tail:
        # delta is a raw distance (center +/- delta defines the tail
        # edge) here, not a [0,1] height fraction -- tau bounded
        # strictly positive (the tail formula's exp(u*tau) blows up for
        # tau<=0 as x moves away from the edge). Matches
        # hycline.inference.mle_trait._bounds_for_trait's own bounds for
        # the same HZAR-parameterized tail.
        if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
            bounds += [(1e-3, d_span * 5 + 1e-3), (1e-3, 50.0)]
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            bounds += [(1e-3, d_span * 5 + 1e-3), (1e-3, 50.0)]
    else:
        if tail_type in (TailType.LEFT, TailType.BOTH):
            bounds += [(0.0, 1.0), (1e-3, d_span * 10 + 1e-3)]
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            bounds += [(0.0, 1.0), (1e-3, d_span * 10 + 1e-3)]
        if tail_type == TailType.MIRROR:
            bounds += [(0.0, 1.0), (1e-3, d_span * 10 + 1e-3)]
    return bounds


def _numerical_hessian(fun, x0, eps=1e-4) -> np.ndarray:
    """Central-difference Hessian of `fun` at `x0` (small, dependency-light;
    fine for the handful of parameters in a cline model)."""
    n = len(x0)
    hess = np.zeros((n, n))
    f0 = fun(x0)
    for i in range(n):
        for j in range(n):
            xpp, xpm, xmp, xmm = [x0.copy() for _ in range(4)]
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            hess[i, j] = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4 * eps * eps)
    return hess


def fit_mle(transect: AlleleFreqTransect, tail_type: TailType = TailType.NONE,
            compute_std_errors: bool = True, hzar_tail: bool = False) -> FitResult:
    """Fit a cline model to allele-frequency data by maximum likelihood.

    hzar_tail : if True, use HZAR's own tail parameterization
        (`hycline.models.cline.hzar_cline_function`, verified directly
        against HZAR's R source) instead of `cline_function`'s own
        (differently parameterized, not HZAR-verified) tail model.
        Default False preserves this project's original, unchanged
        behavior -- no previously-fitted results change unless this is
        explicitly requested. Ignored for `TailType.NONE` (both
        functions are identical there). See BACKEND_TRAIT.md sections
        1-2 for the full derivation and verification, and
        `hzar_cline_function`'s own docstring for the real, named
        difference in what `delta_l`/`tau_l`/`delta_r`/`tau_r` mean
        between the two parameterizations.
    """
    x0 = _initial_guess(transect, tail_type, hzar_tail=hzar_tail).as_array(tail_type)
    bounds = _bounds_for(transect, tail_type, hzar_tail=hzar_tail)

    result = minimize(
        negative_log_likelihood, x0, args=(transect, tail_type, hzar_tail),
        method="L-BFGS-B", bounds=bounds,
    )

    params = ClineParams.from_array(result.x, tail_type)
    ll = log_likelihood(transect, params, tail_type, hzar_tail=hzar_tail)
    k = len(result.x)
    n = len(transect)
    aicc = _aicc(ll, k, n)

    std_errors = None
    if compute_std_errors:
        try:
            hess = _numerical_hessian(
                lambda v: negative_log_likelihood(v, transect, tail_type, hzar_tail), result.x
            )
            cov = np.linalg.inv(hess)
            variances = np.diag(cov)
            std_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
        except np.linalg.LinAlgError:
            std_errors = None

    return FitResult(
        params=params, tail_type=tail_type, log_likelihood=ll, aicc=aicc,
        n_params=k, n_obs=n, success=bool(result.success), hzar_tail=hzar_tail,
        std_errors=std_errors, param_vec=result.x,
    )


def _aicc(log_lik: float, k: int, n: int) -> float:
    """Small-sample-corrected AIC, as used by HZAR's autofit model comparison."""
    aic = 2 * k - 2 * log_lik
    if n - k - 1 <= 0:
        return np.inf  # not enough data to correct reliably
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _derive_pmin_pmax_single_locus(distance: np.ndarray, n_alleles: np.ndarray,
                                    n_focal: np.ndarray, n_edge_sites: int = 1) -> tuple:
    """Single-locus analog of `hycline.inference.mcmc_hierarchical.
    _derive_pmin_pmax`, used by `fit_mle_fixed_extremes` below.

    `n_edge_sites=1` (the default here) reproduces the original single-
    site convention *exactly* -- HZAR's own convention for "fixed"
    scaling, and what Phase 1's Manakin validation was checked against
    (BACKEND_PHASE1.md's own cross-validation). This default is
    deliberately unchanged: switching it would silently alter previously-
    validated results for a case (larger, well-behaved sample sizes like
    Manakin's) where BACKEND_PHASE4.md section 3 already found the
    single-site fragility wasn't a dominant issue.

    Pass `n_edge_sites` > 1 to use the more robust multi-site-averaged
    version instead -- pooling counts (sum(n_focal)/sum(n_alleles)) over
    the `n_edge_sites` sites nearest each end of the transect, rather
    than anchoring on one site's frequency. This matters most for
    smaller sample sizes, where a true pmin near 0 (or pmax near 1) can
    easily draw an exact 0/n (or n/n) at a single site by chance -- not a
    rare event, and one that locks the fixed asymptote at the wrong value
    and can force the optimizer to substantially widen the fitted cline
    to compensate. Found and fixed for the hierarchical model first
    (BACKEND_PHASE4.md section 3, with a worked example); this is the
    same fix, made available here too rather than left as a documented-
    but-unaddressed gap.
    """
    order = np.argsort(distance)
    n_alleles_sorted = n_alleles[order]
    n_focal_sorted = n_focal[order]
    k = min(n_edge_sites, len(n_alleles_sorted))

    pmin = float(n_focal_sorted[:k].sum() / n_alleles_sorted[:k].sum())
    pmax = float(n_focal_sorted[-k:].sum() / n_alleles_sorted[-k:].sum())
    return pmin, pmax


def fit_mle_fixed_extremes(transect: AlleleFreqTransect,
                            tail_type: TailType = TailType.NONE,
                            pmin: float | None = None,
                            pmax: float | None = None,
                            n_edge_sites: int = 1,
                            compute_std_errors: bool = True) -> FitResult:
    """Fit with pmin/pmax fixed rather than estimated -- HZAR's
    ``scaling="fixed"`` mode. If not supplied, pmin/pmax default to the
    observed frequency at the transect's geographic extremes -- by
    default, a single site at each end (`n_edge_sites=1`), matching
    HZAR's own convention exactly and what this project's Manakin
    cross-validation was checked against. Pass `n_edge_sites` > 1 (e.g.
    3, matching the hierarchical model's own default) for the more
    sample-size-robust multi-site-averaged version instead -- see
    `_derive_pmin_pmax_single_locus`'s docstring for why this matters and
    when to reach for it.

    This is currently only supported for TailType.NONE (sufficient to
    reproduce HZAR's published "Model I" comparison; tail support can be
    added the same way once needed).
    """
    if tail_type != TailType.NONE:
        raise NotImplementedError(
            "fit_mle_fixed_extremes currently only supports TailType.NONE"
        )

    if pmin is None or pmax is None:
        derived_pmin, derived_pmax = _derive_pmin_pmax_single_locus(
            transect.distance, transect.n_alleles, transect.n_focal, n_edge_sites)
        if pmin is None:
            pmin = derived_pmin
        if pmax is None:
            pmax = derived_pmax

    guess = _initial_guess(transect, tail_type)
    x0 = np.array([guess.center, guess.width])
    d_span = transect.distance.max() - transect.distance.min()
    bounds = [
        (transect.distance.min() - d_span, transect.distance.max() + d_span),
        (1e-3, d_span * 10 + 1e-3),
    ]

    def neg_ll(cw):
        center, width = cw
        if width <= 0:
            return np.inf
        params = ClineParams(center=center, width=width, pmin=pmin, pmax=pmax)
        return -log_likelihood(transect, params, tail_type)

    result = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)

    center, width = result.x
    params = ClineParams(center=center, width=width, pmin=pmin, pmax=pmax)
    ll = log_likelihood(transect, params, tail_type)
    k = 2  # only center and width are free parameters here
    n = len(transect)
    aicc = _aicc(ll, k, n)

    std_errors = None
    if compute_std_errors:
        try:
            hess = _numerical_hessian(neg_ll, result.x)
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
