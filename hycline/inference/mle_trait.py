"""Maximum-likelihood fitting of a cline model to quantitative trait
data -- the trait analog of `hycline.inference.mle.fit_mle`, using the
full HZAR-parity Gaussian likelihood in `hycline.likelihoods.gaussian`.
See that module's docstring for the complete model (mean cline, variance
cline, tail models) and BACKEND_TRAIT.md for the full design history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from hycline.data.schema import TraitTransect
from hycline.likelihoods.gaussian import log_likelihood, negative_log_likelihood
from hycline.models.cline import ClineParams, TailType, param_names
from hycline.inference.mle import _numerical_hessian, _aicc


@dataclass
class TraitFitResult:
    """Result of `fit_mle_trait` -- a mean-cline shape (`mean_params`,
    reusing `ClineParams`: center, width, pmin(=muL), pmax(=muR), tail
    parameters) plus HZAR's own three variance parameters (`var_l`,
    `var_r`, `var_h` -- see `hycline.likelihoods.gaussian.
    variance_cline`'s docstring for exactly what they mean)."""
    mean_params: ClineParams
    var_l: float
    var_r: float
    var_h: float
    tail_type: TailType
    log_likelihood: float
    aicc: float
    n_params: int
    n_obs: int
    success: bool
    std_errors: np.ndarray | None = None
    param_vec: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        cline_param_count = self.n_params - 3  # var_l, var_r, var_h are always fitted
        names = param_names(self.tail_type)[:cline_param_count] + ["var_l", "var_r", "var_h"]
        lines = [
            f"Trait cline fit (tail={self.tail_type.value}, HZAR-parity variance model): "
            f"success={self.success}, logL={self.log_likelihood:.3f}, "
            f"AICc={self.aicc:.3f}, n={self.n_obs}, k={self.n_params}",
        ]
        for i, name in enumerate(names):
            val = self.param_vec[i]
            if self.std_errors is not None and i < len(self.std_errors) and np.isfinite(self.std_errors[i]):
                se = self.std_errors[i]
                lines.append(f"  {name:>10s} = {val:10.4f}  (SE {se:.4f}, "
                             f"95% CI [{val - 1.96*se:.4f}, {val + 1.96*se:.4f}])")
            else:
                lines.append(f"  {name:>10s} = {val:10.4f}")
        return "\n".join(lines)


def _initial_guess_trait(transect: TraitTransect, tail_type: TailType) -> np.ndarray:
    """Rough, data-driven starting values."""
    d = transect.distance
    m = transect.trait_mean
    v = transect.trait_sd ** 2
    order = np.argsort(d)
    d_sorted, m_sorted, v_sorted = d[order], m[order], v[order]

    k = max(1, len(m_sorted) // 5)
    pmin_guess = float(np.mean(m_sorted[:k]))
    pmax_guess = float(np.mean(m_sorted[-k:]))
    var_l_guess = float(max(np.mean(v_sorted[:k]), 1e-3))
    var_r_guess = float(max(np.mean(v_sorted[-k:]), 1e-3))

    mid = (pmin_guess + pmax_guess) / 2.0
    idx = np.argmin(np.abs(m_sorted - mid))
    center_guess = float(d_sorted[idx])
    width_guess = float(max((d.max() - d.min()) / 4.0, 1e-3))

    # var_h: excess variance in the center beyond a straight interpolation
    # of var_l/var_r (floored at a small positive value, since var_h >= 0
    # by construction -- see variance_cline).
    mid_var = float(np.median(v_sorted[len(v_sorted) // 3: 2 * len(v_sorted) // 3])) \
        if len(v_sorted) >= 3 else float(np.median(v_sorted))
    var_h_guess = max(mid_var - (var_l_guess + var_r_guess) / 2.0, 1e-3)

    params = ClineParams(center=center_guess, width=width_guess,
                          pmin=pmin_guess, pmax=pmax_guess)
    # HZAR's own tail edges are `center +/- delta` directly (a raw
    # distance, not a width-relative fraction -- see
    # hycline.likelihoods.gaussian's module docstring), so the initial
    # guess for delta should be a real distance, not a small fraction.
    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        params.delta_l, params.tau_l = width_guess / 2.0, 0.5
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        params.delta_r, params.tau_r = width_guess / 2.0, 0.5

    x0 = params.as_array(tail_type)
    x0 = np.append(x0, [var_l_guess, var_r_guess, var_h_guess])
    return x0


def _bounds_for_trait(transect: TraitTransect, tail_type: TailType) -> list[tuple]:
    d_span = transect.distance.max() - transect.distance.min()
    m_span = float(transect.trait_mean.max() - transect.trait_mean.min())
    m_span = max(m_span, 1e-6)
    m_min, m_max = transect.trait_mean.min(), transect.trait_mean.max()
    v_max = float(max(transect.trait_sd.max() ** 2, 1e-3))

    bounds = [
        (transect.distance.min() - d_span, transect.distance.max() + d_span),  # center
        (1e-3, d_span * 10 + 1e-3),                                            # width
        (m_min - 3 * m_span, m_max + 3 * m_span),                              # pmin (muL)
        (m_min - 3 * m_span, m_max + 3 * m_span),                              # pmax (muR)
    ]
    # delta_l/delta_r are HZAR's raw tail-edge distances (center +/- delta),
    # bounded to keep the edge within (or not too far outside) the observed
    # transect span; tau bounded strictly positive (see the tail formula's
    # exp(u*tau) blowing up for tau<=0 as x moves away from the edge).
    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        bounds += [(1e-3, d_span * 5 + 1e-3), (1e-3, 50.0)]
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        bounds += [(1e-3, d_span * 5 + 1e-3), (1e-3, 50.0)]
    bounds += [
        (1e-6, v_max * 10 + 1e-6),   # var_l
        (1e-6, v_max * 10 + 1e-6),   # var_r
        (0.0, v_max * 10 + 1e-6),    # var_h (>= 0, unlike var_l/var_r)
    ]
    return bounds


def fit_mle_trait(transect: TraitTransect, tail_type: TailType = TailType.NONE,
                   compute_std_errors: bool = True) -> TraitFitResult:
    """Fit a cline model to quantitative trait data with full HZAR
    parity: the mean follows the usual sigmoid cline shape, and the
    variance follows its OWN cline shape (`var_l`/`var_r`/`var_h`,
    sharing the same center/width/tail parameters) rather than a single
    constant -- see `hycline.likelihoods.gaussian`'s module docstring
    for the complete model, including tail models (also full HZAR
    parity, verified continuous at the tail/center boundary).

    Requires `transect.trait_sd` (raises ValueError if not set) -- HZAR's
    own likelihood always uses each site's observed sample variance as
    data, not just its mean.
    """
    if transect.trait_sd is None:
        raise ValueError(
            "fit_mle_trait requires transect.trait_sd (per-site SD of individual "
            "measurements) -- HZAR's own quantitative-trait likelihood always uses each "
            "site's observed sample variance as data, not just its mean."
        )
    if tail_type == TailType.T_CLINE:
        raise ValueError(
            "fit_mle_trait does not support tail_type=TailType.T_CLINE -- the T-cline "
            "(Baird & Daley 2025) was built for geographic/genomic allele-frequency "
            "clines and has no quantitative-trait variance-model counterpart here. "
            "Use TailType.NONE/LEFT/RIGHT/BOTH/MIRROR for trait clines instead."
        )

    x0 = _initial_guess_trait(transect, tail_type)
    bounds = _bounds_for_trait(transect, tail_type)

    result = minimize(
        negative_log_likelihood, x0, args=(transect, tail_type),
        method="L-BFGS-B", bounds=bounds,
    )

    mean_params = ClineParams.from_array(result.x[:-3], tail_type)
    var_l, var_r, var_h = float(result.x[-3]), float(result.x[-2]), float(result.x[-1])

    ll = log_likelihood(transect, mean_params, var_l, var_r, var_h, tail_type)
    k = len(result.x)
    n = len(transect)
    aicc = _aicc(ll, k, n)

    std_errors = None
    if compute_std_errors:
        try:
            hess = _numerical_hessian(
                lambda v: negative_log_likelihood(v, transect, tail_type),
                result.x
            )
            cov = np.linalg.inv(hess)
            variances = np.diag(cov)
            std_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
        except np.linalg.LinAlgError:
            std_errors = None

    return TraitFitResult(
        mean_params=mean_params, var_l=var_l, var_r=var_r, var_h=var_h,
        tail_type=tail_type, log_likelihood=ll, aicc=aicc, n_params=k, n_obs=n,
        success=bool(result.success), std_errors=std_errors, param_vec=result.x,
    )
