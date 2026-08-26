"""Maximum-likelihood fitting of the genomic cline model to per-individual
locus data against hybrid index.

Mirrors `hycline.inference.mle`'s structure and approach (L-BFGS-B via
scipy.optimize, data-driven initial guess, numerical-Hessian standard
errors) -- see that module and BACKEND_PHASE1.md for the general rationale
behind these choices; this module doesn't repeat that reasoning, only
what's specific to the genomic cline model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from hycline.data.schema import HybridIndexLocusData
from hycline.likelihoods.genomic_binomial import log_likelihood, negative_log_likelihood
from hycline.models.genomic_cline import GenomicClineParams, param_names


@dataclass
class GenomicFitResult:
    params: GenomicClineParams
    log_likelihood: float
    aicc: float
    n_params: int
    n_obs: int
    success: bool
    std_errors: np.ndarray | None = None
    param_vec: np.ndarray = None

    def summary(self) -> str:
        names = param_names()
        lines = [
            f"Genomic cline fit: success={self.success}, logL={self.log_likelihood:.3f}, "
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
        # a quick, interpretable note on what the fitted v means
        if np.isfinite(self.param_vec[1]):
            v = self.param_vec[1]
            if v > 1.05:
                interp = "steeper than genome-wide average (elevated barrier to introgression)"
            elif v < 0.95:
                interp = "shallower than genome-wide average (elevated introgression)"
            else:
                interp = "close to the neutral genome-wide average (v ~ 1)"
            lines.append(f"  interpretation: {interp}")
        return "\n".join(lines)


def _initial_guess(data: HybridIndexLocusData) -> GenomicClineParams:
    """Data-driven starting values, same motivation as
    hycline.inference.mle._initial_guess: land the optimizer in a
    reasonable basin rather than relying on generic defaults."""
    # crude initial u: logit of the observed frequency at h close to 0.5
    freq = data.observed_freq
    mid_mask = np.abs(data.hybrid_index - 0.5) < 0.25
    if np.any(mid_mask):
        mid_freq = np.clip(np.mean(freq[mid_mask]), 0.02, 0.98)
    else:
        mid_freq = np.clip(np.mean(freq), 0.02, 0.98)
    u0 = float(np.log(mid_freq / (1 - mid_freq)))
    return GenomicClineParams(u=u0, v=1.0)


def _bounds() -> list[tuple]:
    return [(-15.0, 15.0), (1e-3, 20.0)]


def _numerical_hessian(fun, x0, eps=1e-4) -> np.ndarray:
    n = len(x0)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp, xpm, xmp, xmm = [x0.copy() for _ in range(4)]
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            hess[i, j] = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4 * eps * eps)
    return hess


def _aicc(log_lik: float, k: int, n: int) -> float:
    aic = 2 * k - 2 * log_lik
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def fit_genomic_cline_mle(data: HybridIndexLocusData,
                           compute_std_errors: bool = True) -> GenomicFitResult:
    """Fit the logit-logistic genomic cline model to one locus's data by
    maximum likelihood."""
    x0 = _initial_guess(data).as_array()
    bounds = _bounds()

    result = minimize(negative_log_likelihood, x0, args=(data,),
                       method="L-BFGS-B", bounds=bounds)

    params = GenomicClineParams.from_array(result.x)
    ll = log_likelihood(data, params)
    k = len(result.x)
    n = len(data)
    aicc = _aicc(ll, k, n)

    std_errors = None
    if compute_std_errors:
        try:
            hess = _numerical_hessian(lambda v: negative_log_likelihood(v, data), result.x)
            cov = np.linalg.inv(hess)
            variances = np.diag(cov)
            std_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
        except np.linalg.LinAlgError:
            std_errors = None

    return GenomicFitResult(
        params=params, log_likelihood=ll, aicc=aicc, n_params=k, n_obs=n,
        success=bool(result.success), std_errors=std_errors, param_vec=result.x,
    )
