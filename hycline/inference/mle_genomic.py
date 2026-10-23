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
from hycline.likelihoods.genomic_binomial import (log_likelihood, log_likelihood_at,
                                                  negative_log_likelihood)
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
    # Parental frequencies: with their panel posteriors on the data, their
    # posterior means (integrated out of the fit), otherwise the given ones.
    parental_freqs: tuple | None = None
    parental_freqs_estimated: bool = False

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
        if self.parental_freqs_estimated:
            for k, name in enumerate(("p0", "p1")):
                se = (self.std_errors[2 + k] if self.std_errors is not None
                      and len(self.std_errors) > 2 + k else np.nan)
                lines.append(f"  {name:>10s} = {self.parental_freqs[k]:10.4f}"
                             + (f"  (posterior SD {se:.4f}, from the reference panel "
                                f"and the hybrids)" if np.isfinite(se) else ""))
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
    # crude initial u: logit of the observed ancestry proportion at h close
    # to 0.5, i.e. the observed frequency mapped onto the phi scale
    p0, p1 = data.parental_freq_pop0, data.parental_freq_pop1
    freq = (data.observed_freq - p0) / (p1 - p0)
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


def fit_with_parental_panels(data, loglik_at, x0_uv, bounds_uv,
                             compute_std_errors: bool = True,
                             n_nodes: int = 16, max_recentre: int = 5) -> GenomicFitResult:
    """Fit u and v with the parental frequencies integrated out, for data
    that carry their Beta posteriors (`parental_beta_pop0/1`, from the
    reference panels: `hycline.data.filter_aims.panel_beta_posteriors`).
    The fit maximizes the hybrids' likelihood averaged over both:

        L(u, v) = E_{p0, p1} [ L_hybrids(u, v, p0, p1) ],

    by Gauss-Jacobi quadrature (`beta_quadrature`, n_nodes per
    frequency). Averaging rather than maximizing over p0/p1 matters:
    maximizing jointly (a profile likelihood) biases v upward, because
    pulling p0 and p1 together while steepening the cline fits a few
    dozen hybrids almost as well, and every locus has its own two
    nuisance frequencies.

    The quadrature is adaptive. A first fit uses nodes spread over the
    panel posteriors; the fit is then repeated with nodes placed over
    p0/p1's posterior given the panels, the hybrids and the last cline
    (a moment-matched Beta, widened by half), each node weighted by panel
    density / node density, until that posterior's mean and variance
    settle (at most max_recentre times). With many hybrids the hybrids pin
    p0/p1 down more tightly than the panel does, and the re-centred nodes
    are where the integrand actually is.

    The u/v standard errors (from the Hessian of log L(u, v)) include the
    panels' sampling error. The reported p0/p1 are their posterior means
    given the panels, the hybrids and the fitted cline, with posterior
    SDs in `std_errors[2:]`. `loglik_at(data, params, p0, p1)` returns the
    hybrids' log-likelihood for each of an array of (p0, p1) pairs.
    """
    from scipy.special import logsumexp
    from hycline.models.genomic_cline import parental_nodes, widened_beta
    panels = [data.parental_beta_pop0, data.parental_beta_pop1]

    def node_set(betas):
        return parental_nodes(panels, betas, n_nodes)

    def fit(nodes, x0):
        p0_nodes, p1_nodes, log_w = nodes

        def node_terms(x):
            return log_w + loglik_at(data, GenomicClineParams.from_array(x), p0_nodes, p1_nodes)

        def objective(x):
            if x[1] <= 0:
                return np.inf
            return -float(logsumexp(node_terms(x)))

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds_uv)
        terms = node_terms(result.x)
        post = np.exp(terms - logsumexp(terms))
        mean = np.array([post @ p0_nodes, post @ p1_nodes])
        var = np.maximum(np.array([post @ p0_nodes ** 2, post @ p1_nodes ** 2]) - mean ** 2,
                         1e-14)
        return result, objective, mean, var

    result, objective, p_mean, p_var = fit(node_set(panels), np.asarray(x0_uv, dtype=float))
    for _ in range(max_recentre):
        # nodes over p0/p1's posterior at the last fit (variance widened 1.5x)
        mean, var = p_mean, p_var
        result, objective, p_mean, p_var = fit(
            node_set(list(zip(*widened_beta(mean, var, 1.5)))), result.x)
        if (np.all(np.abs(p_mean - mean) < 0.1 * np.sqrt(p_var))
                and np.all(np.abs(np.log(p_var / var)) < 0.2)):
            break
    ll = -float(result.fun)
    k, n = len(result.x), len(data)

    std_errors = None
    if compute_std_errors:
        try:
            cov = np.linalg.inv(_numerical_hessian(objective, result.x))
            variances = np.diag(cov)
            std_errors = np.concatenate([np.sqrt(np.where(variances > 0, variances, np.nan)),
                                         np.sqrt(p_var)])
        except np.linalg.LinAlgError:
            std_errors = None

    return GenomicFitResult(
        params=GenomicClineParams.from_array(result.x), log_likelihood=ll,
        aicc=_aicc(ll, k, n), n_params=k, n_obs=n, success=bool(result.success),
        std_errors=std_errors, param_vec=np.concatenate([result.x, p_mean]),
        parental_freqs=(float(p_mean[0]), float(p_mean[1])),
        parental_freqs_estimated=True,
    )


def fit_genomic_cline_mle(data: HybridIndexLocusData,
                           compute_std_errors: bool = True) -> GenomicFitResult:
    """Fit the logit-logistic genomic cline model to one locus's data by
    maximum likelihood. With parental-frequency posteriors on the data
    (`parental_beta_pop0/1`), the parental frequencies are integrated out
    (`fit_with_parental_panels`); otherwise they are taken as given."""
    if data.parental_beta_pop0 is not None:
        return fit_with_parental_panels(data, log_likelihood_at,
                                        _initial_guess(data).as_array(), _bounds(),
                                        compute_std_errors)
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
        parental_freqs=(data.parental_freq_pop0, data.parental_freq_pop1),
    )
