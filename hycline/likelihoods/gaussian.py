"""Gaussian likelihood for quantitative trait cline data -- HZAR's other
core data type, alongside `AlleleFreqTransect`'s molecular-marker
binomial likelihood (`hycline.likelihoods.binomial`). Any continuous,
measured phenotype that varies smoothly across a hybrid zone can be fit
with the same sigmoid cline shape used for allele frequencies, using a
Gaussian likelihood in place of the binomial one.

FULL HZAR PARITY, mean AND variance, including tail models -- verified
directly against HZAR's own R source
(https://github.com/cran/hzar: `25-guassianPrep.R`,
`15-modelExpressionFunctions.R`), not a principled reimplementation.
This is the ONLY trait-cline likelihood in this project -- an earlier,
simpler design (a single shared variance, ignoring each site's own
sample variance) was removed once HZAR's real model was checked and
found to genuinely differ; see BACKEND_TRAIT.md section 2 for that
history.

THE MODEL, in full:

- The mean follows a sigmoid cline (muL to muR), exactly as for allele
  frequencies.
- The variance ALSO follows a cline of its own -- from `var_l` to
  `var_r`, sharing the same center/width/tail shape as the mean, plus an
  extra term `var_h` representing excess variance specifically in the
  hybrid-zone center (segregation variance among admixed individuals,
  absent in either pure parental population -- a real, well-documented
  biological phenomenon, not an ad hoc addition).
- The log-likelihood uses each site's *observed sample variance* as
  data (a sufficient statistic, alongside the sample mean and n), not
  just the mean -- this is exactly the log-likelihood of every
  individual raw observation at a site, expressed via sufficient
  statistics rather than requiring the raw data itself. Confirmed
  algebraically and by direct numerical comparison against a raw-data
  computation (`tests/test_mle_trait.py::
  test_hzar_log_likelihood_matches_raw_individual_data_likelihood`).
- Tail models (`LEFT`/`RIGHT`/`BOTH`/`MIRROR`) use HZAR's own tail
  construction for BOTH the mean and the variance -- a distance-scaled
  exponential decay (not the allele-frequency `cline_function`'s own,
  differently-parameterized tail model -- see `_cline_shape` below for
  exactly how these differ and why this module doesn't reuse
  `hycline.models.cline.cline_function` for tail cases). Verified
  continuous at the tail/center boundary by direct construction
  (`tests/test_mle_trait.py::
  test_hzar_tail_formula_is_continuous_with_center_formula_at_boundary`).

A REAL, NAMED DIFFERENCE FROM THE ALLELE-FREQUENCY CLINE MODEL: HZAR's
own tail edges are defined as `center - delta_l` / `center + delta_r`
(the delta parameters ARE the raw distance from center to the tail
edge), not `center - width/2` / `center + width/2` the way
`hycline.models.cline.cline_function`'s own (separately-parameterized,
already-documented-as-not-HZAR-verified) tail model defines them. This
means `delta_l`/`tau_l`/`delta_r`/`tau_r` mean something genuinely
different here than they do for allele-frequency fitting, even though
the same `ClineParams` container (for the mean-cline shape only) is
reused for convenience -- stated plainly here rather than left as an
implicit trap.
"""

from __future__ import annotations

import numpy as np

from hycline.data.schema import TraitTransect
from hycline.models.cline import ClineParams, TailType


def _tail_alpha(delta: float, tau: float, width: float) -> float:
    """HZAR's own `alphaExpF`: the tail-region scaling constant, chosen
    so the tail formula is continuous with the center formula at the
    tail edge (verified directly, not just intended --
    `tests/test_mle_trait.py::
    test_hzar_tail_formula_is_continuous_with_center_formula_at_boundary`)."""
    logit_delta = 4.0 * delta / width
    logit_delta = np.clip(logit_delta, -700, 700)
    return np.exp(logit_delta * (tau - 1.0)) / (1.0 + np.exp(-logit_delta))


def _cline_shape(distance: np.ndarray, center: float, width: float,
                  tail_type: TailType,
                  delta_l: float | None = None, tau_l: float | None = None,
                  delta_r: float | None = None, tau_r: float | None = None
                  ) -> tuple:
    """Shared core for both the mean-cline and variance-cline
    evaluators: computes, at every position, HZAR's 0->1 interpolation
    fraction (`frac`, going from the "L" boundary towards the "R"
    boundary) and the corresponding "extra variance" shape value
    (`var_shape` -- the center-region bump function in the no-tail
    region, or the tail region's own `frac*(1-frac)` term, which is
    HZAR's own construction -- see `vTailExpF` in HZAR's R source).

    Center region (bit-verified against HZAR's `sigmoidExpF`/
    `vCenterExpF`):
        u = 4*(x-center)/width
        frac = 1/(1+exp(-u))
        var_shape = 1/(2+exp(u)+exp(-u))   -- peaks at x=center

    Tail regions (bit-verified against HZAR's `pTailExpF`/`vTailExpF`,
    continuous with the center-region formula at the tail edge): for
    `TailType.MIRROR`, `delta_l`/`tau_l` are reused for *both* tail
    edges (matching this project's own existing convention for the
    allele-frequency tail model, where a single shared value plays the
    same role as HZAR's separate `deltaM`/`tauM`).
    """
    x = np.asarray(distance, dtype=float)
    u = np.clip(4.0 * (x - center) / width, -700, 700)
    frac = 1.0 / (1.0 + np.exp(-u))
    var_shape = 1.0 / (2.0 + np.exp(u) + np.exp(-u))

    if tail_type == TailType.NONE:
        return frac, var_shape

    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        dl = delta_l
        tl = tau_l
        left_edge = center - dl
        mask = x < left_edge
        if np.any(mask):
            alpha = _tail_alpha(dl, tl, width)
            u_tail = np.clip(4.0 * (x[mask] - center) / width * tl, -700, 700)
            f = alpha * np.exp(u_tail)
            frac[mask] = f
            var_shape[mask] = f * (1.0 - f)

    if tail_type in (TailType.RIGHT, TailType.BOTH):
        dr = delta_r
        tr = tau_r
        right_edge = center + dr
        mask = x > right_edge
        if np.any(mask):
            alpha = _tail_alpha(dr, tr, width)
            u_tail = np.clip(-4.0 * (x[mask] - center) / width * tr, -700, 700)
            f_r = alpha * np.exp(u_tail)
            frac[mask] = 1.0 - f_r
            var_shape[mask] = f_r * (1.0 - f_r)

    if tail_type == TailType.MIRROR:
        dr, tr = delta_l, tau_l  # shared value for both tails -- see docstring above
        right_edge = center + dr
        mask = x > right_edge
        if np.any(mask):
            alpha = _tail_alpha(dr, tr, width)
            u_tail = np.clip(-4.0 * (x[mask] - center) / width * tr, -700, 700)
            f_r = alpha * np.exp(u_tail)
            frac[mask] = 1.0 - f_r
            var_shape[mask] = f_r * (1.0 - f_r)

    return frac, var_shape


def mean_cline(distance: np.ndarray, center: float, width: float, mu_l: float, mu_r: float,
                tail_type: TailType = TailType.NONE,
                delta_l: float | None = None, tau_l: float | None = None,
                delta_r: float | None = None, tau_r: float | None = None
                ) -> np.ndarray:
    """The mean-cline value at raw distance(s) -- `mu_l`/`mu_r` play the
    role of HZAR's `muL`/`muR` (the trait value at the two geographic
    extremes). See `_cline_shape`'s docstring for the exact formula in
    each region. The transect-free counterpart of `predicted_mean`, for
    callers (e.g. plotting) that want the curve at arbitrary positions
    rather than at a transect's own observed sites."""
    frac, _ = _cline_shape(distance, center, width, tail_type, delta_l, tau_l, delta_r, tau_r)
    return mu_l + (mu_r - mu_l) * frac


def predicted_mean(transect: TraitTransect, params: ClineParams,
                    tail_type: TailType = TailType.NONE) -> np.ndarray:
    """The mean-cline value at every site in `transect` -- `params.pmin`/
    `params.pmax` play the role of HZAR's `muL`/`muR` (the trait value
    at the two geographic extremes)."""
    return mean_cline(transect.distance, params.center, params.width,
                       params.pmin, params.pmax, tail_type,
                       params.delta_l, params.tau_l, params.delta_r, params.tau_r)


def variance_cline(distance: np.ndarray, center: float, width: float,
                    var_l: float, var_r: float, var_h: float,
                    tail_type: TailType = TailType.NONE,
                    delta_l: float | None = None, tau_l: float | None = None,
                    delta_r: float | None = None, tau_r: float | None = None
                    ) -> np.ndarray:
    """HZAR's variance-cline function -- the model variance at each
    position. See `_cline_shape`'s docstring for the exact formula in
    each region; `var(x) = var_l + (var_r - var_l)*frac(x) +
    4*var_h*var_shape(x)`."""
    frac, var_shape = _cline_shape(distance, center, width, tail_type,
                                    delta_l, tau_l, delta_r, tau_r)
    return var_l + (var_r - var_l) * frac + 4.0 * var_h * var_shape


def log_likelihood(transect: TraitTransect, mean_params: ClineParams,
                    var_l: float, var_r: float, var_h: float,
                    tail_type: TailType = TailType.NONE) -> float:
    """Full HZAR-parity Gaussian log-likelihood, using each site's
    observed sample mean AND sample variance as sufficient statistics --
    verified directly against HZAR's own R source
    (`guassianThetaLLExpF` in `15-modelExpressionFunctions.R`):

        LL = -sum( n * (log(2*pi*var(x)) + (sampleVar + (sampleMean - mu(x))^2) / var(x)) ) / 2

    Requires `transect.trait_sd` (per-site SD of individual
    measurements) -- raises ValueError if not set, since this sufficient
    statistic is required data for this likelihood, not optional. There
    is no "single shared variance" fallback in this module -- HZAR's own
    model always uses per-site sample variance as data (see the module
    docstring for the history of why an earlier, simpler design here was
    removed).
    """
    if transect.trait_sd is None:
        raise ValueError(
            "log_likelihood requires transect.trait_sd (per-site SD of individual "
            "measurements) -- HZAR's own quantitative-trait likelihood always uses each "
            "site's observed sample variance as data, not just its mean."
        )
    mu = predicted_mean(transect, mean_params, tail_type)
    sample_var = transect.trait_sd ** 2
    var = variance_cline(
        transect.distance, mean_params.center, mean_params.width, var_l, var_r, var_h,
        tail_type=tail_type, delta_l=mean_params.delta_l, tau_l=mean_params.tau_l,
        delta_r=mean_params.delta_r, tau_r=mean_params.tau_r,
    )
    n = transect.n
    ll = -0.5 * n * (np.log(2 * np.pi * var) + (sample_var + (transect.trait_mean - mu) ** 2) / var)
    return float(np.sum(ll))


def negative_log_likelihood(param_vec: np.ndarray, transect: TraitTransect,
                             tail_type: TailType) -> float:
    """Objective function for the optimizer (minimizes -log-likelihood).

    `param_vec` layout: the `ClineParams` fields (matching
    `ClineParams.from_array`'s layout: center, width, pmin(=muL),
    pmax(=muR), then tail deltas/taus if `tail_type != NONE`), followed
    by three trailing entries: `var_l`, `var_r`, `var_h`.
    """
    mean_params = ClineParams.from_array(param_vec[:-3], tail_type)
    var_l, var_r, var_h = param_vec[-3], param_vec[-2], param_vec[-1]

    if mean_params.width <= 0 or var_l <= 0 or var_r <= 0 or var_h < 0:
        return np.inf
    for name in ("delta_l", "delta_r"):
        val = getattr(mean_params, name, None)
        if val is not None and val <= 0:
            return np.inf
    for name in ("tau_l", "tau_r"):
        val = getattr(mean_params, name, None)
        if val is not None and val <= 0:
            return np.inf

    return -log_likelihood(transect, mean_params, var_l, var_r, var_h, tail_type)
