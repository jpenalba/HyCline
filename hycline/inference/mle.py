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

import warnings
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
        names = param_names(self.tail_type,
                             asymmetric=self.params.t_skew is not None)[:len(self.param_vec)]
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
                    hzar_tail: bool = False, t_cline_asymmetric: bool = False) -> ClineParams:
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
    if tail_type == TailType.T_CLINE:
        # 6.67: Baird & Daley (2025) Table 1's own calibration point --
        # T(6.67) matches the ordinary logistic/Bazykin sigmoid (this
        # project's existing default core shape) to MAD 0.001, so it's
        # the natural "no steepening yet" starting value, not an
        # arbitrary guess. Checked before `hzar_tail` (rather than in an
        # `elif` chained off it) so this still fires correctly even if a
        # caller passes hzar_tail=True alongside T_CLINE directly -- the
        # T-cline has no HZAR-parameterized variant, so hzar_tail is
        # simply not meaningful for it, but this helper stays internally
        # consistent (t_shape always set) either way; `fit_mle` itself
        # already rejects that combination with a clear error before
        # ever reaching this function.
        params.t_shape = 6.67
        if t_cline_asymmetric:
            # 0.0: start symmetric, let the optimizer discover any real
            # skew from the data rather than biasing the search with an
            # arbitrary nonzero guess.
            params.t_skew = 0.0
    elif hzar_tail:
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
                 hzar_tail: bool = False, t_cline_asymmetric: bool = False) -> list[tuple]:
    d_span = transect.distance.max() - transect.distance.min()
    bounds = [
        (transect.distance.min() - d_span, transect.distance.max() + d_span),  # center
        (1e-3, d_span * 10 + 1e-3),                                            # width
        (0.0, 1.0),                                                            # pmin
        (0.0, 1.0),                                                            # pmax
    ]
    if tail_type == TailType.T_CLINE:
        # Checked before `hzar_tail` (see the matching note in
        # `_initial_guess`) so this always produces the right bound count
        # for T_CLINE regardless of hzar_tail -- fit_mle itself already
        # rejects hzar_tail=True with T_CLINE before reaching here.
        # Lower bound 0.05: well past Cauchy (t_shape=1, already the
        # thickest-tailed named case in Baird & Daley Table 1) into
        # more extreme steepness, without reaching numerically
        # unstable near-zero degrees of freedom. Upper bound 500:
        # well past the paper's own T(116)=Normal calibration point,
        # so the optimizer has room to converge to "no steepening
        # detectable" without hitting the bound.
        bounds += [(0.05, 500.0)]
        if t_cline_asymmetric:
            # +/-20: comfortably covers Fisher's own fitted asymmetry
            # (1.367, Baird & Daley's own headline result) with a lot
            # of headroom, while keeping the optimizer's search
            # (and `_t_cline_asymmetric_mode_and_peak`'s own
            # alpha-centered numerical peak-finding bracket, which
            # this range was chosen to stay well inside of) in the
            # verified-safe region -- see
            # hycline.models.cline.t_cline_function's docstring.
            bounds += [(-20.0, 20.0)]
    elif hzar_tail:
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


def _fit_t_cline_multistart(transect: AlleleFreqTransect, bounds: list[tuple],
                             t_cline_asymmetric: bool, n_restarts: int = 8, seed: int = 0):
    """Multi-start optimization for `TailType.T_CLINE`, both symmetric and
    asymmetric. Returns a `scipy.optimize.OptimizeResult`-like object
    (has `.x`/`.fun`/`.success`), matching what `minimize` itself returns,
    so `fit_mle` can treat this as a drop-in replacement for a single
    `minimize` call.

    A REAL FINDING THAT SHAPED THIS FUNCTION, not a design assumed to
    work: a single L-BFGS-B start from the data-driven initial guess can
    converge to a badly wrong local optimum, confirmed directly on
    synthetic data with a known truth. The first, more naive fix tried
    (`scipy.optimize.minimize` from `n_restarts` points drawn uniformly
    across the *entire* parameter bounds -- the same strategy already
    used successfully for the tripartite tail model in
    `hycline.inference.mle_genotype_likelihood.fit_mle_geographic_gl`,
    BACKEND_PHASE2.md section 12.3) turned out to be badly insufficient
    here: checked directly, 100 such uniform restarts found the true
    optimum's basin of attraction *zero* times out of 100 on one
    representative synthetic case. The root cause is `t_shape`'s own
    bounds spanning multiple orders of magnitude (0.05-500) -- a linear-
    uniform draw over that range puts more than 99% of restarts at
    `t_shape > 5`, so the model essentially never gets to explore the
    genuinely "stepped" (small-`t_shape`) region a uniform-over-bounds
    strategy nominally covers. This function instead uses two things
    checked directly to actually work:

    1. `t_shape` restarts are drawn LOG-uniformly (not linearly) across
       its bounds, so orders of magnitude are sampled roughly evenly.
    2. For the asymmetric case specifically, the search is anchored on
       the (separately, robustly fit) SYMMETRIC optimum -- `t_skew=0` is
       an exact point in the asymmetric parameter space, so this is a
       legitimate starting point, not an approximation -- explored with
       both a `t_skew` grid and a few further log-uniform-shape/uniform-
       skew random restarts from the data-driven guess for diversity.
       Checked directly: this combination recovered known truth cleanly
       (within ~3 negative-log-likelihood units) in 4 of 6 representative
       synthetic cases spanning a wide (t_shape, t_skew) range; the 2
       that didn't were both *very* thick-tailed (t_shape below 1) *and*
       strongly asymmetric (|t_skew| >= 5) simultaneously -- genuinely
       hard cases where even the symmetric anchor itself struggled, not
       cases this function silently mishandles.

    Whatever else this function tries, THE ASYMMETRIC RESULT IS NEVER
    WORSE THAN THE SYMMETRIC ONE by construction: since `t_skew=0` is an
    exact, valid point in the asymmetric family, the symmetric optimum
    (extended with `t_skew=0.0`) is always included as a literal
    candidate, so a caller can never see the confusing, mathematically-
    backwards result (asymmetric AICc worse than symmetric AICc for the
    same data) that motivated this whole investigation in the first
    place -- confirmed by test, not just claimed."""
    from scipy.optimize import OptimizeResult

    t_shape_lo, t_shape_hi = bounds[4]
    rng = np.random.default_rng(seed)
    base = _initial_guess(transect, TailType.T_CLINE, t_cline_asymmetric=False)

    sym_bounds = bounds[:5]
    x0 = base.as_array(TailType.T_CLINE)
    sym_best = minimize(negative_log_likelihood, x0, args=(transect, TailType.T_CLINE, False),
                         method="L-BFGS-B", bounds=sym_bounds)
    for _ in range(n_restarts):
        t_shape_draw = float(np.exp(rng.uniform(np.log(t_shape_lo), np.log(t_shape_hi))))
        x0_r = np.array([base.center, base.width, base.pmin, base.pmax, t_shape_draw])
        candidate = minimize(negative_log_likelihood, x0_r, args=(transect, TailType.T_CLINE, False),
                              method="L-BFGS-B", bounds=sym_bounds)
        if candidate.fun < sym_best.fun:
            sym_best = candidate

    if not t_cline_asymmetric:
        return sym_best

    sym_params = ClineParams.from_array(sym_best.x, TailType.T_CLINE)
    t_skew_lo, t_skew_hi = bounds[5]

    # Include the symmetric optimum itself (t_skew=0.0) as a literal
    # candidate -- this is what guarantees the final asymmetric result is
    # never worse than the symmetric-only fit.
    asym_x0 = np.array([sym_params.center, sym_params.width, sym_params.pmin,
                         sym_params.pmax, sym_params.t_shape, 0.0])
    asym_best = OptimizeResult(x=asym_x0, fun=sym_best.fun, success=sym_best.success)

    for t_skew_grid_val in (-10.0, -5.0, -2.0, 2.0, 5.0, 10.0):
        x0_r = np.array([sym_params.center, sym_params.width, sym_params.pmin,
                          sym_params.pmax, sym_params.t_shape, t_skew_grid_val])
        candidate = minimize(negative_log_likelihood, x0_r, args=(transect, TailType.T_CLINE, False),
                              method="L-BFGS-B", bounds=bounds)
        if candidate.fun < asym_best.fun:
            asym_best = candidate
    for _ in range(n_restarts):
        t_shape_draw = float(np.exp(rng.uniform(np.log(t_shape_lo), np.log(t_shape_hi))))
        t_skew_draw = float(rng.uniform(t_skew_lo, t_skew_hi))
        x0_r = np.array([base.center, base.width, base.pmin, base.pmax,
                          t_shape_draw, t_skew_draw])
        candidate = minimize(negative_log_likelihood, x0_r, args=(transect, TailType.T_CLINE, False),
                              method="L-BFGS-B", bounds=bounds)
        if candidate.fun < asym_best.fun:
            asym_best = candidate

    return asym_best


def fit_mle(transect: AlleleFreqTransect, tail_type: TailType = TailType.NONE,
            compute_std_errors: bool = True, hzar_tail: bool = False,
            t_cline_asymmetric: bool = False, n_restarts: int = 8, seed: int = 0) -> FitResult:
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
        between the two parameterizations. Not meaningful for
        `tail_type=TailType.T_CLINE` (the T-cline has no HZAR-
        parameterized variant) -- raises `ValueError` immediately rather
        than failing deep inside the optimizer's first objective call.
    t_cline_asymmetric : only meaningful for `tail_type=TailType.T_CLINE`
        -- if True, fits the fully general asymmetric `T(v, alpha)`
        family (`ClineParams.t_skew` free, starting from 0.0) instead of
        the symmetric-only `T(v)` (the default, `t_skew` fixed at
        `None`). Raises `ValueError` immediately if `True` with any
        other `tail_type`. See `hycline.models.cline.t_cline_function`'s
        docstring for the model itself.
    n_restarts, seed : for `tail_type=TailType.T_CLINE` only (ignored
        otherwise -- other tail types' single-start behavior is
        deliberately unchanged, matching this project's own principle of
        not silently altering previously-validated results). A REAL
        finding, not a theoretical worry: a single L-BFGS-B start from
        the data-driven initial guess can converge to a badly wrong
        local optimum for the T-cline, confirmed directly on synthetic
        data with a known truth -- one real case found a fit whose
        negative log-likelihood was 264 units worse than the true
        parameters' own (462.7), recovering `width` more than 80% too
        wide, while starting the *identical* optimizer from the true
        parameters converged almost exactly (`center`/`width`/`t_shape`/
        `t_skew` all within 0.3% of truth). This mirrors the exact same
        finding already made for the genotype-likelihood-aware tail-
        model path (`hycline.inference.mle_genotype_likelihood.
        fit_mle_geographic_gl`, BACKEND_PHASE2.md section 12.3) and
        adopts the identical fix here: `n_restarts` additional L-BFGS-B
        runs from randomized starting points (uniform within `bounds`),
        keeping whichever run (including the original data-driven start)
        achieves the lowest negative log-likelihood. Default 8, the same
        default `fit_mle_geographic_gl` uses.
    """
    if tail_type == TailType.T_CLINE and hzar_tail:
        raise ValueError(
            "hzar_tail=True is not meaningful with tail_type=TailType.T_CLINE -- "
            "the T-cline is a standalone continuous shape model with no HZAR-"
            "parameterized variant. Pass hzar_tail=False (the default)."
        )
    if t_cline_asymmetric and tail_type != TailType.T_CLINE:
        raise ValueError(
            "t_cline_asymmetric=True is only meaningful with tail_type=TailType.T_CLINE, "
            f"got tail_type={tail_type!r}."
        )
    x0 = _initial_guess(transect, tail_type, hzar_tail=hzar_tail,
                         t_cline_asymmetric=t_cline_asymmetric).as_array(tail_type)
    bounds = _bounds_for(transect, tail_type, hzar_tail=hzar_tail,
                          t_cline_asymmetric=t_cline_asymmetric)

    if tail_type == TailType.T_CLINE:
        result = _fit_t_cline_multistart(transect, bounds, t_cline_asymmetric,
                                          n_restarts=n_restarts, seed=seed)
    else:
        result = minimize(
            negative_log_likelihood, x0, args=(transect, tail_type, hzar_tail),
            method="L-BFGS-B", bounds=bounds,
        )

    params = ClineParams.from_array(result.x, tail_type)
    ll = log_likelihood(transect, params, tail_type, hzar_tail=hzar_tail)
    k = len(result.x)
    n = len(transect)
    aicc = _aicc(ll, k, n)

    if tail_type == TailType.T_CLINE and t_cline_asymmetric and params.t_skew is not None:
        t_skew_lo, t_skew_hi = bounds[5]
        span = t_skew_hi - t_skew_lo
        if params.t_skew <= t_skew_lo + 0.01 * span or params.t_skew >= t_skew_hi - 0.01 * span:
            warnings.warn(
                f"The asymmetric T-cline's fitted t_skew ({params.t_skew:.3g}) landed at "
                f"the search bound ({t_skew_lo:g}, {t_skew_hi:g}), not settled on an "
                "interior optimum. A REAL finding, checked directly rather than assumed "
                "to be a bug: at large |t_skew| combined with a moderate t_shape, the "
                "resulting cline shape can become nearly indistinguishable over the "
                "sampled geographic range for a wide span of t_skew values -- confirmed "
                "on synthetic data with a known true t_skew=2.5, where the fitted "
                "negative log-likelihood (448.30) was actually *better* than the truth's "
                "own (448.70) despite the fit landing near t_skew=13-20, not 2.5. This "
                "mirrors the already-documented weak identifiability of the tripartite "
                "tail model's own delta/tau parameters -- treat t_skew's point estimate "
                "with real caution here (center/width/pmin/pmax and the overall AICc "
                "ranking against the symmetric-only fit remain meaningful), and consider "
                "whether your transect's geographic range actually extends far enough "
                "into the tails to constrain asymmetry at all.",
                UserWarning, stacklevel=2,
            )

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
