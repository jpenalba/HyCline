"""
Core geographic cline model.

Implements the classical sigmoid cline of Szymura & Barton (1986, 1991)

    p(x) = pmin + (pmax - pmin) / (1 + exp(-4 * (x - c) / w))

where `c` is the cline center and `w` is the cline width (defined, following
the original parameterization, as the inverse of the maximum slope -- hence
the factor of 4 in the exponent).

An optional exponential-decay tail extension (Barton & Gale 1993) is also
provided for clines that do not plateau cleanly outside the sigmoid core.

NOTE ON THE TAIL MODEL
-----------------------
`cline_function`'s tail parameterization (below) follows a general form
described in the hybrid-zone cline literature (a continuous exponential
decay grafted onto the sigmoid core at x = c +/- w/2) -- this is NOT
HZAR's own tail parameterization, and this function's default behavior
is deliberately left unchanged to avoid silently altering any previously
-fitted results. `hzar_cline_function` (below) is the actual,
bit-verified-against-HZAR's-R-source alternative -- see its own
docstring, and BACKEND_TRAIT.md sections 1-2 for the full derivation and
verification (built first for quantitative trait clines, where the
variance model made checking against the real source unavoidable, and
ported here once verified). Use `hzar_cline_function` when exact HZAR
parity matters; `cline_function` remains the default for `fit`/
`fit-multi` and everywhere else in this project that doesn't opt in.

NOTE ON THE T-CLINE MODEL
--------------------------
`t_cline_function` (below) is a third, structurally different cline
model: Baird & Daley (2025)'s Student's-t-distribution-based family,
continuous everywhere (no corners), addressing a real, documented
convergence difficulty the tripartite tail models above have under
gradient-based (NUTS) Bayesian fitting -- see BACKEND_PHASE2.md
section 12. Not a tail bolted onto the logistic core the way
`TailType.LEFT`/`RIGHT`/`BOTH`/`MIRROR` are; see `t_cline_function`'s
own docstring for the full model, including both the symmetric case
(`params.t_skew` unset) and the asymmetric case (`params.t_skew` set)
now supported for classic MLE fitting.

NOTE ON WHY THE T-CLINE HAS NO BAYESIAN (NUTS) SUPPORT, EITHER SHAPE
---------------------------------------------------------------------
Not a todo-list gap -- checked directly and found genuinely infeasible
with the JAX/NumPyro versions available at the time of writing (jax
0.10.2), for a real, specific, verified reason: `jax.scipy.special.
betainc`, which both a hand-built JAX Student's-t CDF and NumPyro's own
official `numpyro.distributions.StudentT.cdf()` are built on, explicitly
does not support gradients with respect to its own shape parameters
(confirmed directly against NumPyro's own code, not just a homemade
implementation: `jax.grad` on `StudentT(df=...).cdf()` with respect to
`df` raises `ValueError: Betainc gradient with respect to a and b not
supported`). Since `t_shape` enters exactly there, and NUTS needs the
gradient of the log-density with respect to every parameter it samples
(`t_shape` included, for a Bayesian T-cline fit), this is a real
capability gap in the JAX/NumPyro ecosystem itself, not something this
project's own code could close by porting more of Clinalyse -- porting
the reference implementation's own closed-form asymmetric peak-location
formula (built from Hermite/confluent-hypergeometric functions) doesn't
help either, since it has its own, separate numerical problem in JAX
(`jax.scipy.special.hyp1f1` diverges from `scipy.special.hyp1f1` by a
factor of ~150,000 at `t_shape=500`, well within this project's own
existing bounds) -- and is moot regardless, since the CDF-gradient
problem above blocks Bayesian fitting before that formula would even be
reached. `--tail t-cline --method bayesian` fails immediately with a
clear CLI error rather than attempting this and failing deep inside
NumPyro. See README.md's Tier 1 backlog for the full investigation.

A narrower alternative exists if Bayesian T-cline credible intervals are
ever wanted badly enough: fix `t_shape` (and `t_skew`, if asymmetric) at
its classic-MLE-estimated value -- a constant, not a parameter NUTS
samples -- and let NUTS explore only `center`/`width`/`pmin`/`pmax`.
This sidesteps the broken gradient entirely, at the cost of not
propagating shape-parameter uncertainty into the reported credible
intervals. Not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats as _stats


class TailType(str, Enum):
    NONE = "none"        # pure sigmoid, no tails
    MIRROR = "mirror"     # symmetric exponential tails on both sides
    LEFT = "left"         # exponential tail on the left only
    RIGHT = "right"       # exponential tail on the right only
    BOTH = "both"         # independent exponential tails on both sides
    T_CLINE = "t-cline"   # Student's-t-based continuous stepped-cline alternative
                           # (Baird & Daley 2025) -- see `t_cline_function` below.
                           # NOT a tail bolted onto the logistic core the way the
                           # other four members are: a standalone continuous shape
                           # model. `cline_function`/`hzar_cline_function` both
                           # reject it explicitly rather than mishandle it.


@dataclass
class ClineParams:
    """Parameters for a single cline fit.

    Attributes
    ----------
    center : cline center (c), in the same distance units as the transect
    width : cline width (w); must be > 0
    pmin : frequency/trait value far to the left of the cline
    pmax : frequency/trait value far to the right of the cline
    delta_l, tau_l : left-tail height fraction and decay length (TailType.LEFT/BOTH/MIRROR)
    delta_r, tau_r : right-tail height fraction and decay length (TailType.RIGHT/BOTH)
    t_shape : T-cline shape/steepness parameter (TailType.T_CLINE only -- the
        Student's-t distribution's degrees of freedom, matching Baird & Daley
        (2025)'s own `v`). Must be > 0.
    t_skew : T-cline asymmetry/noncentrality parameter (TailType.T_CLINE
        only, optional -- matching Baird & Daley (2025)'s own `alpha`).
        `None` or `0.0` means the symmetric T-cline (the paper's `T(v)`);
        any other value means the asymmetric `T(v, alpha)` family. See
        `t_cline_function`'s docstring for both cases.
    """

    center: float
    width: float
    pmin: float = 0.0
    pmax: float = 1.0
    delta_l: Optional[float] = None
    tau_l: Optional[float] = None
    delta_r: Optional[float] = None
    tau_r: Optional[float] = None
    t_shape: Optional[float] = None
    t_skew: Optional[float] = None

    def as_array(self, tail_type: TailType) -> np.ndarray:
        """Flatten to a parameter vector for the optimizer, in a fixed
        order determined by the tail type. For TailType.T_CLINE,
        `t_skew` is included only when set (not None) -- whether this
        instance represents a symmetric or asymmetric T-cline is decided
        by `self.t_skew`, not by a separate flag, so constructing a
        `ClineParams` with `t_skew=0.0` explicitly (rather than leaving
        it `None`) opts into the asymmetric parameter count even though
        0.0 is itself symmetric."""
        vec = [self.center, self.width, self.pmin, self.pmax]
        if tail_type in (TailType.LEFT, TailType.BOTH):
            vec += [self.delta_l, self.tau_l]
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            vec += [self.delta_r, self.tau_r]
        if tail_type == TailType.MIRROR:
            vec += [self.delta_l, self.tau_l]
        if tail_type == TailType.T_CLINE:
            vec += [self.t_shape]
            if self.t_skew is not None:
                vec += [self.t_skew]
        return np.array(vec, dtype=float)

    @staticmethod
    def from_array(vec: np.ndarray, tail_type: TailType) -> "ClineParams":
        """For TailType.T_CLINE, whether `vec` carries `t_skew` is
        inferred from its length (5 elements total = symmetric, 6 =
        asymmetric) -- the caller doesn't need to pass a separate flag,
        matching `as_array`'s own symmetric convention."""
        center, width, pmin, pmax = vec[:4]
        rest = vec[4:]
        kwargs = dict(center=center, width=width, pmin=pmin, pmax=pmax)
        if tail_type == TailType.LEFT:
            kwargs["delta_l"], kwargs["tau_l"] = rest[0], rest[1]
        elif tail_type == TailType.RIGHT:
            kwargs["delta_r"], kwargs["tau_r"] = rest[0], rest[1]
        elif tail_type == TailType.BOTH:
            kwargs["delta_l"], kwargs["tau_l"] = rest[0], rest[1]
            kwargs["delta_r"], kwargs["tau_r"] = rest[2], rest[3]
        elif tail_type == TailType.MIRROR:
            kwargs["delta_l"], kwargs["tau_l"] = rest[0], rest[1]
            kwargs["delta_r"], kwargs["tau_r"] = rest[0], rest[1]
        elif tail_type == TailType.T_CLINE:
            kwargs["t_shape"] = rest[0]
            if len(rest) > 1:
                kwargs["t_skew"] = rest[1]
        return ClineParams(**kwargs)

    def n_params(self, tail_type: TailType) -> int:
        return len(self.as_array(tail_type))


def _sigmoid_core(x: np.ndarray, center: float, width: float,
                   pmin: float, pmax: float) -> np.ndarray:
    z = -4.0 * (x - center) / width
    # clip to avoid overflow in exp() for points far from the cline
    z = np.clip(z, -700, 700)
    return pmin + (pmax - pmin) / (1.0 + np.exp(z))


def cline_function(x: np.ndarray, params: ClineParams,
                    tail_type: TailType = TailType.NONE) -> np.ndarray:
    """Evaluate the cline model at geographic position(s) `x`.

    Parameters
    ----------
    x : array-like of geographic distances along the transect
    params : ClineParams
    tail_type : which tail variant to apply

    Returns
    -------
    np.ndarray of predicted frequencies/trait values at each x, same shape as x.
    """
    if tail_type == TailType.T_CLINE:
        raise ValueError(
            "cline_function does not handle TailType.T_CLINE -- the T-cline is a "
            "standalone continuous shape model, not a tail added to the logistic "
            "core the way NONE/LEFT/RIGHT/BOTH/MIRROR are. Use t_cline_function "
            "instead."
        )

    x = np.asarray(x, dtype=float)
    core = _sigmoid_core(x, params.center, params.width, params.pmin, params.pmax)

    if tail_type == TailType.NONE:
        return core

    left_edge = params.center - params.width / 2.0
    right_edge = params.center + params.width / 2.0
    out = core.copy()

    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        mask = x < left_edge
        if np.any(mask):
            out[mask] = params.pmin + (params.pmax - params.pmin) * params.delta_l * np.exp(
                (x[mask] - left_edge) / params.tau_l
            )

    if tail_type in (TailType.RIGHT, TailType.BOTH, TailType.MIRROR):
        mask = x > right_edge
        if np.any(mask):
            dr = params.delta_r if tail_type != TailType.MIRROR else params.delta_l
            tr = params.tau_r if tail_type != TailType.MIRROR else params.tau_l
            out[mask] = params.pmax - (params.pmax - params.pmin) * dr * np.exp(
                -(x[mask] - right_edge) / tr
            )

    return out


def _hzar_tail_alpha(delta: float, tau: float, width: float) -> float:
    """HZAR's own `alphaExpF`, verified against HZAR's R source
    (`15-modelExpressionFunctions.R`) while building
    `hycline.likelihoods.gaussian`'s trait-cline tail model -- the
    tail-region scaling constant chosen so the tail formula is
    continuous with the sigmoid core at the tail edge (`center +/-
    delta`). See `hzar_cline_function`'s docstring for the full formula
    this feeds into."""
    logit_delta = 4.0 * delta / width
    logit_delta = np.clip(logit_delta, -700, 700)
    return np.exp(logit_delta * (tau - 1.0)) / (1.0 + np.exp(-logit_delta))


def hzar_cline_function(x: np.ndarray, params: ClineParams,
                         tail_type: TailType = TailType.NONE) -> np.ndarray:
    """Evaluate the cline model at geographic position(s) `x`, using
    HZAR's own tail parameterization -- verified directly against
    HZAR's actual R source (`pTailExpF`/`qTailExpF` in
    `15-modelExpressionFunctions.R`, https://github.com/cran/hzar), not
    `cline_function`'s own (differently parameterized, not HZAR-
    verified) tail model. First built and verified for quantitative
    trait clines (`hycline.likelihoods.gaussian`, BACKEND_TRAIT.md
    sections 1-2, including a direct check that the tail formula is
    continuous with the sigmoid core at the tail edge); ported here
    unchanged once verified, for allele-frequency fitting that wants
    exact HZAR parity.

    A REAL, NAMED DIFFERENCE FROM `cline_function`'s OWN TAIL MODEL,
    stated plainly rather than left as an implicit trap: HZAR's tail
    edges are `center - delta_l` / `center + delta_r` (the delta
    parameters ARE the raw distance from center to the tail edge), not
    `center - width/2` / `center + width/2` the way `cline_function`
    defines them. `delta_l`/`tau_l`/`delta_r`/`tau_r` mean something
    genuinely different between these two functions, even though both
    reuse the same `ClineParams` container -- fitted tail parameters
    from one are not directly comparable to, or interchangeable with,
    the other's.

    For `TailType.NONE`, this is identical to `cline_function` (both
    reproduce the same, unambiguous sigmoid core -- see
    `_sigmoid_core`).
    """
    if tail_type == TailType.T_CLINE:
        raise ValueError(
            "hzar_cline_function does not handle TailType.T_CLINE -- the T-cline "
            "has no HZAR-parameterized variant. Use t_cline_function instead."
        )

    x = np.asarray(x, dtype=float)
    core = _sigmoid_core(x, params.center, params.width, params.pmin, params.pmax)

    if tail_type == TailType.NONE:
        return core

    out = core.copy()

    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        dl = params.delta_l
        tl = params.tau_l
        left_edge = params.center - dl
        mask = x < left_edge
        if np.any(mask):
            alpha = _hzar_tail_alpha(dl, tl, params.width)
            u = np.clip(4.0 * (x[mask] - params.center) / params.width * tl, -700, 700)
            frac = alpha * np.exp(u)
            out[mask] = params.pmin + (params.pmax - params.pmin) * frac

    if tail_type in (TailType.RIGHT, TailType.BOTH):
        dr = params.delta_r
        tr = params.tau_r
        right_edge = params.center + dr
        mask = x > right_edge
        if np.any(mask):
            alpha = _hzar_tail_alpha(dr, tr, params.width)
            u = np.clip(-4.0 * (x[mask] - params.center) / params.width * tr, -700, 700)
            frac_r = alpha * np.exp(u)
            out[mask] = params.pmin + (params.pmax - params.pmin) * (1.0 - frac_r)

    if tail_type == TailType.MIRROR:
        dr, tr = params.delta_l, params.tau_l  # shared value for both tails, matching
        right_edge = params.center + dr        # cline_function's own MIRROR convention
        mask = x > right_edge
        if np.any(mask):
            alpha = _hzar_tail_alpha(dr, tr, params.width)
            u = np.clip(-4.0 * (x[mask] - params.center) / params.width * tr, -700, 700)
            frac_r = alpha * np.exp(u)
            out[mask] = params.pmin + (params.pmax - params.pmin) * (1.0 - frac_r)

    return out


def _t_cline_scale(t_shape: float) -> float:
    """Peak density of the standard (zero-centered, unit-scale) Student's-t
    PDF with `t_shape` degrees of freedom -- the factor that converts a
    width-normalized x into the standard Student's-t CDF's own native
    argument, chosen so the resulting cline's maximum gradient is exactly
    1/width (this project's existing HZAR/Barton width convention:
    "width" = inverse of the cline's maximum gradient).

    Derivation: for `p(x) = F_t((x-center)/(width*S); t_shape)` (F_t the
    standard, unscaled Student's-t CDF), the chain rule gives
    `dp/dx|x=center = f_t(0; t_shape) / (width*S)`. Setting `S = f_t(0;
    t_shape)` makes this exactly `1/width`, matching the tripartite core's
    own convention (`_sigmoid_core`'s factor of 4, chosen the same way for
    the logistic case) -- verified directly by finite-difference check in
    `tests/test_cline.py`, not just asserted.

    Cross-checked against the reference Python implementation this model
    is ported from (Baird & Daley 2023, `Clinalyse`,
    github.com/Studenecivb/clinalyse, `equations.py`): its own
    `PreparationGosset._studentcdfgrad(x) = gamma((x+1)/2) /
    (sqrt(pi*x)*gamma(x/2))` is exactly `f_t(0; x)` in closed form --
    `scipy.stats.t.pdf(0, df=t_shape)` computes the same quantity via
    scipy's own numerically robust Student's-t implementation instead of
    a hand-evaluated Gamma-function ratio (safer for large `t_shape`,
    where `gamma()` alone can overflow before the ratio is taken)."""
    return float(_stats.t.pdf(0.0, df=t_shape))


def _t_cline_asymmetric_mode_and_peak(t_shape: float, t_skew: float) -> tuple[float, float]:
    """Generalizes `_t_cline_scale` to the asymmetric case: the location
    and magnitude of the peak density of the noncentral Student's-t
    distribution (`scipy.stats.nct`) with `t_shape` degrees of freedom
    and noncentrality `t_skew`. Unlike the symmetric case, the point of
    maximum density (and therefore maximum CDF gradient -- this
    project's own cline-center convention) is not at 0 once `t_skew !=
    0`, so both a location correction (the mode) and a magnitude
    correction (the peak density there) are needed, not just the
    magnitude alone.

    WHY NUMERICAL PEAK-FINDING, NOT A CLOSED-FORM FORMULA: the reference
    implementation this project ports from (Clinalyse,
    `PreparationGosset.preparation_gosset_asy`) computes this same
    quantity via a closed-form formula built from Hermite/confluent-
    hypergeometric functions. Checked directly, not assumed: that
    approach is numerically unreliable at exactly the `t_shape` range
    this project already supports for the symmetric case
    (`jax.scipy.special.hyp1f1` diverges from `scipy.special.hyp1f1` by a
    factor of ~150,000 at `t_shape=500`) -- and unnecessary anyway for
    classic MLE specifically, since `fit_mle` never needed
    JAX-differentiable code in the first place (`scipy.optimize.minimize`
    is called without a `jac=` argument, so L-BFGS-B already uses
    finite-difference gradients on the whole objective; nothing inside
    this function needs to be analytically differentiable). A plain
    numerical peak-finder is simpler, avoids that fragility entirely, and
    was verified clean (no NaN/overflow, ~0.8ms/call) across 48
    `(t_shape, t_skew)` combinations spanning this project's own existing
    bounds before being adopted -- see README.md's Tier 1 backlog for the
    full investigation.

    The search bracket is centered on `t_skew` (rather than a fixed wide
    window like (-50, 50)) deliberately: `scipy.stats.nct.pdf` itself can
    overflow deep in its own extreme tail at large `t_shape` (confirmed
    directly, not assumed) -- a symptom of searching somewhere
    numerically meaningless, not of the distribution being unusable. The
    true mode is always close to `t_skew` in practice (confirmed
    empirically across the same 48 combinations), so a +/-20 window
    around it comfortably contains the mode while never approaching that
    failure region.
    """
    from scipy.optimize import minimize_scalar
    lo, hi = t_skew - 20.0, t_skew + 20.0
    res = minimize_scalar(
        lambda t: -_stats.nct.pdf(t, df=t_shape, nc=t_skew),
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-10},
    )
    mode = float(res.x)
    peak = float(_stats.nct.pdf(mode, df=t_shape, nc=t_skew))
    return mode, peak


def t_cline_function(x: np.ndarray, params: ClineParams) -> np.ndarray:
    """T-cline: Baird & Daley (2025)'s Student's-t-distribution-based
    alternative to the tripartite tail model above (`cline_function`/
    `hzar_cline_function` with a tail type other than NONE) -- continuous
    everywhere, with no corners at all, addressing the real, documented
    NUTS convergence difficulty those two functions' tail models have
    (see this module's own top-level docstring and BACKEND_PHASE2.md
    section 12: "the tail model is continuous... but not smooth... a kink
    is a real obstacle" for gradient-based samplers). Baird & Daley give
    the underlying structural reason: the tripartite model's sigmoid-core-
    meets-exponential-tail joins are genuine mathematical discontinuities
    ("corners"), not a numerical inconvenience specific to any particular
    optimizer or sampler.

    `params.t_shape` (matching the paper's own `v`, the Student's-t
    distribution's degrees of freedom) slides the cline continuously
    through a family of shapes, calibrated directly against the paper's
    own Table 1:
    - `t_shape=1`: exactly the Cauchy distribution CDF -- the most
      "stepped" (thick-tailed) symmetric member of this family.
    - `t_shape=2`: Gavrilets (1997)'s single-locus viability-selection
      stepped-cline extreme.
    - `t_shape=6.67`: matches the ordinary logistic/Bazykin sigmoid cline
      (`cline_function`'s own default core shape) to within a maximum
      absolute difference of 0.001 (Table 1's own calibration) -- the
      natural "no steepening" reference point for this family.
    - `t_shape -> infinity`: the Normal distribution CDF, effectively
      indistinguishable from `t_shape=30` onward (Table 1: `t_shape=116`
      matches Normal to MAD 0.001).

    `params.t_skew` (matching the paper's own `alpha`, the non-central
    Student's-t's noncentrality) adds asymmetry -- `None` or `0.0` (the
    default) is the SYMMETRIC T-cline (`T(v)`, above); any other value
    is the fully general `T(v, alpha)` family, letting the two sides of
    the cline steepen/decay at different effective rates the way the
    tripartite tail model's separate `delta_l/tau_l` vs. `delta_r/tau_r`
    do, but with one continuous, corner-free shape rather than three
    joined pieces. `center`/`width` keep their usual meaning either way
    (`center` = the point of maximum gradient, `width` = its inverse) --
    for `t_skew != 0` this point is no longer at the distribution's own
    x=0 or its raw peak density, so both a location and a magnitude
    correction are applied (`_t_cline_asymmetric_mode_and_peak`) rather
    than just the magnitude-only correction the symmetric case needs
    (`_t_cline_scale`). Fisher's (1937) own wave-of-advance data is
    fit best by Baird & Daley's paper at `T(8.365, 1.367)` -- an
    18-fold MAD improvement over the best tripartite fit, using fewer
    free parameters, their own headline result.

    Classic MLE fitting only -- no Bayesian/NUTS support for either the
    symmetric or asymmetric case, and not a todo-list gap: see this
    module's own top-level docstring for the specific, verified reason
    (a real capability gap in the JAX/NumPyro ecosystem itself, not
    something more porting would fix).

    Ported from, and cross-checked directly against, the reference Python
    implementation (Baird & Daley 2023, `Clinalyse`,
    github.com/Studenecivb/clinalyse, `equations.py`): `gossetbar_cline`
    for the symmetric case (evaluates `scipy.stats.t.cdf(x, shape,
    scale=prep[shape])` with `prep[shape]` precomputed as exactly
    `_t_cline_scale` above computes it); the asymmetric case here departs
    from `gossetbar_cline_asy`'s own closed-form peak-location formula
    for verified numerical-robustness reasons (see
    `_t_cline_asymmetric_mode_and_peak`'s docstring) while implementing
    the same underlying model -- see `references/Molecular Ecology -
    2025 - Baird - The Shapes of Clines and Wavefronts.pdf`, Table 1's
    "T-cline T(v)" row, for the full derivation.
    """
    if params.t_shape is None or params.t_shape <= 0:
        raise ValueError("t_cline_function requires params.t_shape > 0, got "
                          f"{params.t_shape!r}")
    if params.width <= 0:
        raise ValueError(f"t_cline_function requires params.width > 0, got {params.width!r}")

    x = np.asarray(x, dtype=float)
    raw = (x - params.center) / params.width

    if params.t_skew is None or params.t_skew == 0.0:
        scale = _t_cline_scale(params.t_shape)
        cdf = _stats.t.cdf(raw, df=params.t_shape, scale=scale)
    else:
        mode, peak = _t_cline_asymmetric_mode_and_peak(params.t_shape, params.t_skew)
        raw_t = raw / peak + mode
        with np.errstate(invalid="ignore"):
            cdf = _stats.nct.cdf(raw_t, df=params.t_shape, nc=params.t_skew)
        # scipy.stats.nct.cdf can itself return NaN deep in its own
        # extreme tail (confirmed directly: e.g. t_shape=100, t_skew=10
        # gives NaN over a real, non-astronomical range of raw_t, not
        # just at +/-inf) -- a genuine numerical limitation of scipy's
        # own noncentral-t implementation at large t_shape combined with
        # substantial t_skew, not a bug in this function's own math (the
        # width-convention and t_skew=0-continuity checks in
        # tests/test_cline.py hold exactly). Left unhandled, a single NaN
        # here would silently propagate into an entire fit's log-
        # likelihood (via binom.logpmf), turning one far-tail site into a
        # completely uninformative optimizer target rather than the
        # correctly-negligible one it should be. Any point far enough
        # into the tail to trigger this is, by construction, far enough
        # that the true CDF value is already negligibly close to 0 or 1
        # (confirmed: the nearby, successfully-computed values in that
        # same region are already ~1e-16 to 1e-29, far below this
        # project's own likelihood floor `_EPS=1e-10` in
        # hycline.likelihoods.binomial) -- so substituting the correct
        # limit is safe, not just convenient.
        if np.any(np.isnan(cdf)):
            cdf = np.where(np.isnan(cdf), np.where(raw_t < mode, 0.0, 1.0), cdf)

    return params.pmin + (params.pmax - params.pmin) * cdf


# Human-readable parameter names, in the same order as ClineParams.as_array(),
# used for reporting fitted results.
def param_names(tail_type: TailType, asymmetric: bool = False) -> list[str]:
    """`asymmetric` only affects TailType.T_CLINE -- adds `t_skew` to the
    name list when True, matching `ClineParams.as_array`'s own
    `t_skew is not None` convention for whether a fit vector carries it."""
    names = ["center", "width", "pmin", "pmax"]
    if tail_type in (TailType.LEFT, TailType.BOTH):
        names += ["delta_l", "tau_l"]
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        names += ["delta_r", "tau_r"]
    if tail_type == TailType.MIRROR:
        names += ["delta_m", "tau_m"]
    if tail_type == TailType.T_CLINE:
        names += ["t_shape"]
        if asymmetric:
            names += ["t_skew"]
    return names
