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
own docstring for the full model and its current (symmetric-only)
scope.
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
        (2025)'s own `v`). Must be > 0. Symmetric-only in this implementation
        (no asymmetry parameter yet -- see `t_cline_function`'s docstring).
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

    def as_array(self, tail_type: TailType) -> np.ndarray:
        """Flatten to a parameter vector for the optimizer, in a fixed
        order determined by the tail type."""
        vec = [self.center, self.width, self.pmin, self.pmax]
        if tail_type in (TailType.LEFT, TailType.BOTH):
            vec += [self.delta_l, self.tau_l]
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            vec += [self.delta_r, self.tau_r]
        if tail_type == TailType.MIRROR:
            vec += [self.delta_l, self.tau_l]
        if tail_type == TailType.T_CLINE:
            vec += [self.t_shape]
        return np.array(vec, dtype=float)

    @staticmethod
    def from_array(vec: np.ndarray, tail_type: TailType) -> "ClineParams":
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

    A REAL, NAMED LIMITATION, stated plainly rather than left implicit:
    this implements only the SYMMETRIC T-cline (`T(v)` in the paper's own
    notation) -- the paper's fully general asymmetric `T(v, alpha)`
    family (needed to reproduce its own headline validation result on
    Fisher's 1937 wave-of-advance data, `T(8.365, 1.367)`, an 18-fold MAD
    improvement over the best tripartite fit) is NOT yet implemented here.
    A symmetric T-cline already has no corners and already offers real
    flexibility the plain logistic core doesn't (continuously variable
    steepness), but a user who specifically needs *asymmetric* barrier
    strength (different behavior on each side) should still reach for
    `hzar_cline_function`'s tail types, with their own documented
    convergence caveat, until asymmetric T-cline support lands -- see
    README.md's Tier 1 backlog.

    Ported from, and cross-checked directly against, the reference Python
    implementation for the symmetric case (Baird & Daley 2023,
    `Clinalyse`, github.com/Studenecivb/clinalyse, `equations.py`'s
    `gossetbar_cline`, which evaluates `scipy.stats.t.cdf(x, shape,
    scale=prep[shape])` with `prep[shape]` precomputed as exactly
    `_t_cline_scale` above computes it) -- see `references/Molecular
    Ecology - 2025 - Baird - The Shapes of Clines and Wavefronts.pdf`,
    Table 1's "T-cline T(v)" row, for the full derivation.
    """
    if params.t_shape is None or params.t_shape <= 0:
        raise ValueError("t_cline_function requires params.t_shape > 0, got "
                          f"{params.t_shape!r}")
    if params.width <= 0:
        raise ValueError(f"t_cline_function requires params.width > 0, got {params.width!r}")

    x = np.asarray(x, dtype=float)
    raw = (x - params.center) / params.width
    scale = _t_cline_scale(params.t_shape)
    cdf = _stats.t.cdf(raw, df=params.t_shape, scale=scale)
    return params.pmin + (params.pmax - params.pmin) * cdf


# Human-readable parameter names, in the same order as ClineParams.as_array(),
# used for reporting fitted results.
def param_names(tail_type: TailType) -> list[str]:
    names = ["center", "width", "pmin", "pmax"]
    if tail_type in (TailType.LEFT, TailType.BOTH):
        names += ["delta_l", "tau_l"]
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        names += ["delta_r", "tau_r"]
    if tail_type == TailType.MIRROR:
        names += ["delta_m", "tau_m"]
    if tail_type == TailType.T_CLINE:
        names += ["t_shape"]
    return names
