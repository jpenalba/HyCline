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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class TailType(str, Enum):
    NONE = "none"        # pure sigmoid, no tails
    MIRROR = "mirror"     # symmetric exponential tails on both sides
    LEFT = "left"         # exponential tail on the left only
    RIGHT = "right"       # exponential tail on the right only
    BOTH = "both"         # independent exponential tails on both sides


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
    """

    center: float
    width: float
    pmin: float = 0.0
    pmax: float = 1.0
    delta_l: Optional[float] = None
    tau_l: Optional[float] = None
    delta_r: Optional[float] = None
    tau_r: Optional[float] = None

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
    return names
