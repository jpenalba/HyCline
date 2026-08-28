"""Primary Bayesian inference backend: NUTS via NumPyro.

See BACKEND_PHASE2.md at the repo root for the full design rationale (priors,
why NUTS, why `numpyro.plate` rather than a hand-written `jax.vmap` for
multi-locus batching, diagnostics interpretation). This module docstring
covers only the API surface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import arviz as az
import jax

# JAX defaults to 32-bit floats unless told otherwise, which is fine for
# most ML workloads but introduces ~1e-7 relative error that's large enough
# to matter when directly comparing against the Phase 1 MLE backend (which
# runs in float64 via NumPy/SciPy) -- see BACKEND_PHASE2.md section 11. Enabling
# x64 must happen before any JAX arrays are created, hence right after import.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import ClineParams, TailType

_EPS = 1e-6


def _cline_core_jax(x, center, width, pmin, pmax):
    """JAX-native reimplementation of the sigmoid cline core.

    Mirrors `hycline.models.cline._sigmoid_core` exactly, but written with
    `jax.numpy` so it's traceable inside a NumPyro model. See BACKEND_PHASE2.md
    section 10 for why this duplication exists rather than sharing code with
    `models/cline.py`, and `tests/test_mcmc_numpyro.py::
    test_jax_core_matches_numpy_core` for the regression test that keeps
    the two in agreement.
    """
    z = -4.0 * (x - center) / width
    z = jnp.clip(z, -700, 700)
    return pmin + (pmax - pmin) / (1.0 + jnp.exp(z))


def _cline_tail_jax(x, core, center, width, pmin, pmax, tail_type,
                     delta_l=None, tau_l=None, delta_r=None, tau_r=None):
    """JAX-native reimplementation of `hycline.models.cline.cline_function`'s
    tail extension (Barton & Gale 1993 exponential-decay tails grafted onto
    the sigmoid core). Mirrors the NumPy version's logic exactly, but uses
    `jnp.where` in place of NumPy boolean-mask assignment (`out[mask] = ...`),
    since JAX arrays are immutable and `jnp.where` is the differentiable,
    traceable equivalent -- the same substitution `_cline_core_jax` already
    makes for the sigmoid core itself. `tail_type` is a plain Python enum
    (not a traced value), so the branches below are resolved once at trace
    time, the same pattern already used for `pmin_fixed`/`pmax_fixed` in
    `_single_locus_model`. Verified to match the NumPy version exactly
    (`atol=1e-10`) for every tail type: `tests/test_mcmc_numpyro.py::
    test_jax_tail_matches_numpy_tail`.
    """
    if tail_type == TailType.NONE:
        return core

    left_edge = center - width / 2.0
    right_edge = center + width / 2.0
    out = core

    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        left_val = pmin + (pmax - pmin) * delta_l * jnp.exp((x - left_edge) / tau_l)
        out = jnp.where(x < left_edge, left_val, out)

    if tail_type in (TailType.RIGHT, TailType.BOTH, TailType.MIRROR):
        dr = delta_r if tail_type != TailType.MIRROR else delta_l
        tr = tau_r if tail_type != TailType.MIRROR else tau_l
        right_val = pmax - (pmax - pmin) * dr * jnp.exp(-(x - right_edge) / tr)
        out = jnp.where(x > right_edge, right_val, out)

    return out


def _hzar_tail_alpha_jax(delta, tau, width):
    """JAX-native version of `hycline.models.cline._hzar_tail_alpha` --
    see that function's docstring for the derivation. Kept in exact
    agreement with the NumPy version, the same way `_cline_core_jax`
    mirrors `_sigmoid_core` -- verified in
    `tests/test_mcmc_numpyro.py::test_jax_hzar_tail_matches_numpy_hzar_tail`.
    """
    logit_delta = jnp.clip(4.0 * delta / width, -700, 700)
    return jnp.exp(logit_delta * (tau - 1.0)) / (1.0 + jnp.exp(-logit_delta))


def _hzar_cline_tail_jax(x, core, center, width, pmin, pmax, tail_type,
                          delta_l=None, tau_l=None, delta_r=None, tau_r=None):
    """JAX-native reimplementation of
    `hycline.models.cline.hzar_cline_function`'s tail extension -- HZAR's
    own tail parameterization, verified directly against HZAR's R source
    (BACKEND_TRAIT.md sections 1-2), as opposed to `_cline_tail_jax`
    above (this project's original, differently-parameterized tail
    model). A REAL, NAMED DIFFERENCE from `_cline_tail_jax`: HZAR's tail
    edges are `center -/+ delta` directly (a raw distance), not
    `center -/+ width/2` -- see `hzar_cline_function`'s own docstring.
    `tail_type` is a plain Python enum, resolved at trace time, the same
    pattern `_cline_tail_jax` already uses.
    """
    if tail_type == TailType.NONE:
        return core

    out = core

    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        left_edge = center - delta_l
        alpha = _hzar_tail_alpha_jax(delta_l, tau_l, width)
        u = jnp.clip(4.0 * (x - center) / width * tau_l, -700, 700)
        frac = alpha * jnp.exp(u)
        left_val = pmin + (pmax - pmin) * frac
        out = jnp.where(x < left_edge, left_val, out)

    if tail_type in (TailType.RIGHT, TailType.BOTH):
        right_edge = center + delta_r
        alpha = _hzar_tail_alpha_jax(delta_r, tau_r, width)
        u = jnp.clip(-4.0 * (x - center) / width * tau_r, -700, 700)
        frac_r = alpha * jnp.exp(u)
        right_val = pmin + (pmax - pmin) * (1.0 - frac_r)
        out = jnp.where(x > right_edge, right_val, out)

    if tail_type == TailType.MIRROR:
        dr, tr = delta_l, tau_l  # shared value for both tails, matching
        right_edge = center + dr  # _cline_tail_jax's own MIRROR convention
        alpha = _hzar_tail_alpha_jax(dr, tr, width)
        u = jnp.clip(-4.0 * (x - center) / width * tr, -700, 700)
        frac_r = alpha * jnp.exp(u)
        right_val = pmin + (pmax - pmin) * (1.0 - frac_r)
        out = jnp.where(x > right_edge, right_val, out)

    return out


def _sample_tail_params(tail_type, width_bound):
    """Sample whichever tail parameters `tail_type` needs, with priors
    matching Phase 1's classic-MLE bounds exactly (delta in [0, 1] as a
    height fraction, tau on the same scale as width -- both flat/Uniform,
    consistent with how center/width/pmin/pmax are already priored here).
    Returns (delta_l, tau_l, delta_r, tau_r), with unused entries left
    `None` -- e.g. TailType.LEFT only samples delta_l/tau_l, matching
    `ClineParams.as_array`'s own per-tail-type parameter set."""
    delta_l = tau_l = delta_r = tau_r = None
    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        delta_l = numpyro.sample("delta_l", dist.Uniform(0.0, 1.0))
        tau_l = numpyro.sample("tau_l", dist.Uniform(_EPS, width_bound))
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        delta_r = numpyro.sample("delta_r", dist.Uniform(0.0, 1.0))
        tau_r = numpyro.sample("tau_r", dist.Uniform(_EPS, width_bound))
    return delta_l, tau_l, delta_r, tau_r


def _sample_tail_params_hzar(tail_type, width_bound):
    """The HZAR-parameterized analog of `_sample_tail_params`: `delta` is
    a raw distance (the tail edge is `center +/- delta` directly, not a
    `[0, 1]` height fraction), bounded the same way
    `hycline.inference.mle_trait._bounds_for_trait` bounds it for
    classic MLE trait fitting -- `tau` bounded strictly positive
    (matching `_sample_tail_params`'s own tau bound, reused as-is since
    both parameterizations use `tau` as a decay-rate scale).

    `delta`'s bound is `width_bound / 2` (not `width_bound` itself, and
    deliberately not `width_bound * 5` as an earlier version of this
    function used) -- since `width_bound` here is already ~10x the
    transect's own geographic span (`fit_mcmc`'s own default), a
    `*5` multiplier gave a delta prior roughly 50x the data's span,
    dramatically wider than the `d_span * 5` bound
    `_bounds_for_trait` actually uses for the same parameter in the
    classic MLE case. That mismatch was caught during validation: a real
    fit against known ground truth (true delta_l=15) returned a
    posterior mean around 3000, driven by the sampler wandering into an
    essentially unconstrained region of a already-known-to-be-weakly-
    identified parameter (BACKEND_PHASE2.md section 12.3) rather than by
    a fault in the likelihood itself -- tightening the prior to a range
    consistent with the classic MLE bound fixed this."""
    delta_l = tau_l = delta_r = tau_r = None
    delta_bound = width_bound / 2.0
    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        delta_l = numpyro.sample("delta_l", dist.Uniform(_EPS, delta_bound))
        tau_l = numpyro.sample("tau_l", dist.Uniform(_EPS, 50.0))
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        delta_r = numpyro.sample("delta_r", dist.Uniform(_EPS, delta_bound))
        tau_r = numpyro.sample("tau_r", dist.Uniform(_EPS, 50.0))
    return delta_l, tau_l, delta_r, tau_r


def _single_locus_model(distance, n_alleles, n_focal, center_bounds, width_bound,
                         pmin_fixed: Optional[float], pmax_fixed: Optional[float],
                         tail_type: TailType = TailType.NONE, hzar_tail: bool = False):
    center = numpyro.sample("center", dist.Uniform(*center_bounds))
    width = numpyro.sample("width", dist.Uniform(_EPS, width_bound))

    if pmin_fixed is None:
        pmin = numpyro.sample("pmin", dist.Uniform(0.0, 1.0))
    else:
        pmin = pmin_fixed
    if pmax_fixed is None:
        pmax = numpyro.sample("pmax", dist.Uniform(0.0, 1.0))
    else:
        pmax = pmax_fixed

    if hzar_tail:
        delta_l, tau_l, delta_r, tau_r = _sample_tail_params_hzar(tail_type, width_bound)
    else:
        delta_l, tau_l, delta_r, tau_r = _sample_tail_params(tail_type, width_bound)

    core = _cline_core_jax(distance, center, width, pmin, pmax)
    tail_fn = _hzar_cline_tail_jax if hzar_tail else _cline_tail_jax
    p = tail_fn(distance, core, center, width, pmin, pmax, tail_type,
                delta_l, tau_l, delta_r, tau_r)
    p = jnp.clip(p, _EPS, 1.0 - _EPS)
    numpyro.sample("obs", dist.Binomial(total_count=n_alleles, probs=p), obs=n_focal)


def _batch_model(distance, n_alleles, n_focal, n_loci, center_bounds, width_bound,
                  tail_type: TailType = TailType.NONE, hzar_tail: bool = False):
    """Same model as `_single_locus_model`, but with an extra `locus` plate
    dimension so a single MCMC run fits all loci at once. See BACKEND_PHASE2.md
    section 4 for why this uses `numpyro.plate` rather than `jax.vmap`
    directly, and why this is *not* a hierarchical (partial-pooling) model
    -- each locus gets its own independent flat priors, tail parameters
    included.

    `n_alleles`/`n_focal` are expected as (n_loci, n_sites) arrays sharing
    the same `distance` (n_sites,) vector across loci.
    """
    with numpyro.plate("locus", n_loci):
        center = numpyro.sample("center", dist.Uniform(*center_bounds))
        width = numpyro.sample("width", dist.Uniform(_EPS, width_bound))
        pmin = numpyro.sample("pmin", dist.Uniform(0.0, 1.0))
        pmax = numpyro.sample("pmax", dist.Uniform(0.0, 1.0))
        if hzar_tail:
            delta_l, tau_l, delta_r, tau_r = _sample_tail_params_hzar(tail_type, width_bound)
        else:
            delta_l, tau_l, delta_r, tau_r = _sample_tail_params(tail_type, width_bound)

    # broadcast: distance is (n_sites,), center/width/pmin/pmax are
    # (n_loci,) -> reshape to (n_loci, 1) so the result is (n_loci, n_sites).
    # This happens outside the plate context -- it's plain array math, not
    # another numpyro.sample call, so there's no dimension bookkeeping to
    # coordinate with the "locus" plate above (an earlier version nested a
    # second "site" plate here and hit a dim-allocation conflict with
    # "locus" -- not needed for correctness under plain MCMC, only for
    # SVI-style guides, so it's dropped rather than fought with).
    def _b(v):
        return v[:, None] if v is not None else None

    p_core = _cline_core_jax(distance[None, :], center[:, None], width[:, None],
                              pmin[:, None], pmax[:, None])
    tail_fn = _hzar_cline_tail_jax if hzar_tail else _cline_tail_jax
    p = tail_fn(distance[None, :], p_core, center[:, None], width[:, None],
                pmin[:, None], pmax[:, None], tail_type,
                _b(delta_l), _b(tau_l), _b(delta_r), _b(tau_r))
    p = jnp.clip(p, _EPS, 1.0 - _EPS)
    numpyro.sample("obs", dist.Binomial(total_count=n_alleles, probs=p), obs=n_focal)


@dataclass
class BayesianFitResult:
    idata: az.InferenceData
    tail_type: TailType
    n_obs: int
    backend: str  # "numpyro" or "pymc"
    param_names: list

    @property
    def params(self) -> ClineParams:
        """Point-estimate ClineParams (posterior mean of each parameter),
        for reuse with `cline_function` / plotting -- analogous to
        Phase 1's FitResult.params, but note this collapses a full
        posterior down to a point estimate; use `.idata` directly for
        anything that needs the full uncertainty."""
        post = self.idata.posterior
        if "locus" in post.dims:
            raise ValueError(
                "params is not meaningful on a multi-locus (fit_mcmc_batch) result -- "
                "averaging posterior samples across the locus batch dimension would "
                "silently produce a meaningless cross-locus average, not a per-locus "
                "point estimate. Use .summary_df() for a per-locus table instead, or "
                "index .idata.posterior by the 'locus' coordinate for a single locus's "
                "posterior before constructing ClineParams by hand."
            )
        kwargs = {}
        for name in ("center", "width", "pmin", "pmax", "delta_l", "tau_l", "delta_r", "tau_r"):
            if name in post:
                kwargs[name] = float(post[name].mean())
            elif name == "pmin":
                kwargs[name] = float(self._fixed_pmin)
            elif name == "pmax":
                kwargs[name] = float(self._fixed_pmax)
        if "delta_r" not in kwargs and "delta_l" in kwargs and self.tail_type == TailType.MIRROR:
            kwargs["delta_r"], kwargs["tau_r"] = kwargs["delta_l"], kwargs["tau_l"]
        return ClineParams(**kwargs)

    _fixed_pmin: Optional[float] = None
    _fixed_pmax: Optional[float] = None
    hzar_tail: bool = False

    def summary_df(self) -> pd.DataFrame:
        return az.summary(self.idata, var_names=self.param_names)

    def hdi(self, param: str, prob: float = 0.95) -> tuple:
        """Highest-density interval for a parameter. Note: uses ArviZ's
        actual, verified `hdi()` API (`hdi_prob=`, `hdi` coordinate with
        'lower'/'higher' labels) -- confirmed directly against the real
        installed ArviZ (checked via `inspect.signature(az.hdi)` and a
        live call, not assumed from changelog text). An earlier version of
        this docstring claimed ArviZ >=1.0 uses a different `prob=`/
        `ci_bound` API; that was wrong -- no such ArviZ release exists on
        PyPI (the latest real release as of this check is 0.23.4, which
        uses `hdi_prob=`/`hdi`), and the wrong claim was never actually
        verified against a real install. See BACKEND_PHASE2.md section 8
        for the corrected version history and how this was caught."""
        interval = az.hdi(self.idata.posterior, var_names=[param], hdi_prob=prob)
        lo = float(interval[param].sel(hdi="lower"))
        hi = float(interval[param].sel(hdi="higher"))
        return lo, hi

    def summary(self) -> str:
        df = self.summary_df()
        tail_label = self.tail_type.value
        if self.tail_type != TailType.NONE and self.hzar_tail:
            tail_label += ", HZAR-parameterized"
        lines = [f"Bayesian cline fit (backend={self.backend}, tail={tail_label}), "
                 f"n_obs={self.n_obs}"]
        lines.append(df.to_string())
        return "\n".join(lines)


def fit_mcmc(transect: AlleleFreqTransect, tail_type: TailType = TailType.NONE,
             pmin_fixed: Optional[float] = None, pmax_fixed: Optional[float] = None,
             width_upper_bound: Optional[float] = None,
             num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 4,
             seed: int = 0, hzar_tail: bool = False) -> BayesianFitResult:
    """Fit a single-locus cline model by NUTS MCMC via NumPyro.

    Parameters
    ----------
    transect : the allele-frequency data to fit
    tail_type : any of TailType.NONE/LEFT/RIGHT/BOTH/MIRROR. Tail
        parameters (delta_l/tau_l/delta_r/tau_r as applicable) get flat
        Uniform priors matching Phase 1's classic-MLE bounds exactly
        (delta in [0, 1], tau on the same scale as width) -- see
        BACKEND_PHASE2.md section 12 for the full design and validation.
        PyMC (`fit_mcmc_pymc`) still only supports TailType.NONE.
    pmin_fixed, pmax_fixed : if given, these are held fixed (not sampled) --
        mirrors HZAR's "scaling=fixed" mode / Phase 1's fit_mle_fixed_extremes
    width_upper_bound : upper bound of the (flat) width prior. Defaults to
        10x the transect's geographic span, matching the Phase 1 MLE's
        bounds for comparability -- but for sparse real datasets (few
        sampling sites) this default can be far looser than the data can
        actually constrain, which shows up as R-hat/ESS/divergence warnings
        from `hycline.diagnostics.convergence.check()` (see BACKEND_PHASE2.md
        section 3). If you see those warnings, tightening this to something
        closer to a biologically plausible width for your organism/system
        is usually the fix, not just running more samples. Tail decay
        lengths (tau_l/tau_r) share this same upper bound.
    num_warmup, num_samples, num_chains : standard NUTS/MCMC controls
    seed : PRNG seed for reproducibility
    hzar_tail : if True, use HZAR's own tail parameterization
        (verified directly against HZAR's R source -- BACKEND_TRAIT.md
        sections 1-2) instead of this project's original, differently-
        parameterized tail model. Default False preserves original
        behavior exactly -- ported from the classic-MLE `fit_mle`
        version of this same flag (BACKEND_TRAIT.md section 5) once that
        was verified; the known Bayesian tail-model convergence
        difficulty above applies equally to either parameterization, not
        specific to this flag. Ignored for `TailType.NONE`.

    Returns
    -------
    BayesianFitResult
    """
    if tail_type == TailType.T_CLINE:
        raise ValueError(
            "fit_mcmc does not support tail_type=TailType.T_CLINE -- Bayesian (NUTS) "
            "fitting is not available for the T-cline, for a real, verified reason "
            "(jax.scipy.special.betainc does not support gradients with respect to "
            "its shape parameters, which t_shape needs -- see "
            "hycline.models.cline.t_cline_function's own docstring and "
            "BACKEND_PHASE2.md section 15.3). Use --method classic / "
            "hycline.inference.mle.fit_mle instead for T-cline fitting."
        )

    d_span = transect.distance.max() - transect.distance.min()
    center_bounds = (float(transect.distance.min() - d_span),
                      float(transect.distance.max() + d_span))
    if width_upper_bound is None:
        width_upper_bound = float(d_span * 10 + 1e-3)

    if tail_type != TailType.NONE:
        warnings.warn(
            "Bayesian (NUTS) tail-model fitting has a known, real convergence "
            "difficulty: the tail model's non-smooth join between the sigmoid core "
            "and the exponential tail creates a kinked log-density that gradient-based "
            "samplers can struggle with severely (poor R-hat/ESS, sometimes no "
            "practical convergence even with many warmup iterations) -- this is not a "
            "bug, and more warmup/samples often does not fix it. Baird & Daley (2025) "
            "give the structural reason directly: the join is a genuine mathematical "
            "discontinuity (a 'corner'), not a numerical inconvenience specific to this "
            "sampler. See BACKEND_PHASE2.md section 12.2 for the full investigation. "
            "Classic MLE (hycline.inference.mle_genotype_likelihood.fit_mle_geographic_gl "
            "or hycline.inference.mle.fit_mle) does not have this issue and is the "
            "recommended path for tail models -- or, for a continuous-everywhere "
            "alternative that avoids the corner at its source rather than working around "
            "it, hycline.models.cline.t_cline_function (--tail t-cline, classic MLE "
            "only -- Bayesian support is not available for it either, for a separate, "
            "specific reason: see t_cline_function's own docstring and BACKEND_PHASE2.md "
            "section 15.3). If you do use this Bayesian path with the tripartite tail "
            "model, always check hycline.diagnostics.convergence.check() before trusting "
            "the tail parameters.",
            UserWarning, stacklevel=2,
        )

    kernel = NUTS(_single_locus_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key,
              distance=jnp.asarray(transect.distance),
              n_alleles=jnp.asarray(transect.n_alleles),
              n_focal=jnp.asarray(transect.n_focal),
              center_bounds=center_bounds, width_bound=width_upper_bound,
              pmin_fixed=pmin_fixed, pmax_fixed=pmax_fixed, tail_type=tail_type,
              hzar_tail=hzar_tail)

    idata = az.from_numpyro(mcmc)
    param_names = [n for n in ("center", "width", "pmin", "pmax",
                                "delta_l", "tau_l", "delta_r", "tau_r")
                   if n in idata.posterior]

    result = BayesianFitResult(idata=idata, tail_type=tail_type, n_obs=len(transect),
                                backend="numpyro", param_names=param_names,
                                hzar_tail=hzar_tail)
    result._fixed_pmin = pmin_fixed
    result._fixed_pmax = pmax_fixed
    return result


def fit_mcmc_batch(distance: np.ndarray, n_alleles: np.ndarray, n_focal: np.ndarray,
                    locus_names: Optional[list[str]] = None,
                    tail_type: TailType = TailType.NONE,
                    width_upper_bound: Optional[float] = None,
                    num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 4,
                    seed: int = 0, hzar_tail: bool = False) -> BayesianFitResult:
    """Fit many loci at once via a single plate-vectorized NUTS run.

    All loci must share the same sampling sites (same `distance` vector).
    Each locus gets independent flat priors -- no cross-locus pooling (see
    BACKEND_PHASE2.md section 4; that's the Phase 4 hierarchical extension).
    Tail parameters, when `tail_type != TailType.NONE`, are likewise
    independent per locus (no pooling of tail shape across loci -- see
    BACKEND_PHASE2.md section 12.4 for why this is explicitly out of
    scope for this pass, not silently assumed away).

    Parameters
    ----------
    distance : (n_sites,) array of geographic distances, shared across loci
    n_alleles : (n_loci, n_sites) array of total sampled alleles per locus/site
    n_focal : (n_loci, n_sites) array of focal-allele counts per locus/site
    locus_names : optional labels for reporting; defaults to locus_0, locus_1, ...
    tail_type : any of TailType.NONE/LEFT/RIGHT/BOTH/MIRROR -- see
        `fit_mcmc`'s docstring for the same tail-model design, applied
        per locus here.
    width_upper_bound : see `fit_mcmc`'s docstring -- same rationale applies
        here, and matters just as much (if not more) with sparse per-locus data.
    hzar_tail : see `fit_mcmc`'s docstring -- same flag, applied per locus
        here. Default False preserves original behavior exactly.

    Returns
    -------
    BayesianFitResult whose posterior has a `locus` batch dimension for
    every sampled parameter -- use `.summary_df()` to get a per-locus table.
    """
    if tail_type == TailType.T_CLINE:
        raise ValueError(
            "fit_mcmc_batch does not support tail_type=TailType.T_CLINE -- see "
            "fit_mcmc's docstring for why Bayesian T-cline fitting is not available "
            "(a real JAX/betainc gradient limitation, not a todo-list gap). Use "
            "--method classic / hycline.inference.mle.fit_mle instead."
        )

    distance = jnp.asarray(distance)
    n_alleles = jnp.asarray(n_alleles)
    n_focal = jnp.asarray(n_focal)
    n_loci = n_alleles.shape[0]

    if locus_names is None:
        locus_names = [f"locus_{i}" for i in range(n_loci)]

    d_span = float(distance.max() - distance.min())
    center_bounds = (float(distance.min()) - d_span, float(distance.max()) + d_span)
    if width_upper_bound is None:
        width_upper_bound = d_span * 10 + 1e-3

    if tail_type != TailType.NONE:
        warnings.warn(
            "Bayesian (NUTS) tail-model fitting has a known, real convergence "
            "difficulty -- a genuine mathematical discontinuity where the sigmoid core "
            "joins the exponential tail (Baird & Daley 2025), not a numerical "
            "inconvenience -- see fit_mcmc's docstring / BACKEND_PHASE2.md section 12.2 "
            "for the full investigation. This applies per-locus here too. Classic MLE "
            "does not have this issue and is the recommended path for tail models -- or "
            "hycline.models.cline.t_cline_function (--tail t-cline) for a continuous-"
            "everywhere alternative, classic MLE only (see BACKEND_PHASE2.md "
            "section 15.3 for why Bayesian support isn't available for it either).",
            UserWarning, stacklevel=2,
        )

    kernel = NUTS(_batch_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, distance=distance, n_alleles=n_alleles, n_focal=n_focal,
              n_loci=n_loci, center_bounds=center_bounds, width_bound=width_upper_bound,
              tail_type=tail_type, hzar_tail=hzar_tail)

    param_names = ["center", "width", "pmin", "pmax"]
    if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
        param_names += ["delta_l", "tau_l"]
    if tail_type in (TailType.RIGHT, TailType.BOTH):
        param_names += ["delta_r", "tau_r"]

    idata = az.from_numpyro(mcmc, coords={"locus": locus_names},
                             dims={name: ["locus"] for name in param_names})

    result = BayesianFitResult(idata=idata, tail_type=tail_type, n_obs=n_loci,
                                backend="numpyro", param_names=param_names,
                                hzar_tail=hzar_tail)
    return result
