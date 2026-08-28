"""High-level, backend-agnostic fitting API.

The single design goal of this module: choosing between the "classic"
(HZAR-equivalent MLE) and "bayesian" (NUTS/MCMC) fitting procedures should
be a keyword argument, not a which-module-do-I-import decision. See
BACKEND_PHASE1.md and BACKEND_PHASE2.md for the full rationale behind each
procedure; this module is deliberately thin -- it dispatches, it doesn't
reimplement anything.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mle import FitResult, fit_mle, fit_mle_fixed_extremes
from hycline.models.cline import TailType


class Method(str, Enum):
    CLASSIC = "classic"    # HZAR-equivalent MLE point estimate + Hessian CIs (Phase 1)
    BAYESIAN = "bayesian"  # NUTS via NumPyro or PyMC, full posterior (Phase 2)


def fit_cline(transect: AlleleFreqTransect,
              method: Union[str, Method] = Method.CLASSIC,
              tail_type: TailType = TailType.NONE,
              pmin_fixed: Optional[float] = None,
              pmax_fixed: Optional[float] = None,
              backend: str = "numpyro",
              **kwargs):
    """Fit a cline model, choosing the fitting procedure explicitly.

    Both methods fit the *same* model (sigmoid cline, binomial likelihood;
    see BACKEND_PHASE2.md section 2) -- they differ in how they quantify
    uncertainty and what they cost to run, not in what they're modeling.
    See BACKEND_PHASE1.md section 8 for guidance on which to reach for.

    Parameters
    ----------
    transect : the allele-frequency data to fit
    method : "classic" (default) -- direct MLE via scipy.optimize, with
        Hessian-based confidence intervals. Reproduces numbers comparable
        to published HZAR-based studies (see BACKEND_PHASE1.md section 1
        for why this is a faster route to the same statistical target HZAR
        itself converges to via its own internal MCMC search).
        "bayesian" -- full posterior via NUTS MCMC (NumPyro or PyMC), with
        credible intervals from the posterior directly rather than a
        Gaussian approximation. Slower per fit, more trustworthy when the
        likelihood surface isn't well-approximated by a Gaussian (common
        for `width` with sparse data -- see BACKEND_PHASE2.md section 9
        for a worked example on real data).
    tail_type : cline tail configuration. The classic path supports the
        full tail family (including TailType.T_CLINE), also available
        via `hycline.selection.aicc.compare_models` for model selection.
        The Bayesian path supports TailType.NONE/LEFT/RIGHT/BOTH/MIRROR
        (with a real, documented NUTS convergence difficulty for the
        non-NONE cases -- a UserWarning fires and BACKEND_PHASE2.md
        section 12.2 has the full investigation) but not
        TailType.T_CLINE, which raises a clear error instead of silently
        fitting a plain sigmoid (a real JAX/betainc gradient limitation --
        see `hycline.models.cline.t_cline_function`'s docstring and
        BACKEND_PHASE2.md section 15.3).
    pmin_fixed, pmax_fixed : if given, held fixed rather than estimated,
        matching HZAR's "scaling=fixed" mode in both methods. For
        method="classic" this dispatches to `fit_mle_fixed_extremes`
        (note: that function derives its own fixed pmin/pmax from the
        transect's observed extremes if you don't pass values here --
        passing explicit values here overrides that).
    backend : for method="bayesian" only -- "numpyro" (default; the
        primary backend, also the only one supporting batch multi-locus
        fitting via `hycline.inference.mcmc_numpyro.fit_mcmc_batch`, not
        wrapped by this function) or "pymc" (secondary backend,
        single-locus only, does not yet support pmin_fixed/pmax_fixed --
        see BACKEND_PHASE2.md section 6). Ignored for method="classic".
    **kwargs : forwarded to the underlying fit function. For
        method="classic": `compute_std_errors`. For method="bayesian":
        `num_warmup`, `num_samples`, `num_chains`, `seed`, and (NumPyro
        backend only) `width_upper_bound` -- see BACKEND_PHASE2.md section
        9 for why that last one matters on sparse real data.

    Returns
    -------
    `hycline.inference.mle.FitResult` for method="classic", or
    `hycline.inference.mcmc_numpyro.BayesianFitResult` for
    method="bayesian". These are genuinely different shapes (point
    estimate + CI vs. full posterior) by design -- this function picks the
    procedure, it doesn't paper over the difference in what comes back.
    """
    method = Method(method)

    if method is Method.CLASSIC:
        if pmin_fixed is not None or pmax_fixed is not None:
            return fit_mle_fixed_extremes(transect, tail_type=tail_type,
                                           pmin=pmin_fixed, pmax=pmax_fixed, **kwargs)
        return fit_mle(transect, tail_type=tail_type, **kwargs)

    # method is Method.BAYESIAN -- import here, not at module level, so
    # that importing hycline.api (and therefore top-level `hycline`) never
    # forces JAX/NumPyro/ArviZ/PyMC to be installed just to use the classic
    # path. See BACKEND_PHASE2.md section 8 for the same reasoning applied
    # to hycline/__init__.py.
    if backend == "numpyro":
        from hycline.inference.mcmc_numpyro import fit_mcmc
        return fit_mcmc(transect, tail_type=tail_type,
                         pmin_fixed=pmin_fixed, pmax_fixed=pmax_fixed, **kwargs)
    elif backend == "pymc":
        if pmin_fixed is not None or pmax_fixed is not None:
            raise NotImplementedError(
                "The PyMC backend doesn't yet support pmin_fixed/pmax_fixed -- "
                "see BACKEND_PHASE2.md section 6. Use backend='numpyro' instead, "
                "which is also the primary/recommended backend generally."
            )
        from hycline.inference.mcmc_pymc import fit_mcmc_pymc
        return fit_mcmc_pymc(transect, tail_type=tail_type, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'numpyro' or 'pymc'.")


def fit_genomic_cline(data, **kwargs):
    """Fit the logit-logistic genomic cline model (Fitzpatrick 2013) to
    per-individual locus data against hybrid index.

    Unlike `fit_cline`, this doesn't take a `method=` choice -- Phase 3
    currently only implements classic MLE fitting (see BACKEND_PHASE3.md
    section 4 for a Bayesian NUTS backend as a tracked, not-yet-built
    extension, following the same pattern Phase 2 added for geographic
    clines). Kept as a separate top-level function rather than overloading
    `fit_cline` because the input data shape is fundamentally different
    (per-individual against hybrid index, not per-site against geographic
    distance) -- see BACKEND_PHASE3.md section 2.

    Parameters
    ----------
    data : a `hycline.data.schema.HybridIndexLocusData`
    **kwargs : forwarded to `hycline.inference.mle_genomic.fit_genomic_cline_mle`
        (currently just `compute_std_errors`)

    Returns
    -------
    `hycline.inference.mle_genomic.GenomicFitResult`
    """
    from hycline.inference.mle_genomic import fit_genomic_cline_mle
    return fit_genomic_cline_mle(data, **kwargs)
