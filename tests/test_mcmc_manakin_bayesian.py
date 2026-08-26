"""
Regression test: the Bayesian (NUTS) fit to the real Manakin Ada locus
data, with a sensibly-bounded width prior, should produce clean convergence
diagnostics and a credible interval that contains HZAR's published width
estimate -- even though the posterior mean itself may sit above the MLE/HZAR
point estimate (expected behavior with a fairly flat prior and only 11
sampling sites; see BACKEND_PHASE2.md section 3 and
examples/demo_bayesian_manakin.py for the full discussion).

This also locks in the current ArviZ (>=1.0) hdi() API usage
(`prob=`, `ci_bound` coordinate) so an ArviZ upgrade that changes the API
again gets caught here rather than silently in a script.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mle import fit_mle_fixed_extremes
from hycline.inference.mcmc_numpyro import fit_mcmc
from hycline.diagnostics.convergence import check
from hycline.models.cline import TailType

DATA_PATH = Path(__file__).parent.parent / "examples" / "manakin_ada.json"

HZAR_CENTER = 170.9
HZAR_WIDTH = 263.1


def _load_ada_transect() -> AlleleFreqTransect:
    with open(DATA_PATH) as f:
        rows = json.load(f)
    distance = np.array([float(r["distance"]) for r in rows])
    freq = np.array([float(r["ada_A"]) for r in rows])
    n_alleles = np.array([float(r["ada_nSamples"]) for r in rows])
    n_focal = np.round(freq * n_alleles)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def test_manakin_ada_bayesian_fit_converges_cleanly_with_bounded_width_prior():
    transect = _load_ada_transect()
    mle_fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    d_span = float(transect.distance.max() - transect.distance.min())

    fit = fit_mcmc(transect, pmin_fixed=mle_fit.params.pmin, pmax_fixed=mle_fit.params.pmax,
                    width_upper_bound=d_span,
                    num_warmup=1500, num_samples=1500, num_chains=4, seed=0)

    report = check(fit.idata)
    assert report.ok, report.warnings
    assert report.n_divergences == 0


def test_manakin_ada_bayesian_credible_intervals_contain_hzar_published_values():
    transect = _load_ada_transect()
    mle_fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    d_span = float(transect.distance.max() - transect.distance.min())

    fit = fit_mcmc(transect, pmin_fixed=mle_fit.params.pmin, pmax_fixed=mle_fit.params.pmax,
                    width_upper_bound=d_span,
                    num_warmup=1500, num_samples=1500, num_chains=4, seed=0)

    center_lo, center_hi = fit.hdi("center")
    width_lo, width_hi = fit.hdi("width")

    assert center_lo <= HZAR_CENTER <= center_hi
    assert width_lo <= HZAR_WIDTH <= width_hi


def test_manakin_ada_default_wide_prior_shows_the_identifiability_problem():
    """Documents (rather than silently avoids) the failure mode: with the
    default width prior (10x transect span) on this sparse 11-site dataset,
    convergence diagnostics should flag a problem. If this test starts
    failing because the default prior stops being flagged, that's worth
    knowing -- it would mean either the diagnostics thresholds changed or
    NUTS got meaningfully better at this geometry, both notable."""
    transect = _load_ada_transect()
    mle_fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)

    fit = fit_mcmc(transect, pmin_fixed=mle_fit.params.pmin, pmax_fixed=mle_fit.params.pmax,
                    num_warmup=1500, num_samples=1500, num_chains=4, seed=0)  # default (loose) prior

    report = check(fit.idata)
    assert not report.ok  # expected: this is the known-bad configuration
