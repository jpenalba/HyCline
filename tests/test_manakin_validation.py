"""
Regression test: HyCline's fit to the real Manakin Ada locus data should
stay close to HZAR's own published result (Derryberry et al. 2014, Table 4)
for the same data and model. This is the Phase 1 acceptance test referenced
in the project roadmap and README.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mle import fit_mle_fixed_extremes
from hycline.likelihoods.binomial import hzar_style_log_likelihood
from hycline.models.cline import TailType

DATA_PATH = Path(__file__).parent.parent / "examples" / "manakin_ada.json"

# HZAR's published Model I result for the Ada locus (Derryberry et al. 2014,
# Table 4): pmin/pmax fixed to observed extremes, no tails.
HZAR_CENTER = 170.9
HZAR_CENTER_CI = (109.9, 201.2)
HZAR_WIDTH = 263.1
HZAR_WIDTH_CI = (96.2, 630.0)
HZAR_LOGL = -9.7  # HZAR's own likelihood convention -- see hzar_style_log_likelihood


def _load_ada_transect() -> AlleleFreqTransect:
    with open(DATA_PATH) as f:
        rows = json.load(f)
    distance = np.array([float(r["distance"]) for r in rows])
    freq = np.array([float(r["ada_A"]) for r in rows])
    n_alleles = np.array([float(r["ada_nSamples"]) for r in rows])
    n_focal = np.round(freq * n_alleles)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def test_manakin_ada_matches_hzar_published_center_and_width():
    transect = _load_ada_transect()
    fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)

    assert fit.success
    assert fit.params.center == pytest.approx(HZAR_CENTER, abs=2.0)
    assert fit.params.width == pytest.approx(HZAR_WIDTH, abs=3.0)
    # also check within HZAR's own published 2-log-likelihood-unit support interval
    assert HZAR_CENTER_CI[0] <= fit.params.center <= HZAR_CENTER_CI[1]
    assert HZAR_WIDTH_CI[0] <= fit.params.width <= HZAR_WIDTH_CI[1]


def test_manakin_ada_hzar_style_loglikelihood_matches_published():
    transect = _load_ada_transect()
    fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    ll = hzar_style_log_likelihood(transect, fit.params, TailType.NONE)
    assert ll == pytest.approx(HZAR_LOGL, abs=0.2)
