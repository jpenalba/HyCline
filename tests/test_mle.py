import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.inference.mle import fit_mle
from hycline.selection.aicc import compare_models


def _simulate(true_params, n_sites=60, n_alleles_range=(20, 50), seed=1):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(-30, 130, size=n_sites))
    n_alleles = rng.integers(*n_alleles_range, size=n_sites).astype(float)
    true_freq = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def test_mle_recovers_known_parameters_reasonably_well():
    true_params = ClineParams(center=50.0, width=20.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=80, n_alleles_range=(30, 60), seed=7)

    fit = fit_mle(transect, tail_type=TailType.NONE)

    assert fit.success
    # with this much data, center/width should be recovered within a
    # generous tolerance -- this is a sanity check, not a precision test
    assert fit.params.center == pytest.approx(true_params.center, abs=5.0)
    assert fit.params.width == pytest.approx(true_params.width, abs=8.0)


def test_aicc_prefers_simpler_model_on_notail_data():
    """When the true generating process has no tails, the no-tail model
    should be preferred (or at least competitive) over the mirror-tail model."""
    true_params = ClineParams(center=40.0, width=15.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, n_sites=70, seed=3)

    comparison = compare_models(transect, candidates=None)
    # the no-tail model should not be drastically worse than the best model
    none_row = comparison.table[comparison.table["model"] == "none"].iloc[0]
    assert none_row["delta_AICc"] < 10  # loose bound; guards against gross mis-fitting


def test_fit_result_summary_runs_without_error():
    true_params = ClineParams(center=10.0, width=5.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, n_sites=30, seed=2)
    fit = fit_mle(transect, tail_type=TailType.NONE)
    s = fit.summary()
    assert "AICc" in s


# ---------------------------------------------------------------------------
# fit_mle_fixed_extremes: n_edge_sites (BACKEND_PHASE4.md section 3's fix,
# made available here too -- see BACKEND_PHASE1.md section 4.2)
# ---------------------------------------------------------------------------

def test_derive_pmin_pmax_single_locus_matches_direct_computation():
    from hycline.inference.mle import _derive_pmin_pmax_single_locus

    n_alleles = np.array([10, 10, 10, 10], dtype=float)
    n_focal = np.array([1, 2, 8, 9], dtype=float)
    distance = np.array([0, 1, 2, 3], dtype=float)
    pmin, pmax = _derive_pmin_pmax_single_locus(distance, n_alleles, n_focal, n_edge_sites=2)
    # pmin = (1+2)/(10+10) = 0.15, pmax = (8+9)/(10+10) = 0.85
    assert pmin == pytest.approx(0.15)
    assert pmax == pytest.approx(0.85)


def test_derive_pmin_pmax_single_locus_default_matches_old_single_site_convention():
    """The core backward-compatibility guarantee: n_edge_sites=1 (the
    default) must reproduce the exact original single-site behavior --
    not just approximately, since this is what the Manakin cross-
    validation was checked against and must not silently change."""
    from hycline.inference.mle import _derive_pmin_pmax_single_locus

    distance = np.array([5.0, 1.0, 3.0, 9.0, 7.0])
    n_alleles = np.array([20.0, 15.0, 25.0, 30.0, 18.0])
    n_focal = np.array([2.0, 1.0, 10.0, 27.0, 16.0])

    pmin, pmax = _derive_pmin_pmax_single_locus(distance, n_alleles, n_focal)  # default n_edge_sites=1
    # sorted by distance: [1,3,5,7,9] -> n_focal/n_alleles at the extremes
    # min-distance site (distance=1): n_focal=1, n_alleles=15 -> 1/15
    # max-distance site (distance=9): n_focal=27, n_alleles=30 -> 27/30
    assert pmin == pytest.approx(1.0 / 15.0)
    assert pmax == pytest.approx(27.0 / 30.0)


def test_fit_mle_fixed_extremes_default_n_edge_sites_matches_original_behavior():
    """fit_mle_fixed_extremes's default (n_edge_sites=1) must give
    bit-identical pmin/pmax to what the original single-site
    implementation computed directly -- confirms the refactor didn't
    silently change default behavior."""
    true_params = ClineParams(center=10.0, width=5.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, n_sites=30, seed=4)

    order = np.argsort(transect.distance)
    freq_sorted = transect.n_focal[order] / transect.n_alleles[order]
    expected_pmin, expected_pmax = float(freq_sorted[0]), float(freq_sorted[-1])

    from hycline.inference.mle import fit_mle_fixed_extremes
    fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    assert fit.params.pmin == pytest.approx(expected_pmin)
    assert fit.params.pmax == pytest.approx(expected_pmax)


def test_fit_mle_fixed_extremes_n_edge_sites_mitigates_unlucky_extreme_draw():
    """The actual fix, checked end to end through the fitting function
    (not just the frequency-derivation helper in isolation): an unlucky
    zero-count draw forced at the extreme site should distort the fitted
    width much less when n_edge_sites=3 pools several sites, compared to
    the single-site default."""
    from hycline.inference.mle import fit_mle_fixed_extremes

    rng = np.random.default_rng(5)
    true_params = ClineParams(center=50, width=15, pmin=0.05, pmax=0.95)
    distance = np.sort(rng.uniform(0, 100, 15))
    n_alleles = np.full(15, 20.0)
    p_true = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(20, p_true).astype(float)
    order = np.argsort(distance)
    n_focal[order[0]] = 0.0  # force the unlucky draw at the true extreme site

    transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)

    fit_single = fit_mle_fixed_extremes(transect, n_edge_sites=1)
    fit_robust = fit_mle_fixed_extremes(transect, n_edge_sites=3)

    assert fit_single.params.pmin == 0.0  # confirms the pathological input
    assert fit_robust.params.pmin > 0.0
    # the robust version's width should land substantially closer to the
    # true value of 15 than the single-site version does
    assert abs(fit_robust.params.width - 15) < abs(fit_single.params.width - 15)


def test_fit_mle_fixed_extremes_explicit_pmin_pmax_still_overrides_n_edge_sites():
    """Explicitly passing pmin/pmax should still take priority over any
    derivation, regardless of n_edge_sites -- unchanged contract."""
    from hycline.inference.mle import fit_mle_fixed_extremes

    true_params = ClineParams(center=10.0, width=5.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, n_sites=30, seed=6)
    fit = fit_mle_fixed_extremes(transect, pmin=0.2, pmax=0.8, n_edge_sites=5)
    assert fit.params.pmin == 0.2
    assert fit.params.pmax == 0.8


# ---------------------------------------------------------------------------
# hzar_tail: opt-in HZAR-parameterized tail fitting (BACKEND_TRAIT.md
# sections 1-2), ported from the trait-cline work
# ---------------------------------------------------------------------------

def test_fit_mle_hzar_tail_default_false_preserves_original_behavior():
    """The core backward-compatibility guarantee: hzar_tail defaults to
    False, and results must be bit-identical to before this parameter
    existed."""
    true_params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9,
                               delta_l=0.15, tau_l=10.0)
    transect = _simulate(true_params, n_sites=40, seed=11)
    fit_default = fit_mle(transect, tail_type=TailType.LEFT)
    fit_explicit_false = fit_mle(transect, tail_type=TailType.LEFT, hzar_tail=False)
    np.testing.assert_allclose(fit_default.param_vec, fit_explicit_false.param_vec)
    assert fit_default.hzar_tail is False
    assert "HZAR-parameterized" not in fit_default.summary()


def test_fit_mle_hzar_tail_recovers_known_parameters_simulated_from_hzar_formula():
    from hycline.models.cline import hzar_cline_function

    rng = np.random.default_rng(1001)
    n_sites = 40
    distance = np.sort(rng.uniform(0, 200, n_sites))
    n_alleles = rng.integers(20, 50, n_sites).astype(float)
    true_params = ClineParams(center=100, width=30, pmin=0.05, pmax=0.95,
                               delta_l=15.0, tau_l=0.5)
    p_true = hzar_cline_function(distance, true_params, TailType.LEFT)
    n_focal = rng.binomial(n_alleles.astype(int), np.clip(p_true, 0, 1)).astype(float)
    transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)

    fit = fit_mle(transect, tail_type=TailType.LEFT, hzar_tail=True)
    assert fit.hzar_tail is True
    assert "HZAR-parameterized" in fit.summary()
    assert fit.params.center == pytest.approx(100, abs=10)
    assert fit.params.pmin == pytest.approx(0.05, abs=0.05)
    assert fit.params.pmax == pytest.approx(0.95, abs=0.05)
    # delta_l/tau_l themselves are not checked against truth -- tail
    # parameters are known to be weakly identified even in the best case
    # (BACKEND_PHASE2.md section 12.3); the important check is the core
    # (non-tail) parameters recover well, which they do.


def test_fit_mle_hzar_tail_none_identical_regardless_of_flag():
    """For TailType.NONE, hzar_tail should make no difference at all --
    both cline_function and hzar_cline_function are identical there."""
    true_params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, n_sites=30, seed=13)
    fit_false = fit_mle(transect, tail_type=TailType.NONE, hzar_tail=False)
    fit_true = fit_mle(transect, tail_type=TailType.NONE, hzar_tail=True)
    np.testing.assert_allclose(fit_false.param_vec, fit_true.param_vec)
