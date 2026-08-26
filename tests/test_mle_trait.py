import numpy as np
import pytest
from scipy.stats import norm

from hycline.data.schema import TraitTransect
from hycline.models.cline import ClineParams, TailType
from hycline.likelihoods.gaussian import (
    mean_cline, variance_cline, log_likelihood, _tail_alpha,
)
from hycline.inference.mle_trait import fit_mle_trait


def _simulate_hzar(true_mean_params, true_var_l, true_var_r, true_var_h,
                    n_sites=30, seed=1, tail_type=TailType.NONE, n_range=(15, 30)):
    """Simulate raw per-individual data from HZAR's actual model
    (mean-cline + variance-cline), aggregate to site mean/sd/n -- the
    realistic data-generating process fit_mle_trait should be checked
    against."""
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 200, n_sites))
    n_per_site = rng.integers(*n_range, n_sites)
    p = true_mean_params
    mu = mean_cline(distance, p.center, p.width, p.pmin, p.pmax, tail_type,
                     p.delta_l, p.tau_l, p.delta_r, p.tau_r)
    var = variance_cline(distance, p.center, p.width, true_var_l, true_var_r, true_var_h,
                          tail_type, p.delta_l, p.tau_l, p.delta_r, p.tau_r)
    site_means, site_sds = [], []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
        site_means.append(obs.mean())
        site_sds.append(obs.std())
    return TraitTransect(distance=distance, trait_mean=np.array(site_means),
                          n=n_per_site.astype(float), trait_sd=np.array(site_sds))


# ---------------------------------------------------------------------------
# TraitTransect schema
# ---------------------------------------------------------------------------

def test_trait_transect_basic_construction():
    t = TraitTransect(distance=[0, 10, 20], trait_mean=[1.0, 2.0, 3.0], n=[5, 5, 5])
    assert len(t) == 3
    assert t.trait_sd is None


def test_trait_transect_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        TraitTransect(distance=[0, 10], trait_mean=[1.0, 2.0, 3.0], n=[5, 5])


def test_trait_transect_nonpositive_n_raises():
    with pytest.raises(ValueError, match="n must be positive"):
        TraitTransect(distance=[0, 10], trait_mean=[1.0, 2.0], n=[5, 0])


def test_trait_transect_negative_sd_raises():
    with pytest.raises(ValueError, match="cannot be negative"):
        TraitTransect(distance=[0, 10], trait_mean=[1.0, 2.0], n=[5, 5], trait_sd=[0.1, -0.1])


def test_trait_transect_zero_sd_with_n_gt_1_raises():
    with pytest.raises(ValueError, match="infinite weight"):
        TraitTransect(distance=[0, 10], trait_mean=[1.0, 2.0], n=[5, 5], trait_sd=[0.1, 0.0])


def test_trait_transect_zero_sd_with_n_eq_1_is_allowed():
    t = TraitTransect(distance=[0, 10], trait_mean=[1.0, 2.0], n=[1, 5], trait_sd=[0.0, 0.1])
    assert len(t) == 2


# ---------------------------------------------------------------------------
# mean_cline / variance_cline -- TailType.NONE boundary conditions,
# verified against HZAR's own R source formula directly
# ---------------------------------------------------------------------------

def test_variance_cline_boundary_conditions_match_hzar():
    v_center = variance_cline(np.array([50.0]), center=50, width=20,
                               var_l=1.0, var_r=4.0, var_h=2.0)
    assert v_center[0] == pytest.approx((1.0 + 4.0) / 2 + 2.0)

    v_far_left = variance_cline(np.array([-1000.0]), center=50, width=20,
                                 var_l=1.0, var_r=4.0, var_h=2.0)
    assert v_far_left[0] == pytest.approx(1.0, abs=1e-6)

    v_far_right = variance_cline(np.array([1000.0]), center=50, width=20,
                                  var_l=1.0, var_r=4.0, var_h=2.0)
    assert v_far_right[0] == pytest.approx(4.0, abs=1e-6)


def test_variance_cline_var_h_zero_reduces_to_simple_interpolation():
    distance = np.linspace(0, 100, 11)
    v = variance_cline(distance, center=50, width=20, var_l=1.0, var_r=1.0, var_h=0.0)
    np.testing.assert_allclose(v, 1.0, atol=1e-6)


def test_mean_cline_boundary_conditions():
    mu_far_left = mean_cline(np.array([-1000.0]), center=50, width=20, mu_l=2.0, mu_r=8.0)
    mu_far_right = mean_cline(np.array([1000.0]), center=50, width=20, mu_l=2.0, mu_r=8.0)
    mu_center = mean_cline(np.array([50.0]), center=50, width=20, mu_l=2.0, mu_r=8.0)
    assert mu_far_left[0] == pytest.approx(2.0, abs=1e-6)
    assert mu_far_right[0] == pytest.approx(8.0, abs=1e-6)
    assert mu_center[0] == pytest.approx(5.0)  # midpoint at the exact center


# ---------------------------------------------------------------------------
# Tail formula -- full HZAR parity, verified continuous at the tail/
# center boundary (the derivation this whole tail implementation depends
# on), and that tail edges use HZAR's own definition (center +/- delta,
# not center +/- width/2)
# ---------------------------------------------------------------------------

def test_hzar_tail_formula_is_continuous_with_center_formula_at_boundary():
    width, delta_l, tau_l, center = 20.0, 8.0, 0.6, 50.0
    edge = center - delta_l

    mu_at_edge_from_center_side = mean_cline(
        np.array([edge + 1e-6]), center, width, mu_l=2.0, mu_r=8.0, tail_type=TailType.NONE)
    mu_at_edge_from_tail_side = mean_cline(
        np.array([edge - 1e-6]), center, width, mu_l=2.0, mu_r=8.0, tail_type=TailType.LEFT,
        delta_l=delta_l, tau_l=tau_l)
    assert mu_at_edge_from_tail_side[0] == pytest.approx(mu_at_edge_from_center_side[0], abs=1e-4)

    var_at_edge_from_center_side = variance_cline(
        np.array([edge + 1e-6]), center, width, var_l=1.0, var_r=3.0, var_h=1.5,
        tail_type=TailType.NONE)
    var_at_edge_from_tail_side = variance_cline(
        np.array([edge - 1e-6]), center, width, var_l=1.0, var_r=3.0, var_h=1.5,
        tail_type=TailType.LEFT, delta_l=delta_l, tau_l=tau_l)
    assert var_at_edge_from_tail_side[0] == pytest.approx(var_at_edge_from_center_side[0], abs=1e-3)


def test_tail_edge_uses_hzar_own_definition_not_width_over_2():
    """HZAR's tail edge is `center - delta_l` (a raw distance), not
    `center - width/2` the way the (separately-parameterized, already-
    documented-as-not-HZAR-verified) allele-frequency cline_function's
    own tail model defines it -- confirmed by checking the tail formula
    only activates past the delta-defined edge, using a delta that
    differs from width/2."""
    center, width, delta_l = 50.0, 20.0, 3.0  # delta_l=3, NOT width/2=10
    edge = center - delta_l  # 47.0

    # just inside the edge (center side): should use the plain center formula
    mu_inside = mean_cline(np.array([edge + 0.5]), center, width, mu_l=1.0, mu_r=9.0,
                            tail_type=TailType.LEFT, delta_l=delta_l, tau_l=0.5)
    mu_inside_no_tail = mean_cline(np.array([edge + 0.5]), center, width, mu_l=1.0, mu_r=9.0,
                                    tail_type=TailType.NONE)
    assert mu_inside[0] == pytest.approx(mu_inside_no_tail[0])

    # just outside the edge (tail side): should differ from the plain center formula
    mu_outside = mean_cline(np.array([edge - 0.5]), center, width, mu_l=1.0, mu_r=9.0,
                             tail_type=TailType.LEFT, delta_l=delta_l, tau_l=0.5)
    mu_outside_no_tail = mean_cline(np.array([edge - 0.5]), center, width, mu_l=1.0, mu_r=9.0,
                                     tail_type=TailType.NONE)
    assert mu_outside[0] != pytest.approx(mu_outside_no_tail[0])


def test_tail_alpha_matches_hand_derivation():
    """Hand-derived from HZAR's own alphaExpF: alpha =
    exp(logitDelta*(tau-1))/(1+exp(-logitDelta)) where
    logitDelta = 4*delta/width."""
    delta, tau, width = 8.0, 0.6, 20.0
    logit_delta = 4 * delta / width
    expected = np.exp(logit_delta * (tau - 1)) / (1 + np.exp(-logit_delta))
    assert _tail_alpha(delta, tau, width) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# log_likelihood -- verified against summing raw individual observations'
# own log-densities, not just self-consistency
# ---------------------------------------------------------------------------

def test_log_likelihood_matches_raw_individual_data_likelihood():
    rng = np.random.default_rng(701)
    n_sites = 5
    distance = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    n_per_site = np.array([8, 12, 6, 10, 9])
    mean_params = ClineParams(center=50, width=30, pmin=1.0, pmax=5.0)
    var_l, var_r, var_h = 0.8, 1.5, 0.6

    mu_at_sites = mean_cline(distance, 50, 30, 1.0, 5.0, TailType.NONE)
    var_at_sites = variance_cline(distance, 50, 30, var_l, var_r, var_h)

    raw_ll = 0.0
    site_means, site_sds = [], []
    for i in range(n_sites):
        obs = rng.normal(mu_at_sites[i], np.sqrt(var_at_sites[i]), n_per_site[i])
        site_means.append(obs.mean())
        site_sds.append(obs.std())
        raw_ll += norm.logpdf(obs, loc=mu_at_sites[i], scale=np.sqrt(var_at_sites[i])).sum()

    transect = TraitTransect(distance=distance, trait_mean=np.array(site_means),
                              n=n_per_site.astype(float), trait_sd=np.array(site_sds))
    sufficient_stat_ll = log_likelihood(transect, mean_params, var_l, var_r, var_h, TailType.NONE)
    assert sufficient_stat_ll == pytest.approx(raw_ll, abs=1e-6)


def test_log_likelihood_requires_trait_sd():
    transect = TraitTransect(distance=[0, 10, 20], trait_mean=[1.0, 2.0, 3.0], n=[5, 5, 5])
    params = ClineParams(center=10, width=10, pmin=1.0, pmax=3.0)
    with pytest.raises(ValueError, match="requires transect.trait_sd"):
        log_likelihood(transect, params, 1.0, 1.0, 0.5)


# ---------------------------------------------------------------------------
# fit_mle_trait -- recovery against known ground truth, no-tail and tail
# ---------------------------------------------------------------------------

def test_fit_mle_trait_recovers_known_parameters_no_tail():
    true_mean_params = ClineParams(center=90.0, width=40.0, pmin=10.0, pmax=25.0)
    transect = _simulate_hzar(true_mean_params, 1.0, 2.5, 3.0, n_sites=30, seed=711)
    fit = fit_mle_trait(transect, tail_type=TailType.NONE)

    assert fit.success
    assert fit.n_params == 7  # center, width, pmin, pmax, var_l, var_r, var_h
    assert fit.mean_params.center == pytest.approx(90.0, abs=5)
    assert fit.mean_params.width == pytest.approx(40.0, abs=10)
    assert fit.mean_params.pmin == pytest.approx(10.0, abs=2)
    assert fit.mean_params.pmax == pytest.approx(25.0, abs=2)
    assert fit.var_l == pytest.approx(1.0, abs=1.0)
    assert fit.var_r == pytest.approx(2.5, abs=1.5)
    assert fit.var_h == pytest.approx(3.0, abs=2.5)


def test_fit_mle_trait_requires_trait_sd():
    transect = TraitTransect(distance=[0, 10, 20], trait_mean=[1.0, 2.0, 3.0], n=[5, 5, 5])
    with pytest.raises(ValueError, match="requires transect.trait_sd"):
        fit_mle_trait(transect)


def test_fit_mle_trait_recovers_asymmetric_variance():
    true_mean_params = ClineParams(center=60.0, width=20.0, pmin=5.0, pmax=20.0)
    transect = _simulate_hzar(true_mean_params, 0.3, 3.0, 0.5, n_sites=35, seed=713)
    fit = fit_mle_trait(transect, tail_type=TailType.NONE)
    assert fit.var_r > fit.var_l


def test_fit_mle_trait_handles_decreasing_and_negative_trait_values():
    true_mean_params = ClineParams(center=50.0, width=15.0, pmin=20.0, pmax=-30.0)
    transect = _simulate_hzar(true_mean_params, 0.5, 0.5, 0.3, n_sites=25, seed=505)
    fit = fit_mle_trait(transect, tail_type=TailType.NONE)
    assert fit.mean_params.pmin > fit.mean_params.pmax
    assert fit.mean_params.pmax < 0


def test_fit_mle_trait_left_tail_recovers_known_parameters():
    true_mean_params = ClineParams(center=100.0, width=30.0, pmin=10.0, pmax=25.0,
                                    delta_l=15.0, tau_l=0.5)
    transect = _simulate_hzar(true_mean_params, 1.0, 2.0, 1.5, n_sites=40, seed=801,
                               tail_type=TailType.LEFT)
    fit = fit_mle_trait(transect, tail_type=TailType.LEFT)

    assert fit.success
    assert fit.n_params == 9  # center, width, pmin, pmax, delta_l, tau_l, var_l, var_r, var_h
    assert fit.mean_params.center == pytest.approx(100.0, abs=5)
    assert fit.mean_params.width == pytest.approx(30.0, abs=10)
    assert fit.var_l == pytest.approx(1.0, abs=1.0)
    assert fit.var_r == pytest.approx(2.0, abs=1.0)
    # delta_l/tau_l themselves are not checked against truth here -- tail
    # parameters are known to be weakly identified even in the allele-
    # frequency case (BACKEND_PHASE2.md section 12.3); the important
    # check is that fitting the tail model doesn't degrade recovery of
    # the core (non-tail) parameters above, which it doesn't.


def test_fit_mle_trait_both_tail_runs_without_error():
    true_mean_params = ClineParams(center=100.0, width=30.0, pmin=10.0, pmax=25.0,
                                    delta_l=15.0, tau_l=0.5, delta_r=12.0, tau_r=0.4)
    transect = _simulate_hzar(true_mean_params, 1.0, 2.0, 1.0, n_sites=35, seed=803,
                               tail_type=TailType.BOTH)
    fit = fit_mle_trait(transect, tail_type=TailType.BOTH)
    assert fit.n_params == 11
    assert fit.mean_params.center == pytest.approx(100.0, abs=10)


def test_fit_mle_trait_mirror_tail_runs_without_error():
    true_mean_params = ClineParams(center=100.0, width=30.0, pmin=10.0, pmax=25.0,
                                    delta_l=15.0, tau_l=0.5)
    transect = _simulate_hzar(true_mean_params, 1.0, 2.0, 1.0, n_sites=35, seed=803,
                               tail_type=TailType.MIRROR)
    fit = fit_mle_trait(transect, tail_type=TailType.MIRROR)
    assert fit.n_params == 9
    assert fit.mean_params.center == pytest.approx(100.0, abs=10)


def test_fit_mle_trait_summary_runs_without_error():
    true_mean_params = ClineParams(center=50.0, width=20.0, pmin=5.0, pmax=15.0)
    transect = _simulate_hzar(true_mean_params, 1.0, 1.5, 0.5, n_sites=20, seed=3)
    fit = fit_mle_trait(transect)
    s = fit.summary()
    assert "var_l" in s
    assert "var_h" in s
    assert "AICc" in s


# ---------------------------------------------------------------------------
# compare_trait_models
# ---------------------------------------------------------------------------

def test_compare_trait_models_selects_true_no_tail_model():
    from hycline.selection.aicc import compare_trait_models

    true_mean_params = ClineParams(center=80.0, width=30.0, pmin=5.0, pmax=20.0)
    transect = _simulate_hzar(true_mean_params, 1.0, 1.2, 0.8, n_sites=30, seed=531)
    comparison = compare_trait_models(transect)
    assert comparison.best_model == TailType.NONE
