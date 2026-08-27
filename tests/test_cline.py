import numpy as np
import pytest

from hycline.models.cline import ClineParams, TailType, cline_function


def test_cline_midpoint_is_average():
    """At x = center, the sigmoid should sit exactly halfway between pmin and pmax."""
    params = ClineParams(center=10.0, width=4.0, pmin=0.1, pmax=0.9)
    val = cline_function(np.array([10.0]), params, TailType.NONE)
    assert val[0] == pytest.approx((0.1 + 0.9) / 2, abs=1e-9)


def test_cline_asymptotes():
    """Far from the center, the cline should approach pmin/pmax."""
    params = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0)
    x = np.array([-1000.0, 1000.0])
    val = cline_function(x, params, TailType.NONE)
    assert val[0] == pytest.approx(0.0, abs=1e-6)
    assert val[1] == pytest.approx(1.0, abs=1e-6)


def test_cline_is_monotonic_no_tails():
    params = ClineParams(center=5.0, width=3.0, pmin=0.1, pmax=0.8)
    x = np.linspace(-20, 30, 200)
    y = cline_function(x, params, TailType.NONE)
    assert np.all(np.diff(y) >= -1e-12)  # non-decreasing (pmax > pmin here)


def test_cline_handles_reversed_direction():
    """pmin > pmax should give a monotonically decreasing cline."""
    params = ClineParams(center=5.0, width=3.0, pmin=0.9, pmax=0.1)
    x = np.linspace(-20, 30, 200)
    y = cline_function(x, params, TailType.NONE)
    assert np.all(np.diff(y) <= 1e-12)


# ---------------------------------------------------------------------------
# hzar_cline_function -- HZAR's own tail parameterization, verified
# directly against HZAR's R source (BACKEND_TRAIT.md sections 1-2), a
# real, different alternative to cline_function's own tail model
# ---------------------------------------------------------------------------

def test_hzar_cline_function_no_tail_identical_to_cline_function():
    params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9)
    x = np.linspace(-50, 150, 50)
    from hycline.models.cline import hzar_cline_function
    np.testing.assert_allclose(
        cline_function(x, params, TailType.NONE),
        hzar_cline_function(x, params, TailType.NONE),
    )


def test_hzar_cline_function_matches_trait_mean_cline_exactly():
    """The core correctness guarantee: hzar_cline_function must be
    identical to hycline.likelihoods.gaussian.mean_cline (the formula
    this was ported from, already verified against HZAR's real R
    source) -- not a fresh, potentially-drifted reimplementation."""
    from hycline.models.cline import hzar_cline_function
    from hycline.likelihoods.gaussian import mean_cline

    params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9, delta_l=8.0, tau_l=0.5)
    x = np.linspace(-50, 150, 60)
    result = hzar_cline_function(x, params, TailType.LEFT)
    expected = mean_cline(x, params.center, params.width, params.pmin, params.pmax,
                           TailType.LEFT, params.delta_l, params.tau_l)
    np.testing.assert_allclose(result, expected)


def test_hzar_cline_function_continuous_at_tail_edge():
    """HZAR's tail edge is center - delta_l (a raw distance), and the
    tail formula must be continuous with the sigmoid core there --
    verified directly, the same check already performed for the
    trait-cline version this was ported from."""
    from hycline.models.cline import hzar_cline_function

    center, width, delta_l, tau_l = 50.0, 20.0, 8.0, 0.6
    edge = center - delta_l
    params = ClineParams(center=center, width=width, pmin=0.1, pmax=0.9,
                          delta_l=delta_l, tau_l=tau_l)

    val_center_side = hzar_cline_function(np.array([edge + 1e-6]), params, TailType.NONE)
    val_tail_side = hzar_cline_function(np.array([edge - 1e-6]), params, TailType.LEFT)
    assert val_tail_side[0] == pytest.approx(val_center_side[0], abs=1e-4)


def test_hzar_cline_function_tail_edge_is_delta_not_width_over_2():
    """A real, named difference from cline_function's own tail model:
    the tail only activates past center-delta_l, not center-width/2."""
    from hycline.models.cline import hzar_cline_function

    center, width, delta_l = 50.0, 20.0, 3.0  # delta_l=3, NOT width/2=10
    params = ClineParams(center=center, width=width, pmin=0.1, pmax=0.9,
                          delta_l=delta_l, tau_l=0.5)
    edge = center - delta_l

    val_inside = hzar_cline_function(np.array([edge + 0.5]), params, TailType.LEFT)
    val_inside_none = hzar_cline_function(np.array([edge + 0.5]), params, TailType.NONE)
    assert val_inside[0] == pytest.approx(val_inside_none[0])

    val_outside = hzar_cline_function(np.array([edge - 0.5]), params, TailType.LEFT)
    val_outside_none = hzar_cline_function(np.array([edge - 0.5]), params, TailType.NONE)
    assert val_outside[0] != pytest.approx(val_outside_none[0])


def test_hzar_cline_function_asymptotes():
    params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9,
                          delta_l=8.0, tau_l=0.5, delta_r=6.0, tau_r=0.4)
    from hycline.models.cline import hzar_cline_function
    val_far_left = hzar_cline_function(np.array([-1e6]), params, TailType.BOTH)
    val_far_right = hzar_cline_function(np.array([1e6]), params, TailType.BOTH)
    assert val_far_left[0] == pytest.approx(0.1, abs=1e-6)
    assert val_far_right[0] == pytest.approx(0.9, abs=1e-6)


def test_hzar_cline_function_mirror_reuses_left_params_for_right_tail():
    from hycline.models.cline import hzar_cline_function
    params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9, delta_l=8.0, tau_l=0.5)
    x = np.array([70.0])  # in the right-tail region for delta_l=8 (edge at 58)
    val_mirror = hzar_cline_function(x, params, TailType.MIRROR)
    # a directly-constructed "right" using the same delta_l/tau_l values as the
    # mirror's shared parameters should match
    params_as_right = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9,
                                   delta_r=8.0, tau_r=0.5)
    val_right = hzar_cline_function(x, params_as_right, TailType.RIGHT)
    assert val_mirror[0] == pytest.approx(val_right[0])


# ---------------------------------------------------------------------------
# t_cline_function -- Baird & Daley (2025)'s Student's-t-based continuous
# alternative to the tripartite tail model, symmetric case only. Each test
# below checks a specific, named calibration fact stated directly in the
# paper's own Table 1, not an arbitrary property.
# ---------------------------------------------------------------------------

def test_t_cline_shape_1_is_exactly_cauchy():
    """Table 1: T(1) = Cauchy, exactly (not an approximation)."""
    from hycline.models.cline import t_cline_function
    params = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0, t_shape=1.0)
    x = np.linspace(-8, 8, 41)
    got = t_cline_function(x, params)
    cauchy = 0.5 + np.arctan(np.pi * x) / np.pi
    np.testing.assert_allclose(got, cauchy, atol=1e-10)


def test_t_cline_shape_6_67_matches_logistic_within_paper_stated_mad():
    """Table 1: Bazykin (the ordinary logistic sigmoid) ~= T(6.67), MAD 0.001.

    Tolerance is 0.0011, not the paper's own stated 0.001, exactly. "6.67"
    is itself a rounded 3-significant-figure value in the paper's table --
    at that rounding, the actual MAD comes out to 0.0010059 (checked
    directly), a hair over their reported figure purely from that input
    rounding, not a sign of a wrong implementation -- already confirmed
    independently by the exact Cauchy match and the width-convention
    finite-difference check elsewhere in this file."""
    from hycline.models.cline import t_cline_function
    params_t = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0, t_shape=6.67)
    params_logistic = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0)
    x = np.linspace(-4, 4, 801)  # paper's own MAD comparison window, +/-4 widths
    t_vals = t_cline_function(x, params_t)
    logistic_vals = cline_function(x, params_logistic, TailType.NONE)
    assert np.max(np.abs(t_vals - logistic_vals)) < 0.0011


def test_t_cline_large_shape_matches_normal_within_paper_stated_mad():
    """Table 1: Normal ~= T(116), MAD 0.001."""
    from scipy.stats import norm
    from hycline.models.cline import t_cline_function
    params = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0, t_shape=116.0)
    x = np.linspace(-4, 4, 801)
    t_vals = t_cline_function(x, params)
    # Normal unit cline: same width convention (max gradient = 1), so scale
    # by the Normal PDF's own peak density (1/sqrt(2*pi)) the same way
    # _t_cline_scale does for the Student's-t case.
    normal_vals = norm.cdf(x * np.sqrt(2 * np.pi))
    assert np.max(np.abs(t_vals - normal_vals)) < 0.001


def test_t_cline_width_convention_matches_max_gradient_inverse():
    """This project's width convention ("width" = 1/max gradient) must hold
    for the T-cline exactly as it does for the logistic core, across a wide
    range of shape and width values -- not just asserted, checked by finite
    difference."""
    from hycline.models.cline import t_cline_function
    eps = 1e-4
    for t_shape in (0.5, 1.0, 2.0, 6.67, 30.0, 200.0):
        for width in (1.0, 10.0, 37.5):
            params = ClineParams(center=20.0, width=width, pmin=0.0, pmax=1.0, t_shape=t_shape)
            y = t_cline_function(np.array([20.0 - eps, 20.0 + eps]), params)
            grad = (y[1] - y[0]) / (2 * eps)
            assert grad * width == pytest.approx(1.0, rel=1e-4)


def test_t_cline_is_monotonic_across_steepness_range():
    from hycline.models.cline import t_cline_function
    x = np.linspace(-50, 50, 2001)
    for t_shape in (0.06, 0.5, 1.0, 6.67, 500.0):
        params = ClineParams(center=0.0, width=2.0, pmin=0.1, pmax=0.9, t_shape=t_shape)
        y = t_cline_function(x, params)
        assert np.all(np.diff(y) >= -1e-12)
        assert not np.any(np.isnan(y))


def test_t_cline_asymptotes():
    from hycline.models.cline import t_cline_function
    params = ClineParams(center=0.0, width=5.0, pmin=0.05, pmax=0.95, t_shape=3.0)
    val = t_cline_function(np.array([-1e6, 1e6]), params)
    assert val[0] == pytest.approx(0.05, abs=1e-6)
    assert val[1] == pytest.approx(0.95, abs=1e-6)


def test_t_cline_requires_positive_t_shape():
    from hycline.models.cline import t_cline_function
    params = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0, t_shape=None)
    with pytest.raises(ValueError):
        t_cline_function(np.array([0.0]), params)
    params.t_shape = -1.0
    with pytest.raises(ValueError):
        t_cline_function(np.array([0.0]), params)


def test_cline_function_rejects_t_cline_tail_type():
    """cline_function is not the entry point for T-cline evaluation --
    should fail clearly rather than mishandle TailType.T_CLINE."""
    params = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0, t_shape=2.0)
    with pytest.raises(ValueError):
        cline_function(np.array([0.0]), params, TailType.T_CLINE)


def test_hzar_cline_function_rejects_t_cline_tail_type():
    from hycline.models.cline import hzar_cline_function
    params = ClineParams(center=0.0, width=1.0, pmin=0.0, pmax=1.0, t_shape=2.0)
    with pytest.raises(ValueError):
        hzar_cline_function(np.array([0.0]), params, TailType.T_CLINE)


def test_t_cline_param_names_and_array_roundtrip():
    params = ClineParams(center=12.0, width=8.0, pmin=0.1, pmax=0.9, t_shape=4.5)
    vec = params.as_array(TailType.T_CLINE)
    from hycline.models.cline import param_names
    assert param_names(TailType.T_CLINE) == ["center", "width", "pmin", "pmax", "t_shape"]
    assert list(vec) == [12.0, 8.0, 0.1, 0.9, 4.5]
    roundtrip = ClineParams.from_array(vec, TailType.T_CLINE)
    assert roundtrip.t_shape == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# t_cline_function, asymmetric case (params.t_skew set) -- T(v, alpha) in
# Baird & Daley (2025)'s own notation.
# ---------------------------------------------------------------------------

def test_t_cline_asymmetric_param_names_and_array_roundtrip():
    from hycline.models.cline import param_names
    params = ClineParams(center=12.0, width=8.0, pmin=0.1, pmax=0.9, t_shape=4.5, t_skew=1.2)
    vec = params.as_array(TailType.T_CLINE)
    assert param_names(TailType.T_CLINE, asymmetric=True) == \
        ["center", "width", "pmin", "pmax", "t_shape", "t_skew"]
    assert list(vec) == [12.0, 8.0, 0.1, 0.9, 4.5, 1.2]
    roundtrip = ClineParams.from_array(vec, TailType.T_CLINE)
    assert roundtrip.t_shape == pytest.approx(4.5)
    assert roundtrip.t_skew == pytest.approx(1.2)
    # symmetric array (5 elements) round-trips to t_skew=None, not 0.0 --
    # from_array infers presence from vector length, not value
    sym_vec = ClineParams(center=1.0, width=2.0, t_shape=3.0).as_array(TailType.T_CLINE)
    assert len(sym_vec) == 5
    assert ClineParams.from_array(sym_vec, TailType.T_CLINE).t_skew is None


def test_t_cline_asymmetric_t_skew_zero_matches_symmetric_exactly():
    """t_skew=0.0 (explicitly asymmetric-mode, not None) must reproduce
    the symmetric path's output exactly -- confirms the asymmetric branch
    is a genuine generalization, not an independently-computed approximation
    that merely happens to agree near t_skew=0."""
    x = np.linspace(-40, 50, 200)
    for v in [0.5, 1.0, 6.67, 30.0, 100.0]:
        p_sym = ClineParams(center=5.0, width=8.0, pmin=0.1, pmax=0.9, t_shape=v, t_skew=None)
        p_asym0 = ClineParams(center=5.0, width=8.0, pmin=0.1, pmax=0.9, t_shape=v, t_skew=0.0)
        from hycline.models.cline import t_cline_function
        np.testing.assert_allclose(t_cline_function(x, p_sym), t_cline_function(x, p_asym0),
                                    atol=0)


def test_t_cline_asymmetric_width_convention_holds():
    """Even with asymmetry, "width" must still mean the inverse of the
    maximum gradient, exactly as it does for every other cline model in
    this project -- checked by finite difference, not assumed."""
    from hycline.models.cline import t_cline_function
    eps = 1e-4
    for v in [0.5, 1.0, 6.67, 100.0, 500.0]:
        for alpha in [-8.0, -1.0, 1.0, 5.0]:
            for width in [1.0, 10.0, 37.5]:
                p = ClineParams(center=20.0, width=width, pmin=0.0, pmax=1.0,
                                 t_shape=v, t_skew=alpha)
                y = t_cline_function(np.array([20.0 - eps, 20.0 + eps]), p)
                grad = (y[1] - y[0]) / (2 * eps)
                assert grad * width == pytest.approx(1.0, rel=1e-3)


def test_t_cline_asymmetric_monotonic_and_bounded_across_wide_parameter_range():
    """Regression test for a real bug found during development: a naive
    implementation returns NaN from scipy.stats.nct.cdf itself deep in its
    own extreme tail (confirmed directly at t_shape=100, t_skew=10 -- NOT
    an astronomically extreme case), which would silently propagate into
    an entire fit's log-likelihood if left unhandled. Sweeps a wide
    (t_shape, t_skew) grid over a generously wide x-range specifically to
    catch this."""
    from hycline.models.cline import t_cline_function
    x = np.linspace(-100, 100, 2001)
    for v in [0.5, 1.0, 2.0, 6.67, 30.0, 100.0, 500.0]:
        for alpha in [-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0]:
            p = ClineParams(center=0.0, width=3.0, pmin=0.05, pmax=0.95, t_shape=v, t_skew=alpha)
            y = t_cline_function(x, p)
            assert not np.any(np.isnan(y)), f"NaN at t_shape={v}, t_skew={alpha}"
            assert np.all(np.diff(y) >= -1e-9), f"non-monotonic at t_shape={v}, t_skew={alpha}"
            assert np.all((y >= 0.05 - 1e-6) & (y <= 0.95 + 1e-6)), \
                f"out of [pmin,pmax] at t_shape={v}, t_skew={alpha}"


def test_t_cline_asymmetric_matches_scipy_nct_cdf_directly():
    """Sanity-checks the asymmetric formula against a direct, independent
    scipy.stats.nct.cdf evaluation at the cline's own center (where the
    formula's mode/peak correction should place the argument exactly at
    the noncentral-t distribution's own mode)."""
    from scipy import stats as _stats
    from hycline.models.cline import t_cline_function, _t_cline_asymmetric_mode_and_peak
    v, alpha = 5.0, 2.0
    mode, peak = _t_cline_asymmetric_mode_and_peak(v, alpha)
    p = ClineParams(center=10.0, width=4.0, pmin=0.0, pmax=1.0, t_shape=v, t_skew=alpha)
    got = t_cline_function(np.array([10.0]), p)[0]
    expected = _stats.nct.cdf(mode, df=v, nc=alpha)
    assert got == pytest.approx(expected, abs=1e-9)
