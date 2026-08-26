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
