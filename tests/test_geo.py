import numpy as np
import pytest

from hycline.data.geo import (
    haversine_distance, distance_from_reference, project_onto_transect_axis, EARTH_RADIUS_KM,
)


def test_haversine_known_distance_nyc_to_la():
    # Well-known approximate great-circle distance, commonly cited as ~3936 km
    d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert d == pytest.approx(3936, rel=0.01)


def test_haversine_zero_distance_for_identical_points():
    d = haversine_distance(10.0, 20.0, 10.0, 20.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_latitude_is_about_111km():
    d = haversine_distance(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111.19, rel=0.01)


def test_distance_from_reference_defaults_to_first_site():
    lat = np.array([10.0, 10.5, 11.0])
    lon = np.array([20.0, 20.0, 20.0])
    d = distance_from_reference(lat, lon)
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(
        d, haversine_distance(lat[0], lon[0], lat, lon)
    )


def test_distance_from_reference_explicit_reference():
    lat = np.array([10.0, 10.5, 11.0])
    lon = np.array([20.0, 20.0, 20.0])
    d = distance_from_reference(lat, lon, ref_lat=10.5, ref_lon=20.0)
    assert d[1] == pytest.approx(0.0, abs=1e-9)


def test_project_onto_transect_axis_preserves_relative_spacing_on_a_line():
    # points evenly spaced along a north-south line -- spacing should be
    # recovered (up to sign/direction, which PCA doesn't fix a priori)
    lat = np.array([10.0, 10.25, 10.5, 10.75, 11.0])
    lon = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    proj = project_onto_transect_axis(lat, lon)
    spacing = np.diff(np.sort(proj))
    np.testing.assert_allclose(spacing, spacing[0], rtol=1e-6)
    assert proj.min() == pytest.approx(0.0, abs=1e-9)


def test_project_onto_transect_axis_matches_reference_distance_on_straight_line():
    """On a perfectly straight transect, PCA projection and
    distance-from-first-site should give the same spacing (possibly
    reversed direction, which is fine -- cline center/width are direction-
    agnostic)."""
    lat = np.array([10.0, 10.2, 10.4, 10.6])
    lon = np.array([20.0, 20.0, 20.0, 20.0])
    proj = project_onto_transect_axis(lat, lon)
    ref = distance_from_reference(lat, lon)
    proj_sorted_diffs = np.diff(np.sort(proj))
    ref_sorted_diffs = np.diff(np.sort(ref))
    np.testing.assert_allclose(proj_sorted_diffs, ref_sorted_diffs, rtol=1e-6)


def test_check_site_coordinate_consistency_passes_when_consistent():
    from hycline.data.geo import check_site_coordinate_consistency
    check_site_coordinate_consistency(["S1", "S1", "S2"], [10.0, 10.0, 20.0], [5.0, 5.0, 6.0])


def test_check_site_coordinate_consistency_raises_when_inconsistent():
    from hycline.data.geo import check_site_coordinate_consistency
    with pytest.raises(ValueError, match="inconsistent coordinates"):
        check_site_coordinate_consistency(["S1", "S1"], [10.0, 10.1], [5.0, 5.0])
