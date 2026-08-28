"""Tests for the candidate-flagging overlay plots in hycline.viz.plotting
(plot_geographic_candidates_overlay/plot_genomic_candidates_overlay): one
fitted-curve line per locus, colored by Fitzpatrick-style candidate
category with non-candidates in light grey. New functions, not a
regression pass, so these check the actual contract (fixed category
color assignment, legend order, non-candidates rendered distinctly)
rather than just "it runs"."""

import matplotlib
matplotlib.use("Agg")  # headless test environment
import matplotlib.colors as mcolors
import numpy as np
import pytest

from hycline.models.cline import ClineParams
from hycline.models.genomic_cline import GenomicClineParams
from hycline.viz.plotting import (
    plot_geographic_candidates_overlay, plot_genomic_candidates_overlay,
    _CANDIDATE_CATEGORY_COLORS, _CANDIDATE_NOT_FLAGGED_COLOR,
)


def _hex(color):
    return mcolors.to_hex(color)


def test_geographic_candidates_overlay_raises_clear_error_on_empty_locus_params():
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_geographic_candidates_overlay({}, {})


def test_genomic_candidates_overlay_raises_clear_error_on_empty_locus_params():
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_genomic_candidates_overlay({}, {})


def _geographic_fixture():
    locus_params = {
        "restricted": ClineParams(center=50.0, width=5.0, pmin=0.05, pmax=0.95),
        "elevated": ClineParams(center=50.0, width=40.0, pmin=0.05, pmax=0.95),
        "asymmetric": ClineParams(center=80.0, width=20.0, pmin=0.05, pmax=0.95),
        "ambiguous": ClineParams(center=20.0, width=3.0, pmin=0.05, pmax=0.95),
        "not_flagged_1": ClineParams(center=50.0, width=20.0, pmin=0.05, pmax=0.95),
        "not_flagged_2": ClineParams(center=52.0, width=22.0, pmin=0.05, pmax=0.95),
    }
    category_by_locus = {
        "restricted": "restricted_gene_flow",
        "elevated": "elevated_gene_flow",
        "asymmetric": "asymmetric_gene_flow",
        "ambiguous": "ambiguous",
        "not_flagged_1": None,
        "not_flagged_2": None,
    }
    return locus_params, category_by_locus


def test_geographic_candidates_overlay_assigns_fixed_category_colors():
    locus_params, category_by_locus = _geographic_fixture()
    fig = plot_geographic_candidates_overlay(locus_params, category_by_locus)
    ax = fig.axes[0]

    # one line per locus, colored by its own category (not-flagged loci
    # drawn first/grey -- see the color-by-line-count check below).
    assert len(ax.lines) == len(locus_params)
    colors_seen = {_hex(line.get_color()) for line in ax.lines}
    expected_colors = set(_CANDIDATE_CATEGORY_COLORS.values()) | {_CANDIDATE_NOT_FLAGGED_COLOR}
    assert colors_seen == expected_colors

    # not-flagged loci (drawn first) are grey; flagged ones (drawn after)
    # use their category's fixed color, not a cycled/arbitrary one.
    n_not_flagged = sum(1 for c in category_by_locus.values() if c is None)
    for line in ax.lines[:n_not_flagged]:
        assert _hex(line.get_color()) == _hex(_CANDIDATE_NOT_FLAGGED_COLOR)
    for line in ax.lines[n_not_flagged:]:
        assert _hex(line.get_color()) in {_hex(c) for c in _CANDIDATE_CATEGORY_COLORS.values()}


def test_geographic_candidates_overlay_legend_uses_fixed_category_order():
    locus_params, category_by_locus = _geographic_fixture()
    fig = plot_geographic_candidates_overlay(locus_params, category_by_locus)
    ax = fig.axes[0]

    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # fixed order: not-flagged first, then restricted/elevated/asymmetric/
    # ambiguous -- regardless of the order categories appear in the data
    # (locus_params/category_by_locus above are already in that order,
    # so this also guards against a future reordering of the fixture
    # silently stopping this test from checking anything).
    assert legend_labels == [
        "Not flagged", "Restricted gene flow", "Elevated gene flow",
        "Asymmetric gene flow", "Ambiguous",
    ]


def test_geographic_candidates_overlay_omits_absent_categories_from_legend():
    locus_params = {
        "a": ClineParams(center=50.0, width=20.0, pmin=0.05, pmax=0.95),
        "b": ClineParams(center=50.0, width=5.0, pmin=0.05, pmax=0.95),
    }
    category_by_locus = {"a": None, "b": "restricted_gene_flow"}
    fig = plot_geographic_candidates_overlay(locus_params, category_by_locus)
    legend_labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend_labels == ["Not flagged", "Restricted gene flow"]


def test_geographic_candidates_overlay_default_distance_range_covers_all_loci():
    locus_params = {
        "left": ClineParams(center=0.0, width=10.0, pmin=0.05, pmax=0.95),
        "right": ClineParams(center=200.0, width=10.0, pmin=0.05, pmax=0.95),
    }
    category_by_locus = {"left": None, "right": None}
    fig = plot_geographic_candidates_overlay(locus_params, category_by_locus)
    ax = fig.axes[0]
    all_x = np.concatenate([line.get_xdata() for line in ax.lines])
    # center +/- 3*width for the widest-spanning loci: left locus needs
    # x down to -30, right locus needs x up to 230.
    assert all_x.min() <= -25
    assert all_x.max() >= 225


def test_geographic_candidates_overlay_explicit_distance_range_respected():
    locus_params = {"a": ClineParams(center=50.0, width=20.0, pmin=0.05, pmax=0.95)}
    category_by_locus = {"a": None}
    fig = plot_geographic_candidates_overlay(
        locus_params, category_by_locus, distance_range=(0.0, 100.0))
    ax = fig.axes[0]
    x = ax.lines[0].get_xdata()
    assert x.min() == pytest.approx(0.0)
    assert x.max() == pytest.approx(100.0)


def _genomic_fixture():
    locus_params = {
        "restricted": GenomicClineParams(u=0.0, v=5.0),
        "elevated": GenomicClineParams(u=0.0, v=0.2),
        "asymmetric": GenomicClineParams(u=3.0, v=1.0),
        "not_flagged": GenomicClineParams(u=0.0, v=1.0),
    }
    category_by_locus = {
        "restricted": "restricted_gene_flow",
        "elevated": "elevated_gene_flow",
        "asymmetric": "asymmetric_gene_flow",
        "not_flagged": None,
    }
    return locus_params, category_by_locus


def test_genomic_candidates_overlay_assigns_fixed_category_colors():
    locus_params, category_by_locus = _genomic_fixture()
    fig = plot_genomic_candidates_overlay(locus_params, category_by_locus)
    ax = fig.axes[0]

    # one line per locus, plus the dashed neutral-null reference line.
    curve_lines = [line for line in ax.lines if line.get_linestyle() not in ("--", (0, (6.4, 1.6)))]
    assert len(curve_lines) == len(locus_params)

    colors_seen = {_hex(line.get_color()) for line in curve_lines}
    expected_colors = {_CANDIDATE_CATEGORY_COLORS["restricted_gene_flow"],
                        _CANDIDATE_CATEGORY_COLORS["elevated_gene_flow"],
                        _CANDIDATE_CATEGORY_COLORS["asymmetric_gene_flow"],
                        _CANDIDATE_NOT_FLAGGED_COLOR}
    assert colors_seen == {_hex(c) for c in expected_colors}


def test_genomic_candidates_overlay_includes_neutral_reference_line():
    locus_params, category_by_locus = _genomic_fixture()
    fig = plot_genomic_candidates_overlay(locus_params, category_by_locus)
    ax = fig.axes[0]
    dashed = [line for line in ax.lines if line.get_linestyle() in ("--", (0, (6.4, 1.6)))]
    assert len(dashed) == 1
    x, y = dashed[0].get_xdata(), dashed[0].get_ydata()
    np.testing.assert_allclose(x, [0, 1])
    np.testing.assert_allclose(y, [0, 1])
    assert ax.get_xlim()[0] < 0 < 1 < ax.get_xlim()[1]  # matches the -0.02/1.02 grid convention
