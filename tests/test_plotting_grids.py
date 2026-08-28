"""Regression tests for the small-multiples grid-plot functions in
hycline.viz.plotting -- real bugs found during a code-review pass, not
general coverage (end-to-end rendering of these functions is already
exercised indirectly by the CLI test suite via --plot-grid)."""

import warnings

import matplotlib
matplotlib.use("Agg")  # headless test environment
import numpy as np
import pandas as pd
import pytest

from hycline.viz.plotting import (
    plot_multi_locus_grid, plot_trait_multi_grid, plot_genomic_multi_locus_grid,
    plot_hierarchical_geographic_grid, plot_hierarchical_genomic_grid,
    plot_gl_multi_locus_grid, plot_genomic_gl_multi_locus_grid,
)

_DICT_GRID_FUNCTIONS = [
    plot_multi_locus_grid, plot_trait_multi_grid, plot_genomic_multi_locus_grid,
    plot_gl_multi_locus_grid, plot_genomic_gl_multi_locus_grid,
]


@pytest.mark.parametrize("fn", _DICT_GRID_FUNCTIONS, ids=lambda f: f.__name__)
def test_grid_functions_raise_clear_error_on_empty_results(fn):
    # Regression test: all 7 grid functions used to crash with
    # `ValueError: Number of rows must be a positive integer, not 0`
    # (matplotlib's own unhelpful message) when called with an empty
    # results dict -- a realistic case: a batch fit where every locus
    # failed. Now raises a clear, purpose-built error instead.
    with pytest.raises(ValueError, match="Nothing to plot"):
        fn({})


def test_hierarchical_geographic_grid_raises_clear_error_on_empty_locus_list():
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_hierarchical_geographic_grid(
            np.array([0., 10.]), np.zeros((0, 2)), np.zeros((0, 2)),
            [], pd.DataFrame(), [], [])


def test_hierarchical_genomic_grid_raises_clear_error_on_empty_locus_list():
    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_hierarchical_genomic_grid(
            np.array([0.1, 0.9]), np.zeros((0, 2)), np.zeros((0, 2)),
            [], pd.DataFrame())


def test_hierarchical_geographic_grid_handles_zero_coverage_locus_without_warning():
    # Regression test: plot_hierarchical_geographic_grid divided raw,
    # unvalidated n_alleles arrays with no zero-guard, producing a
    # RuntimeWarning ("invalid value encountered in divide") and a
    # meaningless plotted point when any locus has zero coverage at a
    # site -- a realistic case for real, sparse genomic data.
    distance = np.array([0., 10., 20., 30., 40.])
    n_alleles = np.array([[10., 10., 10., 10., 10.],
                           [8., 8., 0., 8., 8.]])
    n_focal = np.array([[5., 6., 7., 8., 9.],
                         [3., 4., 0., 5., 6.]])
    locus_names = ["A", "B"]
    summary_df = pd.DataFrame({"locus": ["A", "B"], "center_mean": [20., 20.],
                                "width_mean": [10., 10.]})
    pmin = [0.1, 0.1]
    pmax = [0.9, 0.9]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_hierarchical_geographic_grid(
            distance, n_alleles, n_focal, locus_names, summary_df, pmin, pmax)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings
    assert len(fig.axes) == 2
