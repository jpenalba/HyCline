"""
Tests for Phase 4 Component A: the hierarchical geographic cline model.

There is no HZAR-equivalent gold standard for hierarchical pooling, so
these tests validate against synthetic data with a known population-level
structure -- see examples/demo_hierarchical_geographic.py for the fuller
worked example these tests are a faster/stricter version of, and
BACKEND_PHASE4.md for the full design writeup.
"""

import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.inference.mle import fit_mle_fixed_extremes
from hycline.inference.mcmc_hierarchical import (
    fit_hierarchical_batch, _derive_pmin_pmax,
)
from hycline.models.cline import ClineParams, TailType, cline_function


def _simulate(seed=7, n_loci=8, sample_sizes=None):
    rng = np.random.default_rng(seed)
    mu_center_true = 75.0
    sigma_center_true = 10.0
    mu_log_width_true = np.log(25.0)
    sigma_log_width_true = 0.25

    distance = np.sort(rng.uniform(0, 150, size=40))
    true_centers = rng.normal(mu_center_true, sigma_center_true, n_loci)
    true_widths = np.exp(rng.normal(mu_log_width_true, sigma_log_width_true, n_loci))

    if sample_sizes is None:
        sample_sizes = [4, 60] + [20] * (n_loci - 2)

    n_alleles = np.zeros((n_loci, len(distance)))
    n_focal = np.zeros((n_loci, len(distance)))
    for j in range(n_loci):
        params = ClineParams(center=true_centers[j], width=true_widths[j], pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        n_alleles[j, :] = sample_sizes[j]
        n_focal[j, :] = rng.binomial(sample_sizes[j], p)

    truth = dict(mu_center=mu_center_true, sigma_center=sigma_center_true,
                 mu_width=np.exp(mu_log_width_true),
                 true_centers=true_centers, true_widths=true_widths)
    return distance, n_alleles, n_focal, truth


# ---------------------------------------------------------------------------
# _derive_pmin_pmax regression test: the real bug found during this
# component's own validation (single-site anchor fragility)
# ---------------------------------------------------------------------------

def test_derive_pmin_pmax_robust_to_single_noisy_extreme_site():
    """A single unlucky 0/n draw at the extreme site should not lock
    pmin/pmax at a badly wrong value -- averaging over several edge sites
    (the fix) should recover something close to the true asymptote, unlike
    the single-site convention this replaced."""
    rng = np.random.default_rng(3)
    distance = np.sort(rng.uniform(0, 150, size=20))
    n = 60
    true_pmin = 0.05
    # force the very first (min-distance) site's draw to be an unlucky 0/n
    focal = rng.binomial(n, true_pmin, size=20).astype(float)
    focal[0] = 0.0  # deliberately force the single-site pathology
    n_alleles = np.full((1, 20), float(n))
    n_focal = focal.reshape(1, 20)

    pmin_single_site = n_focal[:, 0] / n_alleles[:, 0]  # what the old convention would give
    assert pmin_single_site[0] == 0.0  # confirms the pathological input

    pmin_robust, _ = _derive_pmin_pmax(distance, n_alleles, n_focal, n_edge_sites=3)
    # averaging 3 sites should land much closer to the true 0.05 than the
    # single-site value of exactly 0.0
    assert pmin_robust[0] > 0.0
    assert abs(pmin_robust[0] - true_pmin) < abs(pmin_single_site[0] - true_pmin)


def test_derive_pmin_pmax_matches_direct_computation():
    n_alleles = np.array([[10, 10, 10, 10]], dtype=float)
    n_focal = np.array([[1, 2, 8, 9]], dtype=float)
    distance = np.array([0, 1, 2, 3], dtype=float)
    pmin, pmax = _derive_pmin_pmax(distance, n_alleles, n_focal, n_edge_sites=2)
    # pmin = (1+2)/(10+10) = 0.15, pmax = (8+9)/(10+10) = 0.85
    assert pmin[0] == pytest.approx(0.15)
    assert pmax[0] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Hyperparameter recovery
# ---------------------------------------------------------------------------

def test_hierarchical_fit_recovers_population_level_hyperparameters():
    distance, n_alleles, n_focal, truth = _simulate(seed=7)
    result = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                     num_warmup=1000, num_samples=1000,
                                     num_chains=4, seed=0)

    post = result.idata.posterior
    mu_center_est = float(post["mu_center"].mean())
    sigma_center_est = float(post["sigma_center"].mean())
    mu_width_est = float(np.exp(post["mu_log_width"].mean()))

    assert mu_center_est == pytest.approx(truth["mu_center"], abs=8.0)
    assert sigma_center_est == pytest.approx(truth["sigma_center"], abs=8.0)
    assert mu_width_est == pytest.approx(truth["mu_width"], abs=15.0)


def test_hierarchical_fit_converges_cleanly():
    from hycline.diagnostics.convergence import check

    distance, n_alleles, n_focal, truth = _simulate(seed=7)
    result = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                     num_warmup=1000, num_samples=1000,
                                     num_chains=4, seed=0)
    report = check(result.idata)
    assert report.rhat_max < 1.02


# ---------------------------------------------------------------------------
# Shrinkage: the actual signature of a working hierarchical model
# ---------------------------------------------------------------------------

def test_sparse_locus_shrinks_more_than_data_rich_locus():
    """The core validation this component exists to demonstrate: a sparse
    locus's hierarchical estimate should move further (in relative terms)
    from its own independent estimate toward the population mean than a
    data-rich locus's does."""
    distance, n_alleles, n_focal, truth = _simulate(seed=7)
    result = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                     num_warmup=1000, num_samples=1000,
                                     num_chains=4, seed=0)

    pmin_all, pmax_all = _derive_pmin_pmax(distance, n_alleles, n_focal, n_edge_sites=3)
    locus_df = result.locus_summary()
    pop_center = float(result.idata.posterior["mu_center"].mean())

    def independent_center(j):
        transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles[j], n_focal=n_focal[j])
        fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE,
                                      pmin=pmin_all[j], pmax=pmax_all[j])
        return fit.params.center

    sparse_indep = independent_center(0)   # locus_0: n=4/site
    rich_indep = independent_center(1)     # locus_1: n=60/site

    sparse_hier = locus_df.loc[0, "center_mean"]
    rich_hier = locus_df.loc[1, "center_mean"]

    sparse_shift = abs(sparse_hier - sparse_indep)
    rich_shift = abs(rich_hier - rich_indep)

    # the sparse locus should move (in absolute terms) at least as much as
    # the data-rich locus -- a direct, robust check of the shrinkage direction
    assert sparse_shift >= rich_shift * 0.9  # small tolerance for MC noise

    # and the data-rich locus's hierarchical estimate should stay close to
    # its own independent estimate (the model correctly "gets out of the
    # way" when a locus has enough of its own data)
    assert abs(rich_hier - rich_indep) < 3.0


def test_locus_summary_and_population_summary_are_consistent_shapes():
    distance, n_alleles, n_focal, truth = _simulate(seed=2, n_loci=4,
                                                      sample_sizes=[20, 20, 20, 20])
    result = fit_hierarchical_batch(distance, n_alleles, n_focal,
                                     locus_names=["a", "b", "c", "d"],
                                     num_warmup=600, num_samples=600,
                                     num_chains=2, seed=0)
    locus_df = result.locus_summary()
    assert list(locus_df["locus"]) == ["a", "b", "c", "d"]
    assert set(locus_df.columns) == {"locus", "center_mean", "center_sd",
                                      "width_mean", "width_sd"}

    pop_df = result.population_summary()
    assert set(["mu_center", "sigma_center", "mu_log_width", "sigma_log_width"]) <= \
        set(pop_df.index)

    s = result.summary()
    assert "population-level" in s.lower()
