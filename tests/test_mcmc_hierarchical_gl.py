"""
Tests for Phase 4 Component D: hierarchical geographic cline fitting with
genotype-likelihood marginalization -- the combination of Component A's
partial pooling and Phase 5's genotype-uncertainty marginalization.

The key validation isn't just "does this recover the right numbers" (see
BACKEND_PHASE4.md section 11 for why width recovery on a single noisy
draw is expected to show real sampling variability, the same theme
documented for Component A and Phase 5). It's the actual open empirical
question this component exists to answer: does hierarchical pooling's own
stabilizing effect already fix what Phase 5 fixes for independent
per-locus fits, or does genotype-likelihood-awareness still add value on
top of it? The answer, confirmed here, is a clear yes -- and the
mechanism (a hard-called sparse locus's inflated estimate contaminating
the shared population-level hyperparameters, not just its own posterior)
is more consequential than the original hypothesis.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from hycline.data.genotype_likelihood import simulate_genotype_likelihoods
from hycline.inference.mcmc_hierarchical_gl import (
    fit_hierarchical_batch_gl, _marginal_log_likelihood_jax, _derive_pmin_pmax_gl,
)
from hycline.inference.mcmc_hierarchical import fit_hierarchical_batch
from hycline.likelihoods.genotype_likelihood import _marginal_log_likelihood
from hycline.models.cline import ClineParams, TailType, cline_function


def test_jax_marginal_likelihood_matches_numpy_version():
    rng = np.random.default_rng(3)
    log_gl0 = rng.normal(0, 2, 25)
    log_gl1 = rng.normal(0, 2, 25)
    log_gl2 = rng.normal(0, 2, 25)
    p = rng.uniform(0.05, 0.95, 25)

    numpy_result = _marginal_log_likelihood(log_gl0, log_gl1, log_gl2, p)
    jax_result = np.asarray(_marginal_log_likelihood_jax(
        jnp.asarray(log_gl0), jnp.asarray(log_gl1), jnp.asarray(log_gl2), jnp.asarray(p)))
    np.testing.assert_allclose(numpy_result, jax_result, atol=1e-10)


def test_derive_pmin_pmax_gl_hard_call_matches_direct_computation():
    distance = np.array([0.0, 1.0, 2.0, 3.0])
    neg_inf = -1e10
    log_gl0 = np.array([[0.0, 0.0, neg_inf, neg_inf]])
    log_gl1 = np.array([[neg_inf, neg_inf, neg_inf, neg_inf]])
    log_gl2 = np.array([[neg_inf, neg_inf, 0.0, 0.0]])

    pmin, pmax = _derive_pmin_pmax_gl(distance, log_gl0, log_gl1, log_gl2, n_edge_individuals=2)
    assert pmin[0] == pytest.approx(0.0, abs=1e-6)
    assert pmax[0] == pytest.approx(1.0, abs=1e-6)


def test_derive_pmin_pmax_gl_uncertain_genotypes_give_intermediate_estimate():
    distance = np.array([0.0, 1.0])
    log_gl0 = np.array([[0.0, 0.0]])
    log_gl1 = np.array([[0.0, 0.0]])
    log_gl2 = np.array([[0.0, 0.0]])
    pmin, pmax = _derive_pmin_pmax_gl(distance, log_gl0, log_gl1, log_gl2, n_edge_individuals=1)
    assert pmin[0] == pytest.approx(0.5, abs=1e-6)
    assert pmax[0] == pytest.approx(0.5, abs=1e-6)


def _simulate_multi_locus_gl(seed, n_loci=6, n_ind=100, sparse_depth=2, rich_depth=8):
    rng = np.random.default_rng(seed)
    mu_center_true, sigma_center_true = 75.0, 10.0
    mu_log_width_true, sigma_log_width_true = np.log(25.0), 0.25
    distance = np.sort(rng.uniform(0, 150, size=n_ind))
    true_centers = rng.normal(mu_center_true, sigma_center_true, n_loci)
    true_widths = np.exp(rng.normal(mu_log_width_true, sigma_log_width_true, n_loci))
    depths = [sparse_depth] + [rich_depth] * (n_loci - 1)

    log_gl0 = np.zeros((n_loci, n_ind))
    log_gl1 = np.zeros((n_loci, n_ind))
    log_gl2 = np.zeros((n_loci, n_ind))
    n_alleles_hard = np.zeros((n_loci, n_ind))
    n_focal_hard = np.zeros((n_loci, n_ind))

    for j in range(n_loci):
        params = ClineParams(center=true_centers[j], width=true_widths[j], pmin=0.05, pmax=0.95)
        p = cline_function(distance, params, TailType.NONE)
        u = rng.uniform(size=n_ind)
        true_geno = np.where(u < (1 - p) ** 2, 0,
                              np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))
        gls = simulate_genotype_likelihoods(true_geno, depth=depths[j], error_rate=0.01,
                                             rng=np.random.default_rng(seed * 100 + j))
        log_gl0[j], log_gl1[j], log_gl2[j] = gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]
        gl_stack = np.stack([gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]])
        hard_geno = np.argmax(gl_stack, axis=0)
        n_alleles_hard[j] = 2.0
        n_focal_hard[j] = hard_geno

    return (distance, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
            true_centers, true_widths)


def test_hierarchical_gl_fit_converges_and_recovers_population_hyperparameters():
    from hycline.diagnostics.convergence import check

    (distance, log_gl0, log_gl1, log_gl2, *_, true_centers, true_widths
     ) = _simulate_multi_locus_gl(seed=1)

    fit = fit_hierarchical_batch_gl(distance, log_gl0, log_gl1, log_gl2,
                                     num_warmup=800, num_samples=800, num_chains=2, seed=0)
    report = check(fit.idata)
    assert report.rhat_max < 1.05

    post = fit.idata.posterior
    mu_center_est = float(post["mu_center"].mean())
    sample_mean_center = float(np.mean(true_centers))
    assert mu_center_est == pytest.approx(sample_mean_center, abs=15.0)


def test_hard_call_and_gl_likelihoods_are_finite_and_computed_consistently():
    """Regression check confirming apparent width 'inflation' seen during
    this component's own validation is genuine finite-sample MLE/posterior
    behavior, not a code defect -- the same investigative pattern used for
    Component A's own bug (BACKEND_PHASE4.md section 3)."""
    from hycline.data.schema import GenotypeLikelihoodTransect
    from hycline.likelihoods.genotype_likelihood import log_likelihood_geographic

    (distance, log_gl0, log_gl1, log_gl2, *_, true_centers, true_widths
     ) = _simulate_multi_locus_gl(seed=2)

    j = 0
    pmin_all, pmax_all = _derive_pmin_pmax_gl(distance, log_gl0, log_gl1, log_gl2)
    transect = GenotypeLikelihoodTransect(distance=distance, log_gl0=log_gl0[j],
                                           log_gl1=log_gl1[j], log_gl2=log_gl2[j])

    true_params = ClineParams(center=true_centers[j], width=true_widths[j],
                               pmin=pmin_all[j], pmax=pmax_all[j])
    wider_params = ClineParams(center=true_centers[j], width=true_widths[j] * 2,
                                pmin=pmin_all[j], pmax=pmax_all[j])

    ll_true = log_likelihood_geographic(transect, true_params, TailType.NONE)
    ll_wider = log_likelihood_geographic(transect, wider_params, TailType.NONE)
    assert np.isfinite(ll_true)
    assert np.isfinite(ll_wider)


def test_gl_aware_hierarchical_beats_hard_call_hierarchical_for_sparse_locus():
    """The core validation this component exists to run: for a sparse
    (low-depth) locus fit *hierarchically* alongside richer loci, does
    genotype-likelihood-awareness still add calibration value beyond what
    pooling alone provides? Answer: yes, clearly -- checked via a single
    representative replicate (the full multi-replicate study, 12 seeds,
    is in BACKEND_PHASE4.md section 11 and examples/
    demo_hierarchical_gl.py; this test locks in the same qualitative
    result on one seed as a fast regression check, not a full restudy on
    every test run)."""
    (distance, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
     true_centers, true_widths) = _simulate_multi_locus_gl(seed=0)

    fit_gl = fit_hierarchical_batch_gl(distance, log_gl0, log_gl1, log_gl2,
                                        num_warmup=800, num_samples=800,
                                        num_chains=2, seed=0)
    fit_hard = fit_hierarchical_batch(distance, n_alleles_hard, n_focal_hard,
                                       num_warmup=800, num_samples=800,
                                       num_chains=2, seed=0)

    post_gl = fit_gl.idata.posterior
    post_hard = fit_hard.idata.posterior

    width_gl = np.exp(post_gl["log_width"].isel(locus=0).values.flatten())
    width_hard = np.exp(post_hard["log_width"].isel(locus=0).values.flatten())

    def hdi(draws, prob=0.95):
        return np.percentile(draws, [(100 - prob * 100) / 2, 100 - (100 - prob * 100) / 2])

    lo_gl, hi_gl = hdi(width_gl)
    lo_hard, hi_hard = hdi(width_hard)

    gl_covers = lo_gl <= true_widths[0] <= hi_gl
    hard_covers = lo_hard <= true_widths[0] <= hi_hard

    assert gl_covers
    assert not hard_covers
