"""
Tests for Phase 4 Component E: hierarchical genomic cline fitting with
genotype-likelihood marginalization -- the genomic-cline analog of
Component D (geographic clines).

The key finding here is genuinely different from Component D's, not a
repeat of it -- see BACKEND_PHASE4.md section 12 for the full
investigation. At a moderate sparse/rich depth contrast (2x vs 8x,
matching Component D's own setup), GL-aware and hard-call hierarchical
fitting show *no detectable difference* in coverage for genomic clines --
unlike the dramatic difference found for geographic clines. Only at a
more extreme contrast (1x vs 20x, fewer individuals) does a real but
modest difference emerge, and it manifests as an *overconfidence* pattern
(hard-call's point estimates are actually closer to truth on average, but
its intervals are too narrow relative to that error) rather than
Component D's "population-level contamination" mechanism. Both findings
are locked in here, not just the more dramatic one.
"""

import numpy as np
import pytest

from hycline.data.genotype_likelihood import simulate_genotype_likelihoods
from hycline.inference.mcmc_genomic_hierarchical_gl import fit_hierarchical_genomic_batch_gl
from hycline.inference.mcmc_genomic_hierarchical import fit_hierarchical_genomic_batch
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function


def _simulate_multi_locus_genomic_gl(seed, n_loci=6, n_ind=100, sparse_depth=2, rich_depth=8):
    rng = np.random.default_rng(seed)
    mu_u_true, sigma_u_true = 0.0, 1.0
    mu_log_v_true, sigma_log_v_true = 0.0, 0.5
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    true_u = rng.normal(mu_u_true, sigma_u_true, n_loci)
    true_v = np.exp(rng.normal(mu_log_v_true, sigma_log_v_true, n_loci))
    depths = [sparse_depth] + [rich_depth] * (n_loci - 1)

    log_gl0 = np.zeros((n_loci, n_ind))
    log_gl1 = np.zeros((n_loci, n_ind))
    log_gl2 = np.zeros((n_loci, n_ind))
    n_alleles_hard = np.zeros((n_loci, n_ind))
    n_focal_hard = np.zeros((n_loci, n_ind))

    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        u_draw = rng.uniform(size=n_ind)
        true_geno = np.where(u_draw < (1 - phi) ** 2, 0,
                              np.where(u_draw < (1 - phi) ** 2 + 2 * phi * (1 - phi), 1, 2))
        gls = simulate_genotype_likelihoods(true_geno, depth=depths[j], error_rate=0.01,
                                             rng=np.random.default_rng(seed * 100 + j))
        log_gl0[j], log_gl1[j], log_gl2[j] = gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]
        gl_stack = np.stack([gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]])
        hard_geno = np.argmax(gl_stack, axis=0)
        n_alleles_hard[j] = 2.0
        n_focal_hard[j] = hard_geno

    return (hybrid_index, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
            true_u, true_v)


def test_hierarchical_genomic_gl_fit_converges_and_recovers_population_hyperparameters():
    from hycline.diagnostics.convergence import check

    (hybrid_index, log_gl0, log_gl1, log_gl2, *_, true_u, true_v
     ) = _simulate_multi_locus_genomic_gl(seed=1)

    fit = fit_hierarchical_genomic_batch_gl(hybrid_index, log_gl0, log_gl1, log_gl2,
                                             num_warmup=800, num_samples=800,
                                             num_chains=2, seed=0)
    report = check(fit.idata)
    assert report.rhat_max < 1.05

    post = fit.idata.posterior
    sample_mean_u = float(np.mean(true_u))
    assert float(post["mu_u"].mean()) == pytest.approx(sample_mean_u, abs=1.5)


def test_moderate_depth_contrast_shows_no_detectable_coverage_difference():
    """Regression test for the first (honest, non-dramatic) half of this
    component's finding: at a moderate sparse/rich depth contrast (the
    same 2x/8x setup that showed a dramatic difference for geographic
    clines in Component D), GL-aware and hard-call hierarchical fitting
    for genomic clines give the *same* coverage -- no meaningful
    advantage from genotype-likelihood-awareness at this contrast. This
    is checked directly rather than assumed, and is a real, different
    result from Component D's, not a weaker version of the same one."""
    (hybrid_index, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
     true_u, true_v) = _simulate_multi_locus_genomic_gl(seed=0, sparse_depth=2, rich_depth=8)

    fit_gl = fit_hierarchical_genomic_batch_gl(hybrid_index, log_gl0, log_gl1, log_gl2,
                                                num_warmup=600, num_samples=600,
                                                num_chains=2, seed=0)
    fit_hard = fit_hierarchical_genomic_batch(hybrid_index, n_alleles_hard, n_focal_hard,
                                               num_warmup=600, num_samples=600,
                                               num_chains=2, seed=0)

    v_gl = np.exp(fit_gl.idata.posterior["log_v"].isel(locus=0).values.flatten())
    v_hard = np.exp(fit_hard.idata.posterior["log_v"].isel(locus=0).values.flatten())

    def hdi(draws, prob=0.95):
        return np.percentile(draws, [(100 - prob * 100) / 2, 100 - (100 - prob * 100) / 2])

    lo_gl, hi_gl = hdi(v_gl)
    lo_hard, hi_hard = hdi(v_hard)

    assert lo_gl <= true_v[0] <= hi_gl
    assert lo_hard <= true_v[0] <= hi_hard


def test_extreme_depth_contrast_gl_aware_achieves_better_coverage():
    """Regression test for the second half of this component's finding:
    at a more extreme depth contrast (1x vs 20x, fewer individuals), a
    real difference does emerge, checked on one representative seed
    (the full 8-replicate study is in BACKEND_PHASE4.md section 12.3).
    Unlike Component D's mechanism, this is an overconfidence pattern,
    not population-level contamination -- hard-call's point estimate can
    even be *closer* to the truth while still failing to cover it,
    because its interval is too narrow relative to its own error."""
    (hybrid_index, log_gl0, log_gl1, log_gl2, n_alleles_hard, n_focal_hard,
     true_u, true_v) = _simulate_multi_locus_genomic_gl(
        seed=3, n_ind=60, sparse_depth=1, rich_depth=20)

    fit_gl = fit_hierarchical_genomic_batch_gl(hybrid_index, log_gl0, log_gl1, log_gl2,
                                                num_warmup=600, num_samples=600,
                                                num_chains=2, seed=0)
    fit_hard = fit_hierarchical_genomic_batch(hybrid_index, n_alleles_hard, n_focal_hard,
                                               num_warmup=600, num_samples=600,
                                               num_chains=2, seed=0)

    v_gl = np.exp(fit_gl.idata.posterior["log_v"].isel(locus=0).values.flatten())
    v_hard = np.exp(fit_hard.idata.posterior["log_v"].isel(locus=0).values.flatten())

    def hdi(draws, prob=0.95):
        return np.percentile(draws, [(100 - prob * 100) / 2, 100 - (100 - prob * 100) / 2])

    lo_gl, hi_gl = hdi(v_gl)
    lo_hard, hi_hard = hdi(v_hard)

    gl_covers = lo_gl <= true_v[0] <= hi_gl
    hard_covers = lo_hard <= true_v[0] <= hi_hard

    assert gl_covers
    assert not hard_covers


def test_fit_hierarchical_genomic_batch_gl_calls_uneven_sampling_warning():
    # Regression test: fit_hierarchical_genomic_batch_gl never called
    # _warn_if_hybrid_index_distribution_uneven (the Bailey 2024
    # steep-cline/sparse-hybrid-index check), so the warning silently
    # never fired for genotype-likelihood input, only for hard-count
    # input via fit_hierarchical_genomic_batch. Verified here at the
    # call-site level (mocked, with minimal MCMC settings for speed) --
    # the full end-to-end statistical behavior of the warning itself is
    # already covered by the hard-count path's own dedicated tests.
    from unittest.mock import patch
    import hycline.inference.mcmc_genomic_hierarchical_gl as mod

    (hybrid_index, log_gl0, log_gl1, log_gl2, *_rest) = _simulate_multi_locus_genomic_gl(
        seed=7, n_loci=2, n_ind=20)

    with patch.object(mod, "_warn_if_hybrid_index_distribution_uneven") as mock_warn:
        mod.fit_hierarchical_genomic_batch_gl(
            hybrid_index, log_gl0, log_gl1, log_gl2,
            num_warmup=20, num_samples=20, num_chains=1, seed=0)
        mock_warn.assert_called_once()
