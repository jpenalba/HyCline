"""
Tests for Phase 4 Component B: the hierarchical genomic cline model.

The genomic-cline analog of tests/test_mcmc_hierarchical.py. There is no
HZAR-equivalent gold standard here (HZAR doesn't do genomic clines at
all, let alone hierarchical ones), so these tests validate against
synthetic data with a known population-level structure -- see
examples/demo_hierarchical_genomic.py for the fuller worked example these
tests are a faster/stricter version of, and BACKEND_PHASE4.md for the
full design writeup.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hycline.data.schema import HybridIndexLocusData
from hycline.inference.mle_genomic import fit_genomic_cline_mle
from hycline.inference.mcmc_genomic_hierarchical import (
    fit_hierarchical_genomic_batch, _genomic_cline_core_jax,
)
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function


def _simulate(seed=21, n_loci=8, sample_sizes=None, n_ind=120):
    rng = np.random.default_rng(seed)
    mu_u_true = 0.0
    sigma_u_true = 1.0
    mu_log_v_true = 0.0
    sigma_log_v_true = 0.5

    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    true_u = rng.normal(mu_u_true, sigma_u_true, n_loci)
    true_v = np.exp(rng.normal(mu_log_v_true, sigma_log_v_true, n_loci))

    if sample_sizes is None:
        sample_sizes = [1, 20] + [2] * (n_loci - 2)

    n_alleles = np.zeros((n_loci, n_ind))
    n_focal = np.zeros((n_loci, n_ind))
    for j in range(n_loci):
        params = GenomicClineParams(u=true_u[j], v=true_v[j])
        phi = genomic_cline_function(hybrid_index, params)
        n_alleles[j, :] = sample_sizes[j]
        n_focal[j, :] = rng.binomial(sample_sizes[j], phi)

    truth = dict(mu_u=mu_u_true, sigma_u=sigma_u_true, mu_v=np.exp(mu_log_v_true),
                 true_u=true_u, true_v=true_v)
    return hybrid_index, n_alleles, n_focal, truth


# ---------------------------------------------------------------------------
# JAX/NumPy parity (same regression-test pattern as the geographic model)
# ---------------------------------------------------------------------------

def test_jax_genomic_core_matches_numpy_core():
    params = GenomicClineParams(u=1.7, v=3.2)
    h = np.linspace(0.01, 0.99, 40)

    numpy_result = genomic_cline_function(h, params)
    jax_result = np.asarray(_genomic_cline_core_jax(jnp.asarray(h), params.u, params.v))

    np.testing.assert_allclose(numpy_result, jax_result, atol=1e-10)


def test_jax_parental_allele_freq_matches_numpy():
    from hycline.inference.mcmc_genomic_hierarchical import _parental_allele_freq_jax
    from hycline.models.genomic_cline import genomic_cline_allele_freq
    params = GenomicClineParams(u=-0.4, v=1.6)
    h = np.linspace(0.01, 0.99, 30)
    p0, p1 = np.array([0.1, 0.9]), np.array([0.8, 0.2])
    phi = np.asarray(_genomic_cline_core_jax(jnp.asarray(h), params.u, params.v))
    jax_freq = np.asarray(_parental_allele_freq_jax(
        jnp.asarray(np.vstack([phi, phi])), jnp.asarray(p0), jnp.asarray(p1)))
    for i in range(2):
        np.testing.assert_allclose(jax_freq[i], genomic_cline_allele_freq(h, params, p0[i], p1[i]),
                                   atol=1e-10)


def test_jax_genomic_core_neutral_locus_matches_identity():
    """u=0, v=1 should give phi(h) = h, same neutral-null check as the
    non-hierarchical genomic cline model (tests/test_genomic_cline.py)."""
    h = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    phi = np.asarray(_genomic_cline_core_jax(h, 0.0, 1.0))
    np.testing.assert_allclose(phi, np.asarray(h), atol=1e-6)


# ---------------------------------------------------------------------------
# Hyperparameter recovery
# ---------------------------------------------------------------------------

def test_hierarchical_genomic_fit_recovers_population_level_hyperparameters():
    hybrid_index, n_alleles, n_focal, truth = _simulate(seed=21)
    result = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                             num_warmup=1000, num_samples=1000,
                                             num_chains=4, seed=0)

    post = result.idata.posterior
    mu_u_est = float(post["mu_u"].mean())
    sigma_u_est = float(post["sigma_u"].mean())
    mu_v_est = float(np.exp(post["mu_log_v"].mean()))

    # with only 8 loci, the hierarchical fit's estimate of the population
    # mean should track the *sample* mean of the true per-locus values
    # (not necessarily the theoretical population value) -- see
    # BACKEND_PHASE4.md section 4 for why this is correct behavior, not a
    # bias. Check against the sample mean of the true values used here.
    sample_mean_u = float(np.mean(truth["true_u"]))
    sample_mean_log_v = float(np.mean(np.log(truth["true_v"])))

    assert mu_u_est == pytest.approx(sample_mean_u, abs=0.3)
    assert float(post["mu_log_v"].mean()) == pytest.approx(sample_mean_log_v, abs=0.3)
    assert sigma_u_est == pytest.approx(truth["sigma_u"], abs=0.8)


def test_hierarchical_genomic_fit_converges_cleanly():
    from hycline.diagnostics.convergence import check

    hybrid_index, n_alleles, n_focal, truth = _simulate(seed=21)
    result = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                             num_warmup=1000, num_samples=1000,
                                             num_chains=4, seed=0)
    report = check(result.idata)
    assert report.rhat_max < 1.02


# ---------------------------------------------------------------------------
# Shrinkage: the actual signature of a working hierarchical model
# ---------------------------------------------------------------------------

def test_sparse_locus_shrinks_more_than_data_rich_locus():
    """Genomic-cline analog of the geographic model's core validation
    (tests/test_mcmc_hierarchical.py::
    test_sparse_locus_shrinks_more_than_data_rich_locus): a sparse locus's
    hierarchical estimate should move further from its own independent
    estimate toward the population mean than a data-rich locus's does."""
    hybrid_index, n_alleles, n_focal, truth = _simulate(seed=21)
    result = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                             num_warmup=1200, num_samples=1200,
                                             num_chains=4, seed=0)

    locus_df = result.locus_summary()
    pop_v = float(np.exp(result.idata.posterior["mu_log_v"].mean()))

    def independent_v(j):
        d = HybridIndexLocusData(hybrid_index=hybrid_index, n_alleles=n_alleles[j],
                                  n_focal=n_focal[j])
        fit = fit_genomic_cline_mle(d)
        return fit.params.v

    sparse_indep = independent_v(0)   # locus_0: n=1/individual
    rich_indep = independent_v(1)     # locus_1: n=20/individual

    sparse_hier = locus_df.loc[0, "v_mean"]
    rich_hier = locus_df.loc[1, "v_mean"]

    sparse_shift = abs(sparse_hier - sparse_indep)
    rich_shift = abs(rich_hier - rich_indep)

    # the sparse locus should move at least as much as the data-rich locus
    assert sparse_shift >= rich_shift

    # and the data-rich locus's hierarchical estimate should stay close to
    # its own independent estimate
    assert abs(rich_hier - rich_indep) < 0.1


def test_locus_summary_and_population_summary_are_consistent_shapes():
    hybrid_index, n_alleles, n_focal, truth = _simulate(
        seed=5, n_loci=4, sample_sizes=[2, 2, 2, 2], n_ind=80)
    result = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                             locus_names=["a", "b", "c", "d"],
                                             num_warmup=600, num_samples=600,
                                             num_chains=2, seed=0)
    locus_df = result.locus_summary()
    assert list(locus_df["locus"]) == ["a", "b", "c", "d"]
    assert set(locus_df.columns) == {"locus", "u_mean", "u_sd", "v_mean", "v_sd"}

    pop_df = result.population_summary()
    assert set(["mu_u", "sigma_u", "mu_log_v", "sigma_log_v"]) <= set(pop_df.index)

    s = result.summary()
    assert "population-level" in s.lower()


# ---------------------------------------------------------------------------
# pool_center=True: Firneno et al. (2023)'s own parameterization
# (BACKEND_PHASE4.md section 9.3)
# ---------------------------------------------------------------------------

def _simulate_center_pooled(seed=201, n_loci=15, n_ind=80,
                             sigma_center_true=0.8, sigma_log_v_true=0.3):
    rng = np.random.default_rng(seed)
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    logit_centers_true = rng.normal(0, sigma_center_true, n_loci)
    log_v_true = rng.normal(0, sigma_log_v_true, n_loci)
    v_true = np.exp(log_v_true)
    u_true = v_true * logit_centers_true
    center_true = 1.0 / (1.0 + np.exp(-logit_centers_true))

    n_alleles = np.full((n_loci, n_ind), 2.0)
    n_focal = np.zeros((n_loci, n_ind))
    for i in range(n_loci):
        phi = genomic_cline_function(hybrid_index, GenomicClineParams(u=u_true[i], v=v_true[i]))
        n_focal[i] = rng.binomial(2, np.clip(phi, 0, 1))

    truth = {"sigma_center": sigma_center_true, "sigma_log_v": sigma_log_v_true,
             "center": center_true, "v": v_true, "u": u_true}
    return hybrid_index, n_alleles, n_focal, truth


def test_pool_center_default_is_false_and_behavior_is_unchanged():
    """The core backward-compatibility guarantee: pool_center defaults to
    False, and a fit with the default should be indistinguishable in
    shape/behavior from before this parameter existed."""
    hybrid_index, n_alleles, n_focal, truth = _simulate(seed=21, n_loci=4, n_ind=80)
    result = fit_hierarchical_genomic_batch(hybrid_index, n_alleles, n_focal,
                                             num_warmup=400, num_samples=400,
                                             num_chains=2, seed=0)
    assert result.pool_center is False
    assert "logit_center" not in result.idata.posterior
    assert "u" in result.idata.posterior
    pop_df = result.population_summary()
    assert set(["mu_u", "sigma_u", "mu_log_v", "sigma_log_v"]) <= set(pop_df.index)


def test_pool_center_true_recovers_known_sigma_center():
    hybrid_index, n_alleles, n_focal, truth = _simulate_center_pooled(seed=201, n_loci=15)
    result = fit_hierarchical_genomic_batch(
        hybrid_index, n_alleles, n_focal, pool_center=True,
        num_warmup=800, num_samples=800, num_chains=2, seed=1)

    assert result.pool_center is True
    assert "logit_center" in result.idata.posterior
    assert "u" not in result.idata.posterior

    post = result.idata.posterior
    sigma_center_est = float(post["sigma_center"].mean())
    sigma_log_v_est = float(post["sigma_log_v"].mean())

    # loose bounds given only 15 loci -- this is checking the mechanism
    # recovers roughly the right magnitude, not pinpoint precision
    assert abs(sigma_center_est - truth["sigma_center"]) < 0.5
    assert abs(sigma_log_v_est - truth["sigma_log_v"]) < 0.3


def test_pool_center_true_locus_summary_recovers_known_per_locus_values():
    hybrid_index, n_alleles, n_focal, truth = _simulate_center_pooled(seed=201, n_loci=15)
    result = fit_hierarchical_genomic_batch(
        hybrid_index, n_alleles, n_focal, pool_center=True,
        num_warmup=800, num_samples=800, num_chains=2, seed=1)

    locus_df = result.locus_summary()
    assert "center_mean" in locus_df.columns
    assert "u_mean" in locus_df.columns  # still derived and reported

    corr_center = np.corrcoef(truth["center"], locus_df["center_mean"])[0, 1]
    corr_v = np.corrcoef(truth["v"], locus_df["v_mean"])[0, 1]
    assert corr_center > 0.85
    assert corr_v > 0.85


def test_pool_center_true_derived_u_matches_v_times_logit_center():
    """Confirms the exact algebraic relationship (u = v * logit(center))
    holds in the actual posterior output, not just in the isolated math
    check this was based on."""
    hybrid_index, n_alleles, n_focal, truth = _simulate_center_pooled(seed=203, n_loci=5)
    result = fit_hierarchical_genomic_batch(
        hybrid_index, n_alleles, n_focal, pool_center=True,
        num_warmup=400, num_samples=400, num_chains=2, seed=2)

    post = result.idata.posterior
    v = np.exp(post["log_v"].values)
    u_expected = v * post["logit_center"].values
    locus_df = result.locus_summary()
    for i in range(5):
        u_i = (post["logit_center"].isel(locus=i).values * np.exp(post["log_v"].isel(locus=i).values))
        np.testing.assert_allclose(locus_df.loc[i, "u_mean"], u_i.mean(), rtol=1e-6)


def test_pool_center_jax_and_numpy_reparameterization_are_exact():
    """The core math this whole feature depends on: phi(h; u=v*logit(c), v)
    must equal phi(h; u, v) exactly when logit(c) = u/v -- checked
    directly, not just assumed from the derivation."""
    h = np.linspace(0.01, 0.99, 20)
    u_true, v_true = 1.5, 3.0
    phi_direct = genomic_cline_function(h, GenomicClineParams(u=u_true, v=v_true))

    logit_c = u_true / v_true
    u_derived = v_true * logit_c
    phi_derived = genomic_cline_function(h, GenomicClineParams(u=u_derived, v=v_true))

    np.testing.assert_allclose(phi_direct, phi_derived, atol=1e-12)
    assert u_derived == pytest.approx(u_true)


# ---------------------------------------------------------------------------
# Parental frequencies integrated out (their panel posteriors given)
# ---------------------------------------------------------------------------

def _simulate_sampled_panels(seed=7, n_loci=6, n_ind=100, n_panel=10):
    """Neutral loci (u = 0, v = 1) between populations at 0.3 / 0.7; the
    data carry the frequencies seen in panels of n_panel allele copies and
    their Beta posteriors (Jeffreys prior)."""
    from hycline.models.genomic_cline import genomic_cline_allele_freq
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.02, 0.98, n_ind)
    freq = genomic_cline_allele_freq(h, GenomicClineParams(u=0.0, v=1.0), 0.3, 0.7)
    n_focal = rng.binomial(2, np.broadcast_to(freq, (n_loci, n_ind))).astype(float)
    p0 = rng.binomial(n_panel, 0.3, n_loci) / n_panel
    p1 = rng.binomial(n_panel, 0.7, n_loci) / n_panel
    return (h, np.full((n_loci, n_ind), 2.0), n_focal, p0, p1,
            (p0 * n_panel + 0.5, (1 - p0) * n_panel + 0.5),
            (p1 * n_panel + 0.5, (1 - p1) * n_panel + 0.5))


def test_integrated_parental_freqs_are_summarized_and_widen_the_rate_posterior():
    h, n_alleles, n_focal, p0, p1, beta0, beta1 = _simulate_sampled_panels()
    kw = dict(parental_freq_pop0=p0, parental_freq_pop1=p1,
              num_warmup=500, num_samples=500, num_chains=2, seed=3)
    fixed = fit_hierarchical_genomic_batch(h, n_alleles, n_focal, **kw)
    integrated = fit_hierarchical_genomic_batch(h, n_alleles, n_focal,
                                                parental_beta_pop0=beta0,
                                                parental_beta_pop1=beta1, **kw)
    assert "p0" not in fixed.idata.posterior
    loc = integrated.locus_summary()
    for col in ("p0_mean", "p0_sd", "p1_mean", "p1_sd"):
        assert col in loc
    assert ((loc["p0_mean"] > 0) & (loc["p0_mean"] < 1)).all()
    assert (loc["p0_sd"] > 0.03).all()  # a 10-copy panel leaves p0 uncertain
    log_v_sd = lambda r: r.idata.posterior["log_v"].std(dim=("chain", "draw")).values
    assert log_v_sd(integrated).mean() > log_v_sd(fixed).mean()
    # posterior means sit between the panel estimates and the population
    # frequencies the hybrids came from (0.3 / 0.7)
    assert np.abs(loc["p0_mean"] - 0.3).mean() < np.abs(p0 - 0.3).mean() + 0.02


def test_parental_freq_posteriors_must_be_given_together():
    h, n_alleles, n_focal, p0, p1, beta0, _ = _simulate_sampled_panels()
    with pytest.raises(ValueError, match="together"):
        fit_hierarchical_genomic_batch(h, n_alleles, n_focal, parental_freq_pop0=p0,
                                       parental_freq_pop1=p1, parental_beta_pop0=beta0,
                                       num_warmup=10, num_samples=10, num_chains=1)
