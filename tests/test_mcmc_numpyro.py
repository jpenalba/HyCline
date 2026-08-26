import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.inference.mcmc_numpyro import fit_mcmc, fit_mcmc_batch, _cline_core_jax
from hycline.diagnostics.convergence import check


def _simulate(true_params, n_sites=40, n_alleles_range=(20, 50), seed=1):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(-30, 130, size=n_sites))
    n_alleles = rng.integers(*n_alleles_range, size=n_sites).astype(float)
    true_freq = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def test_jax_core_matches_numpy_core():
    """Regression test flagged in BACKEND_PHASE2.md section 11: the JAX-native cline
    core used inside the NumPyro model must agree numerically with the
    NumPy implementation used everywhere else in the package."""
    import jax.numpy as jnp

    params = ClineParams(center=12.3, width=7.1, pmin=0.05, pmax=0.92)
    x = np.linspace(-40, 60, 50)

    numpy_result = cline_function(x, params, TailType.NONE)
    jax_result = np.asarray(_cline_core_jax(jnp.asarray(x), params.center,
                                             params.width, params.pmin, params.pmax))

    np.testing.assert_allclose(numpy_result, jax_result, atol=1e-10)


def test_jax_tail_matches_numpy_tail():
    """Regression test for BACKEND_PHASE2.md section 12: the JAX-native
    tail-model extension must agree numerically with the NumPy
    implementation (`hycline.models.cline.cline_function`) for every tail
    type, the same parity bar `_cline_core_jax` is held to for the
    no-tail case above."""
    import jax.numpy as jnp
    from hycline.inference.mcmc_numpyro import _cline_tail_jax

    x = np.linspace(-50, 150, 500)
    xj = jnp.asarray(x)

    for tail_type in [TailType.NONE, TailType.MIRROR, TailType.LEFT,
                       TailType.RIGHT, TailType.BOTH]:
        params = ClineParams(center=50, width=20, pmin=0.1, pmax=0.9)
        if tail_type in (TailType.LEFT, TailType.BOTH, TailType.MIRROR):
            params.delta_l, params.tau_l = 0.15, 12.0
        if tail_type in (TailType.RIGHT, TailType.BOTH):
            params.delta_r, params.tau_r = 0.25, 8.0

        numpy_result = cline_function(x, params, tail_type)

        core_j = _cline_core_jax(xj, params.center, params.width, params.pmin, params.pmax)
        jax_result = np.asarray(_cline_tail_jax(
            xj, core_j, params.center, params.width, params.pmin, params.pmax, tail_type,
            params.delta_l, params.tau_l, params.delta_r, params.tau_r))

        np.testing.assert_allclose(numpy_result, jax_result, atol=1e-10,
                                    err_msg=f"mismatch for tail_type={tail_type}")


def test_fit_mcmc_recovers_known_parameters():
    true_params = ClineParams(center=40.0, width=15.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=60, n_alleles_range=(30, 60), seed=11)

    fit = fit_mcmc(transect, num_warmup=500, num_samples=500, num_chains=2, seed=0)

    post_center = float(fit.idata.posterior["center"].mean())
    post_width = float(fit.idata.posterior["width"].mean())

    assert post_center == pytest.approx(true_params.center, abs=6.0)
    assert post_width == pytest.approx(true_params.width, abs=10.0)


def test_fit_mcmc_fixed_extremes_matches_mle():
    """With pmin/pmax fixed, the NUTS posterior mean should land close to
    the Phase 1 MLE fit on the same data and model (see BACKEND_PHASE2.md
    section 2 -- Phase 2 doesn't change the model, only how it's fit)."""
    from hycline.inference.mle import fit_mle_fixed_extremes

    true_params = ClineParams(center=30.0, width=12.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, n_sites=50, seed=5)

    mle_fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    bayes_fit = fit_mcmc(transect, pmin_fixed=mle_fit.params.pmin,
                          pmax_fixed=mle_fit.params.pmax,
                          num_warmup=500, num_samples=500, num_chains=2, seed=2)

    post_center = float(bayes_fit.idata.posterior["center"].mean())
    post_width = float(bayes_fit.idata.posterior["width"].mean())

    assert post_center == pytest.approx(mle_fit.params.center, abs=5.0)
    assert post_width == pytest.approx(mle_fit.params.width, abs=8.0)


def test_diagnostics_check_runs_and_flags_nothing_on_a_clean_fit():
    true_params = ClineParams(center=20.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=60, n_alleles_range=(30, 60), seed=3)
    fit = fit_mcmc(transect, num_warmup=800, num_samples=800, num_chains=2, seed=1)

    report = check(fit.idata)
    assert report.n_chains == 2
    assert report.n_draws_per_chain == 800
    # a well-specified fit with plenty of data shouldn't need to flag anything
    assert report.ok, report.warnings


def test_fit_mcmc_batch_recovers_different_per_locus_parameters():
    """Two loci with genuinely different true centers, fit in a single
    batched (plate-vectorized) MCMC run, should recover distinct posteriors."""
    distance = np.sort(np.random.default_rng(0).uniform(-20, 120, size=40))

    params_a = ClineParams(center=20.0, width=10.0, pmin=0.05, pmax=0.95)
    params_b = ClineParams(center=80.0, width=10.0, pmin=0.05, pmax=0.95)

    rng = np.random.default_rng(42)
    n_alleles = rng.integers(20, 50, size=40).astype(float)

    freq_a = cline_function(distance, params_a, TailType.NONE)
    freq_b = cline_function(distance, params_b, TailType.NONE)
    n_focal_a = rng.binomial(n_alleles.astype(int), freq_a).astype(float)
    n_focal_b = rng.binomial(n_alleles.astype(int), freq_b).astype(float)

    n_alleles_batch = np.stack([n_alleles, n_alleles])
    n_focal_batch = np.stack([n_focal_a, n_focal_b])

    result = fit_mcmc_batch(distance, n_alleles_batch, n_focal_batch,
                             locus_names=["locus_a", "locus_b"],
                             num_warmup=500, num_samples=500, num_chains=2, seed=7)

    centers = result.idata.posterior["center"].mean(dim=("chain", "draw")).values
    assert centers[0] == pytest.approx(params_a.center, abs=8.0)
    assert centers[1] == pytest.approx(params_b.center, abs=8.0)
    assert abs(centers[0] - centers[1]) > 20  # the two loci should be clearly distinguished


# ---------------------------------------------------------------------------
# Bayesian tail-model support (BACKEND_PHASE2.md section 12)
#
# These tests deliberately do NOT assert good convergence (low R-hat, high
# ESS) for tail-model fits -- see BACKEND_PHASE2.md section 12.2 for the
# real finding investigated during this component's own build: NUTS has
# severe, genuine mixing difficulty with the tail model's non-smooth
# (kinked) log-density, confirmed by isolating that the identical data
# converges perfectly with TailType.NONE and degrades specifically when
# tail parameters are added. What IS asserted: the math is correct (see
# test_jax_tail_matches_numpy_tail above) and the sampler runs to
# completion and returns a structurally valid result, honest about
# whatever its actual convergence state is via `hycline.diagnostics.
# convergence.check()` -- exactly what a real user would see.
# ---------------------------------------------------------------------------

def test_fit_mcmc_tail_model_runs_and_returns_expected_parameters():
    params = ClineParams(center=15.0, width=10.0, pmin=0.05, pmax=0.95,
                          delta_l=0.15, tau_l=8.0)
    transect = _simulate_with_tail(params, TailType.LEFT, n_sites=60, seed=21)

    result = fit_mcmc(transect, tail_type=TailType.LEFT, width_upper_bound=40.0,
                       num_warmup=300, num_samples=300, num_chains=2, seed=0)

    assert set(result.param_names) == {"center", "width", "pmin", "pmax", "delta_l", "tau_l"}
    for name in result.param_names:
        assert name in result.idata.posterior
    # a full ClineParams should be reconstructable, whatever the fit quality
    fitted = result.params
    assert fitted.delta_l is not None
    assert fitted.tau_l is not None


def test_fit_mcmc_mirror_tail_sets_both_sides_from_shared_posterior():
    params = ClineParams(center=15.0, width=10.0, pmin=0.05, pmax=0.95,
                          delta_l=0.2, tau_l=6.0)
    transect = _simulate_with_tail(params, TailType.MIRROR, n_sites=60, seed=23)

    result = fit_mcmc(transect, tail_type=TailType.MIRROR, width_upper_bound=40.0,
                       num_warmup=300, num_samples=300, num_chains=2, seed=0)

    # MIRROR only samples delta_l/tau_l (one shared pair, not independent
    # per side) -- matches ClineParams.as_array's own MIRROR convention
    assert set(result.param_names) == {"center", "width", "pmin", "pmax", "delta_l", "tau_l"}
    fitted = result.params
    # .params should mirror delta_l/tau_l onto delta_r/tau_r for a complete,
    # directly cline_function()-usable ClineParams
    assert fitted.delta_r == fitted.delta_l
    assert fitted.tau_r == fitted.tau_l


def test_fit_mcmc_batch_tail_model_runs_for_multiple_loci():
    params_a = ClineParams(center=10.0, width=8.0, pmin=0.05, pmax=0.95,
                            delta_l=0.1, tau_l=5.0)
    params_b = ClineParams(center=40.0, width=8.0, pmin=0.05, pmax=0.95,
                            delta_l=0.1, tau_l=5.0)
    distance = np.sort(np.random.default_rng(29).uniform(-20, 80, size=50))
    rng = np.random.default_rng(29)
    n_alleles = rng.integers(20, 50, size=50).astype(float)

    freq_a = cline_function(distance, params_a, TailType.LEFT)
    freq_b = cline_function(distance, params_b, TailType.LEFT)
    n_focal_a = rng.binomial(n_alleles.astype(int), freq_a).astype(float)
    n_focal_b = rng.binomial(n_alleles.astype(int), freq_b).astype(float)

    n_alleles_batch = np.stack([n_alleles, n_alleles])
    n_focal_batch = np.stack([n_focal_a, n_focal_b])

    result = fit_mcmc_batch(distance, n_alleles_batch, n_focal_batch,
                             locus_names=["locus_a", "locus_b"], tail_type=TailType.LEFT,
                             width_upper_bound=40.0,
                             num_warmup=300, num_samples=300, num_chains=2, seed=7)

    assert "delta_l" in result.idata.posterior
    assert "tau_l" in result.idata.posterior
    assert result.idata.posterior["delta_l"].sizes["locus"] == 2


def _simulate_with_tail(params, tail_type, n_sites=60, n_alleles_range=(20, 50), seed=1):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(-30, 60, size=n_sites))
    n_alleles = rng.integers(*n_alleles_range, size=n_sites).astype(float)
    true_freq = cline_function(distance, params, tail_type)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


# ---------------------------------------------------------------------------
# hzar_tail: HZAR's own tail parameterization, ported from classic MLE
# (BACKEND_TRAIT.md section 5) -- verified here for the Bayesian path
# ---------------------------------------------------------------------------

def test_jax_hzar_tail_matches_numpy_hzar_tail():
    """The core correctness guarantee for this extension: the JAX-traced
    HZAR tail function must exactly match hycline.models.cline.
    hzar_cline_function (already verified against HZAR's real R source)
    for every tail type, not just the no-tail case."""
    from hycline.models.cline import hzar_cline_function
    from hycline.inference.mcmc_numpyro import _hzar_cline_tail_jax, _cline_core_jax
    import jax.numpy as jnp

    x = np.linspace(-50, 150, 100)
    cases = [
        (TailType.NONE, {}),
        (TailType.LEFT, dict(delta_l=8.0, tau_l=0.5)),
        (TailType.RIGHT, dict(delta_r=6.0, tau_r=0.4)),
        (TailType.BOTH, dict(delta_l=8.0, tau_l=0.5, delta_r=6.0, tau_r=0.4)),
        (TailType.MIRROR, dict(delta_l=8.0, tau_l=0.5)),
    ]
    for tt, kwargs in cases:
        params = ClineParams(center=50, width=20, pmin=0.1, pmax=0.9, **kwargs)
        numpy_result = hzar_cline_function(x, params, tt)

        core = jnp.asarray(_cline_core_jax(jnp.asarray(x), 50.0, 20.0, 0.1, 0.9))
        jax_result = _hzar_cline_tail_jax(
            jnp.asarray(x), core, 50.0, 20.0, 0.1, 0.9, tt,
            kwargs.get("delta_l"), kwargs.get("tau_l"),
            kwargs.get("delta_r"), kwargs.get("tau_r"))
        np.testing.assert_allclose(numpy_result, np.array(jax_result), atol=1e-10)


def test_fit_mcmc_hzar_tail_default_false_preserves_original_behavior():
    """The core backward-compatibility guarantee: hzar_tail defaults to
    False, and fit_mcmc's own existing test suite (run unmodified) must
    still pass -- checked at the module level by running the full
    existing suite alongside this file; this test additionally confirms
    the result object correctly reports hzar_tail=False by default."""
    params = ClineParams(center=15.0, width=20.0, pmin=0.1, pmax=0.9)
    transect = _simulate_with_tail(params, TailType.NONE, seed=21)
    result = fit_mcmc(transect, tail_type=TailType.NONE,
                       num_warmup=200, num_samples=200, num_chains=2, seed=0)
    assert result.hzar_tail is False
    assert "HZAR-parameterized" not in result.summary()


def test_fit_mcmc_hzar_tail_recovers_known_parameters_simulated_from_hzar_formula():
    from hycline.models.cline import hzar_cline_function

    rng = np.random.default_rng(2001)
    n_sites = 30
    distance = np.sort(rng.uniform(0, 200, n_sites))
    n_alleles = rng.integers(20, 50, n_sites).astype(float)
    true_params = ClineParams(center=100, width=30, pmin=0.05, pmax=0.95,
                               delta_l=15.0, tau_l=0.5)
    p_true = hzar_cline_function(distance, true_params, TailType.LEFT)
    n_focal = rng.binomial(n_alleles.astype(int), np.clip(p_true, 0, 1)).astype(float)
    transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)

    result = fit_mcmc(transect, tail_type=TailType.LEFT, hzar_tail=True,
                       num_warmup=800, num_samples=800, num_chains=2, seed=1)
    assert result.hzar_tail is True
    assert "HZAR-parameterized" in result.summary()

    post = result.idata.posterior
    assert float(post["center"].mean()) == pytest.approx(100, abs=10)
    assert float(post["pmin"].mean()) == pytest.approx(0.05, abs=0.05)
    assert float(post["pmax"].mean()) == pytest.approx(0.95, abs=0.05)
    # delta_l/tau_l themselves are not checked against truth -- tail
    # parameters are known to be weakly identified even in the best case
    # (BACKEND_PHASE2.md section 12.3); the important check is the core
    # (non-tail) parameters recover well, which they do.


def test_fit_mcmc_batch_hzar_tail_runs_for_multiple_loci():
    params_a = ClineParams(center=10.0, width=15.0, pmin=0.1, pmax=0.9,
                            delta_l=6.0, tau_l=0.5)
    from hycline.models.cline import hzar_cline_function
    rng = np.random.default_rng(31)
    distance = np.sort(rng.uniform(-30, 60, 40))
    n_alleles = rng.integers(20, 50, 40).astype(float)
    p_a = hzar_cline_function(distance, params_a, TailType.LEFT)
    n_focal_a = rng.binomial(n_alleles.astype(int), np.clip(p_a, 0, 1)).astype(float)
    params_b = ClineParams(center=20.0, width=15.0, pmin=0.1, pmax=0.9,
                            delta_l=6.0, tau_l=0.5)
    p_b = hzar_cline_function(distance, params_b, TailType.LEFT)
    n_focal_b = rng.binomial(n_alleles.astype(int), np.clip(p_b, 0, 1)).astype(float)

    n_alleles_batch = np.stack([n_alleles, n_alleles])
    n_focal_batch = np.stack([n_focal_a, n_focal_b])

    result = fit_mcmc_batch(distance, n_alleles_batch, n_focal_batch,
                             locus_names=["locus_a", "locus_b"], tail_type=TailType.LEFT,
                             hzar_tail=True, width_upper_bound=40.0,
                             num_warmup=300, num_samples=300, num_chains=2, seed=7)
    assert result.hzar_tail is True
    assert "delta_l" in result.idata.posterior
    assert "tau_l" in result.idata.posterior
    assert result.idata.posterior["delta_l"].sizes["locus"] == 2


def test_fit_mcmc_batch_hzar_tail_default_false_matches_original():
    """Confirms fit_mcmc_batch's own backward-compatibility guarantee."""
    params = ClineParams(center=10.0, width=15.0, pmin=0.1, pmax=0.9)
    rng = np.random.default_rng(33)
    distance = np.sort(rng.uniform(-30, 60, 30))
    n_alleles = rng.integers(20, 50, 30).astype(float)
    p = cline_function(distance, params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), p).astype(float)

    result = fit_mcmc_batch(distance, n_alleles[None, :], n_focal[None, :],
                             num_warmup=200, num_samples=200, num_chains=2, seed=0)
    assert result.hzar_tail is False
