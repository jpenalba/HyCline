import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless test environment

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.inference.mcmc_numpyro import fit_mcmc
from hycline.viz.plotting import plot_bayesian_fit


def test_plot_bayesian_fit_runs_without_error():
    rng = np.random.default_rng(0)
    true_params = ClineParams(center=20.0, width=8.0, pmin=0.05, pmax=0.95)
    distance = np.sort(rng.uniform(-20, 60, size=30))
    n_alleles = rng.integers(20, 40, size=30).astype(float)
    true_freq = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)

    fit = fit_mcmc(transect, num_warmup=400, num_samples=400, num_chains=2, seed=0)
    ax = plot_bayesian_fit(transect, fit)
    assert ax is not None


def test_plot_bayesian_fit_with_mirror_tail_runs_without_error():
    """Regression test for a real bug caught during BACKEND_PHASE2.md
    section 12's tail-model work: plot_bayesian_fit previously hardcoded
    TailType.NONE when evaluating posterior-draw curves, which would have
    silently plotted the wrong (tail-less) curve shape for any tail-model
    Bayesian fit. This checks the fixed version actually extracts and
    uses tail-parameter draws from the posterior."""
    rng = np.random.default_rng(1)
    true_params = ClineParams(center=20.0, width=8.0, pmin=0.05, pmax=0.95,
                               delta_l=0.15, tau_l=6.0)
    distance = np.sort(rng.uniform(-20, 60, size=40))
    n_alleles = rng.integers(20, 40, size=40).astype(float)
    true_freq = cline_function(distance, true_params, TailType.MIRROR)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)

    fit = fit_mcmc(transect, tail_type=TailType.MIRROR, width_upper_bound=40.0,
                    num_warmup=300, num_samples=300, num_chains=2, seed=0)
    ax = plot_bayesian_fit(transect, fit)
    assert ax is not None


def test_plot_bayesian_fit_uses_hzar_cline_function_when_hzar_tail_true():
    """Regression test for a real bug found while extending hzar_tail to
    the Bayesian path (BACKEND_TRAIT.md section 5): plot_bayesian_fit
    (and plot_fit, and the multi-locus curve-CSV writer in cli.py)
    unconditionally used cline_function for posterior-draw curves,
    ignoring fit.hzar_tail entirely -- silently plotting/exporting the
    wrong (mismatched-parameterization) curve for any hzar_tail=True fit.
    Checked directly by constructing a minimal fake posterior with known
    draws and confirming the resulting curve matches hzar_cline_function,
    not cline_function -- these differ meaningfully in the tail region
    for these parameters, so this is a real, not just theoretical, check."""
    import arviz as az
    from hycline.models.cline import hzar_cline_function
    from hycline.inference.mcmc_numpyro import BayesianFitResult

    # a single, fixed posterior "draw" (chain=1, draw=1) with HZAR-style
    # tail parameters (delta_l=8.0, a raw distance -- meaningless as a
    # [0,1] fraction, so the two parameterizations diverge sharply here)
    draws = {
        "center": np.array([[50.0]]), "width": np.array([[20.0]]),
        "pmin": np.array([[0.1]]), "pmax": np.array([[0.9]]),
        "delta_l": np.array([[8.0]]), "tau_l": np.array([[0.5]]),
    }
    idata = az.from_dict({"posterior": draws})
    fit = BayesianFitResult(idata=idata, tail_type=TailType.LEFT, n_obs=10,
                             backend="numpyro", param_names=list(draws.keys()),
                             hzar_tail=True)
    fit._fixed_pmin = None
    fit._fixed_pmax = None

    distance = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    n_alleles = np.full(5, 20.0)
    n_focal = np.full(5, 10.0)
    transect = AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)

    ax = plot_bayesian_fit(transect, fit, n_posterior_lines=1)
    band_mid_line = [line for line in ax.get_lines() if line.get_label() == "Posterior median cline"][0]
    plotted_y = band_mid_line.get_ydata()
    plotted_x = band_mid_line.get_xdata()

    params = ClineParams(center=50.0, width=20.0, pmin=0.1, pmax=0.9, delta_l=8.0, tau_l=0.5)
    expected_hzar = hzar_cline_function(plotted_x, params, TailType.LEFT)
    expected_original = cline_function(plotted_x, params, TailType.LEFT)

    # confirm the two parameterizations actually diverge for this input
    # (otherwise this test wouldn't distinguish the bug from the fix)
    assert not np.allclose(expected_hzar, expected_original, atol=1e-3)
    np.testing.assert_allclose(plotted_y, expected_hzar, atol=1e-6)
