import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.api import fit_cline, Method
from hycline.inference.mle import FitResult


def _simulate(true_params, n_sites=50, n_alleles_range=(20, 50), seed=1):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(-30, 130, size=n_sites))
    n_alleles = rng.integers(*n_alleles_range, size=n_sites).astype(float)
    true_freq = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def test_fit_cline_classic_default_matches_fit_mle():
    from hycline.inference.mle import fit_mle

    true_params = ClineParams(center=30.0, width=12.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, seed=5)

    dispatched = fit_cline(transect, method="classic")
    direct = fit_mle(transect)

    assert isinstance(dispatched, FitResult)
    assert dispatched.params.center == pytest.approx(direct.params.center)
    assert dispatched.params.width == pytest.approx(direct.params.width)


def test_fit_cline_classic_with_fixed_extremes_matches_fit_mle_fixed_extremes():
    from hycline.inference.mle import fit_mle_fixed_extremes

    true_params = ClineParams(center=30.0, width=12.0, pmin=0.1, pmax=0.9)
    transect = _simulate(true_params, seed=6)

    dispatched = fit_cline(transect, method="classic", pmin_fixed=0.1, pmax_fixed=0.9)
    direct = fit_mle_fixed_extremes(transect, pmin=0.1, pmax=0.9)

    assert dispatched.n_params == 2
    assert dispatched.params.center == pytest.approx(direct.params.center)
    assert dispatched.params.width == pytest.approx(direct.params.width)


def test_fit_cline_bayesian_numpyro_dispatches_correctly():
    from hycline.inference.mcmc_numpyro import BayesianFitResult

    true_params = ClineParams(center=25.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=60, seed=7)

    result = fit_cline(transect, method="bayesian", backend="numpyro",
                        num_warmup=400, num_samples=400, num_chains=2, seed=0)

    assert isinstance(result, BayesianFitResult)
    assert result.backend == "numpyro"
    post_center = float(result.idata.posterior["center"].mean())
    assert post_center == pytest.approx(true_params.center, abs=8.0)


def test_fit_cline_bayesian_pymc_dispatches_correctly():
    from hycline.inference.mcmc_numpyro import BayesianFitResult

    true_params = ClineParams(center=25.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=60, seed=8)

    result = fit_cline(transect, method="bayesian", backend="pymc",
                        num_warmup=400, num_samples=400, num_chains=2, seed=0)

    assert isinstance(result, BayesianFitResult)  # both backends return this type
    assert result.backend == "pymc"


def test_fit_cline_bayesian_pymc_rejects_fixed_extremes():
    true_params = ClineParams(center=25.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, seed=9)

    with pytest.raises(NotImplementedError):
        fit_cline(transect, method="bayesian", backend="pymc",
                  pmin_fixed=0.05, pmax_fixed=0.95)


def test_fit_cline_rejects_unknown_method():
    true_params = ClineParams(center=25.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, seed=10)

    with pytest.raises(ValueError):
        fit_cline(transect, method="not_a_real_method")


def test_fit_cline_rejects_unknown_backend():
    true_params = ClineParams(center=25.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, seed=11)

    with pytest.raises(ValueError):
        fit_cline(transect, method="bayesian", backend="not_a_real_backend")


def test_fit_cline_importable_from_top_level_hycline():
    """fit_cline should be reachable as `hycline.fit_cline`, not just
    `hycline.api.fit_cline` -- see hycline/__init__.py."""
    import hycline
    assert hycline.fit_cline is fit_cline
    assert hycline.Method is Method
