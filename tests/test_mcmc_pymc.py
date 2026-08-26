import numpy as np
import pytest

from hycline.data.schema import AlleleFreqTransect
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.inference.mcmc_pymc import fit_mcmc_pymc


def _simulate(true_params, n_sites=50, n_alleles_range=(20, 50), seed=1):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(-30, 130, size=n_sites))
    n_alleles = rng.integers(*n_alleles_range, size=n_sites).astype(float)
    true_freq = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def test_fit_mcmc_pymc_recovers_known_parameters():
    true_params = ClineParams(center=35.0, width=14.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=60, n_alleles_range=(30, 60), seed=21)

    fit = fit_mcmc_pymc(transect, num_warmup=500, num_samples=500, num_chains=2, seed=0)

    post_center = float(fit.idata.posterior["center"].mean())
    post_width = float(fit.idata.posterior["width"].mean())

    assert post_center == pytest.approx(true_params.center, abs=6.0)
    assert post_width == pytest.approx(true_params.width, abs=10.0)
    assert fit.backend == "pymc"


def test_fit_mcmc_pymc_and_numpyro_agree():
    """The two backends fit the same model, so their posterior means on the
    same data should agree closely (BACKEND_PHASE2.md section 6)."""
    from hycline.inference.mcmc_numpyro import fit_mcmc

    true_params = ClineParams(center=25.0, width=10.0, pmin=0.05, pmax=0.95)
    transect = _simulate(true_params, n_sites=60, n_alleles_range=(30, 60), seed=99)

    pymc_fit = fit_mcmc_pymc(transect, num_warmup=500, num_samples=500,
                              num_chains=2, seed=1)
    numpyro_fit = fit_mcmc(transect, num_warmup=500, num_samples=500,
                            num_chains=2, seed=1)

    pymc_center = float(pymc_fit.idata.posterior["center"].mean())
    numpyro_center = float(numpyro_fit.idata.posterior["center"].mean())

    assert pymc_center == pytest.approx(numpyro_center, abs=5.0)
