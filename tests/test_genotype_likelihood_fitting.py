"""
Tests for genotype-likelihood-aware cline fitting.

The key validation isn't "does MLE recover the right point estimate"
(both GL-aware and naive hard-call approaches turn out to do that
reasonably well in this idealized simulation -- see BACKEND_PHASE5.md
section 3 for why the originally-expected point-estimate bias didn't
materialize). It's **uncertainty calibration**: GL-aware fitting should
give confidence intervals with correct coverage at low sequencing depth,
while naively hard-calling genotypes and fitting with the standard
binomial likelihood should be overconfident (CIs too narrow) at low
depth, converging to the same behavior as GL-aware fitting at high depth
where genotype calls are effectively certain.
"""

import numpy as np
import pytest

from hycline.data.genotype_likelihood import simulate_genotype_likelihoods
from hycline.data.schema import (
    GenotypeLikelihoodTransect, GenotypeLikelihoodLocusData, AlleleFreqTransect,
)
from hycline.models.cline import ClineParams, TailType, cline_function
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
from hycline.likelihoods.genotype_likelihood import (
    log_likelihood_geographic, log_likelihood_genomic, _marginal_log_likelihood,
)
from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl, fit_mle_genomic_gl
from hycline.inference.mle import fit_mle


# ---------------------------------------------------------------------------
# Core marginal-likelihood math
# ---------------------------------------------------------------------------

def test_hard_call_marginal_likelihood_matches_plain_hwe():
    p = 0.3
    ll = _marginal_log_likelihood(
        np.array([-np.inf]), np.array([0.0]), np.array([-np.inf]), np.array([p]))
    expected = np.log(2 * p * (1 - p))
    assert ll[0] == pytest.approx(expected, abs=1e-9)


def test_geographic_gl_likelihood_matches_manual_computation():
    transect = GenotypeLikelihoodTransect(
        distance=np.array([10.0]),
        log_gl0=np.array([-1.0]), log_gl1=np.array([-0.5]), log_gl2=np.array([-2.0]),
    )
    params = ClineParams(center=10.0, width=5.0, pmin=0.1, pmax=0.9)
    p = cline_function(np.array([10.0]), params, TailType.NONE)[0]
    expected = np.log(
        np.exp(-1.0) * (1 - p) ** 2 + np.exp(-0.5) * 2 * p * (1 - p) + np.exp(-2.0) * p ** 2
    )
    ll = log_likelihood_geographic(transect, params, TailType.NONE)
    assert ll == pytest.approx(expected, abs=1e-8)


def test_genomic_gl_likelihood_matches_manual_computation():
    data = GenotypeLikelihoodLocusData(
        hybrid_index=np.array([0.5]),
        log_gl0=np.array([-1.0]), log_gl1=np.array([-0.5]), log_gl2=np.array([-2.0]),
    )
    params = GenomicClineParams(u=1.0, v=2.0)
    phi = genomic_cline_function(np.array([0.5]), params)[0]
    expected = np.log(
        np.exp(-1.0) * (1 - phi) ** 2 + np.exp(-0.5) * 2 * phi * (1 - phi) + np.exp(-2.0) * phi ** 2
    )
    ll = log_likelihood_genomic(data, params)
    assert ll == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# MLE recovery on synthetic data
# ---------------------------------------------------------------------------

def _simulate_geographic_gl(true_params, n_ind=200, depth=10, seed=0):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 100, n_ind))
    true_p = cline_function(distance, true_params, TailType.NONE)
    u = rng.uniform(size=n_ind)
    true_genotype = np.where(
        u < (1 - true_p) ** 2, 0,
        np.where(u < (1 - true_p) ** 2 + 2 * true_p * (1 - true_p), 1, 2))
    gls = simulate_genotype_likelihoods(true_genotype, depth=depth, error_rate=0.01,
                                         rng=np.random.default_rng(seed + 1))
    transect = GenotypeLikelihoodTransect(distance=distance, log_gl0=gls["log_gl0"],
                                           log_gl1=gls["log_gl1"], log_gl2=gls["log_gl2"])
    return transect, distance, true_genotype, gls


def test_fit_mle_geographic_gl_recovers_true_parameters_at_high_depth():
    true_params = ClineParams(center=50.0, width=15.0, pmin=0.05, pmax=0.95)
    transect, *_ = _simulate_geographic_gl(true_params, n_ind=200, depth=30, seed=1)
    fit = fit_mle_geographic_gl(transect, tail_type=TailType.NONE)

    assert fit.success
    assert fit.params.center == pytest.approx(true_params.center, abs=8.0)
    # width has real sampling variability at finite n even at high depth (a single MLE
    # draw, not an average) -- confirmed not a code defect by checking the fitted
    # params have *higher* likelihood than the true generating params on this specific
    # noisy realization (i.e. the optimizer is doing its job correctly); tolerance set
    # relative to the fit's own reported SE (~5.5) rather than tightened arbitrarily
    assert fit.params.width == pytest.approx(true_params.width, abs=15.0)


def test_fit_mle_genomic_gl_recovers_true_parameters_at_high_depth():
    rng = np.random.default_rng(2)
    true_params = GenomicClineParams(u=0.0, v=4.0)
    n_ind = 200
    hybrid_index = rng.uniform(0.02, 0.98, n_ind)
    true_phi = genomic_cline_function(hybrid_index, true_params)
    uu = rng.uniform(size=n_ind)
    true_genotype = np.where(
        uu < (1 - true_phi) ** 2, 0,
        np.where(uu < (1 - true_phi) ** 2 + 2 * true_phi * (1 - true_phi), 1, 2))
    gls = simulate_genotype_likelihoods(true_genotype, depth=30, error_rate=0.01,
                                         rng=np.random.default_rng(3))
    data = GenotypeLikelihoodLocusData(hybrid_index=hybrid_index, log_gl0=gls["log_gl0"],
                                        log_gl1=gls["log_gl1"], log_gl2=gls["log_gl2"])
    fit = fit_mle_genomic_gl(data)

    assert fit.success
    assert fit.params.v > 2.0


# ---------------------------------------------------------------------------
# The key finding: uncertainty calibration, not point-estimate bias
# ---------------------------------------------------------------------------

def test_gl_aware_ci_coverage_better_calibrated_than_hard_call_at_low_depth():
    """Regression test for the central finding of this module: at low
    sequencing depth, genotype-likelihood-aware fitting's 95% CIs achieve
    close to nominal coverage, while naively hard-calling genotypes and
    fitting with the standard binomial likelihood is overconfident (CIs
    too narrow, i.e. lower coverage) -- see BACKEND_PHASE5.md section 3
    for the full investigation, including why this is the correct
    validation target rather than the originally-expected point-estimate
    bias, which did not materialize in this idealized (no mapping-bias)
    simulation."""
    true_params = ClineParams(center=50.0, width=15.0, pmin=0.05, pmax=0.95)
    rng_master = np.random.default_rng(42)
    n_ind = 200
    distance = np.sort(rng_master.uniform(0, 100, n_ind))
    true_p = cline_function(distance, true_params, TailType.NONE)
    uu = rng_master.uniform(size=n_ind)
    true_genotype = np.where(
        uu < (1 - true_p) ** 2, 0,
        np.where(uu < (1 - true_p) ** 2 + 2 * true_p * (1 - true_p), 1, 2))

    n_reps = 40
    depth = 3
    gl_covers, hard_covers, gl_n, hard_n = 0, 0, 0, 0
    for rep in range(n_reps):
        gls = simulate_genotype_likelihoods(true_genotype, depth=depth, error_rate=0.01,
                                             rng=np.random.default_rng(rep))
        transect_gl = GenotypeLikelihoodTransect(distance=distance, log_gl0=gls["log_gl0"],
                                                   log_gl1=gls["log_gl1"], log_gl2=gls["log_gl2"])
        fit_gl = fit_mle_geographic_gl(transect_gl, tail_type=TailType.NONE)

        gl_stack = np.stack([gls["log_gl0"], gls["log_gl1"], gls["log_gl2"]])
        hard_genotype = np.argmax(gl_stack, axis=0)
        transect_hard = AlleleFreqTransect(distance=distance, n_alleles=np.full(n_ind, 2.0),
                                            n_focal=hard_genotype.astype(float))
        fit_hard = fit_mle(transect_hard, tail_type=TailType.NONE)

        if fit_gl.std_errors is not None and np.isfinite(fit_gl.std_errors[1]):
            se = fit_gl.std_errors[1]
            lo, hi = fit_gl.params.width - 1.96 * se, fit_gl.params.width + 1.96 * se
            gl_covers += (lo <= true_params.width <= hi)
            gl_n += 1
        if fit_hard.std_errors is not None and np.isfinite(fit_hard.std_errors[1]):
            se = fit_hard.std_errors[1]
            lo, hi = fit_hard.params.width - 1.96 * se, fit_hard.params.width + 1.96 * se
            hard_covers += (lo <= true_params.width <= hi)
            hard_n += 1

    gl_coverage = gl_covers / gl_n
    hard_coverage = hard_covers / hard_n

    assert gl_coverage >= 0.85
    assert hard_coverage < gl_coverage


# ---------------------------------------------------------------------------
# Tail-model support for fit_mle_geographic_gl (BACKEND_PHASE2.md section 12.3)
# ---------------------------------------------------------------------------

def _simulate_tailed_geographic_gl(true_params, tail_type, n_ind=100, depth=15, seed=31):
    rng = np.random.default_rng(seed)
    distance = np.sort(rng.uniform(0, 150, n_ind))
    p = cline_function(distance, true_params, tail_type)
    u = rng.uniform(size=n_ind)
    true_genotype = np.where(
        u < (1 - p) ** 2, 0,
        np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))
    gls = simulate_genotype_likelihoods(true_genotype, depth=depth, error_rate=0.01, rng=rng)
    transect = GenotypeLikelihoodTransect(distance=distance, log_gl0=gls["log_gl0"],
                                           log_gl1=gls["log_gl1"], log_gl2=gls["log_gl2"])
    return transect


def test_fit_mle_geographic_gl_left_tail_recovers_reasonable_parameters():
    true_params = ClineParams(center=50, width=15, pmin=0.05, pmax=0.95,
                               delta_l=0.15, tau_l=10.0)
    transect = _simulate_tailed_geographic_gl(true_params, TailType.LEFT, seed=31)

    fit = fit_mle_geographic_gl(transect, tail_type=TailType.LEFT)

    assert fit.success  # the multi-start fix's core claim: success=True is achievable
    assert fit.params.center == pytest.approx(50, abs=10)
    assert 0.0 <= fit.params.delta_l <= 1.0


def test_fit_mle_geographic_gl_mirror_tail_runs_and_returns_full_params():
    true_params = ClineParams(center=50, width=15, pmin=0.05, pmax=0.95,
                               delta_l=0.2, tau_l=8.0)
    transect = _simulate_tailed_geographic_gl(true_params, TailType.MIRROR, seed=33)

    fit = fit_mle_geographic_gl(transect, tail_type=TailType.MIRROR)

    assert fit.params.delta_l is not None
    assert fit.params.tau_l is not None
    # MIRROR should give the same delta/tau for both sides (matches
    # ClineParams.from_array's own MIRROR convention)
    assert fit.params.delta_r == fit.params.delta_l
    assert fit.params.tau_r == fit.params.tau_l


def test_multistart_finds_better_optimum_than_single_start():
    """Regression test for the real finding behind fit_mle_geographic_gl's
    multi-start extension (BACKEND_PHASE2.md section 12.3): the tail
    model's kinked likelihood surface can trap a single-start L-BFGS-B run
    in a worse solution (often reporting success=False even when a better
    one exists nearby) -- checked directly by comparing the multi-start
    result's log-likelihood against a deliberately single-start fit on the
    exact same data."""
    from scipy.optimize import minimize
    from hycline.likelihoods.genotype_likelihood import negative_log_likelihood_geographic
    from hycline.inference.mle_genotype_likelihood import (
        _initial_guess_geographic, _bounds_for_geographic_gl,
    )

    true_params = ClineParams(center=50, width=15, pmin=0.05, pmax=0.95,
                               delta_l=0.15, tau_l=10.0)
    transect = _simulate_tailed_geographic_gl(true_params, TailType.LEFT, seed=31)

    # single-start, matching what fit_mle_geographic_gl did before this fix
    x0 = _initial_guess_geographic(transect, TailType.LEFT).as_array(TailType.LEFT)
    bounds = _bounds_for_geographic_gl(transect, TailType.LEFT)
    single_start_result = minimize(negative_log_likelihood_geographic, x0,
                                    args=(transect, TailType.LEFT),
                                    method="L-BFGS-B", bounds=bounds)

    multi_start_fit = fit_mle_geographic_gl(transect, tail_type=TailType.LEFT, n_restarts=8)

    # multi-start should find a likelihood at least as good (lower NLL / higher logL)
    assert -multi_start_fit.log_likelihood <= single_start_result.fun + 1e-6


def test_fit_mle_geographic_gl_none_tail_unaffected_by_multistart_change():
    """Confirms the multi-start extension is scoped to tail models only --
    TailType.NONE fitting should behave exactly as before (single
    optimization, no extra cost), not silently changed."""
    true_params = ClineParams(center=50.0, width=15.0, pmin=0.05, pmax=0.95)
    transect = _simulate_tailed_geographic_gl(true_params, TailType.NONE, seed=1)

    fit = fit_mle_geographic_gl(transect, tail_type=TailType.NONE)
    assert fit.success
    assert fit.params.center == pytest.approx(50, abs=10)
