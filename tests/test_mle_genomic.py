import numpy as np
import pytest

from hycline.data.schema import HybridIndexLocusData
from hycline.models.genomic_cline import GenomicClineParams, genomic_cline_function
from hycline.inference.mle_genomic import fit_genomic_cline_mle


def _simulate(true_params, n_ind=200, ploidy=2, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.02, 0.98, n_ind)
    phi = genomic_cline_function(h, true_params)
    n_alleles = np.full(n_ind, float(ploidy))
    n_focal = rng.binomial(ploidy, phi).astype(float)
    return HybridIndexLocusData(hybrid_index=h, n_alleles=n_alleles, n_focal=n_focal)


def test_mle_recovers_neutral_locus():
    true_params = GenomicClineParams(u=0.0, v=1.0)
    data = _simulate(true_params, n_ind=300, seed=1)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.u == pytest.approx(0.0, abs=0.4)
    assert fit.params.v == pytest.approx(1.0, abs=0.3)


def test_mle_recovers_barrier_locus():
    true_params = GenomicClineParams(u=0.0, v=5.0)
    data = _simulate(true_params, n_ind=300, seed=2)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.v == pytest.approx(5.0, abs=1.5)
    assert fit.params.v > 2.0  # unambiguously steeper than neutral


def test_mle_recovers_permeable_locus():
    true_params = GenomicClineParams(u=0.0, v=0.3)
    data = _simulate(true_params, n_ind=300, seed=3)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.v < 0.7  # unambiguously shallower than neutral


def test_mle_recovers_biased_locus():
    true_params = GenomicClineParams(u=2.5, v=1.0)
    data = _simulate(true_params, n_ind=300, seed=4)
    fit = fit_genomic_cline_mle(data)

    assert fit.success
    assert fit.params.u == pytest.approx(2.5, abs=0.8)
    assert fit.params.u > 1.0  # unambiguously biased toward population 1


def test_summary_runs_and_labels_barrier_locus_correctly():
    true_params = GenomicClineParams(u=0.0, v=6.0)
    data = _simulate(true_params, n_ind=300, seed=5)
    fit = fit_genomic_cline_mle(data)
    s = fit.summary()
    assert "barrier" in s.lower()


def test_summary_runs_and_labels_permeable_locus_correctly():
    true_params = GenomicClineParams(u=0.0, v=0.2)
    data = _simulate(true_params, n_ind=300, seed=6)
    fit = fit_genomic_cline_mle(data)
    s = fit.summary()
    assert "introgression" in s.lower()


# ---------------------------------------------------------------------------
# Parental frequencies integrated out over the reference panels' posteriors
# ---------------------------------------------------------------------------

def _simulate_with_panels(u=0.0, v=1.0, p0=0.3, p1=0.7, n_panel=10, n_ind=150, seed=0):
    """Hybrids drawn from the population frequencies p0/p1; the data carry
    the frequencies seen in reference panels of n_panel allele copies."""
    from hycline.models.genomic_cline import genomic_cline_allele_freq
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.02, 0.98, n_ind)
    freq = genomic_cline_allele_freq(h, GenomicClineParams(u=u, v=v), p0, p1)
    f0, f1 = rng.binomial(n_panel, p0) / n_panel, rng.binomial(n_panel, p1) / n_panel
    return HybridIndexLocusData(
        hybrid_index=h, n_alleles=np.full(n_ind, 2.0),
        n_focal=rng.binomial(2, freq).astype(float),
        parental_freq_pop0=f0, parental_freq_pop1=f1,
        parental_beta_pop0=_panel_beta(f0, n_panel), parental_beta_pop1=_panel_beta(f1, n_panel))


def _panel_beta(freq, n):
    """Beta posterior for a panel frequency from n allele copies (Jeffreys
    prior)."""
    return freq * n + 0.5, (1 - freq) * n + 0.5


def test_panel_fit_maximizes_the_integrated_likelihood():
    # Brute force: average the hybrids' likelihood over the two Beta panel
    # posteriors on a 120 x 120 grid of Beta quantiles, and maximize that.
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.stats import beta as beta_dist
    from hycline.likelihoods.genomic_binomial import log_likelihood_at
    data = _simulate_with_panels(v=2.0, seed=4)
    fit = fit_genomic_cline_mle(data)
    assert fit.parental_freqs_estimated and fit.n_params == 2
    q = (np.arange(120) + 0.5) / 120
    g0 = beta_dist.ppf(q, *data.parental_beta_pop0)
    g1 = beta_dist.ppf(q, *data.parental_beta_pop1)
    p0, p1 = (g.ravel() for g in np.meshgrid(g0, g1, indexing="ij"))

    def integrated(x):
        if x[1] <= 0:
            return np.inf
        ll = log_likelihood_at(data, GenomicClineParams(u=x[0], v=x[1]), p0, p1)
        return -(logsumexp(ll) - np.log(ll.size))

    assert fit.log_likelihood == pytest.approx(-integrated([fit.params.u, fit.params.v]), abs=2e-3)
    brute = minimize(integrated, [0.5, 1.5], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8})
    assert fit.params.u == pytest.approx(brute.x[0], abs=0.02)
    assert fit.params.v == pytest.approx(brute.x[1], abs=0.02)
    assert np.isfinite(fit.std_errors).all()
    # p0/p1 are reported as posterior means between the panel estimates
    # and what the hybrids indicate, with posterior SDs
    assert 0 < fit.parental_freqs[0] < 1 and 0 < fit.std_errors[2] < 0.2


def test_panel_fit_with_huge_panels_matches_fixed_frequencies():
    data = _simulate_with_panels(n_panel=10, seed=5)
    data.parental_beta_pop0 = _panel_beta(data.parental_freq_pop0, 1e8)
    data.parental_beta_pop1 = _panel_beta(data.parental_freq_pop1, 1e8)
    joint = fit_genomic_cline_mle(data)
    data.parental_beta_pop0 = data.parental_beta_pop1 = None
    fixed = fit_genomic_cline_mle(data)
    assert not fixed.parental_freqs_estimated
    assert joint.params.u == pytest.approx(fixed.params.u, abs=1e-3)
    assert joint.params.v == pytest.approx(fixed.params.v, abs=1e-3)
    assert joint.std_errors[1] == pytest.approx(fixed.std_errors[1], rel=1e-2)


def test_panel_fit_widens_the_rate_interval_for_small_panels():
    # The panels' sampling error is what made neutral loci look like
    # barriers or permeable loci when the frequencies were taken as exact.
    data = _simulate_with_panels(n_panel=10, seed=6)
    joint = fit_genomic_cline_mle(data)
    data.parental_beta_pop0 = data.parental_beta_pop1 = None
    fixed = fit_genomic_cline_mle(data)
    assert joint.std_errors[1] > fixed.std_errors[1]
