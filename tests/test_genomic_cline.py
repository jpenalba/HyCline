import numpy as np
import pytest

from hycline.models.genomic_cline import (
    GenomicClineParams, genomic_cline_allele_freq, genomic_cline_function,
)


def test_neutral_locus_reproduces_identity_line():
    """u=0, v=1 should give phi(h) = h exactly -- the neutral null
    expectation where a locus tracks genome-wide admixture exactly."""
    params = GenomicClineParams(u=0.0, v=1.0)
    h = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    phi = genomic_cline_function(h, params)
    np.testing.assert_allclose(phi, h, atol=1e-9)


def test_phi_at_half_hybrid_index_equals_inv_logit_u():
    """At h=0.5, logit(h)=0, so phi(0.5) should equal inv_logit(u)
    regardless of v."""
    for v in [0.1, 1.0, 5.0]:
        params = GenomicClineParams(u=2.0, v=v)
        phi = genomic_cline_function(np.array([0.5]), params)
        expected = 1.0 / (1.0 + np.exp(-2.0))
        assert phi[0] == pytest.approx(expected, abs=1e-9)


def test_high_v_gives_steeper_transition_than_low_v():
    """Higher v should mean a steeper (more barrier-like) transition:
    phi should be closer to 0 below h=0.5 and closer to 1 above it."""
    h = np.array([0.3, 0.7])
    steep = genomic_cline_function(h, GenomicClineParams(u=0.0, v=5.0))
    shallow = genomic_cline_function(h, GenomicClineParams(u=0.0, v=0.3))
    assert steep[0] < shallow[0]  # steeper cline is further from 0.5 below midpoint
    assert steep[1] > shallow[1]  # and further from 0.5 above midpoint


def test_positive_u_shifts_cline_toward_higher_ancestry():
    h = np.array([0.5])
    baseline = genomic_cline_function(h, GenomicClineParams(u=0.0, v=1.0))
    shifted = genomic_cline_function(h, GenomicClineParams(u=3.0, v=1.0))
    assert shifted[0] > baseline[0]


def test_phi_bounded_in_unit_interval():
    h = np.linspace(0.001, 0.999, 50)
    for u in [-10, 0, 10]:
        for v in [0.01, 1, 10]:
            phi = genomic_cline_function(h, GenomicClineParams(u=u, v=v))
            assert np.all(phi >= 0) and np.all(phi <= 1)


def test_monotonic_in_h_for_positive_v():
    h = np.linspace(0.01, 0.99, 100)
    phi = genomic_cline_function(h, GenomicClineParams(u=1.0, v=2.0))
    assert np.all(np.diff(phi) > 0)


# ---------------------------------------------------------------------------
# allele-frequency form: p0 + (p1 - p0) * phi(h)
# ---------------------------------------------------------------------------

def test_allele_freq_defaults_reproduce_phi():
    params = GenomicClineParams(u=0.7, v=2.1)
    h = np.linspace(0, 1, 25)
    np.testing.assert_allclose(genomic_cline_allele_freq(h, params),
                               genomic_cline_function(h, params))


def test_allele_freq_runs_between_parental_frequencies_either_direction():
    params = GenomicClineParams(u=0.0, v=1.0)
    h = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(genomic_cline_allele_freq(h, params, 0.57, 0.87),
                               [0.57, 0.72, 0.87], atol=1e-9)
    np.testing.assert_allclose(genomic_cline_allele_freq(h, params, 0.9, 0.3),
                               [0.9, 0.6, 0.3], atol=1e-9)


@pytest.mark.parametrize("p0,p1", [(0.0, 1.0), (1.0, 0.0), (0.57, 0.87), (0.9, 0.3)])
def test_neutral_loci_recovered_for_any_parental_frequencies(p0, p1):
    """Regression test: without parental frequencies the model forced the
    counted allele from 0 at h = 0 to 1 at h = 1, so reversed loci fitted at
    the v lower bound and a neutral 0.57/0.87 locus fitted as u ~ 1, v ~ 0.3.
    With p0/p1 every kind of neutral locus should come back near u=0, v=1."""
    from hycline.data.schema import GenotypeLikelihoodLocusData, HybridIndexLocusData
    from hycline.inference.mle_genomic import fit_genomic_cline_mle
    from hycline.inference.mle_genotype_likelihood import fit_mle_genomic_gl

    rng = np.random.default_rng(5)
    h = rng.uniform(0, 1, 600)
    g = rng.binomial(2, p0 + (p1 - p0) * h)
    hard = fit_genomic_cline_mle(HybridIndexLocusData(
        h, np.full(h.size, 2), g, parental_freq_pop0=p0, parental_freq_pop1=p1))
    gl = [np.log(np.where(g == k, 0.98, 0.01)) for k in range(3)]
    soft = fit_mle_genomic_gl(GenotypeLikelihoodLocusData(
        h, *gl, parental_freq_pop0=p0, parental_freq_pop1=p1))
    for fit in (hard, soft):
        assert abs(fit.params.u) < 0.35
        assert 0.7 < fit.params.v < 1.5


def test_equal_parental_frequencies_rejected():
    from hycline.data.schema import HybridIndexLocusData
    with pytest.raises(ValueError, match="equal"):
        HybridIndexLocusData([0.2, 0.8], [2, 2], [1, 1],
                             parental_freq_pop0=0.4, parental_freq_pop1=0.4)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        HybridIndexLocusData([0.2, 0.8], [2, 2], [1, 1], parental_freq_pop0=1.2)
