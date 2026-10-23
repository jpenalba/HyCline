import numpy as np
import pytest

from hycline.data.hybrid_index import estimate_hybrid_index_ml


# ---------------------------------------------------------------------------
# estimate_hybrid_index_ml
# ---------------------------------------------------------------------------

def _simulate_genotypes(seed=3, n_ind=60, n_loci=200):
    rng = np.random.default_rng(seed)
    p0, p1 = rng.uniform(0, 1, n_loci), rng.uniform(0, 1, n_loci)
    h = rng.uniform(0, 1, n_ind)
    h[:5], h[5:10] = 0.0, 1.0
    g = rng.binomial(2, (1 - h)[:, None] * p0 + h[:, None] * p1)
    return h, g, p0, p1


def _one_hot(g):
    return [(g == k).astype(float) for k in range(3)]


def test_ml_matches_brute_force_likelihood_maximum_and_interval():
    h, g, p0, p1 = _simulate_genotypes()
    r = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)
    grid = np.linspace(0, 1, 20001)
    q0, q1 = np.clip(p0, 1e-3, 1 - 1e-3), np.clip(p1, 1e-3, 1 - 1e-3)
    pp = (1 - grid)[:, None] * q0 + grid[:, None] * q1
    P = np.stack([(1 - pp) ** 2, 2 * pp * (1 - pp), pp ** 2], axis=-1)
    for i in range(20):  # includes the h = 0 and h = 1 individuals
        L = np.stack([x[i] for x in _one_hot(g)], axis=-1)
        ll = np.log((P * L[None]).sum(axis=-1)).sum(axis=1)
        inside = grid[ll >= ll.max() - 1.920729]
        assert r["hybrid_index"][i] == pytest.approx(grid[np.argmax(ll)], abs=1e-4)
        assert r["lower"][i] == pytest.approx(inside.min(), abs=1e-4)
        assert r["upper"][i] == pytest.approx(inside.max(), abs=1e-4)


def test_ml_is_invariant_to_allele_polarity():
    _, g, p0, p1 = _simulate_genotypes()
    base = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)["hybrid_index"]
    flip = np.zeros(g.shape[1], bool)
    flip[::2] = True
    g_f = np.where(flip, 2 - g, g)
    p0_f, p1_f = np.where(flip, 1 - p0, p0), np.where(flip, 1 - p1, p1)
    np.testing.assert_allclose(estimate_hybrid_index_ml(*_one_hot(g_f), p0_f, p1_f)["hybrid_index"],
                               base, atol=1e-8)


def test_ml_missing_genotypes_are_the_same_as_dropping_them():
    _, g, p0, p1 = _simulate_genotypes()
    include = np.ones(g.shape, bool)
    include[:, :50] = False
    with_missing = estimate_hybrid_index_ml(*_one_hot(g), p0, p1, include=include)
    dropped = estimate_hybrid_index_ml(*[x[:, 50:] for x in _one_hot(g)], p0[50:], p1[50:])
    np.testing.assert_allclose(with_missing["hybrid_index"], dropped["hybrid_index"], atol=1e-8)
    assert (with_missing["n_loci_used"] == 150).all()


def test_ml_tolerates_alleles_unseen_in_the_reference_panels():
    # A hybrid carries the alternate allele at a locus fixed for the
    # reference allele in both panels: without the frequency floor this
    # locus would make every h impossible (log 0).
    g = np.array([[2, 0, 0, 2, 2]])
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    r = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)
    assert np.isfinite(r["hybrid_index"]).all()
    assert r["hybrid_index"][0] == pytest.approx(0.5, abs=0.02)


def test_ml_individual_without_usable_loci_raises():
    _, g, p0, p1 = _simulate_genotypes(n_ind=3, n_loci=10)
    include = np.ones(g.shape, bool)
    include[1] = False
    with pytest.raises(ValueError, match="no usable genotype"):
        estimate_hybrid_index_ml(*_one_hot(g), p0, p1, include=include)


# ---------------------------------------------------------------------------
# reference-panel uncertainty (parental_beta_pop0 / parental_beta_pop1)
# ---------------------------------------------------------------------------

def _panel_beta(freq, n):
    """Beta posteriors for panel frequencies from n allele copies (Jeffreys
    prior), as (a, b) arrays."""
    freq, n = np.asarray(freq, dtype=float), np.asarray(n, dtype=float)
    return freq * n + 0.5, (1 - freq) * n + 0.5


def _beta_averaged_loglik(g_row, beta0, beta1, grid):
    """log L(h) averaged over the Beta panel posteriors by Gauss-Jacobi
    quadrature (exact for the genotype probabilities, which are
    polynomials in the frequencies), one locus at a time."""
    from scipy.special import roots_jacobi

    def nodes(a, b):
        # Beta(a, b) on [0, 1] is the Jacobi weight (1-x)^(b-1) (1+x)^(a-1)
        x, w = roots_jacobi(8, b - 1, a - 1)
        return (x + 1) / 2, w / w.sum()

    out = np.zeros(grid.size)
    for l in range(len(beta0[0])):
        q0, w0 = nodes(beta0[0][l], beta0[1][l])
        q1, w1 = nodes(beta1[0][l], beta1[1][l])
        pp = (1 - grid)[:, None, None] * q0[None, :, None] + grid[:, None, None] * q1[None, None, :]
        geno = [(1 - pp) ** 2, 2 * pp * (1 - pp), pp ** 2][g_row[l]]
        out += np.log((geno * w0[None, :, None] * w1[None, None, :]).sum(axis=(1, 2)))
    return out


def test_ml_with_panel_sizes_matches_numerical_beta_average():
    # The closed form (Beta means plus gamma * Var[p]) against averaging the
    # genotype probabilities over the two Beta posteriors numerically.
    _, g, p0, p1 = _simulate_genotypes(seed=5, n_ind=6, n_loci=12)
    p0, p1 = np.round(p0 * 10) / 10, np.round(p1 * 10) / 10  # 5 + 5 diploid panels
    n0 = np.full(12, 10.0)
    n1 = np.full(12, 10.0)
    beta0, beta1 = _panel_beta(p0, n0), _panel_beta(p1, n1)
    r = estimate_hybrid_index_ml(*_one_hot(g), p0, p1, parental_beta_pop0=beta0,
                                 parental_beta_pop1=beta1)
    grid = np.linspace(0, 1, 20001)
    for i in range(6):
        ll = _beta_averaged_loglik(g[i], beta0, beta1, grid)
        inside = grid[ll >= ll.max() - 1.920729]
        assert r["hybrid_index"][i] == pytest.approx(grid[np.argmax(ll)], abs=1e-4)
        assert r["lower"][i] == pytest.approx(inside.min(), abs=1e-4)
        assert r["upper"][i] == pytest.approx(inside.max(), abs=1e-4)


def test_ml_with_large_panels_matches_fixed_frequencies():
    _, g, p0, p1 = _simulate_genotypes()
    fixed = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)
    big = np.full(g.shape[1], 1e7)
    panels = estimate_hybrid_index_ml(*_one_hot(g), p0, p1,
                                      parental_beta_pop0=_panel_beta(p0, big),
                                      parental_beta_pop1=_panel_beta(p1, big))
    np.testing.assert_allclose(panels["hybrid_index"], fixed["hybrid_index"], atol=1e-4)


def test_ml_small_panels_widen_the_interval():
    _, g, p0, p1 = _simulate_genotypes()
    p0, p1 = np.round(p0 * 10) / 10, np.round(p1 * 10) / 10
    small = np.full(g.shape[1], 10.0)
    fixed = estimate_hybrid_index_ml(*_one_hot(g), p0, p1)
    panels = estimate_hybrid_index_ml(*_one_hot(g), p0, p1,
                                      parental_beta_pop0=_panel_beta(p0, small),
                                      parental_beta_pop1=_panel_beta(p1, small))
    hybrids = slice(10, None)  # the first 10 individuals are parentals (h = 0 or 1)
    width = lambda r: (r["upper"] - r["lower"])[hybrids]
    assert width(panels).mean() > 1.05 * width(fixed).mean()


def test_ml_panel_sizes_must_be_given_together():
    _, g, p0, p1 = _simulate_genotypes(n_ind=3, n_loci=10)
    with pytest.raises(ValueError, match="together"):
        estimate_hybrid_index_ml(*_one_hot(g), p0, p1,
                                 parental_beta_pop0=_panel_beta(p0, np.full(10, 10.0)))
