"""
Demo: simulate allele-frequency data across a hybrid zone transect, fit
cline models, compare via AICc, and plot the best fit.

Run with:
    python examples/demo_synthetic_fit.py
"""

import numpy as np
import matplotlib.pyplot as plt

from hycline import AlleleFreqTransect, TailType, cline_function, ClineParams
from hycline.inference.mle import fit_mle
from hycline.selection.aicc import compare_models
from hycline.viz.plotting import plot_fit


def simulate_transect(seed: int = 0) -> AlleleFreqTransect:
    """Simulate a noisy sigmoid cline, similar in spirit to a real hybrid
    zone transect (e.g. Brumfield et al. 2001's Manacus dataset)."""
    rng = np.random.default_rng(seed)

    true_params = ClineParams(center=50.0, width=15.0, pmin=0.02, pmax=0.97)

    # Uneven sampling along the transect, denser near the cline center --
    # a realistic field-sampling pattern.
    d_center = rng.normal(loc=50, scale=20, size=25)
    d_edges = rng.uniform(low=-40, high=140, size=15)
    distance = np.sort(np.concatenate([d_center, d_edges]))

    n_alleles = rng.integers(low=10, high=40, size=len(distance)).astype(float)
    true_freq = cline_function(distance, true_params, TailType.NONE)
    n_focal = rng.binomial(n_alleles.astype(int), true_freq).astype(float)

    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal), true_params


def main():
    transect, true_params = simulate_transect(seed=42)

    print("=" * 70)
    print("HyCline Phase 1 demo -- synthetic hybrid zone transect")
    print("=" * 70)
    print(f"True parameters: center={true_params.center}, width={true_params.width}, "
          f"pmin={true_params.pmin}, pmax={true_params.pmax}")
    print(f"n sites = {len(transect)}, "
          f"total alleles sampled = {int(transect.n_alleles.sum())}")
    print()

    # --- Single fit (no tails) ---
    fit = fit_mle(transect, tail_type=TailType.NONE)
    print("--- MLE fit (no-tail model) ---")
    print(fit.summary())
    print()

    # --- Model comparison across tail variants ---
    comparison = compare_models(transect, candidates=[TailType.NONE, TailType.MIRROR])
    print("--- Model comparison (AICc) ---")
    print(comparison.summary())
    print()

    best_fit = comparison.results[comparison.best_model]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fit(transect, best_fit, ax=ax,
             title=f"Best model: {comparison.best_model.value} "
                   f"(true center={true_params.center}, true width={true_params.width})")
    fig.tight_layout()
    fig.savefig("demo_fit.png", dpi=150)
    print("Saved plot to demo_fit.png")


if __name__ == "__main__":
    main()
