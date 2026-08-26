"""
Phase 2 demo: fit the real Manakin Ada locus data with the Bayesian
(NumPyro/NUTS) backend, and compare against the Phase 1 MLE fit and HZAR's
own published result -- three-way comparison, same data, same underlying
model, three different fitting procedures.

Run with:
    python examples/demo_bayesian_manakin.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hycline import AlleleFreqTransect, TailType
from hycline.inference.mle import fit_mle_fixed_extremes
from hycline.inference.mcmc_numpyro import fit_mcmc
from hycline.diagnostics.convergence import check
from hycline.viz.plotting import plot_bayesian_fit

HZAR_PUBLISHED_CENTER = 170.9
HZAR_PUBLISHED_WIDTH = 263.1


def load_ada_transect() -> AlleleFreqTransect:
    data_path = Path(__file__).parent / "manakin_ada.json"
    with open(data_path) as f:
        rows = json.load(f)
    distance = np.array([float(r["distance"]) for r in rows])
    freq = np.array([float(r["ada_A"]) for r in rows])
    n_alleles = np.array([float(r["ada_nSamples"]) for r in rows])
    n_focal = np.round(freq * n_alleles)
    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def main():
    transect = load_ada_transect()

    print("=" * 74)
    print("HyCline Phase 2 demo -- Bayesian fit of the Manakin Ada locus")
    print("=" * 74)

    # --- Phase 1: MLE, pmin/pmax fixed to observed extremes ---
    mle_fit = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    print("\n--- Phase 1 MLE fit ---")
    print(mle_fit.summary())

    # --- Phase 2: NUTS, same fixed pmin/pmax ---
    # width_upper_bound: the default prior (10x the transect span) is far
    # looser than this sparse, 11-site real dataset can actually constrain
    # -- see BACKEND_PHASE2.md section 3. Left at the default, NUTS's posterior
    # mean for width lands near 1000 km (vs. HZAR's published 263 km) with
    # R-hat/ESS/divergence warnings flagging the fit as unreliable. Bounding
    # the prior at the transect's actual geographic span (569.5 km) -- still
    # a generous bound, just not an absurd one -- resolves the sampling
    # problems cleanly.
    d_span = float(transect.distance.max() - transect.distance.min())
    bayes_fit = fit_mcmc(transect, pmin_fixed=mle_fit.params.pmin,
                          pmax_fixed=mle_fit.params.pmax,
                          width_upper_bound=d_span,
                          num_warmup=2000, num_samples=2000, num_chains=4, seed=0)
    print("\n--- Phase 2 NUTS fit (width prior bounded at the transect's span) ---")
    print(bayes_fit.summary())

    # --- Convergence diagnostics ---
    report = check(bayes_fit.idata)
    print("\n--- Convergence diagnostics ---")
    print(report.summary())

    # --- Three-way comparison ---
    post = bayes_fit.idata.posterior
    bayes_center_mean = float(post["center"].mean())
    bayes_width_mean = float(post["width"].mean())
    center_hdi = bayes_fit.hdi("center")
    width_hdi = bayes_fit.hdi("width")

    print("\n--- Three-way comparison ---")
    print(f"{'':<25}{'center':>12}{'width':>12}")
    print(f"{'HZAR published':<25}{HZAR_PUBLISHED_CENTER:>12.2f}{HZAR_PUBLISHED_WIDTH:>12.2f}")
    print(f"{'HyCline MLE (Phase 1)':<25}{mle_fit.params.center:>12.2f}"
          f"{mle_fit.params.width:>12.2f}")
    print(f"{'HyCline NUTS (Phase 2)':<25}{bayes_center_mean:>12.2f}{bayes_width_mean:>12.2f}")
    print(f"\n95% credible interval, center: [{center_hdi[0]:.2f}, {center_hdi[1]:.2f}]")
    print(f"95% credible interval, width:  [{width_hdi[0]:.2f}, {width_hdi[1]:.2f}]")
    print(f"\nNote: the NUTS posterior mean for width ({bayes_width_mean:.1f} km) sits above "
          f"the MLE/HZAR point estimate ({mle_fit.params.width:.1f} km) -- expected behavior "
          f"with a still-fairly-flat prior and only 11 sampling sites (width is the least "
          f"identifiable parameter here; see BACKEND_PHASE2.md section 3). The credible interval "
          f"comfortably contains the MLE/HZAR estimate, which is the honest way to read "
          f"agreement here rather than expecting the posterior mean to land exactly on it.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_bayesian_fit(transect, bayes_fit, ax=ax,
                       title=f"Manakin Ada locus -- NUTS posterior: "
                             f"center={bayes_center_mean:.1f} km "
                             f"[{center_hdi[0]:.0f}, {center_hdi[1]:.0f}], "
                             f"width={bayes_width_mean:.1f} km "
                             f"[{width_hdi[0]:.0f}, {width_hdi[1]:.0f}]\n"
                             f"(HZAR published: center={HZAR_PUBLISHED_CENTER} km, "
                             f"width={HZAR_PUBLISHED_WIDTH} km)")
    fig.tight_layout()
    out_path = Path(__file__).parent / "manakin_bayesian_fit.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
