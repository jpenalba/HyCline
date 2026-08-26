"""
Phase 1 acceptance test: validate HyCline against HZAR's own published
results on real data.

Dataset: the Ada locus from the Manacus (manakin) hybrid zone in Panama
(Brumfield et al. 2001), as bundled with the HZAR R package and re-analyzed
in HZAR's own paper (Derryberry et al. 2014, Table 3 & Table 4).

Published HZAR result for Ada, Model I (pmin/pmax fixed to the observed
frequencies at the transect's geographic extremes, no tails):
    center = 170.9 km   (2-LL-unit support: 109.9 - 201.2)
    width  = 263.1 km   (2-LL-unit support: 96.2 - 630.0)
    pmin   = 0.1  (fixed)
    pmax   = 0.675 (fixed)
    logL   = -9.7
    AICc   = 23.3

Published HZAR result for Ada, Model 2 (pmin/pmax estimated, no tails):
    logL = -8.2
    AICc = 24.6

Run with:
    python examples/validate_manakin.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hycline import AlleleFreqTransect, TailType
from hycline.inference.mle import fit_mle, fit_mle_fixed_extremes
from hycline.likelihoods.binomial import hzar_style_log_likelihood
from hycline.models.cline import cline_function
from hycline.viz.plotting import plot_fit

HZAR_PUBLISHED = {
    "model_1_fixed": dict(center=170.9, center_ci=(109.9, 201.2),
                           width=263.1, width_ci=(96.2, 630.0),
                           pmin=0.1, pmax=0.675, logL=-9.7, aicc=23.3),
    "model_2_estimated": dict(logL=-8.2, aicc=24.6),
}


def load_ada_transect() -> AlleleFreqTransect:
    data_path = Path(__file__).parent / "manakin_ada.json"
    with open(data_path) as f:
        rows = json.load(f)

    distance = np.array([float(r["distance"]) for r in rows])
    freq = np.array([float(r["ada_A"]) for r in rows])
    n_alleles = np.array([float(r["ada_nSamples"]) for r in rows])
    # The published table gives rounded frequencies; recover integer counts.
    n_focal = np.round(freq * n_alleles)

    return AlleleFreqTransect(distance=distance, n_alleles=n_alleles, n_focal=n_focal)


def main():
    transect = load_ada_transect()
    print("=" * 74)
    print("HyCline Phase 1 validation -- Manacus (manakin) Ada locus")
    print("Data: Brumfield et al. (2001), via HZAR's manakinMolecular dataset")
    print("=" * 74)
    print(f"n sites = {len(transect)}")
    for d, n, nf in zip(transect.distance, transect.n_alleles, transect.n_focal):
        print(f"  distance={d:7.2f}  n_alleles={int(n):3d}  n_focal={int(nf):3d}  "
              f"freq={nf/n:.3f}")
    print()

    # --- Model I: pmin/pmax fixed to observed extremes, no tails ---
    fit1 = fit_mle_fixed_extremes(transect, tail_type=TailType.NONE)
    ref1 = HZAR_PUBLISHED["model_1_fixed"]

    print("--- Model I (pmin/pmax fixed to observed extremes, no tails) ---")
    print(fit1.summary())
    print()
    print(f"{'Parameter':<12}{'HyCline':>12}{'HZAR':>12}{'HZAR 2-LL support':>24}")
    print(f"{'center':<12}{fit1.params.center:>12.2f}{ref1['center']:>12.2f}"
          f"{'[' + f'{ref1['center_ci'][0]:.1f}, {ref1['center_ci'][1]:.1f}' + ']':>24}")
    print(f"{'width':<12}{fit1.params.width:>12.2f}{ref1['width']:>12.2f}"
          f"{'[' + f'{ref1['width_ci'][0]:.1f}, {ref1['width_ci'][1]:.1f}' + ']':>24}")
    print(f"{'logL':<12}{fit1.log_likelihood:>12.3f}{ref1['logL']:>12.3f}")
    print(f"{'AICc':<12}{fit1.aicc:>12.3f}{ref1['aicc']:>12.3f}")
    print()

    center_in_range = ref1["center_ci"][0] <= fit1.params.center <= ref1["center_ci"][1]
    width_in_range = ref1["width_ci"][0] <= fit1.params.width <= ref1["width_ci"][1]
    print(f"center within HZAR's published support interval: {center_in_range}")
    print(f"width within HZAR's published support interval:  {width_in_range}")
    print()

    # --- Model 2: pmin/pmax freely estimated, no tails ---
    fit2 = fit_mle(transect, tail_type=TailType.NONE)
    ref2 = HZAR_PUBLISHED["model_2_estimated"]

    print("--- Model 2 (pmin/pmax freely estimated, no tails) ---")
    print(fit2.summary())
    print()

    # --- HZAR-comparable log-likelihood ---
    # HZAR reports log-likelihood *relative to the saturated model* (see
    # hycline.likelihoods.binomial.hzar_style_log_likelihood docstring for
    # the full derivation from HZAR's R source), not the raw binomial
    # log-likelihood HyCline uses internally for fitting. The two differ by
    # a constant that doesn't depend on model parameters, so it doesn't
    # change the fitted values -- but it does mean the raw numbers can't be
    # compared directly. Recomputing with HZAR's own formula here:
    ll1_hzar_style = hzar_style_log_likelihood(transect, fit1.params, TailType.NONE)
    ll2_hzar_style = hzar_style_log_likelihood(transect, fit2.params, TailType.NONE)
    aic1 = 2 * 2 - 2 * ll1_hzar_style   # Model I: 2 free params (center, width)
    aic2 = 2 * 4 - 2 * ll2_hzar_style   # Model 2: 4 free params (+ pmin, pmax)

    print("--- Log-likelihood, recomputed HZAR's way (relative to the saturated model) ---")
    print(f"{'':<24}{'HyCline':>12}{'HZAR published':>16}")
    print(f"{'Model I logL':<24}{ll1_hzar_style:>12.3f}{ref1['logL']:>16.3f}")
    print(f"{'Model 2 logL':<24}{ll2_hzar_style:>12.3f}{ref2['logL']:>16.3f}")
    print(f"{'Model I AIC (uncorrected)':<24}{aic1:>12.3f}{ref1['aicc']:>16.3f}  "
          f"<- HZAR's published 'AICc' column matches plain AIC far more closely\n"
          f"{'':<24}{'':<12}{'':<16}     than a small-sample-corrected AICc at this n; "
          f"noted as an open question,\n{'':<24}{'':<12}{'':<16}     "
          f"not something this scaffold silently papers over.")
    print(f"{'Model 2 AIC (uncorrected)':<24}{aic2:>12.3f}{ref2['aicc']:>16.3f}")
    print()

    # --- AICc model selection agreement (using HyCline's own internal AICc,
    #     consistent across models, for HyCline's own model-selection logic) ---
    print("--- Model selection check (HyCline's internal AICc, self-consistent) ---")
    print(f"HZAR selected Model I as best fit (published AICc {ref1['aicc']} < {ref2['aicc']})")
    hycline_best = "Model I" if fit1.aicc < fit2.aicc else "Model 2"
    print(f"HyCline's AICc comparison selects: {hycline_best} "
          f"(internal AICc {fit1.aicc:.2f} vs {fit2.aicc:.2f})")
    print()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fit(transect, fit1, ax=ax,
             title=f"Manakin Ada locus -- HyCline fit: center={fit1.params.center:.1f} km, "
                   f"width={fit1.params.width:.1f} km\n"
                   f"(HZAR published: center={ref1['center']} km, width={ref1['width']} km)")
    fig.tight_layout()
    out_path = Path(__file__).parent / "manakin_validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
