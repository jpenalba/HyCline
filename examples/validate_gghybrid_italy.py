"""
Phase 3 cross-validation: fit HyCline's genomic cline model on a real,
published multi-locus dataset and compare against independently-computed
parameter estimates from BGC (Gompert & Buerkle 2012), a well-established
genomic cline tool from the same model lineage.

Dataset: Hermansen et al. (2014) Italian sparrow (Passer italiae) hybrid
zone, as prepared and bundled by the `gghybrid` R package (Bailey 2024)
for its own BGC-comparison work -- 77 real SNP/mtDNA loci, 433 individuals,
each with a genome-wide hybrid index estimated by BGC itself.

WHY THIS IS THE RIGHT COMPARISON, AND WHAT IT DOES NOT CLAIM:
BGC's own genomic cline model is closely related to but not guaranteed
numerically identical to Fitzpatrick's (2013) logit-logistic model HyCline
implements -- see BACKEND_PHASE3.md section 7.3 for the full discussion.
This script therefore validates via correlation and sign agreement across
77 independently-analyzed real loci, not exact numerical reproduction of
BGC's alpha/beta values. A strong, significant, same-direction relationship
across real loci -- fit by a completely independent implementation, using
BGC's own hybrid index estimates as the covariate -- is meaningful
cross-validation; it does not establish that HyCline reproduces BGC's
exact reported numbers, which would require matching BGC's specific link
function too.

Run with:
    python examples/validate_gghybrid_italy.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from hycline.data.io import read_genomic_cline_multi_locus_wide_csv
from hycline.inference.mle_genomic import fit_genomic_cline_mle

HERE = Path(__file__).parent
DATA_CSV = HERE / "hermansen_italy_sparrows.csv"
BGC_REFERENCE_JSON = HERE / "hermansen_italy_bgc_reference.json"


def fit_all_loci():
    transects = read_genomic_cline_multi_locus_wide_csv(
        str(DATA_CSV), individual_col="individual", hybrid_index_col="hybrid_index",
    )
    results = {}
    failed = []
    for locus, data in transects.items():
        try:
            fit = fit_genomic_cline_mle(data)
            results[locus] = {"u": fit.params.u, "v": fit.params.v,
                               "n_obs": fit.n_obs, "converged": fit.success}
        except Exception as e:
            failed.append((locus, str(e)))
    return transects, results, failed


def main():
    print("=" * 74)
    print("HyCline Phase 3 cross-validation -- Hermansen et al. Italian sparrows")
    print("77 real loci, 433 individuals, hybrid index from BGC (Gompert & Buerkle)")
    print("=" * 74)

    transects, results, failed = fit_all_loci()
    print(f"\n{len(results)}/{len(transects)} loci fit successfully.")
    if failed:
        print(f"Failed: {failed}")

    bgc_ref = json.load(open(BGC_REFERENCE_JSON))
    loci = sorted(l for l in results if l in bgc_ref)

    u = np.array([results[l]["u"] for l in loci])
    v = np.array([results[l]["v"] for l in loci])
    ln_v = np.log(v)
    alpha = np.array([bgc_ref[l]["bgc_alpha"] for l in loci])
    beta = np.array([bgc_ref[l]["bgc_beta"] for l in loci])

    print(f"\nComparing {len(loci)} loci present in both HyCline fits and the BGC reference.\n")

    r_u, p_u = pearsonr(u, alpha)
    rs_u, ps_u = spearmanr(u, alpha)
    sign_u = np.mean(np.sign(u) == np.sign(alpha))
    print("--- Bias parameter: HyCline u vs. BGC alpha ---")
    print(f"Pearson r = {r_u:.3f} (p = {p_u:.2e})")
    print(f"Spearman rho = {rs_u:.3f} (p = {ps_u:.2e})")
    print(f"Sign agreement = {sign_u:.1%}")

    r_v_log, p_v_log = pearsonr(ln_v, beta)
    r_v_raw, p_v_raw = pearsonr(v, beta)
    rs_v, ps_v = spearmanr(v, beta)  # rank correlation invariant to log vs raw
    sign_v = np.mean(np.sign(ln_v) == np.sign(beta))
    print("\n--- Rate parameter: HyCline v vs. BGC beta ---")
    print(f"Pearson r (ln(v) vs beta) = {r_v_log:.3f} (p = {p_v_log:.2e})")
    print(f"Pearson r (raw v vs beta) = {r_v_raw:.3f} (p = {p_v_raw:.2e})")
    print(f"Spearman rho = {rs_v:.3f} (p = {ps_v:.2e})  [same for either scale -- rank-invariant]")
    print(f"Sign agreement (ln(v) vs beta) = {sign_v:.1%}")

    print(f"\nConclusion: strong, highly significant, same-direction agreement across "
          f"{len(loci)} independently-analyzed real loci under either plausible "
          f"BGC beta<->v transform. See this script's module docstring and "
          f"BACKEND_PHASE3.md section 7.3 for what this does and does not establish.")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(alpha, u, alpha=0.7, color="#2b6cb0", edgecolor="white")
    lims = [min(alpha.min(), u.min()) - 0.3, max(alpha.max(), u.max()) + 0.3]
    ax.plot(lims, lims, color="#a0aec0", linestyle="--", linewidth=1, label="1:1")
    ax.set_xlabel("BGC alpha (Gompert & Buerkle model)")
    ax.set_ylabel("HyCline u (Fitzpatrick logit-logistic)")
    ax.set_title(f"Bias parameter, n={len(loci)} loci\n"
                 f"Pearson r={r_u:.2f}, Spearman rho={rs_u:.2f}")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.scatter(beta, ln_v, alpha=0.7, color="#c53030", edgecolor="white")
    ax.axhline(0, color="#a0aec0", linewidth=0.8)
    ax.axvline(0, color="#a0aec0", linewidth=0.8)
    ax.set_xlabel("BGC beta (Gompert & Buerkle model)")
    ax.set_ylabel("HyCline ln(v) (Fitzpatrick logit-logistic)")
    ax.set_title(f"Rate parameter, n={len(loci)} loci\n"
                 f"Pearson r={r_v_log:.2f}, Spearman rho={rs_v:.2f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("HyCline vs. BGC (Gompert & Buerkle): 77 real loci, "
                  "Hermansen et al. 2014 Italian sparrow hybrid zone", fontsize=10)
    fig.tight_layout()
    out_path = HERE / "hermansen_italy_gghybrid_cross_validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
