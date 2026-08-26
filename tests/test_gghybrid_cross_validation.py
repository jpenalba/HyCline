"""
Regression test: HyCline's genomic cline fits on the real Hermansen et al.
Italian sparrow dataset (77 loci, 433 individuals) should stay in strong,
significant agreement with BGC's (Gompert & Buerkle) independently-computed
parameter estimates on the same data.

See examples/validate_gghybrid_italy.py for the full analysis and its
module docstring / BACKEND_PHASE3.md section 7.3 for what this validation
does and does not establish (correlation and sign agreement across real
loci from an independent implementation of a closely related model, not
exact numerical reproduction of BGC's specific link function).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr

from hycline.data.io import read_genomic_cline_multi_locus_wide_csv
from hycline.inference.mle_genomic import fit_genomic_cline_mle

EXAMPLES = Path(__file__).parent.parent / "examples"
DATA_CSV = EXAMPLES / "hermansen_italy_sparrows.csv"
BGC_REFERENCE_JSON = EXAMPLES / "hermansen_italy_bgc_reference.json"


@pytest.fixture(scope="module")
def fitted_loci():
    transects = read_genomic_cline_multi_locus_wide_csv(
        str(DATA_CSV), individual_col="individual", hybrid_index_col="hybrid_index",
    )
    results = {}
    for locus, data in transects.items():
        fit = fit_genomic_cline_mle(data)
        results[locus] = {"u": fit.params.u, "v": fit.params.v}
    return results


def test_all_77_real_loci_fit_successfully(fitted_loci):
    assert len(fitted_loci) == 77


def test_u_correlates_strongly_with_bgc_alpha(fitted_loci):
    bgc_ref = json.load(open(BGC_REFERENCE_JSON))
    loci = sorted(l for l in fitted_loci if l in bgc_ref)
    assert len(loci) == 77

    u = np.array([fitted_loci[l]["u"] for l in loci])
    alpha = np.array([bgc_ref[l]["bgc_alpha"] for l in loci])

    r, p = pearsonr(u, alpha)
    assert r > 0.7
    assert p < 1e-10

    sign_agreement = np.mean(np.sign(u) == np.sign(alpha))
    assert sign_agreement > 0.65


def test_v_correlates_strongly_with_bgc_beta(fitted_loci):
    bgc_ref = json.load(open(BGC_REFERENCE_JSON))
    loci = sorted(l for l in fitted_loci if l in bgc_ref)

    v = np.array([fitted_loci[l]["v"] for l in loci])
    ln_v = np.log(v)
    beta = np.array([bgc_ref[l]["bgc_beta"] for l in loci])

    # check both plausible transforms -- see module docstring
    r_log, p_log = pearsonr(ln_v, beta)
    r_raw, p_raw = pearsonr(v, beta)
    assert r_log > 0.5 and p_log < 1e-8
    assert r_raw > 0.5 and p_raw < 1e-8

    sign_agreement = np.mean(np.sign(ln_v) == np.sign(beta))
    assert sign_agreement > 0.7


def test_locus_names_match_between_hycline_and_bgc_reference(fitted_loci):
    """Sanity check on the data wrangling itself: every locus HyCline fit
    should have a corresponding BGC reference entry, confirming the
    locus-name join between the genotype data and the BGC output files
    (via loci.csv) didn't silently drop or misalign anything."""
    bgc_ref = json.load(open(BGC_REFERENCE_JSON))
    assert set(fitted_loci.keys()) == set(bgc_ref.keys())
