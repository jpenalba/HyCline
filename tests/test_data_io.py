import numpy as np
import pandas as pd
import pytest

from hycline.data.io import read_allele_freq_csv, TransectCSVError


def test_read_csv_with_counts(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "distance,n_alleles,n_focal\n"
        "0,10,1\n"
        "50,20,10\n"
        "100,10,9\n"
    )
    transect = read_allele_freq_csv(str(csv_path))
    assert len(transect) == 3
    np.testing.assert_allclose(transect.n_focal, [1, 10, 9])


def test_read_csv_with_frequency_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "distance,n_alleles,freq\n"
        "0,10,0.1\n"
        "50,20,0.5\n"
        "100,10,0.9\n"
    )
    transect = read_allele_freq_csv(str(csv_path), n_focal_col=None, freq_col="freq")
    np.testing.assert_allclose(transect.n_focal, [1, 10, 9])


def test_read_csv_with_custom_column_names(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "dist_km,total_n,count_A\n"
        "0,10,1\n"
        "50,20,10\n"
    )
    transect = read_allele_freq_csv(
        str(csv_path), distance_col="dist_km",
        n_alleles_col="total_n", n_focal_col="count_A",
    )
    np.testing.assert_allclose(transect.distance, [0, 50])


def test_read_csv_with_site_labels(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "site,distance,n_alleles,n_focal\n"
        "A,0,10,1\n"
        "B,50,20,10\n"
    )
    transect = read_allele_freq_csv(str(csv_path), site_col="site")
    assert transect.site_names == ["A", "B"]


def test_missing_file_raises_actionable_error():
    with pytest.raises(TransectCSVError, match="not found"):
        read_allele_freq_csv("/nonexistent/path/data.csv")


def test_missing_column_lists_available_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("distance,n_alleles,n_focal\n0,10,1\n")
    with pytest.raises(TransectCSVError, match="wrong_name.*distance.*n_alleles.*n_focal"):
        read_allele_freq_csv(str(csv_path), n_focal_col="wrong_name")


def test_both_n_focal_col_and_freq_col_is_an_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("distance,n_alleles,n_focal,freq\n0,10,1,0.1\n")
    with pytest.raises(TransectCSVError, match="not both"):
        read_allele_freq_csv(str(csv_path), n_focal_col="n_focal", freq_col="freq")


def test_non_numeric_value_raises_actionable_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "distance,n_alleles,n_focal\n"
        "0,10,1\n"
        "abc,20,10\n"
    )
    with pytest.raises(TransectCSVError, match="non-numeric"):
        read_allele_freq_csv(str(csv_path))


def test_missing_value_raises_actionable_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "distance,n_alleles,n_focal\n"
        "0,10,1\n"
        ",20,10\n"
    )
    with pytest.raises(TransectCSVError, match="missing"):
        read_allele_freq_csv(str(csv_path))


def test_frequency_out_of_range_raises_actionable_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "distance,n_alleles,freq\n"
        "0,10,1.5\n"
    )
    with pytest.raises(TransectCSVError, match="outside \\[0, 1\\]"):
        read_allele_freq_csv(str(csv_path), n_focal_col=None, freq_col="freq")


def test_empty_csv_raises_actionable_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("distance,n_alleles,n_focal\n")
    with pytest.raises(TransectCSVError, match="no data rows"):
        read_allele_freq_csv(str(csv_path))


# ---------------------------------------------------------------------------
# n_ind / ploidy
# ---------------------------------------------------------------------------

def test_read_csv_with_n_ind_and_ploidy(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "distance,n_ind,n_focal\n"
        "0,10,1\n"
        "50,20,10\n"
    )
    transect = read_allele_freq_csv(
        str(csv_path), n_alleles_col=None, n_ind_col="n_ind", ploidy=2, n_focal_col="n_focal"
    )
    np.testing.assert_allclose(transect.n_alleles, [20, 40])


def test_n_alleles_col_and_n_ind_col_together_is_an_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("distance,n_alleles,n_ind,n_focal\n0,20,10,1\n")
    with pytest.raises(TransectCSVError, match="not both"):
        read_allele_freq_csv(str(csv_path), n_alleles_col="n_alleles", n_ind_col="n_ind")


# ---------------------------------------------------------------------------
# lat/lon coordinates
# ---------------------------------------------------------------------------

def test_read_csv_with_lat_lon_computes_distance(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "lat,lon,n_alleles,n_focal\n"
        "10.0,20.0,10,1\n"
        "10.5,20.0,20,10\n"
    )
    transect = read_allele_freq_csv(
        str(csv_path), distance_col=None, lat_col="lat", lon_col="lon"
    )
    assert transect.distance[0] == pytest.approx(0.0, abs=1e-6)
    assert transect.distance[1] > 0


def test_lat_col_without_lon_col_is_an_error(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("lat,lon,n_alleles,n_focal\n10.0,20.0,10,1\n")
    with pytest.raises(TransectCSVError, match="both --lat-col and --lon-col"):
        read_allele_freq_csv(str(csv_path), distance_col=None, lat_col="lat", lon_col=None)


# ---------------------------------------------------------------------------
# Multi-locus wide-format reader
# ---------------------------------------------------------------------------

def test_read_multi_locus_wide_csv_detects_loci(tmp_path):
    from hycline.data.io import read_multi_locus_wide_csv, detect_loci
    import pandas as pd

    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "distance,locusA.nSamples,locusA.A,locusB.nSamples,locusB.A\n"
        "0,10,0.1,10,0.9\n"
        "50,20,0.5,20,0.5\n"
        "100,10,0.9,10,0.1\n"
    )
    df = pd.read_csv(csv_path)
    assert detect_loci(df) == ["locusA", "locusB"]

    transects = read_multi_locus_wide_csv(str(csv_path))
    assert set(transects.keys()) == {"locusA", "locusB"}
    np.testing.assert_allclose(transects["locusA"].n_focal, [1, 10, 9])
    np.testing.assert_allclose(transects["locusB"].n_focal, [9, 10, 1])


def test_read_multi_locus_wide_csv_detects_loci_via_n_ind_suffix(tmp_path):
    """Regression test for a real bug: locus detection previously always
    checked n_alleles_suffix ('.nSamples' by default), even when the
    caller passed n_ind_suffix instead -- so a file using individual
    (not allele) counts under a custom suffix, with no '.nSamples'-
    suffixed columns at all, silently detected zero loci, even though
    the downstream value-extraction step already correctly prioritized
    n_ind_suffix once loci were (supposedly) found. Detection now checks
    whichever suffix the caller actually asked for."""
    from hycline.data.io import read_multi_locus_wide_csv, detect_loci
    import pandas as pd

    csv_path = tmp_path / "n_ind.csv"
    csv_path.write_text(
        "distance,locusA.n_ind,locusA.n_focal\n"
        "0,10,1\n"
        "50,20,9\n"
        "100,15,14\n"
    )
    df = pd.read_csv(csv_path)
    # detect_loci itself, called with the n_alleles default, finds nothing --
    # confirms this file genuinely has no '.nSamples' columns to begin with
    assert detect_loci(df) == []

    transects = read_multi_locus_wide_csv(
        str(csv_path), n_ind_suffix=".n_ind", freq_suffix=None,
        n_focal_suffix=".n_focal", ploidy=2)
    assert set(transects.keys()) == {"locusA"}
    np.testing.assert_allclose(transects["locusA"].n_alleles, [20, 40, 30])  # n_ind * ploidy
    np.testing.assert_allclose(transects["locusA"].n_focal, [1, 9, 14])


def test_read_multi_locus_wide_csv_matches_real_hzar_manakin_dataset():
    """The real, unmodified HZAR bundled multi-locus dataset (converted to
    plain CSV -- see examples/manakin_multilocus.csv's provenance) should
    be readable with this function's *default* suffixes, no configuration
    needed -- HZAR's own naming convention is exactly what the defaults
    match."""
    from hycline.data.io import read_multi_locus_wide_csv
    from pathlib import Path

    path = Path(__file__).parent.parent / "examples" / "manakin_multilocus.csv"
    transects = read_multi_locus_wide_csv(str(path), site_col="locationID")

    assert set(transects.keys()) == {"ada", "ak2", "gsr", "pgm2", "l5", "pscn3", "mtDNA"}
    for t in transects.values():
        assert len(t) == 11


def test_read_multi_locus_wide_csv_ada_locus_matches_single_locus_validation():
    """Cross-check: the 'ada' locus extracted from the multi-locus wide
    file should give the exact same fit as the previously-validated
    single-locus Manakin Ada data (tests/test_manakin_validation.py)."""
    from hycline.data.io import read_multi_locus_wide_csv
    from hycline.inference.mle import fit_mle_fixed_extremes
    from hycline.models.cline import TailType
    from pathlib import Path

    path = Path(__file__).parent.parent / "examples" / "manakin_multilocus.csv"
    transects = read_multi_locus_wide_csv(str(path), site_col="locationID")
    fit = fit_mle_fixed_extremes(transects["ada"], tail_type=TailType.NONE)

    assert fit.params.center == pytest.approx(170.9, abs=2.0)
    assert fit.params.width == pytest.approx(263.1, abs=3.0)


def test_read_multi_locus_wide_csv_restrict_to_subset(tmp_path):
    from hycline.data.io import read_multi_locus_wide_csv

    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "distance,locusA.nSamples,locusA.A,locusB.nSamples,locusB.A\n"
        "0,10,0.1,10,0.9\n50,20,0.5,20,0.5\n"
    )
    transects = read_multi_locus_wide_csv(str(csv_path), loci=["locusA"])
    assert set(transects.keys()) == {"locusA"}


def test_read_multi_locus_wide_csv_drops_missing_rows_per_locus_only(tmp_path):
    """Different loci can have data missing at different sites -- only the
    rows missing for a given locus should be dropped from that locus's
    transect, not from every locus."""
    from hycline.data.io import read_multi_locus_wide_csv

    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "distance,locusA.nSamples,locusA.A,locusB.nSamples,locusB.A\n"
        "0,10,0.1,10,0.9\n"
        "50,,,20,0.5\n"  # locusA missing here, locusB present
        "100,10,0.9,,\n"  # locusB missing here, locusA present
    )
    transects = read_multi_locus_wide_csv(str(csv_path))
    assert len(transects["locusA"]) == 2  # dropped the row missing locusA
    assert len(transects["locusB"]) == 2  # dropped the row missing locusB
    np.testing.assert_allclose(transects["locusA"].distance, [0, 100])
    np.testing.assert_allclose(transects["locusB"].distance, [0, 50])


def test_read_multi_locus_wide_csv_no_loci_detected_raises_actionable_error(tmp_path):
    from hycline.data.io import read_multi_locus_wide_csv

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("distance,foo,bar\n0,1,2\n")
    with pytest.raises(TransectCSVError, match="No loci detected"):
        read_multi_locus_wide_csv(str(csv_path))



# ---------------------------------------------------------------------------
# Genomic cline readers
# ---------------------------------------------------------------------------

def test_read_genomic_cline_csv_with_counts(tmp_path):
    from hycline.data.io import read_genomic_cline_csv

    csv_path = tmp_path / "genomic.csv"
    csv_path.write_text(
        "hybrid_index,n_focal\n"
        "0.1,0\n"
        "0.5,1\n"
        "0.9,2\n"
    )
    data = read_genomic_cline_csv(str(csv_path))
    np.testing.assert_allclose(data.n_alleles, [2, 2, 2])  # default ploidy=2
    np.testing.assert_allclose(data.n_focal, [0, 1, 2])


def test_read_genomic_cline_csv_with_freq(tmp_path):
    from hycline.data.io import read_genomic_cline_csv

    csv_path = tmp_path / "genomic.csv"
    csv_path.write_text(
        "hybrid_index,freq\n"
        "0.1,0.0\n"
        "0.5,0.5\n"
        "0.9,1.0\n"
    )
    data = read_genomic_cline_csv(str(csv_path), n_focal_col=None, freq_col="freq")
    np.testing.assert_allclose(data.n_focal, [0, 1, 2])


def test_read_genomic_cline_csv_hybrid_index_out_of_range_raises(tmp_path):
    from hycline.data.io import read_genomic_cline_csv, TransectCSVError

    csv_path = tmp_path / "genomic.csv"
    csv_path.write_text("hybrid_index,n_focal\n1.5,1\n")
    with pytest.raises(TransectCSVError, match="outside \\[0, 1\\]"):
        read_genomic_cline_csv(str(csv_path))


def test_read_genomic_cline_multi_locus_wide_csv(tmp_path):
    from hycline.data.io import read_genomic_cline_multi_locus_wide_csv

    csv_path = tmp_path / "genomic_multi.csv"
    csv_path.write_text(
        "hybrid_index,locusA.n,locusA.freq,locusB.n,locusB.freq\n"
        "0.1,2,0.0,2,1.0\n"
        "0.5,2,0.5,2,0.5\n"
        "0.9,2,1.0,2,0.0\n"
    )
    results = read_genomic_cline_multi_locus_wide_csv(str(csv_path))
    assert set(results.keys()) == {"locusA", "locusB"}
    np.testing.assert_allclose(results["locusA"].n_focal, [0, 1, 2])
    np.testing.assert_allclose(results["locusB"].n_focal, [2, 1, 0])


def test_read_genomic_cline_multi_locus_wide_csv_matches_synthetic_validation():
    """The synthetic multi-locus dataset used for CLI end-to-end validation
    should be readable with default settings and give known-recoverable
    parameter estimates for each locus."""
    from hycline.data.io import read_genomic_cline_multi_locus_wide_csv
    from hycline.inference.mle_genomic import fit_genomic_cline_mle
    from pathlib import Path

    path = Path(__file__).parent.parent / "examples" / "synthetic_genomic_clines.csv"
    results = read_genomic_cline_multi_locus_wide_csv(str(path), individual_col="individual")

    assert set(results.keys()) == {"neutral1", "neutral2", "barrier1", "permeable1", "biased1"}

    barrier_fit = fit_genomic_cline_mle(results["barrier1"])
    assert barrier_fit.params.v > 2.0  # true v=5.0, clearly steep

    permeable_fit = fit_genomic_cline_mle(results["permeable1"])
    assert permeable_fit.params.v < 0.7  # true v=0.3, clearly shallow

    biased_fit = fit_genomic_cline_mle(results["biased1"])
    assert biased_fit.params.u > 1.0  # true u=2.5, clearly biased


def test_read_genomic_cline_multi_locus_wide_csv_ploidy_without_n_suffix(tmp_path):
    from hycline.data.io import read_genomic_cline_multi_locus_wide_csv

    csv_path = tmp_path / "genomic_multi.csv"
    csv_path.write_text(
        "hybrid_index,locusA.freq\n"
        "0.1,0.0\n0.5,0.5\n0.9,1.0\n"
    )
    results = read_genomic_cline_multi_locus_wide_csv(
        str(csv_path), n_alleles_suffix=None, ploidy=2
    )
    np.testing.assert_allclose(results["locusA"].n_alleles, [2, 2, 2])


# ---------------------------------------------------------------------------
# Genotype-likelihood multi-locus readers (for Phase 4 Components D/E)
# ---------------------------------------------------------------------------

def test_read_gl_multi_locus_wide_csv_basic(tmp_path):
    from hycline.data.io import read_gl_multi_locus_wide_csv

    csv_path = tmp_path / "gl_multi.csv"
    csv_path.write_text(
        "distance,locusA.gl0,locusA.gl1,locusA.gl2,locusB.gl0,locusB.gl1,locusB.gl2\n"
        "0,0.0,-1.0,-2.0,-0.5,-1.5,-3.0\n"
        "10,-2.0,-1.0,0.0,-3.0,-1.5,-0.5\n"
    )
    results = read_gl_multi_locus_wide_csv(str(csv_path))
    assert set(results.keys()) == {"locusA", "locusB"}
    np.testing.assert_allclose(results["locusA"].log_gl0, [0.0, -2.0])
    np.testing.assert_allclose(results["locusB"].log_gl2, [-3.0, -0.5])


def test_read_gl_multi_locus_wide_csv_phred_conversion(tmp_path):
    from hycline.data.io import read_gl_multi_locus_wide_csv
    from hycline.data.genotype_likelihood import phred_to_log_likelihood

    csv_path = tmp_path / "gl_multi_phred.csv"
    csv_path.write_text(
        "distance,locusA.gl0,locusA.gl1,locusA.gl2\n"
        "0,0,10,20\n"
        "10,20,10,0\n"
    )
    results = read_gl_multi_locus_wide_csv(str(csv_path), phred=True)
    expected = phred_to_log_likelihood(np.array([0.0, 10.0, 20.0]))
    np.testing.assert_allclose(results["locusA"].log_gl0, [expected[0], phred_to_log_likelihood(np.array([20.0]))[0]])


def test_read_gl_multi_locus_wide_csv_no_loci_detected_raises(tmp_path):
    from hycline.data.io import read_gl_multi_locus_wide_csv, TransectCSVError

    csv_path = tmp_path / "gl_empty.csv"
    csv_path.write_text("distance,foo,bar\n0,1,2\n")
    with pytest.raises(TransectCSVError, match="No loci detected"):
        read_gl_multi_locus_wide_csv(str(csv_path))


def test_read_gl_multi_locus_wide_csv_drops_missing_rows_per_locus(tmp_path):
    from hycline.data.io import read_gl_multi_locus_wide_csv

    csv_path = tmp_path / "gl_missing.csv"
    csv_path.write_text(
        "distance,locusA.gl0,locusA.gl1,locusA.gl2\n"
        "0,0.0,-1.0,-2.0\n"
        "10,,,\n"
        "20,-1.0,0.0,-1.0\n"
    )
    results = read_gl_multi_locus_wide_csv(str(csv_path))
    assert len(results["locusA"]) == 2


def test_read_genomic_gl_multi_locus_wide_csv_basic(tmp_path):
    from hycline.data.io import read_genomic_gl_multi_locus_wide_csv

    csv_path = tmp_path / "genomic_gl_multi.csv"
    csv_path.write_text(
        "hybrid_index,locusA.gl0,locusA.gl1,locusA.gl2\n"
        "0.1,0.0,-1.0,-2.0\n"
        "0.9,-2.0,-1.0,0.0\n"
    )
    results = read_genomic_gl_multi_locus_wide_csv(str(csv_path))
    assert set(results.keys()) == {"locusA"}
    np.testing.assert_allclose(results["locusA"].log_gl1, [-1.0, -1.0])


def test_read_genomic_gl_multi_locus_wide_csv_hybrid_index_validation(tmp_path):
    from hycline.data.io import read_genomic_gl_multi_locus_wide_csv, TransectCSVError

    csv_path = tmp_path / "genomic_gl_bad.csv"
    csv_path.write_text(
        "hybrid_index,locusA.gl0,locusA.gl1,locusA.gl2\n"
        "1.5,0.0,-1.0,-2.0\n"
    )
    with pytest.raises(TransectCSVError, match="outside \\[0, 1\\]"):
        read_genomic_gl_multi_locus_wide_csv(str(csv_path))


def test_detect_gl_loci_requires_all_three_columns(tmp_path):
    import pandas as pd
    from hycline.data.io import detect_gl_loci

    df = pd.DataFrame({"locusA.gl0": [0], "locusA.gl1": [0], "other": [0]})
    assert detect_gl_loci(df) == []  # missing .gl2 for locusA

    df2 = pd.DataFrame({"locusA.gl0": [0], "locusA.gl1": [0], "locusA.gl2": [0]})
    assert detect_gl_loci(df2) == ["locusA"]


# ---------------------------------------------------------------------------
# Single-locus genotype-likelihood readers (for Phase 5's own CLI)
# ---------------------------------------------------------------------------

def test_read_gl_csv_basic(tmp_path):
    from hycline.data.io import read_gl_csv

    csv_path = tmp_path / "gl_single.csv"
    csv_path.write_text(
        "distance,log_gl0,log_gl1,log_gl2\n"
        "0,0.0,-1.0,-2.0\n"
        "10,-2.0,-1.0,0.0\n"
    )
    transect = read_gl_csv(str(csv_path))
    assert len(transect) == 2
    np.testing.assert_allclose(transect.log_gl0, [0.0, -2.0])


def test_read_gl_csv_phred(tmp_path):
    from hycline.data.io import read_gl_csv
    from hycline.data.genotype_likelihood import phred_to_log_likelihood

    csv_path = tmp_path / "gl_single_phred.csv"
    csv_path.write_text("distance,log_gl0,log_gl1,log_gl2\n0,0,10,20\n")
    transect = read_gl_csv(str(csv_path), phred=True)
    expected = phred_to_log_likelihood(np.array([0.0]))
    np.testing.assert_allclose(transect.log_gl0, expected)


def test_read_genomic_gl_csv_basic(tmp_path):
    from hycline.data.io import read_genomic_gl_csv

    csv_path = tmp_path / "genomic_gl_single.csv"
    csv_path.write_text(
        "hybrid_index,log_gl0,log_gl1,log_gl2\n"
        "0.1,0.0,-1.0,-2.0\n"
        "0.9,-2.0,-1.0,0.0\n"
    )
    data = read_genomic_gl_csv(str(csv_path))
    assert len(data) == 2
    np.testing.assert_allclose(data.log_gl2, [-2.0, 0.0])


def test_read_genomic_gl_csv_hybrid_index_validation(tmp_path):
    from hycline.data.io import read_genomic_gl_csv, TransectCSVError

    csv_path = tmp_path / "genomic_gl_bad.csv"
    csv_path.write_text("hybrid_index,log_gl0,log_gl1,log_gl2\n-0.5,0.0,-1.0,-2.0\n")
    with pytest.raises(TransectCSVError, match="outside \\[0, 1\\]"):
        read_genomic_gl_csv(str(csv_path))


def test_read_gl_csv_individual_col(tmp_path):
    from hycline.data.io import read_gl_csv

    csv_path = tmp_path / "gl_named.csv"
    csv_path.write_text(
        "sample,distance,log_gl0,log_gl1,log_gl2\n"
        "ind1,0,0.0,-1.0,-2.0\n"
        "ind2,10,-2.0,-1.0,0.0\n"
    )
    transect = read_gl_csv(str(csv_path), individual_col="sample")
    assert transect.individual_names == ["ind1", "ind2"]


# ---------------------------------------------------------------------------
# Hard-genotype-to-GL CSV converter (the "hard-genotype convenience path")
# ---------------------------------------------------------------------------

def test_convert_genotype_csv_to_gl_csv_basic(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_gl_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text(
        "individual,distance,locusA,locusB\n"
        "ind0,0,0,2\n"
        "ind1,10,1,0\n"
        "ind2,20,2,1\n"
    )
    output_path = tmp_path / "converted.csv"
    loci = convert_genotype_csv_to_gl_csv(str(input_path), str(output_path))
    assert loci == ["locusA", "locusB"]

    out_df = pd.read_csv(output_path)
    assert set(out_df.columns) == {
        "individual", "distance", "locusA.gl0", "locusA.gl1", "locusA.gl2",
        "locusB.gl0", "locusB.gl1", "locusB.gl2",
    }
    assert out_df.loc[0, "locusA.gl0"] == 0.0
    assert out_df.loc[0, "locusA.gl1"] == -1e10
    assert out_df.loc[1, "locusB.gl0"] == 0.0  # ind1 locusB=0


def test_convert_genotype_csv_to_gl_csv_handles_missing(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_gl_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text(
        "individual,locusA\n"
        "ind0,0\n"
        "ind1,\n"
    )
    output_path = tmp_path / "converted.csv"
    convert_genotype_csv_to_gl_csv(str(input_path), str(output_path))
    out_df = pd.read_csv(output_path)
    assert out_df.loc[1, "locusA.gl0"] == 0.0
    assert out_df.loc[1, "locusA.gl1"] == 0.0
    assert out_df.loc[1, "locusA.gl2"] == 0.0


def test_convert_genotype_csv_to_gl_csv_explicit_loci(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_gl_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text("individual,locusA,locusB\nind0,0,1\nind1,1,2\n")
    output_path = tmp_path / "converted.csv"
    loci = convert_genotype_csv_to_gl_csv(str(input_path), str(output_path), loci=["locusA"])
    assert loci == ["locusA"]
    out_df = pd.read_csv(output_path)
    assert "locusA.gl0" in out_df.columns
    assert "locusB.gl0" not in out_df.columns
    assert "locusB" in out_df.columns  # passed through unchanged, not converted


def test_convert_genotype_csv_to_gl_csv_no_loci_detected_raises(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_gl_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text("individual,distance\nind0,0\n")
    output_path = tmp_path / "converted.csv"
    with pytest.raises(ValueError, match="No locus columns detected"):
        convert_genotype_csv_to_gl_csv(str(input_path), str(output_path))


def test_converted_gl_csv_fits_correctly_with_fit_gl_multi(tmp_path):
    """End-to-end: convert a hard-genotype CSV, then fit it with
    fit-gl-multi's underlying reader/fitter, confirming the converter's
    output is actually consumable by the rest of the project, not just
    shaped correctly on paper."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.converter import convert_genotype_csv_to_gl_csv
    from hycline.data.io import read_gl_multi_locus_wide_csv
    from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl

    rng = np.random.default_rng(51)
    n_ind = 60
    distance = np.sort(rng.uniform(0, 100, n_ind))
    true_params = ClineParams(center=50.0, width=15.0, pmin=0.05, pmax=0.95)
    p = cline_function(distance, true_params, TailType.NONE)
    u = rng.uniform(size=n_ind)
    dosage = np.where(u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({"individual": [f"ind{i}" for i in range(n_ind)],
                  "distance": distance, "locusA": dosage}).to_csv(input_path, index=False)
    output_path = tmp_path / "converted.csv"
    convert_genotype_csv_to_gl_csv(str(input_path), str(output_path))

    transects = read_gl_multi_locus_wide_csv(str(output_path), individual_col="individual")
    fit = fit_mle_geographic_gl(transects["locusA"], tail_type=TailType.NONE)
    assert fit.params.center == pytest.approx(50.0, abs=15.0)


# ---------------------------------------------------------------------------
# Reverse hard-count converter (genotype dosage -> site-aggregated counts)
# ---------------------------------------------------------------------------

def test_convert_genotype_csv_to_hard_count_csv_matches_hand_computation(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_hard_count_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text(
        "individual,distance,site,locusA\n"
        "ind0,0,X,0\n"
        "ind1,0,X,1\n"
        "ind2,10,Y,2\n"
        "ind3,10,Y,\n"
    )
    output_path = tmp_path / "converted.csv"
    loci = convert_genotype_csv_to_hard_count_csv(
        str(input_path), str(output_path), site_col="site", distance_col="distance")
    assert loci == ["locusA"]

    out_df = pd.read_csv(output_path)
    row_x = out_df[out_df["site"] == "X"].iloc[0]
    # site X: dosage=[0,1], both valid -> n_alleles=2*2=4, n_focal=0+1=1, freq=0.25
    assert row_x["locusA.nSamples"] == 4
    assert row_x["locusA.A"] == pytest.approx(0.25)

    row_y = out_df[out_df["site"] == "Y"].iloc[0]
    # site Y: dosage=[2, NaN], one valid -> n_alleles=1*2=2, n_focal=2, freq=1.0
    assert row_y["locusA.nSamples"] == 2
    assert row_y["locusA.A"] == pytest.approx(1.0)


def test_convert_genotype_csv_to_hard_count_csv_inconsistent_distance_raises(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_hard_count_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text(
        "individual,distance,site,locusA\n"
        "ind0,0,X,0\n"
        "ind1,5,X,1\n"
    )
    output_path = tmp_path / "converted.csv"
    with pytest.raises(ValueError, match="inconsistent"):
        convert_genotype_csv_to_hard_count_csv(
            str(input_path), str(output_path), site_col="site", distance_col="distance")


def test_convert_genotype_csv_to_hard_count_csv_custom_ploidy(tmp_path):
    from hycline.data.converter import convert_genotype_csv_to_hard_count_csv

    input_path = tmp_path / "genotypes.csv"
    input_path.write_text(
        "individual,distance,site,locusA\n"
        "ind0,0,X,0\n"
        "ind1,0,X,2\n"
    )
    output_path = tmp_path / "converted.csv"
    convert_genotype_csv_to_hard_count_csv(
        str(input_path), str(output_path), site_col="site", distance_col="distance", ploidy=4)
    out_df = pd.read_csv(output_path)
    # 2 valid individuals * ploidy 4 = 8 total alleles
    assert out_df.loc[0, "locusA.nSamples"] == 8


def test_convert_genotype_csv_to_hard_count_csv_output_readable_by_fit_multi(tmp_path):
    """End-to-end: the converter's output must be directly consumable by
    the existing multi-locus reader with zero extra configuration --
    matching every other converter's own validation standard."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.converter import convert_genotype_csv_to_hard_count_csv
    from hycline.data.io import read_multi_locus_wide_csv
    from hycline.inference.mle import fit_mle

    rng = np.random.default_rng(51)
    n_sites, n_per_site = 8, 6
    site_distances = np.linspace(0, 140, n_sites)
    individual_distance = np.repeat(site_distances, n_per_site)
    n_ind = n_sites * n_per_site

    true_params = ClineParams(center=70, width=25, pmin=0.05, pmax=0.95)
    p = cline_function(individual_distance, true_params, TailType.NONE)
    u = rng.uniform(size=n_ind)
    dosage = np.where(u < (1 - p) ** 2, 0, np.where(u < (1 - p) ** 2 + 2 * p * (1 - p), 1, 2))

    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame({
        "individual": [f"ind{i}" for i in range(n_ind)], "distance": individual_distance,
        "site": [f"site{i}" for i in np.repeat(np.arange(n_sites), n_per_site)],
        "locusA": dosage,
    }).to_csv(input_path, index=False)

    output_path = tmp_path / "converted.csv"
    convert_genotype_csv_to_hard_count_csv(
        str(input_path), str(output_path), site_col="site", distance_col="distance")

    transects = read_multi_locus_wide_csv(str(output_path))
    fit = fit_mle(transects["locusA"], tail_type=TailType.NONE)
    assert fit.params.center == pytest.approx(70, abs=20)


def test_convert_genotype_csv_to_gl_csv_no_fragmentation_warning_at_scale(tmp_path):
    """Regression test for a real pandas PerformanceWarning found during
    a "thousands of loci" stress test (BACKEND_SCALE.md): looping
    column-by-column assignment onto an existing DataFrame triggers
    pandas' own fragmentation warning at scale. Confirms the fix (building
    new columns as a dict, concatenated once) produces identical output
    with no warning, at a scale large enough to have triggered it before
    the fix (500+ loci)."""
    import warnings
    import numpy as np
    from hycline.data.converter import convert_genotype_csv_to_gl_csv

    rng = np.random.default_rng(1)
    n_loci = 600
    n_ind = 20
    df = {"individual": [f"ind{i}" for i in range(n_ind)]}
    for i in range(n_loci):
        df[f"locus{i}"] = rng.integers(0, 3, size=n_ind)
    input_path = tmp_path / "genotypes.csv"
    pd.DataFrame(df).to_csv(input_path, index=False)
    output_path = tmp_path / "converted.csv"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loci = convert_genotype_csv_to_gl_csv(str(input_path), str(output_path))
        frag_warnings = [x for x in w if "fragmented" in str(x.message)]
        assert len(frag_warnings) == 0

    assert len(loci) == n_loci
    out_df = pd.read_csv(output_path)
    assert out_df.shape[0] == n_ind
    assert "locus0.gl0" in out_df.columns


# ---------------------------------------------------------------------------
# Quantitative trait readers (hycline.data.schema.TraitTransect)
# ---------------------------------------------------------------------------

def test_read_trait_csv_basic(tmp_path):
    from hycline.data.io import read_trait_csv

    csv_path = tmp_path / "trait.csv"
    csv_path.write_text(
        "site,distance,trait_mean,n,trait_sd\n"
        "s0,0,10.2,8,1.1\n"
        "s1,50,15.5,10,1.3\n"
        "s2,100,20.1,12,0.9\n"
    )
    t = read_trait_csv(str(csv_path), site_col="site", sd_col="trait_sd")
    assert len(t) == 3
    np.testing.assert_allclose(t.trait_mean, [10.2, 15.5, 20.1])
    np.testing.assert_allclose(t.n, [8, 10, 12])
    np.testing.assert_allclose(t.trait_sd, [1.1, 1.3, 0.9])
    assert t.site_names == ["s0", "s1", "s2"]


def test_read_trait_csv_no_sd_col_leaves_trait_sd_none(tmp_path):
    from hycline.data.io import read_trait_csv

    csv_path = tmp_path / "trait.csv"
    csv_path.write_text("distance,trait_mean,n\n0,10.2,8\n50,15.5,10\n")
    t = read_trait_csv(str(csv_path))
    assert t.trait_sd is None


def test_read_trait_csv_missing_mean_col_actionable_error(tmp_path):
    from hycline.data.io import read_trait_csv, TransectCSVError

    csv_path = tmp_path / "trait.csv"
    csv_path.write_text("distance,n\n0,8\n50,10\n")
    with pytest.raises(TransectCSVError, match="--mean-col"):
        read_trait_csv(str(csv_path))


def test_read_multi_trait_wide_csv_auto_detects_traits(tmp_path):
    from hycline.data.io import read_multi_trait_wide_csv

    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "distance,mass.mean,mass.n,mass.sd,beak.mean,beak.n\n"
        "0,10.0,8,1.0,5.0,8\n"
        "50,15.0,10,1.2,7.0,10\n"
        "100,20.0,12,0.8,9.0,12\n"
    )
    traits = read_multi_trait_wide_csv(str(csv_path))
    assert set(traits.keys()) == {"mass", "beak"}
    assert len(traits["mass"]) == 3
    assert traits["mass"].trait_sd is not None
    assert traits["beak"].trait_sd is None  # no beak.sd column present


def test_read_multi_trait_wide_csv_drops_missing_rows_independently_per_trait(tmp_path):
    from hycline.data.io import read_multi_trait_wide_csv

    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "distance,mass.mean,mass.n,beak.mean,beak.n\n"
        "0,10.0,8,5.0,8\n"
        "50,,10,7.0,10\n"
        "100,20.0,12,9.0,12\n"
    )
    traits = read_multi_trait_wide_csv(str(csv_path))
    assert len(traits["mass"]) == 2  # row 1 dropped (missing mass.mean)
    assert len(traits["beak"]) == 3  # unaffected


def test_read_multi_trait_wide_csv_no_traits_detected_actionable_error(tmp_path):
    from hycline.data.io import read_multi_trait_wide_csv, TransectCSVError

    csv_path = tmp_path / "multi.csv"
    csv_path.write_text("distance,foo\n0,1\n50,2\n")
    with pytest.raises(TransectCSVError, match="No traits detected"):
        read_multi_trait_wide_csv(str(csv_path))


def test_read_multi_trait_wide_csv_output_chains_into_fit_mle_trait_recovers_known_cline(tmp_path):
    """End-to-end: reader output must be directly usable by fit_mle_trait
    with zero extra configuration, recovering known true parameters."""
    from hycline.data.io import read_multi_trait_wide_csv
    from hycline.inference.mle_trait import fit_mle_trait
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(521)
    n_sites = 25
    distance = np.sort(rng.uniform(0, 200, n_sites))
    n = rng.integers(8, 20, n_sites)
    true_center, true_width, true_pmin, true_pmax = 80.0, 30.0, 5.0, 20.0
    mu = mean_cline(distance, true_center, true_width, true_pmin, true_pmax, TailType.NONE)
    var = variance_cline(distance, true_center, true_width, 1.0, 1.5, 1.0, TailType.NONE)

    site_means, site_sds = [], []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n[i])
        site_means.append(obs.mean())
        site_sds.append(obs.std())

    csv_path = tmp_path / "trait.csv"
    pd.DataFrame({"distance": distance, "size.mean": site_means, "size.n": n,
                  "size.sd": site_sds}).to_csv(csv_path, index=False)

    traits = read_multi_trait_wide_csv(str(csv_path))
    fit = fit_mle_trait(traits["size"])
    assert fit.mean_params.center == pytest.approx(true_center, abs=10)


# ---------------------------------------------------------------------------
# read_raw_trait_csv -- raw individual-level trait input (BACKEND_TRAIT.md
# section 5)
# ---------------------------------------------------------------------------

def test_read_raw_trait_csv_aggregates_correctly_with_site_col(tmp_path):
    from hycline.data.io import read_raw_trait_csv

    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "individual,site,distance,trait_value\n"
        "i1,A,0,1.0\n"
        "i2,A,0,1.2\n"
        "i3,A,0,0.9\n"
        "i4,B,50,5.0\n"
        "i5,B,50,5.5\n"
    )
    t = read_raw_trait_csv(str(csv_path), trait_col="trait_value", site_col="site")
    assert len(t) == 2
    np.testing.assert_allclose(t.n, [3, 2])
    np.testing.assert_allclose(t.trait_mean, [np.mean([1.0, 1.2, 0.9]), np.mean([5.0, 5.5])])
    np.testing.assert_allclose(t.trait_sd,
                                [np.std([1.0, 1.2, 0.9]), np.std([5.0, 5.5])])


def test_read_raw_trait_csv_groups_by_exact_distance_without_site_col(tmp_path):
    from hycline.data.io import read_raw_trait_csv

    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "distance,trait_value\n0,1.0\n0,1.2\n0,0.9\n50,5.0\n50,5.5\n"
    )
    t = read_raw_trait_csv(str(csv_path), trait_col="trait_value")
    assert len(t) == 2
    np.testing.assert_allclose(sorted(t.n), [2, 3])


def test_read_raw_trait_csv_inconsistent_distance_within_site_raises(tmp_path):
    from hycline.data.io import read_raw_trait_csv, TransectCSVError

    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("site,distance,trait_value\nA,0,1.0\nA,5,2.0\n")
    with pytest.raises(TransectCSVError, match="inconsistent"):
        read_raw_trait_csv(str(csv_path), trait_col="trait_value", site_col="site")


def test_read_raw_trait_csv_drops_missing_values_not_counted_in_n(tmp_path):
    from hycline.data.io import read_raw_trait_csv

    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("site,distance,trait_value\nA,0,1.0\nA,0,\nA,0,2.0\n")
    t = read_raw_trait_csv(str(csv_path), trait_col="trait_value", site_col="site")
    assert t.n[0] == 2


def test_read_raw_trait_csv_missing_trait_col_actionable_error(tmp_path):
    from hycline.data.io import read_raw_trait_csv, TransectCSVError

    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("site,distance\nA,0\n")
    with pytest.raises(TransectCSVError, match="--trait-col"):
        read_raw_trait_csv(str(csv_path))


def test_read_raw_trait_csv_output_chains_into_fit_mle_trait_recovers_known_cline(tmp_path):
    """End-to-end: raw individual-level data, aggregated by this reader,
    must recover known true cline parameters through fit_mle_trait."""
    from hycline.data.io import read_raw_trait_csv
    from hycline.inference.mle_trait import fit_mle_trait
    from hycline.models.cline import TailType
    from hycline.likelihoods.gaussian import mean_cline, variance_cline

    rng = np.random.default_rng(1101)
    n_sites = 25
    site_distance = np.sort(rng.uniform(0, 150, n_sites))
    n_per_site = rng.integers(8, 20, n_sites)
    true_center, true_width = 70.0, 25.0
    mu = mean_cline(site_distance, true_center, true_width, 10.0, 25.0, TailType.NONE)
    var = variance_cline(site_distance, true_center, true_width, 1.0, 2.0, 1.5, TailType.NONE)

    rows = []
    for i in range(n_sites):
        obs = rng.normal(mu[i], np.sqrt(var[i]), n_per_site[i])
        for j, v in enumerate(obs):
            rows.append({"site": f"site{i}", "distance": site_distance[i], "trait_value": v})
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    transect = read_raw_trait_csv(str(csv_path), trait_col="trait_value", site_col="site")
    fit = fit_mle_trait(transect)
    assert fit.mean_params.center == pytest.approx(true_center, abs=10)


# ---------------------------------------------------------------------------
# genomic parental-frequency columns
# ---------------------------------------------------------------------------

def test_genomic_readers_pick_up_parental_frequency_columns(tmp_path):
    import pandas as pd
    from hycline.data.io import (
        TransectCSVError, read_genomic_cline_multi_locus_wide_csv,
        read_genomic_gl_multi_locus_wide_csv,
    )
    hard = pd.DataFrame({"individual": ["a", "b"], "hybrid_index": [0.2, 0.7],
                         "L1.n": [2, 2], "L1.freq": [0.5, 1.0], "L1.p0": [0.3, 0.3],
                         "L1.p1": [0.9, 0.9], "L2.n": [2, 2], "L2.freq": [0.0, 0.5]})
    path = tmp_path / "hard.csv"
    hard.to_csv(path, index=False)
    data = read_genomic_cline_multi_locus_wide_csv(str(path), individual_col="individual")
    assert set(data) == {"L1", "L2"}
    assert (data["L1"].parental_freq_pop0, data["L1"].parental_freq_pop1) == (0.3, 0.9)
    assert (data["L2"].parental_freq_pop0, data["L2"].parental_freq_pop1) == (0.0, 1.0)

    gl = pd.DataFrame({"individual": ["a", "b"], "hybrid_index": [0.2, 0.7],
                       "L1.gl0": [-0.1, -3.0], "L1.gl1": [-2.0, -0.2], "L1.gl2": [-5.0, -2.0],
                       "L1.p0": [0.8, 0.8], "L1.p1": [0.1, 0.1]})
    path = tmp_path / "gl.csv"
    gl.to_csv(path, index=False)
    data = read_genomic_gl_multi_locus_wide_csv(str(path), individual_col="individual")
    assert (data["L1"].parental_freq_pop0, data["L1"].parental_freq_pop1) == (0.8, 0.1)

    hard["L1.p0"] = [0.3, 0.4]
    hard.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(TransectCSVError, match="one parental frequency"):
        read_genomic_cline_multi_locus_wide_csv(str(tmp_path / "bad.csv"))
    hard.drop(columns=["L1.p0"]).to_csv(tmp_path / "half.csv", index=False)
    with pytest.raises(TransectCSVError, match="only one of"):
        read_genomic_cline_multi_locus_wide_csv(str(tmp_path / "half.csv"))
