import gzip
import numpy as np
import pytest

from hycline.data.beagle import read_beagle, write_beagle, BeagleData


BEAGLE_CONTENT = (
    "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\tind2\tind2\tind2\n"
    "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.02\t0.98\t0.49\t0.02\t0.49\n"
    "chr1_200\t0\t1\t0.60\t0.30\t0.10\t0.10\t0.30\t0.60\t0.33\t0.34\t0.33\n"
    "chr1_300\t0\t1\t1.00\t0.00\t0.00\t0.00\t0.00\t1.00\t1.00\t0.00\t0.00\n"
)


@pytest.fixture
def beagle_path(tmp_path):
    path = tmp_path / "test.beagle"
    path.write_text(BEAGLE_CONTENT)
    return str(path)


@pytest.fixture
def beagle_gz_path(tmp_path):
    path = tmp_path / "test.beagle.gz"
    with gzip.open(path, "wt") as f:
        f.write(BEAGLE_CONTENT)
    return str(path)


def test_read_beagle_basic_values(beagle_path):
    data = read_beagle(beagle_path)
    assert data.marker_names == ["chr1_100", "chr1_200", "chr1_300"]
    assert data.sample_names == ["ind0", "ind1", "ind2"]
    np.testing.assert_allclose(data.gl0[0], [0.98, 0.0, 0.49])
    np.testing.assert_allclose(data.gl1[0], [0.02, 0.02, 0.02])
    np.testing.assert_allclose(data.gl2[0], [0.0, 0.98, 0.49])
    assert len(data) == 3


def test_read_beagle_gzip_matches_plain_text(beagle_path, beagle_gz_path):
    data_plain = read_beagle(beagle_path)
    data_gz = read_beagle(beagle_gz_path)
    assert data_plain.marker_names == data_gz.marker_names
    assert data_plain.sample_names == data_gz.sample_names
    np.testing.assert_array_equal(data_plain.gl0, data_gz.gl0)
    np.testing.assert_array_equal(data_plain.gl2, data_gz.gl2)


def test_read_beagle_bad_header_raises(tmp_path):
    path = tmp_path / "bad.beagle"
    path.write_text("not\ta\tbeagle\tfile\n")
    with pytest.raises(ValueError, match="doesn't look like an ANGSD beagle file"):
        read_beagle(str(path))


def test_read_beagle_wrong_column_count_raises(tmp_path):
    path = tmp_path / "bad.beagle"
    path.write_text("marker\tallele1\tallele2\tind0\tind0\n")  # only 2, not a multiple of 3
    with pytest.raises(ValueError, match="multiple of 3"):
        read_beagle(str(path))


def test_read_beagle_empty_file_raises(tmp_path):
    path = tmp_path / "empty.beagle"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_beagle(str(path))


def test_read_beagle_no_data_rows_raises(tmp_path):
    path = tmp_path / "headeronly.beagle"
    path.write_text("marker\tallele1\tallele2\tind0\tind0\tind0\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_beagle(str(path))


def test_write_beagle_filters_rows_and_preserves_values(beagle_path, tmp_path):
    data = read_beagle(beagle_path)
    mask = np.array([True, False, True])
    out_path = tmp_path / "filtered.beagle"
    write_beagle(data, str(out_path), mask)

    filtered = read_beagle(str(out_path))
    assert filtered.marker_names == ["chr1_100", "chr1_300"]
    assert filtered.sample_names == data.sample_names
    np.testing.assert_allclose(filtered.gl0[0], data.gl0[0])
    np.testing.assert_allclose(filtered.gl2[1], data.gl2[2])  # chr1_300 preserved exactly


def test_write_beagle_gzip_output(beagle_path, tmp_path):
    data = read_beagle(beagle_path)
    mask = np.array([True, True, True])
    out_path = tmp_path / "filtered.beagle.gz"
    write_beagle(data, str(out_path), mask)
    filtered = read_beagle(str(out_path))
    assert len(filtered) == 3


# ---------------------------------------------------------------------------
# Beagle + metadata -> HyCline CSV writers
# ---------------------------------------------------------------------------

import pandas as pd
from hycline.data.beagle import (
    write_beagle_geographic_gl_csv, write_beagle_genomic_gl_csv,
    write_beagle_geographic_hard_count_csv, write_beagle_genomic_hard_count_csv,
)


@pytest.fixture
def beagle_metadata_df():
    return pd.DataFrame({
        "individual": ["ind0", "ind1", "ind2"],
        "site": ["X", "X", "Y"],
        "distance": [0, 0, 50],
        "hybrid_index": [0.1, 0.3, 0.7],
    })


def test_write_beagle_geographic_gl_csv_matches_hand_computation(beagle_path, beagle_metadata_df, tmp_path):
    data = read_beagle(beagle_path)
    out_path = tmp_path / "out.csv"
    info = write_beagle_geographic_gl_csv(data, beagle_metadata_df, "individual", "distance",
                                           str(out_path))
    assert info["n_individuals"] == 3
    assert info["n_loci"] == 3

    out_df = pd.read_csv(out_path)
    # chr1_100, ind0: gl0=0.98 -> log(0.98)
    assert out_df.loc[0, "chr1_100.gl0"] == pytest.approx(np.log(0.98))
    # a true-zero probability should floor to -1e10, not -inf or NaN
    assert out_df.loc[0, "chr1_100.gl2"] == pytest.approx(-1e10)


def test_write_beagle_geographic_hard_count_csv_uses_expected_dosage_not_argmax(
        beagle_path, beagle_metadata_df, tmp_path):
    """Regression test for a real fix: this writer used to collapse each
    individual to its most-likely genotype (argmax) before aggregating,
    discarding genotype uncertainty. It now uses the expected dosage
    (posterior mean under a flat prior), so the site frequency genuinely
    reflects the GLs, not just a point estimate of each individual's
    genotype."""
    data = read_beagle(beagle_path)
    out_path = tmp_path / "out.csv"
    info = write_beagle_geographic_hard_count_csv(
        data, beagle_metadata_df, "individual", "site", "distance", str(out_path))
    assert info["n_sites"] == 2

    out_df = pd.read_csv(out_path)
    row_x = out_df[out_df["site"] == "X"].iloc[0]
    # site X: ind0 (gl=0.98,0.02,0.00 -> E[dosage]=0.02), ind1 (gl=0,0.02,0.98 -> E[dosage]=1.98)
    # n_alleles = 4, n_focal = 0.02+1.98 = 2.0, freq = 0.5 (matches the old argmax result here,
    # since these two individuals' uncertainty happens to cancel out symmetrically)
    assert row_x["chr1_100.nSamples"] == 4
    assert row_x["chr1_100.A"] == pytest.approx(0.5)

    row_y = out_df[out_df["site"] == "Y"].iloc[0]
    # site Y: ind2 alone, gl=(0.49, 0.02, 0.49) -- genuinely ambiguous between the two
    # homozygous classes. Argmax would arbitrarily pick dosage=0 (freq=0.0); the expected
    # dosage correctly reflects the near-50/50 uncertainty: E[dosage] = (0.02+0.98)/1.00 = 1.0,
    # freq = 1.0/2 = 0.5 -- a real, not just theoretical, difference from the old behavior.
    assert row_y["chr1_100.nSamples"] == 2
    assert row_y["chr1_100.A"] == pytest.approx(0.5)


def test_write_beagle_geographic_hard_count_csv_excludes_missing_individuals_from_count(
        beagle_metadata_df, tmp_path):
    """The expected-dosage calculation can produce NaN for an individual
    with all-zero genotype likelihoods (genuinely missing data) -- unlike
    argmax, which always returns some dosage even for degenerate input.
    That individual must be excluded from both n_alleles and the
    frequency's numerator, the same way hycline.data.vcf's hard-count
    writer already excludes missing (NaN) dosage."""
    content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\tind2\tind2\tind2\n"
        "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.00\t0.00\t0.00\t0.49\t0.02\t0.49\n"
    )
    beagle_path = tmp_path / "test_missing.beagle"
    beagle_path.write_text(content)
    data = read_beagle(str(beagle_path))

    out_path = tmp_path / "out.csv"
    write_beagle_geographic_hard_count_csv(
        data, beagle_metadata_df, "individual", "site", "distance", str(out_path))
    out_df = pd.read_csv(out_path)
    row_x = out_df[out_df["site"] == "X"].iloc[0]
    # site X: ind0 valid (E[dosage]=0.02), ind1 all-zero GLs (missing, excluded)
    # n_alleles = 1 individual * 2 ploidy = 2 (not 4), n_focal = 0.02, freq = 0.01
    assert row_x["chr1_100.nSamples"] == 2
    assert row_x["chr1_100.A"] == pytest.approx(0.02 / 1.00 / 2, abs=1e-6)


def test_write_beagle_genomic_gl_csv_hybrid_index_validation(beagle_path, tmp_path):
    bad_metadata = pd.DataFrame({
        "individual": ["ind0", "ind1", "ind2"], "hybrid_index": [0.1, 1.5, 0.5],
    })
    data = read_beagle(beagle_path)
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        write_beagle_genomic_gl_csv(data, bad_metadata, "individual", "hybrid_index",
                                     str(tmp_path / "out.csv"))


def test_write_beagle_genomic_hard_count_csv_basic(beagle_path, beagle_metadata_df, tmp_path):
    data = read_beagle(beagle_path)
    out_path = tmp_path / "out.csv"
    info = write_beagle_genomic_hard_count_csv(
        data, beagle_metadata_df, "individual", "hybrid_index", str(out_path))
    assert info["n_individuals"] == 3
    out_df = pd.read_csv(out_path)
    assert "chr1_100.n" in out_df.columns
    assert "chr1_100.freq" in out_df.columns
    assert (out_df["chr1_100.n"] == 2).all()


def test_all_four_beagle_writer_outputs_are_directly_readable(beagle_path, beagle_metadata_df, tmp_path):
    """The core end-to-end correctness check: every beagle writer's
    output must be readable by the corresponding existing reader with
    zero extra configuration -- the same standard the VCF writers are
    held to."""
    from hycline.data.io import (
        read_multi_locus_wide_csv, read_gl_multi_locus_wide_csv,
        read_genomic_cline_multi_locus_wide_csv, read_genomic_gl_multi_locus_wide_csv,
    )

    data = read_beagle(beagle_path)

    p1 = tmp_path / "geo_gl.csv"
    write_beagle_geographic_gl_csv(data, beagle_metadata_df, "individual", "distance", str(p1))
    assert len(read_gl_multi_locus_wide_csv(str(p1), individual_col="individual")) == 3

    p2 = tmp_path / "geo_hard.csv"
    write_beagle_geographic_hard_count_csv(data, beagle_metadata_df, "individual", "site",
                                            "distance", str(p2))
    assert len(read_multi_locus_wide_csv(str(p2))) == 3

    p3 = tmp_path / "gen_gl.csv"
    write_beagle_genomic_gl_csv(data, beagle_metadata_df, "individual", "hybrid_index", str(p3))
    assert len(read_genomic_gl_multi_locus_wide_csv(str(p3), individual_col="individual")) == 3

    p4 = tmp_path / "gen_hard.csv"
    write_beagle_genomic_hard_count_csv(data, beagle_metadata_df, "individual",
                                         "hybrid_index", str(p4))
    assert len(read_genomic_cline_multi_locus_wide_csv(str(p4), individual_col="individual")) == 3


def test_write_beagle_geographic_gl_csv_recovers_known_cline(tmp_path):
    """Real end-to-end validation against known ground truth, not just
    format-level round-tripping."""
    import numpy as np
    from hycline.models.cline import ClineParams, TailType, cline_function
    from hycline.data.io import read_gl_multi_locus_wide_csv
    from hycline.inference.mle_genotype_likelihood import fit_mle_geographic_gl

    rng = np.random.default_rng(81)
    n_hybrid = 80
    distance = np.sort(rng.uniform(0, 150, n_hybrid))
    sample_names = [f"ind{i}" for i in range(n_hybrid)]

    true_params = ClineParams(center=70, width=25, pmin=0.05, pmax=0.95)
    p_true = cline_function(distance, true_params, TailType.NONE)
    u = rng.uniform(size=n_hybrid)
    true_geno = np.where(u < (1 - p_true) ** 2, 0,
                          np.where(u < (1 - p_true) ** 2 + 2 * p_true * (1 - p_true), 1, 2))

    def make_gl(true_g, noise=0.05):
        gl = np.full(3, noise)
        gl[true_g] = 1.0 - 2 * noise
        return gl / gl.sum()

    header = ["marker", "allele1", "allele2"] + sum([[s, s, s] for s in sample_names], [])
    row = ["locusA", "0", "1"]
    for g in true_geno:
        row += [f"{v:.4f}" for v in make_gl(g)]
    beagle_path = tmp_path / "test.beagle"
    beagle_path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n")

    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"individual": sample_names, "distance": distance}).to_csv(
        metadata_path, sep="\t", index=False)

    data = read_beagle(str(beagle_path))
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    gl_path = tmp_path / "converted.csv"
    write_beagle_geographic_gl_csv(data, metadata_df, "individual", "distance", str(gl_path))

    transects = read_gl_multi_locus_wide_csv(str(gl_path), individual_col="individual")
    fit = fit_mle_geographic_gl(transects["locusA"], tail_type=TailType.NONE)
    assert fit.params.center == pytest.approx(70, abs=15)


def test_beagle_hard_count_writers_treat_equal_likelihoods_as_missing(beagle_metadata_df, tmp_path):
    """ANGSD writes 0.333333 x3 for an individual with no reads. The
    geographic writer counted it as expected dosage 1 (a heterozygote) and
    the genomic writer as argmax dosage 0; both must treat it as missing,
    the same way the VCF writers treat ./."""
    content = (
        "marker\tallele1\tallele2\tind0\tind0\tind0\tind1\tind1\tind1\tind2\tind2\tind2\n"
        "chr1_100\t0\t1\t0.98\t0.02\t0.00\t0.333333\t0.333333\t0.333333\t0.00\t0.02\t0.98\n"
    )
    path = tmp_path / "missing.beagle"
    path.write_text(content)
    data = read_beagle(str(path))

    geo = tmp_path / "geo.csv"
    write_beagle_geographic_hard_count_csv(data, beagle_metadata_df, "individual", "site",
                                            "distance", str(geo))
    row_x = pd.read_csv(geo).set_index("site").loc["X"]
    assert row_x["chr1_100.nSamples"] == 2  # ind0 only
    assert row_x["chr1_100.A"] == pytest.approx(0.01)

    gen = tmp_path / "gen.csv"
    write_beagle_genomic_hard_count_csv(data, beagle_metadata_df, "individual",
                                         "hybrid_index", str(gen))
    out = pd.read_csv(gen).set_index("individual")
    assert out.loc["ind1", "chr1_100.n"] == 0
    assert np.isnan(out.loc["ind1", "chr1_100.freq"])
    assert out.loc["ind2", "chr1_100.freq"] == 1.0
