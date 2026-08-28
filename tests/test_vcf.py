"""
Tests for hycline.data.vcf's pure-Python VCF parser.

Uses small, hand-crafted VCF fixtures built directly by these tests
(not external files) so the exact expected values are known and checked
precisely, the same rigor applied to every other reader in this project.
"""

import gzip
import numpy as np
import pytest

from hycline.data.vcf import read_vcf, _parse_gt, VCFData


TEST_VCF_CONTENT = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsampleA\tsampleB\tsampleC\tsampleD
chr1\t100\t.\tA\tG\t.\tPASS\t.\tGT:PL\t0/0:0,10,20\t0/1:10,0,10\t1/1:20,10,0\t./.:.
chr1\t200\trs123\tC\tT\t.\tPASS\t.\tGT:PL\t0/1:5,0,15\t0/0:0,15,25\t1/1:30,10,0\t0/1:8,0,12
chr1\t300\t.\tA\tAT\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1\t0/0
chr1\t400\t.\tG\tA,C\t.\tPASS\t.\tGT\t0/1\t0/2\t1/1\t0/0
chr1\t500\t.\tT\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t./.
chr1\t600\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\t0|0\t1|0
"""


@pytest.fixture
def test_vcf_path(tmp_path):
    path = tmp_path / "test.vcf"
    path.write_text(TEST_VCF_CONTENT)
    return str(path)


@pytest.fixture
def test_vcf_gz_path(tmp_path):
    path = tmp_path / "test.vcf.gz"
    with gzip.open(path, "wt") as f:
        f.write(TEST_VCF_CONTENT)
    return str(path)


def test_parse_gt_unphased():
    assert _parse_gt("0/0") == 0.0
    assert _parse_gt("0/1") == 1.0
    assert _parse_gt("1/0") == 1.0
    assert _parse_gt("1/1") == 2.0


def test_parse_gt_phased():
    assert _parse_gt("0|1") == 1.0
    assert _parse_gt("1|1") == 2.0


def test_parse_gt_missing():
    assert np.isnan(_parse_gt("./."))
    assert np.isnan(_parse_gt("."))
    assert np.isnan(_parse_gt(".|."))


def test_read_vcf_basic_dosages(test_vcf_path):
    data = read_vcf(test_vcf_path)
    assert data.sample_names == ["sampleA", "sampleB", "sampleC", "sampleD"]
    # 4 biallelic SNPs (100, rs123/200, 500, 600) -- 300 (indel) and 400
    # (multiallelic) should be skipped
    assert data.locus_names == ["chr1:100", "rs123", "chr1:500", "chr1:600"]
    assert data.n_skipped_indel == 1
    assert data.n_skipped_multiallelic == 1

    expected_locus0 = [0.0, 1.0, 2.0, np.nan]  # chr1:100
    np.testing.assert_array_equal(data.dosage[0], expected_locus0)

    expected_locus3 = [1.0, 2.0, 0.0, 1.0]  # chr1:600, phased genotypes
    np.testing.assert_array_equal(data.dosage[3], expected_locus3)


def test_read_vcf_gzip_matches_plain_text(test_vcf_path, test_vcf_gz_path):
    data_plain = read_vcf(test_vcf_path, extract_pl=True)
    data_gz = read_vcf(test_vcf_gz_path, extract_pl=True)
    assert data_plain.locus_names == data_gz.locus_names
    np.testing.assert_array_equal(data_plain.dosage, data_gz.dosage)
    np.testing.assert_array_equal(data_plain.log_gl0, data_gz.log_gl0)


def test_read_vcf_pl_extraction_matches_hand_computed_values(test_vcf_path):
    from hycline.data.genotype_likelihood import phred_to_log_likelihood

    data = read_vcf(test_vcf_path, extract_pl=True)
    # chr1:100, sampleA: PL=0,10,20
    expected = phred_to_log_likelihood(np.array([0.0, 10.0, 20.0]))
    np.testing.assert_allclose(
        [data.log_gl0[0, 0], data.log_gl1[0, 0], data.log_gl2[0, 0]], expected)

    # chr1:500 has no PL field at all -- should be all NaN
    assert np.all(np.isnan(data.log_gl0[2]))
    assert np.all(np.isnan(data.log_gl1[2]))
    assert np.all(np.isnan(data.log_gl2[2]))


def test_read_vcf_tracks_loci_without_pl(test_vcf_path):
    data = read_vcf(test_vcf_path, extract_pl=True)
    # chr1:500 and chr1:600 have no PL field
    assert data.n_loci_without_pl == 2


def test_read_vcf_sample_subset_and_reorder(test_vcf_path):
    data = read_vcf(test_vcf_path, samples=["sampleC", "sampleA"])
    assert data.sample_names == ["sampleC", "sampleA"]
    # chr1:100: sampleC=1/1 (2), sampleA=0/0 (0)
    np.testing.assert_array_equal(data.dosage[0], [2.0, 0.0])


def test_read_vcf_missing_sample_raises_actionable_error(test_vcf_path):
    with pytest.raises(ValueError, match="not found"):
        read_vcf(test_vcf_path, samples=["nonexistent_sample"])


def test_read_vcf_max_variants(test_vcf_path):
    data = read_vcf(test_vcf_path, max_variants=2)
    assert len(data) == 2
    assert data.locus_names == ["chr1:100", "rs123"]


def test_read_vcf_no_biallelic_snps_raises(tmp_path):
    content = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsampleA\n"
        "chr1\t100\t.\tA\tAT\t.\tPASS\t.\tGT\t0/1\n"
    )
    path = tmp_path / "no_snps.vcf"
    path.write_text(content)
    with pytest.raises(ValueError, match="No biallelic SNPs found"):
        read_vcf(str(path))


def test_read_vcf_no_chrom_header_raises(tmp_path):
    content = "##fileformat=VCFv4.2\nchr1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
    path = tmp_path / "bad.vcf"
    path.write_text(content)
    with pytest.raises(ValueError, match="not a valid VCF"):
        read_vcf(str(path))


# ---------------------------------------------------------------------------
# VCF + metadata writers
# ---------------------------------------------------------------------------

import pandas as pd
from hycline.data.vcf import (
    write_geographic_gl_csv, write_genomic_gl_csv,
    write_genomic_hard_count_csv, write_geographic_hard_count_csv,
)


@pytest.fixture
def test_metadata_df():
    return pd.DataFrame({
        "individual": ["sampleA", "sampleB", "sampleC", "sampleD"],
        "site": ["X", "X", "Y", "Y"],
        "distance": [0, 0, 50, 50],
        "hybrid_index": [0.1, 0.3, 0.7, 0.9],
    })


def test_write_geographic_hard_count_csv_matches_hand_computation(
        test_vcf_path, test_metadata_df, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path)
    out_path = tmp_path / "out.csv"
    info = write_geographic_hard_count_csv(
        vcf_data, test_metadata_df, "individual", "site", "distance", str(out_path))
    assert info["n_sites"] == 2
    assert info["n_samples_dropped"] == 0

    out_df = pd.read_csv(out_path)
    row_y = out_df[out_df["site"] == "Y"].iloc[0]
    # chr1:500, site Y = sampleC(dosage=2), sampleD(dosage=NaN)
    # valid=1 individual -> n_alleles=2, n_focal=2, freq=1.0
    assert row_y["chr1:500.nSamples"] == 2
    assert row_y["chr1:500.A"] == pytest.approx(1.0)


def test_write_geographic_hard_count_csv_inconsistent_distance_raises(
        test_vcf_path, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path)
    bad_metadata = pd.DataFrame({
        "individual": ["sampleA", "sampleB"], "site": ["X", "X"], "distance": [0, 5],
    })
    with pytest.raises(ValueError, match="inconsistent"):
        write_geographic_hard_count_csv(
            vcf_data, bad_metadata, "individual", "site", "distance", str(tmp_path / "out.csv"))


def test_write_no_sample_overlap_raises(test_vcf_path, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path)
    no_overlap = pd.DataFrame({
        "individual": ["nope1", "nope2"], "site": ["X", "X"], "distance": [0, 0],
    })
    with pytest.raises(ValueError, match="No sample names in common"):
        write_geographic_hard_count_csv(
            vcf_data, no_overlap, "individual", "site", "distance", str(tmp_path / "out.csv"))


def test_write_partial_sample_overlap_drops_extras(test_vcf_path, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path)
    partial = pd.DataFrame({
        "individual": ["sampleA", "sampleB"], "site": ["X", "X"], "distance": [0, 0],
    })
    info = write_geographic_hard_count_csv(
        vcf_data, partial, "individual", "site", "distance", str(tmp_path / "out.csv"))
    assert info["n_individuals"] == 2
    assert info["n_samples_dropped"] == 2  # sampleC, sampleD not in metadata


def test_write_geographic_gl_csv_requires_pl(test_vcf_path, test_metadata_df, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path, extract_pl=False)
    with pytest.raises(ValueError, match="extract_pl=True"):
        write_geographic_gl_csv(
            vcf_data, test_metadata_df, "individual", "distance", str(tmp_path / "out.csv"))


def test_write_geographic_gl_csv_excludes_loci_with_no_pl_data(
        test_vcf_path, test_metadata_df, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path, extract_pl=True)
    out_path = tmp_path / "out.csv"
    info = write_geographic_gl_csv(
        vcf_data, test_metadata_df, "individual", "distance", str(out_path))
    # chr1:500 and chr1:600 have no PL field in the test VCF
    assert set(info["excluded_loci_no_pl"]) == {"chr1:500", "chr1:600"}
    out_df = pd.read_csv(out_path)
    assert "chr1:500.gl0" not in out_df.columns
    assert "chr1:100.gl0" in out_df.columns


def test_write_genomic_hard_count_csv_hybrid_index_validation(test_vcf_path, tmp_path):
    from hycline.data.vcf import read_vcf

    vcf_data = read_vcf(test_vcf_path)
    bad_metadata = pd.DataFrame({
        "individual": ["sampleA", "sampleB"], "hybrid_index": [0.5, 1.5],
    })
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        write_genomic_hard_count_csv(
            vcf_data, bad_metadata, "individual", "hybrid_index", str(tmp_path / "out.csv"))


def test_all_four_writer_outputs_are_directly_readable(test_vcf_path, test_metadata_df, tmp_path):
    """The core end-to-end correctness check: every writer's output must
    be readable by the corresponding existing reader with zero extra
    configuration (the whole point of matching each reader's own default
    column-suffix conventions)."""
    from hycline.data.vcf import read_vcf
    from hycline.data.io import (
        read_multi_locus_wide_csv, read_gl_multi_locus_wide_csv,
        read_genomic_cline_multi_locus_wide_csv, read_genomic_gl_multi_locus_wide_csv,
    )

    vcf_data = read_vcf(test_vcf_path, extract_pl=True)

    p1 = tmp_path / "geo_hard.csv"
    write_geographic_hard_count_csv(vcf_data, test_metadata_df, "individual", "site",
                                     "distance", str(p1))
    result1 = read_multi_locus_wide_csv(str(p1))
    assert len(result1) == 4  # all 4 SNPs (chr1:100, rs123, chr1:500, chr1:600)

    p2 = tmp_path / "geo_gl.csv"
    write_geographic_gl_csv(vcf_data, test_metadata_df, "individual", "distance", str(p2))
    result2 = read_gl_multi_locus_wide_csv(str(p2), individual_col="individual")
    assert len(result2) == 2  # chr1:500/chr1:600 excluded (no PL)

    p3 = tmp_path / "gen_hard.csv"
    write_genomic_hard_count_csv(vcf_data, test_metadata_df, "individual",
                                  "hybrid_index", str(p3))
    result3 = read_genomic_cline_multi_locus_wide_csv(str(p3), individual_col="individual")
    assert len(result3) == 4

    p4 = tmp_path / "gen_gl.csv"
    write_genomic_gl_csv(vcf_data, test_metadata_df, "individual", "hybrid_index", str(p4))
    result4 = read_genomic_gl_multi_locus_wide_csv(str(p4), individual_col="individual")
    assert len(result4) == 2


def test_read_vcf_skips_invariant_alt_dot_records(tmp_path):
    # Regression test: VCF records with ALT="." (the spec's no-alternate
    # -allele/invariant-site marker, common in gVCF-derived joint
    # -genotyped call sets) used to pass both the multiallelic filter
    # (alt.split(",") on "." gives a length-1 list) and the indel filter
    # (len(".") == 1) undetected, getting parsed as spurious "SNPs"
    # instead of being skipped as the non-variant sites they are.
    content = (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsampleA\tsampleB\n"
        "chr1\t100\tlocusA\tA\tT\t.\tPASS\t.\tGT\t0/0\t1/1\n"
        "chr1\t200\tinvariant1\tA\t.\t.\tPASS\t.\tGT\t0/0\t0/0\n"
    )
    path = tmp_path / "invariant.vcf"
    path.write_text(content)

    data = read_vcf(str(path))

    assert "invariant1" not in data.locus_names
    assert data.locus_names == ["locusA"]
    assert data.n_skipped_invariant == 1
    assert data.n_skipped_multiallelic == 0
    assert data.n_skipped_indel == 0


def test_write_geographic_hard_count_csv_shared_missing_distance_not_inconsistent(tmp_path):
    # Regression test: the site-distance consistency check used `!=`
    # against possibly-NaN values, so a site where every individual
    # genuinely has a missing (NaN) distance always raised a spurious
    # "inconsistent distance values (nan vs nan)" error instead of
    # treating identically-missing values as consistent.
    from hycline.data.vcf import write_geographic_hard_count_csv
    import pandas as pd

    vcf_data = VCFData(locus_names=["locus1"], sample_names=["ind1", "ind2"],
                        dosage=np.array([[1.0, 1.0]]))
    metadata_df = pd.DataFrame({
        "individual": ["ind1", "ind2"], "site": ["S1", "S1"],
        "distance": [np.nan, np.nan],
    })
    out_path = tmp_path / "out.csv"
    info = write_geographic_hard_count_csv(vcf_data, metadata_df, "individual", "site",
                                            "distance", str(out_path))
    assert info["n_sites"] == 1
