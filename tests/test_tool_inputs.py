"""benchmarking/make_tool_inputs.py: matching inputs for HyCline, gghybrid,
bgchm and bgc, checked against hand-computed values on a tiny VCF."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarking"))
from make_tool_inputs import write_inputs  # noqa: E402

SAMPLES = ["a1", "a2", "b1", "b2", "h1", "h2"]
POPS = ["P", "P", "Q", "Q", "zone", "zone"]
# Two loci; GT:AD per sample. a = pop1 (hybrid index 0), b = pop2, h = test.
ROWS = [
    ("snp1", ["0/0:3,0", "0/1:2,2", "1/1:0,4", "1/1:0,3", "0/1:1,1", "./.:0,0"]),
    ("snp2", ["1/1:0,2", "1/1:0,5", "0/0:4,0", "0/1:3,1", "0/0:2,0", "1/1:0,6"]),
]


def _write(tmp_path):
    lines = ["##fileformat=VCFv4.2",
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES)]
    for i, (sid, cells) in enumerate(ROWS):
        lines.append(f"chr1\t{100 * (i + 1)}\t{sid}\tA\tG\t.\tPASS\t.\tGT:AD\t" + "\t".join(cells))
    vcf = tmp_path / "in.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    meta = tmp_path / "meta.tsv"
    pd.DataFrame({"individual": SAMPLES, "population": POPS}).to_csv(meta, sep="\t", index=False)
    return vcf, meta


def test_inputs_for_every_tool(tmp_path):
    vcf, meta = _write(tmp_path)
    out = tmp_path / "out"
    info = write_inputs(vcf, meta, out, "P", "Q")
    assert (info["n_p0"], info["n_p1"], info["n_test"]) == (2, 2, 2)
    assert info["has_reads"]

    # gghybrid: two rows per individual, ALT = allele 1, NA missing
    gg = pd.read_csv(out / "gghybrid" / "gghybrid_input.csv")
    assert list(gg.columns) == ["INDLABEL", "POPID", "snp1", "snp2"]
    a2 = gg[gg["INDLABEL"] == "a2"]
    assert sorted(a2["snp1"]) == [0, 1] and list(a2["snp2"]) == [1, 1]
    assert gg[gg["INDLABEL"] == "h2"]["snp1"].isna().all()
    assert set(gg["POPID"]) == {"P", "Q", "test"}

    # bgchm: individuals x loci ALT dosage, parental sets split out
    gx = pd.read_csv(out / "bgchm" / "Gx.tsv", sep="\t", index_col=0)
    assert list(gx.index) == ["h1", "h2"]
    assert gx.loc["h1", "snp1"] == 1 and np.isnan(gx.loc["h2", "snp1"])
    g0 = pd.read_csv(out / "bgchm" / "G0.tsv", sep="\t", index_col=0)
    assert g0.to_numpy().tolist() == [[0, 2], [1, 2]]

    # bgc known genotypes: parental REF ALT allele counts; admixed per individual
    p0 = (out / "bgc" / "genotypes" / "p0in.txt").read_text().split("\n")
    assert p0[:4] == ["locus 0", "3 1", "locus 1", "0 4"]
    adm = (out / "bgc" / "genotypes" / "admixedin.txt").read_text().split("\n")
    assert adm[:4] == ["locus 0", "pop 0", "1 1", "0 0"]
    # bgc read counts: per-individual REF ALT reads from AD
    p1r = (out / "bgc" / "reads" / "p1in.txt").read_text().split("\n")
    assert p1r[:3] == ["locus 0", "0 4", "0 3"]
    assert (out / "bgc" / "individuals.txt").read_text().split() == ["h1", "h2"]

    # HyCline: the VCF itself plus labels, and run scripts for each tool
    m = pd.read_csv(out / "hycline" / "metadata.tsv", sep="\t")
    assert list(m["population"]) == ["P", "P", "Q", "Q", "test", "test"]
    for script in ("hycline/run_hycline.sh", "gghybrid/run_gghybrid.R", "bgchm/run_bgchm.R"):
        assert (out / script).exists()
    assert 'S0 = "P", S1 = "Q"' in (out / "gghybrid" / "run_gghybrid.R").read_text()
