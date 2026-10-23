#!/usr/bin/env python3
"""Real-world variations of VCF and beagle input to generate-geographic-input
and generate-genomic-input: is each converted correctly, or refused with a
clear message?

A small simulated hybrid zone (simulate_vcf.py) gives a plain base VCF (GT
only) and beagle file. Each variant rewrites the input the way some real
pipeline or file might -- phased genotypes, GT not first in FORMAT,
GATK-style FORMAT fields, half-missing genotypes, multiallelic sites,
indels, symbolic or '*' alleles, monomorphic sites (ALT '.'), gzip and
BGZF-style compression, Windows line endings, missing or duplicate IDs,
IDs containing dots, haploid or polyploid genotypes, duplicate sample
names, untidy metadata -- and is paired with a reference file stating, in
plain form, what the variant should mean (e.g. a half-missing genotype is
missing; a multiallelic site is skipped). Both files are converted by
both commands and the outputs compared; each output is also read back
with the fitting commands' own readers.

Outcomes:
  same          output identical to the reference's
  DIFFERS       output differs from the reference's (details given)
  error         the command refused the input (message given); for
                variants marked 'should refuse', that is the intended outcome
Writes <outdir>/results.tsv and prints it.

Example:
    python benchmarking/conversion_input_variants.py --outdir conversion_variants
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
HYCLINE = [sys.executable, str(HERE.parent / "run_hycline.py")]
LABELS = ["--pop1-label", "pop1", "--pop2-label", "pop2"]


# ---------------------------------------------------------------------------
# Base data
# ---------------------------------------------------------------------------

def load_vcf(path):
    meta, header, rows = [], None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("##"):
            meta.append(line)
        elif line.startswith("#CHROM"):
            header = line.split("\t")
        else:
            rows.append(line.split("\t"))
    return meta, header, rows


def vcf_text(meta, header, rows, nl="\n"):
    return nl.join(meta + ["\t".join(header)] + ["\t".join(r) for r in rows]) + nl


def gts(row):
    return row[9:]


def with_gts(row, cells, fmt=None):
    return row[:8] + [fmt or row[8]] + list(cells)


# ---------------------------------------------------------------------------
# VCF variants: name -> function(meta, header, rows, rng) returning
# (variant spec, reference spec, note). A spec is a dict with 'text' (or
# 'bytes' for compressed files), 'suffix', and optionally 'metadata' (a
# function taking the base metadata text).
# ---------------------------------------------------------------------------

def spec(text, suffix=".vcf", **kw):
    return {"text": text, "suffix": suffix, **kw}


def v_phased(m, h, rows, rng):
    v = [with_gts(r, [c.replace("/", "|") for c in gts(r)]) for r in rows]
    return spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)), "phased genotypes (0|1)"


def v_gt_not_first(m, h, rows, rng):
    v = [with_gts(r, [f"{rng.integers(1, 20)}:{c}" for c in gts(r)], "DP:GT") for r in rows]
    return spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)), "FORMAT DP:GT"


def gatk_cell(c, rng):
    d = int(rng.integers(4, 20))
    if c == "./.":
        return "./.:0,0:0:.:."
    alt = c.count("1")
    ad = {0: f"{d},0", 1: f"{d // 2},{d - d // 2}", 2: f"0,{d}"}[alt]
    pl = {0: f"0,{3 * d},{30 * d}", 1: f"{3 * d},0,{3 * d}", 2: f"{30 * d},{3 * d},0"}[alt]
    return f"{c}:{ad}:{d}:{min(99, 3 * d)}:{pl}"


def v_gatk(m, h, rows, rng):
    v = [with_gts(r, [gatk_cell(c, rng) for c in gts(r)], "GT:AD:DP:GQ:PL") for r in rows]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)),
            "GATK-style FORMAT GT:AD:DP:GQ:PL")


def v_truncated(m, h, rows, rng):
    v = [with_gts(r, [c if c == "./." else f"{c}:5,5:10" for c in gts(r)], "GT:AD:DP")
         for r in rows]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)),
            "missing samples' trailing FORMAT fields dropped (./. only)")


def v_dot_missing(m, h, rows, rng):
    v = [with_gts(r, ["." if c == "./." else c for c in gts(r)]) for r in rows]
    return spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)), "missing written as '.'"


def v_half_missing(m, h, rows, rng):
    v, ref = [], []
    for r in rows:
        cells, refc = [], []
        for c in gts(r):
            if c != "./." and rng.random() < 0.05:
                cells.append(rng.choice([f"./{c[2]}", f"{c[0]}/."]))
                refc.append("./.")
            else:
                cells.append(c)
                refc.append(c)
        v.append(with_gts(r, cells))
        ref.append(with_gts(r, refc))
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, ref)),
            "5% of genotypes half-missing (./1, 0/.) -> treated as missing")


def extra_records(rows, every, make):
    out = []
    for i, r in enumerate(rows):
        out.append(r)
        if i % every == 0:
            out.append(make(r))
    return out


def v_multiallelic(m, h, rows, rng):
    def make(r):
        alts = [b for b in "ACGT" if b != r[3]][:2]
        return [r[0], str(int(r[1]) + 1), r[2] + "_m", r[3], ",".join(alts)] + r[5:9] + [
            c if c == "./." else "1/2" for c in gts(r)]
    return (spec(vcf_text(m, h, extra_records(rows, 10, make))), spec(vcf_text(m, h, rows)),
            "multiallelic records added -> skipped")


def v_indels(m, h, rows, rng):
    def make(r):
        return [r[0], str(int(r[1]) + 1), r[2] + "_i", r[3] + "T", r[4]] + r[5:]
    return (spec(vcf_text(m, h, extra_records(rows, 10, make))), spec(vcf_text(m, h, rows)),
            "indel records added -> skipped")


def v_alt_dot(m, h, rows, rng):
    def make(r):
        return [r[0], str(int(r[1]) + 1), r[2] + "_mono", r[3], "."] + r[5:9] + [
            c if c == "./." else "0/0" for c in gts(r)]
    return (spec(vcf_text(m, h, extra_records(rows, 10, make))), spec(vcf_text(m, h, rows)),
            "monomorphic records (ALT '.', all-sites VCF) added -> should be skipped")


def v_symbolic(m, h, rows, rng):
    def make(r):
        return [r[0], str(int(r[1]) + 1), r[2] + "_sym", r[3], "<NON_REF>"] + r[5:9] + [
            c if c == "./." else "0/0" for c in gts(r)]
    return (spec(vcf_text(m, h, extra_records(rows, 10, make))), spec(vcf_text(m, h, rows)),
            "symbolic ALT <NON_REF> records (gVCF) added -> skipped")


def v_star(m, h, rows, rng):
    def make(r):
        return [r[0], str(int(r[1]) + 1), r[2] + "_star", r[3], "*"] + r[5:]
    return (spec(vcf_text(m, h, extra_records(rows, 10, make))), spec(vcf_text(m, h, rows)),
            "spanning-deletion ALT '*' records added -> should be skipped")


def v_gzip(m, h, rows, rng):
    return (spec(None, ".vcf.gz", bytes=gzip.compress(vcf_text(m, h, rows).encode())),
            spec(vcf_text(m, h, rows)), "gzip, .vcf.gz")


def bgzf_like(text):
    data = text.encode()
    return b"".join(gzip.compress(data[i:i + 4096]) for i in range(0, len(data), 4096))


def v_bgzf(m, h, rows, rng):
    return (spec(None, ".vcf.gz", bytes=bgzf_like(vcf_text(m, h, rows))),
            spec(vcf_text(m, h, rows)), "BGZF-style multi-member gzip, .vcf.gz")


def v_bgz_ext(m, h, rows, rng):
    return (spec(None, ".vcf.bgz", bytes=bgzf_like(vcf_text(m, h, rows))),
            spec(vcf_text(m, h, rows)), "BGZF-style gzip named .vcf.bgz")


def v_crlf(m, h, rows, rng):
    return (spec(vcf_text(m, h, rows, "\r\n")), spec(vcf_text(m, h, rows)),
            "Windows line endings (CRLF)")


def v_id_dot(m, h, rows, rng):
    v = [r[:2] + ["."] + r[3:] for r in rows]
    ref = [r[:2] + [f"{r[0]}:{r[1]}"] + r[3:] for r in rows]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, ref)),
            "ID '.' -> loci named CHROM:POS")


def v_id_dots(m, h, rows, rng):
    v = [r[:2] + [f"scaffold_{i % 7}.1_{r[1]}"] + r[3:] for i, r in enumerate(rows)]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, v)),
            "IDs containing dots (scaffold_3.1_1000); checked by reading back")


def v_dup_ids(m, h, rows, rng):
    v = [r[:] for r in rows]
    v[1][2] = v[0][2]
    return (spec(vcf_text(m, h, v)), None,
            "two records share an ID -> should refuse (IDs become column names)")


def v_haploid(m, h, rows, rng):
    v = [with_gts(r, ["." if c == "./." else c[0] for c in gts(r)]) if i % 5 == 0 else r
         for i, r in enumerate(rows)]
    ref = [r for i, r in enumerate(rows) if i % 5]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, ref)),
            "haploid genotypes (0, 1) at 20% of records -> should skip or refuse, not "
            "write all-missing loci")


def v_polyploid(m, h, rows, rng):
    v = [with_gts(r, ["./././." if c == "./." else f"{c}/{c}" for c in gts(r)]) if i % 5 == 0
         else r for i, r in enumerate(rows)]
    ref = [r for i, r in enumerate(rows) if i % 5]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, ref)),
            "tetraploid genotypes (0/0/1/1) at 20% of records -> should skip or refuse")


def v_lowercase(m, h, rows, rng):
    v = [r[:3] + [r[3].lower(), r[4].lower()] + r[5:] for r in rows]
    return spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)), "lowercase REF/ALT"


def v_filter(m, h, rows, rng):
    v = [r[:6] + ["LowQual" if i % 10 == 0 else r[6]] + r[7:] for i, r in enumerate(rows)]
    return (spec(vcf_text(m, h, v)), spec(vcf_text(m, h, rows)),
            "FILTER LowQual on 10% of records (FILTER is not applied: same as reference)")


def v_dup_sample(m, h, rows, rng):
    h2 = h[:]
    h2[10] = h2[9]
    return (spec(vcf_text(m, h2, rows)), None,
            "two VCF columns with the same sample name -> should refuse")


def v_no_gt(m, h, rows, rng):
    v = [with_gts(r, ["5,5" for _ in gts(r)], "AD") for r in rows]
    return spec(vcf_text(m, h, v)), None, "no GT field (FORMAT AD only) -> should refuse"


def v_meta_crlf(m, h, rows, rng):
    return (spec(vcf_text(m, h, rows), metadata=lambda t: t.replace("\n", "\r\n")),
            spec(vcf_text(m, h, rows)), "metadata with Windows line endings")


def v_meta_space(m, h, rows, rng):
    def meta(t):
        lines = t.split("\n")
        first = lines[1].split("\t")
        first[0] += " "
        lines[1] = "\t".join(first)
        return "\n".join(lines)
    return (spec(vcf_text(m, h, rows), metadata=meta), spec(vcf_text(m, h, rows)),
            "a trailing space after one metadata sample name")


VCF_VARIANTS = {
    "phased": v_phased, "gt_not_first": v_gt_not_first, "gatk_format": v_gatk,
    "truncated_fields": v_truncated, "dot_missing": v_dot_missing,
    "half_missing": v_half_missing, "multiallelic": v_multiallelic, "indels": v_indels,
    "monomorphic_alt_dot": v_alt_dot, "symbolic_alt": v_symbolic, "star_alt": v_star,
    "gzip": v_gzip, "bgzf": v_bgzf, "bgz_extension": v_bgz_ext, "crlf": v_crlf,
    "id_dot": v_id_dot, "ids_with_dots": v_id_dots, "duplicate_ids": v_dup_ids,
    "haploid": v_haploid, "polyploid": v_polyploid, "lowercase_alleles": v_lowercase,
    "filter_lowqual": v_filter, "duplicate_sample": v_dup_sample, "no_gt": v_no_gt,
    "metadata_crlf": v_meta_crlf, "metadata_trailing_space": v_meta_space,
}
SHOULD_REFUSE = {"duplicate_ids", "duplicate_sample", "no_gt"}


# ---------------------------------------------------------------------------
# beagle variants
# ---------------------------------------------------------------------------

def b_gzip(text):
    return spec(None, ".beagle.gz", bytes=gzip.compress(text.encode())), spec(text, ".beagle"), \
        "gzip, .beagle.gz"


def b_crlf(text):
    return (spec(text.replace("\n", "\r\n"), ".beagle"), spec(text, ".beagle"),
            "Windows line endings (CRLF)")


def b_sci(text):
    lines = text.split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if not line:
            continue
        f = line.split("\t")
        out.append("\t".join(f[:3] + [f"{float(x):.6e}" for x in f[3:]]))
    return (spec("\n".join(out) + "\n", ".beagle"), spec("\n".join(out) + "\n", ".beagle"),
            "likelihoods in scientific notation (1.000000e-05); checked by reading back")


def b_marker_dots(text):
    lines = text.split("\n")
    out = [lines[0]]
    for i, line in enumerate(lines[1:]):
        if line:
            f = line.split("\t")
            out.append("\t".join([f"scaffold_{i % 7}.1_{f[0]}"] + f[1:]))
    t = "\n".join(out) + "\n"
    return spec(t, ".beagle"), spec(t, ".beagle"), "marker names with dots; checked by reading back"


def b_ind_names(text):
    lines = text.split("\n")
    h = lines[0].split("\t")
    n = (len(h) - 3) // 3
    lines[0] = "\t".join(h[:3] + [f"Ind{i}" for i in range(n) for _ in range(3)])
    return (spec("\n".join(lines), ".beagle"), None,
            "ANGSD's own header (Ind0, Ind1, ...) with metadata using real sample names -> "
            "should refuse clearly")


BEAGLE_VARIANTS = {"gzip": b_gzip, "crlf": b_crlf, "scientific_notation": b_sci,
                   "marker_names_with_dots": b_marker_dots, "angsd_ind_header": b_ind_names}
BEAGLE_SHOULD_REFUSE = {"angsd_ind_header"}


# ---------------------------------------------------------------------------
# Running and comparing
# ---------------------------------------------------------------------------

def write_spec(s, path_stem):
    path = Path(str(path_stem) + s["suffix"])
    if "bytes" in s:
        path.write_bytes(s["bytes"])
    else:
        path.write_bytes(s["text"].encode())
    return path


def convert(command, inp, meta, out, fmt):
    args = [command, "--input", str(inp), "--metadata", str(meta), "--output", str(out)]
    if fmt:
        args += ["--format", fmt]
    if command == "generate-genomic-input":
        args += LABELS
    return subprocess.run(HYCLINE + args, capture_output=True, text=True)


def read_back(command, kind, out):
    from hycline.data import io
    try:
        if command == "generate-geographic-input":
            data = (io.read_multi_locus_wide_csv(str(out)) if kind == "vcf"
                    else io.read_gl_multi_locus_wide_csv(str(out), individual_col="individual"))
        else:
            data = (io.read_genomic_cline_multi_locus_wide_csv(str(out), individual_col="individual")
                    if kind == "vcf"
                    else io.read_genomic_gl_multi_locus_wide_csv(str(out), individual_col="individual"))
        return str(len(data))
    except Exception as e:  # noqa: BLE001 -- reported, not raised
        return f"FAILED: {str(e).splitlines()[0][:120]}"


def diff_outputs(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if list(a.columns) == list(b.columns) and a.shape == b.shape:
        num = a.select_dtypes("number")
        other = b[num.columns]
        same_nan = (num.isna() == other.isna()).all().all()
        close = np.allclose(num.fillna(0).to_numpy(float), other.fillna(0).to_numpy(float),
                            rtol=0, atol=1e-12)
        if same_nan and close and a.drop(columns=num.columns).equals(b.drop(columns=num.columns)):
            return ""
    parts = []
    extra = [c for c in a.columns if c not in set(b.columns)]
    missing = [c for c in b.columns if c not in set(a.columns)]
    if len(a.columns) != len(set(a.columns)) or any(".1" == c[-2:] for c in extra):
        parts.append("duplicate column names")
    if extra:
        parts.append(f"{len(extra)} extra columns (e.g. {extra[0]})")
    if missing:
        parts.append(f"{len(missing)} columns missing (e.g. {missing[0]})")
    if len(a) != len(b):
        parts.append(f"{len(a)} rows vs {len(b)}")
    if not parts:
        parts.append("values differ")
    return "; ".join(parts)


def run_variant(kind, name, variant, reference, note, outdir, meta_text, should_refuse):
    d = outdir / f"{kind}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "metadata.tsv"
    meta_path.write_text(variant.get("metadata", lambda t: t)(meta_text))
    ref_meta = d / "metadata_ref.tsv"
    ref_meta.write_text(meta_text)
    inp = write_spec(variant, d / "input")
    ref_inp = write_spec(reference, d / "reference") if reference else None
    fmt = "beagle" if kind == "beagle" else None
    rows = []
    for command in ("generate-geographic-input", "generate-genomic-input"):
        short = command.split("-")[1]
        out, ref_out = d / f"{short}.csv", d / f"{short}_reference.csv"
        r = convert(command, inp, meta_path, out, fmt)
        row = {"input": kind, "variant": name, "command": short, "description": note}
        if r.returncode != 0:
            msg = [l for l in r.stderr.splitlines() if l.strip()]
            row.update(outcome="error (intended)" if should_refuse else "ERROR",
                       details=(msg[-1] if msg else "")[:200], read_back="")
        elif should_refuse:
            row.update(outcome="ACCEPTED (should refuse)", details="", read_back=read_back(command, kind, out))
        else:
            rr = convert(command, ref_inp, ref_meta, ref_out, fmt)
            if rr.returncode != 0:
                row.update(outcome="reference failed", details=rr.stderr.strip()[-200:], read_back="")
            else:
                diff = diff_outputs(pd.read_csv(out), pd.read_csv(ref_out))
                notes = [l.strip() for l in r.stdout.splitlines() if "NOTE" in l or "Skipped" in l]
                row.update(outcome="same" if not diff else "DIFFERS",
                           details=diff + (" | " if diff and notes else "") + " / ".join(notes)[:200],
                           read_back=read_back(command, kind, out))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default="conversion_variants")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "base"
    subprocess.run([sys.executable, str(HERE / "simulate_vcf.py"), "--n-pop1", "10",
                    "--n-pop2", "10", "--n-popH", "30", "--n-snps", "200",
                    "--aim-percent", "50", "--missing-global", "0.05", "--format-beagle",
                    "--mean-depth", "5", "-o", f"{stem}.vcf", "--seed", str(args.seed)],
                   check=True, capture_output=True)
    meta = pd.read_csv(f"{stem}_metadata.tsv", sep="\t")[["individual", "population",
                                                          "hybrid_index"]]
    rng = np.random.default_rng(args.seed)
    meta["site"] = [f"site{int(h * 5)}" for h in meta["hybrid_index"]]
    meta["distance"] = [int(h * 5) * 10.0 for h in meta["hybrid_index"]]
    meta_text = meta.to_csv(sep="\t", index=False)

    m, h, rows = load_vcf(f"{stem}.vcf")
    results = []
    for name, fn in VCF_VARIANTS.items():
        variant, reference, note = fn(m, h, rows, rng)
        results += run_variant("vcf", name, variant, reference, note, out, meta_text,
                               name in SHOULD_REFUSE)
    btext = Path(f"{stem}.beagle").read_text()
    for name, fn in BEAGLE_VARIANTS.items():
        variant, reference, note = fn(btext)
        results += run_variant("beagle", name, variant, reference, note, out, meta_text,
                               name in BEAGLE_SHOULD_REFUSE)
    df = pd.DataFrame(results)
    df.to_csv(out / "results.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 110):
        print(df[["input", "variant", "command", "outcome", "read_back", "details"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
