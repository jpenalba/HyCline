#!/usr/bin/env python3
"""Check generate-geographic-input and generate-genomic-input value by value
against an independent recomputation from the input files.

For each seed, simulate_vcf.py writes a hybrid zone as a VCF (true
genotypes) and a beagle file (reads at --mean-depth), with missing
genotypes and one individual missing at 90% of SNPs. The metadata file is
built to exercise the sample join: its rows are shuffled, a few input
individuals are left out of it (they must be dropped), and a few
individuals absent from the input are added (they must be ignored). Each
individual gets a sampling site and a distance (parental populations at the
two ends, hybrids spread over sites in between) and its true hybrid index.

Both commands are run on both files, and every output value is compared
with what a separate, minimal parser (written for this script, sharing no
code with HyCline) computes from the raw files:

  geographic, VCF     per site: allele copies (.nSamples) and ALT frequency
                      (.A), sites in metadata order
  geographic, beagle  per individual: natural-log genotype likelihoods
  genomic, VCF        per test individual: .n and .freq; parental
                      frequencies (.p0/.p1) from the labelled references,
                      clipped to [0.001, 0.999]; references left out; loci
                      with equal parental frequencies left out
  genomic, beagle     per test individual: log likelihoods; parental
                      frequencies as the references' maximum-likelihood
                      frequency (found here by a grid plus bounded search,
                      not EM)

Reports, per seed and output: rows, loci, whether the row order, row set
and column set match exactly, and the largest absolute difference. Writes
<outdir>/results.tsv.

Example:
    python benchmarking/conversion_exactness.py --outdir conversion_exactness
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
HYCLINE = [sys.executable, str(HERE.parent / "run_hycline.py")]
GT = re.compile(r"^([01])[/|]([01])$")
FLOOR = 1e-3


# ---------------------------------------------------------------------------
# Independent parsers
# ---------------------------------------------------------------------------

def parse_vcf(path):
    """(locus ids, sample names, dosage array (loci, samples) with NaN)."""
    ids, rows, samples = [], [], None
    for line in open(path):
        if line.startswith("##"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[0] == "#CHROM":
            samples = f[9:]
            continue
        if len(f[3]) != 1 or len(f[4]) != 1:
            continue
        gt_at = f[8].split(":").index("GT")
        row = []
        for cell in f[9:]:
            m = GT.match(cell.split(":")[gt_at])
            row.append(int(m.group(1)) + int(m.group(2)) if m else np.nan)
        ids.append(f[2] if f[2] != "." else f"{f[0]}:{f[1]}")
        rows.append(row)
    return ids, samples, np.array(rows, dtype=float)


def parse_beagle(path):
    """(markers, sample names, likelihoods (loci, samples, 3))."""
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        samples = head[3::3]
        markers, rows = [], []
        for line in fh:
            f = line.rstrip("\n").split("\t")
            markers.append(f[0])
            rows.append([float(x) for x in f[3:]])
    gl = np.array(rows).reshape(len(markers), len(samples), 3)
    return markers, samples, gl


def ml_freq(gl, include):
    """Maximum-likelihood allele frequency of the included individuals under
    Hardy-Weinberg, by a fine grid then a bounded search around its best
    point."""
    g = gl[include]
    if len(g) == 0:
        return np.nan
    g = g / g.sum(axis=1, keepdims=True)

    def nll(p):
        with np.errstate(divide="ignore"):
            return -np.log(g[:, 0] * (1 - p) ** 2 + g[:, 1] * 2 * p * (1 - p)
                           + g[:, 2] * p ** 2).sum()

    grid = np.linspace(0, 1, 2001)
    best = grid[np.argmin([nll(p) for p in grid])]
    res = minimize_scalar(nll, bounds=(max(best - 1e-3, 0), min(best + 1e-3, 1)),
                          method="bounded", options={"xatol": 1e-12})
    return min((res.x, best), key=nll)


def beagle_missing(gl):
    total = gl.sum(axis=-1)
    equal = np.isclose(gl[..., 0], gl[..., 1]) & np.isclose(gl[..., 1], gl[..., 2])
    return ~np.isfinite(total) | (total <= 0) | equal


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------

def expected_geographic_vcf(vcf, meta):
    ids, samples, dos = vcf
    pos = {s: i for i, s in enumerate(samples)}
    m = meta[meta["individual"].isin(pos)]
    sites = list(dict.fromkeys(m["site"]))
    out = {"site": sites, "distance": [m.loc[m["site"] == s, "distance"].iloc[0] for s in sites]}
    for j, locus in enumerate(ids):
        n_col, a_col = [], []
        for s in sites:
            d = dos[j, [pos[x] for x in m.loc[m["site"] == s, "individual"]]]
            called = ~np.isnan(d)
            n = 2 * int(called.sum())
            n_col.append(n)
            a_col.append(np.nansum(d) / n if n else np.nan)
        out[f"{locus}.nSamples"] = n_col
        out[f"{locus}.A"] = a_col
    return pd.DataFrame(out)


def expected_geographic_beagle(bgl, meta):
    markers, samples, gl = bgl
    pos = {s: i for i, s in enumerate(samples)}
    m = meta[meta["individual"].isin(pos)]
    idx = [pos[x] for x in m["individual"]]
    out = {"individual": m["individual"].tolist(), "distance": m["distance"].tolist()}
    with np.errstate(divide="ignore"):
        logs = np.where(gl > 0, np.log(gl), -1e10)
    for j, mk in enumerate(markers):
        for k in range(3):
            out[f"{mk}.gl{k}"] = logs[j, idx, k]
    return pd.DataFrame(out)


def expected_genomic(data, meta, kind):
    names, samples, values = data
    pos = {s: i for i, s in enumerate(samples)}
    m = meta[meta["individual"].isin(pos)]
    ref0 = [pos[x] for x in m.loc[m["population"] == "pop1", "individual"]]
    ref1 = [pos[x] for x in m.loc[m["population"] == "pop2", "individual"]]
    test = m[~m["population"].isin(["pop1", "pop2"])]
    idx = [pos[x] for x in test["individual"]]
    out = {"individual": test["individual"].tolist(), "hybrid_index": test["hybrid_index"].tolist()}
    if kind == "beagle":
        miss = beagle_missing(values)
        with np.errstate(divide="ignore"):
            logs = np.where(values > 0, np.log(values), -1e10)
    for j, name in enumerate(names):
        if kind == "vcf":
            p = [np.nanmean(values[j, r]) / 2 if np.isfinite(values[j, r]).any() else np.nan
                 for r in (ref0, ref1)]
        else:
            p = [ml_freq(values[j, r], ~miss[j, r]) for r in (ref0, ref1)]
        p = [np.clip(x, FLOOR, 1 - FLOOR) for x in p]
        if not (np.isfinite(p[0]) and np.isfinite(p[1])) or p[0] == p[1]:
            continue
        if kind == "vcf":
            d = values[j, idx]
            out[f"{name}.n"] = np.where(np.isnan(d), 0, 2)
            out[f"{name}.freq"] = d / 2
        else:
            for k in range(3):
                out[f"{name}.gl{k}"] = logs[j, idx, k]
        out[f"{name}.p0"] = [p[0]] * len(idx)
        out[f"{name}.p1"] = [p[1]] * len(idx)
    return pd.DataFrame(out)


def compare(got: pd.DataFrame, exp: pd.DataFrame, key: str, p_tol: float) -> dict:
    """Exact comparison of two output tables: same rows in the same order,
    same columns, values equal to within 1e-12 (NaN = NaN). Parental-frequency columns from
    beagle input are compared to within p_tol (the expected value comes from
    a different optimizer)."""
    res = {"rows": len(got), "loci": sum(c.endswith((".A", ".freq", ".gl0")) for c in got.columns),
           "row_order_matches": got[key].tolist() == exp[key].tolist(),
           "columns_match": list(got.columns) == list(exp.columns)}
    common = [c for c in exp.columns if c in got.columns and c != key]
    diff, p_diff = 0.0, 0.0
    for c in common:
        a = pd.to_numeric(got[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(exp[c], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            diff = np.inf
            continue
        d = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0.0
        if c.endswith((".p0", ".p1")):
            p_diff = max(p_diff, d)
        else:
            diff = max(diff, d)
    res["max_abs_diff"] = diff
    res["max_abs_diff_parental_freq"] = p_diff
    # 1e-12: values computed in a different order differ in the last bits
    res["exact"] = (res["row_order_matches"] and res["columns_match"] and diff <= 1e-12
                    and p_diff <= p_tol)
    return res


def run_seed(seed: int, a) -> list[dict]:
    d = Path(a.outdir) / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / "sim"
    subprocess.run([sys.executable, str(HERE / "simulate_vcf.py"),
                    "--n-pop1", str(a.n_parental), "--n-pop2", str(a.n_parental),
                    "--n-popH", str(a.n_hybrids), "--n-snps", str(a.n_snps),
                    "--aim-percent", "40", "--missing-global", "0.1",
                    "--missing-individual", "sample45_popH=0.9", "--hybrid-freqs", "population",
                    "--format-beagle", "--beagle-output", f"{stem}.beagle",
                    "--mean-depth", str(a.mean_depth), "-o", f"{stem}.vcf",
                    "--seed", str(seed)], check=True, capture_output=True)

    rng = np.random.default_rng(seed)
    meta = pd.read_csv(f"{stem}_metadata.tsv", sep="\t")[["individual", "population", "hybrid_index"]]
    hyb = meta["population"] == "popH"
    sites = np.where(meta["population"] == "pop1", "site00",
                     np.where(meta["population"] == "pop2", f"site{a.n_sites + 1:02d}", ""))
    sites[hyb.to_numpy()] = [f"site{k:02d}" for k in rng.integers(1, a.n_sites + 1, hyb.sum())]
    meta["site"] = sites
    meta["distance"] = meta["site"].str[4:].astype(int) * 10.0
    dropped = rng.choice(meta["individual"], 5, replace=False)
    meta = meta[~meta["individual"].isin(dropped)]
    ghosts = pd.DataFrame({"individual": [f"ghost{k}" for k in range(5)], "population": "popH",
                           "hybrid_index": 0.5, "site": "site01", "distance": 10.0})
    meta = pd.concat([meta, ghosts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    meta_path = d / "metadata.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)

    vcf, bgl = parse_vcf(f"{stem}.vcf"), parse_beagle(f"{stem}.beagle")
    labels = ["--pop1-label", "pop1", "--pop2-label", "pop2"]
    runs = {
        "geographic, VCF": (["generate-geographic-input", "--input", f"{stem}.vcf"],
                            lambda: expected_geographic_vcf(vcf, meta), "site"),
        "geographic, beagle": (["generate-geographic-input", "--input", f"{stem}.beagle"],
                               lambda: expected_geographic_beagle(bgl, meta), "individual"),
        "genomic, VCF": (["generate-genomic-input", "--input", f"{stem}.vcf", *labels],
                         lambda: expected_genomic(vcf, meta, "vcf"), "individual"),
        "genomic, beagle": (["generate-genomic-input", "--input", f"{stem}.beagle", *labels],
                            lambda: expected_genomic(bgl, meta, "beagle"), "individual"),
    }
    rows = []
    for name, (cmd, expect, key) in runs.items():
        out = d / (name.replace(", ", "_") + ".csv")
        r = subprocess.run(HYCLINE + cmd + ["--metadata", str(meta_path), "--output", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            rows.append({"seed": seed, "output": name, "exact": False, "error": r.stderr.strip()})
            continue
        got = pd.read_csv(out)
        rows.append({"seed": seed, "output": name,
                     **compare(got, expect(), key, a.p_tol)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-parental", type=int, default=20)
    ap.add_argument("--n-hybrids", type=int, default=120)
    ap.add_argument("--n-snps", type=int, default=3000)
    ap.add_argument("--n-sites", type=int, default=10)
    ap.add_argument("--mean-depth", type=float, default=4)
    ap.add_argument("--p-tol", type=float, default=1e-6,
                    help="Tolerance for beagle parental frequencies (different optimizers).")
    ap.add_argument("--outdir", default="conversion_exactness")
    args = ap.parse_args(argv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(args.reps):
        rows += run_seed(args.seed + r, args)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.outdir) / "results.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))
    return 0 if df["exact"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
