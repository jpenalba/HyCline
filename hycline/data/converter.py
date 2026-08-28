"""Conversion utilities for getting real-world data into HyCline's
expected input formats.

This module currently holds the hard-genotype-to-genotype-likelihood
converter (the "hard-genotype convenience path" -- see
`hycline.data.genotype_likelihood.hard_genotype_to_log_likelihood` for
the underlying per-value conversion, and BACKEND_PHASE5.md section 5 for
why this was flagged as a real gap). The VCF + metadata converter lives
alongside it in `hycline.data.vcf`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hycline.data.genotype_likelihood import hard_genotype_to_log_likelihood
from hycline.data.schema import values_consistent

#: Column names never treated as a locus when auto-detecting locus columns
#: from a wide genotype-dosage CSV -- covers every reserved column name
#: used anywhere else in this project's CSV conventions (individual/sample
#: labels, geographic covariates, genomic covariates, population labels
#: for hybrid-index estimation). Not exhaustive for every possible
#: metadata scheme -- pass `loci` explicitly for anything not covered.
DEFAULT_RESERVED_COLUMNS = frozenset({
    "individual", "sample", "distance", "lat", "lon", "latitude", "longitude",
    "hybrid_index", "population", "pop", "site", "locality", "location",
})


def detect_genotype_loci(df: pd.DataFrame,
                          reserved_columns=DEFAULT_RESERVED_COLUMNS) -> list[str]:
    """Auto-detect locus columns in a wide genotype-dosage CSV: every
    column not in `reserved_columns` (case-insensitive match)."""
    reserved_lower = {c.lower() for c in reserved_columns}
    return [c for c in df.columns if c.lower() not in reserved_lower]


def convert_genotype_csv_to_gl_csv(input_path: str, output_path: str,
                                    loci: list[str] | None = None,
                                    reserved_columns=DEFAULT_RESERVED_COLUMNS) -> list[str]:
    """Convert a wide genotype-dosage CSV (one row per individual, one
    column per locus holding 0/1/2 focal-allele copies, blank/NaN for
    missing) into a wide genotype-likelihood CSV with the same shape
    every GL-consuming command in this project already expects
    (`<locus>.gl0`/`<locus>.gl1`/`<locus>.gl2` columns), so ordinary
    hard-called genotype data can be fed into any GL-aware fitting
    command without constructing one-hot likelihoods by hand.

    Every column not identified as a locus (see `reserved_columns`/
    `loci`) is passed through unchanged -- typically `individual`,
    `distance`, `hybrid_index`, or similar covariate columns the
    downstream fitting command will also need.

    Parameters
    ----------
    input_path : path to the genotype-dosage CSV
    output_path : path to write the genotype-likelihood CSV to
    loci : optional explicit list of locus column names. If not given,
        every column not in `reserved_columns` (case-insensitive) is
        treated as a locus -- see `DEFAULT_RESERVED_COLUMNS` for what's
        excluded by default.
    reserved_columns : column names to exclude from locus auto-detection
        when `loci` isn't given explicitly

    Returns
    -------
    list of the locus column names that were converted
    """
    df = pd.read_csv(input_path)

    if loci is None:
        loci = detect_genotype_loci(df, reserved_columns)
        if not loci:
            raise ValueError(
                f"No locus columns detected in '{input_path}' (every column matched "
                f"the reserved-column list). Pass `loci` explicitly, or check that "
                f"your genotype columns aren't accidentally named the same as a "
                f"reserved column ({sorted(reserved_columns)})."
            )

    reserved_present = [c for c in df.columns if c not in loci]
    new_columns = {}
    for locus in loci:
        if locus not in df.columns:
            raise ValueError(f"Requested locus column '{locus}' not found in '{input_path}'.")
        genotype = df[locus].to_numpy(dtype=float)
        gls = hard_genotype_to_log_likelihood(genotype)
        new_columns[f"{locus}.gl0"] = gls["log_gl0"]
        new_columns[f"{locus}.gl1"] = gls["log_gl1"]
        new_columns[f"{locus}.gl2"] = gls["log_gl2"]

    # built as one concat, not looped column-by-column assignment, to avoid
    # pandas' DataFrame-fragmentation performance warning at scale (found
    # during a real "thousands of loci" stress test -- see BACKEND_SCALE.md)
    out = pd.concat([df[reserved_present].reset_index(drop=True),
                      pd.DataFrame(new_columns)], axis=1)

    out.to_csv(output_path, index=False)
    return loci


def convert_genotype_csv_to_hard_count_csv(input_path: str, output_path: str,
                                            site_col: str, distance_col: str,
                                            loci: list[str] | None = None,
                                            reserved_columns=DEFAULT_RESERVED_COLUMNS,
                                            ploidy: int = 2) -> list[str]:
    """Aggregate a wide genotype-dosage CSV (one row per individual, one
    column per locus holding 0/1/2 focal-allele copies) into HZAR-style
    site-level allele counts (`<locus>.nSamples`/`<locus>.A` columns,
    matching `read_multi_locus_wide_csv`'s own default convention exactly
    -- see BACKEND_PHASE1.md section 9.3), ready for `fit`/`fit-multi`.

    This is the reverse direction of `convert_genotype_csv_to_gl_csv`,
    which deliberately stays per-individual (genotype likelihoods can't
    be losslessly aggregated the way hard counts can -- see
    `hycline.data.schema.GenotypeLikelihoodTransect`'s docstring). Going
    from per-individual dosage to site-aggregated counts loses no
    information a hard genotype call carries, so this direction *can*
    aggregate, and does. Individuals sharing a site are pooled into one
    row; every individual at a given site must share the same distance
    value, checked explicitly (the same requirement and error-reporting
    convention as `hycline.data.vcf.write_geographic_hard_count_csv`,
    which this function mirrors for genotype-CSV input instead of a VCF).

    Parameters
    ----------
    input_path : path to the genotype-dosage CSV
    output_path : path to write the site-aggregated hard-count CSV to
    site_col : column identifying each individual's sampling site
    distance_col : column holding each individual's geographic distance
        (must agree within a site)
    loci : optional explicit list of locus column names (default:
        auto-detected, excluding `site_col`/`distance_col` in addition to
        `reserved_columns`)
    reserved_columns : as in `convert_genotype_csv_to_gl_csv`
    ploidy : ploidy used to convert a count of non-missing individuals at
        a site into a total sampled-allele count (default: 2)

    Returns
    -------
    list of the locus column names that were converted
    """
    df = pd.read_csv(input_path)

    if site_col not in df.columns:
        raise ValueError(f"--site-col '{site_col}' not found in '{input_path}'. "
                          f"Columns present: {', '.join(df.columns)}")
    if distance_col not in df.columns:
        raise ValueError(f"--distance-col '{distance_col}' not found in '{input_path}'. "
                          f"Columns present: {', '.join(df.columns)}")

    if loci is None:
        exclude = set(reserved_columns) | {site_col.lower(), distance_col.lower()}
        loci = detect_genotype_loci(df, reserved_columns=exclude)
        if not loci:
            raise ValueError(
                f"No locus columns detected in '{input_path}' (every column matched "
                f"the reserved-column list, or was --site-col/--distance-col). Pass "
                f"`loci` explicitly, or check your genotype columns' names."
            )
    else:
        missing = [l for l in loci if l not in df.columns]
        if missing:
            raise ValueError(f"Requested loci {missing} not found in '{input_path}'.")

    sites = df[site_col].tolist()
    distances = df[distance_col].tolist()

    site_distance = {}
    site_order = []
    site_row_positions = {}
    for pos, (site, dist) in enumerate(zip(sites, distances)):
        if site not in site_distance:
            site_distance[site] = dist
            site_order.append(site)
            site_row_positions[site] = []
        elif not values_consistent(site_distance[site], dist):
            raise ValueError(
                f"Site '{site}' has inconsistent {distance_col} values across its "
                f"individuals ({site_distance[site]} vs {dist}) -- every individual at "
                f"the same site must share the same distance value."
            )
        site_row_positions[site].append(pos)

    out_rows = {site_col: site_order, "distance": [site_distance[s] for s in site_order]}

    for locus in loci:
        dosage_all = df[locus].to_numpy(dtype=float)
        n_alleles_col = []
        freq_col = []
        for site in site_order:
            positions = site_row_positions[site]
            dosage = dosage_all[positions]
            valid = ~np.isnan(dosage)
            n_alleles = int(valid.sum()) * ploidy
            n_focal = float(np.nansum(dosage))
            n_alleles_col.append(n_alleles)
            freq_col.append(n_focal / n_alleles if n_alleles > 0 else np.nan)
        out_rows[f"{locus}.nSamples"] = n_alleles_col
        out_rows[f"{locus}.A"] = freq_col

    out = pd.DataFrame(out_rows)
    out.to_csv(output_path, index=False)
    return loci
