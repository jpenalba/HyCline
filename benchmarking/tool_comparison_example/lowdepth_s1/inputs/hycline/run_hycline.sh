#!/usr/bin/env bash
# HyCline: hybrid index, genomic-cline input, then genomic clines (MLE and
# hierarchical Bayesian). Run from this folder. HYCLINE defaults to the
# installed `hycline` command; e.g. HYCLINE="python /path/to/run_hycline.py".
set -euo pipefail
HYCLINE=${HYCLINE:-hycline}
INPUT=input.beagle
$HYCLINE estimate-hybrid-index --input $INPUT --metadata metadata.tsv \
    --pop1-label pop1 --pop2-label pop2 --output hybrid_index.tsv
$HYCLINE generate-genomic-input --input $INPUT --metadata hybrid_index.tsv \
    --parental-metadata metadata.tsv --pop1-label pop1 --pop2-label pop2 \
    --output genomic_input.csv
$HYCLINE fit-genomic --input genomic_input.csv --individual-col individual \
    --no-plot --output-dir fit_mle
$HYCLINE fit-hierarchical-genomic --input genomic_input.csv --individual-col individual \
    --no-plot --output-dir fit_hier
