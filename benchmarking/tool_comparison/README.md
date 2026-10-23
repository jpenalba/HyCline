# HyCline, gghybrid, bgchm and bgc: benchmark data

The results are summarized in `../HyCline_Tool_Comparison.pdf`
(rebuild with `python benchmarking/make_comparison_report_pdf.py`).

## Layout

One folder per design, each with three replicate data sets (seeds 1-3):

| Folder | Design |
|---|---|
| `baseline/` | 20 + 20 parentals, 100 hybrids (uniform h), 100 loci differing by >= 0.6; `genotypes_s*` (true genotypes) and `lowdepth_s*` (reads at mean depth 3) |
| `weak_differentiation/` | loci differing by >= 0.3 |
| `small_panels_exact_freqs/` | 5 + 5 parentals, hybrids drawn from the panels' own frequencies |
| `population_panels/` | 20 + 20 parentals sampled from population frequencies |
| `small_panels/` | 5 + 5 parentals sampled from population frequencies |
| `missing_data/` | 20% of genotypes missing |
| `bimodal_hybrids/` | hybrid indices ~ Beta(0.5, 0.5) |
| `skewed_hybrids/` | hybrid indices ~ Beta(1.5, 6) |
| `italy_sparrows/` | scores and timings on the Hermansen et al. (2014) Italian sparrows |

Every design has 10 barrier loci (v = 3), 10 permeable (v = 0.33), 10 with a
shifted centre (u = +/-1.5) and 70 neutral. `design.json` records the exact
settings.

Each data set folder (`<scenario>_s<seed>/`) holds:

- `sim/` -- `sim_truth.tsv` (per locus: `cline_class`, `u`, `v` in HyCline's
  convention, panel frequencies), `sim_metadata.tsv` (per individual:
  population and true `hybrid_index`), `sim_params.tsv` (the simulator call).
- `inputs/` -- input files and a run script for each tool, and
  `inputs/README.md` on the formats and how the tools' cline parameters
  correspond:
  - `hycline/` -- `input.vcf` (and `input.beagle` for low depth),
    `metadata.tsv`, `run_hycline.sh`
  - `gghybrid/` -- `gghybrid_input.csv`, `run_gghybrid.R`
  - `bgchm/` -- `G0.tsv`, `G1.tsv`, `Gx.tsv` (and `GL*_*.tsv` genotype
    likelihoods for low depth), `run_bgchm.R`
  - `bgc/` -- bgc 1.03 input in `genotypes/` (and `reads/` for low depth)
  - the outputs each tool produced in the benchmark (`hybrid_index.tsv`,
    `clines.tsv`, HyCline's `fit_mle/` and `fit_hier/` summaries)
- `times.json` -- wall-clock seconds per tool.

## Rerunning

From a tool's folder: `bash run_hycline.sh` (`HYCLINE="python
/path/to/run_hycline.py"` if HyCline isn't installed), `Rscript
run_gghybrid.R`, or `Rscript run_bgchm.R genotype` (`glik` for low depth).
Running a tool replaces its saved outputs. To score:

    python benchmarking/compare_tools.py --score-only --reps 3 \
        --outdir benchmarking/tool_comparison/<design>

New data sets: `benchmarking/compare_tools.py` (simulate, run, score) and
`benchmarking/archive_comparison.py` (save a run in this layout);
`benchmarking/make_tool_inputs.py` writes the four tools' inputs from any
VCF (plus beagle) and metadata file.

The Italian sparrow genotypes are not copied here (they come from
gghybrid's GPL-3 repository); `benchmarking/italy_sparrow_tools.py`
rebuilds the inputs from a gghybrid checkout and reruns the comparison.
