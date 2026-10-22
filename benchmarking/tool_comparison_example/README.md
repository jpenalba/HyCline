# HyCline, gghybrid, bgchm and bgc on the same simulated hybrid zones

Two simulated data sets (seed 1 of `benchmarking/compare_tools.py`), each
with inputs for every tool and the simulation truth:

| Folder | Data | HyCline and bgchm use | gghybrid uses | bgc files |
|---|---|---|---|---|
| `genotypes_s1/` | true genotypes | genotypes (VCF) | genotypes | `bgc/genotypes/` |
| `lowdepth_s1/` | reads at mean depth 3 | genotype likelihoods (beagle / bgchm `glik`) | genotypes called from the reads | `bgc/genotypes/` (called) and `bgc/reads/` (read counts, `-N 1`) |

Both: 20 + 20 parental individuals (`pop1` at hybrid index 0, `pop2` at 1),
100 hybrids with uniform hybrid indices, 100 loci with parental allele
frequencies differing by at least 0.6. Clines: 10 barrier loci (v = 3), 10
permeable loci (v = 0.33), 10 biased loci (u = +/-1.5, centre shifted) and 70
neutral loci (v = 1, centre 0.5). The exact simulator call is in
`sim/sim_params.tsv`.

- `sim/sim_truth.tsv` -- per locus: `cline_class`, `u`, `v` (HyCline's
  convention, logit(phi) = u + v logit(h)), and the parental panels'
  frequencies.
- `sim/sim_metadata.tsv` -- per individual: population and true
  `hybrid_index`.
- `inputs/` -- one folder per tool with its input files and run script, and
  `inputs/README.md` describing the files and how the tools' cline
  parameters correspond (gghybrid and bgchm: centre c = invlogit(-u / v)).

## Running

From each tool's folder:

    bash run_hycline.sh            # HYCLINE="python /path/to/run_hycline.py" if not installed
    Rscript run_gghybrid.R
    Rscript run_bgchm.R genotype   # lowdepth_s1: Rscript run_bgchm.R glik

Each writes `hybrid_index.tsv` (HyCline also `fit_mle/` and `fit_hier/`,
gghybrid and bgchm `clines.tsv`). To score whatever has been run against the
truth:

    python benchmarking/compare_tools.py --score-only --reps 1 \
        --outdir benchmarking/tool_comparison_example

bgc 1.03 is not run by the script; see `inputs/README.md` for its files and a
command template.

## Results (3 seeds; `compare_tools.py --reps 3`)

Hybrid index, 100 test individuals per data set:

| Scenario | Tool | Mean abs. error | r | 95% coverage | Interval width | Time (s) |
|---|---|---|---|---|---|---|
| genotypes | HyCline | 0.038 | 0.986 | 0.92 | 0.167 | 91 |
| | bgchm | 0.038 | 0.986 | 0.92 | 0.168 | 279 |
| | gghybrid | 0.038 | 0.986 | 0.92 | 0.166 | 552 |
| lowdepth | HyCline | 0.041 | 0.983 | 0.94 | 0.193 | 136 |
| | bgchm | 0.042 | 0.983 | 0.95 | 0.195 | 862 |
| | gghybrid | 0.043 | 0.982 | 0.87 | 0.172 | 534 |

Genomic clines, 100 loci per data set (median absolute errors; v outliers =
95% interval excludes 1, centre outliers = excludes 0.5; power among the 20
barrier/permeable or 10 biased loci, false positives among the 70 neutral):

| Scenario | Tool | logit(c) error | log(v) error | v power | v false pos. | c power | c false pos. |
|---|---|---|---|---|---|---|---|
| genotypes | HyCline MLE | 0.182 | 0.199 | 0.80 | 0.052 | 0.97 | 0.033 |
| | HyCline hierarchical | 0.128 | 0.177 | 0.78 | 0.019 | 0.97 | 0.019 |
| | bgchm | 0.132 | 0.190 | 0.82 | 0.024 | 0.97 | 0.029 |
| | gghybrid | 0.215 | 0.235 | 0.58 | 0.005 | 0.77 | 0.038 |
| lowdepth | HyCline MLE | 0.240 | 0.238 | 0.58 | 0.048 | 0.87 | 0.057 |
| | HyCline hierarchical | 0.168 | 0.203 | 0.60 | 0.014 | 0.90 | 0.043 |
| | bgchm | 0.167 | 0.209 | 0.65 | 0.010 | 0.90 | 0.057 |
| | gghybrid | 0.287 | 0.292 | 0.52 | 0.033 | 0.87 | 0.114 |

Times are wall-clock for the whole run script (hybrid index and clines; for
HyCline both cline fits) on 4 cores; the low-depth runs shared the machine
with other work, so treat their times as rough. HyCline's hierarchical
intervals are normal approximations from its per-locus posterior mean and
SD. See BACKEND_PHASE3.md section 9.5 for details.
