# HyCline User Guide: Coupling / Concordance Analysis

**Document status: working draft, updated alongside HyCline's own
development.**

This guide covers HyCline's coupling analysis -- measuring how concordant
cline shapes are across loci, the empirical proxy for Barton's (1983)
coupling coefficient and the basis for Firneno et al. (2023)'s theta
estimator. It applies to both geographic and genomic clines, via two
separate commands sharing the same underlying idea.

Every command shown below was run for real against synthetic and real
(published) test data while writing this guide -- flags, output messages,
and file names are exact.

## Why this matters

A hybrid zone's genome isn't one cline -- it's a collection of loci, each
with its own cline shape, shaped by its own combination of selection,
recombination, and linkage to other selected loci. When many loci share
similar, concordant cline shapes (similar centers, similar widths or
rates), that's evidence of strong reproductive isolation acting genome-
wide -- a "coupled" barrier to gene flow. When cline shapes vary widely
across loci, isolation is weaker or more locus-specific. Coupling
analysis quantifies this directly from a set of already-fitted, per-locus
cline parameters -- it's a downstream step, not a fitting step itself.

### A real shape-mismatch caveat, not a hypothetical one

Both coupling commands measure concordance through `width`/`slope`
(`coupling-geographic`) or `u`/`v` (`coupling-genomic`) -- summary numbers
that assume each locus's cline is shaped like the others being compared.
Baird & Daley (2025) point out a real, two-directional way that
assumption can quietly break down when clines actually differ in *shape*,
not just in the parameters being compared:

- **Asymmetry read as a spurious outlier.** A locus with a genuinely
  asymmetric cline (steeper on one side of the center than the other) can
  show a shifted center or a distorted width estimate relative to
  symmetric loci in the same set, purely because a symmetric model is
  being asked to summarize an asymmetric shape -- not because that locus
  has a different underlying selection/recombination balance.
- **Steppedness inflates width, masking real outliers.** A locus with a
  cline shape closer to a step function (sharper transition, heavier
  tails) than the tripartite/logistic default tends to have its width
  systematically overestimated by those models -- which can either
  manufacture a false low-concordance outlier or, just as problematically,
  hide a locus whose width is a genuine barrier-effect outlier for a
  different reason.

**HyCline's `--tail t-cline` model (a continuous alternative to the
tripartite tail model; see `HyCline_UserGuide_Geographic.md`) does not by
itself close this gap for coupling analysis.** `coupling-geographic`
reads only `width` from the per-locus summary CSV, and `coupling-genomic`
reads only `u`/`v` -- neither currently extracts the T-cline's own shape
parameters (`t_shape`, and `t_skew` under `--t-cline-asymmetric`) from the
summary, so a locus fit with the T-cline still enters coupling analysis
as a single width or `u`/`v` number, with no signal about how stepped or
asymmetric that locus actually was. If your locus set mixes clearly
different cline shapes (some loci visibly stepped or asymmetric when
plotted, others closer to plain logistic), treat cross-locus variance
comparisons with that caveat in mind -- a low-concordance signal may
reflect a shape mismatch between the model and a subset of your data
rather than a real difference in selection/recombination balance across
loci.

---

## `coupling-geographic`

Computes the across-locus variance in geographic cline slope from a
per-locus summary CSV -- the exact kind of file `fit-geographic`/
`fit-hierarchical-geographic` already produce.

```bash
hycline coupling-geographic \
    --input results/hycline_summary.csv \
    --output-dir coupling_results/
```

Real output, using independent per-locus fits (`fit-geographic`) on the
HZAR-bundled Manakin dataset (7 loci):

```
Reading locus summary from 'results/hycline_summary.csv' ...
  7 loci, using column 'width' for cline width.

Geographic cline variance across 7 loci:
  mean slope (beta) = 0.4119
  SD slope (sigma_beta) = 0.6680
  CV slope = 1.6216
  (lower SD/CV = more concordant clines across loci = stronger coupling; see this module's docstring -- no theta estimate is provided for geographic clines, only genomic, since Firneno et al. 2023 calibrated theta against real data using genomic clines only.)

All output written to: coupling_results/
  hycline_coupling_geographic_summary.txt/.json
```

Cline "slope" here is `1/width` -- steeper (narrower) clines get a higher
slope value. **Lower SD/CV across loci means more concordant clines**,
i.e. stronger coupling; there's no absolute threshold, this is a relative
comparison across your own locus set (or against another study's).

**No theta estimate for geographic clines.** Firneno et al. (2023)
calibrated their theta estimator against real published data using
genomic clines specifically (`u`/`v` from the logit-logistic model) --
there's no equivalent calibration for geographic cline slope, so
`coupling-geographic` reports only the raw variance metrics
(`mean_slope`, `sd_slope`, `cv_slope`), not theta. Use `coupling-genomic`
if you want a theta estimate and have genomic-cline fits available.

### Flags

| Flag | Meaning |
|---|---|
| `--input` | Per-locus summary CSV from `fit-geographic` or `fit-hierarchical-geographic` |
| `--locus-col` | Column with locus names (default: `locus`) |
| `--width-col` | Column with each locus's fitted cline width. Default: auto-detect `width_mean` (`fit-hierarchical-geographic`'s convention) then `width` (`fit-geographic`'s convention) |

### Independent vs. hierarchical input: a real, worth-knowing difference

`coupling-geographic` accepts either `fit-geographic`'s (independent
per-locus) or `fit-hierarchical-geographic`'s (partially pooled) summary
CSV, auto-detecting which convention it's reading. Running it on both,
from the same underlying data:

```
independent fits:    SD slope = 0.6680,  CV slope = 1.6216
hierarchical fits:    SD slope = 0.0828,  CV slope = 0.2612
```

**The hierarchical estimates show substantially lower variance -- and
this is expected, not a sign hierarchical fitting is "more correct" for
this specific measurement.** Hierarchical pooling deliberately shrinks
each locus's estimate toward a shared population-level mean, borrowing
statistical strength across loci -- so measuring the *spread* of
already-shrunk estimates will mechanically show less spread than
measuring the spread of independent ones, regardless of how concordant
the loci actually are. This doesn't make hierarchical input the wrong
choice (it's still recommended, since independent per-locus estimates
carry more sampling noise that inflates apparent variance for the
opposite reason), but it's worth knowing you're not comparing the same
quantity when you switch between the two input types -- pick one
convention and stay consistent within a given comparison (e.g. across
several populations or species), rather than mixing independent and
hierarchical inputs within the same coupling comparison.

---

## `coupling-genomic`

The genomic-cline analog, computing across-locus variance in `logit(u/v)`
and `log(v)` (Firneno et al. 2023's exact definitions), plus an estimated
coupling coefficient (theta).

```bash
hycline coupling-genomic \
    --input results/hycline_hierarchical_locus_summary.csv \
    --output-dir coupling_results/
```

Real output, using hierarchical fits (`fit-hierarchical-genomic`) on a
5-locus synthetic dataset:

```
Reading locus summary from 'results/hycline_hierarchical_locus_summary.csv' ...
  5 loci, using columns 'u_mean'/'v_mean' for u/v.

Genomic cline variance across 5 loci:
  SD logit(center) (sigma_c) = 1.0313
  SD log(v) (sigma_v) = 1.0935
  estimated coupling coefficient (theta) = -1.647
    NOTE: sigma_c and/or sigma_v fall outside the bounding box of the 31 reference data points the calibration was fit on -- theta here is an extrapolation, treat it with extra caution.
    WARNING: theta is a ratio of selection to recombination (s/r) and is theoretically always positive -- a non-positive estimate here means this (sigma_c, sigma_v) combination falls somewhere the linear calibration doesn't behave well, not a real negative coupling coefficient. Treat the raw sigma_c/sigma_v values as the reliable output for this locus set, not theta.

All output written to: coupling_results/
  hycline_coupling_genomic_summary.txt/.json
```

This real example hits a genuine, documented limitation worth expecting
rather than being surprised by: **theta can come back negative**, which
is theoretically impossible (it's a ratio of selection strength to
recombination rate). This happens when your data's `sigma_c`/`sigma_v`
fall outside the range Firneno et al.'s own published reference points
covered -- the calibration is a linear fit over a limited empirical
range, and extrapolating it past that range can produce nonsensical
values. HyCline flags this explicitly (`within_calibration_range: false`
in the JSON output) rather than returning a believable-looking but
meaningless number silently. When this happens, **treat the raw
`sigma_c`/`sigma_v` values as the reliable output**, not theta itself.

**A second, separate caveat on theta, beyond the calibration-range one
above: the migration rate assumed by Firneno et al.'s own simulations
matters, and they quantified how much.** The calibration this module
implements comes from simulations at migration rate m=0.1/0.2 between
neighboring demes. Firneno et al. also ran a robustness check at a lower
migration rate (m=0.05) and found the relationship between `sigma_c`/
`sigma_v` and theta measurably weaker there on its own (their linear
model's r-squared dropped from 0.523 to 0.415), and, more strikingly,
that a model *fit* at m=0.1/0.2 essentially failed when cross-applied to
predict theta for the m=0.05 simulations (their random-forest model's
out-of-bag variance explained dropped to roughly 0%; even their linear
model, cross-applied the same way, explained only 22.5% of the true
variance) -- despite an m=0.1-vs-m=0.2 cross-application working well in
both directions. In practice, this means `coupling-genomic`'s theta
estimate is best trusted for hybrid zones with dispersal/migration on a
scale broadly similar to the reference data set (real published hybrid
zones spanning many taxa, not a single migration regime -- see the table
in `hycline/analysis/coupling.py::FIRNENO_2023_TABLE1`), and should be
treated with more caution than the calibration-range check alone implies
for a hybrid zone you have independent reason to think has unusually
restricted gene flow between demes -- exactly the regime this robustness
check found the calibration degrades on, even when `sigma_c`/`sigma_v`
themselves still fall inside the calibration's bounding box. Genetic map
size has a smaller version of the same sensitivity (37% vs. the main
calibration's 52.3% variance explained across different simulated
chromosome sizes) -- see `hycline/analysis/coupling.py`'s
`_fit_theta_calibration_from_firneno2023_table1` docstring for the full
numbers on both.

### Flags

| Flag | Meaning |
|---|---|
| `--input` | Per-locus summary CSV from `fit-genomic` or `fit-hierarchical-genomic` |
| `--locus-col` | Column with locus names (default: `locus`) |
| `--u-col` | Column with each locus's fitted `u`. Default: auto-detect `u_mean` (`fit-hierarchical-genomic`'s convention) then `u` (`fit-genomic`'s convention) |
| `--v-col` | Column with each locus's fitted `v`. Default: auto-detect `v_mean` then `v`, same as `--u-col` |
| `--no-theta` | Skip the theta estimate, reporting only the raw `sigma_c`/`sigma_v` variance metrics |

`--pool-center` on `fit-hierarchical-genomic` hierarchically pools
`logit(center) = u/v` directly, rather than the raw bias parameter `u` --
this is the exact quantity Firneno et al.'s own methodology uses, so with
that flag, `sigma_center` from the population summary can be read
directly as `sigma_c`, rather than `coupling-genomic` needing to derive
it from per-locus point estimates. Both approaches feed into
`coupling-genomic` correctly either way; `--pool-center` just makes the
correspondence to Firneno et al.'s original parameterization explicit.

The same independent-vs-hierarchical variance-shrinkage consideration
described above for `coupling-geographic` applies here too -- hierarchical
`u`/`v` estimates will generally show lower across-locus variance (and
correspondingly a lower, possibly out-of-calibration-range, theta) than
independent ones, for the same mechanical reason.

---

## Output files, in brief

Both commands write the same two files to `--output-dir` (default
`./hycline_output`, filename prefix configurable via `--prefix`):

- **`*_summary.txt`** -- human-readable summary
- **`*_summary.json`** -- the same numbers, machine-readable:
  - `coupling-geographic`: `n_loci`, `mean_slope`, `sd_slope`, `cv_slope`, `width_col_used`
  - `coupling-genomic`: `n_loci`, `sd_logit_center`, `sd_log_v`, `theta_estimate`, `within_calibration_range`, `u_col_used`, `v_col_used`

---

## A note on where this guide's examples came from

Every command and every block of real output in this guide was generated
by actually running HyCline against synthetic test data and the real,
HZAR-bundled Manakin dataset while writing it.
