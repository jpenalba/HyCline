# BACKEND_SCALE.md: a real stress test at "thousands of loci" scale

This document exists because of a specific, honest gap flagged in the
README's "Not yet implemented" list: every multi-locus validation in
this project, across every phase, had used on the order of 1-20 loci.
The belief that wide-format CSVs, the VCF/beagle parsers, and sequential
classic-MLE fitting handle "thousands of loci" (the realistic meaning of
"genome-wide" for this kind of analysis -- not tens or hundreds of
thousands) reasonably was a reasoned extrapolation from how pandas,
scipy, and JAX scale generically, not direct evidence from this codebase.
This document is that direct evidence -- what was actually tested, what
the real numbers were, and two real bugs the testing surfaced and fixed
along the way.

Environment this was run on, stated because the numbers below are only
meaningful with it in mind: a single CPU core, ~4GB RAM sandboxed
environment -- a genuinely modest baseline, closer to "an old laptop"
than "a lab workstation," which if anything makes these numbers a
pessimistic (worst-case) estimate for a typical user.

## 1. What was tested, and the honest numbers

All tests used synthetic data with a **known true signal**: a small
fraction of loci (5%) carrying a real, known cline (geographic center=70,
width=25; or genomic barrier locus v=4.0), the rest carrying no true
signal at all (random background frequencies) -- so correctness could be
checked at scale, not just "did it finish."

| Operation | Scale | Wall-clock time | Rate |
|---|---|---|---|
| VCF parsing (`hycline.data.vcf.read_vcf`) | 3,000 loci, 60 samples | 0.24s | ~12,400 loci/sec |
| VCF parsing | 10,000 loci, 60 samples | 0.83s | ~12,000 loci/sec |
| `generate-geographic-input` (full CLI, parse+join+write; VCF input -- was `convert-vcf-geographic` at the time this was measured) | 3,000 loci | 4.91s | -- |
| Wide-format CSV read/write (via the above) | 3,000 loci, 6,002 columns | (included above) | -- |
| `fit-geographic-multi` classic MLE, hard counts, no plot | 3,000 loci | 162.8s | ~0.054s/locus |
| `fit-geographic-gl-multi` classic MLE, genotype-likelihood-aware, no plot | 1,000 loci | 101.4s | ~0.101s/locus |
| `fit-genomic-multi` classic MLE, genomic clines, no plot | 1,000 loci | 15.4s | ~0.015s/locus |
| `genotype-to-gl` conversion (a CLI command at the time this was measured; since removed from the CLI along with the CSV-genotype pipeline it served -- the underlying `hycline.data.converter.convert_genotype_csv_to_gl_csv` function this number reflects is unchanged, still available for direct Python use) | 1,000 loci | 3.6s | -- |
| `estimate-hybrid-index` column detection/loading | 1,000 loci | 1.5s | -- |

**VCF parsing and wide-format CSV I/O are genuinely fine at this scale**
-- the "reasoned extrapolation" in the README was correct here.
Linear scaling confirmed directly (3,000 -> 10,000 loci, same per-locus
rate), not just assumed.

**Classic MLE fitting is exactly what was predicted: a real but
tolerable cost, not a blocker.** 3,000 loci in 2.7 minutes on a single
core is a genuine "go get coffee" wait, not an unusable one -- and this
is the *unparallelized* baseline every Tier 2 speedup item (plate-
vectorized batching, multiprocessing) is trying to improve on, not a
number that needed fixing first.

**Genotype-likelihood-aware fitting is real, and now quantified, about
2x slower per locus than hard-count fitting** (0.101s vs 0.054s) --
expected, since marginalizing over genotype uncertainty is a more
expensive per-evaluation likelihood than the plain binomial one hard
counts use, and now a concrete number instead of "probably somewhat
slower."

**Genomic-cline fitting is meaningfully faster per locus** than
geographic (0.015s vs 0.054s) -- fewer free parameters (u, v vs. center,
width, pmin, pmax), consistent with what the model sizes would predict.

## 2. Scientific correctness was checked, not just "did it complete"

At every scale tested, the known true signal was correctly recovered
from among the background noise, not just "the command exited 0":

- Geographic, hard counts (3,000 loci): true center=70/width=25 recovered
  as mean center=69.9/width=23.5 across the 150 truly informative loci;
  background loci correctly showed high, meaningless dispersion
  (SD=28.6) rather than false convergence on a shared value.
- Geographic, genotype-likelihood-aware (1,000 loci): true center=70/
  width=25 recovered as mean center=70.9/width=23.9.
- Genomic, hard counts (1,000 loci): true barrier locus v=4.0 recovered
  as median v=4.3 across the 50 truly informative loci.

This matters because a performance stress test that only checks timing
could miss a correctness regression introduced under scale-specific code
paths (e.g. a batching/chunking bug that silently mishandles some loci) --
checking against known ground truth at the same time closes that gap.

## 3. A real, now-quantified problem: grid plots

This was flagged as a likely issue before testing (`README.md`'s Tier 2
"scale-aware default outputs" item), but the actual severity was
unknown. Now it is: at just **300 loci**, `fit-geographic-multi`'s default grid plot
(one small-multiples panel per locus) is already **2,025 x 52,500
pixels** (a 52,500-pixel-tall image), 9.3MB, and Pillow's own
decompression-bomb heuristic (`DecompressionBombWarning`, intended to
catch maliciously oversized images) flags it as looking like an attack
pattern -- 106 million pixels against a default 89 million-pixel limit.

This confirms the Tier 2 item's premise with real numbers, and reveals
the threshold is much lower than "thousands" -- a few hundred loci is
already enough to produce an image no viewer will render sensibly and
that some tooling will refuse to open at all. Not fixed in this pass
(that's the scoped Tier 2 work); this section exists to give that item a
concrete number to design against, rather than a vague "this is probably
bad at scale."

## 4. Two real bugs found and fixed, not just documented

Neither of these was visible at the 1-20 loci scale every existing test
in this project used before now -- both are genuine "only shows up at
real scale" findings, the exact kind of thing this stress test existed
to surface.

### 4.1 Pandas DataFrame fragmentation (a real `PerformanceWarning`, not a correctness bug)

Running `genotype-to-gl` (a CLI command at the time this was found;
since removed from the CLI along with the CSV-genotype pipeline it
served -- the underlying `hycline.data.converter.
convert_genotype_csv_to_gl_csv` function this finding applied to is
unchanged) on a 1,000-locus file triggered a genuine
`pandas.errors.PerformanceWarning: DataFrame is highly fragmented`,
caused by assigning new columns one at a time in a loop
(`out[f"{locus}.gl0"] = ...`) onto an already-large DataFrame, rather
than building all new columns at once and concatenating. Checking for
the same pattern elsewhere found it in **six more places** -- every
VCF and beagle GL/hard-count writer that isn't already site-aggregating
into a plain dict first
(`write_geographic_gl_csv`/`write_genomic_gl_csv`/
`write_genomic_hard_count_csv` in `hycline/data/vcf.py`;
`write_beagle_geographic_gl_csv`/`write_beagle_genomic_gl_csv`/
`write_beagle_genomic_hard_count_csv` in `hycline/data/beagle.py`).

All seven were fixed the same way: build new columns as a plain dict,
then `pd.concat([existing, pd.DataFrame(new_columns)], axis=1)` once,
rather than looping column assignment. Verified byte-identical output
(`pd.DataFrame.equals`) before and after the fix, and confirmed zero
`PerformanceWarning`s at a scale (500-600+ loci) that reliably triggered
it before -- `tests/test_data_io.py::
test_convert_genotype_csv_to_gl_csv_no_fragmentation_warning_at_scale`,
plus the existing VCF/beagle test suites re-run to confirm no output
changed (`tests/test_vcf.py`, `tests/test_beagle.py`, 82 tests, all
still passing).

The two already-safe writers
(`write_geographic_hard_count_csv`/`write_beagle_geographic_hard_count_csv`)
were left unchanged -- they already accumulate into a plain Python dict
during site aggregation (an unavoidable step for that specific output
shape) before a single `pd.DataFrame(...)` call, which never triggers
this warning in the first place; the fix pattern applied to the other
seven functions is exactly what those two already did organically.

### 4.2 Every multi-locus CLI command dumped the full locus-name list to the console

`fit-geographic-multi`/`fit-genomic-multi`/`fit-geographic-gl-multi`/`fit-genomic-gl-multi`/
`genotype-to-gl`/`genotype-to-hard-counts` (the latter two since removed
from the CLI entirely -- see BACKEND_PHASE1.md section 9.6's closing
note) all printed something like
`"1000 loci detected: ['locus0', 'locus1', ..., 'locus999']"` --
technically correct, completely unreadable at real scale, and something
no one would notice testing with 5-20 loci. Fixed with a small shared
helper, `_format_loci_list` in `hycline/cli.py`, applied consistently
across all 9 occurrences found (not just the first one noticed):
truncates to the first 10 names plus `"... and N more"` when the list is
longer than that. Regression test: `tests/test_cli.py::
test_cli_genotype_to_gl_truncates_long_locus_list_in_console_output`.

## 5. What this stress test does and doesn't establish

**Established, with real evidence**: VCF parsing, wide-format CSV I/O,
and sequential classic-MLE fitting (geographic hard-count, geographic
GL-aware, and genomic) all work correctly and at a tolerable (if not
fast) pace at "thousands of loci" scale, on genuinely modest hardware.
Two real usability bugs that only manifest at this scale were found and
fixed as a direct result of testing at this scale rather than continuing
to rely on extrapolation.

**Not established, and explicitly out of scope for this pass**:
- Bayesian/MCMC fitting at this scale was not tested here -- the
  README's Tier 2 items already correctly identify this as the real
  cost center at "thousands of loci" (each locus's NUTS run being
  seconds-to-tens-of-seconds, not milliseconds), and this document's
  classic-MLE numbers don't change that assessment either way.
- `filter-aims` and `estimate-hybrid-index`'s core computation (as
  opposed to the I/O/column-detection layer timed here) wasn't
  separately stress-tested at 1,000+ loci -- both use simple, fully
  vectorized numpy operations per locus (no per-locus Python loop with
  an expensive optimizer call, unlike the fitting commands), so there's
  less reason to expect a surprise there, but this is inference from the
  code's shape, not a direct timing measurement the way the table in
  section 1 is.
- Genome-scale VCF/beagle files (tens of thousands of loci or more) were
  not tested -- this document is specifically about the "thousands, not
  hundreds of thousands" scale this project targets (see the README's
  "Not yet implemented" Tier 1/2 framing for why that distinction
  matters); nothing here should be read as evidence about that much
  larger regime.

---

## 6. `filter-aims` at genome scale

`benchmarking/time_filter_aims.py` simulates datasets with
`benchmarking/simulate_vcf.py` and times `filter-aims` on them as separate
processes (fastest of N runs, peak RSS from `os.wait4`). With 100
individuals (20 + 20 parental, 60 hybrid), 20% AIMs, `--minor-allele-freq
0.1`, on a 4-core container:

| Loci | VCF time | VCF peak | Beagle time (before) | Beagle peak (before) | Beagle file |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1.5 s | 138 MB | 1.5 s (1.5 s) | 180 MB (158 MB) | 3 MB |
| 10,000 | 3.2 s | 141 MB | 2.7 s (2.7 s) | 324 MB (220 MB) | 27 MB |
| 30,000 | 5.5 s | 144 MB | 4.9 s (5.2 s) | 336 MB (387 MB) | 82 MB |
| 100,000 | 15.6 s | 160 MB | 12.7 s (16.4 s) | 354 MB (895 MB) | 272 MB |
| 300,000 | 44.8 s | 201 MB | 33.7 s (--) | 377 MB (--) | 816 MB |

Start-up (`import hycline.cli`) is 1.2 s of every run. Beyond it, time is
linear in loci: the log-log slope of (time - start-up) on loci is 0.92
(VCF) and 0.91 (beagle) over 10,000-300,000 loci.

The "before" beagle columns are the original implementation, which read
the whole file into memory (`read_beagle`) and peaked at about 3.3x the
file size -- about 9 GB projected for 1,000,000 markers x 100
individuals. `filter_aims_beagle` now streams the file 5,000 markers at a
time (`hycline.data.beagle._iter_beagle_chunks`), accumulating per-sample
missing-data counts in `filter_aims._MissingTally` instead of keeping the
full missing-data matrix; the VCF filter uses the same tally, which cut
its peak at 300,000 loci from 337 MB to 201 MB. Streaming costs the
beagle filter a roughly fixed ~200 MB for the chunk being parsed, which is why it
uses more memory than before below about 30,000 markers. Only
per-marker names, frequencies and called counts are kept across chunks,
so memory now grows by tens of bytes per marker. Frequencies, the AIM
mask and every missing-data count are identical to the previous
implementation on the 10,000-marker file and across chunk sizes 1 to
5,000 (`tests/test_filter_aims.py::
test_filter_aims_beagle_streaming_is_chunk_size_independent`). Retained
beagle lines are now copied verbatim from the input, as VCF lines
already were, rather than re-formatted from parsed values.

