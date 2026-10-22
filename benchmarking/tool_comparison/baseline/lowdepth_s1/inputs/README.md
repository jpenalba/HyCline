# Matched inputs for HyCline, gghybrid, bgchm and bgc

Source: `sim.vcf` and `sim.beagle`, metadata `sim_metadata.tsv`.
100 loci; 20 `pop1` individuals (hybrid index 0), 20 `pop2`
individuals (hybrid index 1) and 100 test individuals. Allele counts and
frequencies are of the VCF ALT allele; `individuals.txt` (in `bgc/`) and the
row names of the bgchm matrices give the individual order.

| Folder | Run | Writes |
|---|---|---|
| `hycline/` | `bash run_hycline.sh` | `hybrid_index.tsv`, `fit_mle/hycline_summary.csv`, `fit_hier/` |
| `gghybrid/` | `Rscript run_gghybrid.R` | `hybrid_index.tsv` (esth), `clines.tsv` (ggcline) |
| `bgchm/` | `Rscript run_bgchm.R genotype` (or `glik`) | `hybrid_index.tsv`, `clines.tsv` |
| `bgc/` | see below | bgc's HDF5/text output |

## Parameter correspondence

HyCline: logit(phi) = u + v logit(h). gghybrid and bgchm:
phi = h^v / (h^v + (1 - h)^v exp(u')), i.e. u' = -u, with cline centre
c = invlogit(-u / v) = invlogit(u' / v) (the h at which phi = 0.5). All three
model a locus's allele frequency as p0 + (p1 - p0) phi(h). bgc 1.03 uses a
different cline function (alpha and beta; Gompert & Buerkle 2011), so its
parameters are compared through outlier calls rather than directly.

## bgc 1.03

bgc is distributed from its authors' site (sites.google.com/site/bgcsoftware;
manual: bgcmanual.pdf). `genotypes/` holds known-genotype input: for each
locus, `p0in.txt`/`p1in.txt` give the parental REF and ALT allele counts, and
`admixedin.txt` gives, under a single `pop 0` line, each test individual's
REF and ALT allele copies (`0 0` = missing). `reads/` holds the same layout with per-individual REF/ALT read counts in all three files, for the genotype-uncertainty model (`-N 1`, with `-E` set to the sequencing error rate). A typical run (check
the flags against the manual for your bgc version):

    bgc -a genotypes/p0in.txt -b genotypes/p1in.txt -h genotypes/admixedin.txt \
        -O 2 -x 50000 -n 25000 -t 10 -p 1 -q 1 -N 0 -o bgc_out

then `estpost` to summarize alpha, beta and the hybrid indexes.
