# Clustering evaluation: conceptual-transition review

26 September 2026. Read the full canonical JSX, all twelve sections and deeper branches, six programs' surrounding explanations, eight exercises and references. Existing core distance, pair and information labs are unusually explicit and were retained. Author status only; independent reading and rendered verification pending.

| Exact location and hurdles | Disposition |
| --- | --- |
| §1 evidence vs intended use, internal/external/stability/task questions | Retained: real two-versus-three Iris disagreement motivates different criteria before formulas. |
| §2 hard partition, arbitrary names, ID alignment, soft conversion | Retained: eight-ID strips and actual pair changes separate renaming from membership change. |
| §3 a/b/minimum of means, normalization, signs, per-point distribution, weighting, singleton/zero conventions | Retained: C's distance fan, movable membership investigation, complete program and changed-location checkpoint. |
| §4 metric/units, rescore vs refit, orthogonal invariance, ring shape, CH/DB/Dunn | Gap fixed for secondary criteria: one number line exposes center separation versus nearest-cross and widest-within pairs. Existing CH mean-centering proof and all three worked values retained. |
| §5 contingency counts, four pair types, RI, fixed margins, expectation, ARI, incompatible margins, degeneracies | Retained: actual pair board, efficient vs enumerated program and exact null assumptions. |
| §6 entropy/MI/refinement, NMI normalizers, empirical independence, AMI, null enumeration, hypergeometric expectation | Gap fixed at cellwise expectation: evaluate overlaps 1,3,4 for one cell, then sum four cells to recover .114844. This makes negative cell contributions and nonnegative total MI explicit. Existing live 70-assignment enumeration and information bars retained. |
| §7 splits/merges, purity, homogeneity/completeness/V, FM, one-to-one mapping, VI | Gap fixed for mapping: existing strict refinement yields 4/8 under one-to-one versus 8/8 purity, with unmatched convention stated. Existing entropy/pair calculations retained. |
| §8 real data, descriptive fit, rescore/refit experiment, projection as view | Retained: complete reproducible Iris pipeline, exact disagreement, freeze-partition controls and contingency diagnosis. |
| §9 noise encoding, coverage, conditional population, common intersections | Retained: same six IDs visibly rejected and all scores recomputed, not merely relabeled. |
| §10 perturbation meaning, sample weights, fixed probes, exact split, bootstrap ID alignment, co-assignment | Gap fixed for co-assignment denominator: four-run support table gives 2/3 rather than 2/4, and distinguishes sparse from extensive support. Existing exact solver lab and program retained. |
| §11 tendency/null, gap statistic and rule, cophenetic distance, cost, subset/focal sampling, proxy definitions | Gaps fixed: actual three-leaf tree makes first shared branch height visible; same C scored against six versus four observations exposes why subset silhouette changes the quantity. Existing gap selection arithmetic and resource bounds retained. |
| §12 fit/selection/report roles, leakage, held-out distortion, practice/transfer | Retained all programs, eight tasks, solution reasoning, exact module sequence and alternatives. |

## Actual reading and representations

Read original author [Zaki/Meira chapter 17 slides](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap17.pdf), specifically the introduction, matching/purity and entropy/VI portion (pages 1–14). Read the full definition and worked output of [SciPy cophenet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html) and the description, arguments and rule definitions in [R clusGap](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html). Used these to assess where an accurate definition still needs a visible local operation. Did not copy the slide deck's unrestricted claims about a true partition or treat its geometric NMI convention as this lesson's arithmetic convention. Existing annotated article/video alternatives retained; no video playback claimed.

`ClusteringEvaluationIntuitionFigures.jsx` provides two bounded static SVGs with scoped `clustering-evaluation-intuition.css`. The distance summary uses a common 0–9 spatial scale; every marked interval is quantitatively derived. The dendrogram uses a numeric height axis and explicitly categorical leaf spacing, avoiding confusing leaf layout with original distance. Tables provide exact support counts and prose walks through calculations at the point they are needed. No duplicated lab shell or mandatory prediction gate added.

All original models, programs, live labs, base figures, data and results retained. Parse, arithmetic and unchanged-source checks recorded in `author-checks.json`. Independent/browser/integration work pending; no fresh numerical fit implied.
