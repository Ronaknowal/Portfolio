# Bias–Variance: content design and continuation

Stable ID: bias-variance-tradeoff-learning-curves. Classical ML, authorized batch position 6. Author: root, 12 September 2026. Mode: research and write only. The central delivery ledger owns phase status and hashes. This packet contains the [manuscript](lesson.md), [visual specifications](visual-specifications.md), [data provenance](data-provenance.md), unchanged real dataset, source metadata and bounded calculations. No implementation/publication or formal independent review has occurred.

## Original source, notes and scope

Read the complete 65,909-byte original src/learn/data/topics/bias-variance-tradeoff-learning-curves.jsx in bounded sections. Baseline 8c5da59f18516be77c29d5aeeafca3decca4f738; original SHA-256 d4376cc6d4c776afee1b7f705fe3c5d7b44edea5e08e0d2d2b31dc8d9e3ac591. Existing runtime remains unchanged. Content preflight read the topic record, domain strategy and authoring notes: no destination note exists, and the unrelated bitwise inbox item is already resolved.

Retain the title and stable identity. The revision teaches the promised decomposition and practical learning curves, with deeper classification, training optimism and modern nonmonotonic behavior. No catalogue rename/reorder is warranted. Feature-selection and regularization come before this topic; imbalanced learning follows. The manuscript has local mean/variance/error and split refreshers, with the matrix/projection material clearly optional.

| Original learning coverage | Retention or correction |
| --- | --- |
| Why error diagnostics matter | Retained, replacing universal treatment rules with observed evidence and controlled next hypotheses |
| Squared-loss decomposition | Full derivation, conditional heteroscedastic noise, test-input weighting and algorithm-inclusive definition |
| Bias versus model expressiveness | Corrected false “zero iff truth belongs to family”; includes pointwise coincidence, shrinkage and optimization effects |
| Simulated repeated fits | Replaced clipped unstable high-degree normal equations and fabricated-looking fixed outputs with exact 8/32-world enumeration and transparent least squares; bootstrap is separately defined |
| Learning/validation curves and boosting trajectory | Three separate axes and complete real-data programs; actual results retained even when no U shape/overfitting turn appears |
| Remedy tables and scaling | Qualitative discriminating experiments replace invented numerical benefit heatmaps and gap thresholds; fit counts and actual partial-fit requirement retained |
| Training optimism/OLS | Fully corrected projection proof, rank/intercept convention, same-X fresh-response target and distinct new-location leverage; effective smoother extension added |
| Classification | Exact Brier identity versus a direct zero-one counterexample; no “approximation to accuracy” claim |
| Averaging | Derived correlated-average variance with explicit shared-mean assumptions; ensemble training algorithms remain prior dedicated lesson |
| Double descent | Preserved with original primary evidence and explicitly asymptotic sample-wise Gaussian-linear calculation; no universal neural min-norm or parameter-count theorem |
| Noise-floor and statistical diagnosis claims | Removed fixed fold-CI/variance-reduction rules, guaranteed future slopes and erroneous fixed label-noise cross-entropy number |
| History/references/practice | Removed unsupported first/coined/supremacy anecdotes; replaced copied-answer examples with eight changed calculation/diagnosis/experiment tasks and annotated alternatives |

## Outcomes, hurdles and progression

Core: distinguish a fitted mistake, sample sensitivity and target noise; calculate and interpret a finite decomposition; construct a reproducible learning curve; tell size/setting/time axes apart; design a next experiment that could refute a diagnosis. Deeper: explain optimism with projections, identify loss dependence and why nonmonotonic trajectories do not refute an identity.

| Hurdle | Local bridge and example | Representation / practice |
| --- | --- | --- |
| Two sources of spread | Three prediction worlds versus fresh target outcomes | Separate dot rulers, I1 editable predictions, practice 1 |
| Why mixed terms disappear | Centered deviations and independent fresh outcome | Expansion figure F1; explicit algebra |
| Procedure averages versus one fit | Exact polynomial worlds and a queried input | I2 fitted bundles/probe; practice 2 |
| Unknown truth on real data | Airfoil physical inputs and held-out rows | F2 fit/split lanes; meaningful mean baseline |
| Curve axes and controlled interpretation | Actual sample-size crossover, restriction failure, last-round minimum | F3/F4, practices 3–5 |
| Training dependence | SameX with fresh outcomes rather than a new-location claim | F5 projection lanes, practice 6 |
| Classification and ensemble difference | Bernoulli risk and correlated average | Worked scalar calculations, practice 7 |
| Nonmonotonicity | Gaussian-linear asymptotic risk at changed sample ratio | F6, practice 8 |

First-pass route is §§1–6/practices1–5. The deeper sections and readiness are explicitly matched. Interesting applications arise from an actual aerodynamic prediction study, a measurement changing conditional uncertainty, and data growth producing unexpected model comparisons. No detached trivia or unsupported deployment claim is added.

## Canonical section-list coverage

Canonical baseline: Hastie/Tibshirani/Friedman, ESL chapter 7. Its identity was verified on the [publisher chapter page](https://link.springer.com/chapter/10.1007/978-0-387-84858-7_7), and the full section list 7.1–7.12 was read in the publisher's freely accessible [front matter, PDF page 15](https://link.springer.com/content/pdf/bfm:978-0-387-84858-7/1). Full chapter subscription content was not accessed. Several author-hosted legacy PDF URLs failed, so no full-book reading is claimed. The primary free Caltech lecture and original papers below establish substantive mechanisms independently.

| Canonical ideas | Decision |
| --- | --- |
| 7.1–7.3: complexity and decomposition | Full core derivation and exact new examples |
| 7.4–7.6: optimism, error estimates and effective parameters | Deeper projection and general smoother derivation, with actual same-X distinction |
| 7.7: Bayesian/BIC selection | Short comparison in §7; the actual earlier Regularization packet§9 develops predictive AIC, evidence-oriented BIC, assumptions and an exact selection disagreement |
| 7.8: description-length selection | Earlier Regularization packet§9 develops a declared two-part bit code and finite-normalizer NML, with scope limits. A direct learner link replaces an unspecified future destination; no duplicate coding survey is needed here |
| 7.9: VC theory | The later PAC/VC and Rademacher topics own uniform bounds; distinguish procedure risk locally rather than add an unexplained bound |
| 7.10: valid CV | Core protocol and earlier CV link; full split/search methods stay with that author |
| 7.11: bootstrap | Explain what resampling can/cannot identify; full bootstrap inference belongs to its established statistical lesson |
| 7.12: conditional versus expected risk | Explicit opening of§7, connected to §2 and observed held-out score |

Supplementary canonical teaching sequence: Caltech Lecture 8's complete 23-slide sequence was read, including repeated-fit distributions, model approximation versus learning, learning curves and fixed-input fresh-response OLS. Its notation/diagram conventions are explained beside the optional video link. A screenshot request failed with cache-miss; no rendered slide/video review is claimed.

## Research record and evidence limits

All retrievals 12 September 2026. Primary documentation/papers were read for technical claims. No empirical values from another source were copied into this lesson's plots.

| Source / locator | Actual review and use |
| --- | --- |
| [Caltech slides08](https://work.caltech.edu/slides/slides08.pdf), full sequence, especially 6–9 and 18–22 | Definitions, centering, fixed-X fresh-response contract; original numerical examples and proofs developed locally. Official course links verified [here](https://work.caltech.edu/telecourse.html). Video metadata plus slides reviewed, video not watched |
| [scikit-learn bagging decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html), explanation and code | Useful alternate visual resource; fresh simulated datasets distinguished from bootstrap. Our example/dataset/results are different |
| [Learning-curve guide](https://scikit-learn.org/stable/modules/learning_curve.html), section list and code | Validation versus learning axes, selection/evaluation boundary. Categorical diagnostic simplifications are qualified rather than blindly reproduced |
| [learning_curve API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html), parameters/returns | Fit sizes, shuffle/cv/scoring semantics; local sklearn 1.9.1 source additionally confirms incremental mode explicitly checks partial_fit, not warm_start |
| [Domingos AAAI 2000 paper](https://homes.cs.washington.edu/~pedrod/papers/aaai00.pdf), definitions/Theorems1–3 | Loss-specific distinctions. Original source conflated publications; this packet links the exact inspected paper rather than inventing a venue/title combination |
| [Belkin et al.](https://arxiv.org/html/1812.11118v2), introduction and §3/tree appendices | Double descent not exclusive to neural networks; no universal model ranking reproduced |
| [Nakkiran note](https://arxiv.org/pdf/1912.07242), setup,§3.1 Claims1–2 and conditioning discussion | Correct zero-initialized linear min-norm context and explicitly asymptotic formulas, original substitutions saved. Requested PDF screenshot failed; text/equation extraction read |
| [UCI Airfoil](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), metadata/license/download | Real measurements, units and redistribution metadata; exact archive/member hash retained. Corrected inconsistent “binary” webtype for numeric attack angle based on actual values/description |
| ESL publisher chapter/frontmatter above | Canonical breadth check, not a claim that inaccessible chapter content was fully read |

Author-hosted ESL/ISLP downloads failed with 404/empty responses; Bach's book exceeded the web tool's size limit and a direct request failed certificate validation. These are retrieval limitations, not unresolved content facts: the canonical ESL section list was ultimately inspected through the official publisher, while substantive claims use available primary sources. No certificate setting, package or shared environment was changed. No failed reference artifact was retained.

## Author calculations and final learning-experience pass

Executed author-calculations.py once in the existing runtime: NumPy 2.3.5, scikit-learn 1.9.1. It enumerates exact finite worlds, computes the prespecified real curves and records the stated asymptotic substitutions. Maximum finite decomposition residual was 1.33e−15. All 100 learning fits, 30 validation-curve fits and one 120-round boosting fit are measured; no reserved 303-row score exists. A follow-up JSON-only calculation checked probe .5 values and the five-input design's pointwise constant-model reversal.

The author reread the complete manuscript, visual specifications, design and provenance in ordered sections, and ran the full learning-experience checklist. This is an author assessment, not an independent learner trial or phase-two review. The pass added explicit conditional-versus-procedure risk and diagnostic fit-cost explanations, checked the actual curve crossovers and changed practice, and repaired compressed wording in the handoff documents. Findings and boundaries:

- Three-world example totals, changed practice fractions and OLS traces were independently calculated.
- The true-curve convention 1+x+c*x² was kept consistent across prose/code/data; the constant's small bias at .5 is an accidental local fit, explicitly distinguished from the global curve.
- Fixtures expose opposite model choices and nontrivial nulls; default prediction choices are unset and input-bound. The real curves are inspection figures, not mislabeled preset-only investigations.
- No train/validation gap is called an exact variance; fold spread is not a confidence interval; class-probability loss is not a proxy theorem for accuracy.
- The actual favorable unrestricted-tree result and still-improving boosting trajectory are retained.
- Whole-section algorithm history, repeated caution boilerplate and numerical remedy heatmaps were removed. Cautions have their main homes in §2 assumptions,§3 experiment identity,§5 population/protocol and §9 asymptotics.
- Code is complete instructional NumPy/scikit-learn code, not an implemented web lab. Verbatim assembled-program execution is deferred; native author results establish the reported outputs only.
- Graph axes/visibility, figure placement and responsive text equivalents are specified. No desktop/phone/browser review is claimed.

## Handoff

Retain all seven topic-owned files plus the exact data-source metadata (eight files total). The authorJSON is larger than the eventual necessary runtime fixture; export only needed values when implementing. No runtime source, manifest, route, catalogue title or publication was modified. The pending draft/source artifacts are necessary continuation inputs.

When finish is authorized, run the stable-ID finish preflight, consume all packet files, implement topic-owned visual/model/lab/example assets, execute complete displayed programs, independently check numerics and learning experience, inspect browser/keyboard/mobile states and integrate. Update source-bound checkpoints for necessary corrections. Content author checks are not formal independent review or user acceptance.

## Phase two implementation, 15 September 2026

This section is appended by the implementing agent. The manuscript, visual specifications, recorded
calculations, provenance and dataset above are unchanged; only this record is new.

### What was built

| File | Responsibility |
| --- | --- |
| `src/learn/data/bias-variance-models.js` | The verified computation layer. Every quantity a figure or investigation draws is computed here: the three-world decomposition and its direct pair enumeration, the 8- or 32-world polynomial experiment with its per-world curves and pointwise grid decomposition, least squares and the interpolation weights, the recorded-curve summaries with the crossover and restriction claims, the boosting-trace argmin, fixed-design optimism with an explicit hat matrix and the general smoother trace, new-location leverage, the Brier and thresholded-decision accounts, the correlated-average variance, and the piecewise double-descent expressions with an undefined boundary. |
| `src/learn/data/bias-variance-data.js` | Generated real-data module, 12 KB: provenance, the four procedures with all 5x5 fold values each, the six-setting validation curve with its folds, and the 120-round boosting trajectory. |
| `src/learn/data/bias-variance-examples.js` | The three displayed programs with their executed output. |
| `src/learn/components/lesson-labs/BiasVarianceShared.jsx` | Investigation scaffolding: the draft/applied/previous commitment cycle, unset predictions, number fields that publish only valid values, a linear-or-logarithmic plot frame, an additive decomposition track, and tables that stack into labelled rows on a phone. |
| `src/learn/components/lesson-labs/BiasVarianceLabs.jsx` | I1, the prediction and outcome rulers; and I2, the finite worlds. |
| `src/learn/components/lesson-labs/BiasVarianceFigures.jsx` | Seven inline figures. |
| `src/learn/components/lesson-labs/bias-variance-labs.css` | Styles, including the narrow-width table stacking. |
| `src/learn/data/topics/bias-variance-tradeoff-learning-curves.jsx` | The lesson body. |
| `src/learn/data/curriculum/blueprints/bias-variance-tradeoff-learning-curves.js` | The authored plan. |
| `public/learn-assets/bias-variance/` | This lesson's own copy of the unchanged dataset and its attribution. |
| `scripts/verify-bias-variance-models.mjs`, `-examples.py`, `-data.py`, `-browser.cjs` | The four verifiers. |

### Agreement with the recorded calculations

No disagreement was found with any number in the packet. Before any code was written, the airfoil protocol
was reimplemented from scratch (KFold, one permutation stream consumed in fold order, prefixes of each
fold's training rows) and shown to reproduce scikit-learn's learning_curve exactly, and then the packet. All
21 recorded finite experiments were compared world by world and point by point across their 61-point grids:
19,764 recorded values. The manuscript's stated tables in sections 1, 3, 5, 7, 8 and 9, and every practice
answer, were recomputed independently and matched.

### Departures from the specification, and why

1. **Seven inline figures rather than six.** A figure was added for section 8, "Two losses, two different
   accounts": the straight-line zero-one error against the class-1 rate makes "increased variation may help
   or hurt" visible instead of asserted, which the surrounding prose otherwise asks the reader to assemble
   mentally. Its content is the manuscript's own stated arithmetic.
2. **A concrete fixed design in the section 7 figure.** The claims about leverage are drawn on an explicitly
   constructed six-row design with inputs -2.5, -1.5, -0.5, 0.5, 1.5, 2.5 and an intercept, so that n = 6 and
   p = 2 are the manuscript's own numbers. Three things become exact rather than asserted: the hat matrix
   reproduces 8/3, 16/3 and 4/3 through the general smoother formula, since its trace is 2; leverage averaged
   over the six rows returns p sigma squared over n exactly; and a truth carrying a curvature term the design
   cannot represent adds 6.222222 to both errors while leaving the gap at 8/3. The design is named as a
   constructed calculation in the lesson's closing paragraph.
3. **A checkpoint on the five-input reversal.** The visual specification records the observed contrast
   0.340278 becoming 0.3625 while the variance falls, and asks that feedback show both terms. It is taught as
   a checkpoint beside the investigation, and the investigation additionally renders a persistent table of
   both designs at all three degrees so the contrast is legible without hunting for it.
4. **I1's decomposition is an additive track, not a typeset equation.** The specification asks for one
   equation directly beneath the dot geometry. Because the graded quantity here is a direction of change, the
   implemented form is an additive bias/variance/noise track plus a before-and-after table naming which
   operand moved. The equation itself is stated in section 2 and in the figure beside it.
5. **I2's panels stack rather than switching.** The specification asks for one fit at a time on phones. The
   implementation renders the reference and candidate panels in a grid that collapses to a single column
   below 620 px, which shows one fit at a time while keeping the comparison table and the
   reference/candidate selects permanently visible.
6. **The section 5 figure's own table carries the training means.** The manuscript's validation table sits
   immediately above the figure, so repeating it inside the figure would have been duplication; the figure
   tabulates the other line each panel draws instead, and the per-fold values stay behind a disclosure.
7. **Two computed facts added to the prose.** The crossover size, 240 fitted rows per fold, and the sizes at
   which the leaf-20 restriction helps and hurts are printed from the model layer, so the manuscript's
   qualitative claims become checkable rather than merely stated. Both are comparisons over the recorded
   values and are asserted in the verifier.
8. **One sentence about -0.0.** Executing the manuscript's own program verbatim prints -0.0 in the leaf-1
   tree's training column, because the scoring interface returns negative MSE and negating an exact zero
   keeps its sign bit. The lesson says so once, where that output appears.
9. **A numbered readiness section.** The closing readiness paragraph and next-topic link were given a heading,
   section 11, so the route list in the introduction reaches them.
10. **The boosting trace's wiggles are drawn.** The recorded monitoring trace rises again at individual
    rounds while its minimum over all 120 is the last one. The model layer reports both facts and the figure
    states them, rather than smoothing the curve into a monotone shape.

### Integration observation for the increment owner

Contrary to the phase-two briefing, this stable ID is **already registered** in
`src/learn/data/lesson-manifest.json`, pointing at `./topics/bias-variance-tradeoff-learning-curves.jsx`.
Replacing that file therefore changed a live, reachable page rather than an inert draft. The previous
65,909-byte body, SHA-256 `d4376cc6d4c776afee1b7f705fe3c5d7b44edea5e08e0d2d2b31dc8d9e3ac591` and matching
the baseline recorded above, is preserved in git at HEAD and copied to
`scratch/bias-variance/original-topic-baseline.jsx`. A production build succeeded and this lesson code-split
into its own 159 KB chunk, 52 KB gzipped. The build regenerated one line of
`src/learn/data/generated/navigation.js`: this topic's readTime, which changed from `~40 min` to an estimate
carrying explicit reading and practice units. The blueprint is **not** registered in `blueprints/index.js`;
that registration remains the increment owner's.

### Verification completed in this phase

| Check | Result |
| --- | --- |
| `node scripts/verify-bias-variance-models.mjs` | 179 grouped checks across 45 groups, including 19,764 recorded grid values |
| `scripts/verify-bias-variance-examples.py` | 3 displayed programs executed, 55 oracle assertions; recorded output reproduced byte for byte |
| `scripts/verify-bias-variance-data.py` | 100 learning fits, 30 validation fits and one 120-round trajectory recomputed twice, matched against the packet, and the module regenerated byte-identically; read-only without `--write` |
| `npx vite build --outDir dist-bv` | Succeeded |
| Server-side render of the lesson body | 2 investigations, 7 figures, 8 practice tasks, 3 programs, 20 tables, 48 drawings; no radio preselected and no verdict on first paint |
| SVG label extent and pairwise overlap estimate over the rendered body | No label outside its own viewBox and no label-on-label overlap in the default state |

### What remains

Browser verification against a production preview is deferred to the next phase: real interaction, keyboard
and focus behaviour, KaTeX display widths measured on the real page, the visual-layout inspector at five
widths, screenshots of the informative states, and the post-commit render paths of both investigations,
which a server-side render cannot reach. Independent review and user acceptance remain separate.

### Phase C: browser review, 15 September 2026

Registration completed by the increment owner; the lesson was rebuilt to `dist-bv`, previewed at
127.0.0.1:4187 and reviewed in Edge at 1366, 1024, 768, 390 and 320 px. `scripts/verify-bias-variance-browser.cjs`
passes with **11 cases and 30 screenshots**, writing `docs/teaching/evidence/bias-variance-browser.json`.

**Every screenshot was opened and looked at**, and the drawings that a full-page capture renders too small to
judge were re-captured as tight crops at device-scale 2. That second pass is what found most of the defects
below; four of them were invisible to a green verifier.

#### Defects found by the layout inspector, and fixed

| Defect | Fix |
| --- | --- |
| Three labels crossed by a foreground line at 1366 px: the outcome ruler's value label under the true-mean line, and the section 7 residual label across both the projection diagonal and its vertical | Row values moved into a left gutter; the residual label shortened to `(I − H)y` and placed beside the segment it names |
| The section 2 bracket end ticks rose into their own span labels | The ticks now drop below the span |
| Four overlapping label pairs: the section 8 axis title against the tick row, and the γ = 1 boundary label against the tick row and against an inset value | The loss plot gained bottom padding; the boundary's two labels merged into one above the frame; the inset's leftmost value is anchored away from the axis |
| Three mark symbols left the top of the ruler viewBox at 1024 px | Baselines lowered to clear the glyph ascent |

#### Defects found only by looking at the images

1. **Table captions collapsed into a vertical column of single words at narrow width.** A `table-caption`
   inside a table that the stacking rules blockify shrinks to its longest word. Captions are now blocks below
   the breakpoint. This affected every stacked table in the lesson.
2. **The boosting trace's nine local rises were not perceptible, and the figure claimed they were drawn.**
   They are at most 0.0383 squared decibels on a 0–50 axis. Rather than exaggerate them, the figure now says
   they are too small to see at this scale, lists the nine rounds, and names rounds 108 and 109 to read in the
   round inspector, where 13.0848 becomes 13.1231. This replaces the claim recorded as departure 10 above: the
   wiggles are *recorded and checkable*, not visible.
3. **The two double-descent branches read as a single spike.** They were never joined in the path data, but a
   0.06-wide gap in γ looks like a notch. The boundary is now a shaded band from γ = 0.97 to 1.03 that the two
   branches stop at, with one arrow up its centre; the two converging arrows that previously collided head to
   head are gone.
4. **The nested-subset diagram's flow curves ran straight through the row counts.** The layout inspector
   covers straight foreground lines, not curved paths, so it passed this. The counts moved to a left gutter.
5. **The mean and truth reference lines coincided in the ruler and one hid the other.** The default state and
   six of the eight suggested setups have m = f. The dashed mean is now drawn over the solid truth so gold
   shows through its gaps, and the caption says so when they coincide.
6. **Inset values sat on vertical gridlines.** A `bv-halo` backplate, opaque in both the figure and the
   investigation ground, now sits behind any value printed over a line.
7. **`crossover at 240` overstated the drawing.** The two lines visually meet near 200, on a segment that
   joins two fitted sizes rather than a measured one. The annotation now reads `tree better from 240`, and a
   new sentence says the segments are joins and gives the 120-row values where Ridge was still ahead.
8. Smaller polish: two labels flush against a frame edge nudged inward, and a value flush against the
   decomposition key's edge given padding.

#### Checks made on the real page

- **KaTeX display math at 320 px**, measured on the rendered page rather than through injected CSS: no
  formula overflows, and the widest display renders at 280 px inside a 288 px column.
- **Stacked tables** at 390 and 320 px: rows become blocks, cells become two-column grids carrying the heading
  they lost, and no table scrolls sideways.
- **SVG label extents** against their own viewBox, plus pairwise label overlap, over all 48 drawings.
- **Every practice instruction that names a value** against the rendered controls: practice 2's σ = 1 at the
  0.05 step, the suggested prediction setups at the 0.25 step, and every preset inside its control's range.
- Load closure, completion, Previous/Next, and recovery from an import failure and a render failure.

#### Not fixed, and why

- One screenshot path is written twice: the figure-4 capture in the per-figure loop runs after the round-30
  selection, so `bias-variance-figure-4-desktop.png` shows the inspected round rather than the default. The
  state it captures is the more informative one, so the order was left alone; 30 captures produce 29 files.
- The visual-layout inspector does not examine curved paths, so defect 4 above could recur elsewhere in this
  lesson without being caught. It was checked by eye in every drawing that uses a curved flow: figures 2 and 5.

## Disposition of the independent review

The [independent review](../../BIAS-VARIANCE-INDEPENDENT-REVIEW.md) recomputed every stated number from first
principles across **22,098 constructed checks**, importing nothing from `bias-variance-models.js`,
`bias-variance-data.js` or any verifier: exact rational arithmetic wherever a rational answer exists, the generated
module parsed as text rather than imported, the airfoil pipeline refit twice — once by running the manuscript's
displayed programs verbatim and once through a hand-rolled protocol that never calls `learning_curve` — and a
finite-sample Monte-Carlo of the minimum-norm ridgeless estimator. **It found no disagreement with any published
number** in the manuscript, the visual specification, the practice solutions, the lesson body, the generated module or
the packet's own trust root. It also ruled that where the model layer's double-descent bias/variance split departs
from the manuscript, the model layer is correct and the manuscript merely silent, and it confirmed the Phase C
self-correction by measuring the largest boosting rise at 0.038342 on a 0–50 axis.

It raised three blocking findings, four to fix and eight observations. All seven blocking and should-fix findings are
resolved.

| Finding | What was wrong | Resolution |
|---|---|---|
| B1 | The record's only "not fixed" entry named the wrong file. The doubly-written path was `figure-3-desktop.png`, not figure 4; `figure-4-desktop.png` was a distinct path whose bytes duplicated the round-30 capture, so figure 4's default state was captured nowhere and the justification for leaving it — "the state it captures is the more informative one" — described a state that already had its own file | The duplicate figure-3 write is removed, the round selector is restored to its default before the per-figure loop, and the verifier now asserts that every capture has its own path. Thirty captures, thirty files, figure 4's default among them. This paragraph replaces that entry: the reviewer's reading is correct and mine was wrong |
| B2 | Four data curves ran through value labels — worst, figure 7's risk curve along 22.8% of `noise 0.04`; most damaging, figure 6's `0.20`, the one label in that plot with no backplate. Phase C predicted this class and then scoped its by-eye sweep to flow arrows rather than data curves | All four labels moved clear: the asymptote is named at the left end where the curve is an order of magnitude above it, figure 6's last value sits below its point, the inset's `0.50` sits above-left of the rise, and the crossover annotation clears the baseline. Figure 6's four values gained the backplate. The browser verifier now samples every `<path>`, `<polyline>` and `<polygon>` along its own geometry and tests it against every label box at all five widths |
| B3 | §9's prose said the shaded band "is where this approximation has no value", then the table printed two of its values (4.01, 4.04) from inside that band — widening the singularity from a point to an interval, the opposite of what §9 exists to teach | Dissolved by fixing S4 rather than by rewording. The band is gone; the prose now says exactly one point is missing, gives the ordinary values at γ = 0.999 and 1.001, and explains that the gap between the branch ends is that single point drawn at the width the axis gives it |
| S1 | `design.md` said four defects were invisible to a green verifier, then listed eight | The count was transposed with the four the inspector did catch. It is **eight** image-only defects and four inspector-caught ones; this table is the correction, since the record is append-only |
| S2 | `bias-variance-models.js` still carried the retracted Phase A claim that the boosting wiggles "are drawn rather than smoothed away" | The JSDoc now says they are recorded and checkable, not visible, and why a figure must say where to read them |
| S3 | Figure 3's combined plot separated Ridge and the leaf-1 tree — the pair whose crossover is its headline claim — by stroke colour alone | Ridge is now dash-dot. The plot carries four distinguishable patterns: fine dotted, dash-dot, solid and dashed |
| S4 | Figure 7's branches stopped at γ = 0.97/1.03, where the approximation is ≈1.37, so the drawn peak was a fifth of the tabulated 4.01/4.04, the band read as a bar over a continuous spike, and the arrow floated above the curve | The branches now run to γ = 0.99 and 1.01, the last ratios the table gives, putting the drawn peak at the tabulated height near the top of the axis. The band is replaced by a single dashed line at γ = 1 that arrows off the frame, which is the visual specification's own "mark 1 as singular/asymptotic boundary" |

**Observations acted on.** O1 was the most valuable: `calculated-inputs.json`'s `finiteExperiments` and
`doubleDescentApproximation` blocks feed all 19,764 comparisons in the model verifier and were re-derived by nothing in
the repository. `verify-bias-variance-data.py` now re-derives every value in both blocks from the recorded settings
alone — through normal equations rather than the author script's `lstsq`, and through the manuscript's piecewise
expression — and records that file's SHA-256. It reports **21,993 trust-root values**, matching the reviewer's own
count. O2's blind spots are closed in three new browser cases: plotted coordinates are now read back from the combined
learning-curve plot and asserted to be each series' validation mean at its own fitted size rather than its training
mean, with the crossover stem at the size its label names; the grading contract is exercised off its happy path, so a
wrong direction must render the mismatch branch naming both outcomes, a numeric guess must be reported outside its
tolerance and the committed state must be echoed; and keyboard focus, focus moving on rather than being dropped, and
the survival of the displayed program's Python indentation are each asserted. O4's correction is recorded: the drawn
segments cross at ≈138 fitted rows, not "near 200" as Phase C said — the number was wrong, the fix it justified was
right. O5's first half and O6 are now in the reference annotations: the scikit-learn page's own title is given
alongside the manuscript's wording for it, and Claim 2 is noted as credited in that paper to Hastie and colleagues.
O7's cosmetic residue is gone: figure 6's table was the one still needing a sideways scroll at 1366 px, and its
headings are shortened so it fits.

**Observations left, with reasons.** O3 — investigation 1 states the squared bias before any prediction is recorded —
is declined. The graded quantity is the *direction* of change of the total against the previous applied state, which
cannot be read off the bracket; the visual specification explicitly requires "a distance bracket [that] identifies the
signed mean offset"; and the waterfall and the moved-terms table, which are the answer, are both gated behind the
commit. Removing the bracket would cost the section its main picture to satisfy a phrase the contract does not apply to
input displays. O5's second half — the incremental-mode claim is true of scikit-learn but not stated on the cited API
page — is left as written: that sentence is frozen manuscript text, the packet's design record already documents that
it was verified against the local scikit-learn source, and the reviewer independently confirmed the claim is true. O8
needs no action. One correction to the review itself, recorded because it affected a fix: in figure 6, `0.20` was not
the only value label without a backplate — none of the four had one. All four have one now.

After the fixes all four verifiers pass: **179 model groups over 45 groups including 19,764 recorded grid values**;
**3 displayed programs with 55 oracles**; **100 learning fits, 30 validation fits, one 120-round trajectory and 21,993
trust-root values**; and **14 browser cases with 30 captures**, up from 11 and 29. Every changed figure was re-opened
in the images afterwards to confirm the collisions are gone rather than moved — which is how the inset's `0.50` was
caught being relocated *into* the curve by its own first fix, and corrected.
